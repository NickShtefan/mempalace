"""Embedding function factory with hardware acceleration.

Returns a ChromaDB-compatible embedding function bound to a user-selected
ONNX Runtime execution provider.

The model is selected via ``MEMPALACE_EMBEDDING_MODEL`` or
``embedding_model`` in ``~/.mempalace/config.json``:

* ``minilm`` (default) — ``all-MiniLM-L6-v2``, 384-dim, English-only training.
  ChromaDB's default; what every existing palace was built with.
* ``embeddinggemma`` — ``onnx-community/embeddinggemma-300m-ONNX`` (q8), 384-dim
  via Matryoshka truncation, multilingual (100+ languages). Cross-lingual cos
  ~0.88 on parallel translations vs MiniLM's ~0.35. Recommended for any
  non-English use; onboarding offers it as the default. The ~300 MB ONNX
  model is lazy-downloaded from HuggingFace on first use. Switching models
  on an existing palace requires ``mempalace repair rebuild-index``
  (different vector space).
* any other value — treated as a `sentence-transformers
  <https://www.sbert.net/>`_ model id (a HuggingFace repo id such as
  ``intfloat/multilingual-e5-base`` or ``BAAI/bge-m3``, or a local
  checkpoint path) and served through chromadb's
  ``SentenceTransformerEmbeddingFunction``. Requires
  ``pip install mempalace[multilingual]``; a missing dependency raises an
  actionable ``ImportError`` instead of silently falling back to the
  default model (a palace stamped with one model but embedded with
  another is exactly the corruption the identity check exists to stop).
  This path runs on torch, not ONNX Runtime — see
  :func:`_resolve_torch_device` for how ``embedding_device`` maps.

Supported devices (env ``MEMPALACE_EMBEDDING_DEVICE`` or ``embedding_device``
in ``~/.mempalace/config.json``):

* ``auto`` — prefer CUDA ▸ CoreML ▸ DirectML, fall back to CPU
* ``cpu`` — force CPU (the historical default)
* ``cuda`` — NVIDIA GPU via ``onnxruntime-gpu`` (``pip install mempalace[gpu]``)
* ``coreml`` — Apple Neural Engine (macOS)
* ``dml`` — DirectML (Windows / AMD / Intel GPUs)

Requesting an unavailable accelerator emits a warning and falls back to CPU
rather than hard-failing — mining must still work on a laptop without CUDA.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_DEVICE_EXTRA = {
    "cuda": "mempalace[gpu]",
    "coreml": "mempalace[coreml]",
    "dml": "mempalace[dml]",
}

_AUTO_ORDER = [
    ("CUDAExecutionProvider", "cuda"),
    ("CoreMLExecutionProvider", "coreml"),
    ("DmlExecutionProvider", "dml"),
]

# Model ids resolved by this module's own ONNX classes. Anything else is a
# sentence-transformers model id (HF repo or local path).
_BUILTIN_MODELS = {"minilm", "embeddinggemma"}

# Config device vocabulary (ONNX-EP-shaped) → torch device for the
# sentence-transformers path. DirectML has no torch equivalent.
_TORCH_DEVICE_MAP = {"cpu": "cpu", "cuda": "cuda", "coreml": "mps"}

_EF_CACHE: dict = {}
# Check-then-construct on the cache must be atomic: without it, two threads
# resolving the same key each keep their own EF instance, and each instance
# later lazy-loads its own copy of the model.
_EF_CACHE_LOCK = threading.Lock()
_WARNED: set = set()


def is_sentence_transformer_model(model: Optional[str]) -> bool:
    """True when ``model`` names a sentence-transformers checkpoint.

    Any value other than the built-in ``minilm`` / ``embeddinggemma``
    identifiers (and empty/unset, which mean minilm) is routed to the
    sentence-transformers path — e.g. ``intfloat/multilingual-e5-base``,
    ``BAAI/bge-m3``, or a local checkpoint path.
    """
    if not model:
        return False
    return str(model).strip().lower() not in _BUILTIN_MODELS


class EmbeddingDependencyError(ImportError):
    """A configured embedding model needs a package that is not installed.

    A dedicated subclass so error arms in mcp_server / searcher / layers can
    match *this* failure specifically — a bare ``except ImportError`` would
    also swallow unrelated import failures (e.g. a missing backend driver
    like ``pymilvus``) and mislabel them with the ``[multilingual]`` hint.
    """


class EmbeddingModelError(RuntimeError):
    """The configured sentence-transformers model failed to load.

    Wraps download / checkout failures (bad model id, no network, corrupt
    cache) with the model name and recovery options, so the raw huggingface
    error does not surface bare — and is never silently swallowed into a
    fallback to the default model.
    """


def _sentence_transformers_available() -> bool:
    """Cheap availability probe for the ``sentence_transformers`` package.

    Checks ``sys.modules`` first (also lets tests stub the module in or
    block it with a ``None`` entry), then falls back to ``find_spec`` — a
    metadata lookup, NOT an import, so callers on startup-budget paths
    (init, collection open) don't pay the multi-second torch import just
    to learn the package exists.
    """
    mod = sys.modules.get("sentence_transformers")
    if mod is not None:
        return True
    if "sentence_transformers" in sys.modules:
        # Explicit None entry — the import is blocked.
        return False
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def ensure_model_dependencies(model: Optional[str]) -> None:
    """Raise :class:`EmbeddingDependencyError` when ``model`` needs packages
    that are not installed. Cheap — a metadata probe, no import, no download.

    The probe targets ``sentence_transformers`` directly: chromadb's
    ``SentenceTransformerEmbeddingFunction.__init__`` converts the
    underlying ``ImportError`` into a bare ``ValueError``, which callers
    cannot distinguish from a real configuration error. ``init --model``
    calls this up front so a missing dependency fails *init* loudly rather
    than surfacing mid-mine (or worse, being swallowed into a silent
    fallback that binds the palace to the wrong model — PR #442 review).
    """
    if not is_sentence_transformer_model(model):
        return
    if not _sentence_transformers_available():
        raise EmbeddingDependencyError(
            f"embedding_model={str(model).strip()!r} requires the "
            "sentence-transformers package, which is not installed.\n"
            "  Install it with: pip install 'mempalace[multilingual]'\n"
            "  Or revert embedding_model to a built-in value "
            "('minilm' / 'embeddinggemma')."
        )


def _resolve_torch_device(device: Optional[str]) -> str:
    """Map the configured ``embedding_device`` to a torch device string.

    The config vocabulary is ONNX-provider-shaped (``auto`` / ``cpu`` /
    ``cuda`` / ``coreml`` / ``dml``); sentence-transformers runs on torch,
    so ``coreml`` maps to ``mps`` and ``dml`` (no torch equivalent) falls
    back to CPU with a one-shot warning — the same fallback-not-fail policy
    as :func:`_resolve_providers`.
    """
    normalized = (device or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    # torch.backends.mps.is_available() returns True on Intel Macs where MPS
    # does not actually work — require arm64 as well.
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = (
        mps_backend is not None and mps_backend.is_available() and platform.machine() == "arm64"
    )

    if normalized == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    mapped = _TORCH_DEVICE_MAP.get(normalized)
    if mapped == "cuda" and cuda_ok:
        return "cuda"
    if mapped == "mps" and mps_ok:
        return "mps"
    if mapped != "cpu":
        warn_key = f"torch:{normalized}"
        if warn_key not in _WARNED:
            logger.warning(
                "embedding_device=%r is not available on the "
                "sentence-transformers (torch) path — falling back to CPU.",
                normalized,
            )
            _WARNED.add(warn_key)
    return "cpu"


class _LazySentenceTransformerEF:
    """Defers chromadb's ``SentenceTransformerEmbeddingFunction`` to first use.

    The chromadb wrapper's ``__init__`` imports torch and loads the model
    eagerly. Collections are opened on paths that never embed (MCP status,
    hooks, wake-up) and must stay inside the startup budgets, so the load is
    deferred to the first embed call — same lazy contract as the built-in
    :class:`EmbeddinggemmaONNX`. ``name()`` matches the real chromadb class
    so collections created by either open interchangeably.

    Construction failures at load time (bad model id, no network) are
    wrapped in :class:`EmbeddingModelError` with recovery options — never
    silently replaced with the default model.
    """

    @staticmethod
    def name() -> str:
        return "sentence_transformer"

    def __init__(self, model_name: str, device: str):
        self._model_name = model_name
        self._device = device
        self._inner = None
        # Shared across threads via _EF_CACHE; serialize the one-time load.
        self._load_lock = threading.Lock()

    def _load(self):
        if self._inner is not None:
            return self._inner
        with self._load_lock:
            if self._inner is not None:
                return self._inner
            ensure_model_dependencies(self._model_name)
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            try:
                inner = SentenceTransformerEmbeddingFunction(
                    model_name=self._model_name,
                    device=self._device,
                )
            except Exception as e:
                raise EmbeddingModelError(
                    f"Failed to load sentence-transformers model "
                    f"{self._model_name!r} (device={self._device!r}): {e}\n"
                    "  Check the model id and network access, or revert "
                    "embedding_model to a built-in value "
                    "('minilm' / 'embeddinggemma')."
                ) from e
            self._inner = inner
        return self._inner

    def __call__(self, input):  # noqa: A002 — ChromaDB EF protocol
        return self._load()(input)

    def embed_query(self, input):  # noqa: A002 — ChromaDB EF protocol
        """Embed query documents (ChromaDB EF protocol)."""
        return self(input)

    def embed_documents(self, input):  # noqa: A002
        """Embed a batch of documents (ChromaDB EF protocol)."""
        return self(input)


def _build_sentence_transformer_ef(model: str, device: Optional[str]):
    """Build the lazy sentence-transformers EF for ``model``.

    Dependency check runs eagerly (actionable
    :class:`EmbeddingDependencyError`, cheap metadata probe); the model
    itself downloads / loads on the first embed call via
    :class:`_LazySentenceTransformerEF`.
    """
    ensure_model_dependencies(model)

    return _LazySentenceTransformerEF(
        model_name=str(model).strip(),
        device=_resolve_torch_device(device),
    )


def _resolve_providers(device: str) -> tuple[list, str]:
    """Return ``(provider_list, effective_device)`` for ``device``.

    Falls back to CPU (with a one-shot warning) when the requested
    accelerator is not compiled into the installed ``onnxruntime``.
    """
    device = (device or "auto").strip().lower()

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for provider, name in _AUTO_ORDER:
            if provider in available:
                return ([provider, "CPUExecutionProvider"], name)
        return (["CPUExecutionProvider"], "cpu")

    requested = _PROVIDER_MAP.get(device)
    if requested is None:
        if device not in _WARNED:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred = requested[0]
    if preferred == "CPUExecutionProvider":
        return (requested, "cpu")

    if preferred not in available:
        if device not in _WARNED:
            extra = _DEVICE_EXTRA.get(device, "the matching mempalace extra for your device")
            logger.warning(
                "embedding_device=%r requested but %s is not installed — "
                "falling back to CPU. Install %s.",
                device,
                preferred,
                extra,
            )
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return (requested, device)


def _intra_op_session_options(intra_op_num_threads: int):
    """Build ORT ``SessionOptions`` capping the intra-op thread pool (#1068).

    Returns ``None`` when ``intra_op_num_threads <= 0`` so the caller leaves
    ORT at its default (≈ physical core count). ChromaDB's embedder ignores
    ``OMP_NUM_THREADS`` — ORT owns its own intra-op pool, settable only via
    ``SessionOptions`` at session construction — so a cap has to be threaded
    through here rather than via the environment.
    """
    if not intra_op_num_threads or intra_op_num_threads <= 0:
        return None
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_num_threads
    return so


def _resolve_intra_op_threads() -> int:
    """Read the configured ORT intra-op thread cap (``0`` = uncapped, #1068)."""
    try:
        from .config import MempalaceConfig

        return MempalaceConfig().embedding_threads
    except Exception:
        logger.debug("embedding_threads resolution failed; leaving ORT default", exc_info=True)
        return 0


def _build_ef_class():
    """Subclass ``ONNXMiniLM_L6_V2`` with name ``"default"``.

    Why the rename: ChromaDB 1.5 persists the EF identity on the collection
    and rejects reads that pass a differently-named EF (``onnx_mini_lm_l6_v2``
    vs ``default``). The vectors and model are identical — only the
    ``name()`` tag differs — so spoofing the name lets one EF class serve
    palaces created with ``DefaultEmbeddingFunction`` *and* palaces we
    create ourselves, with the same GPU-capable ``preferred_providers``.
    """
    from functools import cached_property

    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _MempalaceONNX(ONNXMiniLM_L6_V2):
        def __init__(self, preferred_providers=None, intra_op_num_threads=0):
            super().__init__(preferred_providers=preferred_providers)
            self._intra_op_num_threads = intra_op_num_threads

        @staticmethod
        def name() -> str:
            return "default"

        @cached_property
        def model(self):
            # Upstream builds the InferenceSession with no intra-op thread cap,
            # so ORT defaults its pool to the physical core count and a
            # background mine pins every core (#1068). Rebuild the session the
            # same way upstream does (same SessionOptions, same CoreML pruning,
            # same model path) but with our cap applied. If upstream's
            # internals shift, fall back to its uncapped build so embedding
            # still works.
            cap = getattr(self, "_intra_op_num_threads", 0)
            if not cap or cap <= 0:
                return super().model
            try:
                ort = self.ort
                providers = self._preferred_providers or ort.get_available_providers()
                providers = [p for p in providers if p != "CoreMLExecutionProvider"]
                so = ort.SessionOptions()
                so.log_severity_level = 3
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = cap
                return ort.InferenceSession(
                    os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
                    providers=providers,
                    sess_options=so,
                )
            except Exception:
                logger.warning(
                    "thread-capped ORT session build failed; using ORT defaults",
                    exc_info=True,
                )
                return super().model

    return _MempalaceONNX


# Embeddinggemma-300m ONNX (q8) — 100+ languages, MRL-truncated to 384 dims so
# it drops into existing ChromaDB collections without a schema change. Lazy:
# the model (~300 MB) downloads on first call and is cached by huggingface_hub.
_EMBEDDINGGEMMA_REPO = "onnx-community/embeddinggemma-300m-ONNX"
_EMBEDDINGGEMMA_ONNX = "model_quantized.onnx"
_EMBEDDINGGEMMA_PREFIX = "task: sentence similarity | query: "
_EMBEDDINGGEMMA_DIM = 384  # Matryoshka truncation — first 384 dims of the 768
_EMBEDDINGGEMMA_MAX_LEN = 2048
# Default docs per session.run. The ONNX graph has no internal batching,
# so one unchunked run over a repair-scale batch (5000 docs, repair.py/
# cli.py) allocates attention buffers that grow with batch size and
# superlinearly with padded length (score tensors are batch x heads x
# len^2 per layer), and the kernel OOM-kills the process (#1770). 32
# matches the internal batch size of chromadb's ONNXMiniLM_L6_V2, whose
# chunked _forward survives the same call sites. embeddinggemma's
# sentence_embedding output is attention-masked, so sub-batch padding
# does not change any row's vector.
_EMBEDDINGGEMMA_BATCH_SIZE = 32


class EmbeddinggemmaONNX:
    """ChromaDB-compatible EF using embeddinggemma-300m ONNX (q8, MRL→384d).

    Cross-lingual cosine similarity on parallel-translated text averages 0.88
    across DE/FR/HI/IT/KO/RU vs 0.35 for ``all-MiniLM-L6-v2``. Output dim is
    truncated to 384 via Matryoshka Representation Learning so the model is a
    drop-in replacement for the MiniLM-shaped 384-dim collections ChromaDB
    creates by default — same vector width, no schema change.

    Switching an existing palace from minilm → embeddinggemma still requires
    re-embedding (different vector space) — collections persist the EF name
    and ChromaDB rejects mismatched reads. Run ``mempalace repair rebuild-index``.
    """

    @staticmethod
    def name() -> str:
        # ChromaDB persists this on the collection and refuses reads with a
        # mismatched EF — that's the signal that forces users to rebuild_index
        # when switching models. Keep it stable.
        return "embeddinggemma_300m"

    def __init__(
        self,
        preferred_providers=None,
        batch_size: int = _EMBEDDINGGEMMA_BATCH_SIZE,
        intra_op_num_threads: int = 0,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._providers = (
            list(preferred_providers) if preferred_providers else ["CPUExecutionProvider"]
        )
        self._batch_size = batch_size
        self._intra_op_num_threads = intra_op_num_threads
        self._session = None
        self._tokenizer = None
        self._np = None
        self._output_idx = None
        # Instances are shared across threads via _EF_CACHE; serialize the
        # one-time model load so concurrent cold calls cannot build (and
        # transiently hold) two full model sessions.
        self._load_lock = threading.Lock()

    def _lazy_load(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            try:
                import numpy as np
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer
            except ImportError as e:
                raise ImportError(
                    "EmbeddinggemmaONNX requires huggingface_hub, tokenizers, and "
                    "numpy — these ship with mempalace core, so this error usually "
                    "means one was uninstalled or pinned to an incompatible version. "
                    "Reinstall with: pip install --upgrade --force-reinstall mempalace"
                ) from e

            logger.info(
                "Downloading %s/%s (cached after first run)…",
                _EMBEDDINGGEMMA_REPO,
                _EMBEDDINGGEMMA_ONNX,
            )
            model_path = hf_hub_download(
                _EMBEDDINGGEMMA_REPO, subfolder="onnx", filename=_EMBEDDINGGEMMA_ONNX
            )
            hf_hub_download(
                _EMBEDDINGGEMMA_REPO, subfolder="onnx", filename=_EMBEDDINGGEMMA_ONNX + "_data"
            )
            tok_path = hf_hub_download(_EMBEDDINGGEMMA_REPO, filename="tokenizer.json")

            session = ort.InferenceSession(
                model_path,
                sess_options=_intra_op_session_options(self._intra_op_num_threads),
                providers=self._providers,
            )
            out_names = [o.name for o in session.get_outputs()]
            # Model card: sentence_embedding is the pooled output (last_hidden_state
            # is the per-token output we don't want).
            output_idx = (
                out_names.index("sentence_embedding") if "sentence_embedding" in out_names else 1
            )

            tokenizer = Tokenizer.from_file(tok_path)
            tokenizer.enable_padding()
            tokenizer.enable_truncation(max_length=_EMBEDDINGGEMMA_MAX_LEN)
            self._output_idx = output_idx
            self._tokenizer = tokenizer
            self._np = np
            # Session is assigned last: the unlocked fast path above treats a
            # non-None session as "fully loaded", so every other attribute
            # must already be in place when it becomes visible.
            self._session = session

    def __call__(self, input: str | list[str] | None) -> list[list[float]]:  # noqa: A002 — ChromaDB EF protocol
        if isinstance(input, str):
            # A bare string would be iterated character by character below,
            # silently producing one garbage vector per character.
            input = [input]
        if input is None or len(input) == 0:
            # None or zero docs: nothing to embed; skip the lazy model
            # download. len() over truthiness so an array-like documents
            # sequence is not rejected by ambiguous-truth-value semantics.
            return []
        self._lazy_load()
        np = self._np
        embeddings: list[list[float]] = []
        # Tokenize and run per sub-batch, not over the whole input: padding
        # is to the longest sequence in the sub-batch, and the ONNX runtime
        # only ever holds batch_size rows of attention buffers at a time
        # (#1770).
        for start in range(0, len(input), self._batch_size):
            chunk = input[start : start + self._batch_size]
            texts = [_EMBEDDINGGEMMA_PREFIX + t for t in chunk]
            encs = self._tokenizer.encode_batch(texts)
            input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
            attention_mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
            outputs = self._session.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            sent_emb = outputs[self._output_idx][:, :_EMBEDDINGGEMMA_DIM]
            # L2-normalize so cosine similarity == dot product (matches what the
            # MTEB methodology assumes; ChromaDB's distance is configured for it).
            norms = np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-12
            embeddings.extend((sent_emb / norms).tolist())
        return embeddings

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002 — ChromaDB EF protocol
        """Embed query documents (ChromaDB EF protocol)."""
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a batch of documents (ChromaDB EF protocol)."""
        return self(input)


def get_embedding_function(device: Optional[str] = None, model: Optional[str] = None):
    """Return a cached embedding function for the requested device + model.

    ``device=None`` reads :attr:`MempalaceConfig.embedding_device`;
    ``model=None`` reads :attr:`MempalaceConfig.embedding_model`.
    The returned function is shared across calls with the same resolved
    provider list + model so we only pay model-load cost once per process.
    """
    if device is None or model is None:
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        if device is None:
            device = cfg.embedding_device
        if model is None:
            model = cfg.embedding_model

    if is_sentence_transformer_model(model):
        # torch runtime — the ONNX provider list is meaningless here, so the
        # cache key carries the resolved torch device instead.
        providers = None
        effective = _resolve_torch_device(device)
        runtime = ("torch", effective)
    else:
        providers, effective = _resolve_providers(device)
        runtime = tuple(providers)
    cache_key = (model, runtime)
    cached = _EF_CACHE.get(cache_key)  # lock-free fast path; dict.get is GIL-atomic
    if cached is not None:
        return cached
    with _EF_CACHE_LOCK:
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if providers is None:
            ef = _build_sentence_transformer_ef(model, device)
        elif model == "embeddinggemma":
            threads = _resolve_intra_op_threads()
            ef = EmbeddinggemmaONNX(preferred_providers=providers, intra_op_num_threads=threads)
        else:
            # Default: minilm (empty / legacy-unset values land here).
            threads = _resolve_intra_op_threads()
            ef_cls = _build_ef_class()
            ef = ef_cls(preferred_providers=providers, intra_op_num_threads=threads)

        _EF_CACHE[cache_key] = ef
    logger.info(
        "Embedding function initialized (model=%s device=%s runtime=%s)",
        model,
        effective,
        runtime,
    )
    return ef


def describe_device(device: Optional[str] = None, model: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved device.

    Used by the miner CLI header so users can see at a glance whether GPU
    acceleration actually engaged. Model-aware: sentence-transformers models
    run on torch, so the label is the resolved torch device (``mps`` rather
    than ``coreml``), not an ONNX provider name.
    """
    if device is None or model is None:
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        if device is None:
            device = cfg.embedding_device
        if model is None:
            model = cfg.embedding_model
    if is_sentence_transformer_model(model):
        return _resolve_torch_device(device)
    _, effective = _resolve_providers(device)
    return effective


# Probed vector widths, keyed by resolved model name. Populated once per
# process the first time an identity is resolved for a model.
_DIM_CACHE: dict = {}


def current_model_name(model: Optional[str] = None) -> str:
    """Resolve the canonical embedder model name (cheap, no model load).

    This is the configured ``embedding_model`` (``"minilm"`` /
    ``"embeddinggemma"`` / a sentence-transformers id), not the embedding
    function's internal ``name()`` (which is spoofed to ``"default"`` for
    ChromaDB compatibility). Normalization matches
    ``MempalaceConfig._normalize_embedding_model`` in BOTH branches —
    builtins lowercase, everything else verbatim — so the identity recorded
    at mine time (from config) and one derived from an explicit ``model``
    argument (e.g. ``palace set-embedder --model``) can never disagree on
    case alone.
    """
    from .config import MempalaceConfig

    if model is not None:
        return MempalaceConfig._normalize_embedding_model(model)
    return MempalaceConfig().embedding_model


def probe_dimension(device: Optional[str] = None, model: Optional[str] = None) -> int:
    """Return the embedder's output dimension by embedding a short probe.

    Model-agnostic — works for any model without a hardcoded table — and
    cached per resolved model name so the probe is paid at most once per
    process. Returns ``0`` if the probe fails (treated as "dimension unknown"
    by the identity check, so a probe failure never blocks normal operation).
    """
    name = current_model_name(model)
    cached = _DIM_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        ef = get_embedding_function(device=device, model=model)
        vectors = ef(input=["probe"])
        dim = len(vectors[0]) if vectors and vectors[0] is not None else 0
    except Exception:
        logger.debug("Embedding dimension probe failed for model=%s", name, exc_info=True)
        dim = 0
    _DIM_CACHE[name] = dim
    return dim


def get_embedder_identity(device: Optional[str] = None, model: Optional[str] = None):
    """Resolve the current embedder identity (RFC 001).

    ``model_name`` from config (cheap); ``dimension`` from a cached one-time
    probe. Returns an :class:`~mempalace.backends.base.EmbedderIdentity`.
    """
    from .backends.base import EmbedderIdentity

    return EmbedderIdentity(
        model_name=current_model_name(model),
        dimension=probe_dimension(device=device, model=model),
    )
