"""Small, auditable local vector search for schema and exemplar retrieval.

The application searches 71 schema documents and 39 golden exemplars. A vector
database is unnecessary at that scale: normalized embeddings plus an exact dot
product are faster to reason about, preserve deterministic ranking, and avoid a
networked/multi-tenant data-store dependency.

Embeddings use the same all-MiniLM-L6-v2 ONNX artifact as the former backend.
The model archive is checksum-pinned, extracted as an allow-list of regular
files, and loaded locally. No question, schema, or business data leaves the
machine.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
MODEL_FILES = frozenset(
    {
        "config.json",
        "model.onnx",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    }
)
MODEL_FILE_SHA256 = {
    "config.json": "b567c7d5a55b636c95186aaf993f9a8920842b7e05a9e703e68b23cab2c3a670",
    "model.onnx": "4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer_config.json": "7702051bbc4953b94d47fa1d61b42ed4cbb3c71b501a8dd7183a823f8bea1f20",
    "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_TOKENS = 256

_model_lock = threading.RLock()
_embedder: "MiniLMEmbedder | None" = None


def _cache_root() -> Path:
    configured = (os.getenv("ASK_EMBEDDING_CACHE_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "ask-your-data" / "models"


def _model_dir() -> Path:
    return _cache_root() / MODEL_NAME / "onnx"


def _legacy_model_dir() -> Path:
    """Reuse an existing verified cache from older app releases without Chroma."""
    return Path.home() / ".cache" / "chroma" / "onnx_models" / MODEL_NAME / "onnx"


def _complete(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file() and _sha256(path / name) == digest
        for name, digest in MODEL_FILE_SHA256.items()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(destination: Path) -> None:
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "ask-your-data/enterprise"},
    )
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("embedding model archive exceeds the configured safety limit")
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("embedding model archive exceeded the configured safety limit")
            out.write(chunk)
    if _sha256(destination) != MODEL_SHA256:
        raise RuntimeError("embedding model archive failed SHA-256 verification")


def _extract_archive(archive: Path, destination: Path) -> None:
    """Extract only the six expected regular files, never archive-supplied paths."""
    destination.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            name = Path(member.name).name
            if name not in MODEL_FILES or not member.isfile():
                continue
            if name in found:
                raise RuntimeError(f"embedding model archive repeats {name}")
            source = bundle.extractfile(member)
            if source is None:
                continue
            target = destination / name
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            found.add(name)
    missing = MODEL_FILES - found
    if missing:
        raise RuntimeError(f"embedding model archive is incomplete: {sorted(missing)}")


def ensure_model() -> Path:
    """Return a complete local model directory, downloading it once if needed."""
    target = _model_dir()
    if _complete(target):
        return target
    legacy = _legacy_model_dir()
    if _complete(legacy):
        return legacy

    with _model_lock:
        if _complete(target):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="minilm-", dir=target.parent))
        try:
            archive = work / "model.tar.gz"
            extracted = work / "onnx"
            _download_archive(archive)
            _extract_archive(archive, extracted)
            if not _complete(extracted):
                raise RuntimeError("embedding model files failed SHA-256 verification")
            if target.exists():
                shutil.rmtree(target)
            os.replace(extracted, target)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return target


class MiniLMEmbedder:
    """Lazy all-MiniLM-L6-v2 inference through ONNX Runtime."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = ensure_model()
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_TOKENS)

        options = ort.SessionOptions()
        options.log_severity_level = 3
        # The default CPU arena and extended graph optimizer retain several
        # model-sized buffers. Together with the in-memory DuckDB warehouse,
        # that pushed the otherwise healthy public service above a 512 MB
        # instance limit during its first semantic-index warm-up. This corpus
        # is tiny and inference is sequential, so deterministic low-memory
        # settings are the right trade: one thread, no reusable arena/pattern,
        # and basic graph rewrites only.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )

    def __call__(self, documents: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if not documents:
            return np.empty((0, 384), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            encoded = [self._tokenizer.encode(str(text)) for text in batch]
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            token_types = np.zeros_like(input_ids, dtype=np.int64)
            hidden = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention,
                    "token_type_ids": token_types,
                },
            )[0]
            expanded = np.expand_dims(attention, -1)
            pooled = np.sum(hidden * expanded, axis=1) / np.clip(
                np.sum(expanded, axis=1), a_min=1e-9, a_max=None
            )
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            batches.append((pooled / np.clip(norms, a_min=1e-12, a_max=None)).astype(np.float32))
        return np.concatenate(batches, axis=0)


def embed(documents: Sequence[str]) -> np.ndarray:
    global _embedder
    with _model_lock:
        if _embedder is None:
            _embedder = MiniLMEmbedder()
        model = _embedder
    return model(documents)


def _embedding_cache_path(collection: str, fingerprint: str) -> Path | None:
    configured = (os.getenv("ASK_RETRIEVAL_PERSIST_DIR") or "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char for char in collection if char.isalnum() or char in "-_")
    return root / f"{safe_name}-{fingerprint}.npy"


def _load_or_embed(
    collection: str,
    fingerprint: str,
    documents: Sequence[str],
) -> np.ndarray:
    cache = _embedding_cache_path(collection, fingerprint)
    if cache and cache.is_file():
        try:
            vectors = np.load(cache, allow_pickle=False)
            if _valid_vectors(vectors, len(documents)):
                return vectors
        except (OSError, ValueError):
            pass

    vectors = embed(documents)
    if not _valid_vectors(vectors, len(documents)):
        raise RuntimeError("embedding model returned invalid or unnormalized vectors")
    if cache:
        temporary = cache.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temporary.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
    return vectors


def _valid_vectors(vectors: np.ndarray, rows: int) -> bool:
    if vectors.shape != (rows, 384) or vectors.dtype != np.float32:
        return False
    if not np.isfinite(vectors).all():
        return False
    norms = np.linalg.norm(vectors, axis=1)
    return bool(np.allclose(norms, 1.0, rtol=1e-3, atol=1e-4))


@dataclass
class LocalVectorIndex:
    """Exact, read-only cosine index with the query shape used by the app."""

    name: str
    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[dict[str, Any], ...]
    embeddings: np.ndarray
    fingerprint: str
    _alive: bool = True

    @classmethod
    def build(
        cls,
        name: str,
        rows: Sequence[dict[str, Any]],
        fingerprint: str,
    ) -> "LocalVectorIndex":
        documents = tuple(str(row["document"]) for row in rows)
        vectors = _load_or_embed(name, fingerprint, documents)
        return cls(
            name=name,
            ids=tuple(str(row["id"]) for row in rows),
            documents=documents,
            metadatas=tuple(dict(row.get("metadata") or {}) for row in rows),
            embeddings=vectors,
            fingerprint=fingerprint,
        )

    @property
    def metadata(self) -> dict[str, str]:
        return {"corpus_fingerprint": self.fingerprint, "space": "cosine"}

    def count(self) -> int:
        self._require_alive()
        return len(self.ids)

    def invalidate(self) -> None:
        """Test/operations hook representing a stale or replaced in-memory index."""
        self._alive = False

    def _require_alive(self) -> None:
        if not self._alive:
            raise RuntimeError("vector index handle is no longer valid")

    def query(
        self,
        *,
        n_results: int,
        query_texts: Sequence[str] | None = None,
        query_embeddings: Any = None,
    ) -> dict[str, list[list[Any]]]:
        self._require_alive()
        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("provide exactly one of query_texts or query_embeddings")
        vectors = (
            embed(tuple(str(text) for text in query_texts or ()))
            if query_embeddings is None
            else np.asarray(query_embeddings, dtype=np.float32)
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query embedding has the wrong shape")
        if not np.isfinite(vectors).all():
            raise ValueError("query embedding contains a non-finite value")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("query embedding must have a non-zero norm")
        vectors = vectors / np.clip(norms, a_min=1e-12, a_max=None)

        limit = max(1, min(int(n_results), len(self.ids)))
        scores = vectors @ self.embeddings.T
        ranked = np.argsort(-scores, axis=1, kind="stable")[:, :limit]
        return {
            "ids": [[self.ids[index] for index in row] for row in ranked],
            "documents": [[self.documents[index] for index in row] for row in ranked],
            "metadatas": [[self.metadatas[index] for index in row] for row in ranked],
            "distances": [[float(1.0 - scores[q, index]) for index in row]
                          for q, row in enumerate(ranked)],
        }
