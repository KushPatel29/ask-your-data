"""Security and determinism contracts for the small local embedding index."""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pytest

from engine import vector_index


def _rows():
    return [
        {"id": "claims", "document": "denied healthcare claims", "metadata": {"domain": "health"}},
        {"id": "salary", "document": "employee salary", "metadata": {"domain": "hr"}},
    ]


def test_exact_index_ranks_cosine_similarity_and_normalizes_queries(monkeypatch):
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    monkeypatch.setattr(vector_index, "_load_or_embed", lambda *_args: vectors)

    index = vector_index.LocalVectorIndex.build("test", _rows(), "fingerprint")
    query = np.zeros(384, dtype=np.float32)
    query[0] = 10.0  # normalization must make scale irrelevant
    result = index.query(query_embeddings=query, n_results=2)

    assert result["ids"][0] == ["claims", "salary"]
    assert result["distances"][0] == pytest.approx([0.0, 1.0])
    assert result["metadatas"][0][0]["domain"] == "health"


def test_invalidated_handle_fails_closed(monkeypatch):
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[:, 0] = 1.0
    monkeypatch.setattr(vector_index, "_load_or_embed", lambda *_args: vectors)
    index = vector_index.LocalVectorIndex.build("test", _rows(), "fingerprint")

    index.invalidate()

    with pytest.raises(RuntimeError, match="no longer valid"):
        index.count()


def test_query_rejects_non_finite_embedding(monkeypatch):
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[:, 0] = 1.0
    monkeypatch.setattr(vector_index, "_load_or_embed", lambda *_args: vectors)
    index = vector_index.LocalVectorIndex.build("test", _rows(), "fingerprint")
    query = np.zeros(384, dtype=np.float32)
    query[0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        index.query(query_embeddings=query, n_results=1)


def test_query_rejects_zero_embedding(monkeypatch):
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[:, 0] = 1.0
    monkeypatch.setattr(vector_index, "_load_or_embed", lambda *_args: vectors)
    index = vector_index.LocalVectorIndex.build("test", _rows(), "fingerprint")

    with pytest.raises(ValueError, match="non-zero norm"):
        index.query(query_embeddings=np.zeros(384, dtype=np.float32), n_results=1)


def test_archive_paths_cannot_escape_destination(tmp_path):
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in vector_index.MODEL_FILES:
            payload = name.encode()
            member = tarfile.TarInfo(f"../../outside/{name}")
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "model"
    vector_index._extract_archive(archive, destination)

    assert {path.name for path in destination.iterdir()} == vector_index.MODEL_FILES
    assert not (tmp_path / "outside").exists()


def test_archive_rejects_duplicate_expected_file(tmp_path):
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in ("first/config.json", "second/config.json"):
            payload = b"config"
            member = tarfile.TarInfo(path)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="repeats config.json"):
        vector_index._extract_archive(archive, tmp_path / "model")


def test_vector_cache_validation_rejects_wrong_shape_and_nan():
    good = np.zeros((2, 384), dtype=np.float32)
    good[:, 0] = 1.0
    assert vector_index._valid_vectors(good, 2)
    assert not vector_index._valid_vectors(good[:, :100], 2)
    good[0, 0] = np.nan
    assert not vector_index._valid_vectors(good, 2)
