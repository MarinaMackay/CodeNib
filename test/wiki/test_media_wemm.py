# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from codenib.wiki import (
    build_multimodal_repository_knowledge,
    build_visual_vector_index,
    create_visual_vector_store,
    update_visual_vector_store,
    visual_vector_search_results,
)
from codenib.wiki.media_wemm import WeMMVisualEmbeddingBackend


class _FakeWeMMModel:
    def __init__(self, *, dimensions=(4, 8)):
        config = SimpleNamespace(matryoshka_dimensions=list(dimensions))
        self.module = SimpleNamespace(auto_model=SimpleNamespace(config=config))
        self.calls = []
        self.staged_paths = []

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.module

    def encode_document(self, inputs, **kwargs):
        return self._encode("document", inputs, kwargs)

    def encode_query(self, inputs, **kwargs):
        return self._encode("query", inputs, kwargs)

    def _encode(self, method, inputs, kwargs):
        self.calls.append((method, list(inputs), dict(kwargs)))
        vectors = []
        dimensions = kwargs["truncate_dim"]
        for item in inputs:
            vector = [0.0] * dimensions
            if isinstance(item, dict):
                path = Path(item["image"])
                assert path.is_file()
                self.staged_paths.append(path)
                vector[0] = 1.0
            elif "cache" in item.lower():
                vector[0] = 1.0
            else:
                vector[1] = 1.0
            vectors.append(vector)
        return vectors


def _document(path, *, mime_type, payload, text="grounded visual text"):
    return {
        "artifact_path": path,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "mime_type": mime_type,
        "text": text,
    }


def test_wemm_backend_embeds_verified_images_and_text_queries(tmp_path):
    png = b"bounded-png-bytes"
    svg = b"<svg>EmbeddingCache</svg>"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "screen.png").write_bytes(png)
    (tmp_path / "docs" / "architecture.svg").write_bytes(svg)
    model = _FakeWeMMModel()
    backend = WeMMVisualEmbeddingBackend(
        repo_path=tmp_path,
        model="tencent/WeMM-Embedding-2B",
        dimensions=4,
        batch_size=2,
        model_instance=model,
    )

    documents = backend.embed_documents(
        [
            _document(
                "docs/screen.png",
                mime_type="image/png",
                payload=png,
            ),
            _document(
                "docs/architecture.svg",
                mime_type="image/svg+xml",
                payload=svg,
                text="architecture cache flow",
            ),
        ]
    )
    queries = backend.embed_queries(["cache flow"])

    assert documents == [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    assert queries == [[1.0, 0.0, 0.0, 0.0]]
    assert model.calls[0][0] == "document"
    assert model.calls[0][2] == {
        "batch_size": 2,
        "truncate_dim": 4,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert model.calls[0][1][0]["text"] == "grounded visual text"
    assert model.calls[0][1][1] == "architecture cache flow"
    assert model.calls[1][0] == "query"
    assert model.staged_paths
    assert all(not path.exists() for path in model.staged_paths)
    assert backend.document_modalities == ("image", "text")
    assert backend.query_modality == "text"


def test_wemm_vectors_materialize_and_search_in_code_vector_store(tmp_path):
    payload = b"bounded-png-bytes"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "cache.png").write_bytes(payload)
    (tmp_path / "README.md").write_text(
        "![Embedding cache architecture](docs/cache.png)",
        encoding="utf-8",
    )
    (tmp_path / "cache.py").write_text(
        "class EmbeddingCache: pass",
        encoding="utf-8",
    )
    model = _FakeWeMMModel()
    backend = WeMMVisualEmbeddingBackend(
        repo_path=tmp_path,
        model="tencent/WeMM-Embedding-2B",
        dimensions=4,
        model_instance=model,
    )
    bundle = build_multimodal_repository_knowledge(tmp_path)
    index = build_visual_vector_index(
        bundle["knowledge_view"],
        document_embedder=backend.embed_documents,
        provider=backend.provider,
        model=backend.model_name,
        dimensions=backend.dimensions,
        document_modalities=backend.document_modalities,
        query_modality=backend.query_modality,
    )

    store = create_visual_vector_store(
        index,
        query_embedder=backend.embed_queries,
    )
    update_visual_vector_store(store, index)
    results = visual_vector_search_results(store, index, "cache", limit=1)

    assert results[0]["artifact_path"] == "docs/cache.png"
    assert "cache.py" in results[0]["source_paths"]
    assert any(
        isinstance(item, dict)
        for _method, inputs, _options in model.calls
        for item in inputs
    )
    assert any(
        item == "cache" for _method, inputs, _options in model.calls for item in inputs
    )


def test_wemm_backend_rejects_changed_or_escaping_artifacts(tmp_path):
    (tmp_path / "docs").mkdir()
    payload = b"image"
    image = tmp_path / "docs" / "screen.png"
    image.write_bytes(payload)
    backend = WeMMVisualEmbeddingBackend(
        repo_path=tmp_path,
        dimensions=4,
        model_instance=_FakeWeMMModel(),
    )
    changed = _document(
        "docs/screen.png",
        mime_type="image/png",
        payload=b"other-image",
    )

    with pytest.raises(ValueError, match="content hash"):
        backend.embed_documents([changed])

    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    outside.write_bytes(payload)
    link = tmp_path / "docs" / "outside.png"
    link.symlink_to(outside)
    escaped = _document(
        "docs/outside.png",
        mime_type="image/png",
        payload=payload,
    )
    with pytest.raises(ValueError, match="inside the repository"):
        backend.embed_documents([escaped])


def test_wemm_backend_validates_model_dimensions(tmp_path):
    payload = b"image"
    image = tmp_path / "screen.png"
    image.write_bytes(payload)
    backend = WeMMVisualEmbeddingBackend(
        repo_path=tmp_path,
        dimensions=16,
        model_instance=_FakeWeMMModel(dimensions=(4, 8)),
    )

    with pytest.raises(ValueError, match="unsupported"):
        backend.embed_documents(
            [
                _document(
                    "screen.png",
                    mime_type="image/png",
                    payload=payload,
                )
            ]
        )


def test_wemm_remote_code_requires_an_immutable_revision(tmp_path):
    with pytest.raises(ValueError, match="full 40-character"):
        WeMMVisualEmbeddingBackend(
            repo_path=tmp_path,
            model="tencent/WeMM-Embedding-2B",
            dimensions=256,
            trust_remote_code=True,
        )

    with pytest.raises(ValueError, match="remote WeMM models require"):
        WeMMVisualEmbeddingBackend(
            repo_path=tmp_path,
            model="tencent/WeMM-Embedding-2B",
            dimensions=256,
        )


@pytest.mark.parametrize("batch_size", [0, 65, True])
def test_wemm_backend_rejects_invalid_batch_sizes(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch size"):
        WeMMVisualEmbeddingBackend(
            repo_path=tmp_path,
            dimensions=4,
            batch_size=batch_size,
            model_instance=_FakeWeMMModel(),
        )
