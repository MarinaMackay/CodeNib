"""
Tests for Phase 4: Delta FAISS update optimization.

Verifies that CodeVectorStore.delta_update() correctly patches
the index for small deltas and falls back to full rebuild for large ones.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Set
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from codeminer.index.embedding.vector_store import CodeVectorStore


DIM = 8


def _md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


def _mock_embedding():
    model = MagicMock()

    def embed_documents(texts: List[str]) -> List[List[float]]:
        return [
            [(hash(t) % (2**16) + i) / (2**16) for i in range(DIM)]
            for t in texts
        ]

    def embed_query(text: str) -> List[float]:
        return embed_documents([text])[0]

    model.embed_documents.side_effect = embed_documents
    model.embed_query.side_effect = embed_query
    return model


def _build_store_with_docs(contents: List[str]):
    """Build a CodeVectorStore populated with the given contents at L2."""
    import faiss
    from langchain_community.docstore import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    embedding = _mock_embedding()
    store = CodeVectorStore.__new__(CodeVectorStore)
    store.embedding = embedding
    store.dimension = DIM
    store.index_metric = "ip"
    store.profiler = None
    store.embedding_model = "test"
    store.embedding_provider = "test"
    store.index_type = "flat"
    store.store_path = None

    documents = []
    text_embeddings = []
    hashes = {}

    for i, content in enumerate(contents):
        ch = _md5(content)
        hashes[ch] = content
        vec = embedding.embed_documents([content])[0]
        text_embeddings.append((content, vec))
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "chunk_id": i,
                    "chunk_type": "function",
                    "name": f"func_{i}",
                    "file": f"file.py",
                    "start_line": i * 3,
                    "end_line": i * 3 + 2,
                    "node_id": f"file.py:func_{i}()",
                    "level": "l2",
                    "content_hash": ch,
                },
            )
        )

    vs = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding,
        metadatas=[doc.metadata for doc in documents],
    )
    store.l2_vector_store = vs
    store.l2_index = vs.index
    store.l2_documents = documents

    l0_idx = faiss.IndexFlatIP(DIM)
    store.l0_index = l0_idx
    store.l0_vector_store = FAISS(
        embedding_function=embedding,
        index=l0_idx,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    store.l0_documents = []

    return store, documents, hashes


class TestDeltaUpdate:
    """Verify delta_update logic."""

    def test_small_delta_patches_index(self):
        """With 1 out of 10 chunks changed, delta path should be used."""
        contents = [f"def func_{i}():\n    return {i}\n" for i in range(10)]
        store, docs, hashes = _build_store_with_docs(contents)

        # Change one chunk
        new_content = "def func_0():\n    return 999\n"
        new_hash = _md5(new_content)
        old_hash = _md5(contents[0])

        # Build the full new document set
        new_docs = list(docs)
        new_embeddings = [
            np.array(store.embedding.embed_documents([d.page_content])[0], dtype=np.float32)
            for d in new_docs
        ]
        new_docs[0] = Document(
            page_content=new_content,
            metadata={**docs[0].metadata, "content_hash": new_hash},
        )
        new_embeddings[0] = np.array(
            store.embedding.embed_documents([new_content])[0], dtype=np.float32
        )

        changed_hashes = {old_hash, new_hash}
        store.delta_update(
            new_docs, new_embeddings, changed_hashes, level="l2", threshold=0.5
        )

        # Verify the documents list was updated
        assert len(store.l2_documents) == 10
        assert store.l2_documents[0].page_content == new_content

    def test_large_delta_triggers_full_rebuild(self):
        """When > threshold fraction changed, full rebuild should happen."""
        contents = [f"def f_{i}():\n    return {i}\n" for i in range(5)]
        store, docs, hashes = _build_store_with_docs(contents)

        # Change all chunks
        new_contents = [f"def f_{i}():\n    return {i*10}\n" for i in range(5)]
        new_docs = []
        new_embeddings = []
        changed_hashes = set()

        for i, content in enumerate(new_contents):
            ch = _md5(content)
            changed_hashes.add(ch)
            changed_hashes.add(_md5(contents[i]))
            vec = np.array(
                store.embedding.embed_documents([content])[0], dtype=np.float32
            )
            new_docs.append(
                Document(
                    page_content=content,
                    metadata={**docs[i].metadata, "content_hash": ch},
                )
            )
            new_embeddings.append(vec)

        # threshold=0.1 → 100% changed > 10% → full rebuild
        store.delta_update(
            new_docs, new_embeddings, changed_hashes, level="l2", threshold=0.1
        )

        assert len(store.l2_documents) == 5
        for i, doc in enumerate(store.l2_documents):
            assert doc.page_content == new_contents[i]

    def test_empty_changed_set_no_op(self):
        """With no changes, delta_update should still work."""
        contents = ["def a():\n    pass\n", "def b():\n    pass\n"]
        store, docs, hashes = _build_store_with_docs(contents)

        embeddings = [
            np.array(store.embedding.embed_documents([d.page_content])[0], dtype=np.float32)
            for d in docs
        ]

        store.delta_update(docs, embeddings, set(), level="l2")
        assert len(store.l2_documents) == 2

    def test_delta_on_empty_index_does_full_rebuild(self):
        """If the index is empty, delta_update should do a full rebuild."""
        import faiss
        from langchain_community.docstore import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        embedding = _mock_embedding()
        store = CodeVectorStore.__new__(CodeVectorStore)
        store.embedding = embedding
        store.dimension = DIM
        store.index_metric = "ip"
        store.profiler = None
        store.embedding_model = "test"
        store.embedding_provider = "test"
        store.index_type = "flat"
        store.store_path = None

        # Build truly empty L2 store (no FAISS.from_embeddings call)
        l2_idx = faiss.IndexFlatIP(DIM)
        store.l2_index = l2_idx
        store.l2_vector_store = FAISS(
            embedding_function=embedding,
            index=l2_idx,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        store.l2_documents = []
        l0_idx = faiss.IndexFlatIP(DIM)
        store.l0_index = l0_idx
        store.l0_vector_store = FAISS(
            embedding_function=embedding,
            index=l0_idx,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        store.l0_documents = []

        content = "def a():\n    pass\n"
        new_doc = Document(
            page_content=content,
            metadata={"content_hash": _md5(content), "level": "l2"},
        )
        vec = np.array(embedding.embed_documents([content])[0], dtype=np.float32)

        store.delta_update([new_doc], [vec], {_md5(content)}, level="l2")
        assert len(store.l2_documents) == 1
