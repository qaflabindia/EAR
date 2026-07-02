"""LlamaIndexRetriever -- plug a LlamaIndex retriever into the Librarian.

The Librarian's `retriever` seam accepts anything with
`retrieve(query) -> list[Passage]`; this adapter maps a LlamaIndex
retriever's nodes into EAR Passages by duck-typing (NodeWithScore wrappers,
`get_content()`/`text`, and `metadata` for the source label), so it never
imports LlamaIndex itself and works with any of its retriever types --
vector, BM25, fusion. Build and configure the retriever with LlamaIndex
(`pip install 'ear[rag]'` for llama-index-core), then hand it over:

    from llama_index.core import VectorStoreIndex
    from ear.integrations.llamaindex_backend import LlamaIndexRetriever

    runtime.librarian.retriever = LlamaIndexRetriever(
        VectorStoreIndex.from_documents(documents).as_retriever(similarity_top_k=6)
    )

The Librarian's own relevance judgment and the `retrieval` audit record
apply unchanged on top -- the platform narrows, EAR's model still judges
and cites, on the record.
"""

from __future__ import annotations

from typing import Any

from ..knowledge import Passage


class LlamaIndexRetriever:
    """Adapts a LlamaIndex retriever to the Librarian's retriever seam."""

    def __init__(self, retriever: Any, source_label: str = "llamaindex") -> None:
        self._retriever = retriever
        self._source_label = source_label

    def retrieve(self, query: str) -> list[Passage]:
        passages: list[Passage] = []
        for node in self._retriever.retrieve(query):
            inner = getattr(node, "node", node)
            content = getattr(inner, "get_content", None)
            text = content() if callable(content) else getattr(inner, "text", str(inner))
            metadata = getattr(inner, "metadata", None) or {}
            source = metadata.get("file_name") or metadata.get("source") or metadata.get("file_path") or self._source_label
            passages.append(Passage(source=str(source), text=str(text)))
        return passages