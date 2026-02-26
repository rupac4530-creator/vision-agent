# backend/rag_engine.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned RAG (Retrieval Augmented Generation) Engine.

Provides document indexing and retrieval for knowledge-grounded responses.
Includes a simple TF-IDF implementation that works without external dependencies.

Usage:
    rag = SimpleRAG()
    await rag.add_documents([Document(text="...", source="guide.md")])
    results = await rag.search("how to set zones")
"""

import abc
import math
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("rag_engine")


@dataclass
class Document:
    """A document to be indexed in the RAG system."""
    text: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Chunk:
    """A chunk of a document after splitting."""
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAG(abc.ABC):
    """Abstract base class for RAG implementations."""

    @abc.abstractmethod
    async def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the index. Returns number of chunks indexed."""
        ...

    async def add_directory(self, path: str, extensions: Optional[List[str]] = None) -> int:
        """Add all matching files from a directory."""
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        exts = [e.lower() if e.startswith(".") else f".{e.lower()}"
                for e in (extensions or [".md", ".txt"])]

        files = [f for f in directory.iterdir()
                 if f.is_file() and f.suffix.lower() in exts]

        if not files:
            return 0

        docs = [Document(text=f.read_text(encoding="utf-8"), source=f.name) for f in files]
        return await self.add_documents(docs)

    @abc.abstractmethod
    async def search(self, query: str, top_k: int = 3) -> str:
        """Search the knowledge base. Returns formatted results."""
        ...

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear all indexed documents."""
        ...

    async def close(self) -> None:
        """Close resources. Override if needed."""
        pass


# ── Simple TF-IDF RAG (zero external dependencies) ────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


class SimpleRAG(RAG):
    """TF-IDF based RAG implementation — no external dependencies required.

    Good for small-to-medium knowledge bases (up to a few thousand documents).
    For production scale, swap in a vector DB implementation.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self._chunks: List[Chunk] = []
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # TF-IDF index
        self._tf: List[Dict[str, float]] = []     # per-chunk term frequencies
        self._df: Dict[str, int] = defaultdict(int)  # document frequency
        self._total_docs = 0

    async def add_documents(self, documents: List[Document]) -> int:
        """Split documents into chunks and index them."""
        total_chunks = 0

        for doc in documents:
            chunks = self._split_text(doc.text, doc.source, doc.metadata)
            for chunk in chunks:
                tokens = _tokenize(chunk.text)
                if not tokens:
                    continue

                # Compute term frequency
                tf: Dict[str, float] = defaultdict(float)
                for token in tokens:
                    tf[token] += 1
                # Normalize
                max_freq = max(tf.values()) if tf else 1
                for token in tf:
                    tf[token] /= max_freq

                # Update document frequency
                for token in set(tokens):
                    self._df[token] += 1

                self._chunks.append(chunk)
                self._tf.append(dict(tf))
                self._total_docs += 1
                total_chunks += 1

        logger.info("Indexed %d chunks from %d documents", total_chunks, len(documents))
        return total_chunks

    async def search(self, query: str, top_k: int = 3) -> str:
        """Search using TF-IDF scoring."""
        if not self._chunks:
            return "No documents indexed."

        query_tokens = _tokenize(query)
        if not query_tokens:
            return "Empty query."

        scores = []
        for i, chunk in enumerate(self._chunks):
            score = self._tfidf_score(query_tokens, i)
            if score > 0:
                scores.append((score, i))

        scores.sort(reverse=True)
        top_results = scores[:top_k]

        if not top_results:
            return "No relevant results found."

        results = []
        for rank, (score, idx) in enumerate(top_results, 1):
            chunk = self._chunks[idx]
            results.append(
                f"[{rank}] (score: {score:.3f}) [{chunk.source}]\n{chunk.text[:300]}"
            )

        return "\n\n".join(results)

    def _tfidf_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute TF-IDF similarity between query and document."""
        tf = self._tf[doc_idx]
        score = 0.0
        for token in query_tokens:
            if token in tf:
                idf = math.log((self._total_docs + 1) / (self._df.get(token, 0) + 1))
                score += tf[token] * idf
        return score

    def _split_text(self, text: str, source: str,
                    metadata: Optional[Dict] = None) -> List[Chunk]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        i = 0
        chunk_idx = 0
        while i < len(words):
            end = min(i + self._chunk_size, len(words))
            chunk_text = " ".join(words[i:end])
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_index=chunk_idx,
                metadata=metadata or {},
            ))
            chunk_idx += 1
            i += self._chunk_size - self._chunk_overlap
        return chunks

    async def clear(self) -> None:
        """Clear all indexed data."""
        self._chunks.clear()
        self._tf.clear()
        self._df.clear()
        self._total_docs = 0

    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self._chunks),
            "total_documents": self._total_docs,
            "vocabulary_size": len(self._df),
            "sources": list(set(c.source for c in self._chunks)),
        }


# ── Singleton instance ─────────────────────────────────────────────────
rag_engine = SimpleRAG()
