"""
Lightweight RAG pipeline: chunk → embed (TF-IDF) → retrieve → compress → answer.

Uses scikit-learn TF-IDF for zero-dependency vector search and ScaleDown for
compressing the retrieved context before sending to the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from src.core.scaledown import ScaleDownClient


# ── Chunking ────────────────────────────────────────────────────────────────── #

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks of roughly *chunk_size* characters."""
    # Clean excessive whitespace but preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    """
    Split a paper into logical sections by detecting markdown or common
    academic headings.  Returns list of (heading, body) tuples.
    """
    # Match markdown headings or ALL-CAPS section headers
    pattern = re.compile(
        r"(?:^|\n)(?:#{1,3}\s+|\d+\.?\s+)?([A-Z][A-Za-z ]{2,60})\n",
        re.MULTILINE,
    )
    splits = pattern.split(text)
    sections: list[tuple[str, str]] = []
    if not splits:
        return [("full_text", text)]
    # splits alternates: [pre, heading1, body1, heading2, body2, ...]
    if splits[0].strip():
        sections.append(("header", splits[0].strip()))
    for i in range(1, len(splits) - 1, 2):
        heading = splits[i].strip()
        body = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if body:
            sections.append((heading, body))
    return sections or [("full_text", text)]


# ── Retriever ───────────────────────────────────────────────────────────────── #

@dataclass
class Retriever:
    """TF-IDF based retriever over a corpus of text chunks."""

    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)   # source label per chunk
    top_k: int = TOP_K
    _vectorizer: Optional[TfidfVectorizer] = field(default=None, repr=False)
    _matrix: Optional[np.ndarray] = field(default=None, repr=False)

    def add_document(self, text: str, source: str = "unknown") -> int:
        """Chunk a document and add to the index. Returns number of chunks added."""
        new_chunks = chunk_text(text)
        self.chunks.extend(new_chunks)
        self.sources.extend([source] * len(new_chunks))
        self._build_index()
        return len(new_chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Return the top-k most relevant chunks for *query*.

        Each result is a dict: {text, source, score}.
        """
        k = top_k or self.top_k
        if not self.chunks or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:k]

        return [
            {
                "text": self.chunks[i],
                "source": self.sources[i],
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def _build_index(self) -> None:
        """Rebuild the TF-IDF matrix."""
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self.chunks)


# ── RAG Pipeline ────────────────────────────────────────────────────────────── #

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    1. Retrieve relevant chunks via TF-IDF similarity.
    2. Compress the retrieved context through ScaleDown.
    3. Return context + metadata ready for LLM answer generation.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        scaledown: Optional[ScaleDownClient] = None,
        top_k: int = TOP_K,
    ):
        self.retriever = retriever or Retriever(top_k=top_k)
        self.scaledown = scaledown or ScaleDownClient()

    def ingest(self, text: str, source: str = "unknown") -> int:
        """Add a document to the retriever. Returns chunk count."""
        return self.retriever.add_document(text, source)

    def compress_paper(
        self, text: str, question: str, source: str = "unknown"
    ) -> dict:
        """
        Compress entire paper section-by-section through ScaleDown,
        then ingest the compressed version.  Returns stats.

        This addresses the core requirement: compress academic papers
        using the ScaleDown API *before* chunking, so that TF-IDF
        retrieval operates on compressed (higher signal) text.
        """
        stats = {
            "sections": 0,
            "original_tokens": 0,
            "compressed_tokens": 0,
            "chunks_ingested": 0,
            "compression_used": False,
        }

        if not self.scaledown.is_configured:
            # No ScaleDown → ingest raw
            stats["chunks_ingested"] = self.ingest(text, source)
            return stats

        sections = split_sections(text)
        compressed_parts: list[str] = []

        for heading, body in sections:
            if len(body) < 200:  # too short to compress
                compressed_parts.append(f"## {heading}\n{body}")
                continue
            result = self.scaledown.compress(
                context=body,
                prompt=question,
            )
            stats["sections"] += 1
            if result["successful"]:
                stats["original_tokens"] += result["original_tokens"]
                stats["compressed_tokens"] += result["compressed_tokens"]
                stats["compression_used"] = True
                compressed_parts.append(f"## {heading}\n{result['compressed_text']}")
            else:
                compressed_parts.append(f"## {heading}\n{body}")

        compressed_text = "\n\n".join(compressed_parts)
        stats["chunks_ingested"] = self.ingest(compressed_text, source)
        return stats

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Retrieve & compress context for *question*.

        Returns:
            context          – compressed (or raw) context string
            chunks           – list of retrieved chunk dicts
            compressed       – whether compression succeeded
            compression_stats – token counts from ScaleDown
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return {
                "context": "",
                "chunks": [],
                "compressed": False,
                "compression_stats": {},
            }

        # Build a single context block with source citations
        raw_context = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )

        # Compress via ScaleDown
        compressed_context = raw_context
        compression_stats: dict = {}
        is_compressed = False

        if self.scaledown.is_configured:
            result = self.scaledown.compress(context=raw_context, prompt=question)
            if result["successful"]:
                compressed_context = result["compressed_text"]
                is_compressed = True
                compression_stats = {
                    "original_tokens": result["original_tokens"],
                    "compressed_tokens": result["compressed_tokens"],
                    "latency_ms": result["latency_ms"],
                }

        return {
            "context": compressed_context,
            "chunks": chunks,
            "compressed": is_compressed,
            "compression_stats": compression_stats,
        }
