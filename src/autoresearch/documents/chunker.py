"""Semantic chunking for document processing."""

from __future__ import annotations

import re
from typing import List


class DocumentChunker:
    """Split documents into semantic chunks for processing."""

    @staticmethod
    def chunk(
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[str]:
        """Split text into overlapping chunks by sentences."""
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_sentences = (
                    current_chunk[-(overlap // 50) :] if overlap > 0 else []
                )
                current_chunk = overlap_sentences
                current_size = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def chunk_by_paragraphs(
        text: str,
        max_chunk_size: int = 1000,
    ) -> List[str]:
        """Split text by paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para.split())
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
