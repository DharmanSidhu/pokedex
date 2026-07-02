"""
Semantic-aware document chunker.

Splits documents into chunks suitable for embedding and retrieval,
respecting sentence boundaries and keeping Pokemon data coherent.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Target chunk size in characters (roughly 300-500 tokens for MiniLM)
TARGET_CHUNK_SIZE = 1500  # chars
MAX_CHUNK_SIZE = 2500
MIN_CHUNK_SIZE = 100


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling Pokemon-specific formatting."""
    # Split on sentence endings, but not on abbreviations like "Sp. Atk"
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on newlines that separate logical blocks
    result = []
    for s in sentences:
        parts = s.split('\n')
        for part in parts:
            stripped = part.strip()
            if stripped:
                result.append(stripped)
    return result


def chunk_document(doc: dict, target_size: int = TARGET_CHUNK_SIZE) -> list[dict]:
    """
    Split a document into chunks if it exceeds the target size.

    For short documents (most Pokemon data chunks), returns the document as-is.
    For long documents (e.g., lore sections), splits at sentence boundaries.

    Args:
        doc: Document dict with 'id', 'text', 'metadata'
        target_size: Target chunk size in characters

    Returns:
        List of document dicts (possibly just the original if short enough)
    """
    text = doc["text"]

    # Short documents don't need splitting
    if len(text) <= MAX_CHUNK_SIZE:
        return [doc]

    sentences = split_into_sentences(text)
    if not sentences:
        return [doc]

    chunks = []
    current_chunk = []
    current_size = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed the target, finalize current chunk
        if current_size + sentence_len > target_size and current_chunk:
            chunk_text = "\n".join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "id": f"{doc['id']}__chunk{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_idx,
                        "is_chunked": True,
                    },
                })
                chunk_idx += 1
            current_chunk = [sentence]
            current_size = sentence_len
        else:
            current_chunk.append(sentence)
            current_size += sentence_len

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append({
                "id": f"{doc['id']}__chunk{chunk_idx}" if chunk_idx > 0 else doc["id"],
                "text": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": chunk_idx,
                    "is_chunked": chunk_idx > 0,
                },
            })

    if not chunks:
        # Fallback: return original document if splitting produced nothing useful
        return [doc]

    logger.debug(
        f"Chunked {doc['id']}: {len(text)} chars → {len(chunks)} chunks"
    )
    return chunks


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk all documents, splitting only those that are too long.

    Args:
        documents: List of document dicts from ETL

    Returns:
        List of chunked document dicts
    """
    all_chunks = []
    split_count = 0

    for doc in documents:
        chunks = chunk_document(doc)
        if len(chunks) > 1:
            split_count += 1
        all_chunks.extend(chunks)

    logger.info(
        f"Chunking complete: {len(documents)} docs → {len(all_chunks)} chunks "
        f"({split_count} documents were split)"
    )
    return all_chunks
