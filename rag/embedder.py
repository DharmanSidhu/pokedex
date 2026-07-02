"""
Embedding model wrapper.

Wraps sentence-transformers/all-MiniLM-L6-v2 for use with ChromaDB,
using Apple Silicon MPS acceleration when available.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Singleton to avoid loading the model multiple times
_model_instance = None


def get_embedding_model():
    """Get or create the singleton embedding model instance."""
    global _model_instance
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer
        import torch

        # Use MPS on Apple Silicon, fallback to CPU
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon GPU) for embeddings")
        else:
            device = "cpu"
            logger.info("Using CPU for embeddings")

        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model_instance = SentenceTransformer(MODEL_NAME, device=device)
        logger.info("Embedding model loaded successfully")

    return _model_instance


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts using all-MiniLM-L6-v2.

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding

    Returns:
        NumPy array of shape (len(texts), 384)
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Returns a list of floats (384-dimensional).
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()


from chromadb import EmbeddingFunction

class ChromaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function wrapper.

    Implements the __call__ interface expected by ChromaDB collections.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a list of documents/queries for ChromaDB."""
        embeddings = embed_texts(input)
        return embeddings.tolist()

