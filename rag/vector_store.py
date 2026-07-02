"""
ChromaDB persistent vector store for Pokemon knowledge.

Manages the pokemon_knowledge collection in ChromaDB with persistent
storage at ./chroma_pokemon_kb.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb

from rag.embedder import ChromaEmbeddingFunction
from rag.chunker import chunk_all_documents

logger = logging.getLogger(__name__)

CHROMA_PATH = Path(__file__).parent.parent / "chroma_pokemon_kb"
COLLECTION_NAME = "pokemon_knowledge"


class PokemonVectorStore:
    """Persistent ChromaDB vector store for Pokemon knowledge base."""

    def __init__(self, path: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            path: Custom path for ChromaDB storage (default: ./chroma_pokemon_kb)
        """
        store_path = path or str(CHROMA_PATH)
        self.client = chromadb.PersistentClient(path=store_path)
        self.embedding_fn = ChromaEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Pokemon vector store initialized: {self.collection.count()} documents "
            f"in {store_path}"
        )

    def add_documents(self, documents: list[dict], batch_size: int = 100) -> int:
        """
        Add documents to the vector store, chunking if necessary.

        Args:
            documents: List of document dicts from ETL with 'id', 'text', 'metadata'
            batch_size: Number of documents to add per batch

        Returns:
            Number of documents added
        """
        # Chunk documents first
        chunks = chunk_all_documents(documents)

        # Separate into parallel lists for ChromaDB
        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["id"])
            texts.append(chunk["text"])
            # Ensure all metadata values are strings (ChromaDB requirement)
            metadata = {}
            for k, v in chunk["metadata"].items():
                if isinstance(v, bool):
                    metadata[k] = str(v).lower()
                elif isinstance(v, (int, float)):
                    metadata[k] = str(v)
                else:
                    metadata[k] = str(v) if v is not None else ""
            metadatas.append(metadata)

        # Add in batches
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            total_added += len(batch_ids)
            logger.debug(f"Added batch {i // batch_size + 1}: {len(batch_ids)} chunks")

        logger.info(f"Added {total_added} chunks to vector store")
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query the vector store.

        Args:
            query_text: Natural language query
            n_results: Number of results to return
            where: Metadata filter (e.g. {"pokemon_name": "pikachu"})
            where_document: Document content filter

        Returns:
            List of result dicts with 'id', 'text', 'metadata', 'distance'
        """
        kwargs = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count()) or 1,
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self.collection.query(**kwargs)

        # Flatten results into a list of dicts
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })

        return output

    def get_by_pokemon(self, pokemon_name: str) -> list[dict]:
        """Get all documents for a specific Pokemon."""
        results = self.collection.get(
            where={"pokemon_name": pokemon_name.lower()},
        )

        output = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                output.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                })

        return output

    def count(self) -> int:
        """Return the total number of documents in the store."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        # ChromaDB doesn't have a clear method; delete and recreate
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")

    def get_all_pokemon_names(self) -> list[str]:
        """Get a list of all unique Pokemon names in the store."""
        # Get all metadata
        results = self.collection.get(include=["metadatas"])
        names = set()
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                if "pokemon_name" in meta:
                    names.add(meta["pokemon_name"])
        return sorted(names)
