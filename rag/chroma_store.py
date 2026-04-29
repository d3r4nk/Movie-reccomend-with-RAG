

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd

from .embeddings import LocalEmbeddingProvider


class MovieChromaStore:
    """Persistent ChromaDB store for movie description chunks."""

    def __init__(
        self,
        persist_path: str = "chroma_db",
        collection_name: str = "movie_chunks",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedding_provider = LocalEmbeddingProvider(embedding_model_name)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _parse_metadata(value: Any, title: str) -> dict[str, str]:
        if isinstance(value, dict):
            metadata = value
        elif isinstance(value, str):
            try:
                metadata = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                metadata = {"Title": title}
        else:
            metadata = {"Title": title}

        return {"Title": str(metadata.get("Title", title))}

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_from_chunks_csv(
        self,
        chunks_path: str | Path,
        batch_size: int = 128,
        reset: bool = True,
    ) -> None:
        """Embed chunk rows and persist them in ChromaDB."""
        if reset:
            self.reset_collection()

        chunks_df = pd.read_csv(chunks_path)
        required = {"Title", "Chunk", "Metadata"}
        missing = required.difference(chunks_df.columns)
        if missing:
            raise ValueError(f"Missing required chunk columns: {sorted(missing)}")

        total = len(chunks_df)
        for start in range(0, total, batch_size):
            batch = chunks_df.iloc[start : start + batch_size]
            documents = batch["Chunk"].astype(str).tolist()
            embeddings = self.embedding_provider.encode(documents)
            ids = [f"chunk-{idx}" for idx in batch.index.tolist()]
            metadatas = [
                self._parse_metadata(row["Metadata"], row["Title"])
                for _, row in batch.iterrows()
            ]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            print(f"Indexed {min(start + batch_size, total)}/{total} chunks")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search ChromaDB for the most relevant movie chunks."""
        query_embedding = self.embedding_provider.encode([query])[0]
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        docs = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for doc, metadata, distance in zip(docs, metadatas, distances):
            results.append(
                {
                    "Title": metadata.get("Title"),
                    "Chunk": doc,
                    "Metadata": metadata,
                    "Distance": float(distance),
                }
            )

        return results

    def stats(self) -> dict[str, Any]:
        return {
            "collection": self.collection_name,
            "persist_path": self.persist_path,
            "total_chunks": self.collection.count(),
        }
