"""Build the persistent ChromaDB vector database from movie chunks."""

from __future__ import annotations

import argparse

from .chroma_store import MovieChromaStore
from .config import RAGConfig


def build_vector_db(chunks_path: str | None = None, reset: bool = True) -> None:
    config = RAGConfig.from_env()
    store = MovieChromaStore(
        persist_path=config.chroma_path,
        collection_name=config.chroma_collection,
        embedding_model_name=config.embedding_model_name,
    )
    store.build_from_chunks_csv(chunks_path or config.chunks_path, reset=reset)
    print(f"ChromaDB ready: {store.stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB from chunk CSV.")
    parser.add_argument("--chunks-path", default=None)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    build_vector_db(args.chunks_path, reset=not args.no_reset)


if __name__ == "__main__":
    main()
