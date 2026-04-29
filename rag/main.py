from __future__ import annotations

import argparse

from .chroma_store import MovieChromaStore
from .config import RAGConfig
from .rag_pipeline import MovieRAGPipeline


def create_pipeline() -> MovieRAGPipeline:
    config = RAGConfig.from_env()
    store = MovieChromaStore(
        persist_path=config.chroma_path,
        collection_name=config.chroma_collection,
        embedding_model_name=config.embedding_model_name,
    )
    return MovieRAGPipeline(store, config)


def run_once(query: str, top_k: int | None = None) -> None:
    pipeline = create_pipeline()
    response = pipeline.query(query, top_k=top_k)
    print(response["answer"])

    print("\nRetrieved chunks:")
    for result in response["results"]:
        print(f"- {result['Title']} | distance={result['Distance']:.4f}")


def run_chat() -> None:
    pipeline = create_pipeline()
    print("Movie RAG chat. Type 'quit' to exit.")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        response = pipeline.query(query)
        print(f"\n{response['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LM Studio movie RAG.")
    parser.add_argument("--query", default=None, help="Single query to run.")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    if args.query:
        run_once(args.query, args.top_k)
    else:
        run_chat()


if __name__ == "__main__":
    main()
