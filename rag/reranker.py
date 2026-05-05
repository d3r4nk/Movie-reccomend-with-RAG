from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class MovieReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        pairs = [
            [query, f"Title: {result.get('Title')}\nDescription: {result.get('Chunk')}"]
            for result in results
        ]
        scores = self.model.predict(pairs)
        reranked = []
        for result, score in zip(results, scores):
            item = dict(result)
            item["RerankScore"] = float(score)
            reranked.append(item)

        return sorted(
            reranked,
            key=lambda item: item["RerankScore"],
            reverse=True,
        )[:top_k]
