from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .chroma_store import MovieChromaStore
from .config import RAGConfig


class MovieRAGPipeline:
    """Retrieve movie chunks from ChromaDB and generate answers with LM Studio."""

    def __init__(self, store: MovieChromaStore, config: RAGConfig):
        self.store = store
        self.config = config
        self.llm = ChatOpenAI(
            model=config.llm_model_name,
            base_url=config.lm_studio_base_url,
            api_key=config.lm_studio_api_key,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        self.chain = self._create_chain()

    def _create_chain(self):
        prompt = PromptTemplate.from_template(
            """You are a helpful movie recommendation expert.
Use only the retrieved movie context to answer the user.
If the context is not enough, say what is missing and still provide the best nearby suggestions.

User query:
{query}

Retrieved context:
{context}

Answer in English with concise recommendations and short reasons."""
        )
        return prompt | self.llm | StrOutputParser()

    @staticmethod
    def format_context(results: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"Title: {result['Title']}\nDescription: {result['Chunk']}"
            for result in results
        )

    def query(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        if not query.strip():
            return {"answer": "Please enter a movie query.", "results": []}

        k = top_k or self.config.default_top_k
        results = self.store.search(query, top_k=k)
        context = self.format_context(results)
        answer = self.chain.invoke({"query": query, "context": context})

        return {"answer": answer, "results": results}
