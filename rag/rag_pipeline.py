from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .chroma_store import MovieChromaStore
from .config import RAGConfig
from .query_structurer import MovieQueryStructurer
from .reranker import MovieReranker

class MovieRAGPipeline:
    
    def __init__(self, store: MovieChromaStore, config: RAGConfig):
        self.store = store
        self.config = config
        self.reranker = (
            MovieReranker(config.reranker_model_name)
            if config.enable_reranking
            else None
        )
        self.llm = ChatOpenAI(
            model=config.llm_model_name,
            base_url=config.lm_studio_base_url,
            api_key=config.lm_studio_api_key,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        self.query_structurer = (
            MovieQueryStructurer(
                llm=self.llm,
                enable_llm=config.enable_llm_query_structuring,
            )
            if config.enable_query_structuring
            else None
        )
        self.chain = self._create_chain()

    def _create_chain(self):
        prompt = PromptTemplate.from_template(
            """You are a movie recommendation expert for a retrieval-augmented system.
You must follow these rules:
- Use only the retrieved movie context below.
- Recommend only movie titles that appear in the retrieved context.
- Do not add facts, titles, actors, years, ratings, plot details, or reasons that are not supported by the retrieved context.
- If the retrieved context does not contain enough evidence to answer the user well, say that the available context is insufficient and explain what is missing.
- If you provide recommendations, cite the context evidence briefly for each one.
- Do not rely on your own background knowledge.

User query:
{query}

Retrieved context:
{context}

Answer in English. Keep the response concise and grounded in the retrieved context."""
        )
        return prompt | self.llm | StrOutputParser()

    @staticmethod
    def format_context(results: list[dict[str, Any]]) -> str:
        blocks = []
        for result in results:
            metadata = result.get("Metadata") or {}
            details = []
            for key in [
                "Year",
                "Genres",
                "Director",
                "IMDb Rating",
                "MetaScore",
                "Duration (minutes)",
                "Certificates",
                "Star Cast",
            ]:
                if metadata.get(key) not in {None, ""}:
                    details.append(f"{key}: {metadata[key]}")
            detail_text = "\n".join(details)
            blocks.append(
                f"Title: {result['Title']}\n{detail_text}\nDescription: {result['Chunk']}"
            )
        return "\n\n".join(blocks)

    def query(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        if not query.strip():
            return {"answer": "Please enter a movie query.", "results": []}

        k = top_k or self.config.default_top_k
        fetch_k = max(k, self.config.rerank_fetch_k) if self.reranker else k

        structured_query = None
        where_filter = None
        search_query = query
        filter_applied = False
        if self.query_structurer:
            structured_query = self.query_structurer.structure(query)
            search_query = structured_query.semantic_query or query
            where_filter = self.query_structurer.to_chroma_where(structured_query)

        results = self.store.search(search_query, top_k=fetch_k, where=where_filter)
        filter_applied = bool(where_filter)
        if not results and where_filter:
            results = self.store.search(search_query, top_k=fetch_k)
            filter_applied = False

        if self.reranker:
            results = self.reranker.rerank(query, results, top_k=k)
        context = self.format_context(results)
        answer = self.chain.invoke({"query": query, "context": context})

        return {
            "answer": answer,
            "results": results,
            "structured_query": None if structured_query is None else structured_query.to_dict(),
            "metadata_filter": where_filter,
            "metadata_filter_applied": filter_applied,
            "search_query": search_query,
        }
