

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Runtime settings for retrieval and generation."""

    processed_dir: str = "data/processed"
    chunks_path: str = "data/processed/movie_chunks_metadata.csv"
    chroma_path: str = "chroma_db"
    chroma_collection: str = "movie_chunks"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    llm_model_name: str = "qwen2.5-7b-instruct"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 768

    default_top_k: int = 5

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            processed_dir=os.getenv("PROCESSED_DIR", cls.processed_dir),
            chunks_path=os.getenv("CHUNKS_PATH", cls.chunks_path),
            chroma_path=os.getenv("CHROMA_PATH", cls.chroma_path),
            chroma_collection=os.getenv("CHROMA_COLLECTION", cls.chroma_collection),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME", cls.embedding_model_name
            ),
            lm_studio_base_url=os.getenv(
                "LM_STUDIO_BASE_URL", cls.lm_studio_base_url
            ),
            lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY", cls.lm_studio_api_key),
            llm_model_name=os.getenv("LM_STUDIO_MODEL", cls.llm_model_name),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", cls.llm_temperature)),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", cls.llm_max_tokens)),
            default_top_k=int(os.getenv("TOP_K", cls.default_top_k)),
        )
