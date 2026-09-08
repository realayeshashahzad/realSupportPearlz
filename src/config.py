import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    llm_temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0")))

    vector_store_dir: Path = Field(default_factory=lambda: Path(os.getenv("VECTOR_STORE_DIR", "./data/vector_store")))
    knowledge_base_dir: Path = Field(default_factory=lambda: Path(os.getenv("KNOWLEDGE_BASE_DIR", "./data/knowledge_base")))
    
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    retrieval_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "4")))
    score_threshold: float = Field(default_factory=lambda: float(os.getenv("SCORE_THRESHOLD", "0.35")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate_keys(self):
        if not self.openai_api_key:
            raise ValueError("CRITICAL: OPENAI_API_KEY is missing from environment or .env file.")

settings = Settings()