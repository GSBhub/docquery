import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    embed_provider: str = field(default_factory=lambda: os.getenv("EMBED_PROVIDER", "openai"))
    embed_base_url: str = field(default_factory=lambda: os.getenv("EMBED_BASE_URL", ""))
    embed_api_key: str = field(default_factory=lambda: os.getenv("EMBED_API_KEY", ""))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    embed_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "32")))

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "rag.db"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0")))

    def __post_init__(self) -> None:
        if not self.embed_base_url:
            if self.embed_provider == "ollama":
                self.embed_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            else:
                self.embed_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not self.llm_base_url:
            if self.llm_provider == "ollama":
                self.llm_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            else:
                self.llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not self.embed_api_key:
            self.embed_api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.llm_api_key:
            self.llm_api_key = os.getenv("OPENAI_API_KEY", "")
