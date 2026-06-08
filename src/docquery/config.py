import json
import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EntityRule:
    """A structural tagging rule applied to each chunk at ingest time.

    ``pattern`` is a regex (searched with re.MULTILINE). If it has a capturing
    group, group(1) becomes the entity name; otherwise the whole match is used.
    Matching chunks are tagged with metadata ``entity_type=name`` and
    ``entity_name=<match>``, which cursor_enumerate then walks deterministically.

    Example (ARMv7-M instruction headings):
        EntityRule(name="instruction", pattern=r"^A7\\.7\\.\\d+\\s+([A-Z][A-Z0-9.]+)")
    """

    name: str
    pattern: str


def _load_entity_rules_from_env() -> list[EntityRule]:
    """Parse ENTITY_RULES (JSON list of {name, pattern}) if set, else []."""
    raw = os.getenv("ENTITY_RULES", "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [EntityRule(name=r["name"], pattern=r["pattern"]) for r in data]
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring malformed ENTITY_RULES env var: %s", exc)
        return []


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
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "120")))
    llm_num_predict: int = field(default_factory=lambda: int(os.getenv("LLM_NUM_PREDICT", "2048")))

    cursor_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("CURSOR_SCORE_THRESHOLD", "0.6")))
    cursor_max_scan: int = field(default_factory=lambda: int(os.getenv("CURSOR_MAX_SCAN", "2000")))

    # Structural enumeration rules: tag chunks at ingest so cursor_enumerate can
    # walk every instance of an entity type (instructions, registers, …).
    entity_rules: list[EntityRule] = field(default_factory=_load_entity_rules_from_env)

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
