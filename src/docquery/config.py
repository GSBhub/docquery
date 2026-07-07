import json
import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Metadata key prefix for structural entity tags: entity_<type> = ";"-joined names.
ENTITY_PREFIX = "entity_"


def language_directive(doc_language: str) -> str:
    """Prompt clause for a non-English source document; "" when unset."""
    if not doc_language.strip():
        return ""
    return (
        f"The source document is written in {doc_language.strip()}. "
        "Interpret retrieved passages and search results in that language. "
        "Respond in the language the user's question is written in, "
        "unless the user requests a specific language."
    )


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
class StructureRule:
    """Classifies a column-aligned document table into a structured kind.

    ``headers`` is a list of synonym groups; a table's header row matches when
    every group is matched by a distinct header cell. A cell matches a synonym
    when any of its words starts with it, case-insensitively — so "Bit(s)"
    matches "bit" and "Exception type" matches "type". Matched columns are
    keyed by the group's *first* synonym, so downstream consumers see stable
    column names regardless of the document's exact wording.

    ``name_column`` (a group's first synonym) marks the column whose values
    name the table's instances; when set, matched documents are tagged
    ``entity_<entity_type>`` at ingest so cursor_enumerate can walk the rows.
    ``entity_type`` defaults to ``kind``.
    """

    kind: str
    headers: list[list[str]]
    name_column: str | None = None
    entity_type: str | None = None


# Shipped defaults — generic hardware/ISA table vocabulary, not tied to any
# vendor. Fully replaceable: an env/CLI rule with the same kind overrides its
# default, and new kinds are just appended.
_DEFAULT_STRUCTURE_RULES = [
    StructureRule(
        kind="register_fields",
        headers=[["bit", "bits"],
                 ["field", "name", "symbol", "function", "meaning", "description"]],
        name_column="field",
        entity_type="register_field",
    ),
    StructureRule(
        kind="interrupt_table",
        headers=[["irq", "exception", "vector", "position", "interrupt"],
                 ["acronym", "name", "type", "source"]],
        name_column="acronym",
        entity_type="interrupt",
    ),
    StructureRule(
        kind="pin_map",
        headers=[["pin", "ball", "pad"],
                 ["function", "signal", "alternate", "af", "name", "description"]],
        name_column="pin",
        entity_type="pin",
    ),
]


def _load_structure_rules_from_env() -> list[StructureRule]:
    """Shipped defaults merged with STRUCTURE_RULES (JSON list of rule dicts).

    An env rule whose ``kind`` matches a default replaces that default;
    other env rules are appended. Malformed JSON keeps the defaults.
    """
    rules = {r.kind: r for r in _DEFAULT_STRUCTURE_RULES}
    raw = os.getenv("STRUCTURE_RULES", "").strip()
    if raw:
        try:
            for r in json.loads(raw):
                rule = StructureRule(
                    kind=r["kind"],
                    headers=[list(g) for g in r["headers"]],
                    name_column=r.get("name_column"),
                    entity_type=r.get("entity_type"),
                )
                rules[rule.kind] = rule
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring malformed STRUCTURE_RULES env var: %s", exc)
    return list(rules.values())


@dataclass
class Settings:
    embed_provider: str = field(default_factory=lambda: os.getenv("EMBED_PROVIDER", "openai"))
    embed_base_url: str = field(default_factory=lambda: os.getenv("EMBED_BASE_URL", ""))
    embed_api_key: str = field(default_factory=lambda: os.getenv("EMBED_API_KEY", ""))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    embed_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "32")))
    # Azure only: API version for the embeddings deployment (embed_model).
    embed_api_version: str = field(default_factory=lambda: os.getenv("EMBED_API_VERSION", ""))

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    # Azure only: API version for the chat deployment (llm_model).
    llm_api_version: str = field(default_factory=lambda: os.getenv("LLM_API_VERSION", ""))

    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "rag.db"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "2000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0")))
    # Natural language the source document is written in ("" = unspecified).
    doc_language: str = field(default_factory=lambda: os.getenv("DOC_LANGUAGE", ""))
    # Grounding enforcement for LLM output against retrieved document text:
    # "strict" retries/bounces on unsupported values, "warn" annotates and
    # logs, "off" disables the checks. See docquery._grounding.
    grounding: str = field(default_factory=lambda: os.getenv("GROUNDING", "strict"))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "120")))
    llm_num_predict: int = field(default_factory=lambda: int(os.getenv("LLM_NUM_PREDICT", "2048")))

    cursor_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("CURSOR_SCORE_THRESHOLD", "0.6")))
    cursor_max_scan: int = field(default_factory=lambda: int(os.getenv("CURSOR_MAX_SCAN", "2000")))

    # Structural enumeration rules: tag chunks at ingest so cursor_enumerate can
    # walk every instance of an entity type (instructions, registers, …).
    entity_rules: list[EntityRule] = field(default_factory=_load_entity_rules_from_env)

    # Table-classification rules: turn column-aligned document tables into
    # kind-tagged structure documents at ingest (register fields, interrupt
    # vectors, pin maps, …). Defaults + STRUCTURE_RULES env additions.
    structure_rules: list[StructureRule] = field(default_factory=_load_structure_rules_from_env)

    def __post_init__(self) -> None:
        if not self.embed_base_url:
            if self.embed_provider == "ollama":
                self.embed_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            elif self.embed_provider == "azure":
                self.embed_base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            else:
                self.embed_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not self.llm_base_url:
            if self.llm_provider == "ollama":
                self.llm_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            elif self.llm_provider == "azure":
                self.llm_base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            else:
                self.llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not self.embed_api_version:
            self.embed_api_version = os.getenv("OPENAI_API_VERSION", "")

        if not self.llm_api_version:
            self.llm_api_version = os.getenv("OPENAI_API_VERSION", "")

        if not self.embed_api_key:
            if self.embed_provider == "azure":
                self.embed_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            if not self.embed_api_key:
                self.embed_api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.llm_api_key:
            if self.llm_provider == "azure":
                self.llm_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            if not self.llm_api_key:
                self.llm_api_key = os.getenv("OPENAI_API_KEY", "")

        if self.grounding not in ("strict", "warn", "off"):
            logger.warning("Unknown GROUNDING value %r; using 'strict'", self.grounding)
            self.grounding = "strict"
