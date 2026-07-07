import logging

from langchain_core.tools import BaseTool, tool

from langchain_chroma.vectorstores import Chroma

from docquery.config import ENTITY_PREFIX

logger = logging.getLogger(__name__)


def _where(conds: dict) -> dict | None:
    """Build a Chroma ``where`` filter from non-empty conditions.

    Chroma rejects flat multi-key filters — two or more conditions must be
    wrapped in ``{"$and": [...]}``; a single condition is passed as-is.
    """
    conds = {k: v for k, v in conds.items() if v not in (None, "")}
    if not conds:
        return None
    if len(conds) == 1:
        return dict(conds)
    return {"$and": [{k: v} for k, v in conds.items()]}


def _entity_names(meta: dict) -> list[str]:
    """All ";"-joined entity names across a doc's ``entity_*`` metadata keys."""
    names: list[str] = []
    for key, val in meta.items():
        if key.startswith(ENTITY_PREFIX):
            names.extend(n.strip() for n in str(val).split(";") if n.strip())
    return names


def _structure_lookup(
    vector_store: Chroma,
    kind: str = "",
    page: int | None = None,
    entity: str | None = None,
    section: str | None = None,
) -> list[dict]:
    """Return kind-tagged structure docs matching the given filters."""
    try:
        collection = vector_store._collection  # type: ignore[attr-defined]
    except Exception:
        logger.error("structure_lookup: underlying ChromaDB collection not available")
        return []

    where = _where({"kind": kind, "page": page})
    kwargs = {"where": where} if where else {}
    raw = collection.get(include=["documents", "metadatas"], **kwargs)
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    results: list[dict] = []
    for content, meta in zip(docs, metas, strict=False):
        meta = meta or {}
        if not content or not meta.get("kind"):
            continue
        if section is not None:
            sec = str(meta.get("section") or "")
            path = str(meta.get("section_path") or "")
            if section != sec and section not in path:
                continue
        if entity is not None:
            names = _entity_names(meta)
            if entity not in names and not any(
                entity.lower() in n.lower() for n in names
            ):
                continue
        results.append({
            "content": content,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "kind": meta.get("kind"),
        })
    results.sort(key=lambda r: (str(r["source"]), r["page"] if isinstance(r["page"], int) else 0))
    return results


def _kind_counts(vector_store: Chroma) -> dict[str, int]:
    """``{kind: doc count}`` across the whole store (metadata scan)."""
    try:
        collection = vector_store._collection  # type: ignore[attr-defined]
    except Exception:
        return {}
    raw = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in raw.get("metadatas") or []:
        kind = (meta or {}).get("kind")
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def make_structure_lookup_tool(vector_store: Chroma) -> BaseTool:
    @tool
    def structure_lookup(kind: str = "", page: int | None = None,
                         entity: str | None = None, section: str | None = None) -> str:
        """Retrieve machine-parseable structured regions extracted from the document \
— deterministic, far more reliable than prose search for bit layouts and tables. \
Kinds include "encoding_grid" (bit-layout encodings as ENCODING lines) and \
rule-detected tables such as "register_fields", "interrupt_table", "pin_map" \
(TABLE/ROW lines). Call with NO arguments to list the kinds available in this \
document and their counts. Filter by kind, page number, entity name (e.g. an \
instruction mnemonic or interrupt name), or section title fragment."""
        if not kind and page is None and entity is None and section is None:
            counts = _kind_counts(vector_store)
            if not counts:
                return ("No structured regions were extracted from this document. "
                        "Fall back to similarity_search.")
            listing = ", ".join(f"{k} ({n} docs)" for k, n in counts.items())
            return (f"Available structured kinds: {listing}. "
                    f"Call structure_lookup(kind=...) to retrieve them.")

        results = _structure_lookup(vector_store, kind, page, entity, section)
        logger.info("structure_lookup: kind=%r page=%r entity=%r section=%r → %d docs",
                    kind, page, entity, section, len(results))
        if not results:
            counts = _kind_counts(vector_store)
            if counts:
                listing = ", ".join(f"{k} ({n} docs)" for k, n in counts.items())
                return (f"No structured regions match those filters. "
                        f"Available kinds: {listing}.")
            return ("No structured regions were extracted from this document. "
                    "Fall back to similarity_search.")
        parts = [
            f"[Source: {r['source']}, Page: {r['page']}, Kind: {r['kind']}]\n{r['content']}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)

    return structure_lookup
