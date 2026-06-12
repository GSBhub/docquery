"""Coverage instrumentation: count distinct tagged entities per type/section.

Turns "discovery fails a lot" into a number a caller can gate on. Pure metadata
scan over the knowledge collection — no embedding or LLM calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from docquery.config import ENTITY_PREFIX

if TYPE_CHECKING:
    from docquery.config import Settings

logger = logging.getLogger(__name__)


def entity_coverage(settings: "Settings", collection_name: str = "db_knowledge") -> dict[str, dict[str, Any]]:
    """Count distinct tagged entities per type, broken down by section.

    Returns ``{entity_type: {"count": <distinct names>,
    "by_section": {section_title: <distinct names>}}}``. Chunks without a section
    aggregate under the ``""`` key. Returns ``{}`` for an empty/unreachable store.
    """
    from docquery._ingest import _build_chroma
    vs = _build_chroma(settings, collection_name)
    try:
        collection = vs._collection  # type: ignore[attr-defined]
        raw = collection.get(include=["metadatas"])
    except Exception:
        logger.warning("entity_coverage: collection %r not accessible", collection_name)
        return {}

    metas = raw.get("metadatas") or []
    # entity_type -> set(names); entity_type -> section -> set(names)
    per_type: dict[str, set[str]] = {}
    per_section: dict[str, dict[str, set[str]]] = {}
    for meta in metas:
        meta = meta or {}
        section = str(meta.get("section") or "")
        for k, v in meta.items():
            if not k.startswith(ENTITY_PREFIX) or not v:
                continue
            etype = k[len(ENTITY_PREFIX):]
            names = {n.strip() for n in str(v).split(";") if n.strip()}
            if not names:
                continue
            per_type.setdefault(etype, set()).update(names)
            per_section.setdefault(etype, {}).setdefault(section, set()).update(names)

    return {
        etype: {
            "count": len(names),
            "by_section": {sec: len(s) for sec, s in sorted(per_section.get(etype, {}).items())},
        }
        for etype, names in sorted(per_type.items())
    }
