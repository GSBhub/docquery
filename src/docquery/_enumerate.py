"""Deterministic entity enumeration — the grounded "list to iterate on".

At ingest, entity rules and structure rules tag chunks with ``entity_<type>``
metadata (``;``-joined names): every instance of whatever entity types those
rules define. This module turns the tags into a complete, ordered list of the
distinct instances of a type — the ground truth an extraction can iterate over
instead of asking the LLM to produce (and hallucinate) the whole list in one
shot.

It is target-agnostic: the vocabulary of types comes entirely from the
configured rules; nothing here is hardcoded to a domain. ``cursor_enumerate``
(the chat tool) and the schema-driven extraction engine both build on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docquery.config import ENTITY_PREFIX

# Structural non-entities that leak into name columns: reserved slots and
# footnote/list markers ("a.", "12."). They are never a thing to iterate.
_SKIP_NAMES = frozenset({"reserved", "name"})
_MARKER_RE = re.compile(r"^(?:[a-z]|\d{1,2})\.$")


def _is_noise(name: str) -> bool:
    return name.lower() in _SKIP_NAMES or bool(_MARKER_RE.match(name))


@dataclass
class EntityItem:
    """One distinct tagged entity instance and where it was found."""

    entity_type: str
    name: str
    content: str
    source: str | None = None
    page: int | None = None
    section: str | None = None
    section_order: int | None = None


def _section_sort_key(item: EntityItem, order: int) -> tuple[int, int, int]:
    """Document order: section, then page, then first-seen order."""
    return (
        item.section_order if isinstance(item.section_order, int) else 1_000_000,
        item.page if isinstance(item.page, int) else 1_000_000,
        order,
    )


def available_entity_types(vector_store) -> list[str]:
    """Every ``entity_<type>`` type tagged in the store, sorted."""
    try:
        collection = vector_store._collection  # type: ignore[attr-defined]
    except Exception:
        return []
    metas = collection.get(include=["metadatas"]).get("metadatas") or []
    return sorted({k[len(ENTITY_PREFIX):] for m in metas if m
                  for k in m if k.startswith(ENTITY_PREFIX)})


def enumerate_entities(
    vector_store,
    entity_type: str,
    section: str | None = None,
    kind: str | None = None,
) -> list[EntityItem]:
    """Distinct entities of *entity_type*, in document order.

    One :class:`EntityItem` per distinct name (a chunk may list several,
    ``;``-joined). ``section`` scopes to a section title / path fragment;
    ``kind`` restricts to entities carried by one structured-region kind.
    Empty list when the type is not tagged — the signal to derive or fall back.
    """
    try:
        collection = vector_store._collection  # type: ignore[attr-defined]
    except Exception:
        return []

    key = f"{ENTITY_PREFIX}{entity_type}"
    raw = collection.get(include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    items: list[EntityItem] = []
    seen: set[str] = set()
    for content, meta in zip(docs, metas, strict=False):
        meta = meta or {}
        val = meta.get(key)
        if not val:
            continue
        if kind is not None and meta.get("kind") != kind:
            continue
        if section is not None:
            sec = str(meta.get("section") or "")
            path = str(meta.get("section_path") or "")
            if section != sec and section not in path:
                continue
        for ename in str(val).split(";"):
            ename = ename.strip()
            if not ename or ename in seen or _is_noise(ename):
                continue
            seen.add(ename)
            items.append(EntityItem(
                entity_type=entity_type,
                name=ename,
                content=content,
                source=meta.get("source"),
                page=meta.get("page"),
                section=meta.get("section"),
                section_order=meta.get("section_order"),
            ))

    order = {id(it): i for i, it in enumerate(items)}
    items.sort(key=lambda it: _section_sort_key(it, order[id(it)]))
    return items
