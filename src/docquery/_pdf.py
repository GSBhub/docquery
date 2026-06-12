"""PyMuPDF-backed helpers for pre-chunk entity tagging and outline extraction.

These run a lightweight pass over the *original* PDF, independent of the
Unstructured loader's chunking. That matters because Unstructured chunks inside
``lazy_load()`` (``chunking_strategy="by_title"``), so ``^``-anchored entity
rules applied to 2000-char chunks can miss a heading split across a chunk
boundary. Matching against whole-page text here avoids that, and the page
outline gives us deterministic section scoping for cursor_enumerate.

No langchain/chromadb imports — pure pymupdf so the module is cheap to import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docquery.config import Settings

logger = logging.getLogger(__name__)

# Sidecar collection holding the document outline (one row per bookmark). Kept
# separate from db_knowledge so it never pollutes similarity search.
OUTLINE_COLLECTION = "db_outline"


def extract_page_text(pdf_path: str | Path) -> dict[int, str]:
    """Return ``{1-based page number: full page text}`` via pymupdf.

    Returns an empty dict if the file cannot be opened or has no extractable
    text (e.g. a scanned PDF) — callers then fall back to chunk-level tagging.
    """
    try:
        import fitz  # pymupdf
    except ImportError:  # pragma: no cover - dependency is declared
        logger.warning("pymupdf not available; skipping pre-chunk page extraction")
        return {}

    pages: dict[int, str] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc, start=1):
                pages[i] = page.get_text() or ""
    except Exception as exc:  # noqa: BLE001 - any pymupdf failure → fall back
        logger.warning("extract_page_text failed for %s: %s", pdf_path, exc)
        return {}
    return pages


def extract_outline(pdf_path: str | Path) -> list[dict]:
    """Return the PDF bookmark outline as an ordered list.

    Each entry is ``{"title": str, "page": int, "level": int, "order": int}`` in
    document order (``order`` is the 0-based index). Returns ``[]`` when the PDF
    has no bookmarks or cannot be opened.
    """
    try:
        import fitz  # pymupdf
    except ImportError:  # pragma: no cover
        return []

    try:
        with fitz.open(str(pdf_path)) as doc:
            toc = doc.get_toc(simple=True) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_outline failed for %s: %s", pdf_path, exc)
        return []

    outline: list[dict] = []
    for order, entry in enumerate(toc):
        # get_toc(simple=True) yields [level, title, page]
        level, title, page = entry[0], entry[1], entry[2]
        outline.append({
            "title": str(title).strip(),
            "page": int(page),
            "level": int(level),
            "order": order,
        })
    return outline


def _outline_client(settings: "Settings"):
    """Return the chromadb client backing the store (building one if needed)."""
    client = getattr(settings, "db_client", None)
    if client is not None:
        return client
    import chromadb
    if settings.db_path:
        client = chromadb.PersistentClient(path=settings.db_path)
    else:  # pragma: no cover - ephemeral has no cross-process persistence
        client = chromadb.EphemeralClient()
    settings.db_client = client
    return client


def persist_outline(settings: "Settings", outline: list[dict], source: str) -> None:
    """Store the outline in the sidecar ``db_outline`` collection.

    Rows use dummy 1-dim embeddings so no embedding model is ever invoked and the
    knowledge collection's similarity search is untouched. Ids are
    ``f"{source}:{order}"`` so re-ingesting the same file overwrites cleanly.
    """
    if not outline:
        return
    client = _outline_client(settings)
    coll = client.get_or_create_collection(OUTLINE_COLLECTION)
    coll.upsert(
        ids=[f"{source}:{e['order']}" for e in outline],
        documents=[e["title"] for e in outline],
        embeddings=[[0.0] for _ in outline],
        metadatas=[{
            "title": e["title"], "page": e["page"], "level": e["level"],
            "order": e["order"], "source": source,
        } for e in outline],
    )
    logger.info("persist_outline: stored %d entries for %s", len(outline), source)


def read_outline(settings: "Settings") -> list[dict]:
    """Return the persisted outline as ``[{"title", "page", "level"}]``, ordered.

    Returns ``[]`` when nothing was ingested with an outline.
    """
    client = _outline_client(settings)
    try:
        coll = client.get_collection(OUTLINE_COLLECTION)
    except Exception:
        return []
    raw = coll.get(include=["metadatas"])
    metas = raw.get("metadatas") or []
    rows = sorted(metas, key=lambda m: (str(m.get("source") or ""), int(m.get("order", 0))))
    return [{"title": m.get("title"), "page": m.get("page"), "level": m.get("level")}
            for m in rows]
