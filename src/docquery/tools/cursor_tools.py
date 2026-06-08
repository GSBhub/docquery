import hashlib
import logging

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from langchain_chroma.vectorstores import Chroma

from docquery.config import Settings

logger = logging.getLogger(__name__)


class CursorState:
    """Per-session iteration state shared by the three cursor tools.

    A single ``cursor_count`` call freezes an ordered, deduplicated list of
    matching chunks; ``cursor_current`` / ``cursor_next`` then walk that list by
    a monotonic index. Because the list is frozen and the index only advances,
    iteration can never loop back to the start or get stuck on ``next == current``.
    """

    def __init__(self) -> None:
        self.criteria: str | None = None
        self.items: list[dict] = []
        self.pos: int = 0

    def reset(self, criteria: str, items: list[dict]) -> None:
        self.criteria = criteria
        self.items = items
        self.pos = 0

    @property
    def active(self) -> bool:
        return self.criteria is not None


def _page_sort_key(item: dict) -> tuple[int, int, int]:
    """Sort by page ascending, then by original retrieval order.

    Pages that are missing or non-numeric sort last but keep a stable order.
    """
    order = item["_order"]
    page = item.get("page")
    try:
        return (0, int(page), order)
    except (TypeError, ValueError):
        return (1, 0, order)


def _format_item(state: CursorState) -> str:
    item = state.items[state.pos]
    name = item.get("entity_name")
    label = f", {name}" if name else ""
    return (
        f"Item {state.pos + 1} of {len(state.items)} "
        f"(criteria: {state.criteria!r}{label}, source: {item.get('source')}, "
        f"page: {item.get('page')})\n{item['content']}"
    )


def make_cursor_tools(vector_store: Chroma, settings: Settings | None = None) -> list[BaseTool]:
    """Build the cursor/iteration tools, all sharing one CursorState.

    Two ways to populate the cursor, both walked by cursor_current / cursor_next:
    - cursor_count(criteria): fuzzy — the chunks most *relevant* to a query.
    - cursor_enumerate(entity_type): structural — *every* chunk tagged with that
      entity type at ingest (instructions, registers, …). Deterministic and
      complete, the right tool for "walk every X in the manual".
    """
    if settings is None:
        settings = Settings()
    threshold = settings.cursor_score_threshold
    max_scan = settings.cursor_max_scan
    state = CursorState()

    @tool
    def cursor_count(criteria: str) -> str:
        """START HERE to enumerate every match of a criteria across the whole document. \
Scores all chunks against the criteria, freezes the matching ones into an ordered \
list, and returns how many there are. Then call cursor_current / cursor_next to walk \
the list one chunk at a time. Calling this again starts a fresh iteration. \
Use for tasks like "list every instruction" or "find all error codes"."""
        try:
            count = vector_store._collection.count()  # type: ignore[attr-defined]
        except Exception:
            logger.error("cursor_count: underlying ChromaDB collection not available")
            return "Cursor unavailable: the document store could not be accessed."

        k = min(count, max_scan) if count else max_scan
        if k <= 0:
            state.reset(criteria, [])
            return f"No documents are indexed, so nothing matches {criteria!r}."

        scored = vector_store.similarity_search_with_relevance_scores(criteria, k=k)
        logger.info("cursor_count: criteria=%r, scanned=%d, threshold=%.3f",
                    criteria, len(scored), threshold)

        items: list[dict] = []
        seen: set[str] = set()
        for doc, score in scored:
            if score < threshold:
                continue
            if not isinstance(doc, Document):
                continue
            meta = doc.metadata or {}
            key = doc.id or hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "id": doc.id,
                "content": doc.page_content,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "score": score,
                "_order": len(items),
            })

        items.sort(key=_page_sort_key)
        for it in items:
            it.pop("_order", None)
        state.reset(criteria, items)
        logger.info("cursor_count: criteria=%r, matched=%d", criteria, len(items))
        if not items:
            return (f"Found 0 chunks matching {criteria!r} at score threshold "
                    f"{threshold}. Try broader criteria or lower CURSOR_SCORE_THRESHOLD.")
        return (f"Found {len(items)} matching chunks for {criteria!r}. "
                f"Call cursor_current to see the first, then cursor_next to advance.")

    @tool
    def cursor_enumerate(entity_type: str) -> str:
        """Enumerate EVERY tagged entity of a given type (e.g. "instruction", "register") \
across the whole document — deterministic and complete, unlike fuzzy cursor_count. \
Requires the document to have been ingested with entity rules. Freezes one item per \
distinct entity (page-ordered); then call cursor_current / cursor_next to iterate. \
Calling this with an unknown type lists the available types."""
        try:
            collection = vector_store._collection  # type: ignore[attr-defined]
        except Exception:
            logger.error("cursor_enumerate: underlying ChromaDB collection not available")
            return "Cursor unavailable: the document store could not be accessed."

        raw = collection.get(where={"entity_type": entity_type},
                             include=["documents", "metadatas"])
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        ids = raw.get("ids") or []

        if not docs:
            all_meta = collection.get(include=["metadatas"]).get("metadatas") or []
            types = sorted({m.get("entity_type") for m in all_meta
                            if m and m.get("entity_type")})
            if types:
                return (f"No entities of type {entity_type!r}. "
                        f"Available types: {', '.join(types)}.")
            return ("No structural entities are tagged in this store. Ingest with entity "
                    "rules (Settings.entity_rules or the --entity CLI flag) to enable "
                    "cursor_enumerate.")

        items: list[dict] = []
        seen: set[str] = set()
        for cid, content, meta in zip(ids, docs, metas, strict=False):
            meta = meta or {}
            ename = meta.get("entity_name") or ""
            key = ename or cid
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "id": cid,
                "content": content,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "entity_name": ename,
                "_order": len(items),
            })
        items.sort(key=_page_sort_key)
        for it in items:
            it.pop("_order", None)
        state.reset(f"{entity_type} entities", items)
        logger.info("cursor_enumerate: type=%r, distinct entities=%d", entity_type, len(items))
        return (f"Found {len(items)} distinct {entity_type} entities. "
                f"Call cursor_current to see the first, then cursor_next to advance.")

    @tool
    def cursor_current() -> str:
        """Return the chunk at the cursor's current position. \
Call cursor_count first to start an iteration. Does not advance the cursor."""
        if not state.active:
            return "No active cursor. Call cursor_count(criteria) first to start an iteration."
        if not state.items:
            return f"No matches were found for {state.criteria!r}; nothing to show."
        return _format_item(state)

    @tool
    def cursor_next() -> str:
        """Advance the cursor to the next matching chunk and return it. \
Call cursor_count first to start an iteration. Reports when iteration is exhausted."""
        if not state.active:
            return "No active cursor. Call cursor_count(criteria) first to start an iteration."
        if not state.items:
            return f"No matches were found for {state.criteria!r}; nothing to iterate."
        if state.pos >= len(state.items) - 1:
            return (f"End of iteration: all {len(state.items)} matches for "
                    f"{state.criteria!r} have been visited.")
        state.pos += 1
        return _format_item(state)

    return [cursor_count, cursor_enumerate, cursor_current, cursor_next]
