import logging
import re
from pathlib import Path

import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langsmith import traceable

from docquery.config import ENTITY_PREFIX, EntityRule, Settings
from docquery.embeddings.provider import get_embeddings

logger = logging.getLogger(__name__)


def _merge_entity_tag(meta: dict, key: str, names: list[str]) -> bool:
    """Union ``names`` into the ";"-joined ``meta[key]``, preserving first-seen order.

    Returns True if ``meta`` changed. Used so that two rules sharing an entity
    type (e.g. instructions under different heading conventions), or a chunk-level
    pass following a page-level pass, accumulate rather than clobber each other.
    """
    existing = [n for n in str(meta.get(key, "")).split(";") if n.strip()]
    merged = list(existing)
    for n in names:
        n = n.strip()
        if n and n not in merged:
            merged.append(n)
    if merged != existing:
        meta[key] = ";".join(merged)
        return True
    return False


def tag_entities(docs: list[Document], rules: list[EntityRule]) -> list[Document]:
    """Tag chunks with structural entity metadata in place.

    For each rule, ALL distinct matches in a chunk are recorded (a single
    Unstructured chunk can contain several short entities, e.g. two instruction
    headings) under metadata key ``entity_<rule.name>`` as a ";"-joined string.
    Multiple rules sharing an entity type are UNIONed (not overwritten), so the
    same key can be fed by several heading conventions. cursor_enumerate then
    yields one item per distinct entity. Chunks matching no rule are left
    untouched. Returns the same list.
    """
    if not rules:
        return docs
    compiled = [(r.name, re.compile(r.pattern, re.MULTILINE)) for r in rules]
    tagged = 0
    for d in docs:
        text = d.page_content or ""
        meta = dict(d.metadata or {})
        # Accumulate per entity-type first so rules sharing a name union.
        found_by_name: dict[str, list[str]] = {}
        for name, rx in compiled:
            bucket = found_by_name.setdefault(name, [])
            for m in rx.finditer(text):
                ent = (m.group(1) if m.groups() else m.group(0)).strip()
                if ent and ent not in bucket:
                    bucket.append(ent)
        matched = False
        for name, found in found_by_name.items():
            if found and _merge_entity_tag(meta, f"{ENTITY_PREFIX}{name}", found):
                matched = True
        if matched:
            d.metadata = meta
            tagged += 1
    logger.info("tag_entities: tagged %d/%d chunks using %d rule(s)", tagged, len(docs), len(rules))
    return docs


def _normalize_pages(docs: list[Document]) -> None:
    """Ensure each chunk carries a ``page`` key.

    langchain-unstructured emits ``page_number``; docquery's cursor/keyword tools
    and consumers read ``page``. Without this, page-ordering silently degrades to
    insertion order on real ingests. In place.
    """
    for d in docs:
        meta = d.metadata or {}
        if meta.get("page") is None and meta.get("page_number") is not None:
            meta["page"] = meta["page_number"]
            d.metadata = meta


def match_page_entities(
    page_text: dict[int, str],
    rules: list[EntityRule],
) -> dict[int, dict[str, list[tuple[str, str]]]]:
    """Match entity rules against whole-page text (pre-chunk).

    Returns ``{page: {rule_name: [(entity_name, full_matched_line)]}}``. Matching
    on full pages — not chunks — means ``^``-anchored headings are never lost to a
    chunk boundary. The full matched text is kept so propagation can locate the
    chunk that actually contains the heading.
    """
    if not rules:
        return {}
    compiled = [(r.name, re.compile(r.pattern, re.MULTILINE)) for r in rules]
    result: dict[int, dict[str, list[tuple[str, str]]]] = {}
    for page, text in page_text.items():
        text = text or ""
        per_rule: dict[str, list[tuple[str, str]]] = {}
        for name, rx in compiled:
            bucket = per_rule.setdefault(name, [])
            seen = {ent for ent, _ in bucket}
            for m in rx.finditer(text):
                ent = (m.group(1) if m.groups() else m.group(0)).strip()
                if ent and ent not in seen:
                    seen.add(ent)
                    bucket.append((ent, m.group(0).strip()))
        per_rule = {k: v for k, v in per_rule.items() if v}
        if per_rule:
            result[page] = per_rule
    return result


def _source_matches(meta: dict, source: str) -> bool:
    s = meta.get("source")
    if s is None:
        return False
    return s == source or Path(str(s)).name == Path(source).name


def propagate_page_entities(
    docs: list[Document],
    page_entities: dict[int, dict[str, list[tuple[str, str]]]],
    source: str,
) -> int:
    """Assign page-level entity matches onto the chunks of that page.

    For each (page, rule, entity), tag the chunk that contains the full matched
    heading line; failing that, the chunk containing the entity name; failing
    that (heading split across a boundary), the page's first chunk. Tags are
    unioned via ``_merge_entity_tag``, so this never clobbers chunk-level tags.
    Returns the number of (chunk, tag) assignments made.
    """
    # index chunks by page for this source, preserving order
    by_page: dict[object, list[Document]] = {}
    for d in docs:
        meta = d.metadata or {}
        if _source_matches(meta, source):
            by_page.setdefault(meta.get("page"), []).append(d)

    assigned = 0
    for page, per_rule in page_entities.items():
        chunks = by_page.get(page) or []
        if not chunks:
            continue
        for name, entities in per_rule.items():
            key = f"{ENTITY_PREFIX}{name}"
            for ename, line in entities:
                target = next((c for c in chunks if line and line in (c.page_content or "")), None)
                if target is None:
                    target = next((c for c in chunks if ename in (c.page_content or "")), None)
                if target is None:
                    target = chunks[0]
                meta = dict(target.metadata or {})
                if _merge_entity_tag(meta, key, [ename]):
                    target.metadata = meta
                    assigned += 1
    return assigned


def _outline_paths(outline: list[dict]) -> list[dict]:
    """Annotate each outline entry with a root→leaf ``path`` string.

    Walks the document-ordered outline keeping a level stack, so a level-3
    heading carries its enclosing level-1/level-2 titles as ``"a > b > c"``.
    """
    annotated: list[dict] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    for entry in outline:
        level = entry["level"]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, entry["title"]))
        path = " > ".join(t for _, t in stack)
        annotated.append({**entry, "path": path})
    return annotated


def assign_sections(docs: list[Document], outline: list[dict], source: str) -> int:
    """Tag each chunk with the section it falls under, from the PDF outline.

    For a chunk on page P, the section is the most-recent outline entry (in
    document order) whose ``page <= P`` — i.e. the deepest heading active at that
    page. Adds scalar metadata ``section`` (title), ``section_path`` (root→leaf),
    and ``section_order`` (outline index, for deterministic sorting). Chunks
    before the first bookmark are left unsectioned. Returns chunks sectioned.
    """
    if not outline:
        return 0
    annotated = _outline_paths(outline)
    sectioned = 0
    for d in docs:
        meta = d.metadata or {}
        if not _source_matches(meta, source):
            continue
        page = meta.get("page")
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            continue
        current = None
        for entry in annotated:
            if entry["page"] <= page_num:
                current = entry
            else:
                break
        if current is None:
            continue
        meta = dict(meta)
        meta["section"] = current["title"]
        meta["section_path"] = current["path"]
        meta["section_order"] = current["order"]
        d.metadata = meta
        sectioned += 1
    return sectioned


def build_encoding_documents(
    source: str,
    encodings_by_page: dict[int, list[str]],
    page_text: dict[int, str],
    page_entities: dict[int, dict[str, list[tuple[str, str]]]],
) -> list[Document]:
    """Build one Document per page that has extracted encoding bit-grids.

    The document leads with the page's entity heading lines (or its first text
    line as a fallback) so similarity search associates the ENCODING lines with
    the instruction they belong to; the lines themselves are machine-parseable
    (see :mod:`docquery._bitgrid`). Each heading is extended with the non-empty
    page line that follows it — ISA manuals typically put the section number and
    the instruction name on consecutive lines, and the *name* is what downstream
    consumers match on.
    """
    docs: list[Document] = []
    for page, enc_lines in sorted(encodings_by_page.items()):
        text_lines = [ln.strip() for ln in (page_text.get(page) or "").splitlines()]
        headings: list[str] = []

        def _add_with_continuation(line: str) -> None:
            if line and line not in headings:
                headings.append(line)
            stripped = line.strip()
            if stripped in text_lines:
                idx = text_lines.index(stripped)
                follow = next((ln for ln in text_lines[idx + 1:idx + 4] if ln), "")
                if follow and follow not in headings:
                    headings.append(follow)

        for entries in (page_entities.get(page) or {}).values():
            for _, line in entries:
                _add_with_continuation(line)
        if not headings:
            first = next((ln for ln in text_lines if ln), "")
            if first:
                headings.append(first)
        content = "\n".join(
            headings
            + ["Instruction encoding bit layout (from the manual's encoding diagram):"]
            + enc_lines
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": source, "page": page, "kind": "encoding_grid"},
        ))
    return docs


def _build_chroma(settings: Settings, collection_name: str = "db_knowledge") -> Chroma:
    """Build (or reuse) the Chroma wrapper on settings.vs."""
    if getattr(settings, "vs", None) is not None:
        return settings.vs

    if settings.db_path:
        client = chromadb.PersistentClient(path=settings.db_path)
    else:
        client = chromadb.EphemeralClient()

    settings.db_client = client
    settings.vs = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embeddings(settings),
    )
    return settings.vs


@traceable(name="ingest_documents")
def ingest_documents(
    items: list,
    settings: Settings,
    collection_name: str = "db_knowledge",
) -> int:
    """Load files (str/Path) and/or pre-loaded Documents into the vector store.

    Returns the number of documents added.
    """
    vs = _build_chroma(settings, collection_name)

    file_paths = [i for i in items if isinstance(i, (str, Path))]
    pre_loaded = [i for i in items if isinstance(i, Document)]

    docs: list[Document] = list(pre_loaded)

    if file_paths:
        from langchain_unstructured import UnstructuredLoader
        loader = UnstructuredLoader(
            [str(p) for p in file_paths],
            chunking_strategy="by_title",
            max_characters=settings.chunk_size,
            # Overlap carries the tail of each chunk into the next so a heading
            # (e.g. an instruction number) split across a boundary still appears
            # intact in one chunk — important for entity enumeration.
            overlap=settings.chunk_overlap,
            overlap_all=bool(settings.chunk_overlap),
        )
        docs.extend(loader.lazy_load())

    _normalize_pages(docs)

    rules = getattr(settings, "entity_rules", [])
    # Pre-chunk pass over each PDF: match headings on whole-page text (immune to
    # chunk boundaries) and propagate onto the derived chunks; recover encoding
    # bit-grids from page geometry (flattened beyond use by the text loader).
    for p in file_paths:
        if str(p).lower().endswith(".pdf"):
            from docquery._bitgrid import extract_document_encodings
            from docquery._pdf import extract_outline, extract_page_text, persist_outline
            page_text = extract_page_text(p)
            page_entities = match_page_entities(page_text, rules) if rules else {}
            encodings_by_page = extract_document_encodings(p)
            if encodings_by_page:
                docs.extend(build_encoding_documents(
                    str(p), encodings_by_page, page_text, page_entities,
                ))
            if page_entities:
                propagate_page_entities(docs, page_entities, str(p))
            outline = extract_outline(p)
            if outline:
                assign_sections(docs, outline, str(p))
                persist_outline(settings, outline, str(p))

    # Chunk-level pass: fallback for pre-loaded Documents, non-PDF sources, and
    # scanned PDFs where pymupdf sees no text. Unions with the page-level tags.
    tag_entities(docs, rules)
    filtered = filter_complex_metadata(docs)
    vs.add_documents(documents=filtered)
    logger.info("Ingested %d documents into %r", len(filtered), collection_name)
    return len(filtered)
