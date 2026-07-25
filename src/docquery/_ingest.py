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


def build_structure_documents(
    source: str,
    kind: str,
    lines_by_page: dict[int, list[str]],
    page_text: dict[int, str],
    page_entities: dict[int, dict[str, list[tuple[str, str]]]],
    intro: str,
    names_by_page: dict[int, list[str]] | None = None,
    entity_type: str | None = None,
) -> list[Document]:
    """Build one ``kind``-tagged Document per page that has structure lines.

    The document leads with the page's entity heading lines (or its first text
    line as a fallback) so similarity search associates the machine-parseable
    lines (see :mod:`docquery._bitgrid` / :mod:`docquery._tables`) with the
    entity they belong to. Each heading is extended with the non-empty page
    line that follows it — reference manuals typically put the section number
    and the entity name on consecutive lines, and the *name* is what downstream
    consumers match on.

    When ``names_by_page`` and ``entity_type`` are given, each page's names are
    tagged as ``entity_<entity_type>`` metadata — the bridge that lets
    cursor_enumerate walk a structured table's rows (interrupts, pins, …) like
    any other entity type.
    """
    docs: list[Document] = []
    for page, enc_lines in sorted(lines_by_page.items()):
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
        content = "\n".join(headings + [intro] + enc_lines)
        metadata: dict = {"source": source, "page": page, "kind": kind}
        if names_by_page and entity_type:
            names = [n for n in names_by_page.get(page, []) if n.strip()]
            if names:
                _merge_entity_tag(metadata, f"{ENTITY_PREFIX}{entity_type}", names)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


_ENCODING_INTRO = "Bit-layout encoding (recovered from the document's bit-numbered diagram):"


def build_encoding_documents(
    source: str,
    encodings_by_page: dict[int, list[str]],
    page_text: dict[int, str],
    page_entities: dict[int, dict[str, list[tuple[str, str]]]],
) -> list[Document]:
    """Build encoding-grid Documents (see :func:`build_structure_documents`)."""
    return build_structure_documents(
        source, "encoding_grid", encodings_by_page, page_text, page_entities,
        intro=_ENCODING_INTRO,
    )


def owner_rule_map(
    page_entities: dict[int, dict[str, list[tuple[str, str]]]],
) -> dict[str, str]:
    """``{entity_name: rule_name}`` — the metadata tag key for each owner."""
    rule_of: dict[str, str] = {}
    for per_rule in page_entities.values():
        for rule, entries in per_rule.items():
            for name, _line in entries:
                rule_of.setdefault(name, rule)
    return rule_of


def build_owned_documents(
    source: str,
    kind: str,
    grouped: dict[tuple[int, str | None], list[str]],
    intro: str,
    rule_of: dict[str, str],
    default_rule: str,
    extra_tags: dict[tuple[int, str | None], tuple[str, list[str]]] | None = None,
) -> list[Document]:
    """One ``kind`` Document per (page, owning entity), owner tagged in metadata.

    *grouped* maps ``(page, owner)`` to the block's rendered lines. The owner
    leads the text (so similarity search associates the machine-parseable lines
    with the right entity) and is tagged as ``entity_<rule>`` metadata, letting
    consumers match a block to its entity **by name** rather than guessing from
    the page. *extra_tags* adds a second ``(entity_type, names)`` tag per group —
    used for tables whose rows name entities of their own (registers, pins, …).
    Blocks with no resolvable owner still get a document, so nothing is lost.
    """
    docs: list[Document] = []
    for (page, owner), lines in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        headings = [owner] if owner else []
        content = "\n".join(headings + [intro] + lines)
        metadata: dict = {"source": source, "page": page, "kind": kind}
        if owner:
            _merge_entity_tag(metadata, f"{ENTITY_PREFIX}{rule_of.get(owner, default_rule)}", [owner])
        if extra_tags and (extra := extra_tags.get((page, owner))):
            entity_type, names = extra
            names = [n for n in names if n and n.strip()]
            if entity_type and names:
                _merge_entity_tag(metadata, f"{ENTITY_PREFIX}{entity_type}", names)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_owned_encoding_documents(
    source: str,
    owned_by_page: dict[int, list[tuple[str, str | None]]],
    page_entities: dict[int, dict[str, list[tuple[str, str]]]],
) -> list[Document]:
    """One encoding-grid Document per (page, owning entity).

    Takes the geometry-attributed output of
    :func:`docquery._bitgrid.extract_document_encodings_owned`.
    """
    grouped: dict[tuple[int, str | None], list[str]] = {}
    for page, items in owned_by_page.items():
        for line, owner in items:
            grouped.setdefault((page, owner), []).append(line)
    return build_owned_documents(
        source, "encoding_grid", grouped, _ENCODING_INTRO,
        owner_rule_map(page_entities), default_rule="instruction",
    )


def build_owned_structure_documents(
    source: str,
    owned_by_page: dict[int, list[tuple[str, list[str], list[str], str | None]]],
    page_entities: dict[int, dict[str, list[tuple[str, str]]]],
    entity_type_by_kind: dict[str, str] | None = None,
) -> list[Document]:
    """Structure Documents per (kind, page, owning entity).

    Takes the geometry-attributed output of
    :func:`docquery._tables.extract_document_tables_owned`. Any kind benefits —
    register fields, register/memory maps, pin and interrupt tables — because
    "which entity owns this block" is the same question regardless of what the
    block holds. When a kind has a name column, its row names are additionally
    tagged as ``entity_<type>`` so enumeration can still walk them.
    """
    rule_of = owner_rule_map(page_entities)
    by_kind: dict[str, dict[tuple[int, str | None], list[str]]] = {}
    names_by_group: dict[str, dict[tuple[int, str | None], list[str]]] = {}
    for page, items in owned_by_page.items():
        for kind, lines, names, owner in items:
            by_kind.setdefault(kind, {}).setdefault((page, owner), []).extend(lines)
            if names:
                names_by_group.setdefault(kind, {}).setdefault((page, owner), []).extend(names)

    docs: list[Document] = []
    for kind, grouped in by_kind.items():
        entity_type = (entity_type_by_kind or {}).get(kind)
        extra = None
        if entity_type:
            extra = {
                key: (entity_type, names)
                for key, names in (names_by_group.get(kind) or {}).items()
            }
        docs.extend(build_owned_documents(
            source, kind, grouped,
            intro=f"Structured {kind.replace('_', ' ')} (recovered from the document's table layout):",
            rule_of=rule_of, default_rule="entity", extra_tags=extra,
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
    structure_rules = getattr(settings, "structure_rules", [])
    # Pre-chunk pass over each PDF: match headings on whole-page text (immune to
    # chunk boundaries) and propagate onto the derived chunks; recover encoding
    # bit-grids and rule-matching tables from page geometry (both flattened
    # beyond use by the text loader).
    for p in file_paths:
        if str(p).lower().endswith(".pdf"):
            from docquery._bitgrid import (
                extract_document_encodings,
                extract_document_encodings_owned,
            )
            from docquery._pdf import extract_outline, extract_page_text, persist_outline
            from docquery._tables import (
                extract_document_tables,
                extract_document_tables_owned,
                split_lines_by_kind,
                table_names,
            )
            page_text = extract_page_text(p)
            page_entities = match_page_entities(page_text, rules) if rules else {}
            if page_entities:
                # Attribute each diagram to its owning entity by reading-order
                # geometry so consumers can match by name, not by page.
                owned = extract_document_encodings_owned(p, page_entities)
                if owned:
                    docs.extend(build_owned_encoding_documents(str(p), owned, page_entities))
            else:
                encodings_by_page = extract_document_encodings(p)
                if encodings_by_page:
                    docs.extend(build_encoding_documents(
                        str(p), encodings_by_page, page_text, page_entities,
                    ))
            rules_by_kind = {r.kind: r for r in structure_rules}
            if page_entities and structure_rules:
                # Attribute each table to its owning entity by reading-order
                # geometry rather than lumping a page's tables together under the
                # page's first text line (often just a running header).
                owned_tables = extract_document_tables_owned(p, structure_rules, page_entities)
                if owned_tables:
                    docs.extend(build_owned_structure_documents(
                        str(p), owned_tables, page_entities,
                        entity_type_by_kind={
                            r.kind: (r.entity_type or r.kind)
                            for r in structure_rules if r.name_column
                        },
                    ))
            else:
                tables_by_page = extract_document_tables(p, structure_rules)
                for kind, kind_pages in split_lines_by_kind(tables_by_page).items():
                    rule = rules_by_kind.get(kind)
                    names_by_page = None
                    entity_type = None
                    if rule is not None and rule.name_column:
                        names_by_page = {
                            pg: table_names(lines, rule.name_column)
                            for pg, lines in kind_pages.items()
                        }
                        entity_type = rule.entity_type or rule.kind
                    docs.extend(build_structure_documents(
                        str(p), kind, kind_pages, page_text, page_entities,
                        intro=f"Structured {kind.replace('_', ' ')} (recovered from the document's table layout):",
                        names_by_page=names_by_page,
                        entity_type=entity_type,
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
