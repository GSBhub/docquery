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


def tag_entities(docs: list[Document], rules: list[EntityRule]) -> list[Document]:
    """Tag chunks with structural entity metadata in place.

    For each rule, ALL distinct matches in a chunk are recorded (a single
    Unstructured chunk can contain several short entities, e.g. two instruction
    headings) under metadata key ``entity_<rule.name>`` as a ";"-joined string.
    cursor_enumerate then yields one item per distinct entity. Chunks matching
    no rule are left untouched. Returns the same list.
    """
    if not rules:
        return docs
    compiled = [(r.name, re.compile(r.pattern, re.MULTILINE)) for r in rules]
    tagged = 0
    for d in docs:
        text = d.page_content or ""
        meta = dict(d.metadata or {})
        matched = False
        for name, rx in compiled:
            found: list[str] = []
            for m in rx.finditer(text):
                ent = (m.group(1) if m.groups() else m.group(0)).strip()
                if ent and ent not in found:
                    found.append(ent)
            if found:
                meta[f"{ENTITY_PREFIX}{name}"] = ";".join(found)
                matched = True
        if matched:
            d.metadata = meta
            tagged += 1
    logger.info("tag_entities: tagged %d/%d chunks using %d rule(s)", tagged, len(docs), len(rules))
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

    tag_entities(docs, getattr(settings, "entity_rules", []))
    filtered = filter_complex_metadata(docs)
    vs.add_documents(documents=filtered)
    logger.info("Ingested %d documents into %r", len(filtered), collection_name)
    return len(filtered)
