import logging
import re
from pathlib import Path

import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langsmith import traceable

from docquery.config import EntityRule, Settings
from docquery.embeddings.provider import get_embeddings

logger = logging.getLogger(__name__)


def tag_entities(docs: list[Document], rules: list[EntityRule]) -> list[Document]:
    """Tag chunks with structural entity metadata in place.

    For each chunk, the first rule whose pattern matches sets
    ``entity_type=<rule.name>`` and ``entity_name=<match>`` on the chunk's
    metadata, so cursor_enumerate can later walk every entity of a type.
    Chunks that match no rule are left untouched. Returns the same list.
    """
    if not rules:
        return docs
    compiled = [(r.name, re.compile(r.pattern, re.MULTILINE)) for r in rules]
    tagged = 0
    for d in docs:
        text = d.page_content or ""
        for name, rx in compiled:
            m = rx.search(text)
            if not m:
                continue
            ent = (m.group(1) if m.groups() else m.group(0)).strip()
            d.metadata = {**(d.metadata or {}), "entity_type": name, "entity_name": ent}
            tagged += 1
            break
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
            max_characters=2000,
        )
        docs.extend(loader.lazy_load())

    tag_entities(docs, getattr(settings, "entity_rules", []))
    filtered = filter_complex_metadata(docs)
    vs.add_documents(documents=filtered)
    logger.info("Ingested %d documents into %r", len(filtered), collection_name)
    return len(filtered)
