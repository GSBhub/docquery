import logging
from pathlib import Path

import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langsmith import traceable

from docquery.config import Settings
from docquery.embeddings.provider import get_embeddings

logger = logging.getLogger(__name__)


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

    filtered = filter_complex_metadata(docs)
    vs.add_documents(documents=filtered)
    logger.info("Ingested %d documents into %r", len(filtered), collection_name)
    return len(filtered)
