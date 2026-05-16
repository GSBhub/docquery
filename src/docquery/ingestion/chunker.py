import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docquery.config import Settings

logger = logging.getLogger(__name__)


def chunk(docs: list[Document], settings: Settings | None = None) -> list[Document]:
    if settings is None:
        settings = Settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks: list[Document] = []
    for doc in docs:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_index"] = i
            logger.debug("Chunk %d/%d: %d chars (page %s)", i, len(splits), len(split.page_content), doc.metadata.get("page"))
            chunks.append(split)

    logger.info("Produced %d chunks from %d pages", len(chunks), len(docs))
    return chunks
