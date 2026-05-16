import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_LINES_PER_SECTION = 100


def load(path: str) -> list[Document]:
    source = str(Path(path).resolve())
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ValueError(f"Cannot read {path}: {e}") from e

    lines = text.splitlines(keepends=True)
    if not lines:
        logger.warning("Empty file: %s", path)
        return [Document(page_content="", metadata={"source": source, "page": 0})]

    docs = []
    for start in range(0, len(lines), _LINES_PER_SECTION):
        section = "".join(lines[start : start + _LINES_PER_SECTION])
        docs.append(Document(
            page_content=section,
            metadata={"source": source, "page": start + 1},
        ))

    logger.info("Loaded %d sections (%d lines) from %s", len(docs), len(lines), path)
    return docs
