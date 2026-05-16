import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from langchain_core.documents import Document
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_OCR_THRESHOLD = 50  # chars; below this, fall back to OCR


def _tables_as_markdown(plumb_page) -> str:
    """Extract tables from a pdfplumber page and format as markdown pipe-tables."""
    tables = plumb_page.extract_tables()
    if not tables:
        return ""
    parts = []
    for table in tables:
        rows = [
            [str(cell or "").replace("\n", " ").strip() for cell in row]
            for row in table
            if any(cell for cell in row)
        ]
        if not rows:
            continue
        col_count = max(len(r) for r in rows)
        rows = [r + [""] * (col_count - len(r)) for r in rows]
        header = "| " + " | ".join(rows[0]) + " |"
        sep    = "| " + " | ".join(["---"] * col_count) + " |"
        body   = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
        parts.append("\n".join(filter(None, [header, sep, body])))
    return "\n\n".join(parts)


def load(path: str) -> list[Document]:
    source = str(Path(path).resolve())
    fitz_pdf = fitz.open(path)
    docs: list[Document] = []
    ocr_count = 0
    table_count = 0

    with pdfplumber.open(path) as plumb_pdf:
        pages = list(zip(fitz_pdf, plumb_pdf.pages))
        for fitz_page, plumb_page in tqdm(pages, desc="Loading PDF pages", unit="page"):
            page_num = fitz_page.number
            text = fitz_page.get_text()
            logger.debug("Page %d: %d chars", page_num, len(text.strip()))

            if len(text.strip()) < _OCR_THRESHOLD:
                logger.debug("Page %d: falling back to OCR", page_num)
                pix = fitz_page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                ocr_count += 1

            table_md = _tables_as_markdown(plumb_page)
            if table_md:
                text = text + "\n\n" + table_md
                table_count += 1
                logger.debug("Page %d: appended %d-char table block", page_num, len(table_md))

            docs.append(Document(page_content=text, metadata={"source": source, "page": page_num}))

    fitz_pdf.close()
    logger.info(
        "Loaded %d pages from %s (%d via OCR, %d with tables)",
        len(docs), path, ocr_count, table_count,
    )
    return docs
