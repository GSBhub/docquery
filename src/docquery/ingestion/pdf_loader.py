import logging
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from langchain_core.documents import Document
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_OCR_THRESHOLD = 50  # chars; below this, fall back to OCR


def load(path: str) -> list[Document]:
    source = str(Path(path).resolve())
    pdf = fitz.open(path)
    docs: list[Document] = []
    ocr_count = 0

    for page in tqdm(pdf, desc="Loading PDF pages", unit="page"):
        page_num = page.number
        text = page.get_text()
        logger.debug("Page %d: %d chars", page_num, len(text.strip()))

        if len(text.strip()) < _OCR_THRESHOLD:
            logger.debug("Page %d: falling back to OCR", page_num)
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            ocr_count += 1

        docs.append(Document(page_content=text, metadata={"source": source, "page": page_num}))

    pdf.close()
    logger.info("Loaded %d pages from %s (%d via OCR)", len(docs), path, ocr_count)
    return docs
