import tempfile
from pathlib import Path

from docquery.ingestion.text_loader import load


def _write_tmp(content: str, suffix: str = ".c") -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


def test_single_section_small_file():
    path = _write_tmp("int main() { return 0; }\n")
    docs = load(path)
    assert len(docs) == 1
    assert "int main" in docs[0].page_content
    assert docs[0].metadata["page"] == 1


def test_metadata_source_is_absolute():
    path = _write_tmp("hello\n")
    docs = load(path)
    assert Path(docs[0].metadata["source"]).is_absolute()


def test_large_file_splits_into_sections():
    lines = "\n".join(f"// line {i}" for i in range(250))
    path = _write_tmp(lines)
    docs = load(path)
    assert len(docs) == 3  # 100 + 100 + 50 lines → 3 sections


def test_section_page_metadata_is_line_start():
    lines = "\n".join(f"x={i}" for i in range(150))
    path = _write_tmp(lines)
    docs = load(path)
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 101


def test_empty_file_returns_one_doc():
    path = _write_tmp("")
    docs = load(path)
    assert len(docs) == 1
    assert docs[0].page_content == ""


def test_txt_and_asm_extensions_work():
    for ext in [".txt", ".asm", ".s", ".h"]:
        path = _write_tmp("content\n", suffix=ext)
        docs = load(path)
        assert len(docs) >= 1
