"""Tests for generic rule-driven table extraction (docquery._tables)."""

from __future__ import annotations

import pytest

from docquery.config import StructureRule, _DEFAULT_STRUCTURE_RULES
from docquery._ingest import build_structure_documents
from docquery._tables import (
    extract_document_tables,
    parse_table_lines,
    render_table_lines,
    split_lines_by_kind,
    table_names,
    tables_from_words,
)


def _word(x: float, y: float, text: str, w: float = 8.0):
    """Build a minimal pymupdf-style word tuple centred around x."""
    return (x - w / 2, y, x + w / 2, y + 8, text)


def _row(texts: list[str], y: float, xs: list[float] | None = None):
    xs = xs or [100.0 + i * 100 for i in range(len(texts))]
    return [_word(x, y, t) for x, t in zip(xs, texts)]


RULES = _DEFAULT_STRUCTURE_RULES


# ---------------------------------------------------------------------------
# tables_from_words — happy paths
# ---------------------------------------------------------------------------

def test_register_field_table():
    words = _row(["Bits", "Field", "Description"], y=50)
    words += _row(["31:16", "RES0", "Reserved"], y=62)
    words += _row(["7:0", "EN", "Enable"], y=74)
    (t,) = tables_from_words(words, RULES)
    assert t["kind"] == "register_fields"
    assert t["columns"] == ["bit", "field", "description"]
    assert t["rows"] == [
        {"bit": "31:16", "field": "RES0", "description": "Reserved"},
        {"bit": "7:0", "field": "EN", "description": "Enable"},
    ]
    assert t["names"] == ["RES0", "EN"]


def test_interrupt_vector_table():
    words = _row(["Position", "Acronym", "Description"], y=50)
    words += _row(["1", "WWDG", "Window watchdog"], y=62)
    words += _row(["2", "PVD", "Power voltage detect"], y=74)
    (t,) = tables_from_words(words, RULES)
    assert t["kind"] == "interrupt_table"
    assert t["columns"] == ["irq", "acronym", "description"]
    assert t["names"] == ["WWDG", "PVD"]


def test_pin_map_table():
    words = _row(["Pin", "Signal", "Notes"], y=50)
    words += _row(["PA0", "ADC_IN0", "Analog input"], y=62)
    words += _row(["PA1", "USART2_RTS", ""], y=62 + 12)
    (t,) = tables_from_words(words, RULES)
    assert t["kind"] == "pin_map"
    assert t["columns"] == ["pin", "function", "notes"]
    assert t["names"] == ["PA0", "PA1"]


def test_custom_rule_detects_novel_kind():
    rule = StructureRule(
        kind="dma_channels",
        headers=[["channel"], ["request", "peripheral"]],
        name_column="channel",
    )
    words = _row(["Channel", "Request source"], y=50, xs=[100, 250])
    words += _row(["DMA1_CH1", "ADC1"], y=62, xs=[100, 250])
    words += _row(["DMA1_CH2", "SPI1_RX"], y=74, xs=[100, 250])
    (t,) = tables_from_words(words, [rule])
    assert t["kind"] == "dma_channels"
    assert t["columns"] == ["channel", "request"]
    assert t["names"] == ["DMA1_CH1", "DMA1_CH2"]


def test_multiword_description_and_continuation_row():
    words = _row(["Bits", "Field", "Description"], y=50)
    words += _row(["7:0", "EN", "Enable"], y=62)
    words += [_word(320, 62, "bits")]                # 2nd word of description
    words += [_word(300, 74, "for"), _word(320, 74, "channels")]  # wrapped line
    words += _row(["15:8", "RES0", "Reserved"], y=86)
    (t,) = tables_from_words(words, RULES)
    assert t["rows"][0]["description"] == "Enable bits for channels"
    assert t["rows"][1]["field"] == "RES0"


def test_header_synonyms_canonicalised():
    # "Bit(s)" and "Name" must map to the same canonical keys as "Bits"/"Field".
    words = _row(["Bit(s)", "Name", "Meaning"], y=50)
    words += _row(["3", "TXE", "Transmit empty"], y=62)
    words += _row(["0", "RXNE", "Receive not empty"], y=74)
    (t,) = tables_from_words(words, RULES)
    assert t["columns"][:2] == ["bit", "field"]
    assert t["names"] == ["TXE", "RXNE"]


def test_wrapped_multiline_header():
    # Column headers wrapped over two tightly-spaced lines: "Exception number",
    # "Exception type", "Vector address". No single line shows all columns.
    words = [
        _word(100, 50, "Exception"), _word(200, 50, "Exception"), _word(300, 50, "Vector"),
        _word(100, 56, "number"), _word(200, 56, "type"), _word(300, 56, "address"),
    ]
    words += _row(["2", "NMI", "0x08"], y=76)
    words += _row(["3", "HardFault", "0x0C"], y=88)
    (t,) = tables_from_words(words, RULES)
    assert t["kind"] == "interrupt_table"
    assert t["columns"] == ["irq", "acronym", "vector_address"]
    assert t["rows"][0] == {"irq": "2", "acronym": "NMI", "vector_address": "0x08"}
    assert t["names"] == ["NMI", "HardFault"]


def test_footnote_marker_rows_are_dropped():
    words = _row(["Bits", "Field", "Description"], y=50)
    words += _row(["7:0", "EN", "Enable"], y=62)
    words += _row(["15:8", "TXE", "Transmit"], y=74)
    words += _row(["a.", "See the", "reference manual"], y=86)
    (t,) = tables_from_words(words, RULES)
    assert len(t["rows"]) == 2


# ---------------------------------------------------------------------------
# tables_from_words — false-positive gates
# ---------------------------------------------------------------------------

def test_prose_page_yields_no_tables():
    words = []
    for line, y in enumerate([50, 62, 74, 86]):
        for i, t in enumerate(["The", "bit", "field", "description", "prose"]):
            words.append(_word(80 + i * 12, y, t))
    assert tables_from_words(words, RULES) == []


def test_numbered_list_is_not_a_table():
    words = _row(["1.", "Introduction"], y=50, xs=[100, 200])
    words += _row(["2.", "Overview"], y=62, xs=[100, 200])
    assert tables_from_words(words, RULES) == []


def test_header_matching_single_group_is_rejected():
    # "Address | Value" matches no rule (no second synonym group).
    words = _row(["Address", "Value"], y=50)
    words += _row(["0x4000", "0"], y=62)
    words += _row(["0x4004", "1"], y=74)
    assert tables_from_words(words, RULES) == []


def test_one_row_table_is_rejected():
    words = _row(["Bits", "Field", "Description"], y=50)
    words += _row(["7:0", "EN", "Enable"], y=62)
    assert tables_from_words(words, RULES) == []


def test_table_ends_at_large_vertical_gap():
    words = _row(["Bits", "Field"], y=50)
    words += _row(["7:0", "EN"], y=62)
    words += _row(["15:8", "TXE"], y=74)
    # far below: an unrelated aligned pair must not join the table
    words += _row(["31:16", "RES0"], y=400)
    (t,) = tables_from_words(words, RULES)
    assert len(t["rows"]) == 2


def test_long_first_cell_rows_are_not_data_rows():
    # Prose under the header (spanning columns, many words in column 1)
    # strikes out and the candidate dies for lack of data rows.
    words = _row(["Bits", "Field", "Description"], y=50)
    for y in (62, 74):
        for i, t in enumerate(["This", "table", "is", "described", "in", "prose"]):
            words.append(_word(90 + i * 40, y, t))
    assert tables_from_words(words, RULES) == []


# ---------------------------------------------------------------------------
# render / parse round-trip
# ---------------------------------------------------------------------------

def test_render_parse_round_trip():
    tables = [
        {"kind": "register_fields", "columns": ["bit", "field", "description"],
         "rows": [{"bit": "31:16", "field": "RES0", "description": "Reserved"},
                  {"bit": "7:0", "field": "EN", "description": "Enable"}],
         "names": ["RES0", "EN"]},
        {"kind": "pin_map", "columns": ["pin", "function"],
         "rows": [{"pin": "PA0", "function": "ADC_IN0"},
                  {"pin": "PA1", "function": "USART2_RTS"}],
         "names": ["PA0", "PA1"]},
    ]
    lines = render_table_lines(tables)
    parsed = parse_table_lines(lines)
    assert parsed == [
        {k: t[k] for k in ("kind", "columns", "rows")} for t in tables
    ]


def test_render_sanitises_pipes_in_cells():
    tables = [{"kind": "register_fields", "columns": ["bit", "field"],
               "rows": [{"bit": "7:0", "field": "A|B"}], "names": []}]
    (header, row) = render_table_lines(tables)
    assert "A/B" in row
    assert parse_table_lines([header, row])[0]["rows"][0]["field"] == "A/B"


def test_parse_skips_prose_and_orphan_rows():
    assert parse_table_lines(["prose line", "ROW a | b"]) == []


def test_parse_pads_short_rows():
    parsed = parse_table_lines(["TABLE pin_map: pin | function", "ROW PA0"])
    assert parsed[0]["rows"] == [{"pin": "PA0", "function": ""}]


# ---------------------------------------------------------------------------
# ingest helpers: split_lines_by_kind / table_names / build_structure_documents
# ---------------------------------------------------------------------------

LINES = [
    "TABLE interrupt_table: irq | acronym | description",
    "ROW 2 | NMI | Non-maskable",
    "ROW 3 | HardFault | All faults",
    "TABLE pin_map: pin | function",
    "ROW PA0 | ADC_IN0",
]


def test_split_lines_by_kind_regroups_pages():
    by_kind = split_lines_by_kind({7: LINES})
    assert set(by_kind) == {"interrupt_table", "pin_map"}
    assert by_kind["interrupt_table"][7] == LINES[:3]
    assert by_kind["pin_map"][7] == LINES[3:]


def test_table_names_reads_name_column():
    assert table_names(LINES[:3], "acronym") == ["NMI", "HardFault"]
    assert table_names(LINES[:3], "missing") == []


def test_build_structure_documents_tags_kind_and_bridges_entities():
    (d,) = build_structure_documents(
        "manual.pdf", "interrupt_table", {7: LINES[:3]},
        {7: "Exception model\nprose"}, {},
        intro="Structured interrupt table:",
        names_by_page={7: ["NMI", "HardFault"]},
        entity_type="interrupt",
    )
    assert d.metadata == {
        "source": "manual.pdf", "page": 7, "kind": "interrupt_table",
        "entity_interrupt": "NMI;HardFault",
    }
    assert "Structured interrupt table:" in d.page_content
    assert "ROW 2 | NMI | Non-maskable" in d.page_content


def test_build_structure_documents_without_names_has_no_entity_tag():
    (d,) = build_structure_documents(
        "manual.pdf", "pin_map", {3: LINES[3:]}, {3: "Pinout"}, {},
        intro="Structured pin map:",
    )
    assert d.metadata == {"source": "manual.pdf", "page": 3, "kind": "pin_map"}


# ---------------------------------------------------------------------------
# extract_document_tables (real PDF via pymupdf)
# ---------------------------------------------------------------------------

def test_extract_document_tables_real_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "table.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    cols = [72, 200, 330]
    for x, text in zip(cols, ["Bits", "Field", "Description"]):
        page.insert_text((x, 100), text, fontsize=8)
    for y, row in [(112, ["31:16", "RES0", "Reserved"]),
                   (124, ["7:0", "EN", "Enable"])]:
        for x, text in zip(cols, row):
            page.insert_text((x, y), text, fontsize=8)
    doc.save(str(pdf))
    doc.close()

    result = extract_document_tables(pdf, RULES)
    assert list(result) == [1]
    assert result[1][0] == "TABLE register_fields: bit | field | description"
    assert "ROW 31:16 | RES0 | Reserved" in result[1]


def test_extract_document_tables_missing_file():
    assert extract_document_tables("/nonexistent/file.pdf", RULES) == {}


def test_extract_document_tables_no_rules():
    assert extract_document_tables("whatever.pdf", []) == {}
