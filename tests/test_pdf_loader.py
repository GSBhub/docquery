from unittest.mock import MagicMock, patch

import pytest

from docquery.ingestion.pdf_loader import _tables_as_markdown


def _mock_page(tables):
    page = MagicMock()
    page.extract_tables.return_value = tables
    return page


# ---------------------------------------------------------------------------
# _tables_as_markdown unit tests
# ---------------------------------------------------------------------------

def test_no_tables_returns_empty():
    assert _tables_as_markdown(_mock_page([])) == ""


def test_all_empty_rows_skipped():
    # Table whose rows are all-None/empty — should produce no output
    page = _mock_page([[["", None, ""], [None, None, None]]])
    assert _tables_as_markdown(page) == ""


def test_single_row_table_header_only():
    # A single row becomes a header with a separator and no body rows
    page = _mock_page([[["Col A", "Col B"]]])
    md = _tables_as_markdown(page)
    assert "| Col A | Col B |" in md
    assert "| --- | --- |" in md


def test_simple_two_row_table():
    page = _mock_page([[["opA", "opB", "Instruction"],
                        ["0000", "0000", "NOP"],
                        ["0001", "0000", "YIELD"]]])
    md = _tables_as_markdown(page)
    assert "| opA | opB | Instruction |" in md
    assert "| --- | --- | --- |" in md
    assert "| 0000 | 0000 | NOP |" in md
    assert "| 0001 | 0000 | YIELD |" in md


def test_arm_nop_bit_table():
    # Simulates a pdfplumber extraction of the ARM NOP T1 bit-field table.
    # Bit 15..8 = 1 0 1 1 1 1 1 1, Bit 7..0 = 0 0 0 0 0 0 0 0
    bit_positions = ["15", "14", "13", "12", "11", "10", "9", "8",
                     "7", "6", "5", "4", "3", "2", "1", "0"]
    bit_values    = ["1",  "0",  "1",  "1",  "1",  "1",  "1", "1",
                     "0",  "0",  "0",  "0",  "0",  "0",  "0", "0"]
    page = _mock_page([[bit_positions, bit_values]])
    md = _tables_as_markdown(page)
    # Header row contains all bit position labels
    assert "| 15 |" in md
    assert "| 0 |" in md
    # Body row contains the bit values
    assert "| 1 |" in md
    assert "| 0 |" in md
    # Separator row present
    assert "| --- |" in md


def test_newlines_in_cells_replaced():
    page = _mock_page([[["line1\nline2", "val"]]])
    md = _tables_as_markdown(page)
    assert "line1\nline2" not in md
    assert "line1 line2" in md


def test_none_cells_become_empty_string():
    page = _mock_page([[["A", None, "C"], ["1", None, "3"]]])
    md = _tables_as_markdown(page)
    assert "| A |  | C |" in md
    assert "| 1 |  | 3 |" in md


def test_ragged_rows_padded_to_max_width():
    # Second row has fewer cells than first — should be padded
    page = _mock_page([[["A", "B", "C"], ["1", "2"]]])
    md = _tables_as_markdown(page)
    lines = [l for l in md.splitlines() if l.startswith("|")]
    col_counts = [l.count("|") - 1 for l in lines]
    assert len(set(col_counts)) == 1, "all rows must have the same column count"


def test_multiple_tables_separated_by_blank_line():
    page = _mock_page([
        [["T1 header"], ["T1 row"]],
        [["T2 header"], ["T2 row"]],
    ])
    md = _tables_as_markdown(page)
    assert "T1 header" in md
    assert "T2 header" in md
    # The two tables must be separated by a blank line
    assert "\n\n" in md
