"""Tests for deterministic encoding bit-grid extraction (docquery._bitgrid)."""

from __future__ import annotations

import pytest

from docquery._bitgrid import (
    _opfield_values,
    encodings_from_words,
    extract_document_encodings,
    parse_encoding_line,
    render_encoding_lines,
)
from docquery._ingest import build_encoding_documents


def _word(x: float, y: float, text: str, w: float = 8.0):
    """Build a minimal pymupdf-style word tuple centred spans around x."""
    return (x - w / 2, y, x + w / 2, y + 8, text)


def _header(bits: list[int], y: float, x0: float = 100.0, step: float = 20.0):
    """Header words for *bits* left-to-right starting at x0."""
    return [_word(x0 + i * step, y, str(b)) for i, b in enumerate(bits)]


def _cells(tokens: list[str], y: float, x0: float = 100.0, step: float = 20.0):
    return [_word(x0 + i * step, y, t) for i, t in enumerate(tokens)]


# ---------------------------------------------------------------------------
# encodings_from_words
# ---------------------------------------------------------------------------

def test_16bit_grid_with_fixed_run_and_fields():
    # ADD (register) T1: 0001100 Rm Rn Rd
    words = _header(list(range(15, -1, -1)), y=100)
    values = ["0", "0", "0", "1", "1", "0", "0"]
    words += _cells(values, y=112)
    # Field labels centred over their bit spans (Rm=8:6, Rn=5:3, Rd=2:0).
    words += [
        _word(100 + 8 * 20, 112, "Rm"),   # centre of bits 8..6 ≈ column 7
        _word(100 + 11 * 20, 112, "Rn"),
        _word(100 + 14 * 20, 112, "Rd"),
    ]
    encs = encodings_from_words(words)
    assert len(encs) == 1
    assert encs[0]["width"] == 16
    lines = render_encoding_lines(encs)
    assert lines == ["ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]"]


def _col(b):  # x-centre of bit b (bit 7 at x=100, 20pt bit-proportional columns)
    return 100 + (7 - b) * 20


def _c6x_style_words():
    """8-bit diagram A[7:5] op[4:2] 1[1] p[0] with an explicit width row."""
    words = [_word(_col(b), 100, str(b)) for b in (7, 5, 4, 2, 1, 0)]  # field-edge bits
    words += [_word(_col(6), 112, "A"), _word(_col(3), 112, "op"),
              _word(_col(1), 112, "1"), _word(_col(0), 112, "p")]       # value row
    words += [_word(_col(6), 126, "3"), _word(_col(3), 126, "3"),
              _word(_col(0), 126, "1")]                                 # width row
    return words


def test_explicit_width_row_gives_exact_boundaries():
    # The width row pins boundaries exactly (no x-interpolation drift).
    encs = encodings_from_words(_c6x_style_words())
    assert render_encoding_lines(encs) == [
        "ENCODING 8-bit: A[7:5] op[4:2] bits[1:1]=1 p[0:0]"
    ]


def test_opfield_table_expands_variable_op_field():
    # A shared diagram with a variable `op` field plus an Opfield opcode-map
    # column yields one encoding per opcode value.
    words = _c6x_style_words()
    words += [_word(300, 150, "Opfield"),
              _word(300, 162, "011"), _word(300, 174, "101")]
    lines = render_encoding_lines(encodings_from_words(words))
    assert lines == [
        "ENCODING 8-bit: A[7:5] bits[4:2]=011 bits[1:1]=1 p[0:0]",
        "ENCODING 8-bit: A[7:5] bits[4:2]=101 bits[1:1]=1 p[0:0]",
    ]


def test_should_be_zero_notation_counts_as_fixed():
    words = _header(list(range(7, -1, -1)), y=50)
    words += _cells(["(0)", "1", "(1)", "0"], y=60)
    words += [_word(100 + 5.5 * 20, 60, "Rd")]
    (enc,) = encodings_from_words(words)
    assert render_encoding_lines([enc]) == ["ENCODING 8-bit: bits[7:4]=0110 Rd[3:0]"]


def test_stacked_32bit_halves_are_combined():
    # ARMv8 style: header 31..16 above header 15..0.
    words = _header(list(range(31, 15, -1)), y=100)
    words += _cells(["1"] * 16, y=110)
    words += _header(list(range(15, -1, -1)), y=140)
    words += _cells(["0"] * 16, y=150)
    (enc,) = encodings_from_words(words)
    assert enc["width"] == 32
    assert render_encoding_lines([enc]) == [
        "ENCODING 32-bit: bits[31:16]=1111111111111111 bits[15:0]=0000000000000000"
    ]


def test_side_by_side_32bit_halves_are_combined():
    # ARMv7 Thumb-2 style: two 15..0 grids on one row; left half = bits 31..16.
    words = _header(list(range(15, -1, -1)), y=100, x0=100)
    words += _cells(["1"] * 16, y=112, x0=100)
    words += _header(list(range(15, -1, -1)), y=100, x0=500)
    words += _cells(["0"] * 16, y=112, x0=500)
    (enc,) = encodings_from_words(words)
    assert enc["width"] == 32
    assert render_encoding_lines([enc]) == [
        "ENCODING 32-bit: bits[31:16]=1111111111111111 bits[15:0]=0000000000000000"
    ]


def test_sparse_header_bits_are_interpolated():
    # ARMv8 style sparse header: only field-boundary bits are labelled.
    words = _header([31, 30, 24, 23, 22, 21, 20, 16, 15, 10, 9, 5, 4, 0], y=100,
                    x0=100, step=40)
    # Columns line up with the *interpolated* 32 evenly spaced positions.
    # Header spans bit31@x=100 … bit0@x=620 → 31 gaps ≈ 16.77pt each... use
    # explicit x from the same interpolation the code applies: place value
    # tokens exactly at the labelled boundary columns.
    xs = {31: 100, 30: 140, 24: 180, 23: 220, 22: 260, 21: 300, 20: 340,
          16: 380, 15: 420, 10: 460, 9: 500, 5: 540, 4: 580, 0: 620}

    def bitx(b: int) -> float:
        known = sorted(xs.items(), reverse=True)
        for (b1, x1), (b2, x2) in zip(known, known[1:]):
            if b2 <= b <= b1:
                return x1 + (x2 - x1) * (b1 - b) / (b1 - b2)
        raise AssertionError(b)

    words += [_word(xs[31], 112, "sf")]
    words += [_word(bitx(b), 112, v) for b, v in zip(range(30, 23, -1), "1001011")]
    words += [_word((xs[23] + xs[22]) / 2, 112, "shift")]
    words += [_word(xs[21], 112, "0")]
    words += [_word((xs[20] + bitx(17)) / 2, 112, "Rm")]
    words += [_word((xs[15] + bitx(11)) / 2, 112, "imm6")]
    words += [_word((xs[9] + bitx(6)) / 2, 112, "Rn")]
    words += [_word((xs[4] + bitx(1)) / 2, 112, "Rd")]
    (enc,) = encodings_from_words(words)
    assert enc["width"] == 32
    line = render_encoding_lines([enc])[0]
    assert "bits[30:24]=1001011" in line
    assert "Rm[20:16]" in line
    assert "Rn[9:5]" in line
    assert "Rd[4:0]" in line


def test_value_row_of_ones_and_zeroes_is_not_a_header():
    # A row of 0/1 digits must never be mistaken for a bit-number header.
    words = _header(list(range(15, -1, -1)), y=100)
    words += _cells(["0", "1"] * 8, y=112)
    (enc,) = encodings_from_words(words)
    assert enc["width"] == 16
    assert render_encoding_lines([enc])[0].startswith("ENCODING 16-bit: bits[15:0]=")


def test_dangling_upper_half_is_dropped():
    words = _header(list(range(31, 15, -1)), y=100)
    words += _cells(["1"] * 16, y=110)
    assert encodings_from_words(words) == []


def test_no_grid_no_encodings():
    words = [_word(100, 50, "This"), _word(120, 50, "is"), _word(135, 50, "prose")]
    assert encodings_from_words(words) == []


# ---------------------------------------------------------------------------
# parse_encoding_line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("encoding", [
    {"width": 16, "segments": [
        {"name": None, "hi": 15, "lo": 9, "value": "0001100"},
        {"name": "Rm", "hi": 8, "lo": 6, "value": None},
        {"name": "Rn", "hi": 5, "lo": 3, "value": None},
        {"name": "Rd", "hi": 2, "lo": 0, "value": None},
    ]},
    {"width": 32, "segments": [
        {"name": None, "hi": 31, "lo": 16, "value": "1111111111111111"},
        {"name": None, "hi": 15, "lo": 0, "value": "0000000000000000"},
    ]},
    {"width": 8, "segments": [
        {"name": None, "hi": 7, "lo": 4, "value": "0110"},
        {"name": None, "hi": 3, "lo": 2, "value": None},   # unreadable → ?[3:2]
        {"name": "Rd", "hi": 1, "lo": 0, "value": None},
    ]},
])
def test_parse_encoding_line_round_trips_renderer(encoding):
    (line,) = render_encoding_lines([encoding])
    assert parse_encoding_line(line) == encoding


def test_parse_encoding_line_rejects_prose_and_malformed():
    assert parse_encoding_line("This is prose.") is None
    assert parse_encoding_line("") is None
    assert parse_encoding_line("ENCODING 16-bit: not!a(segment") is None
    assert parse_encoding_line("ENCODING 16-bit: ") is None


# ---------------------------------------------------------------------------
# extract_document_encodings (real PDF via pymupdf)
# ---------------------------------------------------------------------------

def test_extract_document_encodings_real_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "grid.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    x0, step = 72, 12
    for i, b in enumerate(range(15, -1, -1)):
        page.insert_text((x0 + i * step, 100), str(b), fontsize=7)
    for i, t in enumerate(["0", "0", "0", "1", "1", "0", "0"]):
        page.insert_text((x0 + i * step, 110), t, fontsize=7)
    page.insert_text((x0 + 8 * step, 110), "Rm", fontsize=7)
    page.insert_text((x0 + 11 * step, 110), "Rn", fontsize=7)
    page.insert_text((x0 + 14 * step, 110), "Rd", fontsize=7)
    doc.save(str(pdf))
    doc.close()

    result = extract_document_encodings(pdf)
    assert list(result) == [1]
    assert result[1] == ["ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]"]


def test_extract_document_encodings_missing_file():
    assert extract_document_encodings("/nonexistent/file.pdf") == {}


# ---------------------------------------------------------------------------
# build_encoding_documents
# ---------------------------------------------------------------------------

def test_build_encoding_documents_uses_headings_and_metadata():
    docs = build_encoding_documents(
        "manual.pdf",
        {191: ["ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]"]},
        {191: "A7.7.4\nADD (register)\nprose…"},
        {191: {"instruction": [("A7.7.4", "A7.7.4  ADD (register)")]}},
    )
    (d,) = docs
    assert d.metadata == {"source": "manual.pdf", "page": 191, "kind": "encoding_grid"}
    assert d.page_content.startswith("A7.7.4  ADD (register)\n")
    assert "ENCODING 16-bit: bits[15:9]=0001100" in d.page_content


def test_build_encoding_documents_falls_back_to_first_page_line():
    (d,) = build_encoding_documents(
        "manual.pdf",
        {5: ["ENCODING 8-bit: bits[7:0]=00000000"]},
        {5: "\n  NOP instruction \nbody"},
        {},
    )
    assert d.page_content.startswith("NOP instruction\n")


def test_build_encoding_documents_extends_heading_with_next_line():
    # ISA manuals put the section number and the instruction NAME on consecutive
    # lines; the doc must carry the name or consumers can't associate the grid.
    (d,) = build_encoding_documents(
        "manual.pdf",
        {191: ["ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]"]},
        {191: "A7.7.4\nADD (register)\nThis instruction adds…"},
        {191: {"instruction": [("A7.7.4", "A7.7.4")]}},
    )
    lines = d.page_content.splitlines()
    assert lines[0] == "A7.7.4"
    assert lines[1] == "ADD (register)"


def _c6x_diagram_at(y: float):
    """A c6x-style 8-bit diagram (A[7:5] op[4:2] 1[1] p[0]) anchored at *y*."""
    words = [_word(_col(b), y, str(b)) for b in (7, 5, 4, 2, 1, 0)]
    words += [_word(_col(6), y + 12, "A"), _word(_col(3), y + 12, "op"),
              _word(_col(1), y + 12, "1"), _word(_col(0), y + 12, "p")]
    words += [_word(_col(6), y + 26, "3"), _word(_col(3), y + 26, "3"),
              _word(_col(0), y + 26, "1")]
    return words


def _opfield_table_at(y: float, values: list[str]):
    words = [_word(300, y, "Opfield")]
    for i, v in enumerate(values):
        words.append(_word(300, y + 12 * (i + 1), v))
    return words


def test_each_diagram_uses_only_the_opfield_table_beneath_it():
    """Two diagrams of the SAME op width on one page must not swap opcode values.

    Scanning the whole page gives every diagram every value; the widths only
    disambiguate when they happen to differ, which is not guaranteed.
    """
    words = _c6x_diagram_at(100) + _opfield_table_at(150, ["011", "101"])
    words += _c6x_diagram_at(300) + _opfield_table_at(350, ["110"])

    lines = render_encoding_lines(encodings_from_words(words))
    assert lines == [
        "ENCODING 8-bit: A[7:5] bits[4:2]=011 bits[1:1]=1 p[0:0]",
        "ENCODING 8-bit: A[7:5] bits[4:2]=101 bits[1:1]=1 p[0:0]",
        "ENCODING 8-bit: A[7:5] bits[4:2]=110 bits[1:1]=1 p[0:0]",
    ], f"opcode values crossed between diagrams: {lines}"


def test_opfield_values_are_restricted_to_their_band():
    words = _opfield_table_at(150, ["011", "101"]) + _opfield_table_at(350, ["110"])
    assert _opfield_values(words, 100, 300) == ["011", "101"]
    assert _opfield_values(words, 300, float("inf")) == ["110"]
    assert _opfield_values(words, 0, 100) == []      # no header in band


def test_width_row_rejected_when_widths_do_not_tile_the_word():
    """A width row that does not sum to the encoding width can't be trusted, so
    the decoder falls back to x-interpolation rather than inventing boundaries."""
    words = [_word(_col(b), 100, str(b)) for b in (7, 5, 4, 2, 1, 0)]
    words += [_word(_col(6), 112, "A"), _word(_col(3), 112, "op"),
              _word(_col(1), 112, "1"), _word(_col(0), 112, "p")]
    words += [_word(_col(6), 126, "2"), _word(_col(3), 126, "2"),
              _word(_col(0), 126, "1")]          # 2+2+1+1 = 6, not 8
    lines = render_encoding_lines(encodings_from_words(words))
    assert lines and "A[7:6]" not in lines[0], f"bad width row was trusted: {lines}"
