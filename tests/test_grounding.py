"""Tests for deterministic grounding checks (docquery._grounding)."""

from pydantic import BaseModel

from docquery._grounding import (
    extract_claims,
    ungrounded_fields,
    unsupported_claims,
)


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------

def test_extract_hex_normalizes_width_and_case():
    assert extract_claims("vector at 0x00000004") == {"0x4"}
    assert extract_claims("at 0X04 or 0x0000_0004") == {"0x4"}


def test_extract_bit_ranges_normalizes_brackets():
    assert extract_claims("bits [31:16] and 7:0") == {"31:16", "7:0"}
    assert extract_claims("range 15-8") == {"15:8"}


def test_extract_machine_lines_collapse_whitespace():
    text = "quoted:\nENCODING 16-bit:  bits[15:9]=0001100  Rd[2:0]\nprose"
    claims = extract_claims(text)
    assert "ENCODING 16-bit: bits[15:9]=0001100 Rd[2:0]" in claims
    assert "=0001100" in claims
    assert "15:9" in claims and "2:0" in claims


def test_extract_ignores_bare_decimals_and_prose():
    assert extract_claims("There are 16 exceptions on page 40.") == set()


# ---------------------------------------------------------------------------
# unsupported_claims
# ---------------------------------------------------------------------------

SOURCES = """[Source: m.pdf, Page: 21]
TABLE interrupt_table: irq | acronym | vector_address
ROW 2 | NMI | 0x00000008
ROW 3 | HardFault | 0x0000000C
ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]"""


def test_grounded_answer_has_no_unsupported_claims():
    answer = "NMI's vector is 0x08 and HardFault's is 0xC; Rd is bits [2:0]."
    assert unsupported_claims(answer, SOURCES) == []


def test_invented_hex_is_flagged():
    answer = "Reset is at vector 0x00000000."
    assert unsupported_claims(answer, SOURCES) == ["0x0"]


def test_invented_bit_range_is_flagged():
    assert unsupported_claims("The EN field is bits [12:10].", SOURCES) == ["12:10"]


def test_mutated_machine_line_is_flagged():
    answer = "ENCODING 16-bit: bits[15:9]=0001101 Rm[8:6] Rn[5:3] Rd[2:0]"
    flagged = unsupported_claims(answer, SOURCES)
    assert any(c.startswith("ENCODING") for c in flagged)
    assert "=0001101" in flagged


def test_verbatim_machine_line_is_supported():
    answer = "The encoding is:\nENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]"
    assert unsupported_claims(answer, SOURCES) == []


def test_empty_answer_is_grounded():
    assert unsupported_claims("", SOURCES) == []


# ---------------------------------------------------------------------------
# ungrounded_fields
# ---------------------------------------------------------------------------

class Field(BaseModel):
    name: str
    hi: int
    lo: int


class Reg(BaseModel):
    reg_name: str
    address: str
    description: str
    fields: list[Field]


CONTEXT = """The PRIMASK register at address 0xE000ED10.
[0] PRIMASK — prevents activation of exceptions with configurable priority."""


def test_grounded_extraction_passes():
    reg = Reg(reg_name="PRIMASK", address="0xE000ED10",
              description="Masks configurable-priority exceptions entirely.",
              fields=[Field(name="PRIMASK", hi=0, lo=0)])
    assert ungrounded_fields(reg, CONTEXT) == []


def test_invented_identifier_and_number_are_flagged():
    reg = Reg(reg_name="PRIMASK2", address="0xE000ED14",
              description="whatever", fields=[Field(name="PRIMASK", hi=31, lo=0)])
    misses = ungrounded_fields(reg, CONTEXT)
    assert "reg_name='PRIMASK2'" in misses
    assert "address='0xE000ED14'" in misses
    assert "fields.0.hi=31" in misses


def test_prose_fields_are_exempt():
    reg = Reg(reg_name="PRIMASK", address="0xE000ED10",
              description="A totally invented paraphrase that appears nowhere.",
              fields=[])
    assert ungrounded_fields(reg, CONTEXT) == []


def test_hex_width_mismatch_still_grounds():
    # document says 0xE000ED10; extraction may emit 0xe000ed10
    reg = Reg(reg_name="PRIMASK", address="0xe000ed10", description="x", fields=[])
    assert ungrounded_fields(reg, CONTEXT) == []


def test_int_found_as_hex_in_document():
    class M(BaseModel):
        offset: int
    assert ungrounded_fields(M(offset=4), "the offset is 0x04 in the map") == []


def test_nested_dicts_and_lists_walked():
    data = {"regs": [{"name": "XYZ_MISSING", "bit": 0}]}
    misses = ungrounded_fields(data, CONTEXT)
    assert misses == ["regs.0.name='XYZ_MISSING'"]


def test_booleans_are_not_verifiable():
    class M(BaseModel):
        enabled: bool
    assert ungrounded_fields(M(enabled=True), CONTEXT) == []
