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


# ---------------------------------------------------------------------------
# ungrounded_records (association / co-occurrence)
# ---------------------------------------------------------------------------

ROWS = """TABLE interrupt_table: irq | acronym | vector_address
ROW 2 | NMI | 0x00000008
ROW 3 | HardFault | 0x0000000C"""


class Vector(BaseModel):
    name: str
    address: str


def test_correctly_paired_record_grounds():
    from docquery._grounding import ungrounded_records
    v = Vector(name="NMI", address="0x08")
    assert ungrounded_records(v, ROWS) == []


def test_mispaired_record_is_flagged_despite_presence():
    from docquery._grounding import ungrounded_records
    # both tokens exist in the context — but never on the same line
    v = Vector(name="NMI", address="0x0C")
    (miss,) = ungrounded_records(v, ROWS)
    assert "NMI" in miss and "0x0C" in miss


def test_nested_list_records_use_dotted_paths():
    from docquery._grounding import ungrounded_records

    class Table(BaseModel):
        vectors: list[Vector]

    t = Table(vectors=[Vector(name="NMI", address="0x08"),
                       Vector(name="HardFault", address="0x08")])
    (miss,) = ungrounded_records(t, ROWS)
    assert miss.startswith("vectors.1:")


def test_single_scalar_records_are_not_association_checked():
    from docquery._grounding import ungrounded_records

    class M(BaseModel):
        name: str
        note: str  # prose-length, not verifiable

    m = M(name="NMI", note="A long descriptive sentence that is paraphrased.")
    assert ungrounded_records(m, ROWS) == []


def test_sentence_final_number_grounds_but_decimal_part_does_not():
    class M(BaseModel):
        value: int
    # "42." at end of sentence supports 42 ...
    assert ungrounded_fields(M(value=42), "the value is 42.") == []
    # ... but "42.5" and "3.42" must not
    assert ungrounded_fields(M(value=42), "the value is 42.5") != []
    assert ungrounded_fields(M(value=42), "pi-ish is 3.42") != []


def test_record_named_in_heading_valued_on_machine_line_is_grounded():
    from docquery._grounding import ungrounded_records

    # The mnemonic lives on the heading line; width and bits on the ENCODING
    # line. That is how structure documents are built — must not be flagged.
    class Enc(BaseModel):
        mnemonic: str
        width_bits: int
        fixed_bits: str

    ctx = ("A7.7.4\nADD (register)\n"
           "Bit-layout encoding (recovered from the document's bit-numbered diagram):\n"
           "ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]\n"
           "ENCODING 32-bit: bits[31:21]=11101011000 S[20:20] Rd[11:8]")
    assert ungrounded_records(Enc(mnemonic="ADD", width_bits=16, fixed_bits="0001100"), ctx) == []
    # Values must still come from ONE machine line: mixing the 16-bit width
    # with the 32-bit line's opcode is a wrong association.
    (miss,) = ungrounded_records(Enc(mnemonic="ADD", width_bits=16, fixed_bits="11101011000"), ctx)
    assert "11101011000" in miss


def test_heading_augmentation_does_not_merge_two_data_lines():
    from docquery._grounding import ungrounded_records

    ctx = ("Exception model\nStructured interrupt table:\n" + ROWS)
    v = Vector(name="NMI", address="0x0C")  # HardFault's address
    assert ungrounded_records(v, ctx)  # still flagged with headings present


# ---------------------------------------------------------------------------
# digit-grouped hex ("0xA800 0000")
# ---------------------------------------------------------------------------

def test_extract_spaced_hex_merges_four_digit_groups():
    claims = extract_claims("Reset value: 0x0000 0280 for port B")
    assert "0x280" in claims
    claims = extract_claims("0xA800 0000 for port A")
    assert "0xa8000000" in claims


def test_spaced_hex_does_not_merge_unrelated_numbers():
    # continuation groups must be exactly 4 hex digits
    assert "0x815" not in extract_claims("registers 0x08 15 in the map")
    assert "0x80c" not in extract_claims("0x08 or 0x0C")


def test_field_grounds_against_spaced_hex_in_document():
    class M(BaseModel):
        reset_value: str
    ctx = "Reset value: 0x0000 0280 for port B, and 0x00000000 for other ports."
    assert ungrounded_fields(M(reset_value="0x00000280"), ctx) == []


def test_record_grounds_on_row_with_spaced_hex():
    from docquery._grounding import ungrounded_records

    class RegRow(BaseModel):
        name: str
        offset: str
        reset_value: str

    ctx = ("GPIO register map\n"
           "TABLE register_map: register | offset | reset_value\n"
           "ROW GPIOA_CFGR | 0x00 | 0xA800 0000\n"
           "ROW GPIOA_OMODE | 0x04 | 0x0000 00C0")
    r = RegRow(name="GPIOA_CFGR", offset="0x00", reset_value="0xA8000000")
    assert ungrounded_records(r, ctx) == []
    # reset value taken from the OTHER row is still a wrong association
    r = RegRow(name="GPIOA_CFGR", offset="0x00", reset_value="0x000000C0")
    assert ungrounded_records(r, ctx)


# ---------------------------------------------------------------------------
# block-scoped grounding units (register stanzas)
# ---------------------------------------------------------------------------

class RegStanza(BaseModel):
    name: str
    offset: str
    reset_value: str


AT32_STANZA = """6.3.1 GPIO configuration register (GPIOx_CFGR) (x=A..F)
Address offset: 0x00
Reset value: 0xA800 0000 for port A, 0x0000 0280 for port B."""


def test_small_block_without_machine_lines_is_one_unit():
    from docquery._grounding import ungrounded_records

    r = RegStanza(name="GPIOx_CFGR", offset="0x00", reset_value="0xA8000000")
    assert ungrounded_records(r, AT32_STANZA) == []


def test_two_register_stanzas_do_not_cross_pair():
    from docquery._grounding import ungrounded_records

    ctx = (AT32_STANZA + "\n\n"
           "6.3.2 GPIO output mode register (GPIOx_OMODE)\n"
           "Address offset: 0x04\n"
           "Reset value: 0x0001 0000")
    # offset from stanza 2 paired with stanza 1's register: flagged
    r = RegStanza(name="GPIOx_CFGR", offset="0x04", reset_value="0xA8000000")
    assert ungrounded_records(r, ctx)


def test_large_block_is_not_one_unit():
    from docquery._grounding import ungrounded_records

    ctx = ("GPIOx_CFGR overview\n" + "filler line about behaviour\n" * 4 +
           "Address offset: 0x00\nReset value: 0xA800 0000")
    r = RegStanza(name="GPIOx_CFGR", offset="0x00", reset_value="0xA8000000")
    assert ungrounded_records(r, ctx)


def test_heading_region_of_structure_block_is_one_unit():
    from docquery._grounding import ungrounded_records

    ctx = (AT32_STANZA + "\n"
           "TABLE register_fields: bit | field | description\n"
           "ROW 15:0 | IOMC | mode configuration")
    r = RegStanza(name="GPIOx_CFGR", offset="0x00", reset_value="0xA8000000")
    assert ungrounded_records(r, ctx) == []


# ---------------------------------------------------------------------------
# per-field grounding opt-out (json_schema_extra={"grounding": "off"})
# ---------------------------------------------------------------------------

def test_exempt_field_is_not_presence_checked():
    from pydantic import Field as PField

    class M(BaseModel):
        name: str
        access: str = PField(json_schema_extra={"grounding": "off"})

    assert ungrounded_fields(M(name="NMI", access="read-write"), ROWS) == []


def test_exempt_field_is_not_association_checked():
    from docquery._grounding import ungrounded_records
    from pydantic import Field as PField

    class M(BaseModel):
        name: str
        address: str
        access: str = PField(json_schema_extra={"grounding": "off"})

    # {NMI, 0x08} co-occur on a ROW; "rw" appears nowhere but is exempt
    m = M(name="NMI", address="0x08", access="rw")
    assert ungrounded_records(m, ROWS) == []
    assert ungrounded_fields(m, ROWS) == []


def test_non_exempt_fields_still_enforced_alongside_exempt():
    from pydantic import Field as PField

    class M(BaseModel):
        name: str
        access: str = PField(json_schema_extra={"grounding": "off"})

    misses = ungrounded_fields(M(name="INVENTED", access="rw"), ROWS)
    assert misses == ["name='INVENTED'"]
