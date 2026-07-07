"""Tests for the structured-region parse API (docquery._structures)."""

from langchain_core.documents import Document

from docquery.config import Settings
from docquery._structures import _annotate_bit_ranges, get_structures


def _settings_with_store(vector_store):
    s = Settings()
    s.vs = vector_store  # _build_chroma reuses an existing .vs
    return s


def _populate(vector_store):
    vector_store.add_documents([
        Document(
            page_content="A7.7.4\nADD (register)\n"
                         "Bit-layout encoding (recovered from the document's bit-numbered diagram):\n"
                         "ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]",
            metadata={"source": "m.pdf", "page": 191, "kind": "encoding_grid",
                      "entity_instruction": "ADD", "section": "A7.7"},
        ),
        Document(
            page_content="Control register\n"
                         "Structured register fields (recovered from the document's table layout):\n"
                         "TABLE register_fields: bit | field | description\n"
                         "ROW [31] | N | Negative flag\n"
                         "ROW 7:0 | EN | Enable\n"
                         "not a machine line",
            metadata={"source": "m.pdf", "page": 30, "kind": "register_fields",
                      "entity_register_field": "N;EN"},
        ),
        Document(page_content="prose", metadata={"source": "m.pdf", "page": 2}),
    ])
    return vector_store


def test_get_structures_returns_parsed_records(vector_store):
    results = get_structures(_settings_with_store(_populate(vector_store)))
    assert [r["kind"] for r in results] == ["register_fields", "encoding_grid"]  # page order

    fields = results[0]
    assert fields["entities"] == {"register_field": ["N", "EN"]}
    (table,) = fields["records"]
    assert table["columns"] == ["bit", "field", "description"]
    assert table["rows"][0]["bits"] == {"hi": 31, "lo": 31}
    assert table["rows"][1]["bits"] == {"hi": 7, "lo": 0}
    assert table["rows"][1]["field"] == "EN"

    enc = results[1]
    assert enc["page"] == 191
    assert enc["records"][0]["width"] == 16
    assert enc["records"][0]["segments"][0] == {
        "name": None, "hi": 15, "lo": 9, "value": "0001100"}


def test_get_structures_kind_filter(vector_store):
    settings = _settings_with_store(_populate(vector_store))
    results = get_structures(settings, kind="encoding_grid")
    assert len(results) == 1
    assert results[0]["kind"] == "encoding_grid"


def test_get_structures_keeps_unparseable_lines_out_of_records(vector_store):
    settings = _settings_with_store(_populate(vector_store))
    (fields,) = get_structures(settings, kind="register_fields")
    # "not a machine line" is excluded from lines entirely; headings too
    assert all(l.startswith(("TABLE ", "ROW ")) for l in fields["lines"])
    assert len(fields["records"]) == 1


def test_get_structures_empty_store(vector_store):
    assert get_structures(_settings_with_store(vector_store)) == []


def test_annotate_bit_ranges_variants():
    assert _annotate_bit_ranges({"bit": "[26:25],"})["bits"] == {"hi": 26, "lo": 25}
    assert _annotate_bit_ranges({"bit": "3"})["bits"] == {"hi": 3, "lo": 3}
    assert _annotate_bit_ranges({"bit": "8-15"})["bits"] == {"hi": 15, "lo": 8}
    assert "bits" not in _annotate_bit_ranges({"bit": "prose text"})
