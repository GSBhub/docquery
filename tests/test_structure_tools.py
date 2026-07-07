"""Tests for the structure_lookup tool (docquery.tools.structure_tools)."""

import pytest
from langchain_core.documents import Document

from docquery.tools.structure_tools import _where, make_structure_lookup_tool


@pytest.fixture
def structured_store(vector_store):
    vector_store.add_documents([
        Document(
            page_content="ADD (register)\nBit-layout encoding:\n"
                         "ENCODING 16-bit: bits[15:9]=0001100 Rm[8:6] Rn[5:3] Rd[2:0]",
            metadata={"source": "m.pdf", "page": 191, "kind": "encoding_grid",
                      "entity_instruction": "ADD", "section": "A7.7"},
        ),
        Document(
            page_content="Control register\nStructured register fields:\n"
                         "TABLE register_fields: bit | field | description\n"
                         "ROW 7:0 | EN | Enable",
            metadata={"source": "m.pdf", "page": 300, "kind": "register_fields",
                      "entity_register_field": "EN", "section": "B2.1"},
        ),
        Document(
            page_content="Exception model\nStructured interrupt table:\n"
                         "TABLE interrupt_table: irq | acronym\nROW 2 | NMI",
            metadata={"source": "m.pdf", "page": 40, "kind": "interrupt_table",
                      "entity_interrupt": "NMI;HardFault"},
        ),
        Document(
            page_content="Plain prose chunk about instructions.",
            metadata={"source": "m.pdf", "page": 191},
        ),
    ])
    return vector_store


def test_where_single_condition_is_flat():
    assert _where({"kind": "encoding_grid", "page": None}) == {"kind": "encoding_grid"}


def test_where_multiple_conditions_wrapped_in_and():
    assert _where({"kind": "encoding_grid", "page": 191}) == {
        "$and": [{"kind": "encoding_grid"}, {"page": 191}]
    }


def test_where_empty_returns_none():
    assert _where({"kind": "", "page": None}) is None


def test_lookup_by_kind(structured_store):
    tool = make_structure_lookup_tool(structured_store)
    out = tool.invoke({"kind": "encoding_grid"})
    assert "ENCODING 16-bit" in out
    assert "Kind: encoding_grid" in out
    assert "TABLE" not in out


def test_lookup_by_kind_and_page(structured_store):
    tool = make_structure_lookup_tool(structured_store)
    out = tool.invoke({"kind": "encoding_grid", "page": 191})
    assert "ENCODING 16-bit" in out
    assert tool.invoke({"kind": "encoding_grid", "page": 300}).startswith(
        "No structured regions match")


def test_page_filter_excludes_untagged_chunks(structured_store):
    # Page 191 also holds a prose chunk without kind — it must not appear.
    tool = make_structure_lookup_tool(structured_store)
    out = tool.invoke({"page": 191})
    assert "ENCODING 16-bit" in out
    assert "Plain prose" not in out


def test_lookup_by_entity_name(structured_store):
    tool = make_structure_lookup_tool(structured_store)
    out = tool.invoke({"entity": "HardFault"})
    assert "interrupt_table" in out
    assert "ENCODING" not in out


def test_lookup_by_entity_substring_fallback(structured_store):
    tool = make_structure_lookup_tool(structured_store)
    assert "ENCODING 16-bit" in tool.invoke({"entity": "add"})


def test_lookup_by_section(structured_store):
    tool = make_structure_lookup_tool(structured_store)
    out = tool.invoke({"kind": "register_fields", "section": "B2.1"})
    assert "ROW 7:0 | EN | Enable" in out
    assert tool.invoke({"kind": "register_fields", "section": "Z9"}).startswith(
        "No structured regions match")


def test_no_args_lists_kinds_with_counts(structured_store):
    tool = make_structure_lookup_tool(structured_store)
    out = tool.invoke({})
    assert "encoding_grid (1 docs)" in out
    assert "interrupt_table (1 docs)" in out
    assert "register_fields (1 docs)" in out


def test_unknown_kind_lists_available(structured_store):
    tool = make_structure_lookup_tool(structured_store)
    out = tool.invoke({"kind": "syscall_table"})
    assert out.startswith("No structured regions match")
    assert "encoding_grid" in out


def test_empty_store_suggests_fallback(vector_store):
    tool = make_structure_lookup_tool(vector_store)
    assert "similarity_search" in tool.invoke({})
