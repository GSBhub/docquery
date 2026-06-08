from langchain_core.documents import Document

from docquery.config import EntityRule, Settings
from docquery._ingest import tag_entities
from docquery.tools.cursor_tools import make_cursor_tools


# --- tag_entities ---------------------------------------------------------

def test_tag_entities_sets_metadata():
    docs = [
        Document(page_content="A7.7.1 ADC\nAdd with carry ...", metadata={"page": 1}),
        Document(page_content="A7.7.2 ADD\nAdd ...", metadata={"page": 2}),
        Document(page_content="just prose, no heading", metadata={"page": 3}),
    ]
    rule = EntityRule(name="instruction", pattern=r"^A7\.7\.\d+\s+([A-Z][A-Z0-9.]+)")
    tag_entities(docs, [rule])
    assert docs[0].metadata["entity_instruction"] == "ADC"
    assert docs[1].metadata["entity_instruction"] == "ADD"
    # unmatched chunk is left untouched
    assert "entity_instruction" not in docs[2].metadata


def test_tag_entities_captures_all_matches_in_chunk():
    # a single chunk holding two instruction headings (like Unstructured by_title)
    docs = [Document(page_content="A7.7.10 BKPT\nBreakpoint ...\nA7.7.11 BL\nBranch ...",
                     metadata={"page": 1})]
    rule = EntityRule(name="instruction", pattern=r"^A7\.7\.\d+\s+([A-Z][A-Z0-9]+)")
    tag_entities(docs, [rule])
    assert docs[0].metadata["entity_instruction"] == "BKPT;BL"


def test_tag_entities_multiple_types_same_chunk():
    docs = [Document(page_content="R0 register, used by A7.7.1 ADD", metadata={})]
    rules = [
        EntityRule(name="register", pattern=r"\b(R\d+)\b"),
        EntityRule(name="instruction", pattern=r"A7\.7\.\d+\s+([A-Z][A-Z0-9]+)"),
    ]
    tag_entities(docs, rules)
    assert docs[0].metadata["entity_register"] == "R0"
    assert docs[0].metadata["entity_instruction"] == "ADD"


def test_tag_entities_noop_without_rules():
    docs = [Document(page_content="x", metadata={"page": 1})]
    tag_entities(docs, [])
    assert "entity_instruction" not in docs[0].metadata


# --- cursor_enumerate -----------------------------------------------------

def _store_with_entities(vector_store):
    # insert out of page order; tagged with two entity types. The page-9 chunk
    # lists TWO instructions (";"-joined) and repeats ADD to exercise dedup.
    vector_store.add_documents([
        Document(page_content="ADD desc", metadata={"source": "m.pdf", "page": 5,
                                                    "entity_instruction": "ADD"}),
        Document(page_content="ADC desc", metadata={"source": "m.pdf", "page": 2,
                                                    "entity_instruction": "ADC"}),
        Document(page_content="ADD/AND chunk", metadata={"source": "m.pdf", "page": 9,
                                                         "entity_instruction": "ADD;AND"}),
        Document(page_content="R0 reg", metadata={"source": "m.pdf", "page": 1,
                                                  "entity_register": "R0"}),
    ])
    return vector_store


def test_enumerate_walks_all_distinct_in_page_order(vector_store):
    store = _store_with_entities(vector_store)
    count, enumerate_, current, nxt = make_cursor_tools(store, Settings())
    msg = enumerate_.invoke({"entity_type": "instruction"})
    # ADD, ADC, (ADD;AND) => distinct ADD, ADC, AND => 3
    assert "Found 3 distinct instruction" in msg
    first = current.invoke({})
    assert "ADC" in first and "page: 2" in first  # lowest page first
    second = nxt.invoke({})
    assert "ADD" in second
    third = nxt.invoke({})
    assert "AND" in third
    assert "End of iteration" in nxt.invoke({})


def test_enumerate_unknown_type_lists_available(vector_store):
    store = _store_with_entities(vector_store)
    _, enumerate_, _, _ = make_cursor_tools(store, Settings())
    msg = enumerate_.invoke({"entity_type": "opcode"})
    assert "No entities of type 'opcode'" in msg
    assert "instruction" in msg and "register" in msg


def test_enumerate_no_tags_guides(populated_store):
    # populated_store has no entity metadata
    _, enumerate_, _, _ = make_cursor_tools(populated_store, Settings())
    msg = enumerate_.invoke({"entity_type": "instruction"})
    assert "No structural entities are tagged" in msg
