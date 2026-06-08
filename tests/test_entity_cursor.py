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
    assert docs[0].metadata["entity_type"] == "instruction"
    assert docs[0].metadata["entity_name"] == "ADC"
    assert docs[1].metadata["entity_name"] == "ADD"
    # unmatched chunk is left untouched
    assert "entity_type" not in docs[2].metadata


def test_tag_entities_first_rule_wins():
    docs = [Document(page_content="R0 general purpose register", metadata={})]
    rules = [
        EntityRule(name="register", pattern=r"^(R\d+)\b"),
        EntityRule(name="instruction", pattern=r"register"),
    ]
    tag_entities(docs, rules)
    assert docs[0].metadata["entity_type"] == "register"
    assert docs[0].metadata["entity_name"] == "R0"


def test_tag_entities_noop_without_rules():
    docs = [Document(page_content="x", metadata={"page": 1})]
    tag_entities(docs, [])
    assert "entity_type" not in docs[0].metadata


# --- cursor_enumerate -----------------------------------------------------

def _store_with_entities(vector_store):
    # insert out of page order; tagged with two entity types
    vector_store.add_documents([
        Document(page_content="ADD desc", metadata={"source": "m.pdf", "page": 5,
                                                    "entity_type": "instruction", "entity_name": "ADD"}),
        Document(page_content="ADC desc", metadata={"source": "m.pdf", "page": 2,
                                                    "entity_type": "instruction", "entity_name": "ADC"}),
        Document(page_content="ADD dup", metadata={"source": "m.pdf", "page": 9,
                                                   "entity_type": "instruction", "entity_name": "ADD"}),
        Document(page_content="R0 reg", metadata={"source": "m.pdf", "page": 1,
                                                  "entity_type": "register", "entity_name": "R0"}),
    ])
    return vector_store


def test_enumerate_walks_all_distinct_in_page_order(vector_store):
    store = _store_with_entities(vector_store)
    count, enumerate_, current, nxt = make_cursor_tools(store, Settings())
    msg = enumerate_.invoke({"entity_type": "instruction"})
    # ADD + ADC + ADD-dup => 2 distinct
    assert "Found 2 distinct instruction" in msg
    first = current.invoke({})
    assert "ADC" in first and "page: 2" in first  # lowest page first
    second = nxt.invoke({})
    assert "ADD" in second
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
