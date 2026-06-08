import pytest
from langchain_core.documents import Document

from docquery.config import Settings
from docquery.tools.cursor_tools import make_cursor_tools


@pytest.fixture
def cursor_settings():
    """Settings with a permissive threshold so the mock-embedding relevance
    scores (all equal) always clear the bar — decouples tests from the score scale."""
    s = Settings()
    s.cursor_score_threshold = -1.0
    s.cursor_max_scan = 100
    return s


def _tools(store, settings):
    count, current, nxt = make_cursor_tools(store, settings)
    return count, current, nxt


def test_count_then_current(populated_store, cursor_settings):
    count, current, _ = _tools(populated_store, cursor_settings)
    msg = count.invoke({"criteria": "instruction"})
    assert "Found 3" in msg
    first = current.invoke({})
    assert "Item 1 of 3" in first
    assert "MOV" in first  # page 1 sorts first


def test_next_advances_and_exhausts(populated_store, cursor_settings):
    count, _, nxt = _tools(populated_store, cursor_settings)
    count.invoke({"criteria": "instruction"})
    second = nxt.invoke({})
    assert "Item 2 of 3" in second
    assert "LDR" in second  # page 2
    third = nxt.invoke({})
    assert "Item 3 of 3" in third
    end = nxt.invoke({})
    assert "End of iteration" in end
    # past-the-end does not wrap back to the start
    assert "Item 1 of 3" not in end


def test_current_before_count(populated_store, cursor_settings):
    _, current, _ = _tools(populated_store, cursor_settings)
    assert "No active cursor" in current.invoke({})


def test_next_before_count(populated_store, cursor_settings):
    _, _, nxt = _tools(populated_store, cursor_settings)
    assert "No active cursor" in nxt.invoke({})


def test_recount_resets_position(populated_store, cursor_settings):
    count, current, nxt = _tools(populated_store, cursor_settings)
    count.invoke({"criteria": "instruction"})
    nxt.invoke({})
    nxt.invoke({})
    # a fresh cursor_count rewinds to the first item
    count.invoke({"criteria": "instruction"})
    assert "Item 1 of 3" in current.invoke({})


def test_items_sorted_by_page(vector_store, cursor_settings):
    # inserted out of page order; cursor must walk them in ascending page order
    vector_store.add_documents([
        Document(page_content="gamma", metadata={"source": "d.pdf", "page": 3}),
        Document(page_content="alpha", metadata={"source": "d.pdf", "page": 1}),
        Document(page_content="beta", metadata={"source": "d.pdf", "page": 2}),
    ])
    count, current, nxt = _tools(vector_store, cursor_settings)
    count.invoke({"criteria": "anything"})
    assert "alpha" in current.invoke({})
    assert "beta" in nxt.invoke({})
    assert "gamma" in nxt.invoke({})


def test_count_no_matches(populated_store):
    # an unreachably high threshold yields zero matches
    s = Settings()
    s.cursor_score_threshold = 2.0
    count, current, _ = _tools(populated_store, s)
    msg = count.invoke({"criteria": "instruction"})
    assert "Found 0" in msg
    assert "nothing to show" in current.invoke({})
