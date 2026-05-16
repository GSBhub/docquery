from docquery.tools.page_tools import make_page_lookup_tool


def test_page_tool_returns_content(populated_store):
    tool = make_page_lookup_tool(populated_store)
    result = tool.invoke("2")
    assert "LDR" in result


def test_page_tool_empty_page(populated_store):
    tool = make_page_lookup_tool(populated_store)
    result = tool.invoke("999")
    assert "No content found" in result


def test_page_tool_natural_language_input(populated_store):
    tool = make_page_lookup_tool(populated_store)
    result = tool.invoke("page 1")
    assert "MOV" in result


def test_page_tool_bad_input(populated_store):
    tool = make_page_lookup_tool(populated_store)
    result = tool.invoke("no numbers here at all!")
    assert "Could not parse" in result
