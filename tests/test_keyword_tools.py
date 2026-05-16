from docquery.tools.keyword_tools import make_keyword_tool


def test_keyword_tool_finds_exact_match(populated_store):
    tool = make_keyword_tool(populated_store)
    result = tool.invoke("MOV")
    assert "MOV" in result


def test_keyword_tool_finds_opcode(populated_store):
    tool = make_keyword_tool(populated_store)
    result = tool.invoke("0xfe")
    assert "0xfe" in result


def test_keyword_tool_no_match(populated_store):
    tool = make_keyword_tool(populated_store)
    result = tool.invoke("NOSUCHTOKEN_ZZZXXX")
    assert "No exact matches" in result
