from docquery.tools.retrieval_tools import make_similarity_tool


def test_similarity_tool_returns_string(populated_store, settings):
    tool = make_similarity_tool(populated_store, settings)
    result = tool.invoke("MOV instruction")
    assert isinstance(result, str)
    assert len(result) > 0


def test_similarity_tool_no_results(vector_store, settings):
    tool = make_similarity_tool(vector_store, settings)
    result = tool.invoke("anything")
    assert "No relevant content found" in result
