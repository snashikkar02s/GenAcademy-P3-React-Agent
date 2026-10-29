from langchain_core.messages import AIMessage, HumanMessage

from code_review_agent.config import _without_thinking


def test_without_thinking_keeps_text_and_tool_use():
    msg = AIMessage(content=[
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "redacted_thinking", "data": "x"},
        {"type": "text", "text": "Calling a tool."},
        {"type": "tool_use", "id": "t1", "name": "get_pr_overview", "input": {}},
    ], tool_calls=[{"name": "get_pr_overview", "args": {}, "id": "t1"}])
    stripped = _without_thinking(msg)
    assert [b["type"] for b in stripped.content] == ["text", "tool_use"]
    assert stripped.tool_calls == msg.tool_calls
    assert _without_thinking(HumanMessage("hi")).content == "hi"
