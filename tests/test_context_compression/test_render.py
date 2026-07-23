"""Tests for render.py (§5.4 flat estimator + §5.8 structured summarizer input)."""

from __future__ import annotations

from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    FilePart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_extensions.context_compression.render import (
    render_messages_to_text,
    render_structured,
)


def test_flat_includes_instructions_and_parts():
    msgs = [
        ModelRequest(
            parts=[UserPromptPart(content="Hello")],
            instructions="Be helpful.",
        ),
        ModelResponse(parts=[TextPart(content="Hi there.")]),
    ]
    text = render_messages_to_text(msgs)
    assert "Be helpful." in text
    assert "Hello" in text
    assert "Hi there." in text


def test_flat_thinking_included_by_default_skipped_when_disabled():
    msgs = [ModelResponse(parts=[ThinkingPart(content="reasoning here"), TextPart(content="answer")])]
    assert "reasoning here" in render_messages_to_text(msgs)
    assert "reasoning here" not in render_messages_to_text(msgs, include_thinking=False)
    assert "answer" in render_messages_to_text(msgs, include_thinking=False)


def test_flat_tool_call_and_return():
    msgs = [
        ModelResponse(parts=[ToolCallPart(tool_name="get_weather", args={"city": "SF"}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="get_weather", content={"temp": 60}, tool_call_id="c1")]),
    ]
    text = render_messages_to_text(msgs)
    assert "assistant called tool get_weather with" in text
    assert '"city": "SF"' in text
    assert "tool get_weather returned" in text


def test_flat_compaction_part_content_and_none():
    with_content = [ModelResponse(parts=[CompactionPart(content="prior summary")])]
    assert "prior summary" in render_messages_to_text(with_content)
    none_content = [ModelResponse(parts=[CompactionPart(content=None)])]
    assert "<encrypted compaction>" in render_messages_to_text(none_content)


def test_flat_file_part_is_media():
    binary = BinaryContent(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    msgs = [ModelRequest(parts=[FilePart(content=binary)])]
    assert "<media>" in render_messages_to_text(msgs)


def test_structured_file_part_is_media():
    binary = BinaryContent(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    msgs = [ModelRequest(parts=[FilePart(content=binary)])]
    out = render_structured(msgs)
    assert '<message role="user"><media /></message>' in out


def test_flat_user_prompt_sequence_with_binary_is_media():
    binary = BinaryContent(data=b"\x89PNG", media_type="image/png")
    msgs = [ModelRequest(parts=[UserPromptPart(content=["look", binary])])]
    text = render_messages_to_text(msgs)
    assert "look" in text
    assert "<media>" in text


def test_structured_basic_roles_and_wrapper():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="hi")], instructions="sys"),
        ModelResponse(parts=[TextPart(content="hello")]),
    ]
    out = render_structured(msgs)
    assert out.startswith("<conversation-history>")
    assert out.endswith("</conversation-history>")
    assert '<message role="system">sys</message>' in out
    assert '<message role="user">hi</message>' in out
    assert '<message role="assistant">hello</message>' in out


def test_structured_tool_call_attributes():
    msgs = [
        ModelResponse(parts=[ToolCallPart(tool_name="get_weather", args={"city": "SF"}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="get_weather", content={"temp": 60}, tool_call_id="c1")]),
    ]
    out = render_structured(msgs)
    # tool_call attribute uses the tool name; arguments carries the JSON (single-quoted attr,
    # because the JSON contains double quotes; quoteattr picks single quotes automatically).
    assert 'tool_call="get_weather"' in out
    assert 'arguments=\'{"city": "SF"}\'' in out
    assert 'role="tool"' in out
    assert 'tool_name="get_weather"' in out
    assert 'tool_call_id="c1"' in out


def test_structured_escapes_special_characters():
    msgs = [ModelRequest(parts=[UserPromptPart(content="a < b & c > d")])]
    out = render_structured(msgs)
    assert "a &lt; b &amp; c &gt; d" in out
    assert '<message role="user">a &lt; b &amp; c &gt; d</message>' in out


def test_structured_compaction_part_none_is_self_closing():
    msgs = [ModelResponse(parts=[CompactionPart(content=None)])]
    out = render_structured(msgs)
    assert '<message role="system" compaction />' in out


def test_structured_retry_prompt():
    msgs = [ModelRequest(parts=[RetryPromptPart(content="bad args", tool_name="t", tool_call_id="c1")])]
    out = render_structured(msgs)
    assert 'role="tool"' in out
    assert "retry" in out
    assert "bad args" in out
