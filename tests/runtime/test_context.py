from __future__ import annotations

import pytest

from vela.context import ContextBudget, ContextEngine, ContextOverflowError, estimate_text_tokens
from vela.types import Message


def test_context_under_threshold_is_unchanged():
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    manager = ContextEngine(ContextBudget(10_000, 1_000))

    result = manager.prepare(messages, system_prompt="system")

    assert not result.compressed
    assert result.messages == messages


def test_context_compresses_old_turns_and_keeps_latest_user_message():
    messages = []
    for index in range(8):
        messages.extend(
            [
                Message(role="user", content=f"request {index} " + "x" * 240),
                Message(role="assistant", content=f"answer {index} " + "y" * 240),
            ]
        )
    manager = ContextEngine(
        ContextBudget(
            context_window=900,
            max_output_tokens=150,
            compression_threshold=0.6,
            compression_target=0.4,
            reserve_tokens=50,
        ),
        min_recent_messages=4,
    )

    result = manager.prepare(messages, system_prompt="system")

    assert result.compressed
    assert result.summarized_messages > 0
    assert result.messages[0].role == "assistant"
    assert "conversation-summary" in str(result.messages[0].content)
    assert any(
        message.content.startswith("request 7")
        for message in result.messages
        if message.role == "user"
    )
    assert result.estimated_tokens_after < result.estimated_tokens_before


def test_context_boundary_keeps_tool_call_and_result_together():
    messages = [
        Message(role="user", content="old" * 300),
        Message(role="assistant", content="old answer" * 200),
        Message(role="user", content="read file"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="result", tool_call_id="call_1"),
        Message(role="assistant", content="done"),
    ]
    manager = ContextEngine(ContextBudget(700, 100, 0.5, 0.35, 25), min_recent_messages=3)

    result = manager.prepare(messages)

    roles = [message.role for message in result.messages]
    assert roles[-4:] == ["user", "assistant", "tool", "assistant"]
    assert result.messages[-2].tool_call_id == "call_1"


def test_context_truncates_oversized_tool_payload():
    messages = [
        Message(role="user", content="inspect"),
        Message(role="assistant", content="", tool_calls=[{"id": "call_1"}]),
        Message(role="tool", content="x" * 10_000, tool_call_id="call_1"),
        Message(role="assistant", content="done"),
    ]
    manager = ContextEngine(ContextBudget(1_000, 100, 0.5, 0.35, 20), tool_result_max_chars=300)

    result = manager.prepare(messages)

    tool_message = next(message for message in result.messages if message.role == "tool")
    assert "tool result truncated" in str(tool_message.content)
    assert result.truncated_tool_results == 1
    assert result.omitted_tool_characters == 9_700


def test_context_prunes_large_tool_result_even_under_model_threshold():
    messages = [
        Message(role="user", content="inspect"),
        Message(role="tool", content="x" * 1_000, tool_call_id="call_1"),
    ]
    engine = ContextEngine(ContextBudget(100_000, 1_000), tool_result_max_chars=300)

    result = engine.prepare(messages)

    assert result.compressed
    assert result.summarized_messages == 0
    assert result.truncated_tool_results == 1
    assert result.estimated_tokens_after < result.estimated_tokens_before


def test_context_summary_surfaces_goals_decisions_files_and_unfinished_work():
    messages = [
        Message(
            role="user",
            content="目标：重构 src/vela/agent.py，决定改为显式循环，下一步补测试。" + "x" * 600,
        ),
        Message(role="assistant", content="已修改 src/vela/agent.py。" + "y" * 600),
        Message(role="user", content="继续处理 tests/agent/test_agent.py" + "z" * 600),
        Message(role="assistant", content="处理中" + "w" * 600),
        Message(role="user", content="latest request"),
        Message(role="assistant", content="latest answer"),
    ]
    engine = ContextEngine(
        ContextBudget(900, 100, 0.5, 0.35, 20),
        min_recent_messages=2,
    )

    result = engine.prepare(messages)
    summary = str(result.messages[0].content)

    assert "Goals:" in summary
    assert "Decisions:" in summary
    assert "Files:" in summary
    assert "src/vela/agent.py" in summary
    assert "Unfinished:" in summary


def test_mixed_language_token_estimator_is_nonzero_and_conservative():
    assert estimate_text_tokens("你好 world()") >= 5


def test_message_limit_alone_compacts_complete_old_turns():
    messages = [
        message
        for index in range(6)
        for message in (
            Message(role="user", content=f"request {index}"),
            Message(role="assistant", content=f"answer {index}"),
        )
    ]
    engine = ContextEngine(ContextBudget(100_000, 1_000), max_history_messages=5)

    result = engine.prepare(messages)

    assert result.compressed
    assert len(result.messages) <= 5
    assert "conversation-summary" in str(result.messages[0].content)
    assert result.messages[-2].content == "request 5"


def test_context_never_returns_more_than_available_input_budget():
    messages = [
        Message(role="user", content=f"old request {index} " + "x" * 400) for index in range(6)
    ]
    engine = ContextEngine(ContextBudget(700, 120, 0.8, 0.5, 50), min_recent_messages=1)

    result = engine.prepare(messages, system_prompt="system")

    assert result.estimated_tokens_after <= engine.budget.available_input_tokens


def test_context_fails_explicitly_when_latest_request_cannot_fit():
    engine = ContextEngine(ContextBudget(300, 100, reserve_tokens=50))

    with pytest.raises(ContextOverflowError, match="exceed the model input budget"):
        engine.prepare([Message(role="user", content="x" * 2_000)], system_prompt="system")


def test_overflow_recovery_removes_an_old_complete_turn():
    messages = [
        Message(role="user", content="old request"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current request"),
    ]
    engine = ContextEngine(ContextBudget(10_000, 1_000))

    result = engine.recover_from_overflow(messages, system_prompt="system")

    assert result.compressed
    assert [message.content for message in result.messages] == ["current request"]
    assert result.estimated_tokens_after < result.estimated_tokens_before


def test_overflow_recovery_does_not_truncate_the_only_user_request():
    engine = ContextEngine(ContextBudget(10_000, 1_000))

    with pytest.raises(ContextOverflowError, match="no older complete turn"):
        engine.recover_from_overflow([Message(role="user", content="only request")])
