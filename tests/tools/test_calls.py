from __future__ import annotations

import pytest

from vela.tools.calls import tool_call_arguments, tool_call_name


@pytest.mark.parametrize(
    ("call", "name", "arguments"),
    [
        (
            {"function": {"name": "read_file", "arguments": '{"path":"note.txt"}'}},
            "read_file",
            {"path": "note.txt"},
        ),
        (
            {"name": "read_file", "arguments": {"path": "note.txt"}},
            "read_file",
            {"path": "note.txt"},
        ),
        (
            {"function": {"name": "read_file", "arguments": "not json"}},
            "read_file",
            {"raw": "not json"},
        ),
        ({"function": {"name": "read_file", "arguments": "[1]"}}, "read_file", {"value": [1]}),
        ({"function": "invalid", "arguments": 7}, "", {}),
        ([], "", {}),
    ],
)
def test_tool_call_decoders_handle_nested_flat_and_malformed_envelopes(
    call,
    name,
    arguments,
) -> None:
    assert tool_call_name(call) == name
    assert tool_call_arguments(call) == arguments
