"""Normalize provider tool-call envelopes at the runtime/tool boundary."""

from __future__ import annotations

import json
from typing import Any


def tool_call_name(call: object) -> str:
    """Return a nested or flat tool name, or an empty string for malformed calls."""
    if not isinstance(call, dict):
        return ""
    function = call.get("function")
    nested = function.get("name") if isinstance(function, dict) else None
    return str(nested or call.get("name") or "")


def tool_call_arguments(call: object) -> dict[str, Any]:
    """Decode nested or flat arguments without letting malformed provider data escape."""
    if not isinstance(call, dict):
        return {}
    function = call.get("function")
    arguments = (
        function.get("arguments", call.get("arguments", {}))
        if isinstance(function, dict)
        else call.get("arguments", {})
    )
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"raw": arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
