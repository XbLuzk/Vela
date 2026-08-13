# Vela Python Parity

This file tracks the Python port against the existing Java and TypeScript implementations.

## Implemented

- CLI:
  - `vela`
  - `vela -p`
  - `--provider`
  - `--model`
  - `--mode react|plan`
  - `--json` usage output
  - `--cwd`
  - `vela doctor`
- REPL:
  - `/help`
  - `/clear`
  - `/cancel`
  - `/sessions`
  - `/resume [session-id-or-index]`
  - `/context`
  - `/memory`
  - `/save`
  - `/config`
  - `/tools`
  - `/hitl`
  - `/policy`
  - `/audit`
  - `/plan`
  - `/plan --resume`
  - `/model` built-in model selector and direct provider/model switching
  - `/usage`
  - `/skill`
  - `/mcp`
  - `/exit`
  - project-scoped persistent interactive sessions in `~/.vela/sessions/sessions.db`
  - `vela --resume` restores the latest non-empty session for the current project
  - cooperative cancellation for ReAct, Plan, and active shell tools via `/cancel`, Escape, or Ctrl+C
  - unified `planning -> running -> cancelling -> cancelled|completed|failed` interactive state
  - Plan review gate with execute, modify/replan, and cancel decisions
  - interrupted sessions remain resumable with incomplete tool calls closed explicitly
- Agent:
  - OpenAI-compatible streaming LLM client
  - DeepSeek default
  - LangChain ReAct runtime with Vela text/thinking/tool-call/tool-result/done events
  - LangGraph-only Plan-and-Execute agent with `StateGraph`, dynamic `Send` fan-out,
    `Command` routing, native `interrupt()` review, and async SQLite checkpoints
  - session-aligned Graph threads and `/plan --resume` recovery from the last successful batch
  - tool-boundary execution journal for mutating calls with stable execution keys, completed-result
    replay, uncertain retry gates, and overwrite `write_file` reconciliation
  - isolated Skill context per parallel Plan task
- Configuration:
  - defaults
  - user config
  - project config
  - project `.env`
  - CLI overrides
  - process env
  - provider-specific keys such as `DEEPSEEK_API_KEY`, `GLM_API_KEY`, `STEP_API_KEY`, `KIMI_API_KEY`
- Tools:
  - `read_file`
  - `write_file`
  - `list_dir`
  - `glob`
  - `grep`
  - `bash`
  - `save_memory`
  - `search_memory`
  - `load_skill`
- Safety:
  - PathGuard
  - CommandGuard
  - HITL approval
  - JSONL AuditLog
- Memory:
  - static project memory files `AGENTS.md`, `PAI.md`, `.vela/PAI.md`, local variants
  - governed SQLite dynamic memory with metadata, normalized deduplication, TTL, quota, access tracking, and relevance recall
  - automatic request-specific Top-K recall plus model-initiated `search_memory`
  - bounded short-term history and deterministic context compression
  - cache-friendly static Prompt plus per-request dynamic Prompt
- Skills:
  - built-in/user/project skill layers
  - user/project `.vela/skills/*/SKILL.md`
  - `load_skill` with one-shot SkillContextBuffer injection
  - current-query next-turn Skill injection (no one-request delay)
  - name/description/tag Top-K matcher with Chinese n-gram support
  - read-only `/skill list/show` inspection
- MCP:
  - official MCP Python SDK client
  - stdio MCP server connection
  - Streamable HTTP MCP server connection
  - dynamic `mcp__server__tool` registration
  - virtual resource tools
  - virtual prompt tools
  - network and browser capabilities supplied by configured MCP servers
  - `vela mcp init-chrome`
  - `vela mcp list`
- Chrome DevTools MCP:
  - project/user config writer for `npx chrome-devtools-mcp@latest`
  - `--browser-url`
  - `--headless`
  - `--slim`
  - usage-statistics opt-out flag by default
- Image input:
  - `@image:path`
  - `@image:<path with spaces>`
  - `@image:file:///path`
  - `@image:https://...`
  - macOS `Ctrl+V` clipboard capture
  - `@clipboard`
  - local image resize/compress
  - transparent PNG white background handling
  - provider/model capability fallback
- Diagnostics:
  - Python syntax diagnostics after `write_file`
- Usage:
  - OpenAI-compatible streaming usage-only chunks
  - input/output/cache-hit/cache-miss/reasoning token aggregation
  - ReAct/Plan CLI aggregation

## Live Dependencies

These features need external credentials or platform state for live verification:

- Real LLM calls need API keys.
- Chrome DevTools MCP needs Node.js LTS, npm/npx, and Chrome.

## Intentionally Excluded Java-Only Area

The Java implementation has a WeChat iLink channel. Vela intentionally does not implement or plan to migrate this channel because it depends on private protocol details, private account credentials, and scan-login state. It is outside Vela's local development Agent product scope and is not a parity requirement.

## Remaining Public Parity Gaps

- MCP operations in the interactive CLI: server status, restart, logs, runtime enable/disable,
  resources, and prompts management.
- Browser session operations: connect/status/tabs/disconnect, isolated/shared switching, and
  authenticated Chrome session reuse.
- Direct MCP resource references such as `@server:protocol://path` in ordinary prompt input.
- User-level and project-level prompt template overrides comparable to the Java implementation.
- Live end-to-end acceptance covering a real model, MCP, browser interaction, cancellation, and
  interrupted Plan recovery.

Vela already exceeds the Java implementation in persistent project sessions, LangGraph Plan
checkpoints, tool-boundary execution journaling, completed-result replay, and uncertain-call
recovery. Parity therefore means closing the public product workflow, not reproducing every
Java-only integration.

## Intentionally Simplified Areas

- Team mode was removed because it duplicated LangGraph Plan scheduling while adding a second
  planner/worker execution model. Vela keeps one recoverable planning engine.
- Skills are read-only. Vela discovers, matches, and loads existing `SKILL.md` files but does not
  create, overwrite, enable, or disable them at runtime.

## Verification

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev python -m pytest
uv build
uv run vela --help
uv run vela doctor --cwd .
```
