---
date: 2026-08-27
topic: runtime-simplification
---

# Vela Runtime Simplification

## Problem Frame

Vela currently exposes observability, evaluation, policy, session, context, and permission
infrastructure directly in the main Agent and REPL paths. The product should remain a capable
terminal coding agent and retain its interview-worthy LangGraph, MCP, RAG, Memory, Skill, and
multimodal features, while making the ordinary ReAct path and interactive surface easier to read.
The default mental model is only input -> model -> optional tool -> output; advanced capabilities
must be discoverable on demand rather than shape the first-use path.

## Requirements

**Remove non-core observability and evaluation**

- R1. Remove persisted Run Trace behavior, storage, commands, configuration, documentation, and
  tests. Keep only an immutable non-persisted run summary and a rendering function shared by ReAct
  and Plan. Existing callers settle their own normal completion, explicit errors, cancellation, and
  premature stream exhaustion. The summary reports only status, duration, turns, tool-call count,
  and token usage and owns no storage, IDs, spans, event interception, or stream lifecycle.
- R2. Remove the Eval runner, CLI commands, documentation, fixtures, and tests from the product.
- R3. Remove persistent Audit logging and its commands/configuration while preserving PathGuard,
  CommandGuard, dangerous-tool approval in `ask` mode, and Plan execution safety. Guard or approval
  denials and dangerous-tool failures remain visible during the current run as concise, redacted
  events, but are not persisted.

**Simplify the interactive surface**

- R4. Replace the read-only REPL display commands `/config`, `/policy`, `/tools`, `/usage`, and
  `/mcp` with `/status [config|policy|tools|usage|mcp]`; bare `/status` shows the compact summary.
  Operational MCP APIs and CLI subcommands remain unchanged. Status output uses field allowlists
  and never prints credentials or raw configuration object representations. Removed display
  commands are removed from routing, help, and completion. Unknown sections
  list the valid choices; unavailable integrations show `unavailable` instead of looking like zero
  resources or aborting the REPL.
- R5. Replace `/sessions` and `/resume` with `/session [list|current|resume <id|position>]`; bare
  `/session` lists the current project's sessions. Resume references may only resolve against that
  project-scoped list, never arbitrary paths, and must retain the Session-to-LangGraph-thread
  binding. Invalid, cross-project, and damaged records fail closed.
- R6. Ctrl+C never exits Vela. It cancels the active task when one exists. When idle it clears the
  current draft, keeps the REPL running, and shows a short `Ctrl+D or /exit to exit` hint. Ctrl+D
  and `/exit` remain the exit mechanisms.
- R7. Keep the permanent status line compact: model, context usage, approval mode, and active task
  state. Detailed resource counts move to `/status`. The REPL layout has exactly three layers:
  scrollable history, one fixed composer, then one fixed status line. Composer and status content
  never enter the conversation transcript or accumulate as repeated history rows. The status line
  never wraps; on narrow terminals it preserves textual task and approval state first, then hides
  context and model detail in that order. Color is supplementary, never the sole state signal.

**Simplify core concepts without removing retained capabilities**

- R8. Simplify HITL to `ask` and `auto` approval modes. `ask` automatically runs safe tools and
  prompts for tools marked as requiring approval; `auto` skips tool approval prompts. PathGuard and
  CommandGuard remain enabled and cannot be bypassed in either mode. LangGraph plan review remains
  a separate `execute / modify / cancel` decision. Fresh installs default to `ask`; removed HITL
  modes and environment variables are not accepted.
- R9. Keep only `max_conversation_history` as public Context configuration; compression threshold,
  target, reserve, recent-message count, and summary size become named internal defaults. Unknown
  keys use the normal config warning. Compression follows one readable bounded flow: trim
  oversized tool results, preserve recent messages, compress older history, and retry a provider
  context-overflow failure at most once.
- R10. Keep ordinary ReAct independent from Plan recovery details. LangGraph-only execution scope,
  uncertain retry, Checkpoint, and Tool Journal concerns must not clutter the normal ReAct API.
  ReAct modules must not import Checkpoint, Session recovery, Plan execution scope, or Tool Journal
  types; only the Plan call site may attach its recovery options to tool execution.
- R11. Prefer small typed runtime/request objects over long parameter lists, without hiding the
  model -> tool -> model loop behind a framework abstraction. Reuse the existing `ReplRuntime` and
  `_ReactContext` first; introduce a new value object only for a repeated parameter group shared by
  multiple call sites, never a generic registry, interface hierarchy, or factory layer.

## Success Criteria

- The ordinary ReAct call chain is readable without understanding Trace, Audit, Eval, Checkpoint,
  or Tool Journal internals.
- All retained behavior remains covered on Python 3.11 and 3.13.
- Full LangGraph DAG planning, human plan review, SQLite Checkpoint recovery, and side-effect-safe
  Tool Journal replay still work.
- MCP, built-in Code RAG, Memory, Skill, image input, model selection, Session persistence, and core
  tools continue to work.
- LangGraph and RAG each retain one clear entry point, one independently readable call chain, and
  one runnable demonstration without adding concepts to the ordinary ReAct path.
- The CLI and REPL contain no Trace, Eval, or Audit command or persisted data path.
- One-shot JSON retains a terminal `status` field but drops the persisted `run_id` field.
  Removing `run_id` is an intentional one-shot JSON compatibility break and must have a contract
  test plus updated examples/migration text.

## Scope Boundaries

- Do not remove or reduce LangGraph Plan functionality, Checkpoint recovery, Tool Journal recovery,
  MCP, Code RAG, Memory, Skill, image input, model selection, or existing core tools.
- Do not replace the explicit ReAct loop with LangGraph or another high-level Agent framework.
- Do not push, merge, publish, or delete user data as part of this implementation.
- Session changes are limited to command routing, help, and presentation. IDs, persistence format,
  project scoping, active-session lifecycle, and LangGraph thread mapping remain unchanged.
- Context changes are limited to budgeting, trimming, compression, overflow retry, and their public
  knobs. Memory persistence, recall, ranking, limits, and prompt semantics remain unchanged.

## Key Decisions

- LangGraph remains complete because its DAG, interrupt, Checkpoint, and recovery behavior is an
  important project and interview capability.
- RAG remains built in because it is also an important project and interview capability.
- Tool Journal remains for resumable Plan execution, but ordinary ReAct bypasses it.
- Journal identities remain scoped to the LangGraph thread, plan execution, step sequence, tool,
  normalized input hash, and project context. A missing or mismatched recovery record fails closed.
- Ctrl+C is cancellation-only; Ctrl+D and `/exit` are the explicit exits.

## Dependencies / Assumptions

- Existing tests provide characterization coverage for LangGraph recovery, Sessions, Context,
  permissions, MCP/RAG, Memory, Skill, images, and rendering.
- Removing historical Trace and Audit persistence is an intentional compatibility break.

## Next Steps

Before deletion, record a green Python 3.11 and 3.13 baseline and map retained terminal paths to
characterization tests. Then implement and validate in three checkpoints: A) remove Trace, Eval, and Audit; B) consolidate REPL
commands, Sessions, status, Ctrl+C, and approval presentation; C) simplify Context and the ReAct
API. Run retained-capability regression tests after every checkpoint, then run full validation,
including small/large context windows, oversized tool results, and image-bearing requests.
