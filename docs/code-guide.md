# Vela 代码阅读路线

这份路线面向刚开始阅读 Python 项目的开发者。不要从最大的文件逐行啃，先只跟一条普通 ReAct
请求，再分别补 Plan 和持久化。

## 1. 先看最短主链路

一次普通请求只需要跟下面六步：

1. `src/vela/entrypoints/cli.py::main`：解析命令行参数，选择交互模式或单次任务。
2. `src/vela/entrypoints/repl.py::start_repl`：组装模型、工具、Agent 和 Session。
3. `src/vela/entrypoints/repl.py::_repl_loop`：读取输入；具体命令和任务执行分别交给
   `repl_commands.py`、`repl_tasks.py`。
4. `src/vela/agent/agent.py::Agent.run`：根据 `react / plan` 选择执行方式。
5. `src/vela/run_trace/tracker.py::RunTracker.stream`：增加 Run ID、分层 Span，并统一收尾。
6. `src/vela/agent/react_runtime.py::run_react_agent`：执行“模型回复 → 工具调用 → 工具结果 → 再次回复”。

先忽略界面样式、MCP、Memory 和恢复逻辑。能解释这六步，就已经理解了项目最核心的运行链路。

## 2. ReAct 循环怎么读

`run_react_agent()` 只保留轮次循环，可以按五段理解：

1. 把历史消息、本次请求、Skill 和系统提示词组装成模型输入。
2. `_stream_react_turn()` 执行一轮；`agent/model_turn.py::stream_model_turn` 把 Provider 的
   `LlmEvent` 拼成完整回复和工具调用。
3. 没有工具调用就结束；有工具调用就进入 `_execute_tool_round()`。
4. `ToolExecutor` 负责并发只读工具、串行写工具、HITL 和工具 Journal。
5. 工具结果写回消息历史，下一轮模型调用继续处理。

`InteractiveTaskController` 统一编排任务生命周期、工具审批、Plan 确认和取消。一个终端同一时间只
运行一个 Agent 请求；任务运行期间提交的普通消息会被拒绝，不创建隐式队列或并发 Agent。

这条普通 ReAct 链路不依赖高层 Agent 框架，因此可以直接看到循环条件、消息变化和异常出口。

`src/vela/events.py` 用 `AgentEvent` 和 `LlmEvent` 两个 TypedDict 集中声明事件名称和字段；
阅读事件流时先看 `type`，再查看该事件使用的可选字段。

工具本身分两层：

- `src/vela/tools/builtins.py` 只声明工具名称、参数和处理函数。
- `src/vela/tools/file_ops.py` 等模块实现真正的文件或命令操作。

## 3. 再看 Plan

普通 ReAct 看懂后再进入 Plan 分支：

- Plan：`src/vela/agent/plan_graph.py::LangGraphPlanAgent.run`
  - Planner 生成 DAG。
  - `run()` 只串联“准备输入 → 流式执行 → 收尾”三步，恢复确认和 Journal 清理分别由小函数负责。
  - LangGraph 保存状态、等待人工确认并并行派发可执行节点。
  - 每个节点仍然复用 `run_react_agent()`，不是另一套模型或工具系统。

工具执行入口是 `src/vela/tools/executor.py::ToolExecutor._execute_single`。按顺序读它调用的
`_prepare_mutation()`、`_claim_mutation()` 和 `_run_tool()`，就能区分恢复、占位和真正执行三个阶段。

## 4. 最后补持久化和终端 UI

- `src/vela/session.py`：保存和恢复对话消息。
- `src/vela/tools/journal.py`：记录有副作用的工具调用，恢复时避免重复执行已完成操作。
- `src/vela/task_control.py`：管理 planning、running、cancelled 等前台任务状态。
- `src/vela/run_trace/models.py`：定义 Run 和 Span 的可序列化结构。
- `src/vela/run_trace/tracker.py`：把 Agent 事件归入 Plan、Turn 和 Tool 父子 Span。
- `src/vela/run_trace/store.py`：以 JSONL 追加保存 Trace，并提供倒序和 Run ID 查询。
- `src/vela/run_trace/context.py`：只在拉取 Agent 工作时绑定当前 Run ID，供工具 Audit 自动关联。
- `src/vela/entrypoints/repl_commands.py`：实现配置、上下文、Memory、Skill 等斜杠命令。
- `src/vela/entrypoints/repl_tasks.py`：运行 ReAct / Plan，并确保取消或失败后仍保存 Session。
- `src/vela/entrypoints/repl_ui.py`：输入框、快捷键和底部状态栏；它不参与 Agent 决策。
- `src/vela/render/rich_renderer.py`：把 Agent 事件显示到终端。

## 5. 独立能力怎么读

- `src/vela/context/manager.py::ContextEngine.prepare`：计算输入预算、裁剪工具结果并压缩历史。
- `src/vela/context/manager.py::ContextEngine.recover_from_overflow`：Provider 拒绝上下文后再缩减一次旧轮次。
- `src/vela_rag/server.py`：只负责暴露三个 MCP Tool；索引实现位于 `src/vela_rag/index.py`。
- `src/vela/eval/runner.py::EvalRunner.run`：在隔离目录运行固定任务并执行确定性断言。
- `src/vela/trust.py`：记录项目 Trust；CLI 在加载项目配置、MCP 和 Skills 前先解析这项决定。

## 6. 推荐阅读节奏

每次只读 10 到 20 行，并回答三个问题：

1. 这段代码收到了什么数据？
2. 它修改或返回了什么数据？
3. 下一步调用哪个函数？

遇到 `await` 时，把它理解为“当前协程暂停，事件循环等待异步结果；结果回来后从这里继续”，不要把
整个项目同时展开。先跑通一条链路，再回头看数据类和异常分支会轻松很多。
