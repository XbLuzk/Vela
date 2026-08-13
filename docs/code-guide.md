# Vela 代码阅读路线

这份路线面向刚开始阅读 Python 项目的开发者。不要从最大的文件逐行啃，先只跟一条普通 ReAct
请求，再分别补 Plan 和持久化。

## 1. 先看最短主链路

一次普通请求只需要跟下面五步：

1. `src/vela/entrypoints/cli.py::main`：解析命令行参数，选择交互模式或单次任务。
2. `src/vela/entrypoints/repl.py::start_repl`：组装模型、工具、Agent 和 Session。
3. `src/vela/entrypoints/repl.py::_repl_loop`：读取用户输入并启动当前任务。
4. `src/vela/agent/agent.py::Agent.run`：根据 `react / plan` 选择执行方式。
5. `src/vela/agent/langchain_runtime.py::run_langchain_agent`：启动 LangChain Agent Graph，执行“模型回复 → 工具调用 → 工具结果 → 再次回复”。

先忽略界面样式、MCP、Memory 和恢复逻辑。能解释这五步，就已经理解了项目最核心的运行链路。

## 2. ReAct 循环怎么读

`run_langchain_agent()` 可以按四段理解：

1. 把历史消息、本次请求、Skill 和系统提示词组装成模型输入。
2. `create_agent()` 管理模型与工具之间的标准 ReAct 循环。
3. `VelaChatModel` 保留模型流式协议，`VelaToolMiddleware` 继续调用 `ToolExecutor`。
4. LangChain 回填工具结果，Vela 同步 Session 历史并输出终端事件。

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
  - 每个节点仍然复用 `run_langchain_agent()`，不是另一套工具系统。

工具执行入口是 `src/vela/tools/executor.py::ToolExecutor._execute_single`。按顺序读它调用的
`_prepare_mutation()`、`_claim_mutation()` 和 `_run_tool()`，就能区分恢复、占位和真正执行三个阶段。

## 4. 最后补持久化和终端 UI

- `src/vela/session.py`：保存和恢复对话消息。
- `src/vela/tools/journal.py`：记录有副作用的工具调用，恢复时避免重复执行已完成操作。
- `src/vela/task_control.py`：管理 planning、running、cancelled 等前台任务状态。
- `src/vela/entrypoints/repl_commands.py`：实现配置、上下文、Memory、Skill 等斜杠命令。
- `src/vela/entrypoints/repl_ui.py`：输入框、快捷键和底部状态栏；它不参与 Agent 决策。
- `src/vela/render/rich_renderer.py`：把 Agent 事件显示到终端。

## 5. 推荐阅读节奏

每次只读 10 到 20 行，并回答三个问题：

1. 这段代码收到了什么数据？
2. 它修改或返回了什么数据？
3. 下一步调用哪个函数？

遇到 `await` 时，把它理解为“当前协程暂停，事件循环等待异步结果；结果回来后从这里继续”，不要把
整个项目同时展开。先跑通一条链路，再回头看数据类和异常分支会轻松很多。
