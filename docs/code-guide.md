# Vela 代码阅读路线

这份路线面向刚开始阅读 Python 项目的开发者。不要从最大的文件逐行啃，先只跟一条普通 ReAct
请求，再分别补 Plan、Team 和持久化。

## 1. 先看最短主链路

一次普通请求只需要跟下面五步：

1. `src/vela/entrypoints/cli.py::main`：解析命令行参数，选择交互模式或单次任务。
2. `src/vela/entrypoints/repl.py::start_repl`：组装模型、工具、Agent 和 Session。
3. `src/vela/entrypoints/repl.py::_repl_loop`：读取用户输入并启动当前任务。
4. `src/vela/agent/agent.py::Agent.run`：根据 `react / plan / team` 选择执行方式。
5. `src/vela/agent/query.py::run_react_loop`：执行“模型回复 → 工具调用 → 工具结果 → 再次回复”。

先忽略界面样式、MCP、Memory 和恢复逻辑。能解释这五步，就已经理解了项目最核心的运行链路。

## 2. ReAct 循环怎么读

`run_react_loop()` 可以按四段理解：

1. 把历史消息、本次请求、Skill 和系统提示词组装成模型输入。
2. 流式读取模型事件，累积文本、Token 用量和工具调用参数。
3. 如果模型请求工具，交给 `src/vela/tools/executor.py::ToolExecutor.execute_stream`。
4. 把工具结果追加到消息列表，进入下一轮；没有工具调用时结束。

工具本身分两层：

- `src/vela/tools/builtins.py` 只声明工具名称、参数和处理函数。
- `src/vela/tools/file_ops.py` 等模块实现真正的文件或命令操作。

## 3. 再看 Plan 和 Team

普通 ReAct 看懂后再进入两个分支：

- Plan：`src/vela/agent/plan_graph.py::LangGraphPlanAgent.run`
  - Planner 生成 DAG。
  - LangGraph 保存状态、等待人工确认并并行派发可执行节点。
  - 每个节点仍然复用 `run_react_loop()`，不是另一套工具系统。
- Team：`src/vela/agent/orchestrator.py::AgentOrchestrator.run`
  - Planner 拆步骤。
  - Worker 执行步骤。
  - Reviewer 验证结果并决定是否重试。

## 4. 最后补持久化和终端 UI

- `src/vela/session.py`：保存和恢复对话消息。
- `src/vela/tools/journal.py`：记录有副作用的工具调用，恢复时避免重复执行已完成操作。
- `src/vela/task_control.py`：管理 planning、running、cancelled 等前台任务状态。
- `src/vela/entrypoints/repl_ui.py`：输入框、快捷键和底部状态栏；它不参与 Agent 决策。
- `src/vela/render/rich_renderer.py`：把 Agent 事件显示到终端。

## 5. 推荐阅读节奏

每次只读 10 到 20 行，并回答三个问题：

1. 这段代码收到了什么数据？
2. 它修改或返回了什么数据？
3. 下一步调用哪个函数？

遇到 `await` 时，把它理解为“当前协程暂停，事件循环等待异步结果；结果回来后从这里继续”，不要把
整个项目同时展开。先跑通一条链路，再回头看数据类和异常分支会轻松很多。
