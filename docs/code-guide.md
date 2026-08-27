# Vela 代码阅读路线

这份路线面向第一次阅读 Python Agent + React 项目的开发者。不要从最大的文件逐行看，先跟一条
普通 ReAct 请求，再分别补 Plan、持久化和 Web 展示。

## 1. 最短主链路

一次普通请求只经过六步：

1. `web/src/App.tsx` 调用 `api.send()`。
2. `src/vela/web/app.py::send_message` 校验 HTTP 请求。
3. `src/vela/web/runtime.py::WebRuntime.send` 启动唯一前台任务。
4. `src/vela/agent/agent.py::Agent.run` 根据 `react / plan` 选择执行器。
5. `src/vela/agent/react_runtime.py::run_react_agent` 执行“模型 → 工具 → 模型”。
6. `WebRuntime` 将 Agent event 发送到 SSE，`web/src/state.ts` 把事件归并为页面状态。

先忽略样式、MCP、Memory 和恢复逻辑。能解释这六步，就已经理解项目核心。

## 2. ReAct 循环

`run_react_agent()` 可以按五段阅读：

1. 把历史消息、本次请求、Skill 和系统提示组装为模型输入。
2. `_stream_react_turn()` 执行一轮模型流式输出。
3. 没有工具调用就结束；有工具调用就进入 `_execute_tool_round()`。
4. `ToolExecutor` 并发执行只读工具、串行执行写工具，并在需要时等待 Web 审批。
5. 工具结果写回消息历史，下一轮继续。

`src/vela/events.py` 集中声明事件字段；阅读事件时先看 `type`。工具声明位于
`src/vela/tools/builtins.py`，真正的文件、搜索和 Shell 操作位于相邻实现模块。

## 3. Plan 分支

普通 ReAct 看懂后再进入 `src/vela/agent/plan_graph.py::LangGraphPlanAgent.run`：

- Planner 生成 DAG。
- LangGraph 保存状态、等待人工确认，并派发当前可执行节点。
- 每个节点仍复用 `run_react_agent()`，不是第二套模型或工具循环。
- Checkpoint 与 Tool Journal 只服务 Plan 恢复，不进入普通 ReAct API。

工具恢复入口是 `src/vela/tools/executor.py::ToolExecutor._execute_single`。依次阅读
`_prepare_mutation()`、`_claim_mutation()` 和 `_run_tool()`，即可区分恢复、占位和真正执行。

## 4. Web 边界

- `src/vela/entrypoints/web.py`：只启动本机 Uvicorn 服务并打开浏览器。
- `src/vela/web/app.py`：薄 HTTP/SSE 路由，不做 Agent 决策。
- `src/vela/web/runtime.py`：持有 Agent、Session、MCP 与当前任务。
- `src/vela/task_control.py`：管理 running、planning、approval、cancelled 等状态。
- `web/src/state.ts`：把流式事件归并成可渲染状态。
- `web/src/components/`：展示对话、折叠运行细节、输入框、审批与设置。

前端没有 Redux、组件库或第二套业务模型。React 只负责展示和用户动作；Python 是状态与安全边界。

## 5. 持久化与独立能力

- `src/vela/config.py::load_config`：默认值 → 用户配置 → 环境变量。
- `src/vela/session.py`：保存和恢复项目对话。
- `src/vela/tools/journal.py`：记录 Plan 的有副作用工具，避免恢复时重复执行。
- `src/vela/context/manager.py::ContextEngine.prepare`：计算预算、裁剪工具结果、压缩旧历史。
- `src/vela_rag/server.py`：暴露内置 Code RAG MCP Tool；索引位于 `src/vela_rag/index.py`。
- `src/vela/trust.py`：在加载项目指令、MCP 与 Skill 前处理项目 Trust。

## 6. 推荐节奏

每次只读 10 到 20 行，并回答三个问题：

1. 收到了什么数据？
2. 修改或返回了什么数据？
3. 下一步调用哪个函数？

遇到 `await` 时，只理解成“当前协程等结果，事件循环可继续处理其他工作”。先跑通一条链路，再看
数据类和异常分支，会比同时展开整个项目容易得多。
