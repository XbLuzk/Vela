# Vela Roadmap

Vela 的目标是成为一个专注本地代码仓库工作的终端 AI Agent：普通任务使用 LangChain
ReAct，复杂任务使用可恢复的 LangGraph Plan DAG，并通过 MCP 扩展外部能力。

## 当前基础

- LangChain ReAct 与 LangGraph Plan-and-Execute 共用模型、工具、安全策略和事件协议。
- 项目级 Session、任务取消、Checkpoint、工具执行日志和中断恢复。
- 文件、Shell、代码搜索、Memory、只读 Skills、图片输入和 MCP 工具。
- HITL 确认、路径与命令策略、JSONL 审计日志。
- Python 3.11/3.13 自动化测试、Ruff 检查和发行包构建。

## 近期：稳定公开工作流

- [x] 使用 TypedDict 统一 Agent 与 LLM 流式事件协议。
- [x] 建立 GitHub Actions 质量门禁和发行包构建。
- [x] 提供可选的真实模型、stdio MCP、浏览器和 Plan Live 验收入口。
- [ ] 自动化交互式取消与中断 Plan 恢复的 PTY Live 验收。
- [ ] 增加 MCP Server 状态、重启、日志和运行时启停命令。
- [ ] 支持在普通输入中直接引用 MCP Resource 和 Prompt。

## 下一阶段：可观测和浏览器体验

- 统一 Run ID、节点耗时、Token、模型调用和工具调用摘要。
- 为 ReAct 与 Plan 建立可重复执行的任务评测集和结果对比。
- 增加浏览器连接状态、标签页、断开和登录态复用操作。
- 支持用户级、项目级 Prompt 模板覆盖。

## 后续扩展

- 将代码 RAG 作为独立 MCP 服务接入，保持 Vela 核心轻量。
- 在明确需要角色分工时，用 LangGraph Subgraph 扩展多 Agent 调度；不恢复第二套 Team
  Planner/Worker 运行时。
- 增加模型路由、失败切换和任务级预算控制。
- 评估 PyPI 发布和自动化 Release 流程。

## 产品边界

- 不内置依赖私人账号、扫码登录或私有协议的通信通道。
- 不在核心中重复实现已经由 MCP 提供的网络和浏览器工具。
- Skills 保持只读，只负责发现、匹配和加载现有 `SKILL.md`。
- 同一个 Session 不支持由多个 Vela 进程并发恢复。

## 质量门禁

每次合并应通过：

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev python -m pytest
uv build
```

真实外部依赖的验收方式见 [测试与 Live 验收](testing.md)。
