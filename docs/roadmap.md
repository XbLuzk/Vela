# Vela Roadmap

Vela 的目标是成为一个专注本地代码仓库的 Web AI Agent：普通任务使用显式 ReAct，复杂任务使用
可恢复的 LangGraph Plan DAG，外部能力通过 MCP 扩展。

## 当前基础

- React 本地工作区、固定 Composer、Session 列表和紧凑状态栏。
- FastAPI + SSE 事件桥接，Python 保持唯一 Agent 状态边界。
- ReAct 与 LangGraph Plan 共用模型、工具、安全策略和事件协议。
- Session、任务取消、Checkpoint、Tool Journal 和中断恢复。
- 文件、Shell、代码搜索、Memory、只读 Skill、图片和 MCP。
- Ask/Auto 审批，以及始终启用的路径与命令守卫。

## 近期

- [x] 用 Web 完整替换终端聊天、单次 Prompt 和 Rich/Prompt Toolkit UI。
- [x] Thinking、工具与 Plan 详情折叠，主对话保持紧凑。
- [x] 模型设置、项目 Trust、工具审批和 Plan 确认进入页面。
- [ ] 增加拖拽/选择图片并自动插入 Prompt 引用。
- [ ] 增加 MCP Server 状态、重启、日志和运行时启停页面。
- [ ] 增加 Web 层真实模型与浏览器自动化验收。

## 后续

- 在确有角色分工需求时，用 LangGraph Subgraph 扩展多 Agent，不恢复第二套 Team runtime。
- 增加模型路由、失败切换和任务级预算。
- 评估 PyPI 发布和自动 Release。

## 产品边界

- Web 服务只绑定回环地址，不向局域网提供认证不足的远程入口。
- 不在核心中重复实现 MCP 已提供的网络和浏览器工具。
- Skills 只发现、匹配和加载已有 `SKILL.md`。
- 同一 Session 不支持多个 Vela 进程并发恢复。
- 不恢复终端聊天、单次 Prompt、Trace、Eval 或 Audit。
