# 测试与验收

Vela 将验证分成三层，避免外部模型额度和网络波动影响普通 CI。

## 1. 默认 CI

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
uv build
```

这一层不读取 API Key，也不访问真实模型。它覆盖 Agent、LangGraph、Session、Context、MCP、
Memory、Skill、安全守卫、Web API 和本地 launcher。

## 2. 前端

```bash
cd web
npm ci
npm test
npm run build
```

`npm run build` 同时执行 TypeScript 检查，并将生产资源写入 `src/vela/web/static/`。CI 会重新构建并
检查仓库中的静态资源没有漂移。

## 3. 本地 Web 验收

```bash
uv run vela --no-open
```

浏览器打开 `http://127.0.0.1:3080`，至少确认：

1. 输入框固定在底部，状态栏在输入框下方。
2. 发送普通 ReAct 消息后，用户历史和模型输出样式不同。
3. Thinking 和工具详情默认折叠。
4. 任务运行时可以编辑草稿但不能重复发送，并可点击停止。
5. 危险工具审批与 Plan execute/modify/cancel 可以在页面完成。
6. 刷新页面和切换 Session 后历史仍然存在。

真实模型或 MCP 验收需要个人环境变量或用户配置；不要把凭据写入仓库。报告测试结果时应区分
“确定性测试通过”“生产构建通过”和“真实模型人工验收通过”。
