# Contributing to Vela

Vela 是一个本地 Web AI Agent：React 负责交互，Python 负责 ReAct、LangGraph Plan、工具、安全、
Session、Context、MCP、RAG、Memory 和 Skill。

## 开发环境

- Python 3.11+
- uv
- Node.js 20+ 与 npm

```bash
uv sync --extra dev
cd web && npm install && npm run build && cd ..
uv run vela --no-open
```

主要目录：

- Web API/runtime：`src/vela/web/`
- React：`web/src/`
- ReAct 与 Plan：`src/vela/agent/`
- 模型：`src/vela/llm/`
- 工具与安全：`src/vela/tools/`、`src/vela/policy/`
- MCP：`src/vela/mcp/`
- Memory 与 Skill：`src/vela/memory/`、`src/vela/skill/`

公开行为变化时同步更新 README、文档和测试。不要提交 `.env`、API Key、Token、Cookie、私钥、
本地数据库、日志或个人文件。

## 验证

```bash
cd web && npm test && npm run build && cd ..
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Web 交互改动还应在真实浏览器中检查受影响页面与窄屏布局。模型或外部 MCP 验收使用自己的环境
变量或用户配置，不要把凭据写入仓库。

## Pull Request

PR 说明应包含：问题、实现、验证结果、未验证场景，以及兼容性、持久化、安全或外部副作用影响。
尽量让一次 PR 只处理一个清晰问题，并覆盖错误与恢复路径。

提交代码即表示同意按照 [MIT License](LICENSE) 发布该贡献。
