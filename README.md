<h1 align="center">Vela</h1>

<p align="center">面向真实代码仓库的本地 Web AI Agent</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/XbLuzk/Vela/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/XbLuzk/Vela/ci.yml?branch=main&amp;style=flat-square&amp;label=CI"></a>
  <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/XbLuzk/Vela?style=flat-square"></a>
  <a href="https://docs.langchain.com/oss/python/langgraph/overview"><img alt="LangGraph 1.2+" src="https://img.shields.io/badge/LangGraph-1.2%2B-1C3C3C?style=flat-square"></a>
  <a href="https://modelcontextprotocol.io/docs/getting-started/intro"><img alt="MCP compatible" src="https://img.shields.io/badge/MCP-compatible-7C3AED?style=flat-square"></a>
</p>

Vela 在浏览器中提供一个安静的本地工作区：左侧是 Session，中间是对话，底部固定输入框，状态栏
位于输入框下方。用户消息、模型回答、Thinking、工具调用和计划进度使用不同层级展示，Thinking
与工具详情默认折叠，不再把终端状态行重复写进历史记录。

Agent 仍由 Python 负责。普通任务运行显式 ReAct 循环；复杂任务使用 LangGraph Plan DAG；两者
复用模型客户端、工具、安全策略、Session、Context、Memory、Skill、MCP 和内置 Code RAG。
React 前端只消费事件和提交用户操作，不复制 Agent 业务逻辑。

## 快速开始

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/XbLuzk/Vela.git
cd Vela
uv sync --extra dev
export DEEPSEEK_API_KEY=your_key_here
uv run vela
```

`vela` 只做一件事：在 `127.0.0.1:3080` 启动本地 Web 服务并打开浏览器。它不再提供终端聊天、
单次 Prompt、斜杠命令或终端渲染器。

```bash
uv run vela --cwd /path/to/project
uv run vela --port 4312
uv run vela --no-open
uv run vela --version
```

也可以不设置环境变量，启动后在右上角“设置”中选择模型并填写 API Key。模型设置保存到
`~/.vela/config.json`，不会写进项目目录。

## Web 工作流

- 输入区始终固定在页面底部；`Enter` 发送，`Shift+Enter` 换行。
- `ReAct` 适合普通任务；`Plan` 使用 LangGraph 先生成 DAG，再等待确认。
- 任务运行时仍可编辑下一条消息，但完成前不能再次发送；点击“停止”取消当前任务。
- Thinking、工具输入/结果和计划步骤以折叠详情显示，主对话只突出最终结果。
- 危险工具在 `Ask` 模式下显示行内确认条；`Auto` 模式仍然保留路径和命令守卫。
- Session 在左侧切换；刷新页面不会丢失已持久化的历史。
- 第一次打开包含 `AGENTS.md`、项目 MCP 或项目 Skill 的仓库时，页面会先请求项目 Trust。

图片仍使用 Prompt 引用：

```text
分析这张截图 @image:./screenshots/page.png
分析这张截图 @image:</Users/me/Desktop/screen shot.png>
看看剪贴板图片 @clipboard
```

## 保留的核心能力

| 能力 | 实现边界 |
| --- | --- |
| ReAct | `Agent.run()` 驱动显式“模型 → 工具 → 模型”循环 |
| Plan | LangGraph DAG、人工确认、SQLite Checkpoint 和安全恢复 |
| Session | 项目级消息持久化与 LangGraph thread 绑定 |
| Context | 工具结果裁剪、旧历史压缩和一次溢出恢复 |
| 工具安全 | 项目 Trust、Ask/Auto 审批、PathGuard、CommandGuard |
| 扩展 | MCP、Memory、只读 Skill 和多模态输入 |
| Code RAG | 本地 SQLite FTS 增量索引，可选语义混合检索 |

Vela 不保留 Run Trace、Eval、Audit、终端 REPL、Rich renderer 或 Prompt Toolkit UI。

## 配置

配置只有三层，后面的值覆盖前面的值：

1. 内置默认值
2. 用户配置 `~/.vela/config.json`
3. 当前进程环境变量

项目目录中的 `.env` 和 `.vela/config.json` 不会作为模型配置自动加载。项目级 `.vela/mcp.json`、
`.vela/skills/` 与指令文件只在通过 Trust 后加载。

常用环境变量：

| 环境变量 | 用途 |
| --- | --- |
| `VELA_PROVIDER` | 模型供应商 |
| `VELA_MODEL` | 模型 ID |
| `VELA_BASE_URL` | OpenAI-compatible API 地址 |
| `VELA_API_KEY` | 通用模型 Key |
| `DEEPSEEK_API_KEY` | DeepSeek Key |
| `ZAI_API_KEY` / `GLM_API_KEY` | GLM Key |
| `STEP_API_KEY` | Step Key |
| `KIMI_API_KEY` | Kimi Key |

## MCP 与 Code RAG

Vela 读取用户级 `~/.vela/mcp.json` 和可信项目的 `.vela/mcp.json`，支持 stdio 与 Streamable
HTTP MCP。远程工具注册名为 `mcp__<server>__<tool>`。stdio MCP 不会默认继承 Vela 的模型密钥；
需要的变量应在对应 server 的 `env` 中显式声明。

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "chrome-devtools-mcp@1.6.0", "--no-usage-statistics"]
    }
  }
}
```

Code RAG 是可信项目的内置能力。第一次搜索自动创建索引，后续只增量更新变化文件；默认使用本地
SQLite FTS，不会把源码发送给 Embedding Provider。语义混合检索可通过
`VELA_RAG_EMBEDDING_API_KEY`、`VELA_RAG_EMBEDDING_MODEL` 和可选的
`VELA_RAG_EMBEDDING_BASE_URL` 启用。

## 代码阅读路线

第一次阅读只跟这条链路：

```text
React App
  → POST /api/messages
  → WebRuntime.send
  → Agent.run
  → run_react_agent / LangGraphPlanAgent
  → Server-Sent Events
  → React reducer
```

细节见 [Vela 代码阅读路线](docs/code-guide.md)。

## 开发与验证

前端源码位于 `web/`，生产资源构建到 `src/vela/web/static/` 并随 wheel 发布。

```bash
uv sync --extra dev
cd web && npm install && npm test && npm run build && cd ..
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

运行本地验收：

```bash
uv run vela --no-open
# 浏览器打开 http://127.0.0.1:3080
```

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，测试边界见 [docs/testing.md](docs/testing.md)。

## License

[MIT](LICENSE)
