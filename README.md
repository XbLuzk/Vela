<h1 align="center">Vela</h1>

<p align="center">
  面向真实代码仓库的终端 AI Agent
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/XbLuzk/Vela/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/XbLuzk/Vela?style=flat-square&amp;logo=github"></a>
  <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/XbLuzk/Vela?style=flat-square"></a>
  <a href="https://docs.langchain.com/oss/python/langgraph/overview"><img alt="LangGraph 1.2+" src="https://img.shields.io/badge/LangGraph-1.2%2B-1C3C3C?style=flat-square"></a>
  <a href="https://docs.langchain.com/oss/python/langchain/overview"><img alt="LangChain 1.3+" src="https://img.shields.io/badge/LangChain-1.3%2B-1C3C3C?style=flat-square"></a>
  <a href="https://modelcontextprotocol.io/docs/getting-started/intro"><img alt="MCP compatible" src="https://img.shields.io/badge/MCP-compatible-7C3AED?style=flat-square"></a>
</p>

<p align="center">
  LangChain ReAct · LangGraph Plan · MCP · Skills · Memory · Multimodal
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#核心能力">核心能力</a> ·
  <a href="#交互工作流">交互工作流</a> ·
  <a href="#开发与验证">参与开发</a>
</p>

Vela 是一个运行在终端中的 AI Agent。它能够读取和修改文件、搜索代码、执行命令、调用
MCP 工具、管理长期记忆，并通过可恢复的 Session、LangGraph Checkpoint 和工具执行日志处理
长任务与中断恢复。

普通任务由 LangChain Agent Graph 驱动 ReAct 循环；复杂任务由 LangGraph 编排 Plan DAG。
两种模式共用 Vela 的模型适配、工具安全策略、执行日志、Session 和终端事件协议。

第一次阅读 Python Agent 项目，可以从 [Vela 代码阅读路线](docs/code-guide.md) 开始，只跟普通
ReAct 请求的五步主链路，再逐步进入 Plan、Session 和终端 UI。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| Agent 运行时 | LangChain ReAct 与可恢复的 LangGraph Plan-and-Execute |
| 任务恢复 | 项目级持久化 Session、任务取消、Graph Checkpoint 和工具结果重放 |
| 工具系统 | 文件、Shell、代码搜索、网页检索、记忆、Skill 和 MCP 工具 |
| 安全控制 | HITL 人工确认、路径与命令策略、JSONL 审计日志和会话级权限切换 |
| 上下文管理 | 静态项目指令、SQLite 长期记忆、相关性召回、Token 预算和上下文压缩 |
| 多模态输入 | 支持本地图片、远程图片、`@image` 引用和 macOS 剪贴板图片 |
| 使用方式 | 交互式 CLI 和单次 Prompt |

## 快速开始

### 环境要求

- Python 3.11 或更新版本
- [uv](https://docs.astral.sh/uv/)
- 可选：`rg`，用于更快的本地搜索
- 可选：Node.js 20.19.0 LTS、npm/npx 和 Chrome，用于 Chrome DevTools MCP

### 安装与启动

```bash
git clone https://github.com/XbLuzk/Vela.git
cd Vela
uv sync --extra dev
```

配置模型 Key 并启动交互模式：

```bash
export DEEPSEEK_API_KEY=your_key_here
uv run vela
```

恢复当前项目最近一次 Session：

```bash
uv run vela --resume
```

检查本地依赖、模型和配置：

```bash
uv run vela doctor --cwd .
```

### 单次任务

```bash
uv run vela -p "帮我总结这个项目"
uv run vela --mode plan -p "先读取 README，再验证项目" --json
```

## 配置

Vela 按以下顺序合并配置，后面的配置覆盖前面的配置：

1. 内置默认配置
2. `~/.vela/config.json`
3. 项目级 `.vela/config.json`
4. 项目级 `.env`
5. CLI 参数
6. 当前进程环境变量

常用环境变量：

| 环境变量 | 用途 |
| --- | --- |
| `VELA_PROVIDER` | 模型供应商 |
| `VELA_MODEL` | 模型 ID |
| `VELA_BASE_URL` | OpenAI-compatible API 地址 |
| `VELA_API_KEY` | Vela 通用模型 Key |
| `DEEPSEEK_API_KEY` | DeepSeek Key |
| `ZAI_API_KEY` / `GLM_API_KEY` | GLM Key |
| `STEP_API_KEY` | Step Key |
| `KIMI_API_KEY` | Kimi Key |

临时切换模型：

```bash
uv run vela --provider deepseek --model deepseek-v4-flash
```

连接本地 OpenAI-compatible 服务：

```bash
VELA_PROVIDER=openai-compatible \
VELA_BASE_URL=http://127.0.0.1:11434/v1 \
VELA_MODEL=qwen2.5-coder \
uv run vela -p "解释这个仓库"
```

## 交互工作流

### Session 与任务取消

普通启动会为当前项目创建新的 Session。Vela 将消息、工具调用和工具结果持久化到
`~/.vela/sessions/`，不会写入项目目录。

```text
/sessions
/resume
/resume <session-id-or-index>
/cancel
```

任务运行期间可以使用 `/cancel`、`Esc` 或 `Ctrl+C` 取消 ReAct、Plan 及正在执行的
Shell 工具。第一次 `Ctrl+C` 取消当前任务，再按一次才退出 Vela。取消或失败的对话仍会保存，
之后可以通过 `/resume` 继续。

### Plan-and-Execute

`/plan` 使用 LangGraph 生成并执行任务 DAG。计划生成后需要人工选择：

- `execute`：执行当前计划
- `modify`：补充要求并重新规划
- `cancel`：放弃当前计划

```text
/plan <task>
/plan --resume
```

交互式 Checkpoint 保存在 `~/.vela/langgraph/checkpoints.sqlite`。恢复 Session 后执行
`/plan --resume`，可以从最后一个成功批次继续。

### 模型选择

输入 `/model` 打开交互式模型选择器：

- 使用上下方向键选择模型
- 按 `Enter` 切换当前 Agent 的模型
- 按 `Esc` 返回对话

也可以使用 `/model <provider> <model>` 临时切换到配置文件或环境变量中指定的模型。

<details>
<summary><strong>查看全部交互命令</strong></summary>

```text
/help
/exit
/clear
/cancel
/sessions
/resume [session-id-or-index]
/context
/memory
/memory search <query>
/memory stats
/memory delete <id>
/memory clear
/save <fact>
/config
/tools
/hitl default|auto
/policy
/audit [N]
/plan <task>
/plan --resume
/model [model-id]
/model <provider> <model-id>
/usage
/skill
/skill list
/skill show <name>
/mcp
```

</details>

## 工具与安全

Vela 内置的主要工具：

| 类别 | 工具 |
| --- | --- |
| 文件 | `read_file`、`write_file`、`list_dir`、`glob` |
| 搜索 | `grep` |
| 命令 | `bash` |
| 记忆 | `save_memory`、`search_memory` |
| Skill | `load_skill` |

联网与浏览器能力统一由 MCP Server 提供，Vela 不再维护一套重复的本地 Web 实现。

写文件、执行命令和远程 MCP 写操作等危险动作会经过 Policy、HITL 和 Audit 处理。

交互模式下按 `Shift+Tab` 切换权限：

- `Default`：启用 HITL、工作区路径限制和命令安全策略
- `Auto (full access)`：当前会话不再请求审批，并关闭路径与命令守卫

再次按 `Shift+Tab` 会恢复启动时的默认策略。

<details>
<summary><strong>了解 Plan 恢复与工具执行语义</strong></summary>

有副作用的工具调用会写入权限为 `0600` 的 `~/.vela/tool-executions.sqlite`。恢复任务时，Vela
直接重放已完成工具的结果；非追加 `write_file` 会先比较目标文件内容，一致时自动对账为完成。
只有状态为 `uncertain` 且无法对账的调用，才会在用户明确确认后重试。

Vela 在工具边界提供 effectively-once 语义，但不承诺外部系统的 exactly-once。Shell、未知 MCP
工具或不支持幂等键的第三方接口仍可能在极端中断窗口中产生重复副作用。这些工具应使用下游
幂等键、唯一约束或自身的状态查询能力。

</details>

## Skills、Memory 与上下文

### Skills

Skill 按以下顺序加载，同名时后层覆盖前层：

1. `builtin`：产品默认能力
2. `user`：`~/.vela/skills/*/SKILL.md`
3. `project`：`.vela/skills/*/SKILL.md`

Vela 先根据名称、描述和标签召回 Top-K 候选，再由模型决定是否调用 `load_skill`。Skill 正文只在
真正加载后进入当前任务。Vela 只读取已有 `SKILL.md`，不会通过模型创建或改写 Skill。

### Memory

| 层级 | 内容 |
| --- | --- |
| 短期记忆 | 当前 Session 的消息、工具调用和工具结果 |
| 静态长期记忆 | `AGENTS.md`、`PAI.md`、`.vela/PAI.md` 和自定义 Prompt 文件 |
| 动态长期记忆 | 按项目隔离的 SQLite 记录，支持去重、TTL、容量治理和相关性召回 |

动态记忆按当前问题召回相关 Top-K，模型也可以调用 `search_memory` 深搜。达到输入预算的 80% 时，
Vela 会压缩旧轮次并保留近期消息和完整的 Tool Call/Result 对。压缩摘要不会自动写入长期记忆。

## 图片输入

在 Prompt 中使用 `@image` 引用本地或远程图片：

```text
分析这张截图 @image:./screenshots/page.png
分析这张截图 @image:</Users/me/Desktop/screen shot.png>
看看这张图片 @image:https://example.com/image.png
```

macOS 终端还支持：

- 在输入框按 `Ctrl+V`，保存剪贴板图片并插入 `@image:<...>`
- 输入 `@clipboard`，发送时读取当前剪贴板图片

图片缓存目录和文件权限分别为 `0700` 和 `0600`。Vela 会自动压缩图片，并在模型不支持多模态时
降级为文本元信息。

## MCP

Vela 可以连接 stdio 或 Streamable HTTP MCP Server，并将远程工具注册为：

```text
mcp__<server-name>__<tool-name>
```

初始化项目级 Chrome DevTools MCP：

```bash
uv run vela mcp init-chrome --scope project
uv run vela mcp list
```

连接已经开启 remote debugging 的 Chrome：

```bash
uv run vela mcp init-chrome \
  --scope project \
  --browser-url http://127.0.0.1:9222
```

授权 Chrome DevTools MCP 前，请确认浏览器中没有不应暴露给 Agent 的个人账号、敏感数据或生产
后台页面。

## 开发与验证

安装开发依赖：

```bash
uv sync --extra dev
```

运行完整检查：

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev python -m pytest
uv build
```

运行 CLI Smoke Test：

```bash
uv run vela --version
uv run vela --help
uv run vela doctor --cwd .
uv run vela -p hello
```

贡献流程与 Pull Request 要求见 [贡献指南](CONTRIBUTING.md)。能力范围和实现状态见
[对齐说明](docs/parity.md)。

## 来源与许可

Vela 基于 MIT 许可的软件 PaiCLI-Python 进行开发，并保留许可证要求的版权和许可声明。产品界面、
命令、Python 包、配置目录和环境变量均使用 Vela 命名，不提供旧运行时命名的兼容入口。

Vela 使用 [MIT License](LICENSE)。
