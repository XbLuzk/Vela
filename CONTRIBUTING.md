# Contributing to Vela

感谢你愿意参与 Vela 的开发。Vela 是一个运行在终端中的 Python AI Agent CLI，支持
ReAct、Plan-and-Execute、MCP、Skill、Memory 和多模态输入。

## 开发环境

- Python 3.11+
- 推荐使用 [uv](https://docs.astral.sh/uv/)

安装依赖并运行：

```bash
uv sync --extra dev
uv run vela --version
uv run vela doctor --cwd .
```

## 提交改动

开始修改前，请先确认相关命令、实现和测试的位置：

- CLI 与 REPL：`src/vela/entrypoints/`
- ReAct 和 Plan：`src/vela/agent/`
- 模型适配：`src/vela/llm/`
- 工具与执行策略：`src/vela/tools/`、`src/vela/policy/`
- MCP：`src/vela/mcp/`
- Memory 与 Skill：`src/vela/memory/`、`src/vela/skill/`

公开行为发生变化时，请同步更新 README、相关文档和测试。不要提交 `.env`、API Key、
Token、Cookie、私钥、本地数据库、日志或其他个人文件。

## 验证

提交前至少运行：

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev python -m pytest
```

如果修改了打包配置，再运行：

```bash
uv build
```

CLI 或终端交互改动除自动化测试外，还应执行相应的真实命令进行验证。需要真实模型的测试
应使用自己的环境变量或本地配置，不要把凭据写入仓库。

## Pull Request

PR 描述应说明：

1. 解决了什么问题。
2. 采用了什么实现方式。
3. 如何验证，以及哪些场景尚未验证。
4. 是否涉及兼容性、持久化数据、安全策略或外部工具副作用。

请尽量保持一次 PR 只解决一个清晰问题，并为错误路径和恢复路径补充测试。

## License

提交代码即表示你同意按照仓库的 [MIT License](LICENSE) 发布该贡献。
