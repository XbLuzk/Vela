# 测试与 Live 验收

Vela 将测试分成三层，避免 Pull Request 因外部模型额度、网络或浏览器环境波动而随机失败。

## 1. 默认 CI

每次向 `main` Push 或创建 Pull Request 时，都在 Python 3.11、3.13 上执行：

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python -m pytest
uv build
```

这一层不读取 API Key，也不访问真实模型。

## 2. 确定性集成测试

普通 pytest 已覆盖真实 SQLite、LangGraph Checkpoint、stdio MCP 子进程、工具取消、执行日志、
Plan 恢复、Session 持久化和 Run Trace 收尾。Run Trace 会验证完成、错误、取消、提前停止消费、
损坏 JSONL 行与敏感异常脱敏；模型响应使用受控测试实现，保证错误和恢复场景可以稳定复现。

## 3. Live E2E

Live 测试通过真实 OpenAI-compatible 模型运行完整 CLI，并按开关验证 MCP、Chrome 和 Plan。
它们默认跳过，只能显式执行：

```bash
export VELA_API_KEY=your_key
export VELA_PROVIDER=deepseek
export VELA_MODEL=deepseek-v4-flash

uv run pytest tests/e2e --run-live -v
```

可选能力：

```bash
# 运行 Chrome DevTools MCP 验收，需要 Node.js、npx 和 Chrome
export VELA_LIVE_BROWSER=true

# 运行真实 Planner 和 Plan DAG，会产生更多模型调用
export VELA_LIVE_PLAN=true

uv run pytest tests/e2e --run-live -v
```

| 场景 | 默认 CI | Live E2E |
| --- | --- | --- |
| ReAct 与真实模型 | 不运行 | 自动 |
| 真实模型调用 stdio MCP | 不运行 | 自动 |
| Chrome DevTools MCP 浏览器 | 不运行 | `VELA_LIVE_BROWSER=true` |
| 真实 Planner 与 Plan 执行 | 不运行 | `VELA_LIVE_PLAN=true` |
| 工具取消、Checkpoint、Plan 恢复 | 确定性集成测试 | 当前人工验收 |

### GitHub 手动运行

1. 在仓库 Settings → Secrets and variables → Actions 中配置 `VELA_LIVE_API_KEY`。
2. 打开 Actions → Live E2E → Run workflow。
3. 选择 Provider、Model，并决定是否启用 Browser 和 Plan。

Live Workflow 只接受手动触发，不在 Pull Request 中读取 Secret。

### 取消与 Plan 恢复人工验收

这两项依赖真实终端按键时机和模型是否按预期调用慢工具，暂不放入普通 CI：

1. 启动 `uv run vela`，要求 Agent 执行一个持续时间较长的工具任务。
2. 工具运行期间按一次 `Ctrl+C`，确认只取消当前任务且 Session 已保存。
3. 再次启动 `uv run vela --resume`，确认历史完整。
4. 使用 `/plan` 启动包含写工具的任务，执行中取消。
5. 恢复 Session 后运行 `/plan --resume`，确认已完成工具被重放，不确定调用要求再次确认。

后续会用 PTY 驱动替代这部分人工步骤；在此之前，报告结果时必须区分“确定性测试通过”和
“真实终端人工验收通过”。
