import { useEffect, useReducer, useState } from "react";

import { api, connectEvents } from "./api";
import { Composer } from "./components/Composer";
import { Conversation } from "./components/Conversation";
import { InteractionBar } from "./components/InteractionBar";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import { TrustDialog } from "./components/TrustDialog";
import { initialState, reducer } from "./state";
import type { AgentMode, Bootstrap, SessionSummary } from "./types";

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [mode, setMode] = useState<AgentMode>("react");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [trustOpen, setTrustOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  async function runRequest<T>(
    action: Promise<T>,
    options: { rethrow?: boolean } = {},
  ): Promise<T | null> {
    try {
      return await action;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      if (options.rethrow) throw error;
      return null;
    }
  }

  useEffect(() => {
    void runRequest(api.bootstrap()).then((bootstrap) => {
      if (!bootstrap) return;
      dispatch({ type: "loaded", bootstrap });
      setMode(bootstrap.config.prompt.agent_mode);
      if (!bootstrap.ready) setSettingsOpen(true);
    });
    return connectEvents(
      (event) => dispatch({ type: "event", event }),
      (connected) => dispatch({ type: "connection", connected }),
    );
  }, []);

  const bootstrap = state.bootstrap;
  if (!bootstrap) {
    return (
      <div className="boot-screen">
        <span className="boot-mark">V</span>
        <p>正在启动本地工作区…</p>
      </div>
    );
  }

  async function refresh(action: Promise<Bootstrap>): Promise<Bootstrap | null> {
    const next = await runRequest(action);
    if (!next) return null;
    dispatch({ type: "loaded", bootstrap: next });
    if (!next.ready) setSettingsOpen(true);
    return next;
  }

  async function changeSession(action: Promise<SessionSummary>) {
    const session = await runRequest(action);
    if (session) dispatch({ type: "event", event: { type: "session_changed", session } });
  }

  const task = bootstrap.task;
  const messages = bootstrap.session?.messages ?? [];
  const disabled = !bootstrap.ready;

  return (
    <div className="app-shell">
      <Sidebar
        sessions={bootstrap.sessions ?? []}
        activeId={bootstrap.session?.id}
        disabled={Boolean(task?.active) || disabled}
        onNew={() => void changeSession(api.newSession())}
        onSelect={(id) => void changeSession(api.switchSession(id))}
      />

      <section className="workspace">
        <header className="topbar">
          <div>
            <strong>{bootstrap.session?.title || "New session"}</strong>
            <span>{shortPath(bootstrap.cwd)}</span>
          </div>
          <div className="topbar-actions">
            <span className={`connection ${state.connected ? "online" : ""}`}>
              {state.connected ? "Local" : "Reconnecting"}
            </span>
            <button className="icon-button" type="button" onClick={() => setSettingsOpen(true)} aria-label="打开设置">⚙</button>
          </div>
        </header>

        <div className="notices">
          {bootstrap.project_extensions_pending ? (
            <div className="extension-banner">
              <div>
                <strong>检测到项目级扩展</strong>
                <p>当前仍使用内置能力；AGENTS.md、MCP 与 Skills 尚未加载。</p>
              </div>
              <div className="banner-actions">
                <button
                  type="button"
                  className="quiet-button"
                  onClick={() => void refresh(api.trust(false))}
                >
                  保持关闭
                </button>
                <button type="button" className="primary-button" onClick={() => setTrustOpen(true)}>
                  查看并启用
                </button>
              </div>
            </div>
          ) : null}

          {bootstrap.error ? (
            <div className="setup-banner">
              <div>
                <strong>完成模型配置后即可开始</strong>
                <p>{bootstrap.error}</p>
              </div>
              <button type="button" className="primary-button" onClick={() => setSettingsOpen(true)}>打开设置</button>
            </div>
          ) : null}

          {bootstrap.warnings.length > 0 ? (
            <details className="warning-strip">
              <summary>{bootstrap.warnings.length} 条启动提示</summary>
              {bootstrap.warnings.map((warning) => <p key={warning}>{warning}</p>)}
            </details>
          ) : null}
        </div>

        <Conversation messages={messages} liveRun={state.liveRun} cwd={bootstrap.cwd} />

        <InteractionBar
          task={task}
          onApprove={async (value) => { await runRequest(api.approve(value)); }}
          onReview={async (value) => { await runRequest(api.reviewPlan(value)); }}
        />

        <Composer
          mode={mode}
          task={task}
          disabled={disabled}
          onModeChange={setMode}
          onSend={async (message) => {
            await runRequest(api.send(message, mode), { rethrow: true });
          }}
          onCancel={async () => { await runRequest(api.cancel()); }}
        />

        <footer className="statusbar">
          <span>{bootstrap.config.llm.provider} / {bootstrap.config.llm.model}</span>
          <span>{mode === "plan" ? "LangGraph Plan" : "ReAct"}</span>
          <span>{bootstrap.tool_count ?? 0} tools</span>
          <span className="status-spacer" />
          <span>{task?.state ?? "idle"}</span>
          <span>v{bootstrap.version}</span>
        </footer>
      </section>

      <SettingsPanel
        open={settingsOpen}
        bootstrap={bootstrap}
        onClose={() => setSettingsOpen(false)}
        onSave={async (settings) => Boolean((await refresh(api.settings(settings)))?.ready)}
      />

      {bootstrap.project_extensions_pending && trustOpen ? (
        <TrustDialog
          cwd={bootstrap.cwd}
          onCancel={() => setTrustOpen(false)}
          onConfirm={async () => {
            if (await refresh(api.trust(true))) setTrustOpen(false);
          }}
        />
      ) : null}

      {notice ? (
        <button type="button" className="notice" onClick={() => setNotice(null)}>{notice}<span>×</span></button>
      ) : null}
    </div>
  );
}

function shortPath(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.length <= 3 ? path : `…/${parts.slice(-3).join("/")}`;
}
