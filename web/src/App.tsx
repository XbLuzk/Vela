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
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    void api.bootstrap().then((bootstrap) => {
      dispatch({ type: "loaded", bootstrap });
      setMode(bootstrap.config.prompt.agent_mode);
      if (!bootstrap.ready && !bootstrap.trust_required) setSettingsOpen(true);
    }).catch((error: Error) => setNotice(error.message));
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
    try {
      const next = await action;
      dispatch({ type: "loaded", bootstrap: next });
      if (!next.ready && !next.trust_required) setSettingsOpen(true);
      return next;
    } catch (error) {
      setNotice((error as Error).message);
      return null;
    }
  }

  async function changeSession(action: Promise<SessionSummary>) {
    try {
      const session = await action;
      dispatch({ type: "event", event: { type: "session_changed", session } });
    } catch (error) {
      setNotice((error as Error).message);
    }
  }

  const task = bootstrap.task;
  const messages = bootstrap.session?.messages ?? [];
  const disabled = !bootstrap.ready || bootstrap.trust_required;

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
          onApprove={async (value) => {
            try { await api.approve(value); } catch (error) { setNotice((error as Error).message); }
          }}
          onReview={async (value) => {
            try { await api.reviewPlan(value); } catch (error) { setNotice((error as Error).message); }
          }}
        />

        <Composer
          mode={mode}
          task={task}
          disabled={disabled}
          onModeChange={setMode}
          onSend={async (message) => {
            try {
              await api.send(message, mode);
            } catch (error) {
              setNotice((error as Error).message);
              throw error;
            }
          }}
          onCancel={async () => {
            try { await api.cancel(); } catch (error) { setNotice((error as Error).message); }
          }}
        />

        <footer className="statusbar">
          <span>{bootstrap.provider ?? bootstrap.config.llm.provider} / {bootstrap.model ?? bootstrap.config.llm.model}</span>
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

      {bootstrap.trust_required ? (
        <TrustDialog
          cwd={bootstrap.cwd}
          onDecide={async (trusted) => {
            await refresh(api.trust(trusted));
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
