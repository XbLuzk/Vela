import { useEffect, useReducer, useState } from "react";

import { api, connectEvents } from "./api";
import { Composer } from "./components/Composer";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { Conversation } from "./components/Conversation";
import { InteractionBar } from "./components/InteractionBar";
import { CloseIcon, FolderIcon, SettingsIcon, VelaMark } from "./components/Icons";
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
  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [projectPicking, setProjectPicking] = useState(false);

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
        <span className="boot-mark"><VelaMark /></span>
        <p>Starting local workspace…</p>
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

  async function chooseProject() {
    setProjectPicking(true);
    try {
      const result = await runRequest(api.pickProject());
      if (!result?.selected || !result.bootstrap) return;
      dispatch({ type: "loaded", bootstrap: result.bootstrap });
      setMode(result.bootstrap.config.prompt.agent_mode);
      setTrustOpen(false);
      if (!result.bootstrap.ready) setSettingsOpen(true);
    } finally {
      setProjectPicking(false);
    }
  }

  const task = bootstrap.task;
  const messages = bootstrap.session?.messages ?? [];
  const disabled = !bootstrap.ready;

  return (
    <div className="app-shell">
      <Sidebar
        sessions={bootstrap.sessions ?? []}
        activeId={bootstrap.session?.id}
        disabled={Boolean(task?.active)}
        onNew={() => void changeSession(api.newSession())}
        onSelect={(id) => void changeSession(api.switchSession(id))}
        onDelete={setPendingDelete}
      />

      <section className={`workspace ${task?.active ? "workspace-active" : ""}`}>
        <header className="topbar">
          <div className="topbar-context">
            <span className="topbar-kicker">Session</span>
            <strong>{bootstrap.session?.title || "New session"}</strong>
            <button
              className="project-switch"
              type="button"
              title={bootstrap.cwd}
              disabled={Boolean(task?.active) || projectPicking}
              onClick={() => void chooseProject()}
            >
              <FolderIcon />
              <span>{projectPicking ? "Selecting directory…" : shortPath(bootstrap.cwd)}</span>
              <small>Change</small>
            </button>
          </div>
          <div className="topbar-actions">
            <span className={`connection ${state.connected ? "online" : ""}`}>
              {state.connected ? "Local" : "Reconnecting"}
            </span>
            <button className="icon-button" type="button" onClick={() => setSettingsOpen(true)} aria-label="Open settings"><SettingsIcon /></button>
          </div>
        </header>

        <div className="notices">
          {bootstrap.project_extensions_pending ? (
            <div className="extension-banner">
              <div>
                <strong>Project extensions found</strong>
                <p>Built-in tools are active. AGENTS.md, MCP servers, and Skills have not been loaded.</p>
              </div>
              <div className="banner-actions">
                <button
                  type="button"
                  className="quiet-button"
                  onClick={() => void refresh(api.trust(false))}
                >
                  Keep disabled
                </button>
                <button type="button" className="primary-button" onClick={() => setTrustOpen(true)}>
                  Review
                </button>
              </div>
            </div>
          ) : null}

          {bootstrap.error ? (
            <div className="setup-banner">
              <div>
                <strong>Configure a model to get started</strong>
                <p>{bootstrap.error}</p>
              </div>
              <button type="button" className="primary-button" onClick={() => setSettingsOpen(true)}>Open settings</button>
            </div>
          ) : null}

          {bootstrap.warnings.length > 0 ? (
            <details className="warning-strip">
              <summary>{bootstrap.warnings.length} startup warning{bootstrap.warnings.length === 1 ? "" : "s"}</summary>
              {bootstrap.warnings.map((warning) => <p key={warning}>{warning}</p>)}
            </details>
          ) : null}
        </div>

        <Conversation messages={messages} liveRun={state.liveRun} cwd={bootstrap.cwd} />

        <InteractionBar
          task={task}
          onApprove={async (value) => Boolean(await runRequest(api.approve(value)))}
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
          <span title={`${bootstrap.config.llm.provider} / ${bootstrap.config.llm.model}`}>{bootstrap.config.llm.model}</span>
          <span>{mode === "plan" ? "LangGraph Plan" : "ReAct"} · {bootstrap.tool_count ?? 0} tools</span>
          <span className="status-spacer" />
          <span className={`runtime-state ${task?.active ? "active" : ""}`}>{task?.state ?? "idle"}</span>
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

      {pendingDelete ? (
        <ConfirmDialog
          title="Delete this session?"
          description={`“${pendingDelete.title || "New session"}” will be permanently deleted from this device. Project files will not be affected.`}
          confirmLabel="Delete session"
          pending={deleting}
          onCancel={() => setPendingDelete(null)}
          onConfirm={async () => {
            setDeleting(true);
            try {
              if (await refresh(api.deleteSession(pendingDelete.id))) setPendingDelete(null);
            } finally {
              setDeleting(false);
            }
          }}
        />
      ) : null}

      {notice ? (
        <button type="button" className="notice" onClick={() => setNotice(null)}>{notice}<CloseIcon /></button>
      ) : null}
    </div>
  );
}

function shortPath(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.length <= 3 ? path : `…/${parts.slice(-3).join("/")}`;
}
