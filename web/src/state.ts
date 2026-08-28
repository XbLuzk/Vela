import type {
  AppState,
  Bootstrap,
  LiveRun,
  Message,
  PlanTaskActivity,
  RuntimeEvent,
  SessionSummary,
  TaskSnapshot,
  ToolActivity,
} from "./types";

export type Action =
  | { type: "loaded"; bootstrap: Bootstrap }
  | { type: "connection"; connected: boolean }
  | { type: "event"; event: RuntimeEvent };

export const initialState: AppState = {
  bootstrap: null,
  liveRun: null,
  completedRun: null,
  connected: false,
};

export function reducer(state: AppState, action: Action): AppState {
  if (action.type === "loaded") {
    return { ...state, bootstrap: action.bootstrap };
  }
  if (action.type === "connection") {
    return { ...state, connected: action.connected };
  }
  return applyRuntimeEvent(state, action.event);
}

function applyRuntimeEvent(state: AppState, event: RuntimeEvent): AppState {
  if (event.type === "connected") return { ...state, connected: true };
  if (event.type === "bootstrap") {
    return { ...state, bootstrap: event.bootstrap as Bootstrap, liveRun: null, completedRun: null };
  }
  if (!state.bootstrap) return state;

  if (event.type === "user_message") {
    const session = state.bootstrap.session;
    const message = event.message as Message;
    const updatedSession = session
      ? { ...session, messages: [...(session.messages ?? []), message] }
      : session;
    return {
      ...state,
      bootstrap: { ...state.bootstrap, session: updatedSession },
      liveRun: newRun(String(event.run_id ?? "")),
      completedRun: null,
    };
  }

  if (event.type === "task_state") {
    return {
      ...state,
      bootstrap: { ...state.bootstrap, task: event.task as TaskSnapshot },
    };
  }

  if (event.type === "session_updated" || event.type === "session_changed") {
    const session = event.session as SessionSummary;
    const sessions = replaceSession(
      state.bootstrap.sessions ?? [],
      session,
      state.bootstrap.session,
    );
    return {
      ...state,
      liveRun: null,
      completedRun: event.type === "session_updated" && hasChangedFiles(state.liveRun)
        ? state.liveRun
        : null,
      bootstrap: { ...state.bootstrap, session, sessions },
    };
  }

  if (event.type === "session_metadata_updated") {
    const session = event.session as SessionSummary;
    const sessions = replaceSessionMetadata(state.bootstrap.sessions ?? [], session);
    const current = state.bootstrap.session?.id === session.id
      ? { ...state.bootstrap.session, ...session }
      : state.bootstrap.session;
    return { ...state, bootstrap: { ...state.bootstrap, session: current, sessions } };
  }

  const run = state.liveRun ?? newRun(String(event.run_id ?? ""));
  if (event.type === "text_delta") {
    return withRun(state, { ...run, text: run.text + String(event.text ?? "") });
  }
  if (event.type === "thinking_delta") {
    return withRun(state, { ...run, thinking: run.thinking + String(event.thinking ?? "") });
  }
  if (event.type === "tool_call") {
    const tool: ToolActivity = {
      id: String(event.tool_call_id ?? `${event.name ?? "tool"}-${run.tools.length}`),
      name: String(event.name ?? "tool"),
      input: (event.input as Record<string, unknown>) ?? {},
    };
    return withRun(state, { ...run, tools: [...run.tools, tool] });
  }
  if (event.type === "tool_result") {
    const id = String(event.tool_call_id ?? "");
    const tools = run.tools.map((tool) =>
      tool.id === id
        ? {
            ...tool,
            result: String(event.result ?? ""),
            isError: Boolean(event.is_error),
            changedFile: event.changed_file as ToolActivity["changedFile"],
          }
        : tool,
    );
    return withRun(state, { ...run, tools });
  }
  if (event.type === "plan_created") {
    const raw = (event.plan as { tasks?: Record<string, { id: string; description: string }> })
      ?.tasks;
    const plan = Object.values(raw ?? {}).map<PlanTaskActivity>((task) => ({
      id: task.id,
      description: task.description,
      status: "pending",
    }));
    return withRun(state, { ...run, plan });
  }
  if (event.type === "plan_task_started" || event.type === "plan_task_done") {
    const id = String(event.task_id ?? "");
    const status =
      event.type === "plan_task_started"
        ? "running"
        : event.task_status === "failed"
          ? "failed"
          : "completed";
    const plan = updatePlanTask(
      run.plan,
      id,
      String(event.task_description ?? id),
      status,
    );
    return withRun(state, { ...run, plan });
  }
  if (event.type === "done") {
    return withRun(state, { ...run, status: "completed" });
  }
  if (event.type === "run_cancelled") {
    return withRun(state, { ...run, status: "cancelled" });
  }
  if (event.type === "run_failed" || event.type === "error") {
    return withRun(state, {
      ...run,
      status: "failed",
      error: String(event.error ?? "Run failed"),
    });
  }
  return state;
}

function newRun(id: string): LiveRun {
  return { id, text: "", thinking: "", tools: [], plan: [], status: "running" };
}

function withRun(state: AppState, liveRun: LiveRun): AppState {
  return { ...state, liveRun };
}

function hasChangedFiles(run: LiveRun | null): run is LiveRun {
  return Boolean(run?.tools.some((tool) => tool.changedFile));
}

function replaceSession(
  sessions: SessionSummary[],
  current: SessionSummary,
  previous?: SessionSummary,
): SessionSummary[] {
  const summary = { ...current, messages: undefined };
  return sortSessions([
    summary,
    ...sessions.filter(
      (session) =>
        session.id !== current.id &&
        !(previous?.message_count === 0 && session.id === previous.id),
    ),
  ]);
}

function replaceSessionMetadata(sessions: SessionSummary[], current: SessionSummary): SessionSummary[] {
  return sortSessions(sessions.map((session) => session.id === current.id ? { ...session, ...current } : session));
}

function sortSessions(sessions: SessionSummary[]): SessionSummary[] {
  return [...sessions].sort((left, right) => Number(right.pinned) - Number(left.pinned)
    || right.updated_at.localeCompare(left.updated_at));
}

function updatePlanTask(
  tasks: PlanTaskActivity[],
  id: string,
  description: string,
  status: PlanTaskActivity["status"],
): PlanTaskActivity[] {
  if (!tasks.some((task) => task.id === id)) {
    return [...tasks, { id, description, status }];
  }
  return tasks.map((task) => (task.id === id ? { ...task, status } : task));
}
