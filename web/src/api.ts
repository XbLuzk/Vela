import type { AgentMode, Bootstrap, RuntimeEvent, SessionSummary } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  bootstrap: () => request<Bootstrap>("/api/bootstrap"),
  send: (message: string, mode: AgentMode) =>
    request<{ run_id: string }>("/api/messages", {
      method: "POST",
      body: JSON.stringify({ message, mode }),
    }),
  cancel: () => request<{ cancelled: boolean }>("/api/cancel", { method: "POST" }),
  approve: (value: string) =>
    request<{ message: string }>("/api/approval", {
      method: "POST",
      body: JSON.stringify({ value }),
    }),
  reviewPlan: (value: string) =>
    request<{ message: string }>("/api/plan-review", {
      method: "POST",
      body: JSON.stringify({ value }),
    }),
  newSession: () => request<SessionSummary>("/api/sessions", { method: "POST" }),
  switchSession: (id: string) =>
    request<SessionSummary>(`/api/sessions/${encodeURIComponent(id)}`, { method: "POST" }),
  trust: (trusted: boolean) =>
    request<Bootstrap>("/api/trust", {
      method: "POST",
      body: JSON.stringify({ trusted }),
    }),
  settings: (settings: Record<string, unknown>) =>
    request<Bootstrap>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    }),
};

export function connectEvents(
  onEvent: (event: RuntimeEvent) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  const source = new EventSource("/api/events");
  source.onopen = () => onConnection(true);
  source.onerror = () => onConnection(false);
  source.onmessage = (message) => {
    onEvent(JSON.parse(message.data) as RuntimeEvent);
  };
  return () => source.close();
}
