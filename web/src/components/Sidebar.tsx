import type { SessionSummary } from "../types";
import { PlusIcon, TrashIcon, VelaMark } from "./Icons";

interface SidebarProps {
  sessions: SessionSummary[];
  activeId?: string;
  disabled: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (session: SessionSummary) => void;
}

export function Sidebar({ sessions, activeId, disabled, onNew, onSelect, onDelete }: SidebarProps) {
  return (
    <aside className={`sidebar ${disabled ? "task-active" : ""}`}>
      <div className="brand">
        <span className="brand-mark" aria-hidden="true"><VelaMark /></span>
        <div>
          <strong>Vela</strong>
          <span>Local workspace</span>
        </div>
      </div>

      <button className="new-session" type="button" onClick={onNew} disabled={disabled}>
        <PlusIcon />
        New session
      </button>

      <nav className="session-list" aria-label="Session history">
        <p className="sidebar-label"><span>Sessions</span><small>{sessions.length}</small></p>
        {sessions.map((session) => (
          <div className={`session-row ${session.id === activeId ? "active" : ""}`} key={session.id}>
            <button
              className="session-item"
              type="button"
              onClick={() => onSelect(session.id)}
              disabled={disabled || session.id === activeId}
            >
              <span>{session.title || "New session"}</span>
              <small>{formatRelativeTime(session.updated_at)}</small>
            </button>
            <button
              className="session-delete"
              type="button"
              aria-label={`Delete session ${session.title || "New session"}`}
              title="Delete session"
              disabled={disabled}
              onClick={() => onDelete(session)}
            >
              <TrashIcon />
            </button>
          </div>
        ))}
      </nav>
    </aside>
  );
}

function formatRelativeTime(value: string): string {
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return "";
  const minutes = Math.max(0, Math.round((Date.now() - timestamp) / 60_000));
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
