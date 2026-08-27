import type { SessionSummary } from "../types";

interface SidebarProps {
  sessions: SessionSummary[];
  activeId?: string;
  disabled: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
}

export function Sidebar({ sessions, activeId, disabled, onNew, onSelect }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">V</span>
        <div>
          <strong>Vela</strong>
          <span>local agent</span>
        </div>
      </div>

      <button className="new-session" type="button" onClick={onNew} disabled={disabled}>
        <span aria-hidden="true">＋</span>
        新对话
      </button>

      <nav className="session-list" aria-label="历史对话">
        <p className="sidebar-label">最近</p>
        {sessions.map((session) => (
          <button
            className={`session-item ${session.id === activeId ? "active" : ""}`}
            type="button"
            key={session.id}
            onClick={() => onSelect(session.id)}
            disabled={disabled || session.id === activeId}
          >
            <span>{session.title || "New session"}</span>
            <small>{formatRelativeTime(session.updated_at)}</small>
          </button>
        ))}
      </nav>
    </aside>
  );
}

function formatRelativeTime(value: string): string {
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return "";
  const minutes = Math.max(0, Math.round((Date.now() - timestamp) / 60_000));
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  return `${Math.round(hours / 24)} 天前`;
}
