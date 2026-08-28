import { useState } from "react";

import type { SessionSummary } from "../types";
import { EditIcon, PinIcon, PlusIcon, TrashIcon, VelaMark } from "./Icons";

interface SidebarProps {
  sessions: SessionSummary[];
  activeId?: string;
  disabled: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (session: SessionSummary) => void;
  onRename: (id: string, title: string) => void;
  onPin: (id: string, pinned: boolean) => void;
}

export function Sidebar({
  sessions, activeId, disabled, onNew, onSelect, onDelete, onRename, onPin,
}: SidebarProps) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [title, setTitle] = useState("");

  function startRename(session: SessionSummary) {
    setRenamingId(session.id);
    setTitle(session.title || "New session");
  }

  function finishRename(session: SessionSummary) {
    const nextTitle = title.trim();
    setRenamingId(null);
    if (nextTitle && nextTitle !== session.title) onRename(session.id, nextTitle);
  }

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
            <div className="session-item">
              {renamingId === session.id ? (
                <input
                  className="session-rename"
                  value={title}
                  autoFocus
                  aria-label="Session title"
                  onChange={(event) => setTitle(event.target.value)}
                  onBlur={() => finishRename(session)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") event.currentTarget.blur();
                    if (event.key === "Escape") setRenamingId(null);
                  }}
                />
              ) : (
                <button
                  className="session-select"
                  type="button"
                  onClick={() => onSelect(session.id)}
                  disabled={disabled || session.id === activeId}
                >
                  <span>{session.pinned ? <PinIcon /> : null}{session.title || "New session"}</span>
                  <small>{formatRelativeTime(session.updated_at)}</small>
                </button>
              )}
            </div>
            <div className="session-actions">
              <button
                className="session-action"
                type="button"
                aria-label={session.pinned ? "Unpin session" : "Pin session"}
                title={session.pinned ? "Unpin session" : "Pin session"}
                disabled={disabled}
                onClick={() => onPin(session.id, !session.pinned)}
              ><PinIcon /></button>
              <button className="session-action" type="button" aria-label="Rename session" title="Rename session" disabled={disabled} onClick={() => startRename(session)}><EditIcon /></button>
              <button
                className="session-delete"
                type="button"
                aria-label={`Delete session ${session.title || "New session"}`}
                title="Delete session"
                disabled={disabled}
                onClick={() => onDelete(session)}
              ><TrashIcon /></button>
            </div>
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
