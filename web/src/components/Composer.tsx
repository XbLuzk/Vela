import { useEffect, useRef, useState } from "react";

import type { AgentMode, TaskSnapshot } from "../types";

interface ComposerProps {
  mode: AgentMode;
  task?: TaskSnapshot;
  disabled: boolean;
  onModeChange: (mode: AgentMode) => void;
  onSend: (message: string) => Promise<void>;
  onCancel: () => Promise<void>;
}

export function Composer({
  mode,
  task,
  disabled,
  onModeChange,
  onSend,
  onCancel,
}: ComposerProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const active = Boolean(task?.active);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [message]);

  async function submit() {
    const value = message.trim();
    if (!value || active || disabled) return;
    setMessage("");
    try {
      await onSend(value);
    } catch {
      setMessage(value);
    }
  }

  return (
    <section className="composer-wrap">
      <div className={`composer ${active ? "busy" : ""}`}>
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submit();
            }
          }}
          placeholder={active ? "任务正在运行，可以先整理下一条消息…" : "描述你想完成的任务"}
          aria-label="消息"
          disabled={disabled}
          rows={1}
        />
        <div className="composer-actions">
          <div className="mode-switch" aria-label="Agent 模式">
            <button
              type="button"
              className={mode === "react" ? "active" : ""}
              onClick={() => onModeChange("react")}
              disabled={active || disabled}
            >
              ReAct
            </button>
            <button
              type="button"
              className={mode === "plan" ? "active" : ""}
              onClick={() => onModeChange("plan")}
              disabled={active || disabled}
            >
              Plan
            </button>
          </div>
          {active ? (
            <button className="cancel-button" type="button" onClick={() => void onCancel()}>
              <span aria-hidden="true" />
              停止
            </button>
          ) : (
            <button
              className="send-button"
              type="button"
              onClick={() => void submit()}
              disabled={!message.trim() || disabled}
              aria-label="发送消息"
            >
              ↑
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
