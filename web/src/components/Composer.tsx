import { useEffect, useRef, useState } from "react";

import type { AgentMode, TaskSnapshot } from "../types";
import { SendIcon } from "./Icons";

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
  const [sending, setSending] = useState(false);
  const sendingRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const active = Boolean(task?.active);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [message]);

  async function submit() {
    const value = messageForSubmission(message, active, disabled, sendingRef.current);
    if (!value) return;
    sendingRef.current = true;
    setSending(true);
    setMessage("");
    try {
      await onSend(value);
    } catch {
      setMessage((current) => current || value);
    } finally {
      sendingRef.current = false;
      setSending(false);
    }
  }

  return (
    <section className="composer-wrap">
      <div className={`composer ${active || sending ? "busy" : ""}`}>
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
          placeholder={active ? "Draft your next message…" : "Describe a task or ask about the code"}
          aria-label="Message"
          disabled={disabled}
          rows={1}
        />
        <div className="composer-actions">
          <div className="mode-switch" aria-label="Agent mode">
            <button
              type="button"
              className={mode === "react" ? "active" : ""}
              onClick={() => onModeChange("react")}
              disabled={active || disabled || sending}
            >
              ReAct
            </button>
            <button
              type="button"
              className={mode === "plan" ? "active" : ""}
              onClick={() => onModeChange("plan")}
              disabled={active || disabled || sending}
            >
              Plan
            </button>
          </div>
          <span className="composer-hint"><kbd>Enter</kbd> send <i>·</i> <kbd>Shift Enter</kbd> newline</span>
          {active ? (
            <button className="cancel-button" type="button" onClick={() => void onCancel()}>
              <span aria-hidden="true" />
              Stop
            </button>
          ) : (
            <button
              className="send-button"
              type="button"
              onClick={() => void submit()}
              disabled={!message.trim() || disabled || sending}
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

export function messageForSubmission(
  message: string,
  active: boolean,
  disabled: boolean,
  sending: boolean,
): string | null {
  const value = message.trim();
  return value && !active && !disabled && !sending ? value : null;
}
