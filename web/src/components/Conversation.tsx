import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { LiveRun, Message } from "../types";
import { VelaMark } from "./Icons";
import { RunDetails } from "./RunDetails";

interface ConversationProps {
  messages: Message[];
  liveRun: LiveRun | null;
  completedRun: LiveRun | null;
  cwd: string;
}

export function Conversation({ messages, liveRun, completedRun, cwd }: ConversationProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, liveRun?.text, liveRun?.thinking, liveRun?.tools.length, completedRun?.id]);

  const visible = messages.filter(
    (message) =>
      message.role === "user" ||
      (message.role === "assistant" && Boolean(textContent(message.content).trim())),
  );
  const empty = visible.length === 0 && !liveRun;

  return (
    <main className="conversation" aria-live="polite">
      <div className={`message-column ${empty ? "message-column-empty" : ""}`}>
        {empty ? <EmptyState cwd={cwd} /> : null}
        {visible.map((message, index) => (
          <article className={`message message-${message.role}`} key={`${message.role}-${index}`}>
            <header className="message-meta">
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{message.role === "user" ? "You" : "Vela"}</strong>
            </header>
            <div className="message-content">
              {message.role === "user" ? (
                <p>{textContent(message.content)}</p>
              ) : (
                <div className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{textContent(message.content)}</ReactMarkdown>
                </div>
              )}
            </div>
          </article>
        ))}

        {liveRun ? (
          <article className="message message-assistant live-message">
            <header className="message-meta">
              <span>{String(visible.length + 1).padStart(2, "0")}</span>
              <strong>Vela</strong>
            </header>
            <div className="message-content">
              <RunDetails run={liveRun} />
              {liveRun.text ? (
                <div className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{liveRun.text}</ReactMarkdown>
                </div>
              ) : liveRun.status === "running" ? (
                <div className="route-progress" aria-label="Vela is working">
                  <span />
                  <small>Finding the next step</small>
                </div>
              ) : null}
              {liveRun.error ? <p className="run-error">{liveRun.error}</p> : null}
            </div>
          </article>
        ) : null}
        {completedRun ? (
          <section className="completed-run" aria-label="Files changed in the latest run">
            <RunDetails run={completedRun} />
          </section>
        ) : null}
        <div ref={endRef} />
      </div>
    </main>
  );
}

function EmptyState({ cwd }: { cwd: string }) {
  const name = cwd.split(/[\\/]/).filter(Boolean).at(-1) ?? "workspace";
  return (
    <section className="empty-state">
      <div className="empty-mark" aria-hidden="true"><VelaMark /></div>
      <p className="empty-context">Ready <span>/</span> {name}</p>
      <h1>What do you want to change?</h1>
      <p>Ask about the code or give Vela a concrete task.</p>
    </section>
  );
}

function textContent(content: Message["content"]): string {
  if (typeof content === "string") return content;
  return content
    .map((item) => {
      const text = item.text;
      if (typeof text === "string") return text;
      const path = item.path;
      return typeof path === "string" ? `[image: ${path}]` : "";
    })
    .filter(Boolean)
    .join("\n");
}
