import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { LiveRun, Message } from "../types";
import { RunDetails } from "./RunDetails";

interface ConversationProps {
  messages: Message[];
  liveRun: LiveRun | null;
  cwd: string;
}

export function Conversation({ messages, liveRun, cwd }: ConversationProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, liveRun?.text, liveRun?.thinking, liveRun?.tools.length]);

  const visible = messages.filter(
    (message) =>
      message.role === "user" ||
      (message.role === "assistant" && Boolean(textContent(message.content).trim())),
  );

  return (
    <main className="conversation" aria-live="polite">
      {visible.length === 0 && !liveRun ? <EmptyState cwd={cwd} /> : null}
      <div className="message-column">
        {visible.map((message, index) => (
          <article className={`message message-${message.role}`} key={`${message.role}-${index}`}>
            {message.role === "user" ? (
              <p>{textContent(message.content)}</p>
            ) : (
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{textContent(message.content)}</ReactMarkdown>
              </div>
            )}
          </article>
        ))}

        {liveRun ? (
          <article className="message message-assistant live-message">
            <RunDetails run={liveRun} />
            {liveRun.text ? (
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{liveRun.text}</ReactMarkdown>
              </div>
            ) : liveRun.status === "running" ? (
              <div className="thinking-indicator">
                <span />
                <span />
                <span />
              </div>
            ) : null}
            {liveRun.error ? <p className="run-error">{liveRun.error}</p> : null}
          </article>
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
      <div className="empty-orbit" aria-hidden="true">
        <span />
      </div>
      <p className="eyebrow">{name}</p>
      <h1>从一个具体任务开始。</h1>
      <p>Vela 可以阅读代码、修改文件、运行命令，也可以用 LangGraph 拆解复杂计划。</p>
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
