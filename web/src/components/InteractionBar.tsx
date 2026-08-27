import { useState } from "react";

import type { TaskSnapshot } from "../types";

interface InteractionBarProps {
  task?: TaskSnapshot;
  onApprove: (value: string) => Promise<void>;
  onReview: (value: string) => Promise<void>;
}

export function InteractionBar({ task, onApprove, onReview }: InteractionBarProps) {
  const [feedback, setFeedback] = useState("");
  if (!task?.approval && !task?.awaiting_plan_review) return null;

  if (task.approval) {
    return (
      <section className="interaction-bar" aria-live="assertive">
        <div>
          <span className="interaction-kicker">需要确认</span>
          <strong>{task.approval.tool_name}</strong>
          <p>{task.approval.description || "此工具可能修改工作区。"}</p>
          <details>
            <summary>查看参数</summary>
            <pre>{JSON.stringify(task.approval.input, null, 2)}</pre>
          </details>
        </div>
        <div className="interaction-actions">
          <button type="button" className="quiet-button" onClick={() => void onApprove("deny")}>拒绝</button>
          <button type="button" className="quiet-button" onClick={() => void onApprove("skip")}>跳过</button>
          <button type="button" className="primary-button" onClick={() => void onApprove("approve")}>允许</button>
        </div>
      </section>
    );
  }

  return (
    <section className="interaction-bar plan-review" aria-live="assertive">
      <div>
        <span className="interaction-kicker">Plan 已生成</span>
        <strong>{task.review_feedback_pending ? "输入修改要求" : "确认后开始执行"}</strong>
        {task.review_feedback_pending ? (
          <input
            value={feedback}
            onChange={(event) => setFeedback(event.target.value)}
            placeholder="例如：先补测试，再修改实现"
            autoFocus
          />
        ) : null}
      </div>
      <div className="interaction-actions">
        <button type="button" className="quiet-button" onClick={() => void onReview("cancel")}>取消</button>
        {task.review_feedback_pending ? (
          <button
            type="button"
            className="primary-button"
            disabled={!feedback.trim()}
            onClick={() => {
              void onReview(feedback);
              setFeedback("");
            }}
          >
            重新规划
          </button>
        ) : (
          <>
            <button type="button" className="quiet-button" onClick={() => void onReview("modify")}>修改</button>
            <button type="button" className="primary-button" onClick={() => void onReview("execute")}>执行计划</button>
          </>
        )}
      </div>
    </section>
  );
}
