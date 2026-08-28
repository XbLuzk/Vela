import { useState } from "react";

import type { TaskSnapshot } from "../types";

interface InteractionBarProps {
  task?: TaskSnapshot;
  onApprove: (value: string) => Promise<boolean>;
  onReview: (value: string) => Promise<void>;
}

export function InteractionBar({ task, onApprove, onReview }: InteractionBarProps) {
  const [feedback, setFeedback] = useState("");
  const [decision, setDecision] = useState<{ approvalId?: number; value: string } | null>(null);
  const approvalId = task?.approval?.id;
  const pendingDecision = decision && decision.approvalId === approvalId ? decision.value : null;

  if (!task?.approval && !task?.awaiting_plan_review) return null;

  if (task.approval) {
    const pendingCount = task.approval.pending_count ?? 1;

    async function decide(value: string) {
      if (pendingDecision) return;
      setDecision({ approvalId, value });
      const accepted = await onApprove(value);
      if (!accepted) setDecision(null);
    }

    return (
      <section
        className={`interaction-bar approval-tray ${pendingDecision ? "is-resolving" : ""}`}
        aria-live="polite"
      >
        <div className="approval-content" key={approvalId ?? task.approval.tool_name}>
          <div className="interaction-heading">
            <span className="interaction-kicker">Approval required</span>
            <span className="approval-queue-count">
              {pendingCount > 1 ? `${pendingCount} requests waiting` : "1 request"}
            </span>
          </div>
          <strong>{task.approval.tool_name}</strong>
          <p>{task.approval.description || "This tool may modify the workspace."}</p>
          <details>
            <summary>View input</summary>
            <pre>{JSON.stringify(task.approval.input, null, 2)}</pre>
          </details>
        </div>
        <div className="interaction-actions">
          <button type="button" className="quiet-button" disabled={Boolean(pendingDecision)} onClick={() => void decide("deny")}>
            {pendingDecision === "deny" ? "Denying…" : "Deny"}
          </button>
          <button type="button" className="quiet-button" disabled={Boolean(pendingDecision)} onClick={() => void decide("skip")}>
            {pendingDecision === "skip" ? "Skipping…" : "Skip"}
          </button>
          <button type="button" className="primary-button" disabled={Boolean(pendingDecision)} onClick={() => void decide("approve")}>
            {pendingDecision === "approve" ? "Allowing…" : "Allow"}
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="interaction-bar plan-review" aria-live="assertive">
      <div>
        <span className="interaction-kicker">Plan ready</span>
        <strong>{task.review_feedback_pending ? "Add revision notes" : "Review before execution"}</strong>
        {task.review_feedback_pending ? (
          <input
            value={feedback}
            onChange={(event) => setFeedback(event.target.value)}
            placeholder="For example: add tests before changing the implementation"
            autoFocus
          />
        ) : null}
      </div>
      <div className="interaction-actions">
        <button type="button" className="quiet-button" onClick={() => void onReview("cancel")}>Cancel</button>
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
            Replan
          </button>
        ) : (
          <>
            <button type="button" className="quiet-button" onClick={() => void onReview("modify")}>Revise</button>
            <button type="button" className="primary-button" onClick={() => void onReview("execute")}>Run plan</button>
          </>
        )}
      </div>
    </section>
  );
}
