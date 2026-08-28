import type { LiveRun } from "../types";

export function RunDetails({ run }: { run: LiveRun }) {
  if (!run.thinking && run.tools.length === 0 && run.plan.length === 0) return null;

  const completed = run.plan.filter((task) => task.status === "completed").length;
  const routeLabel = run.status === "running"
    ? "Working"
    : run.status === "completed"
      ? "Completed"
      : run.status === "cancelled"
        ? "Stopped"
        : "Failed";
  const routeSummary = [
    run.plan.length ? `${completed}/${run.plan.length} steps` : null,
    run.tools.length ? `${run.tools.length} tool${run.tools.length === 1 ? "" : "s"}` : null,
  ].filter(Boolean).join(" · ");
  const changedFiles = run.tools.flatMap((tool) => tool.changedFile ? [tool.changedFile] : []);

  return (
    <div className={`run-details run-${run.status}`}>
      <div className="run-route">
        <span className="route-node" aria-hidden="true" />
        <strong>{routeLabel}</strong>
        {routeSummary ? <small>{routeSummary}</small> : null}
      </div>
      {changedFiles.length > 0 ? (
        <details className="detail-block changed-files">
          <summary><span>Files changed</span><small>{changedFiles.length}</small></summary>
          <div className="file-change-list">
            {changedFiles.map((file, index) => (
              <details key={`${file.path}-${index}`}>
                <summary>{file.path}</summary>
                <pre>{file.diff || "File content changed."}{file.truncated ? "\n\nDiff preview truncated." : ""}</pre>
              </details>
            ))}
          </div>
        </details>
      ) : null}
      {run.plan.length > 0 ? (
        <details className="detail-block plan-block">
          <summary>
            <span>Plan</span>
            <small>{completed}/{run.plan.length}</small>
          </summary>
          <ol className="plan-list">
            {run.plan.map((task) => (
              <li key={task.id} data-status={task.status}>
                <span className="plan-state" aria-hidden="true" />
                <div>
                  <strong>{task.id}</strong>
                  <p>{task.description}</p>
                </div>
              </li>
            ))}
          </ol>
        </details>
      ) : null}

      {run.thinking ? (
        <details className="detail-block">
          <summary>
            <span>Thinking</span>
            <small>{run.thinking.length.toLocaleString()} chars</small>
          </summary>
          <pre>{run.thinking}</pre>
        </details>
      ) : null}

      {run.tools.map((tool) => (
        <details className="detail-block tool-block" key={tool.id}>
          <summary>
            <span className="tool-name">{tool.name}</span>
            <small className={tool.isError ? "tool-error" : ""}>
              {tool.result === undefined ? "Running" : tool.isError ? "Failed" : "Done"}
            </small>
          </summary>
          <div className="tool-content">
            <label>Input</label>
            <pre>{JSON.stringify(tool.input ?? {}, null, 2)}</pre>
            {tool.result !== undefined ? (
              <>
                <label>Result</label>
                <pre>{tool.result}</pre>
              </>
            ) : null}
          </div>
        </details>
      ))}
    </div>
  );
}
