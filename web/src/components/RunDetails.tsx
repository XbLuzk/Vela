import type { LiveRun } from "../types";

export function RunDetails({ run }: { run: LiveRun }) {
  if (!run.thinking && run.tools.length === 0 && run.plan.length === 0) return null;

  return (
    <div className="run-details">
      {run.plan.length > 0 ? (
        <details open className="detail-block plan-block">
          <summary>
            <span>计划</span>
            <small>{run.plan.filter((task) => task.status === "completed").length}/{run.plan.length}</small>
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
            <small>{run.thinking.length.toLocaleString()} 字符</small>
          </summary>
          <pre>{run.thinking}</pre>
        </details>
      ) : null}

      {run.tools.map((tool) => (
        <details className="detail-block tool-block" key={tool.id}>
          <summary>
            <span className="tool-name">{tool.name}</span>
            <small className={tool.isError ? "tool-error" : ""}>
              {tool.result === undefined ? "运行中" : tool.isError ? "失败" : "完成"}
            </small>
          </summary>
          <div className="tool-content">
            <label>输入</label>
            <pre>{JSON.stringify(tool.input ?? {}, null, 2)}</pre>
            {tool.result !== undefined ? (
              <>
                <label>结果</label>
                <pre>{tool.result}</pre>
              </>
            ) : null}
          </div>
        </details>
      ))}
    </div>
  );
}
