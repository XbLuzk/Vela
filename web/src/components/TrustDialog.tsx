export function TrustDialog({
  cwd,
  onDecide,
}: {
  cwd: string;
  onDecide: (trusted: boolean) => Promise<void>;
}) {
  return (
    <div className="modal-backdrop">
      <section className="trust-dialog" role="dialog" aria-modal="true" aria-labelledby="trust-title">
        <span className="trust-symbol" aria-hidden="true">⌁</span>
        <p className="eyebrow">Workspace trust</p>
        <h2 id="trust-title">信任这个项目？</h2>
        <p>信任后，Vela 可以加载项目内的 AGENTS.md、MCP 服务和 Skills。</p>
        <code>{cwd}</code>
        <div>
          <button type="button" className="quiet-button" onClick={() => void onDecide(false)}>仅使用内置能力</button>
          <button type="button" className="primary-button" onClick={() => void onDecide(true)}>信任并继续</button>
        </div>
      </section>
    </div>
  );
}
