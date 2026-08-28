export function TrustDialog({
  cwd,
  onCancel,
  onConfirm,
}: {
  cwd: string;
  onCancel: () => void;
  onConfirm: () => Promise<void>;
}) {
  return (
    <div className="modal-backdrop">
      <section className="trust-dialog" role="dialog" aria-modal="true" aria-labelledby="trust-title">
        <span className="trust-symbol" aria-hidden="true">⌁</span>
        <p className="eyebrow">Project extensions</p>
        <h2 id="trust-title">Enable project extensions?</h2>
        <p>Vela is using this directory as the current workspace. Enable it to load AGENTS.md, MCP servers, and Skills.</p>
        <code>{cwd}</code>
        <div>
          <button type="button" className="quiet-button" onClick={onCancel}>Back</button>
          <button type="button" className="primary-button" onClick={() => void onConfirm()}>Enable extensions</button>
        </div>
      </section>
    </div>
  );
}
