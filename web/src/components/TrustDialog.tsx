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
        <h2 id="trust-title">启用此项目的扩展？</h2>
        <p>Vela 已经将下面的目录作为当前工作区。启用后才会读取其中的 AGENTS.md、MCP 服务和 Skills。</p>
        <code>{cwd}</code>
        <div>
          <button type="button" className="quiet-button" onClick={onCancel}>返回</button>
          <button type="button" className="primary-button" onClick={() => void onConfirm()}>启用项目扩展</button>
        </div>
      </section>
    </div>
  );
}
