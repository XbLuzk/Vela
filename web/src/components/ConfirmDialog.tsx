export function ConfirmDialog({
  title,
  description,
  confirmLabel,
  pending = false,
  onCancel,
  onConfirm,
}: {
  title: string;
  description: string;
  confirmLabel: string;
  pending?: boolean;
  onCancel: () => void;
  onConfirm: () => Promise<void>;
}) {
  return (
    <div className="modal-backdrop">
      <section className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <p className="eyebrow">Session history</p>
        <h2 id="confirm-title">{title}</h2>
        <p>{description}</p>
        <div>
          <button type="button" className="quiet-button" disabled={pending} onClick={onCancel}>Cancel</button>
          <button
            type="button"
            className="danger-button"
            disabled={pending}
            onClick={() => void onConfirm()}
          >
            {pending ? "Deleting…" : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}
