import type { ReactNode } from "react";

interface IconProps {
  className?: string;
}

function IconFrame({ children, className }: IconProps & { children: ReactNode }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export function VelaMark({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path d="M3.7 4.5h7.6l7.1 17.2-4.1 6.1L3.7 4.5Z" fill="currentColor" />
      <path d="M21 4.5h7.4L18.7 28h-4.4l4.1-6.3L21 4.5Z" fill="var(--brand-pop)" />
      <path d="m18.4 21.7-4.1 6.1h4.4l2.6-6.3-2.9.2Z" fill="currentColor" opacity=".58" />
    </svg>
  );
}

export function PlusIcon(props: IconProps) {
  return <IconFrame {...props}><path d="M12 5v14M5 12h14" /></IconFrame>;
}

export function TrashIcon(props: IconProps) {
  return (
    <IconFrame {...props}>
      <path d="M4.5 7h15M9 7V4.5h6V7M7 7l.7 12h8.6L17 7M10 10.5v5M14 10.5v5" />
    </IconFrame>
  );
}

export function PinIcon(props: IconProps) {
  return <IconFrame {...props}><path d="m9 4 6 6m-7.5 1.5 5-5 4.5 4.5-5 5M12 16l-5 5" /></IconFrame>;
}

export function EditIcon(props: IconProps) {
  return <IconFrame {...props}><path d="m4 20 4.3-1 10-10a2.1 2.1 0 0 0-3-3l-10 10L4 20Z" /><path d="m13.8 7.2 3 3" /></IconFrame>;
}

export function FolderIcon(props: IconProps) {
  return (
    <IconFrame {...props}>
      <path d="M3.5 6.5h6l2 2h9v9.5a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 18Z" />
    </IconFrame>
  );
}

export function SettingsIcon(props: IconProps) {
  return (
    <IconFrame {...props}>
      <circle cx="12" cy="12" r="3" />
      <path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.4 1A7 7 0 0 0 14.8 6L14.5 3h-5L9.2 6a7 7 0 0 0-1.7 1L5.1 6 3 9.5 5.1 11a7 7 0 0 0 0 2L3 14.5 5.1 18l2.4-1a7 7 0 0 0 1.7 1l.3 3h5l.3-3a7 7 0 0 0 1.7-1l2.4 1 2.1-3.5-2.1-1.5a7 7 0 0 0 .1-1Z" />
    </IconFrame>
  );
}

export function CloseIcon(props: IconProps) {
  return <IconFrame {...props}><path d="m7 7 10 10M17 7 7 17" /></IconFrame>;
}

export function SendIcon(props: IconProps) {
  return <IconFrame {...props}><path d="M12 19V5M6.5 10.5 12 5l5.5 5.5" /></IconFrame>;
}
