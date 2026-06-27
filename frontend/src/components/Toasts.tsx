import type { ToastItem } from "../types";

interface ToastsProps {
  items: ToastItem[];
}

export function Toasts({ items }: ToastsProps) {
  return (
    <div className="toast-wrap" aria-live="polite">
      {items.map((item) => (
        <div className={`toast ${item.type}`} key={item.id}>
          <span>{item.message}</span>
        </div>
      ))}
    </div>
  );
}
