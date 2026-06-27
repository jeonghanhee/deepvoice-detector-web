import type { ReactNode } from "react";

interface ChartBlockProps {
  title: string;
  children: ReactNode;
}

export function ChartBlock({ title, children }: ChartBlockProps) {
  return (
    <div>
      <div className="chart-block-label">{title}</div>
      <div className="chart-wrap">{children}</div>
    </div>
  );
}
