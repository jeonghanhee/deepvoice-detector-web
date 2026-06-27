export function formatPercent(value?: number | null): string {
  return Number(value) > 0 ? `${Number(value).toFixed(1)}%` : "-";
}

export function formatFileSize(bytes?: number | null): string {
  if (!bytes) return "-";
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export function formatDate(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value.endsWith("Z") ? value : `${value}Z`);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString("ko-KR");
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
