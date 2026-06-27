import { API_BASE } from "../constants";
import type { AnalysisResponse, HistoryResponse, StatsResponse } from "../types";

async function readJson<T>(response: Response): Promise<T> {
  const data = (await response.json()) as T;
  if (!response.ok) {
    const message =
      typeof data === "object" && data !== null && "detail" in data
        ? String((data as { detail?: unknown }).detail)
        : "요청 실패";
    throw new Error(message);
  }
  return data;
}

export async function fetchStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE}/api/history/stats`);
  return readJson<StatsResponse>(response);
}

export async function fetchHistory(): Promise<HistoryResponse> {
  const response = await fetch(`${API_BASE}/api/history?limit=50`);
  return readJson<HistoryResponse>(response);
}

export async function uploadAudio(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/audio/upload`, {
    method: "POST",
    body: formData,
  });
  const data = await readJson<AnalysisResponse>(response);

  if (!data.success) {
    throw new Error(data.detail || data.message || "분석 실패");
  }

  return data;
}
