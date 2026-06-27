export type Page = "home" | "history";

export type ToastType = "info" | "success" | "error";

export interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
}

export interface ProgressState {
  visible: boolean;
  value: number;
  label: string;
}

export interface StatsResponse {
  total_analyses: number;
  fake_count: number;
  real_count: number;
  avg_confidence: number;
  avg_processing_time: number;
}

export interface AnalysisResult {
  id: number;
  filename: string;
  file_size?: number | null;
  duration?: number | null;
  sample_rate?: number | null;
  is_fake: number;
  confidence: number;
  processing_time?: number | null;
  spectral_centroid?: number | null;
  zero_crossing_rate?: number | null;
  created_at: string;
}

export interface AnalysisResponse {
  success: boolean;
  message: string;
  result?: AnalysisResult | null;
  frequency_data?: number[] | null;
  heatmap_url?: string | null;
  heatmap_metadata?: Record<string, unknown> | null;
  detail?: string;
}

export interface HistoryResponse {
  total: number;
  items: AnalysisResult[];
}
