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

export interface ModelAnalysisResult {
  model_key: string;
  model_label: string;
  model_path: string;
  is_fake: number;
  result: "REAL" | "FAKE";
  confidence: number;
  real_prob: number;
  fake_prob: number;
  threshold: number;
}

export interface AnalysisResponse {
  success: boolean;
  message: string;
  result?: AnalysisResult | null;
  frequency_data?: number[] | null;
  decision_model?: string | null;
  final_label?: string | null;
  detail_type?: string | null;
  model_results?: ModelAnalysisResult[] | null;
  heatmap_url?: string | null;
  heatmap_metadata?: Record<string, unknown> | null;
  detail?: string;
}

export interface HistoryResponse {
  total: number;
  items: AnalysisResult[];
}
