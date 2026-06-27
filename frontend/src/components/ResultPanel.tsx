import { Activity, ShieldAlert, ShieldCheck } from "lucide-react";
import { API_BASE } from "../constants";
import { formatFileSize } from "../lib/format";
import type { AnalysisResponse } from "../types";
import { MetaItem } from "./MetaItem";
import { ChartBlock } from "./charts/ChartBlock";
import { FrequencyChart } from "./charts/FrequencyChart";

interface ResultPanelProps {
  analysis: AnalysisResponse | null;
}

export function ResultPanel({ analysis }: ResultPanelProps) {
  if (!analysis?.result) return null;

  const result = analysis.result;
  const isFake = result.is_fake === 1;
  const confidence = Number(result.confidence || 0);
  const heatmapSrc = analysis.heatmap_url ? `${API_BASE}${analysis.heatmap_url}` : null;

  return (
    <section className="result-panel">
      <div className="verdict-card">
        <div className="verdict-header">
          <h3>판정 결과</h3>
        </div>
        <div className={`verdict-result ${isFake ? "fake" : "real"}`}>
          <div className={`verdict-symbol ${isFake ? "fake" : "real"}`}>
            {isFake ? <ShieldAlert size={36} /> : <ShieldCheck size={36} />}
          </div>
          <div>
            <div className={`verdict-label ${isFake ? "fake" : "real"}`}>
              {isFake ? "AI 합성 음성" : "실제 음성"}
            </div>
            <div className="verdict-desc">
              {isFake ? "합성 음성으로 판정되었습니다." : "실제 음성으로 판정되었습니다."}
            </div>
          </div>
        </div>

        <div className="confidence-section">
          <div className="conf-label">
            <span>판정 신뢰도</span>
            <span>{confidence.toFixed(1)}%</span>
          </div>
          <div className="conf-track">
            <div
              className={`conf-fill ${isFake ? "fake" : "real"}`}
              style={{ width: `${Math.min(confidence, 100)}%` }}
            />
          </div>
        </div>

        <div className="meta-section">
          <div className="meta-section-label">음성 상세 정보</div>
          <div className="meta-grid">
            <MetaItem
              label="처리 시간"
              value={result.processing_time ? `${result.processing_time.toFixed(3)}s` : "-"}
            />
            <MetaItem label="음성 길이" value={result.duration ? `${result.duration.toFixed(2)}s` : "-"} />
            <MetaItem
              label="샘플레이트"
              value={result.sample_rate ? `${(result.sample_rate / 1000).toFixed(1)} kHz` : "-"}
            />
            <MetaItem
              label="스펙트럼 중심"
              value={result.spectral_centroid ? `${result.spectral_centroid.toFixed(0)} Hz` : "-"}
            />
            <MetaItem
              label="영교차율"
              value={result.zero_crossing_rate ? result.zero_crossing_rate.toFixed(4) : "-"}
            />
            <MetaItem label="파일 크기" value={formatFileSize(result.file_size)} />
          </div>
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-note">
          <Activity size={16} />
        </div>
        <ChartBlock title="주파수 스펙트럼 요약">
          <FrequencyChart data={analysis.frequency_data || []} isFake={isFake} />
        </ChartBlock>
        {heatmapSrc && (
          <div>
            <div className="chart-block-label">Log-Mel Evidence Heatmap</div>
            <div className="heatmap-wrap">
              <img alt="Log-Mel evidence heatmap" src={heatmapSrc} />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
