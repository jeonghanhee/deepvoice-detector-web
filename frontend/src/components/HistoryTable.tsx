import { formatDate } from "../lib/format";
import type { AnalysisResult } from "../types";

interface HistoryTableProps {
  items: AnalysisResult[];
  total: number;
  isLoading: boolean;
}

export function HistoryTable({ items, total, isLoading }: HistoryTableProps) {
  return (
    <section className="history-card">
      <div className="history-top">
        <h2>분석 기록</h2>
        <span className="history-count">총 {total ?? 0}건</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>파일명</th>
              <th>판정</th>
              <th>신뢰도</th>
              <th>음성 길이</th>
              <th>처리 시간</th>
              <th>분석 시각</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={7}>
                  <div className="empty-state">불러오는 중</div>
                </td>
              </tr>
            ) : items.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className="empty-state">
                    <div className="empty-label">분석 기록이 없습니다</div>
                    <div className="empty-sub">홈에서 음성 파일을 업로드하면 여기에 기록됩니다</div>
                  </div>
                </td>
              </tr>
            ) : (
              items.map((item) => {
                const isFake = item.is_fake === 1;
                return (
                  <tr key={item.id}>
                    <td>{item.id}</td>
                    <td className="filename-cell">{item.filename}</td>
                    <td>
                      <span className={`badge ${isFake ? "fake" : "real"}`}>
                        {isFake ? "가짜 음성" : "실제 음성"}
                      </span>
                    </td>
                    <td className="strong-cell">{Number(item.confidence || 0).toFixed(1)}%</td>
                    <td>{item.duration ? `${item.duration.toFixed(2)}s` : "-"}</td>
                    <td>{item.processing_time ? `${item.processing_time.toFixed(3)}s` : "-"}</td>
                    <td>{formatDate(item.created_at)}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
