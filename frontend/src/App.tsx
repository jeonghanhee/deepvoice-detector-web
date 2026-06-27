import { useCallback, useEffect, useState } from "react";
import { INITIAL_PROGRESS } from "./constants";
import { fetchHistory, fetchStats, uploadAudio } from "./lib/api";
import { errorMessage } from "./lib/format";
import { Header } from "./components/Header";
import { HistoryTable } from "./components/HistoryTable";
import { ResultPanel } from "./components/ResultPanel";
import { StatsRow } from "./components/StatsRow";
import { Toasts } from "./components/Toasts";
import { UploadPanel } from "./components/UploadPanel";
import type {
  AnalysisResponse,
  HistoryResponse,
  Page,
  ProgressState,
  StatsResponse,
  ToastItem,
  ToastType,
} from "./types";

export function App() {
  const [page, setPage] = useState<Page>("home");
  const [stats, setStats] = useState<Partial<StatsResponse>>({});
  const [history, setHistory] = useState<HistoryResponse>({ items: [], total: 0 });
  const [historyLoading, setHistoryLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState<ProgressState>(INITIAL_PROGRESS);
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const addToast = useCallback((message: string, type: ToastType = "info") => {
    const id = crypto.randomUUID();
    setToasts((items) => [...items, { id, message, type }]);
    window.setTimeout(() => {
      setToasts((items) => items.filter((item) => item.id !== id));
    }, 4000);
  }, []);

  const loadStats = useCallback(async () => {
    try {
      setStats(await fetchStats());
    } catch {
      setStats((current) => current);
    }
  }, []);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      setHistory(await fetchHistory());
    } catch {
      addToast("기록을 불러오지 못했습니다", "error");
    } finally {
      setHistoryLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    loadStats();
    const timer = window.setInterval(loadStats, 30000);
    return () => window.clearInterval(timer);
  }, [loadStats]);

  useEffect(() => {
    if (page === "history") loadHistory();
  }, [loadHistory, page]);

  const runAnalysis = useCallback(async () => {
    if (!selectedFile || isAnalyzing) return;

    setIsAnalyzing(true);
    setProgress({ visible: true, value: 15, label: "파일 업로드 중" });
    const timers = [
      window.setTimeout(() => setProgress({ visible: true, value: 45, label: "음성 특성 추출 중" }), 400),
      window.setTimeout(() => setProgress({ visible: true, value: 70, label: "AI 모델 추론 중" }), 1100),
      window.setTimeout(() => setProgress({ visible: true, value: 90, label: "결과 저장 중" }), 1800),
    ];

    try {
      const data = await uploadAudio(selectedFile);
      setProgress({ visible: true, value: 100, label: "완료" });
      setAnalysis(data);
      addToast("분석이 완료되었습니다", "success");
      await loadStats();
      window.setTimeout(() => setProgress(INITIAL_PROGRESS), 500);
    } catch (error) {
      addToast(`분석 실패: ${errorMessage(error)}`, "error");
      setProgress(INITIAL_PROGRESS);
    } finally {
      timers.forEach((timer) => window.clearTimeout(timer));
      setIsAnalyzing(false);
    }
  }, [addToast, isAnalyzing, loadStats, selectedFile]);

  return (
    <>
      <Toasts items={toasts} />
      <Header page={page} onPageChange={setPage} />
      <main>
        {page === "home" ? (
          <div className="page active">
            <StatsRow stats={stats} />
            <UploadPanel
              isAnalyzing={isAnalyzing}
              onAnalyze={runAnalysis}
              onFileSelect={setSelectedFile}
              progress={progress}
              selectedFile={selectedFile}
            />
            <ResultPanel analysis={analysis} />
          </div>
        ) : (
          <div className="page active">
            <HistoryTable items={history.items} total={history.total} isLoading={historyLoading} />
          </div>
        )}
      </main>
    </>
  );
}
