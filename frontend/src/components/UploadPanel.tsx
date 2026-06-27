import { useRef, useState, type DragEvent } from "react";
import { BarChart3, FileAudio, UploadCloud } from "lucide-react";
import { ACCEPTED_AUDIO } from "../constants";
import { formatFileSize } from "../lib/format";
import type { ProgressState } from "../types";

interface UploadPanelProps {
  selectedFile: File | null;
  onFileSelect: (file: File) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  progress: ProgressState;
}

export function UploadPanel({
  selectedFile,
  onFileSelect,
  onAnalyze,
  isAnalyzing,
  progress,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) onFileSelect(file);
  };

  return (
    <section className="upload-card">
      <div className="upload-card-header">
        <h2>음성 파일 업로드</h2>
      </div>

      <button
        className={`drop-zone ${isDragging ? "drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={handleDrop}
        type="button"
      >
        <input
          accept={ACCEPTED_AUDIO}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onFileSelect(file);
          }}
          ref={inputRef}
          type="file"
        />
        <span className="drop-icon-wrap">
          <UploadCloud size={30} />
        </span>
        <span className="drop-title">음성 파일 선택</span>
        <span className="drop-sub">WAV · MP3 · FLAC · OGG · M4A · WEBM / 최대 50MB</span>
      </button>

      {selectedFile && (
        <div className="file-preview">
          <div className="file-icon">
            <FileAudio size={22} />
          </div>
          <div className="file-info">
            <div className="file-name">{selectedFile.name}</div>
            <div className="file-meta">
              {formatFileSize(selectedFile.size)} · {selectedFile.type || "audio"}
            </div>
          </div>
          <button className="btn-primary" disabled={isAnalyzing} onClick={onAnalyze} type="button">
            {isAnalyzing ? (
              <>
                <span className="spinner" />
                분석 중
              </>
            ) : (
              <>
                <BarChart3 size={17} />
                분석 시작
              </>
            )}
          </button>
        </div>
      )}

      {progress.visible && (
        <div className="progress-wrap">
          <div className="progress-top">
            <span>{progress.label}</span>
            <span>{progress.value}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.value}%` }} />
          </div>
        </div>
      )}
    </section>
  );
}
