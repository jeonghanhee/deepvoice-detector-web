import { History, Home } from "lucide-react";
import type { Page } from "../types";

interface HeaderProps {
  page: Page;
  onPageChange: (page: Page) => void;
}

export function Header({ page, onPageChange }: HeaderProps) {
  return (
    <header>
      <a className="logo" href="#" onClick={(event) => event.preventDefault()}>
        <span className="logo-mark">D</span>
        <span className="logo-text">DeepVoice Detector</span>
      </a>
      <nav>
        <button
          className={`nav-btn ${page === "home" ? "active" : ""}`}
          onClick={() => onPageChange("home")}
          type="button"
        >
          <Home size={16} />
          홈
        </button>
        <button
          className={`nav-btn ${page === "history" ? "active" : ""}`}
          onClick={() => onPageChange("history")}
          type="button"
        >
          <History size={16} />
          분석 기록
        </button>
      </nav>
    </header>
  );
}
