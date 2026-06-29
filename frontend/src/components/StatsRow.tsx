import { Activity, Database, ShieldAlert, ShieldCheck } from "lucide-react";
import { formatPercent } from "../lib/format";
import type { StatsResponse } from "../types";
import { StatCard } from "./StatCard";

interface StatsRowProps {
  stats: Partial<StatsResponse>;
}

export function StatsRow({ stats }: StatsRowProps) {
  return (
    <div className="stats-row">
      <StatCard icon={Database} label="총 분석 건수" value={stats.total_analyses ?? 0} />
      <StatCard icon={ShieldAlert} label="가짜 음성" value={stats.fake_count ?? 0} />
      <StatCard icon={ShieldCheck} label="실제 음성" value={stats.real_count ?? 0} />
      <StatCard icon={Activity} label="평균 신뢰도" value={formatPercent(stats.avg_confidence)} />
    </div>
  );
}
