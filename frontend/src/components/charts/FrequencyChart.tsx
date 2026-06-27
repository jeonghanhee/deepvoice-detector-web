import { useMemo } from "react";
import { Line } from "react-chartjs-2";
import { lineOptions } from "../../lib/chart";

interface FrequencyChartProps {
  data: number[];
  isFake: boolean;
}

export function FrequencyChart({ data, isFake }: FrequencyChartProps) {
  const color = isFake ? "#e03e3e" : "#16a34a";
  const chartData = useMemo(
    () => ({
      labels: data.map((_, index) => `${Math.round((index * 8000) / Math.max(data.length, 1))}Hz`),
      datasets: [
        {
          data,
          borderColor: color,
          backgroundColor: isFake ? "rgba(224, 62, 62, 0.08)" : "rgba(22, 163, 74, 0.08)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.36,
        },
      ],
    }),
    [color, data, isFake],
  );

  return <Line data={chartData} options={lineOptions({ showLegend: false, maxTicksLimit: 8 })} />;
}
