import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  type ChartOptions,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

interface LineOptionsConfig {
  showLegend: boolean;
  hideX?: boolean;
  maxTicksLimit?: number;
}

export function lineOptions({
  showLegend,
  hideX = false,
  maxTicksLimit,
}: LineOptionsConfig): ChartOptions<"line"> {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        display: showLegend,
        position: "right",
        labels: { color: "#6b7280", font: { size: 9 }, boxWidth: 10, padding: 6 },
      },
      tooltip: { intersect: false, mode: "index" },
    },
    scales: {
      x: {
        display: !hideX,
        ticks: { color: "#8a9099", maxTicksLimit, font: { size: 10 } },
        grid: { color: "#eceff3" },
      },
      y: {
        ticks: { color: "#8a9099", font: { size: 10 } },
        grid: { color: "#eceff3" },
      },
    },
  };
}
