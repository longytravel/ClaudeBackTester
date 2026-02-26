import { useEffect, useRef } from "react";
import { createChart, ColorType, LineSeries, type IChartApi, type Time } from "lightweight-charts";
import type { CandidateReport } from "../../types/api";

interface Props {
  candidate: CandidateReport;
}

export function EquityCurve({ candidate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#1e1e2e" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      width: containerRef.current.clientWidth,
      height: 300,
      rightPriceScale: {
        borderColor: "#374151",
      },
      timeScale: {
        borderColor: "#374151",
        timeVisible: false,
      },
    });
    chartRef.current = chart;

    const lineSeries = chart.addSeries(LineSeries, {
      color: candidate.trade_stats && candidate.trade_stats.total_pnl_pips >= 0
        ? "#22c55e"
        : "#ef4444",
      lineWidth: 2,
    });

    // Build equity data from trade stats
    // Since we don't have per-trade equity, create a simple linear representation
    const stats = candidate.trade_stats;
    if (stats && stats.n_trades > 0) {
      const avgPnl = stats.avg_pnl_pips;
      const points: { time: Time; value: number }[] = [];
      let cumulative = 0;
      for (let i = 0; i <= stats.n_trades; i++) {
        // Use sequential day-based time values for lightweight-charts (UTCTimestamp)
        points.push({
          time: (1640000000 + i * 86400) as Time,
          value: cumulative,
        });
        cumulative += avgPnl;
      }
      lineSeries.setData(points);
    }

    // Handle resize
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [candidate]);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Equity Curve (Estimated)
      </h3>
      <div ref={containerRef} />
    </div>
  );
}
