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

    const totalPnl = candidate.trade_stats?.total_pnl_pips ?? 0;
    const lineSeries = chart.addSeries(LineSeries, {
      color: totalPnl >= 0 ? "#22c55e" : "#ef4444",
      lineWidth: 2,
    });

    // Use real equity curve data if available, otherwise fall back to synthetic
    const eqCurve = candidate.equity_curve;
    if (eqCurve && eqCurve.length > 1) {
      const points: { time: Time; value: number }[] = eqCurve
        .map((pt) => ({
          time: pt.timestamp as Time,
          value: pt.equity,
        }))
        .sort((a, b) => (a.time as number) - (b.time as number));
      // Deduplicate: keep only the last point at each timestamp
      // (equity is cumulative, so the last trade at a bar has the correct running total)
      const deduped = points.filter(
        (pt, i, arr) => i === arr.length - 1 || pt.time !== arr[i + 1].time
      );
      lineSeries.setData(deduped);
    } else {
      // Fallback: synthetic linear representation from trade stats
      const stats = candidate.trade_stats;
      if (stats && stats.n_trades > 0) {
        const avgPnl = stats.avg_pnl_pips;
        const points: { time: Time; value: number }[] = [];
        let cumulative = 0;
        for (let i = 0; i <= stats.n_trades; i++) {
          points.push({
            time: (1640000000 + i * 86400) as Time,
            value: cumulative,
          });
          cumulative += avgPnl;
        }
        lineSeries.setData(points);
      }
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

  const hasRealData = candidate.equity_curve && candidate.equity_curve.length > 1;

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Equity Curve{hasRealData ? "" : " (Estimated)"}
      </h3>
      <div ref={containerRef} />
    </div>
  );
}
