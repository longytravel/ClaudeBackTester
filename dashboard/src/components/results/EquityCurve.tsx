import { useEffect, useRef } from "react";
import { createChart, ColorType, LineSeries, type IChartApi, type Time } from "lightweight-charts";
import type { CandidateReport } from "../../types/api";

interface Props {
  candidate: CandidateReport;
  splitTimestamp?: number;
  pair?: string;
}

// Account settings — change here to adjust equity curve display
const STARTING_CAPITAL = 3000;  // £3,000
const LOT_SIZE = 0.01;          // Micro lot
const ACCOUNT_CURRENCY = "GBP";
const ACCOUNT_RATE = 1.27;      // GBP/USD
const CURRENCY_SYMBOL = "£";

function getPipValuePerStdLot(pair: string): number {
  if (pair.includes("JPY")) return 1000 / 150;
  if (pair.includes("XAU")) return 1;
  return 10;
}

export function EquityCurve({ candidate, splitTimestamp, pair = "EUR/USD" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const pipValue = getPipValuePerStdLot(pair);
  const pipsToCurrency = (LOT_SIZE * pipValue) / ACCOUNT_RATE;

  useEffect(() => {
    if (!containerRef.current || containerRef.current.clientWidth === 0) return;

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
      height: 500,
      autoSize: true,
      rightPriceScale: {
        borderColor: "#374151",
        scaleMargins: { top: 0.05, bottom: 0.05 },
      },
      timeScale: {
        borderColor: "#374151",
        timeVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      localization: {
        priceFormatter: (price: number) =>
          CURRENCY_SYMBOL + price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
      },
    });
    chartRef.current = chart;

    const totalPnl = candidate.trade_stats?.total_pnl_pips ?? 0;
    const lineColor = totalPnl >= 0 ? "#22c55e" : "#ef4444";

    const eqCurve = candidate.equity_curve;
    let allPoints: { time: Time; value: number }[] = [];

    if (eqCurve && eqCurve.length > 1) {
      allPoints = eqCurve
        .filter((pt) => pt.timestamp != null && pt.equity != null && isFinite(pt.equity))
        .map((pt) => ({
          time: pt.timestamp as Time,
          value: STARTING_CAPITAL + pt.equity * pipsToCurrency,
        }))
        .sort((a, b) => (a.time as number) - (b.time as number));
      allPoints = allPoints.filter(
        (pt, i, arr) => i === arr.length - 1 || pt.time !== arr[i + 1].time
      );
      if (allPoints.length > 2000) {
        const step = (allPoints.length - 1) / 1999;
        const sampled: typeof allPoints = [allPoints[0]];
        for (let i = 1; i < 1999; i++) {
          sampled.push(allPoints[Math.round(i * step)]);
        }
        sampled.push(allPoints[allPoints.length - 1]);
        allPoints = sampled;
      }
    } else {
      const stats = candidate.trade_stats;
      if (stats && stats.n_trades > 0) {
        const avgPnl = stats.avg_pnl_pips;
        let cumulative = 0;
        for (let i = 0; i <= stats.n_trades; i++) {
          allPoints.push({
            time: (1640000000 + i * 86400) as Time,
            value: STARTING_CAPITAL + cumulative * pipsToCurrency,
          });
          cumulative += avgPnl;
        }
      }
    }

    if (splitTimestamp && allPoints.length > 1) {
      const backPoints: { time: Time; value: number }[] = [];
      const forwardPoints: { time: Time; value: number }[] = [];

      for (const pt of allPoints) {
        if ((pt.time as number) <= splitTimestamp) {
          backPoints.push(pt);
        } else {
          forwardPoints.push(pt);
        }
      }

      if (backPoints.length > 0 && forwardPoints.length > 0) {
        forwardPoints.unshift(backPoints[backPoints.length - 1]);
      }

      if (backPoints.length > 1) {
        const backSeries = chart.addSeries(LineSeries, {
          color: lineColor,
          lineWidth: 2,
          title: "Back-test",
        });
        backSeries.setData(backPoints);
      }

      if (forwardPoints.length > 1) {
        const forwardSeries = chart.addSeries(LineSeries, {
          color: "#f59e0b",
          lineWidth: 2,
          lineStyle: 2,
          title: "Forward-test",
        });
        forwardSeries.setData(forwardPoints);
      }
    } else {
      const lineSeries = chart.addSeries(LineSeries, {
        color: lineColor,
        lineWidth: 2,
      });
      if (allPoints.length > 0) {
        lineSeries.setData(allPoints);
      }
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
        chart.timeScale().fitContent();
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [candidate, splitTimestamp, pipsToCurrency]);

  const hasRealData = candidate.equity_curve && candidate.equity_curve.length > 1;

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">
          Equity Curve{hasRealData ? "" : " (Estimated)"}
          <span className="text-xs font-normal text-gray-500 ml-2">
            {CURRENCY_SYMBOL}{STARTING_CAPITAL.toLocaleString()} start · {LOT_SIZE} lot · {ACCOUNT_CURRENCY}
          </span>
        </h3>
        {splitTimestamp && hasRealData && (
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-0.5 bg-green-500 rounded" />
              <span className="text-gray-400">Back-test</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span
                className="inline-block w-4 h-0.5 rounded"
                style={{
                  background: "repeating-linear-gradient(90deg, #f59e0b 0px, #f59e0b 3px, transparent 3px, transparent 6px)",
                }}
              />
              <span className="text-gray-400">Forward-test</span>
            </div>
          </div>
        )}
      </div>
      <div ref={containerRef} style={{ minHeight: 500 }} />
    </div>
  );
}
