import { useRunStore } from "../../stores/useRunStore";
import {
  formatNumber,
  formatDuration,
  percentString,
} from "../../utils/formatters";

function StatCard({
  label,
  value,
  hint,
  color,
}: {
  label: string;
  value: string;
  hint?: string;
  color?: string;
}) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div
        className="text-lg font-semibold font-mono"
        style={{ color: color || "#f3f4f6" }}
      >
        {value}
      </div>
      {hint && <div className="text-[10px] text-gray-600 mt-0.5">{hint}</div>}
    </div>
  );
}

export function LiveStats() {
  const evalsPerSecond = useRunStore((s) => s.evalsPerSecond);
  const totalEvaluated = useRunStore((s) => s.totalEvaluated);
  const validRate = useRunStore((s) => s.validRate);
  const bestQuality = useRunStore((s) => s.bestQuality);
  const bestSharpe = useRunStore((s) => s.bestSharpe);
  const bestTrades = useRunStore((s) => s.bestTrades);
  const elapsedSecs = useRunStore((s) => s.elapsedSecs);
  const batchHistory = useRunStore((s) => s.batchHistory);

  // Rough ETA: remaining trials / evals per sec
  const latestBatch =
    batchHistory.length > 0 ? batchHistory[batchHistory.length - 1] : null;
  const remaining = latestBatch
    ? latestBatch.trials_total - latestBatch.trials_done
    : 0;
  const eta =
    evalsPerSecond > 0 && remaining > 0 ? remaining / evalsPerSecond : 0;

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-1">
        Live Numbers
      </h3>
      <p className="text-xs text-gray-500 mb-3">
        Best result found so far across all parameter combos tested.
      </p>
      <div className="grid grid-cols-2 gap-2">
        <StatCard
          label="Best Quality"
          value={bestQuality < 1 ? bestQuality.toFixed(3) : bestQuality.toFixed(1)}
          hint="Sharpe x R² x consistency"
          color={
            bestQuality >= 50
              ? "#22c55e"
              : bestQuality >= 20
                ? "#eab308"
                : "#ef4444"
          }
        />
        <StatCard
          label="Best Sharpe Ratio"
          value={bestSharpe.toFixed(2)}
          hint="> 1.0 is good, > 2.0 is great"
          color={
            bestSharpe > 1 ? "#22c55e" : bestSharpe > 0 ? "#eab308" : "#ef4444"
          }
        />
        <StatCard
          label="Best Trade Count"
          value={formatNumber(bestTrades, 0)}
          hint="More trades = more reliable"
        />
        <StatCard
          label="Valid Rate"
          value={percentString(validRate)}
          hint="% of combos that made trades"
          color={
            validRate > 0.05
              ? "#22c55e"
              : validRate > 0.01
                ? "#eab308"
                : "#ef4444"
          }
        />
        <StatCard
          label="Speed"
          value={`${formatNumber(evalsPerSecond, 0)}/s`}
          hint={`${formatNumber(totalEvaluated, 0)} tested so far`}
          color="#3b82f6"
        />
        <StatCard
          label="Elapsed"
          value={formatDuration(elapsedSecs)}
          hint={eta > 0 ? `~${formatDuration(eta)} remaining` : ""}
        />
      </div>
    </div>
  );
}
