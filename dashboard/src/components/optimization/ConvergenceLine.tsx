import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { useRunStore } from "../../stores/useRunStore";
import { formatNumber } from "../../utils/formatters";

export function ConvergenceLine() {
  const batchHistory = useRunStore((s) => s.batchHistory);

  // Compute running max of best_quality over cumulative evaluations
  const data: { evals: number; quality: number }[] = [];
  let runningMax = 0;
  for (const batch of batchHistory) {
    if (batch.best_quality > runningMax) {
      runningMax = batch.best_quality;
    }
    data.push({
      evals: batch.trials_done,
      quality: runningMax,
    });
  }

  if (data.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 h-72 flex items-center justify-center text-gray-500 text-sm">
        Waiting for batch data...
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-gray-300">
          Best Quality Score Over Time
        </h3>
        <span className="text-lg font-bold font-mono text-green-400">
          {runningMax < 1 ? runningMax.toFixed(3) : runningMax.toFixed(1)}
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-3">
        Should climb as the optimizer finds better parameter combinations.
        Quality = Sharpe &times; R&sup2; &times; consistency factors.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="evals"
            tickFormatter={(v: number) => formatNumber(v, 0)}
            stroke="#4b5563"
            fontSize={11}
            label={{
              value: "Parameter combinations tested",
              position: "insideBottom",
              offset: -2,
              fill: "#6b7280",
              fontSize: 10,
            }}
          />
          <YAxis
            stroke="#4b5563"
            fontSize={11}
            label={{
              value: "Quality",
              angle: -90,
              position: "insideLeft",
              fill: "#6b7280",
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            labelFormatter={(v) =>
              `${formatNumber(Number(v), 0)} combos tested`
            }
            formatter={(v) => [Number(v).toFixed(2), "Best Quality"]}
          />
          {/* Reference line at quality=30 (decent threshold) */}
          <ReferenceLine
            y={30}
            stroke="#eab308"
            strokeDasharray="3 3"
            strokeOpacity={0.5}
          />
          <Line
            type="stepAfter"
            dataKey="quality"
            stroke="#22c55e"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#22c55e" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
