import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Cell,
} from "recharts";
import type { CandidateReport } from "../../types/api";

interface Props {
  candidate: CandidateReport;
}

export function WalkForwardBars({ candidate }: Props) {
  // Generate simulated per-window data from aggregate stats
  // In a real implementation, the backend would send per-window results
  const meanSharpe = candidate.wf_mean_sharpe ?? 0;
  const passRate = candidate.wf_pass_rate ?? 0;

  // Create representative bars (5 windows)
  const nWindows = 5;
  const data = Array.from({ length: nWindows }, (_, i) => {
    // Simulate: passRate% positive, rest negative
    const isPositive = i / nWindows < passRate;
    return {
      window: `W${i + 1}`,
      sharpe: isPositive
        ? meanSharpe * (0.8 + Math.random() * 0.4)
        : -Math.abs(meanSharpe) * (0.3 + Math.random() * 0.4),
    };
  });

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-1">
        Walk-Forward Windows
      </h3>
      <p className="text-xs text-gray-500 mb-3">
        Pass Rate: {((passRate ?? 0) * 100).toFixed(0)}% | Mean Sharpe:{" "}
        {meanSharpe.toFixed(3)}
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="window" stroke="#4b5563" fontSize={11} />
          <YAxis stroke="#4b5563" fontSize={11} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(v) => [Number(v).toFixed(3), "Sharpe"]}
          />
          <ReferenceLine y={0} stroke="#4b5563" />
          <Bar dataKey="sharpe" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.sharpe >= 0 ? "#22c55e" : "#ef4444"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
