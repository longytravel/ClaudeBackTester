import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { CandidateReport } from "../../types/api";

interface Props {
  candidate: CandidateReport;
}

export function RegimeRadar({ candidate }: Props) {
  const dist = candidate.regime_distribution;
  const robustness = candidate.regime_robustness_score;

  if (!dist || Object.keys(dist).length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex items-center justify-center text-gray-500 text-sm h-64">
        No regime data available
      </div>
    );
  }

  const data = Object.entries(dist).map(([name, value]) => ({
    regime: name.replace(/_/g, " "),
    value: typeof value === "number" ? value : 0,
    fullMark: 100,
  }));

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-1">
        Regime Distribution
      </h3>
      {robustness !== undefined && (
        <p className="text-xs text-gray-500 mb-3">
          Robustness Score: {robustness.toFixed(2)}
        </p>
      )}
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={data}>
          <PolarGrid stroke="#1f2937" />
          <PolarAngleAxis
            dataKey="regime"
            tick={{ fill: "#9ca3af", fontSize: 10 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: "#4b5563", fontSize: 9 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />
          <Radar
            name="Distribution"
            dataKey="value"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.3}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
