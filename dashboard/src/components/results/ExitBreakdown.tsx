import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { TradeStats } from "../../types/api";

interface Props {
  tradeStats: TradeStats;
}

const COLORS = [
  "#3b82f6",
  "#22c55e",
  "#eab308",
  "#ef4444",
  "#a855f7",
  "#f97316",
  "#06b6d4",
  "#ec4899",
];

export function ExitBreakdown({ tradeStats }: Props) {
  const data = Object.entries(tradeStats.exit_breakdown).map(
    ([name, info]) => ({
      name,
      value: info.count,
      pct: info.pct,
      pnl: info.pnl_pips,
    }),
  );

  if (data.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 h-64 flex items-center justify-center text-gray-500 text-sm">
        No exit data available
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Exit Breakdown
      </h3>
      <ResponsiveContainer width="100%" height={240}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            dataKey="value"
            nameKey="name"
            paddingAngle={2}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(_value, _name, props) => {
              const d = props.payload as { name: string; value: number; pct: number; pnl: number };
              return [
                `${d.value} trades (${(d.pct * 100).toFixed(1)}%) | ${d.pnl.toFixed(1)} pips`,
                d.name,
              ];
            }}
          />
          <Legend
            verticalAlign="bottom"
            iconSize={8}
            formatter={(value: string) => (
              <span className="text-xs text-gray-400">{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
