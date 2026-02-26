import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";
import { useRunStore } from "../../stores/useRunStore";
import { qualityColor } from "../../utils/formatters";

export function QualityScatter() {
  const batchHistory = useRunStore((s) => s.batchHistory);

  const data = batchHistory.map((b, i) => ({
    index: i,
    quality: b.batch_best_quality,
    stage: b.stage_name,
    phase: b.phase,
  }));

  if (data.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 h-72 flex items-center justify-center text-gray-500 text-sm">
        Waiting for batch data...
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-1">
        Search Landscape
      </h3>
      <p className="text-xs text-gray-500 mb-3">
        Each dot is a batch of parameter combos tested. Green dots = good
        results found, red = poor. Like MT5's optimization graph — you want to
        see dots climbing higher and turning green.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="index"
            name="Batch"
            stroke="#4b5563"
            fontSize={11}
            label={{
              value: "Batch #",
              position: "insideBottom",
              offset: -2,
              fill: "#6b7280",
              fontSize: 10,
            }}
          />
          <YAxis
            dataKey="quality"
            name="Quality"
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
            formatter={(value, name) => [
              Number(value).toFixed(2),
              String(name),
            ]}
          />
          <Scatter data={data}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={qualityColor(entry.quality)}
                fillOpacity={0.8}
                r={5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
