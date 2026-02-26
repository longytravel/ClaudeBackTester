import type { CandidateReport } from "../../types/api";
import { qualityColor } from "../../utils/formatters";

interface Props {
  candidate: CandidateReport;
}

export function ParamHeatmap({ candidate }: Props) {
  const params = candidate.params;
  const entries = Object.entries(params);

  if (entries.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex items-center justify-center text-gray-500 text-sm h-48">
        No parameter data
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Best Parameters
      </h3>
      <div className="grid grid-cols-2 gap-1">
        {entries.map(([key, value]) => (
          <div
            key={key}
            className="flex items-center justify-between px-2 py-1.5 rounded bg-gray-800/50 border border-gray-700/30"
          >
            <span className="text-xs text-gray-400 truncate mr-2">{key}</span>
            <span
              className="text-xs font-mono font-semibold"
              style={{ color: qualityColor(candidate.back_quality) }}
            >
              {typeof value === "number" ? value.toFixed(4) : String(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
