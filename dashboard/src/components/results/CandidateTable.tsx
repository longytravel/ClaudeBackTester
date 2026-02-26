import clsx from "clsx";
import type { CandidateReport } from "../../types/api";
import { ratingColor } from "../../utils/formatters";

interface Props {
  candidates: CandidateReport[];
  onSelect?: (index: number) => void;
  selectedIndex?: number;
}

export function CandidateTable({ candidates, onSelect, selectedIndex }: Props) {
  // Sort by composite score descending
  const sorted = [...candidates].sort(
    (a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0),
  );

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 overflow-x-auto">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Candidate Results
      </h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-gray-500 uppercase tracking-wider">
            <th className="text-left py-2 px-2">#</th>
            <th className="text-right py-2 px-2">Quality</th>
            <th className="text-right py-2 px-2">Sharpe</th>
            <th className="text-right py-2 px-2">Trades</th>
            <th className="text-right py-2 px-2">Fwd Quality</th>
            <th className="text-right py-2 px-2">FB Ratio</th>
            <th className="text-right py-2 px-2">DSR</th>
            <th className="text-center py-2 px-2">Rating</th>
            <th className="text-center py-2 px-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((c) => (
            <tr
              key={c.index}
              className={clsx(
                "border-l-2 cursor-pointer transition-colors hover:bg-gray-800/50",
                c.eliminated ? "border-l-red-500" : "border-l-green-500",
                selectedIndex === c.index && "bg-gray-800/70",
              )}
              onClick={() => onSelect?.(c.index)}
            >
              <td className="py-2 px-2 text-gray-400">{c.index + 1}</td>
              <td className="py-2 px-2 text-right font-mono">
                {c.back_quality.toFixed(1)}
              </td>
              <td className="py-2 px-2 text-right font-mono">
                {c.trade_stats?.profit_factor?.toFixed(2) ?? "---"}
              </td>
              <td className="py-2 px-2 text-right font-mono">
                {c.trade_stats?.n_trades ?? "---"}
              </td>
              <td className="py-2 px-2 text-right font-mono">
                {c.forward_quality.toFixed(1)}
              </td>
              <td className="py-2 px-2 text-right font-mono">
                {c.forward_back_ratio.toFixed(2)}
              </td>
              <td className="py-2 px-2 text-right font-mono">
                {c.dsr?.toFixed(3) ?? "---"}
              </td>
              <td className="py-2 px-2 text-center">
                {c.rating ? (
                  <span
                    className="px-2 py-0.5 rounded text-xs font-semibold"
                    style={{
                      backgroundColor: ratingColor(c.rating) + "20",
                      color: ratingColor(c.rating),
                    }}
                  >
                    {c.rating}
                  </span>
                ) : (
                  <span className="text-gray-500">---</span>
                )}
              </td>
              <td className="py-2 px-2 text-center">
                {c.eliminated ? (
                  <span className="text-xs text-red-400" title={c.elimination_reason}>
                    Eliminated
                  </span>
                ) : (
                  <span className="text-xs text-green-400">Survived</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
