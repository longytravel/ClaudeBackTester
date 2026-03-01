import type { CandidateReport } from "../../types/api";

interface Props {
  candidate: CandidateReport;
}

export function WalkForwardBars({ candidate }: Props) {
  const meanSharpe = candidate.wf_mean_sharpe ?? 0;
  const passRate = candidate.wf_pass_rate ?? 0;
  const passCount = candidate.wf_pass_rate != null
    ? `~${Math.round(passRate * 5)}/5`
    : "---";

  const passColor = passRate >= 0.6 ? "#22c55e" : passRate >= 0.4 ? "#eab308" : "#ef4444";
  const sharpeColor = meanSharpe >= 0.3 ? "#22c55e" : meanSharpe >= 0 ? "#eab308" : "#ef4444";

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Walk-Forward Summary
      </h3>
      <p className="text-xs text-gray-500 mb-4">
        Per-window results not sent by backend &mdash; showing aggregate stats only.
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800/60 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-500 mb-1">Pass Rate</p>
          <p className="text-2xl font-bold font-mono" style={{ color: passColor }}>
            {(passRate * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">{passCount} windows</p>
        </div>
        <div className="bg-gray-800/60 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-500 mb-1">Mean OOS Sharpe</p>
          <p className="text-2xl font-bold font-mono" style={{ color: sharpeColor }}>
            {meanSharpe.toFixed(3)}
          </p>
          <p className="text-xs text-gray-500 mt-1">threshold: 0.30</p>
        </div>
      </div>
    </div>
  );
}
