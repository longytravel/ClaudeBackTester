import type { CandidateReport } from "../../types/api";

interface Props {
  candidate: CandidateReport;
}

function MCStatCard({
  label,
  value,
  good,
}: {
  label: string;
  value: string;
  good: boolean | null;
}) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div
        className="text-lg font-semibold font-mono"
        style={{
          color:
            good === null ? "#9ca3af" : good ? "#22c55e" : "#ef4444",
        }}
      >
        {value}
      </div>
    </div>
  );
}

export function MonteCarloHist({ candidate }: Props) {
  const dsr = candidate.dsr;
  const permP = candidate.permutation_p;
  const cpcvSharpe = candidate.cpcv_mean_sharpe;
  const cpcvPct = candidate.cpcv_pct_positive;

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Monte Carlo & Statistical Tests
      </h3>
      <div className="grid grid-cols-2 gap-2">
        <MCStatCard
          label="Deflated Sharpe (DSR)"
          value={dsr !== undefined ? dsr.toFixed(3) : "---"}
          good={dsr !== undefined ? dsr >= 0.95 : null}
        />
        <MCStatCard
          label="Permutation p-value"
          value={permP !== undefined ? permP.toFixed(3) : "---"}
          good={permP !== undefined ? permP <= 0.05 : null}
        />
        <MCStatCard
          label="CPCV Mean Sharpe"
          value={cpcvSharpe !== undefined ? cpcvSharpe.toFixed(3) : "---"}
          good={cpcvSharpe !== undefined ? cpcvSharpe > 0 : null}
        />
        <MCStatCard
          label="CPCV % Positive"
          value={
            cpcvPct !== undefined ? `${(cpcvPct * 100).toFixed(0)}%` : "---"
          }
          good={cpcvPct !== undefined ? cpcvPct > 0.5 : null}
        />
      </div>
    </div>
  );
}
