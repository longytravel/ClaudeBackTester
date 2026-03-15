import type { PipelineUpdate, OptimizerFunnel } from "../../types/api";
import { formatNumber } from "../../utils/formatters";

interface Props {
  stages: PipelineUpdate[];
  optimizerFunnel?: OptimizerFunnel;
}

interface FunnelRow {
  label: string;
  count: number;
  total: number;
  section: "optimizer" | "pipeline";
  detail?: string;
}

export function PipelineFunnel({ stages, optimizerFunnel }: Props) {
  if (stages.length === 0 && !optimizerFunnel) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex items-center justify-center text-gray-500 text-sm h-48">
        No pipeline data
      </div>
    );
  }

  // Build unified funnel rows
  const rows: FunnelRow[] = [];

  // Optimizer funnel rows (prepended)
  if (optimizerFunnel) {
    rows.push({
      label: "Total Trials",
      count: optimizerFunnel.total_trials,
      total: optimizerFunnel.total_trials,
      section: "optimizer",
    });
    rows.push({
      label: "Refinement Passing",
      count: optimizerFunnel.refinement_passing,
      total: optimizerFunnel.total_trials,
      section: "optimizer",
    });
    if (optimizerFunnel.dsr_surviving != null) {
      rows.push({
        label: "DSR Prefilter",
        count: optimizerFunnel.dsr_surviving,
        total: optimizerFunnel.refinement_passing || 1,
        section: "optimizer",
      });
    }
    if (optimizerFunnel.dedup_groups != null) {
      rows.push({
        label: "Unique Strategy Groups",
        count: optimizerFunnel.dedup_groups,
        total: optimizerFunnel.dsr_surviving || 1,
        section: "optimizer",
      });
    }
    if (optimizerFunnel.after_dedup != null) {
      rows.push({
        label: "After Dedup",
        count: optimizerFunnel.after_dedup,
        total: optimizerFunnel.dsr_surviving || 1,
        section: "optimizer",
      });
    }
    rows.push({
      label: "Pipeline Candidates",
      count: optimizerFunnel.pipeline_candidates ?? optimizerFunnel.sent_to_pipeline,
      total: (optimizerFunnel.after_dedup ?? optimizerFunnel.refinement_passing) || 1,
      section: "optimizer",
    });
  }

  // Deduplicate pipeline stages: keep latest update per stage_name
  const stageMap = new Map<string, PipelineUpdate>();
  for (const s of stages) {
    stageMap.set(s.stage_name, s);
  }
  const uniqueStages = Array.from(stageMap.values());

  for (const stage of uniqueStages) {
    rows.push({
      label: stage.stage_name,
      count: stage.candidates_surviving,
      total: stage.candidates_total,
      section: "pipeline",
      detail: stage.detail,
    });
  }

  // For the bar width, use the maximum count across all rows
  const maxCount = Math.max(...rows.map((r) => Math.max(r.count, r.total)), 1);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Pipeline Funnel
      </h3>
      <div className="space-y-2">
        {rows.map((row, i) => {
          // Show section header when transitioning from optimizer to pipeline
          const showOptimizerHeader =
            row.section === "optimizer" && (i === 0 || rows[i - 1].section !== "optimizer");
          const showPipelineHeader =
            row.section === "pipeline" && (i === 0 || rows[i - 1].section !== "pipeline");

          const widthPct = (row.count / maxCount) * 100;
          const isZero = row.count === 0;

          // Color logic
          let barColor: string;
          if (isZero) {
            barColor = "#ef4444"; // red
          } else if (row.section === "optimizer") {
            barColor = "#6366f1"; // indigo for optimizer phase
          } else {
            // Pipeline phase: gradient from blue to green
            const pipelineRows = rows.filter((r) => r.section === "pipeline");
            const pipelineIdx = pipelineRows.indexOf(row);
            barColor = pipelineIdx < pipelineRows.length / 2 ? "#3b82f6" : "#22c55e";
          }

          return (
            <div key={`${row.section}-${i}`}>
              {showOptimizerHeader && (
                <div className="text-[10px] uppercase tracking-widest text-indigo-400 font-semibold mb-1 mt-1">
                  Optimizer Phase
                </div>
              )}
              {showPipelineHeader && (
                <div className="text-[10px] uppercase tracking-widest text-blue-400 font-semibold mb-1 mt-3 pt-2 border-t border-gray-800">
                  Validation Phase
                </div>
              )}
              <div className="flex items-center justify-between text-xs mb-1">
                <span className={row.section === "optimizer" ? "text-gray-400" : "text-gray-400"}>
                  {row.label}
                </span>
                <span className={isZero ? "text-red-400" : "text-green-400"}>
                  {row.section === "optimizer" && row.label === "Total Trials"
                    ? formatNumber(row.count, 0)
                    : `${formatNumber(row.count, 0)} / ${formatNumber(row.total, 0)}`}
                </span>
              </div>
              <div className="h-6 bg-gray-800 rounded overflow-hidden">
                <div
                  className="h-full rounded transition-all duration-500"
                  style={{
                    width: `${Math.max(widthPct, 2)}%`,
                    backgroundColor: barColor,
                    opacity: row.section === "optimizer" ? 0.85 : 1,
                  }}
                />
              </div>
              {row.detail && (
                <p className="text-xs text-gray-500 mt-0.5">{row.detail}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
