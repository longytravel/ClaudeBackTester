import type { PipelineUpdate } from "../../types/api";

interface Props {
  stages: PipelineUpdate[];
}

export function PipelineFunnel({ stages }: Props) {
  if (stages.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex items-center justify-center text-gray-500 text-sm h-48">
        No pipeline data
      </div>
    );
  }

  // Deduplicate: keep latest update per stage_name
  const stageMap = new Map<string, PipelineUpdate>();
  for (const s of stages) {
    stageMap.set(s.stage_name, s);
  }
  const uniqueStages = Array.from(stageMap.values());

  const maxCandidates = Math.max(...uniqueStages.map((s) => s.candidates_total), 1);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Pipeline Funnel
      </h3>
      <div className="space-y-2">
        {uniqueStages.map((stage, i) => {
          const widthPct =
            (stage.candidates_surviving / maxCandidates) * 100;
          const allEliminated = stage.candidates_surviving === 0;

          return (
            <div key={i}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-400">{stage.stage_name}</span>
                <span className={allEliminated ? "text-red-400" : "text-green-400"}>
                  {stage.candidates_surviving} / {stage.candidates_total}
                </span>
              </div>
              <div className="h-6 bg-gray-800 rounded overflow-hidden">
                <div
                  className="h-full rounded transition-all duration-500"
                  style={{
                    width: `${Math.max(widthPct, 2)}%`,
                    backgroundColor: allEliminated
                      ? "#ef4444"
                      : i < uniqueStages.length / 2
                        ? "#3b82f6"
                        : "#22c55e",
                  }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-0.5">{stage.detail}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
