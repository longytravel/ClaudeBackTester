import clsx from "clsx";
import { useRunStore, type StageInfo } from "../../stores/useRunStore";

function StagePill({ stage, isLast }: { stage: StageInfo; isLast: boolean }) {
  const batchHistory = useRunStore((s) => s.batchHistory);

  // Find latest batch for this stage to get progress
  const latestBatch = [...batchHistory]
    .reverse()
    .find((b) => b.stage_name === stage.name);
  const progress = latestBatch
    ? latestBatch.trials_done / latestBatch.trials_total
    : 0;

  return (
    <div className="flex items-center">
      <div
        className={clsx(
          "relative flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all min-w-[120px] justify-center",
          stage.status === "complete" && "bg-green-500/20 text-green-400 border border-green-500/30",
          stage.status === "active" && "bg-blue-500/20 text-blue-400 border border-blue-500/30",
          stage.status === "pending" && "bg-gray-800 text-gray-500 border border-gray-700",
        )}
      >
        {stage.status === "complete" && (
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        )}
        {stage.status === "active" && (
          <svg
            className="w-3.5 h-3.5 animate-spin"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        )}
        <span className="capitalize">{stage.name}</span>

        {/* Progress bar for active stage */}
        {stage.status === "active" && progress > 0 && (
          <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-b-lg overflow-hidden">
            <div
              className="h-full bg-blue-400 transition-all duration-300"
              style={{ width: `${Math.min(progress * 100, 100)}%` }}
            />
          </div>
        )}
      </div>

      {/* Arrow connector */}
      {!isLast && (
        <svg
          className="w-6 h-6 text-gray-600 mx-1 flex-shrink-0"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 5l7 7-7 7"
          />
        </svg>
      )}
    </div>
  );
}

export function StageProgress() {
  const stages = useRunStore((s) => s.stages);
  const runStatus = useRunStore((s) => s.status);

  if (stages.length === 0) return null;

  // When run is complete, override all stage statuses to "complete"
  // in case a stage_complete message was missed
  const displayStages =
    runStatus === "complete"
      ? stages.map((s) => ({ ...s, status: "complete" as const }))
      : stages;

  return (
    <div className="px-6 py-4">
      <div className="flex items-center flex-wrap gap-y-2">
        {displayStages.map((stage, i) => (
          <StagePill
            key={stage.name}
            stage={stage}
            isLast={i === displayStages.length - 1}
          />
        ))}
      </div>
    </div>
  );
}
