import clsx from "clsx";
import { useRunStore } from "../../stores/useRunStore";

export function SamplerInfo() {
  const currentPhase = useRunStore((s) => s.currentPhase);
  const entropy = useRunStore((s) => s.entropy);
  const effectiveLr = useRunStore((s) => s.effectiveLr);
  const batchHistory = useRunStore((s) => s.batchHistory);

  const edaUpdates = batchHistory.filter(
    (b) => b.phase === "exploitation",
  ).length;

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-1">
        Search Strategy
      </h3>
      <p className="text-xs text-gray-500 mb-3">
        The optimizer starts by exploring randomly, then narrows down to
        promising areas. The entropy bar shows how focused the search is.
      </p>
      <div className="space-y-4">
        {/* Phase badge */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Current Mode</span>
          <span
            className={clsx(
              "px-2 py-0.5 rounded text-xs font-semibold",
              currentPhase === "exploration"
                ? "bg-blue-500/20 text-blue-400"
                : currentPhase === "exploitation"
                  ? "bg-amber-500/20 text-amber-400"
                  : "bg-gray-700 text-gray-400",
            )}
          >
            {currentPhase === "exploration"
              ? "Exploring (trying random areas)"
              : currentPhase === "exploitation"
                ? "Narrowing down (found promising area)"
                : "---"}
          </span>
        </div>

        {/* Entropy gauge */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-500">Search Focus</span>
            <span className="text-xs text-gray-400">
              {entropy !== null
                ? entropy > 0.7
                  ? "Wide search"
                  : entropy > 0.4
                    ? "Narrowing"
                    : "Highly focused"
                : "---"}
            </span>
          </div>
          <div className="h-2.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${entropy !== null ? entropy * 100 : 0}%`,
                backgroundColor:
                  entropy !== null
                    ? entropy > 0.7
                      ? "#22c55e"
                      : entropy > 0.4
                        ? "#eab308"
                        : "#ef4444"
                    : "#374151",
              }}
            />
          </div>
        </div>

        {/* Learning Rate */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Learning Rate</span>
          <span className="text-sm font-mono text-gray-300">
            {effectiveLr !== null ? effectiveLr.toFixed(4) : "---"}
          </span>
        </div>

        {/* EDA Updates */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Model Updates</span>
          <span className="text-sm font-mono text-gray-300">{edaUpdates}</span>
        </div>
      </div>
    </div>
  );
}
