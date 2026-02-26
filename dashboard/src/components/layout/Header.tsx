import clsx from "clsx";
import { useRunStore } from "../../stores/useRunStore";

function StatusBadge({ status }: { status: string }) {
  const base =
    "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide";

  switch (status) {
    case "running":
      return (
        <span className={clsx(base, "bg-blue-500/20 text-blue-400")}>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
          </span>
          Running
        </span>
      );
    case "complete":
      return (
        <span className={clsx(base, "bg-green-500/20 text-green-400")}>
          Complete
        </span>
      );
    case "connecting":
      return (
        <span className={clsx(base, "bg-yellow-500/20 text-yellow-400")}>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-yellow-500" />
          </span>
          Connecting
        </span>
      );
    default:
      return (
        <span className={clsx(base, "bg-gray-500/20 text-gray-400")}>
          Idle
        </span>
      );
  }
}

export function Header() {
  const status = useRunStore((s) => s.status);
  const strategy = useRunStore((s) => s.strategy);
  const pair = useRunStore((s) => s.pair);
  const timeframe = useRunStore((s) => s.timeframe);
  const preset = useRunStore((s) => s.preset);

  return (
    <header className="h-14 flex items-center justify-between px-6 border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-10">
      <div className="flex items-center gap-3">
        {strategy ? (
          <>
            <span className="text-sm font-semibold text-gray-100">
              {strategy}
            </span>
            <span className="text-gray-600">|</span>
            <span className="text-sm text-gray-400">{pair}</span>
            <span className="text-gray-600">|</span>
            <span className="text-sm text-gray-400">{timeframe}</span>
            <span className="text-gray-600">|</span>
            <span className="text-sm text-gray-400 capitalize">{preset}</span>
          </>
        ) : (
          <span className="text-sm text-gray-500">
            Waiting for optimization run...
          </span>
        )}
      </div>
      <StatusBadge status={status} />
    </header>
  );
}
