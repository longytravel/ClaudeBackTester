export function RunHistory() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-200 mb-4">Run History</h1>
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-8 text-center">
        <svg
          className="w-12 h-12 mx-auto text-gray-600 mb-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <p className="text-gray-400 text-sm">
          Coming soon &mdash; browse past optimization runs
        </p>
      </div>
    </div>
  );
}
