export function formatNumber(n: number, decimals = 2): string {
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(decimals);
}

export function formatDuration(secs: number): string {
  if (secs < 60) return `${secs.toFixed(0)}s`;
  if (secs < 3600)
    return `${Math.floor(secs / 60)}m ${Math.floor(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

export function qualityColor(q: number): string {
  if (q >= 50) return "#22c55e"; // green-500
  if (q >= 30) return "#eab308"; // yellow-500
  if (q >= 15) return "#f97316"; // orange-500
  return "#ef4444"; // red-500
}

export function ratingColor(rating: string): string {
  switch (rating) {
    case "GREEN":
      return "#22c55e";
    case "YELLOW":
      return "#eab308";
    case "RED":
      return "#ef4444";
    default:
      return "#6b7280";
  }
}

export function percentString(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}
