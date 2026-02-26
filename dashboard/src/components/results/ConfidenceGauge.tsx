import { ratingColor } from "../../utils/formatters";

interface Props {
  compositeScore: number;
  rating: string;
  gatesPassed?: Record<string, boolean>;
}

export function ConfidenceGauge({ compositeScore, rating, gatesPassed }: Props) {
  const color = ratingColor(rating);
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(compositeScore / 100, 1);
  const strokeDashoffset = circumference * (1 - progress);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex flex-col items-center">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Confidence Score
      </h3>

      {/* SVG Gauge */}
      <svg width="160" height="160" viewBox="0 0 160 160">
        {/* Background ring */}
        <circle
          cx="80"
          cy="80"
          r={radius}
          fill="none"
          stroke="#1f2937"
          strokeWidth="10"
        />
        {/* Progress ring */}
        <circle
          cx="80"
          cy="80"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          transform="rotate(-90 80 80)"
          style={{ transition: "stroke-dashoffset 0.5s ease" }}
        />
        {/* Center text */}
        <text
          x="80"
          y="74"
          textAnchor="middle"
          fill={color}
          fontSize="28"
          fontWeight="bold"
          fontFamily="monospace"
        >
          {compositeScore.toFixed(0)}
        </text>
        <text
          x="80"
          y="96"
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="12"
        >
          {rating}
        </text>
      </svg>

      {/* Gates list */}
      {gatesPassed && Object.keys(gatesPassed).length > 0 && (
        <div className="mt-3 w-full space-y-1">
          {Object.entries(gatesPassed).map(([gate, passed]) => (
            <div
              key={gate}
              className="flex items-center justify-between text-xs"
            >
              <span className="text-gray-400">{gate}</span>
              <span className={passed ? "text-green-400" : "text-red-400"}>
                {passed ? "PASS" : "FAIL"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
