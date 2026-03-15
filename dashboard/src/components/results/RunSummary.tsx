import type { Report, OptimizerFunnel } from "../../types/api";
import { formatNumber } from "../../utils/formatters";

interface Props {
  report: Report;
  optimizerFunnel?: OptimizerFunnel;
}

type Verdict = "RED" | "YELLOW" | "GREEN";

function determineVerdict(report: Report): { verdict: Verdict; survivors: number; highConfidence: number } {
  const survivors = report.candidates.filter((c) => !c.eliminated);
  const survivorCount = survivors.length;

  if (survivorCount === 0) {
    return { verdict: "RED", survivors: 0, highConfidence: 0 };
  }

  // Count candidates with high composite score (>= 60)
  const highConfidence = survivors.filter(
    (c) => (c.composite_score ?? 0) >= 60,
  ).length;

  if (highConfidence > 0) {
    return { verdict: "GREEN", survivors: survivorCount, highConfidence };
  }

  return { verdict: "YELLOW", survivors: survivorCount, highConfidence: 0 };
}

const verdictConfig: Record<
  Verdict,
  { bg: string; border: string; badge: string; badgeText: string; textColor: string }
> = {
  RED: {
    bg: "bg-red-950/30",
    border: "border-red-900/50",
    badge: "bg-red-900/60",
    badgeText: "text-red-400",
    textColor: "text-red-300",
  },
  YELLOW: {
    bg: "bg-amber-950/20",
    border: "border-amber-900/40",
    badge: "bg-amber-900/50",
    badgeText: "text-amber-400",
    textColor: "text-amber-300",
  },
  GREEN: {
    bg: "bg-green-950/20",
    border: "border-green-900/40",
    badge: "bg-green-900/50",
    badgeText: "text-green-400",
    textColor: "text-green-300",
  },
};

export function RunSummary({ report, optimizerFunnel }: Props) {
  const { verdict, survivors, highConfidence } = determineVerdict(report);
  const config = verdictConfig[verdict];
  const totalCandidates = report.candidates.length;

  // Build the narrative text
  const totalTrials = optimizerFunnel?.total_trials;
  const refinementPassing = optimizerFunnel?.refinement_passing;
  const dsrSurviving = optimizerFunnel?.dsr_surviving;
  const pipelineCandidates = optimizerFunnel?.pipeline_candidates;

  // Count how many pipeline stages ran (look at first candidate for gates_passed keys)
  const sampleCandidate = report.candidates[0];
  const pipelineStageCount = sampleCandidate?.gates_passed
    ? Object.keys(sampleCandidate.gates_passed).length
    : 0;

  let narrative: string;

  if (verdict === "RED") {
    const parts: string[] = [];
    if (totalTrials != null) {
      parts.push(
        `The optimizer evaluated ${formatNumber(totalTrials, 0)} parameter combinations.`,
      );
    }
    if (refinementPassing != null) {
      parts.push(
        `${formatNumber(refinementPassing, 0)} passed quality filters.`,
      );
    }
    if (dsrSurviving != null) {
      parts.push(
        `${formatNumber(dsrSurviving, 0)} passed the DSR statistical significance filter.`,
      );
    }
    if (pipelineCandidates != null) {
      parts.push(
        `${formatNumber(pipelineCandidates, 0)} candidates entered the validation pipeline.`,
      );
    }
    if (pipelineStageCount > 0) {
      parts.push(
        `After ${pipelineStageCount}-stage validation, 0 of ${totalCandidates} candidates survived.`,
      );
    } else {
      parts.push(`0 of ${totalCandidates} candidates survived validation.`);
    }
    narrative = parts.join(" ");
  } else if (verdict === "YELLOW") {
    const parts: string[] = [];
    if (totalTrials != null) {
      parts.push(
        `The optimizer evaluated ${formatNumber(totalTrials, 0)} parameter combinations.`,
      );
    }
    parts.push(
      `${survivors} of ${totalCandidates} candidates survived validation but with moderate confidence.`,
    );
    parts.push("Further testing recommended before live deployment.");
    narrative = parts.join(" ");
  } else {
    const parts: string[] = [];
    if (totalTrials != null) {
      parts.push(
        `The optimizer evaluated ${formatNumber(totalTrials, 0)} parameter combinations.`,
      );
    }
    parts.push(
      `${survivors} of ${totalCandidates} candidates passed all validation stages${
        highConfidence > 0 ? ` (${highConfidence} with high confidence)` : ""
      }.`,
    );
    parts.push("Ready for paper trading evaluation.");
    narrative = parts.join(" ");
  }

  // Headline text
  const headline: Record<Verdict, string> = {
    RED: "No tradeable edge was found.",
    YELLOW: "Weak candidates found -- further testing needed.",
    GREEN: "Strong candidates identified.",
  };

  return (
    <div
      className={`rounded-xl border p-5 ${config.bg} ${config.border}`}
    >
      <div className="flex items-start gap-4">
        {/* Verdict badge */}
        <div
          className={`flex-shrink-0 px-3 py-1.5 rounded-lg text-sm font-bold tracking-wide ${config.badge} ${config.badgeText}`}
        >
          {verdict}
        </div>

        {/* Text content */}
        <div className="flex-1 min-w-0">
          <h3 className={`text-base font-bold mb-1.5 ${config.textColor}`}>
            {headline[verdict]}
          </h3>
          <p className="text-sm text-gray-400 leading-relaxed">
            {narrative}
          </p>

          {/* Quick stats row */}
          <div className="flex flex-wrap gap-4 mt-3 text-xs text-gray-500">
            {totalTrials != null && (
              <span>
                Trials: <span className="text-gray-300 font-mono">{formatNumber(totalTrials, 0)}</span>
              </span>
            )}
            <span>
              Candidates: <span className="text-gray-300 font-mono">{totalCandidates}</span>
            </span>
            <span>
              Survivors: <span className={`font-mono ${survivors > 0 ? "text-green-400" : "text-red-400"}`}>{survivors}</span>
            </span>
            {pipelineStageCount > 0 && (
              <span>
                Validation stages: <span className="text-gray-300 font-mono">{pipelineStageCount}</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
