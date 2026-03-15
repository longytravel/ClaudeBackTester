import { useState } from "react";
import { useRunStore } from "../stores/useRunStore";
import { Header } from "../components/layout/Header";
import { StageProgress } from "../components/optimization/StageProgress";
import { ConvergenceLine } from "../components/optimization/ConvergenceLine";
import { QualityScatter } from "../components/optimization/QualityScatter";
import { SamplerInfo } from "../components/optimization/SamplerInfo";
import { LiveStats } from "../components/optimization/LiveStats";
import { EquityCurve } from "../components/results/EquityCurve";
import { CandidateTable } from "../components/results/CandidateTable";
import { ConfidenceGauge } from "../components/results/ConfidenceGauge";
import { WalkForwardBars } from "../components/results/WalkForwardBars";
import { ExitBreakdown } from "../components/results/ExitBreakdown";
import { PipelineFunnel } from "../components/results/PipelineFunnel";
import { ParamHeatmap } from "../components/results/ParamHeatmap";
import { MonteCarloHist } from "../components/results/MonteCarloHist";
import { RegimeRadar } from "../components/results/RegimeRadar";
import { RunSummary } from "../components/results/RunSummary";

export function OptimizationRun() {
  const status = useRunStore((s) => s.status);
  const report = useRunStore((s) => s.report);
  const pipelineStages = useRunStore((s) => s.pipelineStages);
  const batchHistory = useRunStore((s) => s.batchHistory);

  const [selectedCandidate, setSelectedCandidate] = useState(0);

  const isRunningOrComplete = status === "running" || status === "complete";
  const hasResults = status === "complete" && report !== null;

  // Find the selected candidate from report
  const candidate =
    hasResults && report
      ? report.candidates.find((c) => c.index === selectedCandidate) ??
        report.candidates[0]
      : null;

  // Best survivor for headline stats
  const bestSurvivor =
    hasResults && report
      ? [...report.candidates]
          .filter((c) => !c.eliminated)
          .sort((a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0))[0] ??
        report.candidates[0]
      : null;

  return (
    <div className="min-h-screen">
      <Header />

      {/* Idle state */}
      {status === "idle" && batchHistory.length === 0 && (
        <div className="flex items-center justify-center h-[calc(100vh-3.5rem)]">
          <div className="text-center">
            <div className="text-6xl mb-4 text-gray-700">
              <svg
                className="w-16 h-16 mx-auto"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-gray-400 mb-2">
              No Active Optimization
            </h2>
            <p className="text-sm text-gray-500 max-w-md">
              Start an optimization run from the CLI and this dashboard will
              automatically connect and display real-time progress.
            </p>
          </div>
        </div>
      )}

      {/* Stage progress bar */}
      {isRunningOrComplete && <StageProgress />}

      {/* Live optimization section */}
      {isRunningOrComplete && (
        <div className="px-6 pb-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left column: Charts */}
            <div className="space-y-4">
              <ConvergenceLine />
              <QualityScatter />
            </div>
            {/* Right column: Stats */}
            <div className="space-y-4">
              <SamplerInfo />
              <LiveStats />
            </div>
          </div>
        </div>
      )}

      {/* Results section */}
      {hasResults && report && (
        <div className="px-6 pb-8 space-y-4">
          <div className="border-t border-gray-800 pt-6 mb-2">
            <h2 className="text-lg font-bold text-gray-200">
              Results - {report.strategy} / {report.pair} / {report.timeframe}
            </h2>
          </div>

          {/* Run summary narrative */}
          <RunSummary
            report={report}
            optimizerFunnel={report.optimizer_funnel}
          />

          {/* Confidence + Pipeline row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {bestSurvivor && (
              <ConfidenceGauge
                compositeScore={bestSurvivor.composite_score ?? 0}
                rating={bestSurvivor.rating ?? "UNKNOWN"}
                gatesPassed={bestSurvivor.gates_passed}
                eliminated={bestSurvivor.eliminated}
                eliminatedAt={bestSurvivor.eliminated_at}
                eliminationReason={bestSurvivor.elimination_reason}
              />
            )}
            <PipelineFunnel
              stages={pipelineStages}
              optimizerFunnel={report.optimizer_funnel}
            />
          </div>

          {/* Equity curve */}
          {candidate && (
            <EquityCurve
              candidate={candidate}
              splitTimestamp={report.back_forward_split_timestamp}
              pair={report.pair}
            />
          )}

          {/* Walk-forward + Monte Carlo row */}
          {candidate && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <WalkForwardBars candidate={candidate} />
              <MonteCarloHist candidate={candidate} />
            </div>
          )}

          {/* Candidate table */}
          <CandidateTable
            candidates={report.candidates}
            onSelect={setSelectedCandidate}
            selectedIndex={selectedCandidate}
          />

          {/* Exit + Regime + Params row */}
          {candidate && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {candidate.trade_stats && (
                <ExitBreakdown tradeStats={candidate.trade_stats} />
              )}
              <RegimeRadar candidate={candidate} />
              <ParamHeatmap candidate={candidate} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
