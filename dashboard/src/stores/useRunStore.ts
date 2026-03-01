import { create } from "zustand";
import type {
  BatchUpdate,
  StageComplete,
  PipelineUpdate,
  Report,
  WSMessage,
  RunConfig,
  Snapshot,
} from "../types/api";

export interface StageInfo {
  name: string;
  status: "pending" | "active" | "complete";
  best_quality?: number;
  trials_evaluated?: number;
  valid_count?: number;
  elapsed_secs?: number;
}

interface RunState {
  // Connection
  status: "idle" | "connecting" | "running" | "complete";

  // Run config
  strategy: string;
  pair: string;
  timeframe: string;
  preset: string;

  // Optimization progress
  stages: StageInfo[];
  currentStageIndex: number;
  batchHistory: BatchUpdate[];
  bestQuality: number;
  bestSharpe: number;
  bestTrades: number;
  totalEvaluated: number;
  evalsPerSecond: number;
  elapsedSecs: number;

  // Current batch info
  currentPhase: string;
  entropy: number | null;
  effectiveLr: number | null;
  validRate: number;

  // Pipeline progress
  pipelineStages: PipelineUpdate[];

  // Final report
  report: Report | null;

  // Actions
  handleMessage: (msg: WSMessage) => void;
  setStatus: (s: RunState["status"]) => void;
  reset: () => void;
}

const initialState = {
  status: "idle" as const,
  strategy: "",
  pair: "",
  timeframe: "",
  preset: "",
  stages: [] as StageInfo[],
  currentStageIndex: -1,
  batchHistory: [] as BatchUpdate[],
  bestQuality: 0,
  bestSharpe: 0,
  bestTrades: 0,
  totalEvaluated: 0,
  evalsPerSecond: 0,
  elapsedSecs: 0,
  currentPhase: "",
  entropy: null as number | null,
  effectiveLr: null as number | null,
  validRate: 0,
  pipelineStages: [] as PipelineUpdate[],
  report: null as Report | null,
};

export const useRunStore = create<RunState>((set, get) => ({
  ...initialState,

  setStatus: (status) => {
    // Never downgrade from "complete" to "connecting" (server shutdown after run)
    const current = get().status;
    if (current === "complete" && status === "connecting") return;
    set({ status });
  },

  reset: () => set(initialState),

  handleMessage: (msg) => {
    switch (msg.type) {
      case "snapshot": {
        // Server sends snapshot on reconnect with full state
        const snap = msg as Snapshot;

        if (snap.run_config) {
          // Replay run_config
          get().handleMessage(snap.run_config);
        }

        // Replay stage completions
        if (snap.stage_results) {
          for (const sr of snap.stage_results) {
            get().handleMessage(sr);
          }
        }

        // Replay full batch history for charts (not just last state)
        if (snap.batch_history && snap.batch_history.length > 0) {
          for (const batch of snap.batch_history) {
            get().handleMessage(batch);
          }
        } else if (snap.last_state) {
          // Fallback: replay just the last batch state
          get().handleMessage(snap.last_state);
        }

        // Replay pipeline progress
        if (snap.pipeline_results) {
          for (const pr of snap.pipeline_results) {
            get().handleMessage(pr);
          }
        }

        // Replay final report
        if (snap.final_report) {
          get().handleMessage({
            type: "run_complete",
            report: snap.final_report,
          } as WSMessage);
        }
        break;
      }

      case "run_config": {
        const m = msg as RunConfig;
        set({
          status: "running",
          strategy: m.strategy,
          pair: m.pair,
          timeframe: m.timeframe,
          preset: m.preset,
          stages: m.stages.map((name, i) => ({
            name,
            status: i === 0 ? "active" : "pending",
          })),
          currentStageIndex: 0,
          batchHistory: [],
          pipelineStages: [],
          report: null,
        });
        break;
      }

      case "batch": {
        const m = msg as BatchUpdate;
        const state = get();

        // Update stages array — don't overwrite "complete" back to "active"
        const stages = [...state.stages];
        for (let i = 0; i < stages.length; i++) {
          if (i < m.stage_index)
            stages[i] = { ...stages[i], status: "complete" };
          else if (i === m.stage_index)
            stages[i] = { ...stages[i], status: stages[i].status === "complete" ? "complete" : "active" };
          else stages[i] = { ...stages[i], status: "pending" };
        }

        // Cumulative trials: sum completed stage trials + current stage trials_done
        const completedTrials = stages
          .filter((s) => s.status === "complete" && s.trials_evaluated)
          .reduce((sum, s) => sum + (s.trials_evaluated ?? 0), 0);
        const cumulativeEvaluated = completedTrials + m.trials_done;

        set({
          stages,
          currentStageIndex: m.stage_index,
          batchHistory: [...state.batchHistory, m],
          bestQuality: m.best_quality,
          bestSharpe: m.best_sharpe,
          bestTrades: m.best_trades,
          totalEvaluated: cumulativeEvaluated,
          evalsPerSecond: m.evals_per_sec,
          elapsedSecs: m.elapsed_secs,
          currentPhase: m.phase,
          entropy: m.entropy,
          effectiveLr: m.effective_lr,
          validRate: m.valid_rate,
        });
        break;
      }

      case "stage_complete": {
        const m = msg as StageComplete;
        const state = get();
        const stages = [...state.stages];
        if (m.stage_index < stages.length) {
          stages[m.stage_index] = {
            ...stages[m.stage_index],
            status: "complete",
            best_quality: m.best_quality,
            trials_evaluated: m.trials_evaluated,
            valid_count: m.valid_count,
            elapsed_secs: m.elapsed_secs,
          };
        }
        // Activate next stage
        if (m.stage_index + 1 < stages.length) {
          stages[m.stage_index + 1] = {
            ...stages[m.stage_index + 1],
            status: "active",
          };
        }
        set({ stages, currentStageIndex: m.stage_index + 1 });
        break;
      }

      case "pipeline": {
        const m = msg as PipelineUpdate;
        set((state) => ({
          pipelineStages: [...state.pipelineStages, m],
        }));
        break;
      }

      case "run_complete": {
        const m = msg as { type: "run_complete"; report: Report };
        set({
          status: "complete",
          report: m.report,
        });
        break;
      }

      case "ping":
        // Server keepalive, ignore
        break;
    }
  },
}));
