export interface BatchUpdate {
  type: "batch";
  stage_name: string;
  stage_index: number;
  total_stages: number;
  trials_done: number;
  trials_total: number;
  best_quality: number;
  best_sharpe: number;
  best_trades: number;
  valid_count: number;
  valid_rate: number;
  batch_best_quality: number;
  batch_mean_quality: number;
  phase: "exploration" | "exploitation";
  entropy: number | null;
  effective_lr: number | null;
  evals_per_sec: number;
  elapsed_secs: number;
}

export interface StageComplete {
  type: "stage_complete";
  stage_name: string;
  stage_index: number;
  total_stages: number;
  best_quality: number;
  best_metrics: Record<string, number>;
  trials_evaluated: number;
  valid_count: number;
  elapsed_secs: number;
}

export interface PipelineUpdate {
  type: "pipeline";
  stage_name: string;
  candidates_total: number;
  candidates_surviving: number;
  detail: string;
}

export interface RunConfig {
  type: "run_config";
  strategy: string;
  pair: string;
  timeframe: string;
  preset: string;
  stages: string[];
}

export interface RunComplete {
  type: "run_complete";
  report: Report;
}

export interface Report {
  strategy: string;
  version: string;
  pair: string;
  timeframe: string;
  candidates: CandidateReport[];
}

export interface CandidateReport {
  index: number;
  params: Record<string, number | string | boolean>;
  back_quality: number;
  forward_quality: number;
  forward_back_ratio: number;
  eliminated: boolean;
  eliminated_at?: string;
  elimination_reason?: string;
  composite_score?: number;
  rating?: string;
  gates_passed?: Record<string, boolean>;
  wf_pass_rate?: number;
  wf_mean_sharpe?: number;
  dsr?: number;
  permutation_p?: number;
  cpcv_mean_sharpe?: number;
  cpcv_pct_positive?: number;
  stability_rating?: string;
  stability_mean_ratio?: number;
  regime_distribution?: Record<string, number>;
  regime_robustness_score?: number;
  trade_stats?: TradeStats;
  equity_curve?: { timestamp: number; equity: number }[];
}

export interface TradeStats {
  n_trades: number;
  total_pnl_pips: number;
  total_pnl_usd: number;
  win_rate: number;
  profit_factor: number;
  avg_pnl_pips: number;
  exit_breakdown: Record<
    string,
    { count: number; pct: number; pnl_pips: number }
  >;
}

export interface RunSummary {
  id: string;
  strategy: string;
  pair: string;
  timeframe: string;
  rating?: string;
  composite_score?: number;
  timestamp?: number;
}

export interface Snapshot {
  type: "snapshot";
  run_config: RunConfig | null;
  stage_results: StageComplete[];
  pipeline_results: PipelineUpdate[];
  last_state: BatchUpdate | null;
  batch_history?: BatchUpdate[];
  final_report: Report | null;
}

export interface Ping {
  type: "ping";
}

export type WSMessage =
  | BatchUpdate
  | StageComplete
  | PipelineUpdate
  | RunConfig
  | RunComplete
  | Snapshot
  | Ping;
