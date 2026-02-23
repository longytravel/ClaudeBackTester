"""Full end-to-end run: load data → optimize → validate pipeline.

Usage:
    uv run python scripts/full_run.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_run")

# Suppress noisy loggers
logging.getLogger("numba").setLevel(logging.WARNING)

# ============================================================
# Configuration
# ============================================================
PAIR = "EUR/USD"
TIMEFRAME = "H1"
PIP_VALUE = 0.0001
SLIPPAGE_PIPS = 0.5
DATA_DIR = Path("G:/My Drive/BackTestData")
OUTPUT_DIR = Path("./results/rsi_eurusd_h1")

# ============================================================
# 1. Load data
# ============================================================
pair_file = PAIR.replace("/", "_")
parquet_path = DATA_DIR / f"{pair_file}_{TIMEFRAME}.parquet"

logger.info(f"Loading {parquet_path}...")
if not parquet_path.exists():
    logger.error(f"Data file not found: {parquet_path}")
    sys.exit(1)

df = pd.read_parquet(parquet_path)
logger.info(f"Loaded {len(df):,} bars: {df.index[0]} to {df.index[-1]}")

# ============================================================
# 2. Split 80/20
# ============================================================
from backtester.data.splitting import split_backforward

back_df, fwd_df = split_backforward(df, back_pct=0.80)
logger.info(f"Back: {len(back_df):,} bars ({back_df.index[0]} to {back_df.index[-1]})")
logger.info(f"Forward: {len(fwd_df):,} bars ({fwd_df.index[0]} to {fwd_df.index[-1]})")


def df_to_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert DataFrame to numpy arrays dict for the engine."""
    return {
        "open": frame["open"].to_numpy(dtype=np.float64),
        "high": frame["high"].to_numpy(dtype=np.float64),
        "low": frame["low"].to_numpy(dtype=np.float64),
        "close": frame["close"].to_numpy(dtype=np.float64),
        "volume": frame["volume"].to_numpy(dtype=np.float64),
        "spread": frame["spread"].to_numpy(dtype=np.float64),
        "bar_hour": frame.index.hour.to_numpy(dtype=np.int64),
        "bar_day_of_week": frame.index.dayofweek.to_numpy(dtype=np.int64),
    }


data_back = df_to_arrays(back_df)
data_fwd = df_to_arrays(fwd_df)
data_full = df_to_arrays(df)

# ============================================================
# 3. Create strategy
# ============================================================
from backtester.strategies.rsi_mean_reversion import RSIMeanReversion

strategy = RSIMeanReversion()
logger.info(f"Strategy: {strategy.name} v{strategy.version}")
logger.info(f"Param space: {len(strategy.param_space())} params")

# ============================================================
# 4. Run optimization (turbo preset for speed)
# ============================================================
from backtester.optimizer.config import get_preset
from backtester.optimizer.run import optimize

opt_config = get_preset("turbo")
logger.info(f"Optimization preset: turbo ({opt_config.trials_per_stage} trials/stage)")
logger.info("Starting optimization...")

t0 = time.time()
opt_result = optimize(
    strategy=strategy,
    open_back=data_back["open"],
    high_back=data_back["high"],
    low_back=data_back["low"],
    close_back=data_back["close"],
    volume_back=data_back["volume"],
    spread_back=data_back["spread"],
    open_fwd=data_fwd["open"],
    high_fwd=data_fwd["high"],
    low_fwd=data_fwd["low"],
    close_fwd=data_fwd["close"],
    volume_fwd=data_fwd["volume"],
    spread_fwd=data_fwd["spread"],
    config=opt_config,
    pip_value=PIP_VALUE,
    slippage_pips=SLIPPAGE_PIPS,
    bar_hour_back=data_back["bar_hour"],
    bar_day_back=data_back["bar_day_of_week"],
    bar_hour_fwd=data_fwd["bar_hour"],
    bar_day_fwd=data_fwd["bar_day_of_week"],
)
opt_elapsed = time.time() - t0

logger.info(f"Optimization: {opt_result.total_trials:,} trials in {opt_elapsed:.1f}s "
            f"({opt_result.evals_per_second:.0f} evals/sec)")
logger.info(f"Candidates found: {len(opt_result.candidates)}")

if not opt_result.candidates:
    logger.error("No candidates found! Optimization failed.")
    sys.exit(1)

# Print candidate details
for i, cand in enumerate(opt_result.candidates):
    logger.info(
        f"  Candidate {i}: back_quality={cand.back_metrics.get('quality_score', 0):.2f}, "
        f"back_sharpe={cand.back_metrics.get('sharpe', 0):.3f}, "
        f"back_trades={cand.back_metrics.get('trades', 0):.0f}, "
        f"fwd_quality={cand.forward_metrics.get('quality_score', 0) if cand.forward_metrics else 0:.2f}, "
        f"dsr={cand.dsr:.3f}, fb_ratio={cand.forward_back_ratio:.3f}"
    )
    logger.info(f"    Params: {cand.params}")

# ============================================================
# 5. Run validation pipeline
# ============================================================
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.runner import PipelineRunner
from backtester.pipeline.types import CandidateResult

# Build CandidateResults with optimizer metadata
candidate_results = []
for i, cand in enumerate(opt_result.candidates):
    back_q = cand.back_metrics.get("quality_score", 0)
    fwd_q = cand.forward_metrics.get("quality_score", 0) if cand.forward_metrics else 0
    fb_ratio = fwd_q / back_q if back_q > 0 else 0.0
    cr = CandidateResult(
        candidate_index=i,
        params=cand.params,
        back_quality=back_q,
        forward_quality=fwd_q,
        forward_back_ratio=fb_ratio,
        back_sharpe=cand.back_metrics.get("sharpe", 0),
        back_trades=int(cand.back_metrics.get("trades", 0)),
        n_trials=opt_result.total_trials,
    )
    candidate_results.append(cr)

pipeline_config = PipelineConfig()
runner = PipelineRunner(
    strategy=strategy,
    data_arrays=data_full,
    config=pipeline_config,
    pair=PAIR,
    timeframe=TIMEFRAME,
    pip_value=PIP_VALUE,
    slippage_pips=SLIPPAGE_PIPS,
    output_dir=str(OUTPUT_DIR),
)

logger.info("Starting validation pipeline...")
t1 = time.time()
state = runner.run(candidate_results=candidate_results)
pipe_elapsed = time.time() - t1

# ============================================================
# 6. Print results
# ============================================================
logger.info(f"\nPipeline complete in {pipe_elapsed:.1f}s")
logger.info(f"Completed stages: {state.completed_stages}")

active = [c for c in state.candidates if not c.eliminated]
eliminated = [c for c in state.candidates if c.eliminated]
logger.info(f"Survivors: {len(active)}/{len(state.candidates)}")

for c in eliminated:
    logger.info(f"  ELIMINATED at {c.eliminated_at_stage}: {c.elimination_reason}")

for c in active:
    conf = c.confidence
    if conf:
        logger.info(
            f"  SURVIVOR candidate {c.candidate_index}: "
            f"score={conf.composite_score:.1f}, rating={conf.rating.value}, "
            f"wf_pass={c.walk_forward.pass_rate:.0%} if wf else 'N/A', "
            f"stability={c.stability.rating.value if c.stability else 'N/A'}"
        )

report_path = OUTPUT_DIR / "report.json"
if report_path.exists():
    logger.info(f"\nReport saved to: {report_path}")
    # Print report summary
    import json
    with open(report_path) as f:
        report = json.load(f)
    logger.info(f"Report contains {len(report['candidates'])} candidates")

logger.info(f"\nTotal elapsed: {time.time() - t0:.1f}s")
