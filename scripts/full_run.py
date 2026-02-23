"""Full end-to-end run: load data -> optimize -> validate -> report.

Usage:
    uv run python scripts/full_run.py --pair EUR/USD --timeframe H1 --preset turbo
    uv run python scripts/full_run.py  # defaults: EUR/USD H1 turbo
"""

import argparse
import logging
import sys
import time
from collections import Counter
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
logging.getLogger("numba").setLevel(logging.WARNING)

# ============================================================
# PIP values per pair
# ============================================================
PIP_VALUES = {
    "EUR/USD": 0.0001, "GBP/USD": 0.0001, "AUD/USD": 0.0001,
    "NZD/USD": 0.0001, "USD/CHF": 0.0001, "EUR/GBP": 0.0001,
    "USD/JPY": 0.01, "EUR/JPY": 0.01, "GBP/JPY": 0.01,
    "XAU/USD": 0.01,
}

SLIPPAGE_PIPS = 0.5
DATA_DIR = Path("G:/My Drive/BackTestData")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full backtest pipeline run")
    parser.add_argument("--pair", default="EUR/USD", help="Currency pair (e.g. EUR/USD)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (e.g. H1, H4, D)")
    parser.add_argument("--preset", default="turbo", choices=["turbo", "fast", "default"],
                        help="Optimization preset")
    parser.add_argument("--output", default=None, help="Output directory (default: results/<pair>_<tf>)")
    return parser.parse_args()


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


def print_header(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title: str) -> None:
    print(f"\n--- {title} ---")


# ============================================================
# Section 1: Data Loading & Quality Check
# ============================================================
def load_and_check_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Load data with NaN auto-detection and rebuild."""
    pair_file = pair.replace("/", "_")
    tf_path = DATA_DIR / f"{pair_file}_{timeframe}.parquet"
    m1_path = DATA_DIR / f"{pair_file}_M1.parquet"

    # Check if timeframe file needs (re)building
    needs_rebuild = False
    if not tf_path.exists():
        logger.info(f"{timeframe} file not found, building from M1...")
        needs_rebuild = True
    else:
        df_check = pd.read_parquet(tf_path)
        nan_close = df_check["close"].isna().sum()
        if nan_close > 0:
            logger.warning(f"Found {nan_close} NaN close bars in {timeframe} — rebuilding from M1")
            needs_rebuild = True

    if needs_rebuild:
        if not m1_path.exists():
            logger.error(f"M1 data not found at {m1_path}")
            sys.exit(1)
        from backtester.data.timeframes import convert_timeframes
        convert_timeframes(pair, str(DATA_DIR), [timeframe])
        logger.info(f"Rebuilt {timeframe} from M1 data")

    df = pd.read_parquet(tf_path)

    # Print data summary
    print_header("SECTION 1: DATA SUMMARY")
    print(f"  Pair:           {pair}")
    print(f"  Timeframe:      {timeframe}")
    print(f"  Bars:           {len(df):,}")
    print(f"  Date range:     {df.index[0]} to {df.index[-1]}")

    nan_close = df["close"].isna().sum()
    nan_spread = df["spread"].isna().sum()
    spread_pips = df["spread"] / PIP_VALUES.get(pair, 0.0001)
    print(f"  NaN close:      {nan_close}")
    print(f"  NaN spread:     {nan_spread}")
    print(f"  Spread (pips):  median={spread_pips.median():.2f}, "
          f"mean={spread_pips.mean():.2f}, max={spread_pips.max():.2f}")

    if nan_close > 0:
        logger.error("Data still has NaN close bars after rebuild!")
        sys.exit(1)

    return df


# ============================================================
# Section 2: Optimization
# ============================================================
def run_optimization(strategy, data_back, data_fwd, preset_name, pip_value):
    from backtester.optimizer.config import get_preset
    from backtester.optimizer.run import optimize

    opt_config = get_preset(preset_name)

    print_header("SECTION 2: OPTIMIZATION")
    print(f"  Strategy:       {strategy.name} v{strategy.version}")
    print(f"  Param space:    {len(strategy.param_space())} parameters")
    print(f"  Preset:         {preset_name} ({opt_config.trials_per_stage} trials/stage)")

    t0 = time.time()
    opt_result = optimize(
        strategy=strategy,
        open_back=data_back["open"], high_back=data_back["high"],
        low_back=data_back["low"], close_back=data_back["close"],
        volume_back=data_back["volume"], spread_back=data_back["spread"],
        open_fwd=data_fwd["open"], high_fwd=data_fwd["high"],
        low_fwd=data_fwd["low"], close_fwd=data_fwd["close"],
        volume_fwd=data_fwd["volume"], spread_fwd=data_fwd["spread"],
        config=opt_config,
        pip_value=pip_value,
        slippage_pips=SLIPPAGE_PIPS,
        bar_hour_back=data_back["bar_hour"], bar_day_back=data_back["bar_day_of_week"],
        bar_hour_fwd=data_fwd["bar_hour"], bar_day_fwd=data_fwd["bar_day_of_week"],
    )
    elapsed = time.time() - t0

    print(f"\n  Trials:         {opt_result.total_trials:,}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  Throughput:     {opt_result.evals_per_second:,.0f} evals/sec")
    print(f"  Candidates:     {len(opt_result.candidates)}")

    if not opt_result.candidates:
        print("\n  *** NO CANDIDATES FOUND — optimization failed ***")
        sys.exit(1)

    n_cand = len(opt_result.candidates)
    if n_cand > 1:
        print(f"\n  Selected {n_cand} diverse candidates (from "
              f"{opt_result.staged_result.refinement_metrics.shape[0] if opt_result.staged_result and opt_result.staged_result.refinement_metrics is not None else '?'} "
              f"refinement passing trials)")

    for i, cand in enumerate(opt_result.candidates):
        print_subheader(f"Candidate {i}")
        bm = cand.back_metrics
        fm = cand.forward_metrics or {}
        print(f"  Back:    quality={bm.get('quality_score', 0):.2f}, "
              f"sharpe={bm.get('sharpe', 0):.3f}, trades={bm.get('trades', 0):.0f}")
        print(f"  Forward: quality={fm.get('quality_score', 0):.2f}, "
              f"sharpe={fm.get('sharpe', 0):.3f}, trades={fm.get('trades', 0):.0f}")
        print(f"  DSR:     {cand.dsr:.3f}")
        print(f"  FB ratio:{cand.forward_back_ratio:.3f}")
        print(f"  Params:")
        for k, v in sorted(cand.params.items()):
            print(f"    {k:30s} = {v}")

    return opt_result


# ============================================================
# Section 3-6: Validation Pipeline
# ============================================================
def run_validation(strategy, data_full, opt_result, pair, timeframe, pip_value, output_dir):
    from backtester.pipeline.config import PipelineConfig
    from backtester.pipeline.runner import PipelineRunner
    from backtester.pipeline.types import CandidateResult

    candidate_results = []
    for i, cand in enumerate(opt_result.candidates):
        back_q = cand.back_metrics.get("quality_score", 0)
        fwd_q = (cand.forward_metrics or {}).get("quality_score", 0)
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
        pair=pair,
        timeframe=timeframe,
        pip_value=pip_value,
        slippage_pips=SLIPPAGE_PIPS,
        output_dir=str(output_dir),
    )

    t0 = time.time()
    state = runner.run(candidate_results=candidate_results)
    pipe_elapsed = time.time() - t0

    # --- Print Walk-Forward (Section 3) ---
    print_header("SECTION 3: WALK-FORWARD VALIDATION")
    for c in state.candidates:
        wf = c.walk_forward
        if wf is None:
            print(f"  Candidate {c.candidate_index}: SKIPPED (eliminated earlier)")
            continue

        print(f"\n  Candidate {c.candidate_index}:")
        print(f"  {'Window':>8s}  {'Bars':>6s}  {'Trades':>6s}  {'Sharpe':>7s}  "
              f"{'Quality':>8s}  {'PF':>6s}  {'MaxDD%':>7s}  {'Pass':>5s}")
        print(f"  {'-'*60}")
        for w in wf.windows:
            if not w.is_oos:
                continue
            pass_str = "YES" if w.passed else "NO"
            print(f"  {w.window_index:>8d}  {w.end_bar - w.start_bar:>6d}  {w.n_trades:>6d}  "
                  f"{w.sharpe:>7.3f}  {w.quality_score:>8.2f}  {w.profit_factor:>6.2f}  "
                  f"{w.max_dd_pct:>7.2f}  {pass_str:>5s}")
        print(f"\n  Aggregate:")
        print(f"    OOS windows:     {wf.n_oos_windows}")
        print(f"    Passed:          {wf.n_passed}/{wf.n_oos_windows} ({wf.pass_rate:.0%})")
        print(f"    Mean Sharpe:     {wf.mean_sharpe:.3f}")
        print(f"    Mean Quality:    {wf.mean_quality:.2f}")
        print(f"    Geo Mean Qual:   {wf.geo_mean_quality:.2f}")
        print(f"    Min Quality:     {wf.min_quality:.2f}")
        print(f"    Quality CV:      {wf.quality_cv:.3f}")
        print(f"    WFE:             {wf.wfe:.3f}")
        print(f"    Gate:            {'PASSED' if wf.passed_gate else 'FAILED'}")

    # --- Print CPCV (Section 3b) ---
    has_cpcv = any(c.cpcv is not None for c in state.candidates)
    if has_cpcv:
        print_header("SECTION 3b: CPCV VALIDATION")
        for c in state.candidates:
            cpcv = c.cpcv
            if cpcv is None:
                if c.eliminated:
                    print(f"  Candidate {c.candidate_index}: SKIPPED (eliminated at {c.eliminated_at_stage})")
                continue
            print(f"\n  Candidate {c.candidate_index}:")
            print(f"    Blocks:          {cpcv.n_blocks}")
            print(f"    Folds:           {cpcv.n_folds}")
            print(f"    Mean Sharpe:     {cpcv.mean_sharpe:.3f}")
            print(f"    Median Sharpe:   {cpcv.median_sharpe:.3f}")
            print(f"    Std Sharpe:      {cpcv.std_sharpe:.3f}")
            print(f"    95% CI:          [{cpcv.sharpe_ci_low:.3f}, {cpcv.sharpe_ci_high:.3f}]")
            print(f"    % Positive:      {cpcv.pct_positive_sharpe:.0%}")
            print(f"    Mean Quality:    {cpcv.mean_quality:.2f}")
            print(f"    Median Quality:  {cpcv.median_quality:.2f}")
            print(f"    Gate:            {'PASSED' if cpcv.passed_gate else 'FAILED'}")

    # --- Print Stability (Section 4) ---
    print_header("SECTION 4: STABILITY ANALYSIS")
    for c in state.candidates:
        stab = c.stability
        if stab is None:
            print(f"  Candidate {c.candidate_index}: SKIPPED")
            continue
        print(f"\n  Candidate {c.candidate_index}:")
        print(f"    Rating:          {stab.rating.value}")
        print(f"    Mean ratio:      {stab.mean_ratio:.3f}")
        print(f"    Min ratio:       {stab.min_ratio:.3f}")
        print(f"    Worst param:     {stab.worst_param}")

        # Top 3 most fragile params
        sorted_perturbs = sorted(stab.perturbations, key=lambda p: p.ratio)
        print(f"\n    Top 3 most fragile parameters:")
        for p in sorted_perturbs[:3]:
            print(f"      {p.param_name:30s}  ratio={p.ratio:.3f}  "
                  f"({p.original_value} -> {p.perturbed_value})")

    # --- Print Monte Carlo (Section 5) ---
    print_header("SECTION 5: MONTE CARLO ANALYSIS")
    for c in state.candidates:
        mc = c.monte_carlo
        if mc is None:
            if c.eliminated and c.eliminated_at_stage != "monte_carlo":
                print(f"  Candidate {c.candidate_index}: SKIPPED (eliminated at {c.eliminated_at_stage})")
            continue
        print(f"\n  Candidate {c.candidate_index}:")
        print(f"    DSR:                    {mc.dsr:.4f}  {'PASS' if mc.dsr >= 0.95 else 'FAIL'}")
        print(f"    Permutation p-value:    {mc.permutation_p_value:.4f}  "
              f"{'PASS' if mc.permutation_p_value <= 0.05 else 'FAIL'}")
        print(f"    Bootstrap Sharpe mean:  {mc.bootstrap_sharpe_mean:.3f}")
        print(f"    Bootstrap Sharpe 95%CI: [{mc.bootstrap_sharpe_ci_low:.3f}, "
              f"{mc.bootstrap_sharpe_ci_high:.3f}]")
        for level, quality in sorted(mc.skip_results.items()):
            print(f"    Trade-skip {level:>3s}:        quality={quality:.2f}")
        print(f"    Exec stress ratio:      {mc.stress_quality_ratio:.3f}")
        print(f"    Gate:                   {'PASSED' if mc.passed_gate else 'FAILED'}")

    # --- Print Confidence (Section 6) ---
    print_header("SECTION 6: CONFIDENCE SCORING")
    for c in state.candidates:
        conf = c.confidence
        if conf is None:
            status = "ELIMINATED" if c.eliminated else "SKIPPED"
            stage = f" at {c.eliminated_at_stage}" if c.eliminated else ""
            reason = f" ({c.elimination_reason})" if c.elimination_reason else ""
            print(f"  Candidate {c.candidate_index}: {status}{stage}{reason}")
            continue

        print(f"\n  Candidate {c.candidate_index}:")
        print(f"    Gates:")
        for gate_name, passed in conf.gates_passed.items():
            print(f"      {gate_name:25s}  {'PASS' if passed else 'FAIL'}")
        print(f"      All gates:              {'PASS' if conf.all_gates_passed else 'FAIL'}")

        print(f"\n    Sub-Scores (0-100):")
        print(f"      Walk-forward:    {conf.walk_forward_score:6.1f}  (weight 0.30)")
        if conf.cpcv_score > 0:
            print(f"      CPCV:            {conf.cpcv_score:6.1f}  (blended 40% into WF)")
        print(f"      Monte Carlo:     {conf.monte_carlo_score:6.1f}  (weight 0.25)")
        print(f"      Forward/Back:    {conf.forward_back_score:6.1f}  (weight 0.15)")
        print(f"      Stability:       {conf.stability_score:6.1f}  (weight 0.10)")
        print(f"      DSR:             {conf.dsr_score:6.1f}  (weight 0.10)")
        print(f"      Backtest Qual:   {conf.backtest_quality_score:6.1f}  (weight 0.10)")
        print(f"\n    Composite Score:   {conf.composite_score:.1f}")
        print(f"    Rating:            {conf.rating.value}")

    return state, pipe_elapsed


# ============================================================
# Section 7: Trade Statistics (Telemetry)
# ============================================================
def run_trade_stats(strategy, data_full, state, pip_value):
    from backtester.core.dtypes import EXEC_FULL
    from backtester.core.engine import BacktestEngine
    from backtester.core.telemetry import run_telemetry

    active = [c for c in state.candidates if not c.eliminated]
    if not active:
        return

    engine = BacktestEngine(
        strategy,
        data_full["open"], data_full["high"],
        data_full["low"], data_full["close"],
        data_full["volume"], data_full["spread"],
        pip_value=pip_value, slippage_pips=SLIPPAGE_PIPS,
        bar_hour=data_full.get("bar_hour"),
        bar_day_of_week=data_full.get("bar_day_of_week"),
    )

    print_header("SECTION 7: TRADE STATISTICS")

    for c in active:
        telemetry = run_telemetry(engine, c.params, EXEC_FULL)
        trades = telemetry.trades

        if not trades:
            print(f"\n  Candidate {c.candidate_index}: 0 trades")
            continue

        print(f"\n  Candidate {c.candidate_index}: {len(trades)} trades")

        # Basic stats
        pnls = [t.pnl_pips for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        gross_win = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        print(f"\n    P&L Summary:")
        print(f"      Total P&L:       {total_pnl:+.1f} pips")
        print(f"      Win rate:        {win_rate:.1f}%")
        print(f"      Profit factor:   {pf:.2f}")
        print(f"      Avg P&L/trade:   {np.mean(pnls):+.2f} pips")
        print(f"      Median P&L:      {np.median(pnls):+.2f} pips")
        print(f"      Best trade:      {max(pnls):+.1f} pips")
        print(f"      Worst trade:     {min(pnls):+.1f} pips")
        print(f"      Std dev:         {np.std(pnls):.2f} pips")

        # Exit reason distribution
        exit_counts = Counter(t.exit_reason for t in trades)
        print(f"\n    Exit Reasons:")
        for reason, count in exit_counts.most_common():
            pct = count / len(trades) * 100
            reason_pnl = sum(t.pnl_pips for t in trades if t.exit_reason == reason)
            print(f"      {reason:20s}  {count:>5d} ({pct:5.1f}%)  P&L: {reason_pnl:+.1f} pips")

        # Duration stats
        bars_held = [t.bars_held for t in trades]
        print(f"\n    Duration:")
        print(f"      Mean bars held:  {np.mean(bars_held):.1f}")
        print(f"      Median bars:     {np.median(bars_held):.0f}")
        print(f"      Max bars:        {max(bars_held)}")

        # MFE/MAE
        mfes = [t.mfe_pips for t in trades]
        maes = [t.mae_pips for t in trades]
        print(f"\n    MFE/MAE (pips):")
        print(f"      Mean MFE:        {np.mean(mfes):.1f}")
        print(f"      Mean MAE:        {np.mean(maes):.1f}")
        print(f"      MFE/MAE ratio:   {np.mean(mfes) / np.mean(maes):.2f}" if np.mean(maes) > 0 else "")

        # Direction breakdown
        buys = [t for t in trades if t.direction == "BUY"]
        sells = [t for t in trades if t.direction == "SELL"]
        print(f"\n    Direction Breakdown:")
        if buys:
            buy_pnl = sum(t.pnl_pips for t in buys)
            buy_wr = sum(1 for t in buys if t.pnl_pips > 0) / len(buys) * 100
            print(f"      BUY:  {len(buys):>5d} trades, WR={buy_wr:.1f}%, P&L={buy_pnl:+.1f} pips")
        if sells:
            sell_pnl = sum(t.pnl_pips for t in sells)
            sell_wr = sum(1 for t in sells if t.pnl_pips > 0) / len(sells) * 100
            print(f"      SELL: {len(sells):>5d} trades, WR={sell_wr:.1f}%, P&L={sell_pnl:+.1f} pips")

        # Metrics from telemetry
        m = telemetry.metrics
        if m:
            print(f"\n    Full Metrics:")
            for key in ["sharpe", "sortino", "max_dd_pct", "return_pct", "r_squared",
                         "ulcer_index", "quality_score", "profit_factor", "win_rate"]:
                if key in m:
                    print(f"      {key:20s}  {m[key]:.4f}")


# ============================================================
# Section 8: Final Verdict
# ============================================================
def print_verdict(state):
    print_header("SECTION 8: FINAL VERDICT")

    active = [c for c in state.candidates if not c.eliminated]
    eliminated = [c for c in state.candidates if c.eliminated]

    print(f"  Candidates: {len(active)} survived / {len(state.candidates)} total")

    # Sort survivors by composite score (best first)
    active.sort(
        key=lambda c: c.confidence.composite_score if c.confidence else 0,
        reverse=True,
    )

    for c in eliminated:
        print(f"  ELIMINATED candidate {c.candidate_index} at {c.eliminated_at_stage}: "
              f"{c.elimination_reason}")

    if not active:
        print(f"\n  {'*' * 50}")
        print(f"  *  VERDICT: RED — No candidates survived           *")
        print(f"  *  All candidates eliminated during validation      *")
        print(f"  {'*' * 50}")
        print(f"\n  Recommendation: Try 'fast' or 'default' preset with more trials,")
        print(f"  or investigate why the strategy fails validation.")
        return

    for c in active:
        conf = c.confidence
        if not conf:
            continue

        rating = conf.rating.value
        score = conf.composite_score
        marker = {"GREEN": "+++", "YELLOW": "~~~", "RED": "---"}.get(rating, "???")

        print(f"\n  {marker * 15}")
        print(f"  CANDIDATE {c.candidate_index}: {rating} (score: {score:.1f}/100)")
        print(f"  {marker * 15}")

        # Strengths
        strengths = []
        weaknesses = []

        if conf.walk_forward_score >= 70:
            strengths.append(f"Strong walk-forward consistency ({conf.walk_forward_score:.0f})")
        elif conf.walk_forward_score < 40:
            weaknesses.append(f"Weak walk-forward ({conf.walk_forward_score:.0f})")

        if conf.monte_carlo_score >= 70:
            strengths.append(f"Robust Monte Carlo ({conf.monte_carlo_score:.0f})")
        elif conf.monte_carlo_score < 40:
            weaknesses.append(f"Weak Monte Carlo ({conf.monte_carlo_score:.0f})")

        if conf.forward_back_score >= 70:
            strengths.append(f"Good forward/back ratio ({conf.forward_back_score:.0f})")
        elif conf.forward_back_score < 30:
            weaknesses.append(f"Weak forward/back ({conf.forward_back_score:.0f})")

        if conf.stability_score >= 70:
            strengths.append(f"Parameter stability ({conf.stability_score:.0f})")
        elif conf.stability_score < 30:
            weaknesses.append(f"Fragile parameters ({conf.stability_score:.0f})")

        if conf.cpcv_score >= 70:
            strengths.append(f"Strong CPCV distribution ({conf.cpcv_score:.0f})")
        elif conf.cpcv_score > 0 and conf.cpcv_score < 30:
            weaknesses.append(f"Weak CPCV distribution ({conf.cpcv_score:.0f})")

        if conf.dsr_score >= 80:
            strengths.append(f"High DSR ({conf.dsr_score:.0f})")

        if conf.backtest_quality_score >= 70:
            strengths.append(f"Strong backtest quality ({conf.backtest_quality_score:.0f})")

        if strengths:
            print(f"\n  Strengths:")
            for s in strengths:
                print(f"    + {s}")
        if weaknesses:
            print(f"\n  Weaknesses:")
            for w in weaknesses:
                print(f"    - {w}")

        # Recommendations
        print(f"\n  Recommendation:")
        if rating == "GREEN":
            print(f"    Strategy passes all validation gates with good confidence.")
            print(f"    Safe to proceed to paper trading on demo account.")
        elif rating == "YELLOW":
            print(f"    Strategy passes gates but composite score is moderate.")
            print(f"    Consider running with 'default' preset for more thorough optimization")
            print(f"    before paper trading.")
        else:
            print(f"    Strategy failed one or more validation gates.")
            failed_gates = [g for g, p in conf.gates_passed.items() if not p]
            if failed_gates:
                print(f"    Failed gates: {', '.join(failed_gates)}")
            print(f"    Do NOT paper trade. Investigate failed gates or try different strategy.")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    pair = args.pair
    timeframe = args.timeframe
    preset = args.preset

    pip_value = PIP_VALUES.get(pair, 0.0001)
    output_dir = Path(args.output) if args.output else Path(f"./results/{pair.replace('/', '_').lower()}_{timeframe.lower()}")

    t_start = time.time()

    # ---- Section 1: Load Data ----
    df = load_and_check_data(pair, timeframe)

    # ---- Split 80/20 ----
    from backtester.data.splitting import split_backforward
    back_df, fwd_df = split_backforward(df, back_pct=0.80)
    print(f"\n  Back-test:      {len(back_df):,} bars ({back_df.index[0]} to {back_df.index[-1]})")
    print(f"  Forward-test:   {len(fwd_df):,} bars ({fwd_df.index[0]} to {fwd_df.index[-1]})")

    data_back = df_to_arrays(back_df)
    data_fwd = df_to_arrays(fwd_df)
    data_full = df_to_arrays(df)

    # ---- Section 2: Optimization ----
    from backtester.strategies.rsi_mean_reversion import RSIMeanReversion
    strategy = RSIMeanReversion()
    opt_result = run_optimization(strategy, data_back, data_fwd, preset, pip_value)

    # ---- Sections 3-6: Validation Pipeline ----
    state, pipe_elapsed = run_validation(
        strategy, data_full, opt_result, pair, timeframe, pip_value, output_dir,
    )

    # ---- Section 7: Trade Statistics ----
    run_trade_stats(strategy, data_full, state, pip_value)

    # ---- Section 8: Verdict ----
    print_verdict(state)

    # ---- Summary Footer ----
    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Total elapsed: {total_elapsed:.1f}s "
          f"(optimization: {opt_result.elapsed_seconds:.1f}s, pipeline: {pipe_elapsed:.1f}s)")
    print(f"  Report: {output_dir / 'report.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
