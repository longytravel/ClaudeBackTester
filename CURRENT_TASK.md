# Current Task

## Status: Rust Backend Integrated — Ready for M1 Stress Testing

Phase 6 (live trading) is being built by another agent. Phase 5b optimizer/validation enhancements in progress.

## What's Built
- **Phase 1**: Data pipeline (Dukascopy downloader, timeframes, validation, splitting, MT5 broker)
- **Phase 2**: Strategy framework (14 indicators, base class, param space, SL/TP calc, registry)
- **Phase 3**: Backtest engine (JIT batch evaluator, metrics, encoding, telemetry, orchestrator)
- **Phase 4**: Parameter optimizer (Sobol/EDA samplers, staged optimization, ranking, diversity archive)
- **Execution Cost Modeling**: SELL exit spread, round-trip commission, max spread filter
- **Phase 5**: Validation pipeline (walk-forward, stability, Monte Carlo, confidence scoring, checkpoint/resume, JSON reports)
- **Phase 5b VP-2**: Multi-candidate pipeline — optimizer returns top N diverse candidates, pipeline validates all
- **Phase 5b VP-1**: CPCV validation — C(N,k) purged cross-validation, 45 folds, integrated into confidence scoring
- **Phase 5b OPT-1**: Adaptive LR + entropy diagnostics for EDA sampler (pairwise dependencies skipped — unproven)
- **Phase 5b VP-3**: Regime-aware validation — ADX+NATR 4-quadrant classification, per-regime stats, advisory robustness score
- **Causality Contract**: SignalCausality enum, pipeline/engine guards, automated verification tests, legacy cleanup
- **Subprocess Isolation**: Full pipeline (optimizer + validation) runs in isolated subprocesses
- **Rust Backend**: PyO3 + Rayon native extension replaces Numba JIT hot loop. Bit-for-bit parity verified. Auto-fallback to Numba if not built.
- **522 tests passing**

## Completed Phase 5b Enhancements
- [x] VP-2: Multi-Candidate Pipeline (Increments 1-4)
- [x] VP-1: CPCV Combinatorial Purged Cross-Validation (Increments 5-10)
- [x] OPT-1: CE Exploitation Upgrade (descoped: adaptive LR + entropy, skipped pairwise dependencies)
- [x] VP-3: Regime-Aware Validation (4 increments: classify_bars, regime stats, pipeline integration, reporting)
- [x] Causality Contract + Pipeline Hardening (SignalCausality enum, guards, tests, legacy cleanup)
- [x] Subprocess Isolation: Pipeline runs in isolated subprocess, optimizer skips M1
- [x] Rust Backend: backtester_core (PyO3 + Rayon) replaces Numba, 14 Rust unit tests + 11 parity tests

## Rust Backend — Implementation Details (for context recovery)

### What was done this session
Replaced the Numba @njit hot loop (`backtester/core/jit_loop.py`, 963 lines) with a Rust native extension via PyO3/maturin. The Numba NRT allocator on Windows pools memory and never returns it to the OS, causing segfaults with M1 data (5-10M bars). Rust uses the system allocator with deterministic deallocation.

### Files created
- `rust/Cargo.toml` — pyo3 0.24, numpy 0.24, rayon 1.10 deps
- `rust/pyproject.toml` — maturin build config, module-name = "backtester_core"
- `rust/build.sh` — Windows build script (sets MSVC env vars, auto-detects SDK versions)
- `rust/src/lib.rs` — PyO3 module, `batch_evaluate()` binding with Rayon parallel iter + GIL release
- `rust/src/constants.rs` — All constants mirrored from dtypes.py (27 PL_*, metric indices, modes)
- `rust/src/filter.rs` — `signal_passes_time_filter()` — hour wrap-around + day bitmask
- `rust/src/sl_tp.rs` — `compute_sl_tp()` — 3 SL modes x 3 TP modes
- `rust/src/trade_basic.rs` — `simulate_trade_basic()` — SL/TP with M1 sub-bars
- `rust/src/trade_full.rs` — `simulate_trade_full()` — trailing, BE, partial, stale, max bars, deferred SL
- `rust/src/metrics.rs` — `compute_metrics_inline()` — all 10 metrics
- `backtester/core/rust_loop.py` — Auto-dispatcher: Rust → Numba fallback, `BACKTESTER_BACKEND` env var
- `tests/test_rust_parity.py` — 11 parity tests (EXEC_BASIC, EXEC_FULL, edge cases, 500-trial stress)

### Files modified
- `backtester/core/engine.py` — Changed `from backtester.core.jit_loop import` → `from backtester.core.rust_loop import` (2 places: top-level and inside `_build_param_layout`)
- `.gitignore` — Added `rust/target/`
- `CLAUDE.md` — Updated Tech Stack, Architecture, added Rust build section, updated project structure

### Build environment (Windows)
- Rust 1.93.1 (stable-x86_64-pc-windows-msvc) installed via rustup
- VS Build Tools 2022 installed at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`
- MSVC 14.44.35207, Windows SDK 10.0.26100.0
- maturin 1.12.4 installed in venv
- **CRITICAL**: Git Bash has `/usr/bin/link.exe` (GNU coreutils) that shadows MSVC's linker. `build.sh` handles this by prepending MSVC to PATH.
- Build command: `cd rust && bash build.sh` (or manually set MSVC env + `maturin develop --release`)

### Design decisions
- **Zero-copy**: numpy arrays passed as `PyReadonlyArray1<f64>` raw pointer views. Output via mutable `PyArray2<f64>`.
- **Rayon parallelism**: `into_par_iter()` with `chunks_mut()` for non-overlapping output slices. GIL released via `py.allow_threads()`.
- **Automatic fallback**: `rust_loop.py` tries `import backtester_core`, falls back to `jit_loop`. Env var override available.
- **Line-for-line port**: Each Numba function → one Rust file. Same logic, same constants, same control flow.
- **Parity verified**: Bit-for-bit identical output (atol=1e-12) across EXEC_BASIC, EXEC_FULL, 500+ trials, with/without costs.

### NOT yet done
1. **M1 stress test** — Need to run EUR/USD M15 with M1 data through Rust backend to prove no segfaults
2. **Remove subprocess isolation** — `scripts/full_run.py` still uses subprocess isolation; can simplify once Rust is proven stable
3. **Benchmark** — No throughput comparison yet (Rust vs Numba evals/sec)
4. **`jit_loop.py` NOT deleted** — Kept as Numba fallback. Only `engine.py` changed to use `rust_loop.py`.

## Next Steps
1. **M1 Stress Test** — Run EUR/USD M15 with M1 data (5.7M bars) through Rust backend, verify no segfaults
2. **Remove subprocess isolation workarounds** once Rust is stable with M1
3. **Benchmark Rust vs Numba** throughput comparison
4. **OPT-2: GT-Score Objective** — A/B test vs Quality Score
5. **OPT-3: Batch Size Auto-Tuning** — Benchmark and auto-select

## Phase 6 — Live Trading (separate agent, do NOT build)
- Live Trading Engine (REQ-L01-L15)
- Risk Management (REQ-R01-R13)
- Broker Integration (REQ-L16-L30)

## Blockers
- None
