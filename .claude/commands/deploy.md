# /deploy — Deploy Strategy to VPS

Deploy a backtest result to the VPS for practice or live trading. This is the bridge between "we found a strategy" and "it's running on the VPS making trades."

## Usage
```
/deploy [RUN_DIR]
```
RUN_DIR is optional (e.g., `eur_usd_m15`). If not provided, scan and ask.

## Step 1: Scan Available Runs

List all pipeline results and their validation status:

```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python -c "
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

results_dir = Path('results')
if not results_dir.exists():
    print('No results/ directory found.')
    sys.exit(0)

runs = []
for cp_path in sorted(results_dir.glob('*/checkpoint.json')):
    try:
        data = json.loads(cp_path.read_text())
        candidates = data.get('candidates', [])
        if not candidates:
            continue
        best = candidates[0]
        name = data.get('strategy_name', '?')
        pair = data.get('pair', '?')
        tf = data.get('timeframe', '?')
        eliminated = best.get('eliminated', True)
        elim_at = best.get('eliminated_at_stage', best.get('eliminated_at', ''))
        elim_reason = best.get('elimination_reason', '')
        back_q = best.get('back_quality', 0)
        back_sharpe = best.get('back_sharpe', 0)
        n_trades = best.get('back_trades', 0)
        wf_pass = best.get('walk_forward', {}).get('pass_rate', 0) if best.get('walk_forward') else 0
        wf_sharpe = best.get('walk_forward', {}).get('mean_sharpe', 0) if best.get('walk_forward') else 0

        # Check report for rating
        report_path = cp_path.parent / 'report.json'
        rating = ''
        if report_path.exists():
            rdata = json.loads(report_path.read_text())
            rcands = rdata.get('candidates', [])
            if rcands:
                rating = rcands[0].get('rating', '')

        status = 'ELIMINATED' if eliminated else (rating if rating else 'PASSED')
        print(f'DIR: {cp_path.parent.name}')
        print(f'  Strategy: {name} | Pair: {pair} | TF: {tf}')
        print(f'  Status: {status}')
        if eliminated:
            print(f'  Eliminated at: {elim_at} ({elim_reason})')
        print(f'  Back Quality: {back_q:.2f} | Back Sharpe: {back_sharpe:.3f} | Trades: {n_trades}')
        if wf_pass > 0 or wf_sharpe != 0:
            print(f'  WF Pass Rate: {wf_pass:.0%} | WF Mean Sharpe: {wf_sharpe:.3f}')
        if rating:
            print(f'  Rating: {rating}')
        print()
    except Exception as e:
        print(f'Error reading {cp_path}: {e}')
"
```

Present this as a clear summary table to the user.

## Step 2: Select Run to Deploy

If RUN_DIR was provided in $ARGUMENTS, use that. Otherwise, use AskUserQuestion to ask:

1. **Which run?** — List the directory names as options (e.g., `eur_usd_m15`, `eur_usd_h1`)
2. **Trading mode?** — Options:
   - `practice` (Recommended) — Demo account, no real money
   - `live` — Real money (requires typing LIVE to confirm on VPS)

If the selected run has ALL candidates eliminated, display a prominent warning:

```
WARNING: This strategy was ELIMINATED by the validation pipeline.
  Eliminated at: [stage]
  Reason: [reason]

  Deploying an eliminated strategy means trading with parameters that
  FAILED validation. The pipeline said "this doesn't have an edge."

  You can force-deploy for paper trading to gather live data, but
  DO NOT force-deploy to live trading with real money.
```

Then ask: "Do you want to force-deploy this anyway (for practice/paper trading)?"

## Step 3: Prepare Checkpoint for Deployment

### If the candidate passed validation (not eliminated)
No changes needed. Skip to Step 4.

### If force-deploying an eliminated candidate
Modify the checkpoint to un-eliminate the candidate so `start_all.py` will pick it up:

```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python -c "
import json
from pathlib import Path

cp_path = Path('results/RUN_DIR/checkpoint.json')
data = json.loads(cp_path.read_text())

# Un-eliminate the first candidate
c = data['candidates'][0]
c['eliminated'] = False
c['force_deployed'] = True
c['force_deploy_note'] = 'Manually force-deployed by user. Original elimination: ELIM_REASON'

# Write back atomically
tmp = cp_path.with_suffix('.tmp')
tmp.write_text(json.dumps(data, indent=2))
tmp.rename(cp_path)

print(f'Checkpoint updated: eliminated=False, force_deployed=True')
print(f'Strategy: {data.get(\"strategy_name\")} | {data.get(\"pair\")} | {data.get(\"timeframe\")}')
"
```

Also update the report.json if it exists:

```bash
uv run python -c "
import json
from pathlib import Path

rp_path = Path('results/RUN_DIR/report.json')
if not rp_path.exists():
    print('No report.json to update')
else:
    data = json.loads(rp_path.read_text())
    if data.get('candidates'):
        c = data['candidates'][0]
        c['eliminated'] = False
        c['force_deployed'] = True
        c['force_deploy_note'] = 'Manually force-deployed by user. Original elimination: ELIM_REASON'
    tmp = rp_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(rp_path)
    print('Report updated.')
"
```

## Step 4: Verify live_trade.py Strategy Import

The `scripts/live_trade.py` has hardcoded strategy imports at the top. Check if the strategy being deployed is imported:

```bash
grep -n "import.*STRATEGY_MODULE" scripts/live_trade.py
```

If the strategy isn't imported in `live_trade.py`, add the import. For example, for `ema_crossover`:
```python
import backtester.strategies.ema_crossover  # noqa: F401
```

This is critical — without this import, the strategy won't be in the registry and `start_all.py` will fail to launch it.

## Step 5: Git Commit + Push

Stage the relevant files, commit, and push:

```bash
# Stage the results files
git add results/RUN_DIR/checkpoint.json results/RUN_DIR/report.json

# Also stage live_trade.py if we modified it
git diff --name-only scripts/live_trade.py | head -1 && git add scripts/live_trade.py

# Check what we're committing
git status
git diff --cached --stat
```

Then commit with a descriptive message:

```bash
git commit -m "$(cat <<'EOF'
Deploy STRATEGY PAIR TF to MODE trading

[Force-deployed: original elimination at STAGE (REASON)]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Then push:

```bash
git push origin master
```

## Step 6: VPS Instructions

After pushing, tell the user:

```
Pushed to GitHub. To start trading on VPS:

1. RDP/SSH into your VPS
2. Open the ClaudeBackTester folder
3. Double-click DEPLOY.bat (or run it from terminal)

DEPLOY.bat will:
  - git pull (gets the checkpoint we just pushed)
  - Check dependencies
  - Launch: STRATEGY / PAIR / TF in MODE mode
  - Fixed lots: 0.01 (micro lot)

The trader runs as a background process. Check status with:
  - Logs: state/RUN_DIR/trader.log
  - STATUS.bat (if it exists)

To stop: STOP.bat or kill the python process
```

If mode is `live`, add an extra warning:
```
LIVE MODE: The VPS will ask you to type "LIVE" to confirm.
Real money will be at risk. Make sure you understand the strategy
and its risk parameters before confirming.
```

## Step 7: Post-Deploy Checklist

Remind the user of these checks after deploying:

- [ ] MT5 terminal is running on VPS with AutoTrading enabled (green button)
- [ ] MT5 is logged into the correct account (demo for practice, live for live)
- [ ] Internet connection on VPS is stable
- [ ] Check `trader.log` after 1-2 candle periods to confirm cycles are running
- [ ] Verify no errors in the log (especially MT5 connection or symbol issues)

## Important Notes

- `start_all.py` uses `--fixed-lots 0.01` (micro lot). This is safe for demo accounts
- The trader runs as a detached background process — closing the terminal won't stop it
- Each strategy gets its own state directory under `state/RUN_DIR/`
- State is persisted — restarting picks up where it left off (no duplicate orders)
- Circuit breakers: 3% daily loss limit, 10% max drawdown, 10 max daily trades
- Practice mode connects to MT5 demo account (credentials in .env on VPS)
