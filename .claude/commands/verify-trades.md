# /verify-trades — Backtest vs Live Trade Comparison

Compare live MT5 trades against backtest replay to find discrepancies in signal timing, entry/exit prices, SL/TP levels, exit reasons, and PnL.

## Usage
```
/verify-trades [STRATEGY PAIR TIMEFRAME]
```
All arguments are optional. If missing, ask the user interactively.

## Environment Setup (ALWAYS run first)

Every bash command MUST begin with:
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
```

## Step 1: Gather Inputs

### 1a. If ALL 3 args provided, skip to Step 2

### 1b. If any args missing, ask using AskUserQuestion

Ask ONLY the missing arguments:

1. **Strategy** — which strategy to verify. List deployed strategies:
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
ls state/ 2>/dev/null || echo "No state/ directory"
ls results/*/checkpoint.json 2>/dev/null || echo "No checkpoints found"
```

2. **Pair** — EURUSD, GBPUSD, USDJPY, etc.
3. **Timeframe** — H1, M15, etc.

## Step 2: Check for Live Trades

Before running the full verification, confirm trades exist. Check TWO sources:

### 2a. Check audit.jsonl
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
INSTANCE="STRATEGY_PAIR_TIMEFRAME"
AUDIT="state/$INSTANCE/audit.jsonl"
if [ -f "$AUDIT" ]; then
    echo "=== Audit file found ==="
    echo "Total events: $(wc -l < "$AUDIT")"
    echo "Orders placed: $(grep -c '"order_placed"' "$AUDIT")"
    echo "Positions closed: $(grep -c '"position_closed"' "$AUDIT")"
    echo ""
    echo "=== Last 5 events ==="
    tail -5 "$AUDIT" | python -m json.tool --no-ensure-ascii 2>/dev/null || tail -5 "$AUDIT"
else
    echo "No audit file at $AUDIT"
fi
```

### 2b. Check MT5 (if connected)
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python -c "
from backtester.broker import mt5
from backtester.broker.mt5_orders import get_closed_deals
ok = mt5.connect()
if not ok:
    print('MT5 not connected — will use audit.jsonl instead')
else:
    from backtester.live.trader import _deterministic_magic
    instance_id = 'STRATEGY_PAIR_TIMEFRAME'
    magic = _deterministic_magic(instance_id)
    deals = get_closed_deals(magic=magic)
    pair_deals = [d for d in deals if d['symbol'] == 'PAIR']
    print(f'Magic number: {magic}')
    print(f'Total deals for {\"PAIR\"}: {len(pair_deals)}')
    if pair_deals:
        print(f'First deal: {pair_deals[0][\"time\"]}')
        print(f'Last deal: {pair_deals[-1][\"time\"]}')
    mt5.disconnect()
"
```

### 2c. Report findings to user

Tell the user:
- How many trades were found (from audit and/or MT5)
- Date range of trades
- Ask: **"Ready to run verification? This will download the latest data and replay the backtest."**

If NO trades found at all, tell the user there's nothing to compare yet and suggest they wait for market open / check that traders are running.

## Step 3: Run Verification

Determine the best source (MT5 or audit) and run:

### Option A: Using audit.jsonl (no MT5 needed)
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python scripts/verify_trades.py \
    --strategy STRATEGY \
    --pair PAIR \
    --timeframe TIMEFRAME \
    --checkpoint results/STRATEGY_pair_tf/checkpoint.json \
    --audit state/STRATEGY_PAIR_TIMEFRAME/audit.jsonl
```

### Option B: Using MT5 directly
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python scripts/verify_trades.py \
    --strategy STRATEGY \
    --pair PAIR \
    --timeframe TIMEFRAME \
    --checkpoint results/STRATEGY_pair_tf/checkpoint.json
```

### Option C: Verify ALL deployed strategies
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python scripts/verify_trades.py --all
```

Run with `timeout: 300000` (5 min) — data download can take time.

## Step 4: Interpret Results

Read the output and explain to the user:

### Key Metrics
- **Matched trades**: How many live trades matched a backtest trade
- **Unmatched live**: Trades that happened live but NOT in backtest (signal divergence!)
- **Unmatched backtest**: Trades in backtest but NOT live (missed signals)
- **Entry slippage**: How much worse the live fill was vs backtest (expect 0.1-0.5 pips)
- **PnL delta**: Difference in profit between live and backtest per trade
- **Exit reason match rate**: Should be >80% for a healthy system

### Common Divergence Causes
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Unmatched live trades | Signal filter mismatch between Rust engine and Python `_filter_vectorized_signals()` | Check variant/filter_value mapping in trader.py |
| Unmatched backtest trades | Data source difference (Dukascopy vs IC Markets) | Check price at signal bar — does it differ? |
| Large entry slippage (>1 pip) | High spread at entry, or slippage_pips too low in backtest | Increase backtest `slippage_pips` or add spread filter |
| Exit reason mismatch | Management logic difference (position_manager.py vs telemetry.py) | Check deferred SL, BE trigger, trailing activation |
| SL/TP price mismatch | ATR computed differently (different lookback data) | Check indicator warmup period |
| Time offset in matches | MT5 server time (UTC+2) not properly converted | Check timezone handling in trader.py |
| All trades unmatched | Wrong magic number or data doesn't cover trade period | Verify `--magic` flag or check data freshness |

### Action Items
After reviewing, suggest concrete fixes:
1. If entry slippage consistently >0.5 pips → increase `slippage_pips` in backtest config
2. If exit reasons diverge → compare position_manager.py vs telemetry.py logic
3. If trades are missing → check signal filtering params match exactly
4. If PnL deltas are large → check spread/commission assumptions vs actual

## Step 5: Save and Follow Up

The report is automatically saved to `results/{strategy}_{pair}_{tf}/verification_*.json`.

Tell the user where the report was saved and ask:
- **"Want to investigate any specific diverged trade in detail?"**
- **"Should I adjust any backtest parameters based on these findings?"**

## Notes
- Data download may take 1-2 minutes if the current year needs refreshing
- MT5 must have "Max Bars in Chart" set high (Tools > Options > Charts > 10,000,000+)
- MT5 server time is UTC+2 — all timestamps are converted to UTC for comparison
- The verification uses EXEC_FULL mode (all management features) for telemetry replay
