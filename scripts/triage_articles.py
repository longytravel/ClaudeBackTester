"""Triage MQL5 articles using Claude Haiku for intelligent classification.

Classifies each article into: STRATEGY, MODULE, INDICATOR, SYSTEM, or SKIP
based on title + description, with full context about our trading system.

Usage:
    uv run python scripts/triage_articles.py [--batch-size 50] [--start 0]

Reads Research/mql5_catalogue.json, classifies via Haiku, writes back.
Checkpoints every batch so it can resume from where it left off.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = Path(__file__).parent.parent / "Research" / "mql5_catalogue.json"

SYSTEM_PROMPT = """You are a triage assistant for an automated forex trading research pipeline.

## Our System
We have a fully automated forex backtesting and live trading system built in Python + Rust. It:
- Downloads historical M1/H1 forex data (Dukascopy)
- Implements trading strategies as Python classes with vectorized signal generation
- Evaluates strategies via a Rust hot loop (PyO3 + Rayon) at 20K+ evals/sec
- Runs a staged optimizer (Sobol + EDA + MAP-Elites) across signal, time, risk, and management param groups
- Validates via walk-forward, CPCV, Monte Carlo, stability, regime analysis
- Deploys to MetaTrader 5 for live trading on IC Markets

## What We Need from Articles
We're scanning MQL5.com articles to find content we can USE. The articles are written in MQL4/MQL5 but we translate the LOGIC into our Python system. We don't care about MQL5 code syntax — we care about the trading IDEAS and mathematical CONCEPTS.

## Classification Categories

STRATEGY — The article describes a tradeable system with clear entry/exit rules that we could implement as a strategy class. This includes:
- Signal-based strategies (crossovers, breakouts, reversals, divergences, patterns)
- Complete trading systems with defined rules
- ML/AI-based prediction models that generate trade signals
- Multi-indicator systems combining signals
- Price action systems (order blocks, supply/demand, FVG, Wyckoff, ICT concepts)
- Statistical arbitrage or mean-reversion systems
- Strategies described in series (e.g., "Part 3") still count if they contain implementable logic

MODULE — The article describes a trade MANAGEMENT technique (not entry signals) that we could add as a reusable module. This includes:
- Trailing stop variants (ATR-based, chandelier, parabolic, step-based)
- Breakeven mechanisms
- Partial close / scaling out techniques
- Position sizing methods (Kelly, optimal-f, fixed fractional)
- Risk management systems (drawdown limits, circuit breakers, equity curves)
- Money management approaches
- Dynamic stop-loss or take-profit calculation methods
- Trade discipline enforcement

INDICATOR — The article describes a new technical indicator or a novel computation on price/volume data that we could add to our indicator library. This includes:
- Custom oscillators or trend indicators
- Novel uses of existing indicators
- Volume-based indicators
- Volatility measures
- Market structure indicators (swing detection, fractals)
- Composite/hybrid indicators

SYSTEM — The article describes infrastructure, methodology, or analytical techniques that could improve our backtesting/optimization/validation pipeline. This includes:
- Walk-forward analysis, cross-validation methods
- Overfitting detection (deflated Sharpe, Monte Carlo)
- Optimization algorithms (genetic, swarm, Bayesian)
- Statistical testing (stationarity, correlation, regime detection)
- Data pipeline techniques
- Portfolio construction
- Performance metrics and analysis methods
- Quantitative research methodology

SKIP — The article is NOT useful for our system. This includes:
- MQL5/MQL4 language tutorials (syntax, OOP, classes, libraries)
- MetaTrader platform-specific features (UI, charts, panels, dialogs, 3D graphics)
- Integration with external services (Telegram bots, web requests, databases, APIs)
- Interviews, marketplace guides, freelancing
- MQL5-specific frameworks (DoEasy, standard library internals)
- Content that is purely about MQL5 programming with no trading logic
- Copy trading, signal services, VPS setup

## Important Nuances
- An article about "building an EA" IS a strategy if it has clear trading rules
- An article about "MQL5 Wizard patterns" IS a strategy if it describes indicator-based signals
- A "machine learning" article IS a strategy if it generates trade signals, but SKIP if it's just demonstrating ML in MQL5
- A "risk management" article IS a module, not a strategy
- A "custom indicator" article IS an indicator, even if shown inside an EA
- Series articles (Part 1, Part 2, etc.) — classify based on what THIS part covers
- When in doubt between STRATEGY and INDICATOR, choose STRATEGY if it has entry/exit rules

## Response Format
Reply with ONLY the category word: STRATEGY, MODULE, INDICATOR, SYSTEM, or SKIP
Nothing else. Just the single word."""


def classify_batch(
    client: anthropic.Anthropic,
    articles: list[dict],
) -> list[str]:
    """Classify a batch of articles using individual Haiku calls."""
    results = []
    for article in articles:
        prompt = f"Title: {article['title']}\nDescription: {article['description'][:400]}"

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            category = response.content[0].text.strip().upper()
            if category not in {"STRATEGY", "MODULE", "INDICATOR", "SYSTEM", "SKIP"}:
                category = "SKIP"
            results.append(category)
        except Exception as e:
            print(f"    ERROR on [{article['id']}]: {e}")
            results.append("SKIP")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Triage MQL5 articles via Haiku")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Articles per checkpoint save")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (for resuming)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max articles to process (0 = all)")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add to .env or environment.")

    client = anthropic.Anthropic(api_key=api_key)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    articles = data["articles"]
    total = len(articles)

    # Determine start point — skip already-classified articles if resuming
    start = args.start
    if start == 0:
        # Auto-detect: find first unclassified article
        for i, a in enumerate(articles):
            if not a.get("category") or a["category"] == "new" or a["category"] == "":
                start = i
                break
        else:
            print("All articles already classified!")
            _print_summary(articles)
            return

    end = total if args.limit == 0 else min(start + args.limit, total)
    to_process = articles[start:end]

    print(f"Classifying articles {start}-{end-1} of {total} via Haiku...")
    print(f"Batch size: {args.batch_size}, checkpointing to {INPUT_PATH}")
    print()

    classified = 0
    t0 = time.time()

    for batch_start in range(0, len(to_process), args.batch_size):
        batch = to_process[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(to_process) + args.batch_size - 1) // args.batch_size

        print(f"Batch {batch_num}/{total_batches} "
              f"(articles {start + batch_start}-{start + batch_start + len(batch) - 1})...",
              end=" ", flush=True)

        categories = classify_batch(client, batch)

        # Update articles in-place
        for i, cat in enumerate(categories):
            articles[start + batch_start + i]["category"] = cat
            classified += 1

        # Count categories in this batch
        batch_counts = {}
        for cat in categories:
            batch_counts[cat] = batch_counts.get(cat, 0) + 1
        counts_str = " ".join(f"{k}:{v}" for k, v in sorted(batch_counts.items()))
        elapsed = time.time() - t0
        rate = classified / elapsed if elapsed > 0 else 0
        print(f"done [{counts_str}] ({rate:.1f} articles/sec)")

        # Checkpoint save
        with open(INPUT_PATH, "w") as f:
            json.dump(data, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nClassified {classified} articles in {elapsed:.0f}s "
          f"({classified/elapsed:.1f}/sec)")
    print()
    _print_summary(articles)


def _print_summary(articles: list[dict]) -> None:
    """Print classification summary."""
    counts: dict[str, int] = {}
    for a in articles:
        cat = a.get("category", "UNKNOWN")
        counts[cat] = counts.get(cat, 0) + 1

    total = len(articles)
    print(f"=== TRIAGE SUMMARY ({total} articles) ===\n")
    for cat in ["STRATEGY", "MODULE", "INDICATOR", "SYSTEM", "SKIP", "new", ""]:
        if cat in counts:
            label = cat if cat else "UNCLASSIFIED"
            pct = 100 * counts[cat] / total
            print(f"  {label:14s}  {counts[cat]:5d}  ({pct:4.1f}%)")

    actionable = counts.get("STRATEGY", 0) + counts.get("MODULE", 0) + \
                 counts.get("INDICATOR", 0) + counts.get("SYSTEM", 0)
    print(f"\n  {'ACTIONABLE':14s}  {actionable:5d}")
    print(f"  {'SKIP':14s}  {counts.get('SKIP', 0):5d}")


if __name__ == "__main__":
    main()
