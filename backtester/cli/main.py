import os

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Automated Forex Trading System CLI."""
    pass


@cli.command()
def info():
    """Show system information and configuration."""
    import platform
    import sys

    import numba

    click.echo(f"Python:  {sys.version}")
    click.echo(f"Numba:   {numba.__version__}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"CPU cores (logical): {numba.config.NUMBA_NUM_THREADS}")
    click.echo(f"Threading layer: {numba.config.THREADING_LAYER}")


# ---------- Data commands ----------

@cli.group()
def data():
    """Historical data management (download, update, status)."""
    pass


@data.command()
@click.option("--pair", "-p", default=None, help="Single pair to download (e.g. EUR/USD). Omit for all pairs.")
@click.option("--data-dir", "-d", default=None, help="Data directory override.")
@click.option("--start-year", default=2005, help="First year to download.", show_default=True)
@click.option("--force", is_flag=True, help="Re-download all years, even if cached.")
def download(pair, data_dir, start_year, force):
    """Download M1 data from Dukascopy for one or all pairs."""
    from backtester.data.downloader import (
        ALL_PAIRS,
        DEFAULT_DATA_DIR,
        download_all_pairs,
        download_pair,
    )

    data_dir = data_dir or DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    if pair:
        click.echo(f"Downloading {pair} from {start_year}...")
        path = download_pair(pair, data_dir, start_year, force)
        click.echo(f"Done: {path}")
    else:
        click.echo(f"Downloading all {len(ALL_PAIRS)} pairs from {start_year}...")
        results = download_all_pairs(ALL_PAIRS, data_dir, start_year, force)
        click.echo(f"Done: {len(results)}/{len(ALL_PAIRS)} pairs downloaded.")


@data.command()
@click.option("--pair", "-p", default=None, help="Single pair to update. Omit for all pairs.")
@click.option("--data-dir", "-d", default=None, help="Data directory override.")
def update(pair, data_dir):
    """Incremental update: fetch only new data since last download."""
    from backtester.data.downloader import ALL_PAIRS, DEFAULT_DATA_DIR, update_pair

    data_dir = data_dir or DEFAULT_DATA_DIR

    pairs = [pair] if pair else ALL_PAIRS
    for p in pairs:
        click.echo(f"Updating {p}...")
        try:
            path = update_pair(p, data_dir)
            click.echo(f"  Updated: {path}")
        except FileNotFoundError:
            click.echo(f"  No existing data for {p}. Use 'bt data download -p {p}' first.")


@data.command()
@click.option("--data-dir", "-d", default=None, help="Data directory override.")
def status(data_dir):
    """Show status of all cached data files."""
    from pathlib import Path

    import pandas as pd

    from backtester.data.downloader import ALL_PAIRS, DEFAULT_DATA_DIR

    data_dir = data_dir or DEFAULT_DATA_DIR
    data_path = Path(data_dir)

    if not data_path.exists():
        click.echo(f"Data directory not found: {data_dir}")
        return

    click.echo(f"Data directory: {data_dir}\n")
    click.echo(f"{'Pair':<12} {'M1 Rows':>12} {'From':>14} {'To':>14} {'Size MB':>10} {'TFs':>20}")
    click.echo("-" * 88)

    for pair in ALL_PAIRS:
        pair_file = pair.replace("/", "_")
        m1 = data_path / f"{pair_file}_M1.parquet"

        if not m1.exists():
            click.echo(f"{pair:<12} {'---':>12} {'---':>14} {'---':>14} {'---':>10} {'---':>20}")
            continue

        try:
            df = pd.read_parquet(m1)
            rows = len(df)
            first = df.index[0].strftime("%Y-%m-%d") if rows else "---"
            last = df.index[-1].strftime("%Y-%m-%d") if rows else "---"
            size = m1.stat().st_size / 1024 / 1024
        except Exception:
            rows = 0
            first = last = "ERROR"
            size = 0

        # Check which higher TFs exist
        tfs = []
        for tf in ["M5", "M15", "M30", "H1", "H4", "D", "W"]:
            if (data_path / f"{pair_file}_{tf}.parquet").exists():
                tfs.append(tf)

        tf_str = ",".join(tfs) if tfs else "none"
        click.echo(f"{pair:<12} {rows:>12,} {first:>14} {last:>14} {size:>10.1f} {tf_str:>20}")


@data.command()
@click.option("--pair", "-p", default=None, help="Single pair. Omit for all pairs with M1 data.")
@click.option("--data-dir", "-d", default=None, help="Data directory override.")
def build_timeframes(pair, data_dir):
    """Build higher timeframes (M5, M15, M30, H1, H4, D, W) from M1 data."""
    from pathlib import Path

    from backtester.data.downloader import ALL_PAIRS, DEFAULT_DATA_DIR
    from backtester.data.timeframes import convert_timeframes

    data_dir = data_dir or DEFAULT_DATA_DIR

    if pair:
        pairs = [pair]
    else:
        # Find all pairs that have M1 data
        pairs = []
        for p in ALL_PAIRS:
            m1 = Path(data_dir) / f"{p.replace('/', '_')}_M1.parquet"
            if m1.exists():
                pairs.append(p)

    for p in pairs:
        click.echo(f"Building timeframes for {p}...")
        try:
            results = convert_timeframes(p, data_dir)
            click.echo(f"  Built: {', '.join(results.keys())}")
        except Exception as e:
            click.echo(f"  Error: {e}")


@data.command()
@click.option("--pair", "-p", required=True, help="Pair to validate.")
@click.option("--timeframe", "-t", default="M1", help="Timeframe to validate.", show_default=True)
@click.option("--data-dir", "-d", default=None, help="Data directory override.")
def validate(pair, timeframe, data_dir):
    """Validate data quality for a pair/timeframe."""
    from pathlib import Path

    import pandas as pd

    from backtester.data.downloader import DEFAULT_DATA_DIR
    from backtester.data.validation import validate_data

    data_dir = data_dir or DEFAULT_DATA_DIR
    pair_file = pair.replace("/", "_")
    path = Path(data_dir) / f"{pair_file}_{timeframe}.parquet"

    if not path.exists():
        click.echo(f"File not found: {path}")
        return

    df = pd.read_parquet(path)
    result = validate_data(df, timeframe)

    click.echo(f"\nValidation: {pair} {timeframe}")
    click.echo(f"  Quality Score: {result.get('quality_score', 0)}/100")
    click.echo(f"  Passed: {result.get('passed', False)}")
    click.echo(f"  Candles: {result.get('total_candles', 0):,}")
    click.echo(f"  Date Range: {result.get('date_range', '---')}")

    if "gaps" in result:
        g = result["gaps"]
        click.echo(f"  Gaps: {g['unexpected_gaps']} unexpected, {g['weekend_gaps']} weekend")
    if "zeros_nans" in result:
        z = result["zeros_nans"]
        click.echo(f"  Zeros: {z['zero_count']}, NaNs: {z['nan_count']}")
    if "anomalies" in result:
        a = result["anomalies"]
        click.echo(f"  Anomalies: {a['extreme_range_candles']} extreme, {a['ohlc_violations']} violations")

    if not result.get("passed"):
        click.echo(f"\n  FAILED: {result.get('reason', 'unknown')}")


if __name__ == "__main__":
    cli()
