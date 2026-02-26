"""Smoke tests to verify the project setup works."""

import numpy as np


def test_rust_backend_imports():
    """Verify Rust native extension imports and exports constants."""
    import backtester_core

    assert hasattr(backtester_core, "batch_evaluate")
    assert backtester_core.NUM_METRICS == 10
    assert backtester_core.NUM_PL == 27


def test_parquet_roundtrip(tmp_path):
    """Verify Parquet read/write works."""
    import pandas as pd

    df = pd.DataFrame({
        "open": [1.1, 1.2, 1.3],
        "high": [1.2, 1.3, 1.4],
        "low": [1.0, 1.1, 1.2],
        "close": [1.15, 1.25, 1.35],
        "volume": [100, 200, 300],
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    df2 = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, df2)


def test_cli_exists():
    """Verify CLI entry point imports."""
    from backtester.cli.main import cli

    assert cli is not None
