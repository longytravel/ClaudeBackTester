"""Smoke tests to verify the project setup works."""

import numpy as np


def test_numba_jit_compiles():
    """Verify Numba JIT compilation works."""
    from numba import njit

    @njit
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_numba_parallel_prange():
    """Verify Numba parallel=True with prange and TBB works."""
    import numba
    from numba import njit, prange

    @njit(parallel=True)
    def parallel_sum(arr, results):
        for i in prange(len(arr)):
            results[i] = arr[i] * 2

    arr = np.arange(1000, dtype=np.float64)
    results = np.empty(1000, dtype=np.float64)
    parallel_sum(arr, results)
    np.testing.assert_array_equal(results, arr * 2)
    assert numba.threading_layer() == "tbb"


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
