"""Data splitting: back-test / forward-test and holdout modes.

Splits time-series data chronologically (never randomly) for
optimization and validation stages.
"""

from datetime import timedelta

import pandas as pd
import structlog

log = structlog.get_logger()


def split_backforward(
    df: pd.DataFrame,
    back_pct: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into back-test and forward-test portions by row count.

    Default 80/20 split. Always chronological (earlier = back, later = forward).
    """
    if df.empty:
        return df, df

    split_idx = int(len(df) * back_pct)
    back = df.iloc[:split_idx]
    forward = df.iloc[split_idx:]

    log.info(
        "split_backforward",
        back_rows=len(back),
        forward_rows=len(forward),
        split_date=str(df.index[split_idx]) if split_idx < len(df) else "end",
        back_pct=back_pct,
    )
    return back, forward


def split_holdout(
    df: pd.DataFrame,
    holdout_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the last N months as pure out-of-sample holdout.

    Returns (training_data, holdout_data).
    """
    if df.empty:
        return df, df

    last_ts = df.index[-1]
    cutoff = last_ts - pd.DateOffset(months=holdout_months)

    training = df[df.index < cutoff]
    holdout = df[df.index >= cutoff]

    log.info(
        "split_holdout",
        training_rows=len(training),
        holdout_rows=len(holdout),
        cutoff=str(cutoff),
        holdout_months=holdout_months,
    )
    return training, holdout


def split_data(
    df: pd.DataFrame,
    mode: str = "backforward",
    back_pct: float = 0.80,
    holdout_months: int = 3,
) -> dict[str, pd.DataFrame]:
    """Split data using the specified mode.

    Returns dict with 'back'/'forward' keys (or 'training'/'holdout').
    """
    if mode == "backforward":
        back, forward = split_backforward(df, back_pct)
        return {"back": back, "forward": forward}
    elif mode == "holdout":
        training, holdout = split_holdout(df, holdout_months)
        return {"back": training, "forward": holdout}
    else:
        raise ValueError(f"Unknown split mode: {mode}. Use 'backforward' or 'holdout'.")
