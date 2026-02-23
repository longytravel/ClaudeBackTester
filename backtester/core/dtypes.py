"""Numeric constants for JIT-compiled backtest engine.

All codes are int64/float64 for Numba compatibility.
Extensibility: new exit types, SL/TP modes, or metric columns = add constants here.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Direction codes (match Direction enum values)
# ---------------------------------------------------------------------------
DIR_BUY: int = 1
DIR_SELL: int = -1

# ---------------------------------------------------------------------------
# SL mode codes (index into risk_params sl_mode values list)
# ---------------------------------------------------------------------------
SL_FIXED_PIPS: int = 0   # "fixed_pips"
SL_ATR_BASED: int = 1    # "atr_based"
SL_SWING: int = 2        # "swing"

# ---------------------------------------------------------------------------
# TP mode codes
# ---------------------------------------------------------------------------
TP_RR_RATIO: int = 0     # "rr_ratio"
TP_ATR_BASED: int = 1    # "atr_based"
TP_FIXED_PIPS: int = 2   # "fixed_pips"

# ---------------------------------------------------------------------------
# Trailing mode codes
# ---------------------------------------------------------------------------
TRAIL_OFF: int = 0        # "off"
TRAIL_FIXED_PIP: int = 1  # "fixed_pip"
TRAIL_ATR_CHANDELIER: int = 2  # "atr_chandelier"

# ---------------------------------------------------------------------------
# Exit reason codes
# ---------------------------------------------------------------------------
EXIT_NONE: int = 0         # Trade still open (should not appear in results)
EXIT_SL: int = 1           # Stop-loss hit
EXIT_TP: int = 2           # Take-profit hit
EXIT_TRAILING: int = 3     # Trailing stop hit
EXIT_BREAKEVEN: int = 4    # Breakeven stop hit
EXIT_MAX_BARS: int = 5     # Maximum bars reached
EXIT_STALE: int = 6        # Stale exit (no movement)
EXIT_PARTIAL: int = 7      # Partial close event (not a full exit)
# Future: EXIT_VOLATILITY = 11, EXIT_INDICATOR = 12, etc.

# ---------------------------------------------------------------------------
# Execution mode codes
# ---------------------------------------------------------------------------
EXEC_BASIC: int = 0   # SL/TP only — fast path for early optimization stages
EXEC_FULL: int = 1    # All management features enabled

# ---------------------------------------------------------------------------
# Metric column indices in the output matrix (N, NUM_METRICS)
# Order matters — indexed by these constants in JIT code
# ---------------------------------------------------------------------------
M_TRADES: int = 0
M_WIN_RATE: int = 1
M_PROFIT_FACTOR: int = 2
M_SHARPE: int = 3
M_SORTINO: int = 4
M_MAX_DD_PCT: int = 5
M_RETURN_PCT: int = 6
M_R_SQUARED: int = 7
M_ULCER: int = 8
M_QUALITY: int = 9

NUM_METRICS: int = 10

# ---------------------------------------------------------------------------
# Signal array column indices (for the signal matrix passed to JIT)
# ---------------------------------------------------------------------------
SIG_BAR_INDEX: int = 0
SIG_DIRECTION: int = 1
SIG_ENTRY_PRICE: int = 2
SIG_HOUR: int = 3
SIG_DAY_OF_WEEK: int = 4
SIG_ATR_PIPS: int = 5

NUM_SIG_COLS: int = 6  # Base columns; attrs follow after this

# ---------------------------------------------------------------------------
# Parameter column indices for the encoded param matrix
# These are dynamically assigned by encoding.py based on ParamSpace,
# but we define the STANDARD param names here for reference.
# The actual indices are stored in EncodingSpec.
# ---------------------------------------------------------------------------

# Bitmask encoding for allowed_days
# Mon=bit0, Tue=bit1, Wed=bit2, Thu=bit3, Fri=bit4, Sat=bit5, Sun=bit6
DAYS_BITMASK: dict[int, int] = {
    0: 1,    # Monday
    1: 2,    # Tuesday
    2: 4,    # Wednesday
    3: 8,    # Thursday
    4: 16,   # Friday
    5: 32,   # Saturday
    6: 64,   # Sunday
}
DAYS_MON_FRI: int = 31    # 0b0011111
DAYS_ALL: int = 127       # 0b1111111

# ---------------------------------------------------------------------------
# Default constants for simulation
# ---------------------------------------------------------------------------
DEFAULT_SPREAD_PIPS: float = 1.0
DEFAULT_SLIPPAGE_PIPS: float = 0.5
DEFAULT_PIP_VALUE: float = 0.0001  # Most pairs; JPY = 0.01

# Annualization factor for Sharpe/Sortino (assuming ~252 trading days)
TRADING_DAYS_PER_YEAR: int = 252

# Estimated bars per year for common timeframes (24h forex market, 252 trading days)
BARS_PER_YEAR: dict[str, float] = {
    "M1": 252 * 24 * 60,     # 362,880
    "M5": 252 * 24 * 12,     # 72,576
    "M15": 252 * 24 * 4,     # 24,192
    "M30": 252 * 24 * 2,     # 12,096
    "H1": 252 * 24,          # 6,048
    "H4": 252 * 6,           # 1,512
    "D": 252,                # 252
    "W": 52,                 # 52
}
DEFAULT_BARS_PER_YEAR: float = 6048.0  # H1 default
