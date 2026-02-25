/// Numeric constants mirrored from backtester/core/dtypes.py.
/// Must stay in sync — any change in dtypes.py must be reflected here.

// Direction codes
pub const DIR_BUY: i64 = 1;
pub const DIR_SELL: i64 = -1;

// SL mode codes
pub const SL_FIXED_PIPS: i64 = 0;
pub const SL_ATR_BASED: i64 = 1;
pub const SL_SWING: i64 = 2;

// TP mode codes
pub const TP_RR_RATIO: i64 = 0;
pub const TP_ATR_BASED: i64 = 1;
pub const TP_FIXED_PIPS: i64 = 2;

// Trailing mode codes
pub const TRAIL_OFF: i64 = 0;
pub const TRAIL_FIXED_PIP: i64 = 1;
pub const TRAIL_ATR_CHANDELIER: i64 = 2;

// Exit reason codes
pub const EXIT_NONE: i64 = 0;
pub const EXIT_SL: i64 = 1;
pub const EXIT_TP: i64 = 2;
pub const EXIT_TRAILING: i64 = 3;
pub const EXIT_BREAKEVEN: i64 = 4;
pub const EXIT_MAX_BARS: i64 = 5;
pub const EXIT_STALE: i64 = 6;

// Execution mode codes
pub const EXEC_BASIC: i64 = 0;
pub const EXEC_FULL: i64 = 1;

// Metric column indices in the output matrix (N, NUM_METRICS)
pub const M_TRADES: usize = 0;
pub const M_WIN_RATE: usize = 1;
pub const M_PROFIT_FACTOR: usize = 2;
pub const M_SHARPE: usize = 3;
pub const M_SORTINO: usize = 4;
pub const M_MAX_DD_PCT: usize = 5;
pub const M_RETURN_PCT: usize = 6;
pub const M_R_SQUARED: usize = 7;
pub const M_ULCER: usize = 8;
pub const M_QUALITY: usize = 9;
pub const NUM_METRICS: usize = 10;

// Parameter layout indices — must match jit_loop.py PL_* constants
pub const PL_SL_MODE: usize = 0;
pub const PL_SL_FIXED_PIPS: usize = 1;
pub const PL_SL_ATR_MULT: usize = 2;
pub const PL_TP_MODE: usize = 3;
pub const PL_TP_RR_RATIO: usize = 4;
pub const PL_TP_ATR_MULT: usize = 5;
pub const PL_TP_FIXED_PIPS: usize = 6;
pub const PL_HOURS_START: usize = 7;
pub const PL_HOURS_END: usize = 8;
pub const PL_DAYS_BITMASK: usize = 9;
pub const PL_TRAILING_MODE: usize = 10;
pub const PL_TRAIL_ACTIVATE: usize = 11;
pub const PL_TRAIL_DISTANCE: usize = 12;
pub const PL_TRAIL_ATR_MULT: usize = 13;
pub const PL_BREAKEVEN_ENABLED: usize = 14;
pub const PL_BREAKEVEN_TRIGGER: usize = 15;
pub const PL_BREAKEVEN_OFFSET: usize = 16;
pub const PL_PARTIAL_ENABLED: usize = 17;
pub const PL_PARTIAL_PCT: usize = 18;
pub const PL_PARTIAL_TRIGGER: usize = 19;
pub const PL_MAX_BARS: usize = 20;
pub const PL_STALE_ENABLED: usize = 21;
pub const PL_STALE_BARS: usize = 22;
pub const PL_STALE_ATR_THRESH: usize = 23;
pub const PL_SIGNAL_VARIANT: usize = 24;
pub const PL_BUY_FILTER_MAX: usize = 25;
pub const PL_SELL_FILTER_MIN: usize = 26;
pub const NUM_PL: usize = 27;
