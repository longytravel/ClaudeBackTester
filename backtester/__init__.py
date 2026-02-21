# Bootstrap TBB for Numba parallel support on Windows.
# This MUST run before any Numba parallel code is imported.
from backtester.core.numba_setup import _setup_tbb as _  # noqa: F401
