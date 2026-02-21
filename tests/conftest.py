# Bootstrap TBB before any test imports Numba parallel features
import backtester.core.numba_setup  # noqa: F401
