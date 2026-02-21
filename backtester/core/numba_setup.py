"""Numba + TBB bootstrap module.

MUST be imported before any other module that uses Numba parallel features.
Ensures TBB DLLs are discoverable on Windows.
"""

import os
import sys


def _setup_tbb():
    """Add TBB DLL directory to Windows DLL search path."""
    if sys.platform != "win32":
        return

    # TBB DLLs are installed by pip to {venv}/Library/bin/
    tbb_dir = os.path.join(sys.prefix, "Library", "bin")
    if os.path.exists(tbb_dir):
        os.add_dll_directory(tbb_dir)
        # Also add to PATH for any subprocess or ctypes fallback
        os.environ["PATH"] = tbb_dir + os.pathsep + os.environ.get("PATH", "")


# Run on import
_setup_tbb()
