#!/bin/bash
# Build the Rust native extension for backtester_core.
# Run from the rust/ directory: bash build.sh
#
# Prerequisites:
#   - Rust toolchain: rustup (https://rustup.rs)
#   - MSVC Build Tools: VS Build Tools 2022 with C++ workload
#   - maturin: uv pip install maturin

set -e

# MSVC paths (VS Build Tools 2022)
MSVC_ROOT="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC"
MSVC_VER=$(ls "$MSVC_ROOT" | sort -V | tail -1)
WINSDK_ROOT="/c/Program Files (x86)/Windows Kits/10"
WINSDK_VER=$(ls "$WINSDK_ROOT/Lib" | sort -V | tail -1)

export PATH="$MSVC_ROOT/$MSVC_VER/bin/Hostx64/x64:$WINSDK_ROOT/bin/$WINSDK_VER/x64:$HOME/.cargo/bin:$(dirname "$(which python 2>/dev/null || echo /c/Users/$USER/Projects/ClaudeBackTester/.venv/Scripts/python.exe)"):$PATH"
export LIB="$MSVC_ROOT/$MSVC_VER/lib/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/ucrt/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/um/x64"
export INCLUDE="$MSVC_ROOT/$MSVC_VER/include;$WINSDK_ROOT/Include/$WINSDK_VER/ucrt;$WINSDK_ROOT/Include/$WINSDK_VER/um;$WINSDK_ROOT/Include/$WINSDK_VER/shared"

echo "MSVC version: $MSVC_VER"
echo "Windows SDK: $WINSDK_VER"

cd "$(dirname "$0")"
maturin develop --release
echo "Done! backtester_core installed."
