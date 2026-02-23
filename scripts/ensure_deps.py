"""Auto-install dependencies when requirements.txt changes.

Stores an MD5 hash of requirements.txt in .installed.
Only runs pip when the hash differs (= new packages added).
"""

import hashlib
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    req_file = root / "requirements.txt"
    marker = root / ".installed"

    if not req_file.exists():
        print("  No requirements.txt found, skipping.")
        return

    # Hash current requirements.txt
    current_hash = hashlib.md5(req_file.read_bytes()).hexdigest()

    # Compare to last installed hash
    if marker.exists():
        last_hash = marker.read_text().strip()
        if last_hash == current_hash:
            print("  Dependencies up to date.")
            return

    print("  Installing/updating dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root)],
                   check=False)

    # Save hash
    marker.write_text(current_hash)
    print("  Done.")


if __name__ == "__main__":
    main()
