from __future__ import annotations

import sys
from pathlib import Path

# Ensure pytest can import application modules from the repository root on
# GitHub Actions, local Windows checkouts and IDE test runners.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
root = str(REPOSITORY_ROOT)
if root not in sys.path:
    sys.path.insert(0, root)
