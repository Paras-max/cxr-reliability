"""Single-image demo page (Phase 12).

Responsibility:
    Delegates to the main single-image reliability dashboard application.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.dashboard.app import main

if __name__ == "__main__":
    main()
