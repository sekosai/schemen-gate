"""Keep tests and their child Python processes on this checkout's source tree."""

from __future__ import annotations

import os
from pathlib import Path

SOURCE = str(Path(__file__).resolve().parents[1] / "src")
inherited = os.environ.get("PYTHONPATH", "")
remaining = [entry for entry in inherited.split(os.pathsep) if entry and entry != SOURCE]
os.environ["PYTHONPATH"] = os.pathsep.join([SOURCE, *remaining])
