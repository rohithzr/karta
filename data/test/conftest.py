"""Pytest config — make ``data/`` importable so tests can ``import convert_beam``."""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
