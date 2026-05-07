"""Make `tests/` importable so smoke tests can share helpers via `from synth import ...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
