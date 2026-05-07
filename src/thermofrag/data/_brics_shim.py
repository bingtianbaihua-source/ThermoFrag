"""Shim to make the vendored BBAR fragmentation code importable.

The original BBAR code imports ``bbar.utils.typing`` and ``bbar.utils.common``,
but in the vendor tree those modules live under the flat package
``bbar_utils/``. Install aliases under ``sys.modules`` so the vendored code can
be imported unmodified.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def ensure_brics() -> None:
    """Register the ``bbar.utils.*`` aliases and put ``vendor/`` on sys.path."""
    if "bbar.utils.typing" in sys.modules:
        return
    vendor_dir = Path(__file__).resolve().parents[3] / "vendor"
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))

    import bbar_utils  # type: ignore  # noqa: F401
    import bbar_utils.typing  # type: ignore
    import bbar_utils.common  # type: ignore

    # Create a virtual ``bbar`` package and splice in ``.utils``.
    pkg_bbar = types.ModuleType("bbar")
    pkg_bbar.__path__ = []  # mark as package
    sys.modules["bbar"] = pkg_bbar

    sys.modules["bbar.utils"] = bbar_utils  # type: ignore[assignment]
    sys.modules["bbar.utils.typing"] = bbar_utils.typing  # type: ignore[assignment]
    sys.modules["bbar.utils.common"] = bbar_utils.common  # type: ignore[assignment]
