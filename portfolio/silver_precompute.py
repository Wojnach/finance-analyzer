"""Silver Deep Context Precomputer — delegation wrapper.

Delegates to the consolidated metals_precompute module.
Kept for backwards compatibility (manual runs, existing references).

Run: .venv/Scripts/python.exe portfolio/silver_precompute.py
Or:  .venv/Scripts/python.exe portfolio/metals_precompute.py  (preferred)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.file_utils import load_json


def maybe_precompute_silver(config=None):
    """Delegate to consolidated metals precompute."""
    from portfolio.metals_precompute import maybe_precompute_metals
    return maybe_precompute_metals(config)


def precompute(config=None):
    """Delegate to consolidated metals precompute."""
    from portfolio.metals_precompute import precompute as metals_precompute
    return metals_precompute(config)


if __name__ == "__main__":
    from portfolio.metals_precompute import precompute as metals_precompute

    config = load_json("config.json")
    metals_precompute(config)
