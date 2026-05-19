"""Thin wrapper for the Fin Snipe Manager entry point."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio.fin_snipe_manager import main


if __name__ == "__main__":
    raise SystemExit(main())
