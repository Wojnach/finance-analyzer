"""Backward-compatible wrapper for the Fin Snipe entry point."""

from portfolio.fin_snipe import main


if __name__ == "__main__":
    raise SystemExit(main())
