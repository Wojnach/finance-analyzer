"""Entry point for: python -m portfolio.golddigger"""

import argparse
import logging
import sys

from portfolio.golddigger.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="GoldDigger — Intraday gold certificate trading bot")
    parser.add_argument("--live", action="store_true", help="Live execution via Avanza (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Paper trade (default)")
    parser.add_argument("--once", action="store_true", help="Run a single poll cycle and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    live = args.live
    if live:
        logging.getLogger().info(
            "Starting in live request mode; runner will resolve LIVE vs SIGNAL-ONLY"
        )

    # Propagate exit code so scripts/win/golddigger-loop.bat can short-circuit
    # restarts on EXIT_LOCK_CONFLICT (11) — matches the pattern used by
    # data/crypto_loop.py + data/oil_loop.py.
    return run(live=live, once=args.once) or 0


if __name__ == "__main__":
    sys.exit(main())
