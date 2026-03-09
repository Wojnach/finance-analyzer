"""Entry point for: python -m portfolio.golddigger"""

import argparse
import logging
import sys

from portfolio.golddigger.runner import run


def main():
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
        logging.getLogger().info("Starting in LIVE mode — real orders will be placed!")

    run(live=live, once=args.once)


if __name__ == "__main__":
    main()
