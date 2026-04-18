"""Entry point: python -m portfolio.mstr_loop

Starts the main loop. The PHASE is read from portfolio.mstr_loop.config
(can be overridden with MSTR_LOOP_PHASE env var). Default: shadow.
"""

from __future__ import annotations

import logging
import sys


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    from portfolio.mstr_loop import loop
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("mstr_loop: stopped by KeyboardInterrupt")
        return 0
    except Exception:
        logging.exception("mstr_loop: fatal error — exiting")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
