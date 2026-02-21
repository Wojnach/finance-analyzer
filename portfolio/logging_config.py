"""Structured logging configuration for the finance-analyzer system.

Replaces print()-based logging with Python's logging module.
StreamHandler goes to stdout (captured by pf-loop.bat → loop_out.txt).
RotatingFileHandler writes to data/portfolio.log (10MB, 3 backups).
"""

import logging
import logging.handlers
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_configured = False


def setup_logging(level=logging.INFO):
    """Configure root logger with stream + rotating file handlers.

    Safe to call multiple times — only configures once.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("portfolio")
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # StreamHandler → stdout (same as print, captured by bat redirect)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # RotatingFileHandler → data/portfolio.log
    DATA_DIR.mkdir(exist_ok=True)
    log_path = DATA_DIR / "portfolio.log"
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
