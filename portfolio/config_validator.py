"""Config validation for portfolio system startup.

Validates config.json has all required keys before the main loop starts.
"""

import logging
from pathlib import Path

from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.config_validator")

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

# Required: missing any of these is a fatal error.
# 2026-05-11 fix: Binance creds actually live under ``exchange.key`` /
# ``exchange.secret`` (freqtrade-style config layout, with
# ``exchange.name = "binance"``) — they have never been under a
# ``binance`` top-level section. The original list referencing
# ``binance.key`` / ``binance.secret`` was blocking main-loop startup on
# fresh restarts because validation always failed against the real
# config shape.
REQUIRED_KEYS = [
    ("telegram", "token"),
    ("telegram", "chat_id"),
    ("alpaca", "key"),
    ("alpaca", "secret"),
    ("exchange", "key"),
    ("exchange", "secret"),
]

# Optional: missing these produces a warning but isn't fatal
OPTIONAL_KEYS = [
    ("mistral_api_key",),
    ("iskbets",),
    ("newsapi_key",),
    ("alpha_vantage", "api_key"),
    ("golddigger", "fred_api_key"),
    ("bgeometrics", "api_token"),
]


def validate_config(config: dict) -> list[str]:
    """Validate config dict. Returns list of error strings (empty = valid)."""
    errors = []
    for key_path in REQUIRED_KEYS:
        obj = config
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                errors.append(f"missing required key: {'.'.join(key_path)}")
                break
            obj = obj[key]
        else:
            # Key exists — check it's not empty/placeholder
            if isinstance(obj, str) and not obj.strip():
                errors.append(f"empty value for required key: {'.'.join(key_path)}")
    return errors


def validate_config_file() -> dict:
    """Load config.json, validate, and return it.

    Logs warnings for missing optional keys.
    Raises ValueError if required keys are missing.
    """
    config = load_json(CONFIG_FILE)
    if config is None:
        raise ValueError(f"config.json not found or unreadable at {CONFIG_FILE}")

    # Check optional keys and warn
    for key_path in OPTIONAL_KEYS:
        obj = config
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                logger.warning("optional config key missing: %s", '.'.join(key_path))
                break
            obj = obj[key]

    # Check required keys
    errors = validate_config(config)
    if errors:
        for err in errors:
            logger.error("config validation: %s", err)
        raise ValueError(f"config.json validation failed: {'; '.join(errors)}")

    logger.info("config.json validated successfully")
    return config
