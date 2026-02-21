"""Config validation for portfolio system startup.

Validates config.json has all required keys before the main loop starts.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("portfolio.config_validator")

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

# Required: missing any of these is a fatal error
REQUIRED_KEYS = [
    ("telegram", "token"),
    ("telegram", "chat_id"),
    ("alpaca", "key"),
    ("alpaca", "secret"),
]

# Optional: missing these produces a warning but isn't fatal
OPTIONAL_KEYS = [
    ("mistral_api_key",),
    ("iskbets",),
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
    if not CONFIG_FILE.exists():
        raise ValueError(f"config.json not found at {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Check optional keys and warn
    for key_path in OPTIONAL_KEYS:
        obj = config
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                logger.warning(f"optional config key missing: {'.'.join(key_path)}")
                break
            obj = obj[key]

    # Check required keys
    errors = validate_config(config)
    if errors:
        for err in errors:
            logger.error(f"config validation: {err}")
        raise ValueError(f"config.json validation failed: {'; '.join(errors)}")

    logger.info("config.json validated successfully")
    return config
