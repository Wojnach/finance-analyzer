"""Validates config.json structure at startup."""

import json
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

REQUIRED_KEYS = {
    "telegram": {"token": str, "chat_id": (str, int)},
    "exchange": {"apiKey": str, "secret": str},
}

OPTIONAL_KEYS = {
    "alpaca": {"key": str, "secret": str},
    "dashboard_token": str,
}


def validate_config(config_path=None):
    """Validate config.json structure. Returns list of error strings."""
    if config_path is None:
        config_path = BASE_DIR / "config.json"

    errors = []

    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return ["config.json not found"]
    except json.JSONDecodeError as e:
        return [f"config.json is not valid JSON: {e}"]

    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            errors.append(f"Missing required section: {section}")
            continue
        if isinstance(keys, dict):
            for key, expected_type in keys.items():
                if key not in config[section]:
                    errors.append(f"Missing required key: {section}.{key}")
                elif not config[section][key]:
                    errors.append(f"Empty required key: {section}.{key}")

    return errors


if __name__ == "__main__":
    errors = validate_config()
    if errors:
        print("Config validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Config validation passed.")
