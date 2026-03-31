import importlib.util
import io
import json
from collections import deque
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "data" / "silver_monitor.py"


def _load_silver_monitor():
    real_open = open
    config_payload = json.dumps({"telegram": {"token": "test-token", "chat_id": "test-chat"}})
    state_payload = json.dumps(
        {"silver_test": {"active": True, "entry": 90.55, "leverage": 4.76, "units": 0}}
    )

    def fake_open(file, mode="r", *args, **kwargs):
        path = Path(file)
        if path.name == "config.json":
            return io.StringIO(config_payload)
        if path.name == "metals_positions_state.json":
            return io.StringIO(state_payload)
        return real_open(file, mode, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("test_silver_monitor_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch("builtins.open", side_effect=fake_open):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def silver_monitor():
    return _load_silver_monitor()


def test_calc_velocity_pct_requires_full_window(silver_monitor):
    history = deque([100.0, 99.9, 99.8], maxlen=silver_monitor.VELOCITY_WINDOW)

    assert silver_monitor.calc_velocity_pct(history) is None


def test_velocity_alert_ignores_routine_three_minute_noise(silver_monitor):
    history = deque(
        [100.0] + [99.9] * (silver_monitor.VELOCITY_WINDOW - 2) + [99.6],
        maxlen=silver_monitor.VELOCITY_WINDOW,
    )

    assert silver_monitor.calc_velocity_pct(history) == pytest.approx(-0.4)
    assert silver_monitor.should_send_velocity_alert(history) is False


def test_velocity_alert_keeps_real_flushes_enabled(silver_monitor):
    history = deque(
        [100.0] + [99.4] * (silver_monitor.VELOCITY_WINDOW - 2) + [99.1],
        maxlen=silver_monitor.VELOCITY_WINDOW,
    )

    assert silver_monitor.calc_velocity_pct(history) == pytest.approx(-0.9)
    assert silver_monitor.should_send_velocity_alert(history) is True


def test_velocity_alert_mutes_only_telegram_path(silver_monitor):
    history = deque(
        [100.0] + [99.4] * (silver_monitor.VELOCITY_WINDOW - 2) + [99.1],
        maxlen=silver_monitor.VELOCITY_WINDOW,
    )

    assert silver_monitor.should_send_velocity_alert(history) is True
    assert silver_monitor.VELOCITY_TELEGRAM_ENABLED is False


def test_singleton_lock_round_trip(silver_monitor, tmp_path):
    lock_path = tmp_path / "silver_monitor.singleton.lock"

    silver_monitor.release_singleton_lock()
    assert silver_monitor.acquire_singleton_lock(lock_path) is True
    silver_monitor.release_singleton_lock()
    assert silver_monitor.acquire_singleton_lock(lock_path) is True
    silver_monitor.release_singleton_lock()


def test_main_exits_when_duplicate_instance_detected(silver_monitor, monkeypatch):
    silver_monitor.release_singleton_lock()
    monkeypatch.setattr(silver_monitor, "acquire_singleton_lock", lambda lock_path=silver_monitor.SINGLETON_LOCK_FILE: False)

    with pytest.raises(SystemExit) as exc:
        silver_monitor.main()

    assert exc.value.code == silver_monitor.DUPLICATE_INSTANCE_EXIT_CODE
