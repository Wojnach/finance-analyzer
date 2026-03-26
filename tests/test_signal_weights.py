"""Tests for portfolio.signal_weights.SignalWeightManager.

Covers:
  1. Initial weights default to 1.0
  2. Correct prediction increases weight
  3. Wrong prediction decreases weight
  4. Weight floor (never below 0.01)
  5. Weights persist to disk and reload correctly
  6. Batch update works and auto-saves
  7. Normalised weights average to 1.0
  8. Configurable learning rate (eta)
  9. Multiple updates compound correctly
 10. get_normalized_weights with empty / single signal
"""

import json
import math

import pytest

from portfolio.signal_weights import SignalWeightManager, _WEIGHT_FLOOR, _DEFAULT_ETA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mgr(tmp_path, eta=None):
    path = tmp_path / "signal_weights.json"
    return SignalWeightManager(path=path, eta=eta)


# ===========================================================================
# 1. Default weights
# ===========================================================================

class TestDefaultWeights:
    def test_unknown_signal_returns_one(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        assert mgr.get_weight("rsi") == 1.0

    def test_all_unknown_signals_return_one(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        for name in ("rsi", "macd", "ema", "bb", "volume", "ministral"):
            assert mgr.get_weight(name) == 1.0


# ===========================================================================
# 2. Correct prediction increases weight
# ===========================================================================

class TestCorrectUpdate:
    def test_correct_increases_weight(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        new_w = mgr.update("rsi", correct=True)
        assert new_w > 1.0

    def test_correct_multiplier(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.1)
        new_w = mgr.update("rsi", correct=True)
        assert math.isclose(new_w, 1.0 * 1.1, rel_tol=1e-9)

    def test_correct_reflected_in_get_weight(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.2)
        mgr.update("macd", correct=True)
        assert math.isclose(mgr.get_weight("macd"), 1.2, rel_tol=1e-9)


# ===========================================================================
# 3. Wrong prediction decreases weight
# ===========================================================================

class TestWrongUpdate:
    def test_wrong_decreases_weight(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        new_w = mgr.update("rsi", correct=False)
        assert new_w < 1.0

    def test_wrong_multiplier(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.1)
        new_w = mgr.update("rsi", correct=False)
        assert math.isclose(new_w, 1.0 * 0.9, rel_tol=1e-9)

    def test_wrong_reflected_in_get_weight(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.2)
        mgr.update("macd", correct=False)
        assert math.isclose(mgr.get_weight("macd"), 0.8, rel_tol=1e-9)


# ===========================================================================
# 4. Weight floor
# ===========================================================================

class TestWeightFloor:
    def test_repeated_wrong_hits_floor(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.5)
        for _ in range(100):
            mgr.update("rsi", correct=False)
        assert mgr.get_weight("rsi") >= _WEIGHT_FLOOR

    def test_floor_is_not_zero(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.99)
        for _ in range(50):
            mgr.update("rsi", correct=False)
        assert mgr.get_weight("rsi") > 0.0

    def test_floor_value(self):
        assert _WEIGHT_FLOOR == 0.01

    def test_update_at_floor_stays_at_floor(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.99)
        for _ in range(50):
            mgr.update("rsi", correct=False)
        w_before = mgr.get_weight("rsi")
        mgr.update("rsi", correct=False)
        w_after = mgr.get_weight("rsi")
        assert w_before == w_after == _WEIGHT_FLOOR


# ===========================================================================
# 5. Persistence
# ===========================================================================

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr = SignalWeightManager(path=path)
        mgr.update("rsi", correct=True)
        mgr.save()
        assert path.exists()

    def test_reload_restores_weights(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr1 = SignalWeightManager(path=path, eta=0.1)
        mgr1.update("rsi", correct=True)
        mgr1.update("macd", correct=False)
        mgr1.save()

        mgr2 = SignalWeightManager(path=path)
        assert math.isclose(mgr2.get_weight("rsi"), 1.1, rel_tol=1e-9)
        assert math.isclose(mgr2.get_weight("macd"), 0.9, rel_tol=1e-9)

    def test_reload_unknown_still_defaults_to_one(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr1 = SignalWeightManager(path=path)
        mgr1.update("rsi", correct=True)
        mgr1.save()

        mgr2 = SignalWeightManager(path=path)
        assert mgr2.get_weight("ema") == 1.0

    def test_missing_file_does_not_raise(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        mgr = SignalWeightManager(path=path)
        assert mgr.get_weight("rsi") == 1.0

    def test_corrupt_file_does_not_raise(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{{")
        mgr = SignalWeightManager(path=path)
        assert mgr.get_weight("rsi") == 1.0


# ===========================================================================
# 6. Batch update
# ===========================================================================

class TestBatchUpdate:
    def test_batch_updates_all_signals(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.1)
        mgr.batch_update({"rsi": True, "macd": False, "ema": True})
        assert math.isclose(mgr.get_weight("rsi"), 1.1, rel_tol=1e-9)
        assert math.isclose(mgr.get_weight("macd"), 0.9, rel_tol=1e-9)
        assert math.isclose(mgr.get_weight("ema"), 1.1, rel_tol=1e-9)

    def test_batch_auto_saves(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr = SignalWeightManager(path=path, eta=0.1)
        mgr.batch_update({"rsi": True})
        # Verify persisted without calling save() explicitly
        mgr2 = SignalWeightManager(path=path)
        assert math.isclose(mgr2.get_weight("rsi"), 1.1, rel_tol=1e-9)

    def test_empty_batch_does_not_raise(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.batch_update({})  # should not raise

    def test_batch_saves_even_when_empty(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr = SignalWeightManager(path=path)
        mgr.batch_update({})
        assert path.exists()


# ===========================================================================
# 7. Normalised weights
# ===========================================================================

class TestNormalizedWeights:
    def test_uniform_weights_average_to_one(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        names = ["rsi", "macd", "ema", "bb"]
        nw = mgr.get_normalized_weights(names)
        avg = sum(nw.values()) / len(nw)
        assert math.isclose(avg, 1.0, rel_tol=1e-9)

    def test_all_values_one_when_uniform(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        names = ["rsi", "macd", "ema"]
        nw = mgr.get_normalized_weights(names)
        for v in nw.values():
            assert math.isclose(v, 1.0, rel_tol=1e-9)

    def test_after_updates_average_still_one(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.1)
        mgr.update("rsi", correct=True)
        mgr.update("macd", correct=False)
        mgr.update("ema", correct=True)
        names = ["rsi", "macd", "ema"]
        nw = mgr.get_normalized_weights(names)
        avg = sum(nw.values()) / len(nw)
        assert math.isclose(avg, 1.0, rel_tol=1e-9)

    def test_better_signal_gets_higher_normalized_weight(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.2)
        for _ in range(5):
            mgr.update("rsi", correct=True)
        for _ in range(5):
            mgr.update("macd", correct=False)
        nw = mgr.get_normalized_weights(["rsi", "macd"])
        assert nw["rsi"] > nw["macd"]

    def test_empty_signal_list_returns_empty_dict(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        nw = mgr.get_normalized_weights([])
        assert nw == {}

    def test_single_signal_returns_one(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.5)
        mgr.update("rsi", correct=True)
        nw = mgr.get_normalized_weights(["rsi"])
        assert math.isclose(nw["rsi"], 1.0, rel_tol=1e-9)

    def test_keys_match_input_signals(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        names = ["rsi", "macd", "ema", "bb", "volume"]
        nw = mgr.get_normalized_weights(names)
        assert set(nw.keys()) == set(names)


# ===========================================================================
# 8. Configurable learning rate
# ===========================================================================

class TestLearningRate:
    def test_custom_eta_correct(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.25)
        new_w = mgr.update("rsi", correct=True)
        assert math.isclose(new_w, 1.25, rel_tol=1e-9)

    def test_custom_eta_wrong(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.25)
        new_w = mgr.update("rsi", correct=False)
        assert math.isclose(new_w, 0.75, rel_tol=1e-9)

    def test_default_eta_value(self):
        assert _DEFAULT_ETA == 0.1

    def test_default_eta_applied_when_none_passed(self, tmp_path):
        mgr = _make_mgr(tmp_path)  # eta=None → should use _DEFAULT_ETA
        new_w = mgr.update("rsi", correct=True)
        assert math.isclose(new_w, 1.0 * (1 + _DEFAULT_ETA), rel_tol=1e-9)


# ===========================================================================
# 9. Compounding updates
# ===========================================================================

class TestCompounding:
    def test_two_correct_compounds(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.1)
        mgr.update("rsi", correct=True)
        new_w = mgr.update("rsi", correct=True)
        expected = 1.0 * 1.1 * 1.1
        assert math.isclose(new_w, expected, rel_tol=1e-9)

    def test_correct_then_wrong_partial_cancel(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.1)
        mgr.update("rsi", correct=True)
        new_w = mgr.update("rsi", correct=False)
        expected = 1.0 * 1.1 * 0.9
        assert math.isclose(new_w, expected, rel_tol=1e-9)

    def test_many_corrects_grow_unbounded_above_one(self, tmp_path):
        mgr = _make_mgr(tmp_path, eta=0.1)
        for _ in range(20):
            mgr.update("rsi", correct=True)
        assert mgr.get_weight("rsi") > 2.0


# ===========================================================================
# 10. JSON file structure
# ===========================================================================

class TestFileStructure:
    def test_saved_json_contains_weights_key(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr = SignalWeightManager(path=path)
        mgr.update("rsi", correct=True)
        mgr.save()
        data = json.loads(path.read_text())
        assert "weights" in data

    def test_saved_json_contains_eta_key(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr = SignalWeightManager(path=path, eta=0.15)
        mgr.save()
        data = json.loads(path.read_text())
        assert "eta" in data
        assert math.isclose(data["eta"], 0.15, rel_tol=1e-9)

    def test_saved_weights_are_numeric(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr = SignalWeightManager(path=path)
        mgr.update("rsi", correct=True)
        mgr.update("macd", correct=False)
        mgr.save()
        data = json.loads(path.read_text())
        for v in data["weights"].values():
            assert isinstance(v, (int, float))
