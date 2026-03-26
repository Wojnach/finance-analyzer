"""Multiplicative Weight Updates (MWU) for online signal learning.

Each signal maintains a persistent weight that is multiplied up on correct
outcomes and down on wrong outcomes.  The result is a classic Hedge algorithm:
signals that are consistently wrong rapidly approach zero weight (floor 0.01)
while consistently correct signals grow to dominate the aggregation.

Weights are persisted to JSON via the same atomic I/O used across the project.
"""

import logging
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.signal_weights")

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _BASE_DIR / "data" / "signal_weights.json"

_DEFAULT_ETA = 0.1   # learning rate — 10% multiplicative update per outcome
_WEIGHT_FLOOR = 0.01  # never reaches zero


class SignalWeightManager:
    """Manages MWU weights for all trading signals.

    Thread-safety note: this class is not internally thread-safe.  In the
    current system it is only called from the single-threaded outcome backfill
    path, so no locking is required.  Add a threading.Lock if that changes.
    """

    def __init__(self, path=None, eta=None):
        self._path = Path(path) if path is not None else _DEFAULT_PATH
        self._eta = eta if eta is not None else _DEFAULT_ETA
        self._weights: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weight(self, signal_name: str) -> float:
        """Return the current weight for *signal_name*.

        Defaults to 1.0 for unknown signals (no prior history).
        """
        return self._weights.get(signal_name, 1.0)

    def update(self, signal_name: str, correct: bool) -> float:
        """Update the weight for *signal_name* after one outcome.

        Correct prediction  → multiply by (1 + eta)
        Incorrect prediction → multiply by (1 - eta)

        The weight is clamped to the floor [_WEIGHT_FLOOR, +∞).

        Returns the new weight.
        """
        current = self._weights.get(signal_name, 1.0)
        if correct:
            new_weight = current * (1.0 + self._eta)
        else:
            new_weight = current * (1.0 - self._eta)
        new_weight = max(new_weight, _WEIGHT_FLOOR)
        self._weights[signal_name] = new_weight
        return new_weight

    def batch_update(self, outcomes: dict) -> None:
        """Update multiple signals at once then persist to disk.

        Args:
            outcomes: ``{signal_name: bool}`` — True means correct prediction.
        """
        for signal_name, correct in outcomes.items():
            self.update(signal_name, correct)
        self.save()

    def get_normalized_weights(self, signal_names) -> dict:
        """Return weights normalised so their average equals 1.0.

        Only considers signals in *signal_names*.  If the list is empty or all
        weights are zero, returns a uniform dict with all values set to 1.0.

        This means the total magnitude of the consensus is preserved — signals
        above 1.0 are stronger than average, below 1.0 weaker.
        """
        signal_names = list(signal_names)
        if not signal_names:
            return {}
        raw = {name: self.get_weight(name) for name in signal_names}
        avg = sum(raw.values()) / len(raw)
        if avg == 0.0:
            return {name: 1.0 for name in signal_names}
        return {name: w / avg for name, w in raw.items()}

    def save(self) -> None:
        """Persist weights to JSON atomically."""
        payload = {
            "eta": self._eta,
            "weights": self._weights,
        }
        atomic_write_json(self._path, payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load weights from disk.  No-ops silently if the file is missing."""
        data = load_json(self._path, default=None)
        if data is None:
            return
        if isinstance(data, dict):
            self._weights = {
                k: float(v)
                for k, v in data.get("weights", {}).items()
            }
            # Honour stored eta only if caller did not override it
            # (caller passes None → _DEFAULT_ETA, so we preserve stored value)
