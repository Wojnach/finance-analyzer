"""Circuit breaker for data source API calls.

Prevents repeated calls to failing APIs. States:
  CLOSED  — normal operation, requests pass through
  OPEN    — API is failing, requests blocked until recovery timeout
  HALF_OPEN — testing recovery, one request allowed through
"""

import enum
import logging
import threading
import time

logger = logging.getLogger("portfolio.circuit_breaker")


class State(enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Thread-safe circuit breaker for a single data source."""

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60,
                 max_recovery_timeout: int = 300):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._base_recovery_timeout = recovery_timeout
        self._max_recovery_timeout = max_recovery_timeout
        self._state = State.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()
        self._half_open_probe_sent = False  # BUG-93: Only one request in HALF_OPEN

    @property
    def state(self) -> State:
        return self._state

    def record_success(self) -> None:
        """Record a successful request. Resets failure count; HALF_OPEN -> CLOSED."""
        with self._lock:
            if self._state == State.HALF_OPEN:
                logger.info("Circuit breaker '%s': HALF_OPEN -> CLOSED (recovery confirmed)", self.name)
                self._state = State.CLOSED
                self._half_open_probe_sent = False  # BUG-93: Reset probe flag
                # BUG-245: Reset backoff on successful recovery
                self.recovery_timeout = self._base_recovery_timeout
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request. Increments count; CLOSED -> OPEN at threshold, HALF_OPEN -> OPEN."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == State.HALF_OPEN:
                # BUG-245: Exponential backoff — double timeout on each failed
                # recovery probe, capped at max. Reduces retry pressure during
                # extended outages (e.g., Binance maintenance windows).
                prev_timeout = self.recovery_timeout
                self.recovery_timeout = min(
                    self.recovery_timeout * 2, self._max_recovery_timeout
                )
                logger.warning(
                    "Circuit breaker '%s': HALF_OPEN -> OPEN (recovery failed, %d failures, "
                    "next probe in %ds, was %ds)",
                    self.name, self._failure_count, self.recovery_timeout, prev_timeout,
                )
                self._state = State.OPEN
                self._half_open_probe_sent = False  # BUG-93: Reset probe flag
            elif self._state == State.CLOSED and self._failure_count >= self.failure_threshold:
                logger.warning(
                    "Circuit breaker '%s': CLOSED -> OPEN (threshold %d reached)",
                    self.name, self.failure_threshold,
                )
                self._state = State.OPEN

    def allow_request(self) -> bool:
        """Return True if a request should proceed."""
        with self._lock:
            if self._state == State.CLOSED:
                return True

            if self._state == State.OPEN:
                if self._last_failure_time is None:
                    return False
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        "Circuit breaker '%s': OPEN -> HALF_OPEN (%.1fs elapsed, testing recovery)",
                        self.name, elapsed,
                    )
                    self._state = State.HALF_OPEN
                    self._half_open_probe_sent = True  # BUG-93: This IS the probe
                    return True
                return False

            # BUG-93/BUG-187: HALF_OPEN — the probe request is always sent via
            # the OPEN→HALF_OPEN transition above (which sets probe_sent=True and
            # returns True). This branch handles the case where a second request
            # arrives while still in HALF_OPEN (waiting for success/failure).
            return False

    def get_status(self) -> dict:
        """Return current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Force the breaker back to CLOSED with zero failures.

        Intended use: operational override (manual recovery) and test
        isolation. Production code should NOT call this in normal flow
        — let record_success/record_failure drive the state machine.

        2026-05-02: added when test_consensus xdist flakes traced back
        to module-level breakers tripping during one test and leaking
        into the next on the same xdist worker.
        """
        with self._lock:
            self._state = State.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_probe_sent = False
            self.recovery_timeout = self._base_recovery_timeout
