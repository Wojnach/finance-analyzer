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

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
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
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request. Increments count; CLOSED -> OPEN at threshold, HALF_OPEN -> OPEN."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == State.HALF_OPEN:
                logger.warning(
                    "Circuit breaker '%s': HALF_OPEN -> OPEN (recovery failed, %d failures)",
                    self.name, self._failure_count,
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

            # BUG-93: HALF_OPEN — allow exactly one probe request
            if not self._half_open_probe_sent:
                self._half_open_probe_sent = True
                return True
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
