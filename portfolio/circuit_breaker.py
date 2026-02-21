"""Circuit breaker for data source API calls.

Prevents repeated calls to failing APIs. States:
  CLOSED  — normal operation, requests pass through
  OPEN    — API is failing, requests blocked until recovery timeout
  HALF_OPEN — testing recovery, one request allowed through
"""

import enum
import logging
import time

logger = logging.getLogger("portfolio.circuit_breaker")


class State(enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker for a single data source."""

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = State.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None

    @property
    def state(self) -> State:
        return self._state

    def record_success(self) -> None:
        """Record a successful request. Resets failure count; HALF_OPEN -> CLOSED."""
        if self._state == State.HALF_OPEN:
            logger.info("Circuit breaker '%s': HALF_OPEN -> CLOSED (recovery confirmed)", self.name)
            self._state = State.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request. Increments count; CLOSED -> OPEN at threshold, HALF_OPEN -> OPEN."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == State.HALF_OPEN:
            logger.warning(
                "Circuit breaker '%s': HALF_OPEN -> OPEN (recovery failed, %d failures)",
                self.name, self._failure_count,
            )
            self._state = State.OPEN
        elif self._state == State.CLOSED and self._failure_count >= self.failure_threshold:
            logger.warning(
                "Circuit breaker '%s': CLOSED -> OPEN (threshold %d reached)",
                self.name, self.failure_threshold,
            )
            self._state = State.OPEN

    def allow_request(self) -> bool:
        """Return True if a request should proceed."""
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
                return True
            return False

        # HALF_OPEN — allow the probe request
        return True

    def get_status(self) -> dict:
        """Return current circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
        }
