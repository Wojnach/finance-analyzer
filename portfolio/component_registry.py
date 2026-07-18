"""Single source of truth for per-instrument signal enablement (Phase 4.1).

Today the engine decides "does signal S vote for ticker T at horizon H"
via 4 overlapping hardcoded mechanisms spread across tickers.py and
signal_engine.py (DISABLED_SIGNALS, _DISABLED_SIGNAL_OVERRIDES,
_TICKER_DISABLED_BY_HORIZON, asset-class sets) plus a disconnected
instrument_profile.py that only fin_fish/fish_monitor_smart read and that
contradicts the engine's own gating (see NOTE below). This module reads a
generated snapshot of that behavior (portfolio/registry_defaults.py, see
scripts/gen_registry_defaults.py) plus a live JSON overlay, and answers the
same question through one API.

Nothing imports this module yet — signal_engine.py keeps using its own
constants until Phase 4.2 flips a feature flag after the parity harness in
tests/test_component_registry.py stays green in prod shadow. See
docs/plans/2026-07-18-dashboard-redesign-and-modular-engine.md Phase 4.

NOTE on instrument_profile.py: its XAG/XAU trusted_signals lists include
claude_fundamental, momentum_factors, structure and fibonacci — all
globally disabled in DISABLED_SIGNALS today (structure is additionally
ticker-blacklisted for both XAG-USD and XAU-USD). Its ignored_signals lists
are, by contrast, mostly already-disabled signals. The profile was clearly
built from an accuracy snapshot that predates several disable decisions and
was never wired back into signal_engine's gating. This module does not
attempt to reconcile that — see Phase 4.4 (absorb/delete instrument_profile.py).

Overlay file (data/control/registry_overrides.json), all keys optional::

    {
      "<ticker>": {
        "<signal>": {
          "enabled": true,
          "reason": "operator note",
          "horizons": {"1d": false, "3h": true}
        }
      }
    }

A missing file means no overrides. A malformed file (not a JSON object) is
logged as a warning and ignored — never raises.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from portfolio.file_utils import load_json
from portfolio.registry_defaults import (
    DISABLED_SIGNAL_OVERRIDES,
    SIGNALS,
    TICKER_DISABLED_BY_HORIZON,
)
from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS

logger = logging.getLogger("portfolio.component_registry")

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_OVERLAY_PATH = _BASE_DIR / "data" / "control" / "registry_overrides.json"


class ComponentRegistry:
    """Read API over registry_defaults.py + a live JSON overlay.

    Instantiate directly (e.g. in tests, with a scratch overlay_path) or use
    the module-level `get_registry()` singleton, which reloads the overlay
    whenever its mtime changes (same pattern as portfolio.api_utils.load_config).
    """

    def __init__(self, overlay_path: Path | None = None):
        self._overlay_path = overlay_path or _DEFAULT_OVERLAY_PATH
        self._lock = threading.Lock()
        self._overlay: dict[str, Any] = {}
        self._overlay_mtime: float | None = None

    def _reload_overlay_if_stale(self) -> None:
        with self._lock:
            try:
                mtime = self._overlay_path.stat().st_mtime
            except OSError:
                if self._overlay_mtime is not None:
                    self._overlay = {}
                    self._overlay_mtime = None
                return
            if self._overlay_mtime == mtime:
                return
            data = load_json(str(self._overlay_path), default=None)
            if data is None:
                self._overlay = {}
            elif not isinstance(data, dict):
                logger.warning(
                    "registry overlay %s is not a JSON object, ignoring",
                    self._overlay_path,
                )
                self._overlay = {}
            else:
                self._overlay = data
            self._overlay_mtime = mtime

    def _overlay_entry(self, signal: str, ticker: str | None) -> dict:
        if not ticker:
            return {}
        self._reload_overlay_if_stale()
        ticker_entry = self._overlay.get(ticker)
        if not isinstance(ticker_entry, dict):
            return {}
        sig_entry = ticker_entry.get(signal)
        return sig_entry if isinstance(sig_entry, dict) else {}

    # -- enablement -----------------------------------------------------

    def is_enabled(self, signal: str, ticker: str, horizon: str | None = None) -> bool:
        """Would `signal` vote for `ticker` at `horizon` ("_default" if None)?

        Ignores GPU market-hours skipping — that's a runtime condition
        applied dynamically, not static enablement (see applicable_signals /
        applicable_count, which take skip_gpu explicitly).

        An unknown ticker falls through every per-ticker table untouched
        (no rescue, no per-ticker/horizon blacklist) and is treated as "not
        in any asset class", so asset-class-restricted signals answer False
        and everything else answers on global state alone — this is the
        "unknown ticker -> global-only answer" behavior.
        """
        overlay = self._overlay_entry(signal, ticker)
        if horizon and "horizons" in overlay and horizon in overlay["horizons"]:
            return bool(overlay["horizons"][horizon])
        if "enabled" in overlay:
            return bool(overlay["enabled"])

        meta = SIGNALS.get(signal)
        if meta is None:
            return False  # unknown signal

        if meta["disabled"] and (signal, ticker) not in DISABLED_SIGNAL_OVERRIDES:
            return False

        restriction = meta["asset_class_restriction"]
        if restriction == "crypto_only" and ticker not in CRYPTO_SYMBOLS:
            return False
        if restriction == "metals_only" and ticker not in METALS_SYMBOLS:
            return False
        if restriction == "non_stock" and ticker in STOCK_SYMBOLS:
            return False

        default_disabled = TICKER_DISABLED_BY_HORIZON.get("_default", {}).get(
            ticker, frozenset()
        )
        if signal in default_disabled:
            return False
        if horizon:
            horizon_disabled = TICKER_DISABLED_BY_HORIZON.get(horizon, {}).get(
                ticker, frozenset()
            )
            if signal in horizon_disabled:
                return False
        return True

    def is_globally_disabled(self, signal: str, ticker: str) -> bool:
        """Pure global-disable check: DISABLED_SIGNALS + the static
        per-ticker override table (and the overlay's `enabled` key, which
        takes precedence same as is_enabled) — nothing else.

        Ignores asset-class restriction and the ticker/horizon blacklist
        (see is_enabled, which composes all three). signal_engine's
        registry-flag wrappers need this narrower cut because those two
        axes are orthogonal to shadow-registry promotion, while this one
        (DISABLED_SIGNALS + _DISABLED_SIGNAL_OVERRIDES) is exactly the axis
        a promotion bypasses — see docs/plans/2026-07-18-dashboard-redesign-
        and-modular-engine.md Phase 4.2.
        """
        overlay = self._overlay_entry(signal, ticker)
        if "enabled" in overlay:
            return not bool(overlay["enabled"])
        meta = SIGNALS.get(signal)
        if meta is None:
            return True
        return (
            bool(meta["disabled"]) and (signal, ticker) not in DISABLED_SIGNAL_OVERRIDES
        )

    def is_ticker_horizon_blacklisted(
        self, signal: str, ticker: str, horizon: str | None = None
    ) -> bool:
        """Pure per-ticker/horizon blacklist check: is `signal` in the
        static "_default" ∪ `horizon`-specific TICKER_DISABLED_BY_HORIZON
        list for `ticker`, or overlaid to one via the overlay's `horizons`
        key? Ignores global disable/override (is_globally_disabled) and
        asset-class restriction (is_enabled composes all three).

        The overlay's top-level `enabled` key is deliberately NOT consulted
        here — that's is_globally_disabled's exclusive concern. Keeping
        these two checks independent (rather than folding both into
        is_enabled) is what lets signal_engine preserve the
        shadow-registry-promotion nuance: a promotion only ever bypasses
        the global-disable axis, never this one — see
        docs/plans/2026-07-18-dashboard-redesign-and-modular-engine.md
        Phase 4.2.
        """
        overlay = self._overlay_entry(signal, ticker)
        if horizon and "horizons" in overlay and horizon in overlay["horizons"]:
            return not bool(overlay["horizons"][horizon])
        default_disabled = TICKER_DISABLED_BY_HORIZON.get("_default", {}).get(
            ticker, frozenset()
        )
        if signal in default_disabled:
            return True
        if horizon:
            horizon_disabled = TICKER_DISABLED_BY_HORIZON.get(horizon, {}).get(
                ticker, frozenset()
            )
            if signal in horizon_disabled:
                return True
        return False

    def disabled_reason(self, signal: str, ticker: str | None = None) -> str | None:
        """Why is `signal` disabled? None means "it isn't" (for this query).

        ticker=None asks the global-only question (ignoring per-ticker
        rescues/overlays): reason from tickers.get_disabled_reason(), or
        None if the signal isn't in DISABLED_SIGNALS.
        """
        meta = SIGNALS.get(signal)
        if meta is None:
            return "unknown signal"
        if ticker is None:
            return meta["disabled_reason"] if meta["disabled"] else None
        if self.is_enabled(signal, ticker):
            return None
        overlay = self._overlay_entry(signal, ticker)
        if overlay.get("reason"):
            return str(overlay["reason"])
        if meta["disabled_reason"]:
            return meta["disabled_reason"]
        if meta["asset_class_restriction"]:
            return (
                f"asset-class restricted ({meta['asset_class_restriction']}), "
                f"not applicable to {ticker}"
            )
        # Enabled globally but blacklisted for this ticker/horizon — the
        # accuracy-audit reason for that lives in a signal_engine.py code
        # comment next to _TICKER_DISABLED_BY_HORIZON, not captured in the
        # generated snapshot (Phase 4.1 keeps the snapshot compact).
        return f"disabled for {ticker} (per-ticker/horizon exception list)"

    # -- applicability ----------------------------------------------------

    def applicable_signals(
        self, ticker: str, skip_gpu: bool = False, horizon: str | None = None
    ) -> frozenset[str]:
        """All signals that would vote for `ticker` at `horizon`.

        skip_gpu=True additionally excludes GPU signals (mirrors
        signal_engine._compute_applicable_count's skip_gpu branch).
        """
        return frozenset(
            signal
            for signal, meta in SIGNALS.items()
            if self.is_enabled(signal, ticker, horizon)
            and not (skip_gpu and meta["gpu"])
        )

    def applicable_count(
        self, ticker: str, skip_gpu: bool = False, horizon: str | None = None
    ) -> int:
        return len(self.applicable_signals(ticker, skip_gpu=skip_gpu, horizon=horizon))

    # -- voter state (feeds dashboard Phase 1.4 "voters" card) -------------

    def voter_state(self, signal: str, ticker: str | None = None) -> dict:
        """{state, reason} for `signal` — state in VOTING/DISABLED/SHADOW.

        Deliberately does no network calls / flag-file reads (remote-gate
        state like remote_llm_available() or data/local_llm.disabled stays
        the dashboard's job — see dashboard/system_status.py's parallel,
        richer _voter_state which adds GATED_REMOTE_DOWN/PAUSED_LLM_FLAG on
        top of this). SHADOW means the signal is in _KNOWN_SHADOW_LLMS —
        computed and logged, never voting in consensus, even when rescued
        per-ticker (e.g. phi4_mini stays SHADOW; its per-ticker rescue only
        matters once VOTING is the dashboard-level composite state).
        """
        meta = SIGNALS.get(signal)
        if meta is None:
            return {"state": "DISABLED", "reason": "unknown signal"}
        if meta["shadow_llm"]:
            reason = (
                self.disabled_reason(signal, ticker) or "shadow-tracked, not voting"
            )
            return {"state": "SHADOW", "reason": reason}
        enabled = (
            not meta["disabled"] if ticker is None else self.is_enabled(signal, ticker)
        )
        if enabled:
            return {"state": "VOTING", "reason": None}
        return {"state": "DISABLED", "reason": self.disabled_reason(signal, ticker)}

    # -- bulk dump for dashboard/API ---------------------------------------

    def snapshot(self) -> dict:
        """Full explode: {ticker: {signal: {...}}} for every known ticker."""
        from portfolio.registry_defaults import HORIZONS, TICKERS

        out: dict[str, dict] = {}
        for ticker in TICKERS:
            out[ticker] = {}
            for signal, meta in SIGNALS.items():
                horizons = {
                    h: self.is_enabled(signal, ticker, h)
                    for h in ("_default", *HORIZONS)
                }
                out[ticker][signal] = {
                    "enabled_default": horizons["_default"],
                    "horizons": horizons,
                    "disabled_reason": self.disabled_reason(signal, ticker),
                    "voter_state": self.voter_state(signal, ticker),
                    "gpu": meta["gpu"],
                    "core": meta["core"],
                    "shadow_llm": meta["shadow_llm"],
                    "asset_class_restriction": meta["asset_class_restriction"],
                }
        return out


# -- module-level singleton, mirrors portfolio.api_utils.load_config's
# mtime-cached pattern --------------------------------------------------

_default_registry: ComponentRegistry | None = None
_default_registry_lock = threading.Lock()


def get_registry() -> ComponentRegistry:
    global _default_registry
    if _default_registry is None:
        with _default_registry_lock:
            if _default_registry is None:
                _default_registry = ComponentRegistry()
    return _default_registry


def is_enabled(signal: str, ticker: str, horizon: str | None = None) -> bool:
    return get_registry().is_enabled(signal, ticker, horizon)


def is_globally_disabled(signal: str, ticker: str) -> bool:
    return get_registry().is_globally_disabled(signal, ticker)


def is_ticker_horizon_blacklisted(
    signal: str, ticker: str, horizon: str | None = None
) -> bool:
    return get_registry().is_ticker_horizon_blacklisted(signal, ticker, horizon)


def disabled_reason(signal: str, ticker: str | None = None) -> str | None:
    return get_registry().disabled_reason(signal, ticker)


def applicable_signals(
    ticker: str, skip_gpu: bool = False, horizon: str | None = None
) -> frozenset[str]:
    return get_registry().applicable_signals(ticker, skip_gpu=skip_gpu, horizon=horizon)


def applicable_count(
    ticker: str, skip_gpu: bool = False, horizon: str | None = None
) -> int:
    return get_registry().applicable_count(ticker, skip_gpu=skip_gpu, horizon=horizon)


def voter_state(signal: str, ticker: str | None = None) -> dict:
    return get_registry().voter_state(signal, ticker)


def snapshot() -> dict:
    return get_registry().snapshot()
