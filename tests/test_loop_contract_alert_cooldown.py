"""Tests for per-invariant Telegram alert cooldown in _alert_violations.

Background (2026-04-28): the accuracy_degradation invariant has a
throttled-replay design that re-emits the cached Violation list every cycle
to keep ViolationTracker.consecutive alive. _alert_violations does not
deduplicate, so the same critical Telegram alert ships every ~10 minutes —
192 cycles in a row by the time we noticed.

These tests pin the contract for a per-invariant cooldown that
suppresses identical replays but still ships when the message text changes
(e.g. a new degraded signal joins the alert list).
"""

import hashlib
import time

import pytest

from portfolio.loop_contract import (
    Violation,
    _alert_violations,
)


@pytest.fixture()
def cooldown_state_file(tmp_path, monkeypatch):
    """Redirect contract_state.json to a tmp file so each test starts clean."""
    state_file = tmp_path / "contract_state.json"
    monkeypatch.setattr(
        "portfolio.loop_contract.CONTRACT_STATE_FILE", state_file,
    )
    return state_file


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class TestAlertCooldown:
    """First fire goes out; identical replays suppressed; text-change re-fires."""

    def test_first_critical_alert_is_sent(self, cooldown_state_file):
        """A fresh CRITICAL violation always reaches send_or_store on first fire."""
        from unittest.mock import patch

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v], config={})
        mock_send.assert_called_once()

    def test_identical_replay_within_cooldown_is_suppressed(self, cooldown_state_file):
        """Sending the same (invariant, message) again within the cooldown is dropped."""
        from unittest.mock import patch

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v], config={})  # first fire
            _alert_violations([v], config={})  # duplicate replay
            _alert_violations([v], config={})  # another duplicate replay
        # Only the first should reach the wire.
        assert mock_send.call_count == 1

    def test_message_text_change_refires_immediately(self, cooldown_state_file):
        """Changing the message text bypasses the cooldown — the alert is genuinely new."""
        from unittest.mock import patch

        v_old = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        v_new = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="13 signal(s) dropped >15pp...",  # one more signal added
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v_old], config={})
            _alert_violations([v_new], config={})
        assert mock_send.call_count == 2

    def test_after_cooldown_expires_replay_is_resent(self, cooldown_state_file):
        """Once the cooldown window passes, the same message can ship again."""
        from unittest.mock import patch

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v], config={})
            # Rewind every persisted ts by more than the default cooldown.
            # The cooldown reads recent_hashes[*].ts now (multi-hash dedup);
            # last_sent_ts is a legacy mirror for human inspection.
            from portfolio.file_utils import load_json, atomic_write_json
            state = load_json(cooldown_state_file, default={}) or {}
            forged_ts = time.time() - 5 * 3600  # 5 h ago > 4 h default
            for entry in (state.get("telegram_alert_state") or {}).values():
                entry["last_sent_ts"] = forged_ts
                for r in entry.get("recent_hashes", []) or []:
                    r["ts"] = forged_ts
            atomic_write_json(cooldown_state_file, state)

            _alert_violations([v], config={})
        assert mock_send.call_count == 2

    def test_different_invariants_do_not_share_cooldown(self, cooldown_state_file):
        """Layer 2 alerts and accuracy alerts must not block each other."""
        from unittest.mock import patch

        v1 = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        v2 = Violation(
            invariant="layer2_journal_activity",
            severity="CRITICAL",
            message="Layer 2 trigger fired 30m ago...",
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v1], config={})
            _alert_violations([v2], config={})
        assert mock_send.call_count == 2

    def test_partial_dedup_still_sends_remaining_violations(self, cooldown_state_file):
        """If a bundled alert has a stale invariant + a fresh one, the fresh
        one still reaches the wire (just without the stale duplicate)."""
        from unittest.mock import patch

        v_stale = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        v_fresh = Violation(
            invariant="layer2_journal_activity",
            severity="CRITICAL",
            message="Layer 2 trigger fired 30m ago...",
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v_stale], config={})  # accuracy alert lands first
            _alert_violations([v_stale, v_fresh], config={})  # bundled — accuracy stale, l2 fresh
        # Two sends total: first one (accuracy), second one (l2 only — accuracy filtered)
        assert mock_send.call_count == 2
        second_call_args = mock_send.call_args_list[1]
        sent_msg = second_call_args.args[0] if second_call_args.args else second_call_args.kwargs["msg"]
        assert "layer2_journal_activity" in sent_msg
        assert "accuracy_degradation" not in sent_msg

    def test_cooldown_state_persists_to_contract_state_file(self, cooldown_state_file):
        """After an alert ships, contract_state.json contains the cooldown bookkeeping."""
        from unittest.mock import patch

        from portfolio.file_utils import load_json

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        with patch("portfolio.message_store.send_or_store"):
            _alert_violations([v], config={})

        state = load_json(cooldown_state_file, default={}) or {}
        assert "telegram_alert_state" in state
        per_invariant = state["telegram_alert_state"].get("accuracy_degradation")
        assert per_invariant is not None
        assert per_invariant.get("last_message_hash") == _sha1(v.message)
        assert per_invariant.get("last_sent_ts") > 0

    def test_warning_severity_does_not_use_cooldown(self, cooldown_state_file):
        """The cooldown only applies to the CRITICAL path that actually ships
        Telegram alerts. Warnings stay below the threshold today."""
        from unittest.mock import patch

        v = Violation(
            invariant="health_updated",
            severity="WARNING",
            message="Health state was not updated this cycle.",
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v], config={})
            _alert_violations([v], config={})
        # Warnings never reach send_or_store at all (existing behavior).
        mock_send.assert_not_called()

    def test_no_critical_violations_means_no_send_and_no_state_write(self, cooldown_state_file):
        """A clean call with no CRITICAL violations must not touch
        contract_state.json — the file shouldn't sprout an empty
        telegram_alert_state block on every passing cycle."""
        from unittest.mock import patch

        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([], config={})
        mock_send.assert_not_called()
        # File may not exist — that's fine. If it does, it shouldn't have
        # the cooldown key.
        if cooldown_state_file.exists():
            from portfolio.file_utils import load_json
            state = load_json(cooldown_state_file, default={}) or {}
            assert "telegram_alert_state" not in state

    def test_send_failure_does_not_persist_cooldown(self, cooldown_state_file):
        """If send_or_store raises, we must NOT mark the alert as sent —
        otherwise a transient Telegram outage would suppress the next 4 h
        of legitimate critical alerts."""
        from unittest.mock import patch

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        with patch(
            "portfolio.message_store.send_or_store",
            side_effect=RuntimeError("Telegram API down"),
        ):
            _alert_violations([v], config={})

        from portfolio.file_utils import load_json
        state = load_json(cooldown_state_file, default={}) or {}
        per_invariant = (state.get("telegram_alert_state") or {}).get("accuracy_degradation")
        # Either the key is absent, or last_sent_ts is 0 — both signal "didn't ship".
        assert per_invariant is None or not per_invariant.get("last_sent_ts")

    def test_muted_categories_do_not_persist_cooldown(self, cooldown_state_file):
        """Codex P2-2 follow-up: when telegram.muted_categories includes
        'error', send_or_store stores the message and returns True without
        attempting a Telegram send. The cooldown must not claim its 4 h
        window in that case — otherwise unmuting within the window
        suppresses the first real recurrence even though no operator was
        ever paged."""
        from unittest.mock import patch

        from portfolio.file_utils import load_json

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        config = {"telegram": {"muted_categories": ["error"]}}
        with patch(
            "portfolio.message_store.send_or_store",
            return_value=True,  # "stored, didn't actually send" — same as muted
        ) as mock_send:
            _alert_violations([v], config=config)
        assert mock_send.call_count == 1

        state = load_json(cooldown_state_file, default={}) or {}
        per_inv = (state.get("telegram_alert_state") or {}).get("accuracy_degradation")
        assert per_inv is None or not per_inv.get("last_sent_ts"), (
            "muted_categories suppressed the actual delivery but cooldown "
            "was claimed; first real recurrence after unmute will be silenced"
        )

    def test_mute_all_with_unmuted_whitelist_skips_cooldown(self, cooldown_state_file):
        """Same as above for the global mute_all gate: unless the category
        is in the unmuted whitelist, no Telegram is sent and the cooldown
        must stay clear."""
        from unittest.mock import patch

        from portfolio.file_utils import load_json

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        config = {
            "telegram": {
                "mute_all": True,
                "unmuted_categories": ["other_category"],  # 'error' is muted
            },
        }
        with patch(
            "portfolio.message_store.send_or_store",
            return_value=True,
        ):
            _alert_violations([v], config=config)

        state = load_json(cooldown_state_file, default={}) or {}
        per_inv = (state.get("telegram_alert_state") or {}).get("accuracy_degradation")
        assert per_inv is None or not per_inv.get("last_sent_ts")

    def test_mute_all_with_error_in_unmuted_whitelist_persists_cooldown(
        self, cooldown_state_file,
    ):
        """Sanity: when 'error' IS in the unmute whitelist under mute_all,
        the alert really does ship and the cooldown should engage."""
        from unittest.mock import patch

        from portfolio.file_utils import load_json

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        config = {
            "telegram": {
                "mute_all": True,
                "unmuted_categories": ["error"],
            },
        }
        with patch(
            "portfolio.message_store.send_or_store",
            return_value=True,
        ):
            _alert_violations([v], config=config)

        state = load_json(cooldown_state_file, default={}) or {}
        per_inv = (state.get("telegram_alert_state") or {}).get("accuracy_degradation")
        assert per_inv is not None and per_inv.get("last_sent_ts") > 0

    def test_text_flap_a_b_a_within_cooldown_does_not_re_fire_a(
        self, cooldown_state_file,
    ):
        """Codex P2-3: dedup must remember every recent hash per invariant,
        not just the most recent one. If the alert flaps A -> B -> A
        within the cooldown window (e.g. degraded-signal list grows by
        one and shrinks back), the second A must be suppressed because
        it's the same incident as the first A. Otherwise the user sees
        the same alert again at hour T+0, T+x, T+2x, ..."""
        from unittest.mock import patch

        v_a = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped...",
        )
        v_b = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="13 signal(s) dropped...",
        )
        with patch("portfolio.message_store.send_or_store") as mock_send:
            _alert_violations([v_a], config={})  # A — fires
            _alert_violations([v_b], config={})  # B (text change) — fires
            _alert_violations([v_a], config={})  # A again — should be suppressed

        assert mock_send.call_count == 2, (
            "Text flap A -> B -> A within cooldown window re-fired A. "
            "Per-invariant dedup must remember every recent hash, not "
            "just the most recent one."
        )

    def test_send_returning_false_does_not_persist_cooldown(self, cooldown_state_file):
        """Codex P1 follow-up: send_or_store reports normal delivery failures
        by returning False (missing token, non-OK sendMessage response, etc.)
        rather than raising. The cooldown must respect that boolean — silencing
        4 h of CRITICAL alerts after a transient Telegram outage that no
        operator was paged for would be exactly the kind of silent failure
        the contract framework exists to prevent."""
        from unittest.mock import patch

        v = Violation(
            invariant="accuracy_degradation",
            severity="CRITICAL",
            message="12 signal(s) dropped >15pp...",
        )
        with patch(
            "portfolio.message_store.send_or_store",
            return_value=False,
        ):
            _alert_violations([v], config={})

        from portfolio.file_utils import load_json
        state = load_json(cooldown_state_file, default={}) or {}
        per_invariant = (state.get("telegram_alert_state") or {}).get("accuracy_degradation")
        assert per_invariant is None or not per_invariant.get("last_sent_ts"), (
            "send_or_store returned False (delivery failed) but cooldown was "
            "persisted — next 4 h of legitimate CRITICAL alerts would be "
            "silenced even though no Telegram was actually delivered"
        )
