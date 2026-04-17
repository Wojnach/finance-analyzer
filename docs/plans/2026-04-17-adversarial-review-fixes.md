# Adversarial-Review Fixes (2026-04-17)

## Context

Four parallel adversarial reviewers (pr-review-toolkit: code-reviewer, silent-failure-hunter, type-design-analyzer, pr-test-analyzer) reviewed commits `b88e3ed^..7c9f4ad1` on `main`. This plan addresses every finding.

## Findings to address

### P1 (critical)

- **P1-B** — Horizon-blacklist force-HOLDs don't propagate to `extra_info["_voters"]`. Outer caller sets `_voters = buy+sell` (pre-horizon-blacklist); `_weighted_consensus` then force-HOLDs horizon-blacklisted signals. Stage-4 dynamic_min reads stale `_voters`. (`signal_engine.py:2254+2464+1158`)
- **P1-C** — `_count_active_voters_at_gate` crashes on `None` accuracy values in the live path (replay has `except TypeError`, live doesn't). (`signal_engine.py:953-969`)
- **P1-D** — `blend_accuracy_data` silently drops directional keys when signal is only in `recent`. Directional gate never fires. (`accuracy_stats.py:751-776`)
- **P1-E** — `scripts/replay_consensus.py` has no tests.
- **P1-F** — `TestGenerateSignalHorizonBlacklistE2E.test_mstr_claude_fundamental_is_hold_at_every_horizon` only asserts `isinstance(extra, dict)` — never checks the consensus output.

### P2 (should fix)

- **P2-A** — Guard C (`best_possible >= 2`) still allows 2-voter consensus. Raise to `MIN_VOTERS_BASE` (3).
- **P2-B** — `blend_accuracy_data` called with `min_recent_samples=30` in production vs default `50` in backtester/replay.
- **P2-C** — `_dynamic_min_voters_for_regime` duplicated in `apply_confidence_penalties` — replace with the helper call.
- **P2-D** — Regime string case/typo silently falls to strictest quorum. Normalize.
- **P2-E** — Stale accuracy cache lets circuit-breaker relax against fresh votes. Noting — not fixing here; larger refactor.
- **P2-F** — `_POST_EXCLUSION_MIN = 3` hardcoded, not derived from `MIN_VOTERS_CRYPTO/STOCK`.
- **P2-G** — No module-load assertion for `ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX > _DIRECTIONAL_GATE_THRESHOLD`.
- **P2-H** — `_TICKER_DISABLED_BY_HORIZON` structure invariants not enforced at module load (`_default` required, horizon keys bounded, values must be frozenset).
- **P2-I** — E2E test mocks cache to None, masking the "cache-miss" code path; claude_fundamental goes HOLD for the wrong reason.
- **P2-J** — Replay script catches only `KeyError/ValueError/TypeError`; `AttributeError`/`RuntimeError` crashes full replay.
- **P2-K** — `_GATE_RELAXATION_STEP = 0` would cause `ZeroDivisionError`. Add defensive check.

### P3 (defensive)

- **P3-1** — `_get_horizon_disabled_signals` uses `[]` on `_default` at line 311 — crash on future removal.
- **P3-2** — `_verdict_correct` asymmetric for `change_pct == 0` (counts as miss for both directions). Leave as-is (float precision makes this rare in prod) but document.
- **P3-3** — No logging in `_count_active_voters_at_gate` for post-mortem.
- **P3-4** — `load_cached_accuracy` returning None silently collapses to `{}` in replay — guard intent not met.
- **P3-5** — No NaN/inf adversarial input tests.
- **P3-6** — No pin for `regime=None` vs `regime="unknown"` equivalence.
- **P3-7** — `samples=0` bypasses accuracy gate entirely — document existing behavior.
- **P3-8** — Replay `_load_entries` silently skips malformed JSON rows — add counter.

## Batches

### Batch 1 — P1 critical fixes

- `signal_engine.py`: apply horizon-disabled at the SAME site as regime-gating (line ~2218) so `active_voters` counts post-horizon-disable. Remove duplicate application inside `_weighted_consensus` (or keep as idempotent no-op).
- `signal_engine.py:_count_active_voters_at_gate`: coerce None values to safe defaults.
- `accuracy_stats.py:blend_accuracy_data`: merge directional keys from both alltime and recent (prefer larger-sample source for each key).
- Add new tests for each.

### Batch 2 — P2 consolidation

- Raise Guard C to `_LONE_SIGNAL_FLOOR = MIN_VOTERS_CRYPTO` (3).
- Replace inline dynamic_min block in `apply_confidence_penalties` with `_dynamic_min_voters_for_regime(regime)`.
- Add `_normalize_regime(s)` helper: `s.strip().lower()` + alias mapping. Use at both sites.
- Derive `_POST_EXCLUSION_MIN = max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)`.
- Add module-load assertions: `_MIN_RECOVERY_FLOOR <= _MIN_ACTIVE_VOTERS_SOFT`; relaxation margin vs directional gate; `_GATE_RELAXATION_STEP > 0`; `_default` key exists; values are frozensets.
- Unify `blend_accuracy_data` min_recent_samples via named constant used by all callers.

### Batch 3 — Replay hardening + tests

- `scripts/replay_consensus.py`: broaden per-ticker exception to `Exception`, keep RuntimeError for cache-load; add malformed-row counter; surface "unknown regime" prevalence in report; raise if both caches are None.
- New `tests/test_replay_consensus.py`: smoke test on a synthetic 2-entry signal_log fixture (tmp_path).
- Rewrite `test_mstr_claude_fundamental_is_hold_at_every_horizon` with a real vote spy that captures the post-gate state.

### Batch 4 — P3 defensive

- `_get_horizon_disabled_signals` use `.get("_default", {})`.
- New `tests/test_circuit_breaker_invariants.py` module-load assertion tests + NaN/inf tests.
- Pin `regime=None == regime="unknown"` in an explicit test.
- Add debug logs in `_count_active_voters_at_gate` at each relaxation step (gated behind a level check).
- Add parser-error counter in `_load_entries`.

### Batch 5 — Verify, merge, ship

- Full pytest (-n auto).
- Commit + push via cmd.exe git.
- Restart PF-DataLoop.
- Clean up worktree.

## Risk assessment

- **Batch 1 P1-B** is the highest-risk change (touches core consensus flow). Requires a test that END-TO-END asserts `_voters` = post-horizon count.
- **Batch 2 P2-C** (de-dup dynamic_min) is pure refactor but touches `apply_confidence_penalties` which is heavily tested — run the confidence-penalty suite after.
- All other batches are additive / test-only / assertion-only — low risk.

## Rollback

Each batch is an independent commit. `git revert <sha>` undoes one.

## Out of scope

- P2-E (stale cache vs fresh votes) — larger refactor requiring timestamp tracking in accuracy_data.
- P3-7 (`samples=0` bypass) — long-standing intentional behavior; changing it would force-gate new signals for their entire warm-up window.
- P1-A (replay regime=unknown) — signal_log.jsonl doesn't persist regime. Would require loop-side change to start recording regime; noted for future session.
