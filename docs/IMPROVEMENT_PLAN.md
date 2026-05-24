# Improvement Plan — Auto-Session 2026-05-24

Based on 6-agent parallel exploration of codebase + cross-reference against
FGL adversarial review (2026-05-23, 22 P0s, ~40 P1s).

## Methodology

- FGL 2026-05-23 SYNTHESIS.md identified 22 P0 bugs
- Prior auto-sessions (5/22, 5/23) fixed 25 bugs across 10 batches
- This session targets remaining FGL P0s with clear fixes + low blast radius

## P0 Classification After Verification

### CONFIRMED P0s (10 fixes this session)

| # | FGL P0 | File | Issue | Risk |
|---|--------|------|-------|------|
| 1 | P0-2 | `portfolio/fin_snipe_manager.py:64` | MIN_STOP_DISTANCE_PCT=1.0, violates 3% rule | LOW |
| 2 | P0-6 | `portfolio/signal_decay_alert.py:27,148` | Relative `"data/..."` paths — silent failure | LOW |
| 3 | P0-7 | `portfolio/api_utils.py:60` | `apiKey` vs `key` config mismatch | LOW |
| 4 | P0-5 | `portfolio/signal_engine.py:~4238` | accuracy_gate config override below 47% floor | LOW |
| 5 | P0-8 | `portfolio/signals/connors_rsi2.py:103` | `**kwargs` absorbs `context=`, ticker guard bypassed | LOW |
| 6 | P0-9 | `portfolio/signal_engine.py:~3601-3685` | Promoted shadow signal HOLDs forever | MED |
| 7 | P0-11 | `portfolio/trigger.py:496-509` | Price baseline stales across quiet periods | MED |
| 8 | P0-12 | `portfolio/agent_invocation.py:647-682` | `_agent_proc` never cleared after kill+wait wedge | MED |
| 9 | P0-3 | `portfolio/grid_fisher.py:~1424-1522` | Naked position when stop-rearm fails | MED |
| 10 | P0-4 | `portfolio/agent_invocation.py:~967-971` | 0/3 specialist failure cascades to "success" | MED |

### FALSE POSITIVE (1)

| # | FGL P0 | Verdict | Evidence |
|---|--------|---------|----------|
| 1 | P0-1 | **CORRECT** | kelly_metals.py math verified: code gives half-Kelly=0.587, proportional Kelly formula gives 0.587. FGL review used wrong formula. |

### DEFERRED (11 — too risky or needs design session)

| # | FGL P0 | Why Deferred |
|---|--------|--------------|
| P0-10 | Drawdown FX mixing | Needs design decision: USD-only vs dual-currency breaker |
| P0-13 | Avanza trading whitelist | Unified guard requires testing all 4 order paths |
| P0-14 | get_buying_power returns 0 | Breaking return type change |
| P0-15 | get_positions pension leak | Breaking default change |
| P0-16 | EOD hardcoded | Needs Avanza API integration for `todayClosingTime` |
| P0-17 | EOD crash recovery | Needs atexit + signal handler design |
| P0-18 | yfinance lock | Concurrency refactor across 5+ callers |
| P0-19 | fx_rates stale | Breaking return type (tuple) |
| P0-20 | Core-gate pre-persistence | Complex change in 4400-line signal_engine.py |
| P0-21 | Warrant P&L barrier | Schema change + downstream consumers |
| P0-22 | Journal linear scan | Already partially addressed (load_jsonl_tail cap) |

---

## Execution Batches

### Batch 1: Trivial fixes (4 files, LOW risk)

**Files:** `fin_snipe_manager.py`, `signal_decay_alert.py`, `api_utils.py`, `connors_rsi2.py`

1. **fin_snipe_manager.py:64** — Change `MIN_STOP_DISTANCE_PCT = 1.0` to `3.0`.
   Also audit keep-existing-stop branch (~line 555-563) for same violation.

2. **signal_decay_alert.py:27,148** — Replace relative `"data/..."` with
   `Path(__file__).resolve().parent.parent / "data"`. Same pattern as
   ic_computation.py fix (2026-05-02).

3. **api_utils.py:60** — Change `apiKey` to `key` to match config_validator schema.

4. **connors_rsi2.py:103** — Change signature from `(df, ticker="", **kwargs)`
   to `(df, context=None, **kwargs)`. Extract ticker from context dict.

**Impact:** No downstream callers break. All are leaf changes.

### Batch 2: Signal engine + trigger (3 files, MED risk)

**Files:** `signal_engine.py`, `trigger.py`

5. **signal_engine.py accuracy_gate clamp** — Add
   `accuracy_gate = max(ACCURACY_GATE_THRESHOLD, accuracy_gate)` after
   config read. Prevents sub-47% override.

6. **signal_engine.py promoted shadow fix** — In the throttle block at
   ~line 3663-3685, if `_promoted_override` is True, skip the
   `status == "shadow"` force-HOLD.

7. **trigger.py:496-509** — Refresh `state["last"]["prices"]` every cycle
   (not only when triggered). Keep `last_trigger_time` separate.

### Batch 3: Agent invocation + grid fisher (2 files, MED risk)

**Files:** `agent_invocation.py`, `grid_fisher.py`

8. **agent_invocation.py kill wedge** — After `taskkill` returns 0/128 and
   `wait(15)` times out, force `_agent_proc = None`. Log critical_errors entry.

9. **grid_fisher.py stop-rearm** — Don't overwrite `inst.stop_loss_id`
   when `new_stop_id is None`. Set `inst.stop_needs_rearm = True`. Log
   critical_errors entry.

10. **agent_invocation.py specialist quorum** — After `wait_for_specialists`,
    if `success_count == 0`, skip synthesis and log `specialist_quorum_fail`.

---

## Additional P1 fixes (if time allows)

- `layer2_exec.py:77`, `layer2_action.py:78`, `layer2_invoke.py:75` —
  Replace `json.load(open(...))` with `load_json()` from file_utils
- Top 5-10 most critical bare `except: pass` blocks in production paths
  (metals_loop.py:928, fish_engine.py:654,686,708)
