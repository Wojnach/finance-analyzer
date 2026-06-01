# FGL Adversarial Codebase Review — Synthesis (2026-06-01)

**Scope:** 8 subsystems, ~50K LOC of curated production hot-path code, reviewed by
8 fresh Claude Code subagents in parallel (each on an empty-baseline worktree PR),
plus an independent author pass, then cross-critiqued and severity-re-graded here on
a money/state/auth/silent-failure rubric (not the subagents' self-assigned severity).
Read-only review — no production code changed. Per-subsystem reports: `docs/fgl-review/<sub>.md`.

Reviewers: signals-core, orchestration, portfolio-risk, signals-modules, infrastructure
(pr-review-toolkit:code-reviewer); avanza-api, data-external (caveman:cavecrew-reviewer);
metals-core (pr-review-toolkit). Independent pass: `_independent-pass.md`.

---

## 0. The thing that opened this session — corrected

The mandatory startup check reported **"12 unresolved critical errors,"** prominently
`portfolio_state_corrupt` (×6) at 15:11–15:19 today. **This is a false alarm.** Three
reviewers independently + my own verification converged:

- All **6/6** `portfolio_state_corrupt` entries have `context.path` inside
  `…\Temp\pytest-of-herc2\pytest-2873\popen-gw3\…` with byte counts 4/7/8/12/15 — they
  are **deliberately-corrupt pytest fixtures**, not the live files.
- Live `data/portfolio_state.json` and `_bold.json` both parse clean (6 keys each, UTF-8).
  Zero `.bak`/`.corrupt-*` files in live `data/`. **Nothing was wiped.**

**Why it leaked (the real bug):** the state-resilience fix merged earlier today added a
side-effect to the read path — `_load_state_from → _quarantine_corrupt_state →
atomic_append_jsonl(CRITICAL_ERRORS_LOG, …)` where `CRITICAL_ERRORS_LOG` is the
module-level constant pointing at the **live** journal (`portfolio_mgr.py:24,115,175`).
The pre-existing corruption tests patch `STATE_FILE` to `tmp_path` but **not**
`CRITICAL_ERRORS_LOG`, so every `pytest` run appends real `critical` entries to the live
journal — which the CLAUDE.md startup gate and `PF-FixAgentDispatcher` then read as live
incidents. (The new `test_portfolio_mgr_corrupt_quarantine.py` patches it correctly; the
3 older tests were never updated.) The accuracy_degradation entries are genuine but
benign (already force-HOLD per the gate; self-clear as the 14d baseline rolls).

> Action taken at end of this review: 6 `resolution` lines appended for these phantom
> timestamps to clear the gate. The durable fix (patch the 3 tests) is logged below.

---

## 1. Master findings table (re-graded severity)

| # | Sev | Subsystem(s) | Finding | Converging reviewers |
|---|-----|--------------|---------|----------------------|
| 1 | **P0** | metals-core | `update_trailing_stops`/`_rebuild_stop_orders_for` (`metals_loop.py:5148,2447`) **cancel broker stops before placing** new ones, commit `if orders:`, no rollback → if all re-places reject, the live 5x position is left **naked**. | metals |
| 2 | **P0** | metals-core | Fish-engine BUY (`metals_loop.py:2936-3055`) opens a 5x cert with **no broker stop** (only in-memory per-tick exit); loop death → unprotected until restart. | metals |
| 3 | **P0** | orchestration, portfolio-risk, infrastructure | **Layer 2 LLM persists trades by hand-editing JSON** via `Edit`/`Write` (`TRADING_PLAYBOOK.md:127`, `agent_invocation.py:1159` `--allowedTools "Edit,Read,Bash,Write"`) — bypasses atomic write, per-file lock, schema validation. The corruption *source*. | independent + portfolio-risk + infra |
| 4 | **P0†** | portfolio-risk, infrastructure | `_rotate_backups` (`portfolio_mgr.py:50-68`) `shutil.copy2`s the **unvalidated current file** into `.bak`→`.bak2`→`.bak3`; a corrupt primary launders through the whole ring → "no backup recovered." Turns recoverable corruption into **permanent silent loss**. †conditional on a real corruption event. | independent + portfolio-risk + infra |
| 5 | **P1** | portfolio-risk, infrastructure | Test-isolation leak pollutes live `critical_errors.jsonl` (see §0) → poisons the #1 startup gate + spawns useless fix-agents. | portfolio-risk + infra |
| 6 | **P1** | infrastructure, portfolio-risk, independent | `load_state` is **lockless** and `threading.Lock` doesn't span the 2+ OS processes; on Windows `os.replace` raises `PermissionError` on a concurrent reader → `load_json`→`None`→treated as **corruption**→triggers the (laundering) recovery. Cross-process primitive (`jsonl_sidecar_lock`) exists but is unused here. | infra + portfolio-risk + independent |
| 7 | **P1** | data-external, portfolio-risk | FX 10× SEK risk: `fx_rates.py:44` hardcoded 10.50 fallback + raw `agent_summary.get("fx_rate")` in `monte_carlo_risk.py:408` & `exit_optimizer.py:718` bypass `_resolve_fx_rate` → portfolio valuations silently off 10%+. | data-external + portfolio-risk |
| 8 | **P1** | avanza-api | `avanza_session.get_positions()` (`:681`) does **not** filter to ISK 1625505 → pension account **2674244 leaks** into positions (verified: extracts `account` but never gates it; `metals_avanza_helpers.py:88` does filter). | avanza |
| 9 | **P1** | orchestration | `autonomous._detect_regime` (`:421`) reads `extra["regime"]` but engine writes `extra["_regime"]` (`signal_engine.py:4429`) → regime **always** the "range-bound" fallback in `layer2.enabled=false` mode; feeds the fishing position-size multiplier. | orchestration |
| 10 | **P1** | signals-modules | `residual_pair_reversion` is **dead (always HOLD)**: `MIN_ROWS=200` vs engine's 100-row frame, AND RangeIndex→1970-epoch vs daily-driver join → 0 rows. Targets the user's focus tickers (ETH/XAG). | signals-modules |
| 11 | **P1** | data-external | `futures_data.py:61,87+` `data["openInterest"]`/comprehension `KeyError` on partial Binance response → caught by `_cached()`→`None` → signal silently loses the metric. | data-external |
| 12 | P2 | orchestration | Two **dead safety-net detectors**: `signal_count_stable` (`loop_contract.py:724` keys on `active_voters`, never set — engine writes `_voters`) and `atomic_write_residue` (residue scan counts dirs, breaks at 100 of 713 `data/` files). Both silently never fire. | orchestration |
| 13 | P2 | metals-core | `get_open_orders` swallows API error→returns `[]` (not None) → `reconcile_against_live` marks resting buys CANCELLED, spurious 6h cooldown + cancel/replace churn (asymmetric vs `get_positions` which raises). | metals |
| 14 | P2 | metals-core | Stop-fill check (`metals_loop.py:5042`) queries the **regular order** endpoint for stop-surface IDs → missed fills → stale `pos["units"]` → double-sell/short-reject risk. | metals |
| 15 | P2 | metals-core | EOD close **hardcoded 21:55 CET** in 4 places (`metals_swing_trader.py:2796,2450`, `fish_engine.py:216`, `grid_fisher.py:287`) vs the "don't hardcode — DST/half-days vary" rule; `session_calendar` already available. | metals |
| 16 | P2 | portfolio-risk | 3 divergent ATR stop levels (`risk_management.py:376` vs `:469/:912`, `monte_carlo.py:305`); only `compute_stop_levels` has the 3% floor / 15% cap → displayed stop ≠ probability stop. | portfolio-risk |
| 17 | P2 | portfolio-risk | `equity_curve.py:361-421` FIFO matcher silently drops unmatched SELL shares → realized P&L undercount, no warning. | portfolio-risk |
| 18 | P2 | data-external | Cache hits served without age metadata (`data_collector.py:289`); longer-TF data (12h@300s, 2d@900s) served as live. yfinance fallback `_source` lost at indicator-cache level (`price_source.py:252`). | data-external |
| 19 | P2 | data-external | `onchain_data.py:286` BGeometrics partial failure (1-of-6 metrics) fires with false confidence; `interpret_onchain` checks `zscore` not `mvrv`. | data-external |
| 20 | P2 | orchestration | Primary L2 invocation (`agent_invocation.py:1224` direct `Popen`) bypasses the `CLAUDE_ENABLED` kill switch + `claude_invocations.jsonl` cost log. | orchestration |
| 21 | P2 | signals-core | `("ml","ETH-USD")` per-ticker override is dead — `ml` vote hardcoded HOLD (`signal_engine.py:3471`), no compute path; CLAUDE.md claims ETH gets the ML vote. | signals-core |

Plus ~12 P3 (naming `_sub_commercial_change`, `volatility.py` √365/√252 mismatch, double-Itô drift `metals_risk.py:171`, `GRID_STOP_PCT=3.5` vs 15% rule, fish BUY at ask, oil_grid stale-cache, unwired accuracy-tier boost `signal_engine.py:516`, `forecast_accuracy.py:142` None-guard, intra-bar candlestick/calendar, alpha_vantage budget-exhaustion logged only WARNING, `save_state` can overwrite corrupt-present with defaults — latent). Full detail in per-subsystem reports.

---

## 2. Thematic clusters

**A. State persistence integrity (findings 3,4,5,6 + P3 save-overwrite).** The dominant
theme. The autonomous system's *own* trade-write path (Layer 2 LLM `Edit`) is the one path
that bypasses every guarantee the infra was built to provide; the backup ring can't save
it because it launders unvalidated bytes; the recovery verdict conflates Windows file-lock
contention with corruption; and the tests that exercise corruption pollute the live alert
journal. The earlier-today defensive fix hardened *recovery* but left the *write path* and
*backup validation* untouched — which is why it recurred. **Fix the write path and this
whole cluster collapses.**

**B. Naked leveraged positions (findings 1,2).** The only *real-money* P0s. Both leave a
live 5x Avanza cert without a broker-side stop under realistic failure (rejected re-place;
loop death). Endpoint invariant and barrier-proximity are otherwise sound — these are gaps
in *when* the stop exists, not *how* it's placed.

**C. Silent degradation / stale-as-live (findings 7,11,13,14,18,19 + P3 oil-cache, av-budget).**
Many external/reconcile paths swallow a failure and return empty/stale/`None`/hardcoded,
which downstream code treats as fresh truth. Violates "live prices first." Individually
P1/P2; collectively a reliability pattern worth a lint rule (never `except: return []/None`
on a data-fetch without a staleness signal).

**D. Believed-active-but-dead logic (findings 9,10,12,21).** Four places where code the
maintainers (and CLAUDE.md) believe is contributing is silently inert: two signals on the
user's focus tickers, two loop-safety detectors, one regime feed. No crash, no money loss —
but the system's actual behavior diverges from its documented/believed behavior.

---

## 3. Verified SAFE (do not re-litigate)

- **Dashboard auth**: CF-Access JWT properly verified (RS256/JWKS, aud+iss+email), fails
  CLOSED on cold-start, `hmac.compare_digest`; every API route `@require_auth`; path-traversal
  guarded. No bypass (infra).
- **Stop-loss endpoint invariant holds** — all metals stop paths use `/_api/trading/stoploss/new`;
  regular `order/new` only for entries (metals). Mar-3 incident class not present.
- **Accuracy gate is force-HOLD, not inverted** (signals-core, independent); shadow votes
  excluded from live tally; no lookahead in outcome backfill.
- **Atomic primitives themselves are correct** — `atomic_write_json`/`atomic_append_jsonl`
  (tmp+fsync+os.replace; cross-process sidecar lock). The bug is writers that don't use them.
- **Monte Carlo cores** (GBM, t-copula, Cholesky PSD fallback, antithetic variates) correct
  (portfolio-risk). `gpu_gate` lock, `http_retry` token-redaction + 429 handling, telegram
  send-failure handling — all sound (infra).
- **Subprocess hardening**: `CLAUDECODE` stripped before spawn, no `--bare`, `detect_auth_failure`
  on both completion + timeout-kill paths, crash-backoff floor (orchestration).
- **11/12 signal modules**: directions correct, denominators guarded, short-series→HOLD,
  out-of-range confidence clamped by the engine (signals-modules).

## 4. Out of scope (curation gaps — visible, not silent)

Excluded from review: ~100K LOC of `scripts/` one-offs, backtests, `golddigger`/`elongir`/
`mstr_loop` bots, `dashboard/app.py` beyond auth, the 64 non-curated signal modules, LLM
inference servers. A bug in any of these was not looked for this pass.

---

## 5. Recommended fix order (next implementation session)

1. **Clear the false alarm + stop the leak** (P1 #5): patch `CRITICAL_ERRORS_LOG` to
   `tmp_path` in the 3 old corruption tests (autouse conftest fixture); resolution lines
   already appended. *Trivial, unblocks the startup gate.*
2. **Naked-position P0s** (#1,#2): place-before-cancel (or snapshot+re-arm rollback) in the
   stop rebuild; place a HW stop immediately after every fish BUY. *Real money.*
3. **Layer 2 atomic persistence** (#3): add `python -m portfolio.layer2_apply …` routing
   through `update_state()`; rewrite playbook §Execute to call it; drop `Edit,Write` from the
   L2 allow-list. **Kills the corruption cluster at the source.**
4. **Validate-before-rotate** (#4) + distinguish transient `OSError` from `JSONDecodeError`
   in `_load_state_from` (#6); take a cross-process lock on state read/write.
5. **FX 10× guard** (#7): route all fx reads through `_resolve_fx_rate`.
6. Then the P1 tail (#8 pension filter, #9 regime key, #10 residual_pair, #11 futures KeyError)
   and the dead-detector P2s (#12).

Each is additive/reversible and testable; none requires touching live signal weights/thresholds.
