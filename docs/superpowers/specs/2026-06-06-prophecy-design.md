# Prophecy — Daily AI Price-Prediction System

**Date:** 2026-06-06
**Status:** Approved for build (user said "go ahead and implement" ×4)
**Branch:** `prophecy/build-2026-06-06` (isolated worktree `Q:/finance-analyzer-prophecy`)
**Implementation vehicle:** `/fgl`
**Effort:** ultracode / Opus 4.8 / max effort

---

## 1. Goal

Every day at **10:00 Europe/Stockholm**, an isolated scheduled job runs Claude Code
(most-advanced model = **`claude-opus-4-8`**, max effort = **ultracode**, using
**`/deep-research`**) to predict, for every tracked instrument:

- **direction** (up / down / flat)
- **target price point**
- **P(direction)** + **confidence**
- a **low–high range**

across **10 fixed horizons**: `1d (today), 2d, 3d, 4d, 5d, 6d, 7d, 1mo, 2mo, 6mo`.

Predictions use **as much of the system's stored signals + equations as possible**
AND **fresh online + forum sentiment** ("what people think"), with a **unique
predictive playbook per instrument**. Results are journaled (append-only),
snapshotted for the dashboard/website, accuracy-scored against realized prices,
and the **token/cost of each run is tracked**.

### Cost posture
"Go unhinged to begin with" — no token budget cap in phase 1. Full `/deep-research`
per instrument, Opus 4.8, max effort. Cost is **measured** (not capped) so it can be
controlled later if it proves too expensive.

---

## 2. Naming & separation (collision resolution)

`prophecy` already exists and is load-bearing:
- `portfolio/prophecy.py` — macro-beliefs module (`load_beliefs`, `save_beliefs`,
  `add_belief`, `get_context_for_layer2`, …), imported by `main.py`, `reporting.py`,
  `signals/news_event.py`.
- `data/prophecy.json` — macro-beliefs store.
- `--prophecy-review` CLI.

**These stay 100% untouched.** The new system is a fully separate package:

| Concern | Existing (DO NOT TOUCH) | New "Prophecy" predictor |
|---|---|---|
| Python package | `portfolio/prophecy.py` (`portfolio.prophecy`) | `prophecy/` at repo root (`prophecy.*`) |
| Data | `data/prophecy.json` | `data/prophecy/` subdir |

Root package `prophecy/` does not clash with `portfolio.prophecy` (different import
paths). `data/prophecy/` dir coexists with the `data/prophecy.json` file.

---

## 3. Instrument universe ("everything price-tracked")

13 instruments, each with an independent **enable flag** (all `true` at launch;
flip to `false` to silence Prophecy for that instrument):

| Group | Instruments |
|---|---|
| Crypto | BTC-USD, ETH-USD |
| Metals | XAU-USD, XAG-USD |
| US stock | MSTR |
| Oil | CL=F (WTI), BZ=F (Brent) |
| Warrants | XBT-TRACKER (→BTC), ETH-TRACKER (→ETH), MINI-SILVER (→XAG) |
| Tier-2 equities | SAAB-B, SEB-C, INVE-B |

Canonical list + flags live in `data/prophecy/config.json`. `prep.py` and the prompt
both honor `enabled`; disabled instruments are skipped end-to-end (no context, no
Claude spend, no journal row).

---

## 4. Architecture (script-first; Claude only where it must reason)

```
PF-Prophecy (Win Task, 10:00 CET)
        │
        ▼
scripts/prophecy-daily.bat
  1. prep.py        [ZERO tokens]  gather stored signals+equations + live prices
  2. claude -p      [Claude]       /deep-research per enabled instrument → raw JSON
  3. publish.py     [ZERO tokens]  validate → journal + latest snapshot
  4. outcomes.py    [ZERO tokens]  backfill realized px at matured horizons → accuracy
  5. cost.py        [ZERO tokens]  parse token usage → cost_log
```

### 4.1 `prophecy/prep.py` (zero tokens)
For each **enabled** instrument, assemble a compact context bundle:
- **live price** (Binance FAPI / Alpaca / `portfolio.price_source` — never cached, per
  Critical Rule #3),
- **stored signals + equations**: pull from `data/agent_summary.json`,
  `accuracy_cache.json`, regime state, on-chain (BTC), cross-asset (metals),
  microstructure, momentum/mean-reversion outputs — *as much as exists for that ticker*,
- **macro beliefs** via `portfolio.prophecy.get_context_for_layer2()` (read-only reuse),
- **recent research** (`daily_research_*.json`, `morning_briefing.json`),
- the instrument's **playbook** (§5).

Writes `data/prophecy/context_<date>.json` (one object keyed by instrument).

### 4.2 Claude run (`docs/prophecy-prompt.md` piped to `claude -p`)
- `--model claude-opus-4-8 --verbose --output-format stream-json` (token usage in tail).
- Prompt embeds the keyword **`ultracode`** (enables Workflow orchestration) and
  instructs the agent to, **per enabled instrument**:
  1. load its context bundle + playbook,
  2. run **`/deep-research`** (web) on the playbook's research questions,
  3. search **forums** for crowd sentiment (sources per playbook),
  4. fuse stored signals + equations + research + sentiment via the instrument's
     unique strategy,
  5. emit predictions for all 10 horizons in the **strict JSON schema** (§6),
- Writes raw JSON to `data/prophecy/raw_<date>.json` + stream log to
  `data/prophecy/run_<date>.log`.

### 4.3 `prophecy/publish.py` (zero tokens)
Validate raw JSON against schema (reject/repair malformed horizons), stamp
`spot_at_prediction`, append each instrument record to
`data/prophecy/prediction_journal.jsonl` (atomic), write `data/prophecy/latest.json`
(dashboard snapshot). Uses `portfolio.file_utils` atomic I/O (Critical Rule #4).

### 4.4 `prophecy/outcomes.py` (zero tokens)
For matured horizons (now ≥ prediction_ts + horizon), pull realized price, compute:
- directional hit (predicted dir == realized sign),
- target error (|target − realized| / realized),
- Brier score on `prob_up`.
Append to `data/prophecy/accuracy.jsonl`, roll up `data/prophecy/accuracy.json`
(per-instrument × per-horizon hit-rate, MAE, Brier, n). Reuses
`portfolio.outcome_tracker` / `forecast_accuracy` patterns where possible.

### 4.5 `prophecy/cost.py` (zero tokens)
Parse token counts from the stream-json tail (input/output/cache tokens), price with
Opus 4.8 rates, append `data/prophecy/cost_log.jsonl` (per-run + per-instrument est.),
roll up totals into `latest.json.cost_summary`. This is the lever for the later
"control the spend" decision.

---

## 5. Per-instrument strategy playbooks (`prophecy/strategies.py`)

Each instrument has a unique `Playbook`: `{signal_emphasis, equations, web_questions,
forum_sources, special_factors, price_model}`. Highlights:

- **BTC-USD** — on-chain (MVRV-Z, SOPR, NUPL, exchange netflow), spot-ETF flows, F&G
  contrarian, funding/basis, cycle position; crowd: r/Bitcoin, crypto-Twitter, TradingView.
- **ETH-USD** — ETH/BTC ratio momentum, staking yield/withdrawals, L2/gas activity,
  DeFi TVL, ETF flows, BTC-beta + independent catalysts; crowd: r/ethereum, r/ethfinance.
- **XAU-USD** — real yields (10y TIPS), DXY inverse, central-bank buying, ETF holdings,
  geopolitical premium, seasonality; crowd: Kitco, r/Gold.
- **XAG-USD** — Gold-Silver Ratio mean-reversion (primary equation), industrial/solar
  demand, COT positioning, mine-supply deficit, high gold-beta; crowd: r/Silverbugs.
- **MSTR** — mNAV premium/discount to BTC holdings, BTC beta (~1.5–2×), convertible
  debt/dilution, equity issuance, options skew; crowd: r/MSTR, StockTwits.
- **CL=F (WTI)** — OPEC+ decisions, EIA inventories, term structure
  (contango/backwardation), refinery margins, USD, China demand; crowd: oil Twitter.
- **BZ=F (Brent)** — global supply/geopolitics (Hormuz, Russia), WTI-Brent spread,
  freight, Asian demand.
- **Warrants (XBT/ETH/MINI-SILVER)** — predict **underlying** first, then apply
  turbo/mini pricing `P=(S−K)·FX/r`, leverage `Ω=S/(S−K)`; flag barrier-proximity risk
  (pull parity/barrier/strike live from Avanza). Inherit underlying's crowd sentiment.
- **Tier-2 equities** — SAAB-B (defense spend/orders), SEB-C (rates/credit/NIM),
  INVE-B (holding-co NAV discount); earnings calendar, OMX correlation; crowd: Placera,
  Avanza forum, Flashback finans.

The playbook drives BOTH `prep.py` (what stored signals to bundle) and the prompt
(what to research / how to fuse).

---

## 6. Data schemas (`prophecy/schema.py`)

**Journal record** (`prediction_journal.jsonl`, one line per instrument per run):
```json
{
  "schema_version": 1,
  "run_id": "prophecy-2026-06-06T08:00:00Z",
  "ts": "2026-06-06T08:05:12Z",
  "date": "2026-06-06",
  "instrument": "BTC-USD",
  "strategy": "btc_onchain_flow_cycle",
  "spot_at_prediction": 61000.0,
  "spot_source": "binance_fapi",
  "model": "claude-opus-4-8",
  "regime": "trending-down",
  "horizons": {
    "1d":  {"direction":"down","target":59500,"prob_up":0.35,"prob_down":0.55,"prob_flat":0.10,"confidence":0.60,"low":58000,"high":61000,"rationale":"..."},
    "...": {},
    "6mo": {"direction":"up","target":78000,"prob_up":0.58,"prob_down":0.32,"prob_flat":0.10,"confidence":0.45,"low":52000,"high":95000,"rationale":"..."}
  },
  "key_drivers": ["ETF outflows 13d", "MVRV 1.2 accumulation"],
  "stored_signals_used": ["rsi","mvrv_z","fear_greed","drift_regime_gate"],
  "web_sources": ["https://..."],
  "forum_sentiment": {"net":"bearish","score":-0.4,"sources":["r/Bitcoin","TradingView"]},
  "deep_research_summary": "...",
  "coverage": {
    "data_sufficiency": "high",
    "has_proper_equation": true,
    "missing_inputs": [],
    "low_confidence_horizons": ["6mo"],
    "needs_work": false,
    "note": ""
  },
  "cost": {"input_tokens":0,"output_tokens":0,"cache_read_tokens":0,"est_usd":0.0}
}
```

### 6.1 Coverage / data-sufficiency flag (visible gap-tracking)

Every record carries a **`coverage`** block so the user can see at a glance *where the
system lacks the data or equations to make a credible prediction* — and therefore where
more engineering work is needed:

- `data_sufficiency`: `high | medium | low | insufficient` — graded from how many of the
  playbook's required inputs `prep.py` actually found live (computed deterministically in
  prep, then refined by Claude with what it could/couldn't research).
- `has_proper_equation`: `false` when the instrument has no validated predictive
  model/equation for the horizon set (e.g. Tier-2 equities with no signal feed, or a
  warrant whose barrier/parity couldn't be pulled).
- `missing_inputs`: explicit list (e.g. `["no on-chain feed","Avanza barrier unavailable"]`).
- `low_confidence_horizons`: horizons where the prediction is a near-coin-flip.
- `needs_work`: `true` when `data_sufficiency ∈ {low, insufficient}` OR
  `has_proper_equation == false`. **This is the column the user scans.**

`prep.py` seeds `coverage` deterministically (it knows what feeds were empty); the Claude
run may downgrade it further (e.g. deep-research turned up nothing). `publish.py` never
upgrades a `needs_work=true` to `false`.

**Dashboard:** the prophecy table renders a **⚠ "needs work"** column driven by
`coverage.needs_work`, plus a tooltip listing `missing_inputs`. `latest.json` rolls up a
top-level `coverage_summary` (`{instruments_needing_work: [...]}`) so gaps are one glance
away.

**`config.json`** — `{ "model": "claude-opus-4-8", "horizons": [...10...],
"instruments": { "BTC-USD": {"enabled": true, "strategy": "btc_onchain_flow_cycle"}, ... },
"budget_usd_soft_cap": null }`.

**`latest.json`** — `{date, generated_at, cost_summary, instruments:{INSTR: <record>}}`.

**`accuracy.json`** — `{INSTR: {horizon: {n, dir_hit_rate, target_mae, brier}}}`.

---

## 7. Scheduling (Windows — your preference)

- `scripts/prophecy-daily.bat` mirrors `after-hours-research.bat` conventions
  (progress JSON, JSONL log, timestamps, exit-code handling). Sets `CLAUDECODE=` /
  `CLAUDE_CODE_ENTRYPOINT=` to detach.
- `scripts/win/install-prophecy-task.ps1` registers **`PF-Prophecy`**, daily **10:00**
  Europe/Stockholm. **User runs this** (admin, on Windows). This task is the ONLY guard
  — the `.bat` calls `claude -p` directly and bypasses `claude_gate`, identical to the
  existing research `.bat`s. It is **independent of the frozen tasks**; the freeze is
  not touched.

---

## 8. Dashboard (`/api/prophecy`)

One additive route in `dashboard/app.py` serving `data/prophecy/latest.json` +
`accuracy.json` + `cost_summary`. Tile: per-instrument horizon grid (direction arrows,
target, P, confidence), plus running cost + accuracy. No existing route touched.

---

## 9. Out of scope / deferred

- **Tests** — user: "don't start with the test until I say so." Code is written
  test-ready (pure functions, injectable paths) but **no tests written/run** until
  greenlit.
- **Phase 2 — loop integration** — user: "once this is up and running … then we will
  also want to change the existing loops to read from this data." Wiring `main.py` /
  Layer 2 / metals loop to consume `data/prophecy/latest.json` is a **separate later
  phase**, not built now.
- **Token budget enforcement** — measured now, enforced later if needed.

## 10. Risks

- **Cost** — 13 instruments × full `/deep-research` × Opus 4.8 daily is the most
  expensive run shape, during a token freeze. Mitigated by: explicit user opt-in,
  per-instrument enable flags, and built-in cost tracking to inform throttling.
- **Schema drift from Claude** — `publish.py` validates + quarantines malformed output
  rather than corrupting the journal.
- **Worktree symlink gap** — `config.json` symlink absent in worktree (known); affects
  only runtime, not the build. Runtime executes from main checkout after merge.
