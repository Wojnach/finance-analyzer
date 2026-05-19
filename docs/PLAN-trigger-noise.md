# PLAN — Reduce Layer 2 Trigger Noise (invocation budget)

Date: 2026-04-16 (written), 2026-04-17 (filed)
Status: **DEFERRED** — awaiting user decision
Branch: none yet

## Problem

Layer 2 Claude Code invocations eat Max subscription budget. On 2026-04-16:
27 trigger-causes fired across 21 invocations in ~5h. XAU alone produced 10
triggers — all returned HOLD. The system is correct but expensive.

### Trigger breakdown (2026-04-16, 08:27–13:25 UTC)

| Type | Count | % | Example |
|---|---|---|---|
| Consensus crossing (#1) | 13 | 48% | `XAU-USD consensus BUY (30%)` |
| Sustained flip (#2) | 11 | 41% | `XAU-USD flipped BUY->HOLD (sustained)` |
| Other (post-trade, F&G, price) | 3 | 11% | `post-trade reassessment` |

Consensus crossings by confidence:
- ≥60%: 6 (XAG 62/65/80, ETH 80/80, MSTR 68) — all legitimate
- 40-59%: 4 (ETH 42, XAU 48, BTC 62, BTC 80) — borderline to legitimate
- <40%: 3 (XAU 30, 30, 34) — **noise, all returned HOLD**

Sustained flips:
- Direction flips (BUY↔SELL): 4 — meaningful state changes
- Fade flips (*→HOLD): 5 — signal fading, less actionable without position
- Other (SELL→BUY): 2 — meaningful

### Root cause

XAU has 28 applicable signals but in ranging regime (ATR 0.12%) only 5-6 vote.
At `MIN_VOTERS=3`, a 3B/2S split = BUY at 30% confidence. One voter flipping
toggles BUY↔HOLD↔SELL, producing whipsaw triggers on both paths.

## Options

### Option A — Raise MIN_VOTERS for metals (3 → 6)

**File:** `portfolio/signal_engine.py` (line ~1930-1932)

Pros:
- Simple, one constant change
- Stops XAU noise at source — 5-voter consensus becomes HOLD
- Reduces both consensus AND flip triggers for metals

Cons:
- **Per-ticker magic** — metals at 6, crypto/stocks at 3; no principled boundary
- **SwingTrader collateral** — reads same consensus; could miss real metals setups
  during thin-voter overnight phases
- **Delays real breakouts** — XAU moves historically start with 4-5 voters before
  momentum/volume catch up
- Maintenance debt: needs periodic re-validation as signals are added/removed

Estimated impact (2026-04-16): blocks ~8/10 XAU triggers

### Option B — Trigger-level confidence floor (≥0.40)

**File:** `portfolio/trigger.py` §1 (line ~197-200)

Change: in check_triggers(), gate consensus-crossing trigger (#1) to only fire
when `conf >= 0.40`:

```python
if action in ("BUY", "SELL") and last_tc == "HOLD":
    conf = sig.get("confidence", 0)
    if conf >= TRIGGER_MIN_CONFIDENCE:  # NEW: 0.40
        reasons.append(f"{ticker} consensus {action} ({conf:.0%})")
    triggered_consensus[ticker] = action  # always update baseline
```

Pros:
- One-constant change, low blast radius (trigger.py only)
- Uniform across all tickers — no per-ticker magic
- Preserves contrarian F&G-23 entries (42-44% → pass)
- Consistent with `feedback_no_signal_inversion` (gate, not invert)

Cons:
- **Only ~11% invocation reduction** (3 of 27 triggers blocked)
- Sustained-flip triggers (#2) bypass the floor — 11 of 27 unaffected
- Modest budget relief for the change
- 48% XAU trigger still fires
- Semantic gap: `signal_log.action=BUY` but `trigger_state` didn't fire → may
  confuse `layer2_journal_activity` contract monitor
- Cliff edge at 0.40: 41% triggers, 39% doesn't

Estimated impact (2026-04-16): blocks 3 XAU triggers (30%, 30%, 34%)

### Option C — Per-ticker re-trigger cooldown (90 min)

**File:** `portfolio/trigger.py` + `data/trigger_state.json` schema

Change: track `triggered_consensus_ts[ticker]` with last-trigger timestamp;
suppress same-ticker consensus triggers within `CONSENSUS_RETRIGGER_COOLDOWN_S`.

Pros:
- Targets the actual pattern (whipsaw = repeat triggers)
- First trigger always fires; only repeats suppressed
- Preserves novel events (price move, F&G crossing, new ticker)

Cons:
- **USER REJECTED — no hard cooldowns**
- Could miss real gap mid-cooldown
- Compounds with existing sustained_flip debounce
- Arbitrary time constant (why 90 min, not 60 or 120?)

**Status: RULED OUT per user constraint**

### Option D — Fix consensus semantics (require real majority)

**File:** `portfolio/signal_engine.py` (lines ~1928-1942, ~2121-2137)

Change: require `abs(buy - sell) >= max(2, active_voters // 3)` for non-HOLD.
A 5-voter 3B/2S (majority=1) resolves to HOLD. 7-voter 5B/2S (majority=3)
resolves to BUY.

```python
# After computing buy, sell, active_voters:
min_majority = max(2, active_voters // 3)
if abs(buy - sell) < min_majority:
    action = "HOLD"
```

Pros:
- **Root-cause fix** — whipsaws stop at source, cascades into both trigger paths
- No cooldowns, no per-ticker magic, one conceptual rule
- Principled: "3B/2S in 5 voters" genuinely IS noise
- Thematically aligned with the accuracy-gating work already in-flight

Cons:
- **Largest blast radius** — signal_engine.py is read by SwingTrader, autonomous
  mode, Layer 2, dashboard, accuracy tracker, signal_log
- Overnight metals: only 4-5 voters awake → real signals suppressed when actually
  legitimate (lower participation != lower conviction)
- Test surface: 350+ signal_engine tests may need assertion updates
- Changes historical accuracy baselines (accuracy_cache computed from old semantics)
- Requires careful replay analysis before shipping (like accgate branch did)

Estimated impact (2026-04-16): eliminates most XAU whipsaw + most fade flips

### Option E — Do nothing

Pros:
- Zero regression risk
- System IS correct (all XAU triggers returned HOLD)
- Accgate branch (merging soon) may shift the pattern on its own

Cons:
- Subscription budget continues draining on noise
- ~10 XAU HOLD messages/day → annoying Telegram volume

### Option G — Drop fade-flips (*→HOLD) when no position held

**File:** `portfolio/trigger.py` §2 (lines ~225-229)

Change: if current_action == "HOLD" and no portfolio holds the ticker, suppress
the sustained-flip trigger.

Pros:
- Addresses 5 of 11 sustained-flips without cooldowns
- Preserves fade-to-HOLD when you're long (important re-evaluate signal)
- No timing gates

Cons:
- Position-aware trigger = new coupling between trigger.py and portfolio state
  (currently stateless; portfolio files not read by trigger)
- Which portfolio counts? Patient + Bold + SwingTrader + Warrants — needs rule
- Architecture smell: trigger becomes entangled with portfolio management

## Recommendation

**If budget is the primary constraint:**
- Best single option: **D** (root-cause, largest effect, no cooldowns)
- Best low-risk option: **B** (modest effect, minimal blast radius)
- Best combination: **B + G** (handles both trigger paths, no cooldowns,
  moderate blast radius)

**If risk is the primary constraint:**
- **E** (do nothing), then measure after accgate branch merges

## Pre-implementation checklist (whichever option)

1. Merge `fix/accuracy-gating-20260416` branch first (pending in accgate worktree)
2. Measure trigger cadence post-accgate for 24-48h
3. If still noisy, create worktree on `fix/trigger-noise-<date>`
4. Implement chosen option with tests
5. Run replay script (`scripts/replay_consensus.py`) to measure impact on
   historical 14d window
6. Codex adversarial review
7. Merge, push, restart loops

## Data references

- Raw trigger data: `data/layer2_journal.jsonl` (filter 2026-04-16)
- Telegram traffic: `data/telegram_messages.jsonl` (filter 2026-04-16)
- Trigger code: `portfolio/trigger.py`
- Consensus code: `portfolio/signal_engine.py` lines 1878-2160
- Signal engine tests: `tests/test_signal_engine.py` (350+ tests)
