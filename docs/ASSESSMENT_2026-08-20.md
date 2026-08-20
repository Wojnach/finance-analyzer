# Situation assessment — 2026-08-20

Multi-agent research pass (crypto / semis / macro / test-triage) plus local
system health check. Figures stamped ~08:30–09:20 UTC.

---

## 0. Three of my own hypotheses were killed. Recording that first.

| My claim                                                    | Verdict          | Evidence that killed it                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The equity dispersion is a **balance-sheet / credit** split | **DEAD**         | Independently rejected twice. `corr(ND/EBITDA, Aug19 move) = +0.09` (semis, n=19); `Spearman = −0.224, p=0.371` (macro). CRWV at **10.80x** ND/EBITDA fell only −2.5%; WDC (**net cash**, +$3.5B FCF) fell −6.9%; AAPL (most net debt, −$44B) rose **+2.2%**. NBIS, which I called the flagship levered casualty, carries net debt of just **3.8% of market cap** and is +199% YTD. |
| A **debasement / hard-asset** bid is driving crypto+metals  | **Mostly wrong** | Debasement requires _rising_ breakevens with _falling_ real yields. `T10YIE` unchanged at **2.30**; `DFII30` at **3.03–3.06 = all-time series high**. Also XAG is −42.0% below its Jan ATH and XAU −14.5%: that trade peaked in January and crashed.                                                                                                                                |
| ETH is **not overbought**, so it has more room              | **WRONG**        | I quoted `signals[].rsi` = the **`Now`** horizon bucket (65.6). The daily-equivalent bucket is **83.2**, matching my independent Wilder RSI of 83.5. ETH 4h RSI is **93.4**. ETH is the _most_ overbought instrument in the book, on every timeframe 1h→1d.                                                                                                                         |

Note the RSI item is a **field-selection error of mine, not a code bug** — the
agent that flagged it diagnosed a broken engine. `agent_summary.timeframes[tkr]`
is a list keyed by `horizon`: `Now, 12h, 2d, 7d, 1mo, 3mo, 6mo`. Every bucket
matches an independent recomputation to within ~0.3 pt. The engine is correct.

## What actually drove the divergence

**The real cross-sectional variable is the prior squeeze, not leverage.**

| Variable                   | r vs the 2-day unwind | r²        |
| -------------------------- | --------------------- | --------- |
| **Prior run-up Aug 12→17** | **−0.659**            | **0.434** |
| Distance from 52w high     | +0.557                | 0.310     |
| YTD return                 | −0.369                | 0.136     |
| ND/EBITDA                  | −0.151                | 0.023     |

SNDK ran +40.6% then gave back −12.2%; NBIS +39.1% → −16.7%; AAPL/GOOGL never
ran and never fell. 43% of the variance is the run-up; leverage is 2%.

**Vol-normalised magnitudes settle the macro-vs-crypto question:**

| Asset | 24h     | own 1d σ | in σ       |
| ----- | ------- | -------- | ---------- |
| BTC   | +10.96% | 1.79%    | **+6.12σ** |
| ETH   | +18.31% | 3.33%    | **+5.49σ** |
| XAG   | +5.65%  | 2.40%    | +2.36σ     |
| XAU   | +3.08%  | 1.42%    | +2.17σ     |

Metals are the _highest-beta_ clean expression of a debasement factor, so they
cannot lag crypto by 2.5x if that factor is the driver. Instead they **price the
macro impulse for us at ~2.2σ**. Crypto did 5.8σ. Decomposition: **~38% macro,
~62% crypto-idiosyncratic.**

## The catalysts, primary-sourced

- **Treasury, 2026-08-19** (`home.treasury.gov/news/press-releases/sb0607`,
  fetched): long-end liquidity-support buybacks doubled from **$2B to "at least
  $4B" per operation**, 10-20y and 20-30y sectors, **effective Sept 9 through
  Nov 4**. Rationale cited is _strong_ sponsorship, not distress. **Nothing has
  been bought yet.** Result: 30y −9.1bp, 10y −5.3bp, 5y only −1.4bp, DXY −0.82%
  to a 3-month low. A long-end-only term-premium move with zero easing content.
- Context: 30y closed **5.309 on Aug 17, highest since 2007-06-12**.
- **July FOMC minutes released the same day, Aug 19**: 9-3 hold with **three
  dissents preferring a 25bp HIKE**. Core PCE 3.3% (June). `2y 4.19 − EFFR 3.63
= +56bp`: **the market prices hikes, not cuts.**
- **The Fed Chair is Kevin Warsh**, sworn in **2026-05-22**
  (`federalreserve.gov/newsevents/pressreleases/other20260522a.htm`). Powell is
  a Board member only. Any thesis premised on Powell or on easing is inverted.
- **Crypto-specific cluster:** SEC proposes "Regulation Crypto Assets"
  Aug 18 (Chairman Atkins; $5M/4yr and $75M/12mo exemptions, 60-day comment) —
  _sec.gov 403s all tooling, confirmed only via a verbatim mirror_. White House
  crypto summit Aug 19; Trump pressing the Clarity Act, **Senate cloture
  Sept 15**. Record short liquidation, **$1.44–1.6B, ~87% shorts** — third-hand
  aggregator data, Coinglass unreachable.
- **Equity leg was Aug 18, BEFORE the Treasury news.** Aug 19 was risk-**ON**:
  SPX +0.21%, AAPL +2.2%, yields down, DXY at a 3-month low. A rates-driven
  selloff must reverse when rates fall; it didn't. Duration is rejected for Aug 19.

## There is NO AI-capex cut. Guidance was RAISED.

Alphabet **$195–205B** · Amazon **~$220B** · Meta **$130–145B** · Microsoft
CY26 **$190B** (vs $152B consensus). Five-hyperscaler FY26 aggregate **>$690B**,
guided **>$900B by FY28**, raised twice this spring. **Meta's raise explicitly
cites "higher component pricing"** — a customer raising its budget because your
product got more expensive is the opposite of a demand warning.

The circulating "30-50% of 2026 projects cancelled" dataset is a **2026-04-01**
Bloomberg piece using Sightline Climate data, rebutted by SemiAnalysis
("YE2026 NA hyperscaler self-build forecast moved ~1%"). Four months stale.

One real structural change: hyperscalers shifted to external funding —
incremental annual debt went from **9% of capex in FY24 to 32% LTM**; Alphabet
priced an **$84.75B equity raise** in June 2026. Capex is being _financed_
differently, not cut. Slow-burn risk, not a 72h catalyst.

## 2026-07-29 was the bottom

Every name in the book is **+6% to +54%** above its late-July low. NBIS +51%,
CRWV +49%, SNDK +54% — off the exact day CRWV's CDS implied a ~50% default
probability. Not one name is at a new low. This is a **two-day giveback inside a
three-week recovery**, which is also _why_ the squeeze variable dominates.

---

## Positions — recommended actions

| Sleeve                       | % book | P&L    | Action                                  |
| ---------------------------- | ------ | ------ | --------------------------------------- |
| Crypto (ETH 2.13:1 over BTC) | 20.4%  | +21.6% | **ROTATE ETH → BTC**, keep total weight |
| AI compute core              | 31.0%  | +17.0% | HOLD, don't add pre-NVDA                |
| Memory/NAND                  | 16.6%  | −23.7% | HOLD                                    |
| HDD                          | 8.3%   | −24.2% | HOLD                                    |
| AI server/interconnect       | 7.3%   | −17.5% | SPLIT — DELL & AVGO print Sept 1/2      |
| Neocloud                     | 3.1%   | −12.7% | **REDUCE**                              |
| Semicap                      | 3.0%   | −34.0% | **HOLD or ADD**                         |

**Neocloud is the one place leverage genuinely IS the thesis:** CRWV is S&P
**B+**, CDS ~855bp, ND/EBITDA ≥10.8x and _understated_ (excludes the $8.5B and
$3.1B 2026 facilities), interest expense doubled to $536M, FCF −$4.71B; the May
DDTL 5.0 priced **Ba2/BB+**, down from A3 in March. APLD has **negative
EBITDA**. NBIS announced a **$4.5B convert on 2026-08-19**, its third raise this
year. Trim here, surgically.

**Semicap is the inverse:** worst P&L, best quality. LRCX **net cash +$1.8B,
+$4.9B FCF**; KLAC **0.20x**. Both sold on momentum on a day broad credit didn't
move. HY OAS sits at the **15th percentile (2.75%)** and is +1bp over 63 days.

### The storage thesis is incoherent as currently held

The complex is **223,585 SEK = 35.4% of the book** and **all four legs are
losing** (MU −19.9%, SNDK −28.6%, STX −17.2%, WDC −30.8%). A correctly-expressed
substitution thesis yields one winner and one loser. Four losers proves
substitution is not the axis moving the money.

HDD fell harder on the day because of **customer concentration, not
substitution** — WDC discloses **89% of revenue from cloud, 73% from its top ten
customers**, so it has the highest AI-capex beta in the group. And on 1 week NAND
_beat_ HDD (SNDK +16.7%, MU +2.8% vs WDC +1.8%, STX −5.2%).

The actual economics run the other way: NAND contract pricing Q3-26 is **flat to
−5%** with capacity added at all five suppliers, while HDD ASP/exabyte is **up
four consecutive quarters** and **WDC is sold out for calendar 2026**. QLC still
costs **22.6x HDD per TB at 30TB**. On current evidence the spread is **long HDD
/ short NAND — the opposite of the stated thesis.**

Both HDD names are _accelerating_: WDC Q4FY26 revenue **+44% YoY**, GM 54.1%,
guiding Q1FY27 **+42-49% YoY** with EPS $4.00 (8-K, 2026-08-05, pulled from SEC
directly). STX 13th consecutive quarter of margin gains, guiding EPS **+56% YoY**.
On company-guided EPS only: WDC de-rated **46.6x → 28.9x**, STX **37.5x → 28.5x**.

Two coherent choices: express it as a spread, or stop calling it a thesis and
size it as what it is — a concentrated 35% long in AI storage capex.

## Calendar — the window is dense

| Date          | Event                                                                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Aug 26**    | **NVDA Q2 FY27**, 2:00pm PT (primary: investor.nvidia.com PR 2026-07-29)                                                                                          |
| **Aug 26**    | **BEA Personal Income & Outlays (PCE)** + GDP 2nd est. — live macro catalyst with core PCE 3.3% and three hike dissents                                           |
| **Aug 27-29** | **Jackson Hole** — theme "Financial Innovation: Implications for Payments and Policy"; **Warsh's first address as Chair, Aug 28** (secondary only; bls/kcfed 403) |
| **Sept 1**    | **DELL** Q2 FY27 — direct test of the memory-margin squeeze (non-GAAP GM 21.6%→18.1%)                                                                             |
| **Sept 2**    | **AVGO** Q3 FY26 — direct test of the $370B lease-backstop story                                                                                                  |
| **Sept 9**    | Treasury's enlarged buybacks actually **begin**                                                                                                                   |
| **Sept 15**   | Senate cloture vote, Clarity Act                                                                                                                                  |
|               | **No FOMC in the window.** Next: Sept 15-16, Oct 27-28, Dec 8-9                                                                                                   |

Three earnings prints in 13 days covering **33.8% of the book**.

## Probabilities

Own call-journal record: **n=5, 40.0%, 95% CI [12%–77%], Brier 0.2563 vs 0.25
for a coin flip.** 50% sits inside that interval — not yet distinguishable from
chance. Weight accordingly.

System per-ticker 1d accuracy (rebuilt today from 5,552 outcomes): **BTC 66.7%,
ETH 63.3%** (n=30 each) — but consensus beats chance only at 3h/4h (53.5%); at
≥12h it is 44.8–48.0%.

| Call                                | P       | Horizon     | Basis                                                                                                                                                                                                                     |
| ----------------------------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ETH higher                          | **54%** | 7d          | Base rate after ≥+7% day with RSI>70 is **50.0% (n=42, median +0.04%)** — zero edge; 200d-reclaim bucket 66.7%; blended 58.1%; haircut for RSI 93 on 4h, failed ratio breakout, spent squeeze, negative weekly ETF flows  |
| BTC higher                          | **57%** | 7d          | Overbought-continuation bucket 77.4% (n=31) conflicts with today's day-2 follow-through bucket 43.5%; blended toward 62.6% then haircut for RSI 78, thin +3.2% cushion over the 200d, top-trader distribution 1.695→1.127 |
| ETH/BTC gives back >half its +6.63% | **65%** | 10 sessions | Ratio rejected from a 180-day high on day one; ratio RSI 81.3                                                                                                                                                             |
| SOXX higher                         | **58%** | 30d         | Dip-within-recovery bucket 59.5%, but n_eff ≈7 after de-overlapping, CI [24-95%] — nearly uninformative; both the leverage and capex-cut branches falsified                                                               |
| Neocloud underperforms core         | **65%** | 90d         | INFERENCE, no clean base rate                                                                                                                                                                                             |
| Semicap outperforms core            | **58%** | 90d         | INFERENCE                                                                                                                                                                                                                 |
| HY OAS > 3.00%                      | **30%** | 15 sessions | From 2.75% at the 15th percentile                                                                                                                                                                                         |

**Falsifiers.** `T10YIE` above ~2.45 while `DFII30` falls would resurrect the
debasement thesis — the current configuration is its exact opposite. `HY OAS`
above ~3.10% with IG under 0.90% would resurrect the balance-sheet story, and
**FRED has not published Aug 19+ credit, so that branch is genuinely open —
"could not check", not "checked and absent."** BTC holding >$71,000 for 10+
sessions with ETF creations >$400M/day would mean allocation, not a squeeze.

## Genuine bifurcation worth tracking

HY OAS sits at the **15th percentile (2.75%)** while **CCC & lower is at the
97.5th percentile (10.27%, +88bp over 63 days vs HY +1bp)**. Bottom tier near
its widest, index near its tightest — 87bp of pure quality decompression. A
slow-burn 2026 condition, not an Aug 19 catalyst, but it is exactly where the
neocloud sleeve lives.

---

## System findings

**Fixed this session:** Avanza session recovered and 507 stale critical-error
entries resolved · outcome backfill run (**1,865 → 5,552 outcomes**, newest
advanced Jul 17 → current) · `pf-outcomecheck.timer` and `pf-pickups.timer`
enabled (both were `disabled`; the latter is why ROTATE-LEAKED-CREDS never
fired) · duplicate `pf-dataloop.service` retired (it lacked
`RestartPreventExitStatus=11`, so enabling it would have restart-looped every
30s against the singleton lock).

**Outcome-blackout consequence:** signal accuracy was frozen on July data for
five weeks, so the 47% force-HOLD gate was grading on stale evidence. With 3x
the data, five _active_ signals are now under the gate: `amihud` 46.2% (CLAUDE.md
documents 68.0%), `statistical_jump` 45.4%, `econ_calendar` 44.6% (documents
57.2%), `news_event` 43.2%, `crypto_macro` 35.4% (documents 54.5%). RSI and BB
went the other way: **67.7%** and **62.4%**.

**Monte Carlo volatility is wrong by 7-12x** — see
`docs/BUG_2026-08-20-monte-carlo-vol.md`. Reports `volatility_annual = 0.05` for
BTC and ETH against realized 34% and 63%. Consequences observed live:
`drawdown_1pct_prob = 0.0`, `p_stop_hit_1d = 0.0`, metals ladder bands ~6x too
narrow. **Not fixed** — `fin_fish.py`/`grid_fisher.py` place real Avanza orders
and may be tuned around the wrong vol.

**Test suite:** 75 failures, none from this session (zero code changes). 56 are
local-LLM/GPU (paused + no GPU on this host), 11 metals-loop (not installed
here), 8 core. Triage verdict: `_compute_applicable_count` is **CORRECT**; the
tests are stale tripwires that fired in July and were never acknowledged.

**CLAUDE.md is stale** and should be corrected: live roster is **13 active**
(`SIGNAL_NAMES=90, DISABLED_SIGNALS=77`), and applicable counts are
**crypto=14, stocks=10, metals=12** — not the documented 15/12/12. Cause: two
July commits (`a953f63b` disabled ministral+qwen3; `5c654545` promoted
phi4_mini on BTC/ETH/XAU/XAG).

**One real code bug found:** `portfolio/signal_engine.py:2718`
`best_sig = max(active_in_group, key=_leader_accuracy_key)` — `max()` over a
**set**. On an exact accuracy tie the leader is picked by string-hash order,
which is randomized per process, and the loser eats a 0.3x follower penalty that
can flip consensus direction. Reproduced across `PYTHONHASHSEED` 0-5. Fix is a
deterministic tie-break: `key=lambda s: (_leader_accuracy_key(s), s)`.

**Undocumented live behaviour:** `data/local_llm.disabled` does **not** gate
`phi4_mini` — it votes via _remote_ inference on herc2. So an LLM signal is an
active voter on BTC/ETH/XAU/XAG despite the documented pause, and the crypto
applicable count silently drops 14 → 13 whenever herc2 sleeps.

**Still open, human-only:** ROTATE-LEAKED-CREDS. 11 credentials from the March
leak remain byte-identical to live config; the blob is still HTTP 200. Verify
with `.venv/bin/python scripts/check_leaked_creds_rotated.py` (exits 0 when clean).

---

## REVISION — 2026-08-20 09:25 UTC: rotate, don't reduce

A second research pass produced evidence strong enough to change the crypto
recommendation, and to withdraw one more of my claims.

### The 200-day test — the strongest single finding in the whole study

Both BTC and ETH reclaimed their 200d SMA on Aug 19. That SMA is **falling
hard** (BTC −5.39% over 30d, ETH −8.03%), which structurally makes this a
*bear-market rally* — the 2018/2022 setup the bears invoke.

The empirical record inverts that intuition (3,000 daily candles):

| Signal | BTC 7d | BTC 30d | ETH 7d | ETH 30d |
|---|---|---|---|---|
| Reclaim a **FALLING** 200d | **76.2%** n=21 | 71.4% | **73.3%** n=30 | **86.7%** |
| Reclaim a **RISING** 200d | 46.3% n=41 | 43.9% | 53.3% n=15 | 40.0% |

De-clustered at 60-day separation, the signal survives: **BTC 4/5 independent
episodes positive, ETH 7/8.** Reclaiming a *falling* 200d marks the transition
out of a downtrend, and those transitions are violent. Reclaiming a *rising* one
is a late dip-buy in an extended trend.

**Caveat that governs the sizing:** effective n is **5 and 8**, not 21 and 30.
Binomial CI on 7/8 is roughly [47%, 99%]. And **both historical failures were
exogenous shocks** — COVID (BTC 2020, −7.4%) and Luna/3AC (ETH 2022, −18.7%).
Two dated binaries sit inside the 30d horizon: **Sept 9** (buyback flow actually
starts) and **Sept 15** (Senate cloture, Clarity Act).

### Withdrawn: my "de-levered rally" mechanism

I said BTC's rally was healthy because long positioning flushed to a 14-day low.
The **notional-weighted** top-trader series says otherwise — it **ROSE**:
BTC 1.471 → 1.501, ETH 1.380 → 1.428. Many small long *accounts* closed (which
is what both my global series and the agent's top-account series measured), but
the *money* did not de-lever. Account-count ratios cannot bear the weight either
of us put on them.

The conclusion survives on better evidence: **funding pinned at the +0.0100%/8h
baseline and perp basis still NEGATIVE (BTC −0.0263%, ETH −0.0192%, mark below
index) ~19h after the move** genuinely do evidence a spot-led advance with no
long-leverage premium. Right answer, wrong instrument.

### Revised probabilities

| | BTC | ETH |
|---|---|---|
| P(higher 7d) | **60%** (was 57%) | **57%** (was 54%) |
| P(higher 30d) | **66%** (was 62%) | **65%** (was 60%) |

Both raised; ETH raised more. **They are now effectively equal — and that is
exactly what changes the trade.**

### Revised action: ROTATE ETH → BTC, keep crypto at 20.4%

Trim ETH ~20% (**≈25,000 SEK**) and buy BTC with the proceeds.

| | now | after |
|---|---|---|
| ETH | 124,348 SEK (13.87% of book) | ~99,300 SEK (~11.1%) |
| BTC | 58,507 SEK (6.53%) | ~83,500 SEK (~9.3%) |
| **Ratio** | **2.13 : 1** | **~1.19 : 1** |
| Crypto total | 20.4% | **unchanged 20.4%** |

Rationale — equal odds, unequal risk:
- **ETH ATR 2.80%/day vs BTC 1.93%** — 45% more volatility for the same odds.
- **80–87% of ETH's move is beta to BTC.** You can buy that beta in BTC at lower variance; ETH-idiosyncratic alpha is only +2.5 to +3.6pp and not significant.
- **ETH already printed a new 90-day high; BTC is still 7.1% below its** (77,322) — identifiable headroom versus none.
- **The relative leg is unwinding now:** ETH/BTC 0.032590 → 0.031890 across nine consecutive hours, after tagging 0.032500 = simultaneously the 90d AND 180d high, and rejecting. Ratio RSI 81.3.
- **ETH spot ETFs posted net weekly OUTFLOWS** ($2.26M, snapping a five-week streak, ETHA and FETH both redeeming); BTC's were positive (+$189.3M Aug 18).

**Do NOT add net new crypto.** Adding after a 3.15σ ratio day (1-in-91) into 4h
RSI 88–93 is the worst available entry.

The **rotation is far more robust than the direction call**, because it nets out
direction: whatever crypto does in aggregate, composition is the part with real
evidence behind it.

### The bull-trap case, for symmetry

1. **The catalyst is an announcement with no flow for 20 days, and it is small** — $4B/operation against a >$28T market. The 30y is at a 19-year high for structural fiscal reasons a $4B buyback does not fix. **If long-end yields resume rising the premise evaporates — watch the 30y, not the chart.**
2. **Gold has already stopped.** PAXG +0.01% over the last 6h while BTC ran +3.64%. The clean macro read is finished; what is still moving is the amplification component (crypto moved 4–5x its gold beta), and amplification mean-reverts.
3. **BTC OI was ADDED at the highs** (104,746 → 110,370 coins). Fresh longs with a high cost basis are the flush fuel if 69,035 breaks.
4. **F&G went 46 → 62 in one day** — first Greed print in 217 days. Sentiment repriced faster than fundamentals.
5. **The 200d falls 5–8%/month and price sits only +4.0% above it.** One 2-ATR down week erases the reclaim.

### Invalidation levels

- **BTC:** daily close **< 69,035** (200d, −3.9% ≈ 2.0 ATR). Earlier tell 68,902. Upside confirmation 77,322.
- **ETH:** daily close **< 2,089** (base of the squeeze leg); structural 2,004 (200d). **Relative stop: ETH/BTC < 0.0296** → complete the rotation.
- **Macro master switch: 30y yield back above 5.33%** — kills the premise regardless of price.

### Note on the logged calls

The two calls logged at 09:20 UTC (`ETH-USD TRIM p_up 0.54`, `BTC-USD HOLD
p_up 0.57`) are left untouched — `call_journal` is deliberately append-only and
has no void mechanism, by design ("all you can rewrite is a journal that will
quietly agree with whatever happened").

Be aware the ETH entry is **internally muddled**: `_BEARISH = {SELL, TRIM,
AVOID}`, so a `TRIM` is scored as a directional *down* call, while the attached
`p_up=0.54` says mildly up. If ETH rises it will score as a miss. That is a fair
penalty for conflating a composition decision with a direction call, and it
stays on the record. A relative ETH/BTC call — the thing actually believed —
could not be logged because `ETHBTC` is unpriceable through `price_source` (it
routes to Alpaca equities and 400s), and an unresolvable call would defeat the
scoring pickup's whole purpose.
