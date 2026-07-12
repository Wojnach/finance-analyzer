# Deck Migration — Status + TODO

Goal: everything runs on the Steam Deck **except LLM/GPU jobs**. Loops must
work with herc2 off and LLMs unavailable (gate → abstain). Deck capacity
verified: main loop = 171 MB RAM, ~9 s CPU per 600 s cycle, network-bound.

## Done (2026-07-12)

- [x] Main loop on Deck: `pf-dataloop.service` (systemd user, lingering).
      Invoke with `python -u -m portfolio.main --loop` — `-m` required on
      Linux, direct script path breaks imports.
- [x] Dashboard on Deck: `pf-dashboard.service` (Flask :5055).
- [x] raanman.lol tunnel on Deck: `cloudflared-tunnel.service`,
      `~/.local/bin/cloudflared`, config `~/.cloudflared/config.yml`, same
      tunnel ID `91335725`. Cloudflare Access gate unchanged.
- [x] herc2 disabled: PF-DataLoop, PF-LoopResume, PF-LoopHealthWatchdog,
      cloudflared service (`sc config cloudflared start= disabled`).
- [x] No-LLM guard: `data/local_llm.disabled` armed on Deck (+
      `data/fix_agent.disabled`). Layer 2 `enabled:false` → autonomous mode.
- [x] Deck venv: requirements.txt + venv-drift extras **yfinance, filelock,
      PyJWT[crypto]** (CF Access JWT verify — 500s without it; consider
      adding these to requirements.txt).
- [x] config.json: `~/.config/finance-analyzer/config.json` (0600) +
      repo symlink, herc2-style. Never commit.
- [x] mahalanobis read-only-cov crash fixed (`f5f2d3a6`, pushed).

## Phase A — paper loops + support tasks (approved, no credentials risk)

Move each as systemd user unit on Deck, then disable herc2 twin (schtasks):

- [x] Crypto loop (`data/crypto_loop.py`, 60 s, DRY_RUN) ← PF-CryptoLoop
- [x] Oil loop (`data/oil_loop.py`, 60 s, DRY_RUN) ← PF-OilLoop
- [x] MSTR loop (`portfolio/mstr_loop/`, shadow phase) ← PF-MstrLoop
- [x] Outcome backfill (`--check-outcomes`, daily) ← PF-OutcomeCheck
      **Must move — signal data lives on Deck now.**
- [x] ML retrain (weekly, sklearn HistGradientBoosting — CPU-light,
      NOT an LLM job) ← PF-MLRetrain
- [x] Meta-learner retrain ← PF-MetaLearnerRetrain (verify not GPU first)
- [x] Shadow review ← PF-ShadowReview
- [x] Pending pickups dispatcher (`scripts/process_pending_pickups.py`,
      daily 08:00 CET) ← PF-PendingPickups
- [x] Check each entry script for Windows-only paths before enabling.

## Phase B — Avanza stack (REAL MONEY — confirm with user before executing)

Metals trades run LLM-less: LLM signals abstain via gate; metals has 10
non-LLM voters, MIN_VOTERS=2 since 2026-05-11.

- [x] Install Playwright + Chromium in Deck venv (BankID auth flow).
- [x] **MOVE (not copy)** `data/avanza_session.json` +
      `data/avanza_storage_state.json` from herc2 — two machines holding
      live sessions = conflicting order state. Sessions expire ~24 h;
      re-auth ritual (BankID + phone) moves to Deck.
- [x] Metals loop (`data/metals_loop.py` + embedded silver fast-tick +
      grid-fisher) ← PF-MetalsLoop, PF-SilverMonitor
- [x] GoldDigger ← PF-GoldDigger
- [x] iskbets / fin-snipe — trace how they're invoked before moving.
- [ ] Stop-loss rule applies: `/_api/trading/stoploss/new`, never the
      regular order API.

## Phase C — herc2 cleanup

- [ ] Disable ALL remaining PF-\* on herc2 EXCEPT LLM/GPU jobs
      (PF-LLMBackfill, PF-LocalLlmReport, PF-LoraTraining stay).
- [ ] Health-check tasks (PF-HealthCheck-\*, PF-LoopHealthDaily) — retarget
      at Deck or disable; they currently alert on the missing herc2 loop.
- [ ] PF-VerifyTunnel canary: works as-is (validates raanman.lol → Deck
      origin whenever herc2 is awake). Keep.
- [ ] herc2 end state: optional LLM box, off by default. Wake via
      `wakeherc` when GPU/LLM work needed.

## Known non-blockers

- Reddit 403 on both machines (pre-existing) — sentiment signal is in
  DISABLED_SIGNALS anyway. Only fix: Reddit OAuth app (user registers
  creds) + fetcher rewrite. Parked.
- `fingpt_infer` module absent on Deck → LLM batch skips, harmless noise.
- llm_prewarmer warns "query returned None" each cycle — gate blocks it,
  noise only. Candidate: skip prewarm when gate paused.
- herc2-sourced dashboard tiles (metals/crypto/oil/golddigger) freeze while
  herc2 is off — resolves as Phase A/B lands.

## Phase B execution notes (2026-07-12)

- Deck units: `pf-metalsloop.service` (always-on), `pf-silvermonitor.timer` +
  `pf-golddigger.timer` (daily 07:00, oneshot, golddigger keeps `--dry-run`).
- herc2 twins disabled (PF-MetalsLoop/SilverMonitor/GoldDigger), stray
  processes killed; session files RENAMED to `*.migrated-to-deck` (not
  deleted) — Deck is sole session owner.
- iskbets + fin-snipe need no separate migration: invoked from within
  main loop / metals loop.
- Fixed for Linux: `data/metals_llm.py` hardcoded `os.chdir("Q:/...")`
  (FileNotFoundError at import, not caught by ImportError fallback).
- Gotcha: tar-over-ssh pull TRUNCATED `data/metals_loop.py` (~2 KB tail) —
  restored from git. Compile-sweep all pulled .py files after any snapshot.
- **REMAINING: BankID re-auth on Deck** — `python scripts/avanza_login.py`
  opens visible Chromium on Deck screen, user authenticates with phone,
  session valid ~24 h. Current session expired (236 h) — loops run in
  monitoring mode, no order placement until re-auth.
