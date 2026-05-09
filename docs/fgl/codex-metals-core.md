# Codex Adversarial Review — metals-core subsystem

Reviewer: codex / gpt-5.4 (xhigh reasoning)
Date: 2026-05-09
Branch: review/2026-05-08-metals-core (off empty-baseline)

Format: `[Pri] file.py:line — problem | FIX: repair`

---

[P0] data/metals_loop.py:2456 — Existing stops are cancelled before the “too close” guard runs, so a rejected trailing-stop update can leave the position with zero hardware protection. | FIX: Validate the replacement stop before cancelling live stops, or keep the old stops when the new stop is rejected.
[P0] data/metals_loop.py:4893 — The same-day fast path treats any non-empty `orders` list as “already placed”, so one transient stop-placement failure can leave a position unprotected for the rest of the day with no retry. | FIX: Skip only when placed stop orders cover the intended volume, and retry failed or partial placements.
[P0] data/metals_loop.py:4900 — Initial stop placement cancels existing stops before checking the 3% distance gate, so a rejected replacement can strip all live protection. | FIX: Run the distance check first and only cancel after the new stop plan is confirmed valid.
[P0] data/metals_loop.py:2499 — Rebuilt trailing-stop state writes size under `volume` instead of `units`, breaking the schema expected by the stop-fill path. | FIX: Persist the same `units` field everywhere stop-order state is written.
[P0] data/metals_loop.py:5066 — Stop-fill accounting reads `order["units"]`, so any rebuilt stop written with `volume` throws inside the broad `except` and leaves a broker-sold position marked active locally. | FIX: Normalize stop-order records before reading them or accept both `units` and `volume`.
[P1] data/metals_loop.py:5016 — Stop orders are marked `cancelled` regardless of HTTP status, so a failed DELETE can silently desync local state from broker reality and poison the next sell/replace cycle. | FIX: Only mark a stop cancelled on confirmed success and leave it retryable on non-2xx responses.
[P1] portfolio/fin_fish.py:360 — SHORT fishing feeds `1 - p_up` into `drift_from_probability`, so bullish metals conditions reduce spike probabilities and bearish conditions increase them, reversing BEAR ranking. | FIX: Use the actual upward probability for spike scenarios instead of inverting it.
[P1] portfolio/fin_fish.py:735 — Breached BEAR MINI barriers fall through with `pass`, so the planner can rank a knocked-out short instrument as tradeable if it is more than 5% past the barrier. | FIX: `continue` immediately when a SHORT product is already beyond its knock-out barrier.
[P1] portfolio/fin_snipe.py:160 — Ladder generation never passes `direction_sign=-1` for BEAR or MINI S products, so inverse instruments get long-product price translations and wrong entry/exit levels. | FIX: Derive instrument direction from market metadata or name and pass `-1` for inverse products.
[P1] portfolio/fin_snipe_manager.py:1246 — Flat-state planning carries the previous `entry_underlying` forward, so the next trade on the same orderbook reuses stale entry basis and corrupts exit P&L and targets. | FIX: Clear `entry_underlying` when the position goes flat or recompute it whenever entry price or volume changes.
[P1] portfolio/fin_snipe_manager.py:1311 — A planning exception only logs and skips that instrument, so a live held position can miss an entire cycle of sell/stop maintenance. | FIX: Fail the live cycle hard or put the instrument into an explicit emergency state instead of silently continuing.
[P2] portfolio/fin_snipe.py:49 — Stop-loss inventory failures are collapsed to `[]`, so the manager cannot distinguish “no stops exist” from “stop API unreadable” and may place duplicate protection or false naked-position actions. | FIX: Propagate the read failure or mark the snapshot invalid so the cycle aborts instead of assuming zero stops.
[P2] portfolio/fin_snipe_manager.py:1237 — `next_state` never persists `entry_ts`, so each cycle recreates the position as “just opened” and `HOLD_TIME_EXTENDED` can never accumulate. | FIX: Store `entry_ts` in saved state and clear it only when the position is truly flat.
[P2] portfolio/exit_optimizer.py:617 — Hold EV is computed from five terminal percentiles instead of the full simulated terminal distribution, so skewed path sets can mis-rank hold versus immediate exit. | FIX: Compute hold EV from the mean P&L across all terminal paths.
[P2] portfolio/metals_precompute.py:149 — `_fetch_market_data()` only fills a source when its TTL says “refresh now”, so off-cycle precompute runs overwrite deep-context files with `None` for still-valid market sections instead of last-good data. | FIX: Persist and reload last-good source payloads or skip rewriting contexts when required inputs were not refreshed.
[P2] portfolio/orb_predictor.py:32 — The session windows are hard-coded to winter UTC hours, so from late March to late October the predictor reads the wrong 09:00–11:00 Stockholm opening range by one hour. | FIX: Compute the UTC session boundaries from `Europe/Stockholm` for each trading date.
[P2] portfolio/iskbets.py:313 — The Layer-2 gate defaults to `approved=True` on timeout, missing binary, and most malformed outputs, so operational failures silently become buy approvals. | FIX: Fail closed on gate errors or make the fallback policy explicit and operator-configurable.
[P3] data/metals_loop.py:1836 — Microstructure state is only persisted every 5 cycles while cross-process readers expire it after 120 seconds, so the persisted feed self-invalidates between writes. | FIX: Persist at least once per minute or raise the freshness threshold above the write interval.
[P3] portfolio/microstructure_state.py:227 — `load_persisted_state()` drops any snapshot older than 120 seconds even though the producer only rewrites the file every 5 cycles, so readers see `None` most of the time. | FIX: Align the staleness cutoff with the actual persistence cadence.
COUNT: P0=5 P1=6 P2=6 P3=2
