# Codex prose extract: 4-metals-core-codex.md

_(Codex exec emitted a transcript instead of a final P0/P1/P2 report. This is the model's actual narrative observations, extracted by stripping tool calls and raw file dumps.)_

---

Reviewing the in-scope metals-core files only. I’ll map the recursive directories first, then inspect the code against the 10 adversarial focus areas with line-level references.

---

The shell rejected the first listing form. I’m switching to simpler read-only file enumeration and then pattern scans inside the approved paths.

---

I have the recursive file map. Next I’m doing targeted scans for hours, stop-losses, locking, session handling, and precompute lifecycle before drilling into the exact files that hit.

---

I’ve narrowed this to a small set of likely issues in the hours, EOD, stop-loss, lock, and stale-data paths. I’m reading those code regions directly now to confirm behavior and avoid false positives.

---

I’ve confirmed a few structural issues in `metals_loop` and `grid_fisher`. I’m now checking whether the actual trade sets overlap and whether re-entry on signal flips is gated by anything stronger than a timer.

---

I’m checking the stop planners now. The main question is whether any live stop placement path enforces a MINI barrier-distance guard, or whether that check only exists in ranking/pretrade helpers.

---

I’ve got the session-window mismatches in `golddigger` and `elongir`. I’m reading the session-expiry handlers next to determine whether failures are surfaced or can degrade into silent no-trade loops.

---

I have the main grid/metals findings. I’m doing one more pass on the remaining in-scope directories for anything stronger than the current issues, especially session windows, locks, and stop handling.

---

The remaining pass did not uncover a stronger ORB issue; `orb_backtest.py` is explicitly walk-forward. I’m checking one last thing now: whether the metals-related processes share or separate lock namespaces cleanly enough to prevent concurrent order activity.

---

I have enough for the review. I’m tightening the references now so the final report only includes the findings that are directly supported by the code.

---

I haven’t used the precompute files as findings yet. I’m doing a quick read there now; if they’re refreshed intraday I’ll leave staleness out rather than speculate.

---

The precompute paths are mostly self-refreshing, so I’m dropping staleness as a finding. I’m checking the `fin_fish`/`fin_snipe` timezone and rounding edge cases one last time; if they’re not solid, they stay out.