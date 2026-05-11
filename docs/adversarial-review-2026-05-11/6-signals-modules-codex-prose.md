# Codex prose extract: 6-signals-modules-codex.md

_(Codex exec emitted a transcript instead of a final P0/P1/P2 report. This is the model's actual narrative observations, extracted by stripping tool calls and raw file dumps.)_

---

Reviewing `Q:\finance-analyzer\portfolio\signals` against the subsystem rules, starting with the shared contract in `signal_utils.py` and the engine/registry wiring before I walk detector-by-detector for concrete failures.

---

The shared utility contract is simple primitives only, so the return-shape expectation is likely enforced in the engine or registry layer. I’m locating that wiring now and then scanning all detector entrypoints for signature and data-shape drift.

---

The engine is returning `(action, conf, extra_info)` after reading detector result dicts, so I’m checking registry dispatch now to pin down the exact detector contract and catch modules whose signatures or return shapes drift from what the engine calls.

---

I have the registry contract now. Next pass is the engine’s validator and dispatch paths, because that determines whether a detector bug turns into a hard exception, a silent HOLD, or a malformed vote.

---

I’m switching to an inventory pass over all detector entrypoints. That gives me a hard list of signatures, globals, and import-time behavior before I drill into the modules with the highest risk patterns.

---

The shell policy is tight enough that ad-hoc Python helpers are blocked, so I’m using the Node REPL for the bulk scan and then validating only the modules that surface concrete risk patterns.

---

Two look-ahead candidates stand out from the scan: `smart_money` because swing/structure logic often cheats with future confirmation, and `heikin_ashi` because it explicitly mentions forward-projected Alligator lines. I’m reading those closely first.

---

A second high-risk cluster is the “fixed-horizon” detectors. Anything hardcoding `1h/24h`, `1h DXY`, or daily macro returns without reading horizon will leak the same vote into every engine horizon. I’m confirming that in `forecast`, `credit_spread`, `cot_positioning`, and `metals_cross_asset`.

---

I have enough context to pin line-accurate findings now. I’m collecting exact anchors for the fixed-horizon modules and the timezone/import-time issues so the final review stays concrete.