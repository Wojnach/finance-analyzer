# Codex Review — 4-metals-core

## Summary

Several added modules contain functional issues: the main metals loop can be redirected into the wrong checkout on import, the live warrant refresh path fails against the normal Avanza payload shape, and some trading/analytics calculations are materially miscomputed. These are actionable correctness problems, not just style concerns.

Full review comments:

- [P1] Remove hard-coded checkout switch from metals_llm import — Q:\fa-review\data\metals_llm.py:27-28
  Importing `metals_llm` from `metals_loop.py` now unconditionally switches the whole process into `Q:/finance-analyzer` and prepends that checkout to `sys.path`. In any checkout/CI job where that path does not exist, startup crashes with `FileNotFoundError` (the caller only catches `ImportError`); if the path does exist, all later relative reads and writes go to the other repo copy instead of the one being run.

- [P1] Unwrap Avanza value objects before filtering warrant quotes — Q:\fa-review\data\metals_warrant_refresh.py:171-173
  Avanza market-guide responses expose numeric fields as value objects, which the rest of the repo already unwraps with helpers like `_v`/`_value`. Here `bid` and `ask` stay as dicts, so the normal `bid <= 0` check raises `TypeError` and aborts `refresh_warrant_catalog()`, causing `load_catalog_or_fetch()` to fall back to stale cache instead of discovering live warrants.

- [P2] Compute hold-to-close EV from the full terminal distribution — Q:\fa-review\portfolio\exit_optimizer.py:617-621
  The comment says `hold_ev` should be the mean terminal P&L, but this code only averages P&L at the 10/25/50/75/90th percentiles. That is not the expected value except by coincidence, so skewed payoff distributions (common for leveraged warrants because of the zero floor) can materially mis-rank `hold_to_close` versus the market/limit exits.

- [P2] Floor knocked-out warrant values at zero in ORB translation — Q:\fa-review\portfolio\orb_predictor.py:384-388
  For a MINI long, intrinsic value cannot go below zero after the financing level is breached, but `intrinsic_target` is allowed to go negative here. Any predicted low below the barrier therefore produces impossible outputs like losses worse than `-100%` and a negative `warrant_price_factor`, and `format_prediction()` calls this path on downside targets.

- [P2] Annualize Chronos 24h returns as drift, not volatility — Q:\fa-review\data\metals_execution_engine.py:137-141
  `chronos_24h_pct` is a 24h return forecast, but this converts it with `sqrt(252)`, which is the volatility scaling, not the return scaling. That makes confident Chronos forecasts about 16x too small in live execution recommendations (for example, `1%` becomes `0.16` instead of `2.52`), and it is inconsistent with `portfolio.fin_fish`, which annualizes the same field with `*252`.
Several added modules contain functional issues: the main metals loop can be redirected into the wrong checkout on import, the live warrant refresh path fails against the normal Avanza payload shape, and some trading/analytics calculations are materially miscomputed. These are actionable correctness problems, not just style concerns.

## Full review comments

- [P1] Remove hard-coded checkout switch from metals_llm import — Q:\fa-review\data\metals_llm.py:27-28
  Importing `metals_llm` from `metals_loop.py` now unconditionally switches the whole process into `Q:/finance-analyzer` and prepends that checkout to `sys.path`. In any checkout/CI job where that path does not exist, startup crashes with `FileNotFoundError` (the caller only catches `ImportError`); if the path does exist, all later relative reads and writes go to the other repo copy instead of the one being run.

- [P1] Unwrap Avanza value objects before filtering warrant quotes — Q:\fa-review\data\metals_warrant_refresh.py:171-173
  Avanza market-guide responses expose numeric fields as value objects, which the rest of the repo already unwraps with helpers like `_v`/`_value`. Here `bid` and `ask` stay as dicts, so the normal `bid <= 0` check raises `TypeError` and aborts `refresh_warrant_catalog()`, causing `load_catalog_or_fetch()` to fall back to stale cache instead of discovering live warrants.

- [P2] Compute hold-to-close EV from the full terminal distribution — Q:\fa-review\portfolio\exit_optimizer.py:617-621
  The comment says `hold_ev` should be the mean terminal P&L, but this code only averages P&L at the 10/25/50/75/90th percentiles. That is not the expected value except by coincidence, so skewed payoff distributions (common for leveraged warrants because of the zero floor) can materially mis-rank `hold_to_close` versus the market/limit exits.

- [P2] Floor knocked-out warrant values at zero in ORB translation — Q:\fa-review\portfolio\orb_predictor.py:384-388
  For a MINI long, intrinsic value cannot go below zero after the financing level is breached, but `intrinsic_target` is allowed to go negative here. Any predicted low below the barrier therefore produces impossible outputs like losses worse than `-100%` and a negative `warrant_price_factor`, and `format_prediction()` calls this path on downside targets.

- [P2] Annualize Chronos 24h returns as drift, not volatility — Q:\fa-review\data\metals_execution_engine.py:137-141
  `chronos_24h_pct` is a 24h return forecast, but this converts it with `sqrt(252)`, which is the volatility scaling, not the return scaling. That makes confident Chronos forecasts about 16x too small in live execution recommendations (for example, `1%` becomes `0.16` instead of `2.52`), and it is inconsistent with `portfolio.fin_fish`, which annualizes the same field with `*252`.
