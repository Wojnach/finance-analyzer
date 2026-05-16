# /fgl: cryptotrader_lm real PEFT LoRA inference

## Goal

Flip `portfolio/signals/cryptotrader_lm.py` from scaffold to real inference.
Promote model to a measured shadow voter so the 72%-accuracy / 0.94-Sharpe paper
claim can be validated or refuted on live BTC/ETH data.

## Context

- Model: `Q:/models/cryptotrader-lm/` (PEFT LoRA adapter on Ministral-8B-Instruct + `cryptotrader-lm-lora.gguf` for llama-cpp).
- Base model: Ministral-8B-Instruct-2410 — uses Mistral `[INST]...[/INST]` instruction format.
- `portfolio/llama_server._MODEL_CONFIGS["ministral8_lora"]` already loads Ministral-8B base with `--lora cryptotrader-lm-lora.gguf` extra-arg. No new server config needed.
- Scaffold at `portfolio/signals/cryptotrader_lm.py` already has crypto-only guard (`_CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}`).
- Promotion criteria in `data/shadow_registry.json`: `min_samples=200, min_accuracy=0.60` (higher bar than 0.55 default because of paper claim).

## Approach

Same shape as `finance_llama` (commits `1ecb8d44` + `284596ae`) but:
- Reuse `ministral_trader._build_prompt(context)` because base is Mistral-instruct.
- Reuse `ministral_trader._parse_response`.
- `query_llama_server("ministral8_lora", prompt, stop=["[INST]"])`.
- Crypto-only refusal preserved at top of compute fn.
- Plex-VRAM guard via `model_load_safe()`.
- All abstention paths return canonical HOLD/conf=0 with reason string.

## Files

- `portfolio/signals/cryptotrader_lm.py` — flip `_FEATURE_AVAILABLE=True`, wire real inference body.
- `tests/test_cryptotrader_lm_inference.py` — NEW. Mock `query_llama_server`; verify:
  * Non-crypto ticker still refused (regression guard).
  * `server_unavailable` / `inference_error` / `plex_vram_tight` abstention paths.
  * Well-formed BUY/SELL/HOLD JSON parsing.
  * Confidence regex fallback.
  * Result validates under `signal_engine._validate_signal_result`.
- `tests/test_llm_scaffold_signals.py` — update `test_cryptotrader_lm_returns_abstention_on_crypto` similarly to how finance_llama was updated.
- `data/shadow_registry.json` — restore `finance_llama.cycle_modulo` 1→3 (verification window done) AND set `cryptotrader_lm.cycle_modulo` to 3 (verification phase) with notes.

## Verification

1. `pytest tests/test_cryptotrader_lm_inference.py tests/test_llm_scaffold_signals.py -v` → all green.
2. Module-level probe: real conf > 0 from running llama-server.
3. Codex / cavecrew review.
4. Full pytest -n auto.
5. Merge, push, restart loops.
6. Post-restart check: real conf rows in production log.

## Risks

| Risk | Mitigation |
|---|---|
| LoRA adapter incompatible with current llama-server build | Live probe first; swallow exceptions to abstention. |
| Cycle time blown by Ministral-8B + LoRA load | `cycle_modulo=3` budget. Throttle log-vote fix already shipped. |
| Non-crypto ticker hits inference path | Guard order: ticker check BEFORE _FEATURE_AVAILABLE branch. |

## Sequencing

Single batch (one impl file + one new test + one test update + registry tweak). No need for incremental commits.
