# Unsloth Fine-Tuning Plan for Local LLM Trading Signals

**Date:** 2026-03-18
**Status:** Draft — awaiting review before implementation
**Relation:** Extends Phase 4 of `2026-03-09-local-llm-accuracy-plan.md` and supersedes
the bitsandbytes approach in `lora-custom-training-plan.md`

---

## 1. Problem Statement

Our local LLM signals underperform:

| Model | Accuracy | Status | Issue |
|---|---|---|---|
| Custom LoRA (CryptoTrader-LM) | 20.9% | Disabled | 97% SELL bias, overfitting, data imbalance |
| Kronos | 50-56% | Disabled | Coin-flip, high failure rate |
| Chronos BTC-USD | 54% | Active | Near coin-flip for crypto |
| Ministral-3-8B | 70.2% | Active | Best performer, but generic — not trained on our data |
| Consensus (overall) | 48.1% | Active | Below 50% = worse than random |

**Root cause of LoRA failure:** The original training used (1) only 2 assets (BTC/ETH),
(2) 12h lookahead with +/-2% threshold causing severe class imbalance (~65% HOLD),
(3) templated reasoning (10 templates per class) causing overfitting to phrasing patterns,
(4) no real sentiment/F&G data (neutral placeholders), (5) bitsandbytes QLoRA with no
GGUF-native export path.

**What Unsloth changes:**
- 2x faster training, ~70% less VRAM vs standard HuggingFace + Flash Attention 2
- Native GGUF export with "Dynamic 2.0" per-layer quantization (+1% vs uniform)
- Free tier covers all our needs (single GPU, QLoRA, all models, GGUF export)
- Supports GRPO reinforcement learning (trading has verifiable rewards — price outcomes)
- 500+ model support including all our model families (Mistral, Qwen, Llama, Gemma, Phi)

---

## 2. Current Infrastructure

### Hardware
- **GPU:** RTX 3080 10GB VRAM, CUDA 13.1, driver 591.74
- **QLoRA 4-bit headroom:** Models up to ~8B params fit comfortably (need ~5-9GB)

### Models & Paths
```
Q:/models/
├── llama-cpp-bin/cuda13/llama-completion.exe   # Native llama.cpp inference binary
├── ministral-3-8b-gguf/                        # Active: Ministral-3-8B-Instruct Q5_K_M
├── ministral-8b-gguf/                          # Legacy: Ministral-8B-Instruct-2410 Q4_K_M
├── qwen3-8b-gguf/                              # New: Qwen3-8B Q4_K_M (A/B testing)
├── cryptotrader-lm/cryptotrader-lm-lora.gguf   # Disabled: 20.9% accuracy, 97% SELL bias
├── kronos/                                      # Disabled: coin-flip accuracy
├── .venv-llm/                                   # GPU inference venv (llama-cpp-python)
└── .gpu_lock                                    # File-based GPU serialization
```

### Inference Pipeline
All local LLMs run as subprocesses with JSON stdin/stdout protocol:
1. `signal_engine.py` → `ministral_signal.py` (acquires GPU gate)
2. → subprocess: `ministral_trader.py` (builds prompt, runs inference)
3. → returns `{"action": "BUY|SELL|HOLD", "reasoning": "..."}`

GGUF is the only inference format. Any training output must export to GGUF.

### Available Training Data

| Source | Entries | Fields | Quality |
|---|---|---|---|
| `data/signal_log.jsonl` + SQLite | ~1,400 snapshots | 30 signals + outcomes at 3h/1d/3d/5d/10d | Gold standard |
| `data/sentiment_ab_log.jsonl` | ~2,000 | Multi-model sentiment comparison | High |
| `data/layer2_journal.jsonl` | ~300-400 | Reasoning chains, debates, decisions | Expert quality |
| `data/forecast_predictions.jsonl` | ~500 | Chronos/Kronos predictions + outcomes | Mixed |
| `data/metals_llm_predictions.jsonl` | ~100 | Metals-specific LLM predictions | Sparse |
| Freqtrade feather files | 10K+ candles | BTC/ETH 1h OHLCV (historical) | Raw candles |

**Expanded training data (from signal_log):** ~1,400 snapshots x ~15 tickers/snapshot
x 5 horizons = up to **~100K labeled examples** (signal context → actual price outcome).

### Existing Training Pipeline (to be replaced)
- `training/lora/generate_data.py` — Feather → indicators → 12h labels → prompt pairs
- `training/lora/train_lora.py` — bitsandbytes QLoRA, rank 8, 3 epochs
- Output: HuggingFace adapter → separate `convert_lora_to_gguf.py` step
- Tests: `tests/test_lora_pipeline.py` (80+ tests)

---

## 3. What Unsloth Brings

### vs Current bitsandbytes Pipeline

| Aspect | Current (bitsandbytes) | Unsloth |
|---|---|---|
| Training speed | Baseline | **2x faster** |
| VRAM usage | ~8-9 GB (8B model) | **~5-6 GB** (70% less) |
| GGUF export | Separate tool, fragile | **Built-in, one-step** |
| Quantization quality | Uniform (Q4_K_M) | **Dynamic 2.0** (per-layer optimal) |
| RL training (GRPO) | Not available | **Available** (80% less VRAM) |
| Model support | Manual HF download | **500+ models, auto-download** |
| Iteration speed | ~45 min/run | **~20 min/run** |

### Key Capabilities for Our Use Case

1. **Fix LoRA training + export in one tool** — no more separate conversion step
2. **Try smaller models** — Qwen3-4B or Ministral-3B fit easily, faster iteration
3. **RL with verifiable rewards** — trading outcomes as reward signal (GRPO)
4. **Dynamic quantization** — better accuracy at same GGUF size

---

## 4. Model Candidates

For QLoRA 4-bit fine-tuning on RTX 3080 (10GB):

| Model | Params | Est. VRAM | Strengths | Priority |
|---|---|---|---|---|
| **Qwen3-4B** | 4B | ~4-5 GB | Strong JSON output, fast iterations | **Primary** |
| **Ministral-3B** | 3B | ~3-4 GB | Same family as active model | Secondary |
| **Llama 3.2 3B** | 3B | ~3-4 GB | Best quality/size ratio | Secondary |
| Qwen3-8B | 8B | ~8-9 GB | Best quality, tight fit | Stretch |
| Phi-4 Mini | 3.8B | ~4-5 GB | Math/reasoning focus | Optional |
| Gemma 3 4B | 4B | ~4-5 GB | Google's latest small | Optional |

**Recommendation:** Start with **Qwen3-4B** — comfortable VRAM headroom for
experimentation, strong structured output (critical for JSON responses), and fast
iteration cycles (~15-20 min/run). If Qwen3-4B proves capable, no need to go bigger.

---

## 5. Training Data Strategy

### Why the Original LoRA Failed (and how to fix it)

| Failure Mode | Original Approach | Fix |
|---|---|---|
| 97% SELL bias | Imbalanced classes (~65% HOLD) | Strict 33/33/33 balance + class weights |
| Only 2 assets | BTC + ETH only | All 20 tracked instruments |
| Synthetic reasoning | 10 templates per class | Real reasoning from Layer 2 journal |
| No real sentiment | Neutral placeholders | Actual F&G and sentiment from signal_log |
| 12h lookahead only | Single horizon | Multiple horizons (1d primary, 3d secondary) |
| 3,000 examples | Small, overfit-prone | 10K-30K balanced examples |

### Data Generation Plan

**Primary source:** `signal_log.jsonl` — each entry has 30 signal votes + actual outcomes.

```
For each (snapshot, ticker) pair where outcomes exist:
    1. Extract: price, RSI, MACD, EMA, BB, F&G, sentiment, volume, regime
    2. Extract: outcome at 1d horizon (change_pct)
    3. Label: change_pct > +1% → BUY, < -1% → SELL, else → HOLD
    4. Build prompt in Ministral format (exact _build_prompt() template)
    5. Build completion as structured JSON: {"action": "X", "reasoning": "Y"}
    6. For reasoning: use signal context to generate grounded explanations
```

**Secondary source:** `layer2_journal.jsonl` — 300+ expert reasoning chains.
Use as few-shot examples or RLHF preference data.

**Target dataset:** 10,000-30,000 examples, strictly balanced (33% each class).
Stratified by ticker and market regime to prevent overfitting to one asset or condition.

### Label Thresholds (tunable)

| Horizon | BUY threshold | SELL threshold | Rationale |
|---|---|---|---|
| 1d (primary) | > +1.0% | < -1.0% | Captures meaningful moves, above fee drag |
| 3d (secondary) | > +2.0% | < -2.0% | Longer-term trend confirmation |

HOLD zone is intentionally wider than the original +/-2% on 12h — avoids noise labeling.

---

## 6. Implementation Phases

### Phase A — Environment Setup (1-2 hours)

1. Create Unsloth training environment:
   ```bash
   conda create --name unsloth_env python=3.12 -y
   conda activate unsloth_env
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   pip install unsloth
   pip install pandas numpy pyarrow   # for data prep
   ```
   **Note:** Use conda, not pip-only. Unsloth's CUDA kernels need matched torch+CUDA.

2. Verify GPU detection:
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       "unsloth/Qwen3-4B-bnb-4bit", max_seq_length=2048, load_in_4bit=True
   )
   print("VRAM:", torch.cuda.memory_allocated() / 1e9, "GB")
   ```

3. Test GGUF export path works end-to-end with untrained model.

### Phase B — Training Data Generation (2-3 hours)

1. Write `training/unsloth/generate_training_data.py`:
   - Read `signal_log.jsonl` (or SQLite) + backfilled outcomes
   - For each (snapshot, ticker) with 1d outcome:
     - Extract all 30 signal votes + indicators + macro context
     - Label based on 1d price change (>+1% BUY, <-1% SELL, else HOLD)
     - Build prompt matching `ministral_trader.py` `_build_prompt()` format
     - Build completion as JSON `{"action": "...", "reasoning": "..."}`
   - Balance classes: equal BUY/SELL/HOLD (subsample majority classes)
   - Stratify by ticker (no single ticker > 15% of dataset)
   - 90/10 train/eval split
   - Output: `training/unsloth/data/training.jsonl`, `training/unsloth/data/eval.jsonl`

2. Validate dataset:
   - Class distribution (target: 33/33/33 +/- 2%)
   - Ticker distribution (no single ticker > 15%)
   - No data leakage (eval set from later timestamps than train set)
   - Prompt format matches production exactly

### Phase C — Fine-Tuning (2-3 hours active, ~20 min per run)

1. Write `training/unsloth/train.py`:
   ```python
   from unsloth import FastLanguageModel

   model, tokenizer = FastLanguageModel.from_pretrained(
       "unsloth/Qwen3-4B-bnb-4bit",
       max_seq_length=2048,
       load_in_4bit=True,
   )

   model = FastLanguageModel.get_peft_model(
       model,
       r=16,                  # LoRA rank (higher than original 8)
       lora_alpha=32,         # alpha/r = 2
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
       lora_dropout=0.05,
       use_gradient_checkpointing="unsloth",  # Unsloth-optimized
   )
   ```

2. Training config:
   | Parameter | Value | Notes |
   |---|---|---|
   | Base model | `unsloth/Qwen3-4B-bnb-4bit` | Pre-quantized 4-bit |
   | LoRA rank | 16 | Higher than original 8 for more capacity |
   | LoRA alpha | 32 | Standard 2x rank scaling |
   | Target modules | All attention + MLP | Broader than original q_proj/v_proj only |
   | Epochs | 3 | With early stopping on eval loss |
   | Batch size | 4 x 8 grad accum = 32 effective | |
   | Learning rate | 2e-4, cosine schedule | |
   | Warmup | 5% of steps | |
   | Max seq length | 2048 | |
   | Gradient checkpointing | Unsloth-optimized | Saves ~30% VRAM vs standard |

3. Run training, monitor eval loss. Expect ~15-20 min per run on RTX 3080.

4. **Export to GGUF immediately** (Unsloth native):
   ```python
   model.save_pretrained_gguf(
       "Q:/models/custom-qwen3-trading",
       tokenizer,
       quantization_method="q4_k_m",  # or "dynamic" for Unsloth Dynamic 2.0
   )
   ```
   Output: single GGUF file ready for llama-cpp-python.

### Phase D — Integration & A/B Testing (2-3 hours + 1 week passive)

1. Add new model to inference pipeline:
   - Copy `portfolio/ministral_trader.py` → adapt for Qwen3 prompt format
   - Or: if prompt format is compatible, just swap the GGUF path in config
   - Ensure GPU gate integration works

2. Shadow mode deployment:
   - New model runs alongside Ministral-3, predictions logged but not voted
   - Log to `data/unsloth_ab_log.jsonl`
   - Format: `{ts, ticker, ministral_action, custom_action, price, outcome_1d}`

3. A/B evaluation after 1 week:
   - Per-ticker accuracy comparison
   - Class distribution (watch for SELL bias recurrence)
   - Signal flip rate (lower = more stable)
   - Agreement/disagreement patterns

4. **Promotion criteria** (same as existing plan):
   - Accuracy >= 5% higher (absolute) than Ministral on BUY/SELL signals
   - No single-class bias (no class > 60% of votes)
   - Flip rate does not increase by > 10%
   - Works across all asset classes (not just crypto)

### Phase E — Advanced: RL with GRPO (Future, requires WSL)

**Only after Phase D shows SFT model is viable.**

1. Install WSL2 with Ubuntu (GRPO requires Linux)
2. Define reward function:
   ```python
   def trading_reward(prompt, completion, outcome):
       action = parse_action(completion)  # BUY/SELL/HOLD
       price_change = outcome["1d_change_pct"]

       if action == "BUY" and price_change > 1.0:
           return +1.0  # Correct BUY
       elif action == "SELL" and price_change < -1.0:
           return +1.0  # Correct SELL
       elif action == "HOLD" and abs(price_change) < 1.0:
           return +0.5  # Correct HOLD (lower reward, less interesting)
       else:
           return -1.0  # Wrong direction
   ```
3. GRPO training with Unsloth (80% less VRAM than standard RL)
4. Model learns optimal BUY/SELL/HOLD directly from price outcomes

---

## 7. Sentiment Model Consolidation (Separate Track)

Currently running 3+ sentiment models (CryptoBERT, TradingHero-LLM, FinGPT shadow,
FinBERT CPU). Could consolidate to one fine-tuned model:

1. Fine-tune `SmolLM2-1.7B` or `Qwen3-0.6B` on financial headlines
2. Training data: `sentiment_ab_log.jsonl` (2K+ labeled entries with multi-model consensus)
3. Label strategy: use ensemble agreement as ground truth
4. Export to GGUF (~1-2 GB), replace all 3 models
5. Benefits: less VRAM contention, faster sentiment cycle, single point of maintenance

**Priority:** Lower than trading signal fine-tuning. Only pursue after Phase D.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overfitting (repeat of LoRA failure) | Medium | High | Strict class balance, ticker stratification, eval split, early stopping |
| SELL bias recurrence | Medium | High | Monitor class distribution every epoch, abort if any class > 50% |
| VRAM OOM during training | Low | Low | Qwen3-4B at 4-bit uses ~4-5 GB; 10GB card has headroom |
| GGUF export incompatibility | Low | Medium | Test export before full training run |
| Worse than Ministral-3 baseline | Medium | Low | Shadow mode = zero risk; Ministral stays active |
| Unsloth Windows compatibility | Low | Medium | SFT works natively on Windows; GRPO needs WSL |
| Training data too small | Low | Medium | ~10K-30K examples from signal_log (not 3K like original) |
| GPU contention with live loop | Medium | Medium | Train during off-market hours or pause loop |

---

## 9. Success Criteria

| Metric | Threshold | Stretch Goal |
|---|---|---|
| 1d accuracy (all tickers) | > 55% (beat consensus 48%) | > 65% |
| 1d accuracy (XAG-USD) | > 75% (match Chronos) | > 80% |
| 1d accuracy (BTC-USD) | > 60% (beat Ministral ~54%) | > 70% |
| Class balance | No class > 50% of votes | 30-40% each |
| Signal stability | Flip rate < Ministral | — |
| Training time per run | < 30 min | < 15 min |
| GGUF file size | < 3 GB (4B model) | < 2 GB |
| Inference latency | < 10s per prediction | < 5s |

---

## 10. File Inventory (planned)

| Path | Description |
|---|---|
| `training/unsloth/generate_training_data.py` | Data generation from signal_log |
| `training/unsloth/train.py` | Unsloth QLoRA training script |
| `training/unsloth/evaluate.py` | A/B test evaluation metrics |
| `training/unsloth/data/training.jsonl` | Generated training data |
| `training/unsloth/data/eval.jsonl` | Generated eval data |
| `Q:/models/custom-qwen3-trading/` | Exported GGUF model |
| `data/unsloth_ab_log.jsonl` | A/B test predictions log |

---

## 11. Timeline

| Phase | Tasks | Duration | Dependency |
|---|---|---|---|
| **A** | Conda env + Unsloth install + verify GPU | 1-2 hours | None |
| **B** | Training data generation + validation | 2-3 hours | Phase A |
| **C** | Fine-tuning runs + GGUF export | 2-3 hours | Phase B |
| **D** | Integration + shadow deployment | 2-3 hours | Phase C |
| **D+** | A/B testing (passive) | 1 week | Phase D |
| **E** | GRPO RL (optional, WSL) | 3-5 days | Phase D success |

**Total active effort:** ~8-12 hours over 2-3 sessions.
**Total elapsed:** ~1-2 weeks (including passive A/B testing).

---

## 12. Relation to Existing Plans

- **`2026-03-09-local-llm-accuracy-plan.md`** Phase 1 (safety/measurement) and Phase 2
  (benchmarking) are prerequisites. Phase 1 is complete. Phase 2 is partially done.
  This plan implements Phase 4 (fine-tuning) using Unsloth instead of raw bitsandbytes.

- **`lora-custom-training-plan.md`** is superseded by this plan for the training approach.
  The A/B testing framework and evaluation criteria remain valid and are reused here.
  The bitsandbytes QLoRA approach is replaced by Unsloth QLoRA (faster, less VRAM,
  native GGUF export).

- **Existing `training/lora/` code** remains for reference but new work goes in
  `training/unsloth/`. The `tests/test_lora_pipeline.py` tests remain valid for
  format/prompt compatibility verification.
