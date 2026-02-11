# Custom LoRA Fine-Tuning Plan for Ministral-8B Trading Signals

## 1. Overview

**What:** Train a custom QLoRA adapter for Ministral-8B-Instruct-2410, specialized on BTC/ETH trading decisions using our exact prompt format and historical outcomes.

**Why:** The current CryptoTrader-LM LoRA (`Q:\models\cryptotrader-lm\cryptotrader-lm-lora.gguf`) is a generic crypto-trading adapter not trained on our specific signal format (RSI, MACD, EMA, BB, Fear & Greed, multi-timeframe) or calibrated against actual price movements. A custom LoRA should produce better signal accuracy and more consistent indicator-grounded reasoning.

**Expected outcome:** A drop-in replacement LoRA GGUF file that loads identically via `llama-cpp-python` in `ministral_trader.py`, with measurably better signal accuracy during A/B testing.

---

## 2. Training Data Generation

### Data Source

Freqtrade feather files already on disk:

- `Q:\finance-analyzer\user_data\data\binance\futures\BTC_USDT_USDT-1h-futures.feather`
- `Q:\finance-analyzer\user_data\data\binance\futures\ETH_USDT_USDT-1h-futures.feather`

### Hindsight Labeling

For each 1h candle at time `t`, compute `return_12h = (close[t+12] - close[t]) / close[t]`:

- `> +2%` -> **BUY**, `< -2%` -> **SELL**, else -> **HOLD**

### Prompt Format

Each example uses the **exact** template from `Q:\models\ministral_trader.py` ΓÇö the `[INST]...[/INST]` prompt with Asset, Price, 24h Change, RSI, MACD, EMA, BB, Fear & Greed, Multi-timeframe, Headlines fields. Indicators are computed from the 1h candle data. Sentiment fields use neutral placeholders (no historical sentiment data available).

### Completion Format

```
DECISION: BUY - RSI at 28.3 shows oversold conditions combined with bullish EMA crossover, suggesting upward momentum.
```

Completions are templated with indicator-specific reasoning matching the label, using 10+ reasoning templates per class for diversity.

### Class Balancing

Target: **3000 total** (1000 BUY, 1000 SELL, 1000 HOLD). HOLD dominates naturally (~60-70%), so subsample it. If any class has <1000, use all available and match other classes.

### Output

- Script: `Q:\finance-analyzer\scripts\generate_lora_data.py`
- Data: `Q:\finance-analyzer\training_data\lora_training.jsonl`
- Format: `{"messages": [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<completion>"}]}`

The script reads feather files, computes RSI/MACD/EMA/BB, labels with 12h lookahead, builds prompts in the exact ministral_trader.py format, balances classes, and writes JSONL.

---

## 3. Environment Setup

### New Training Venv

```powershell
cd Q:\finance-analyzer
python -m venv .venv-train
.venv-train\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes datasets accelerate
pip install pandas numpy pyarrow
```

**Alternative (recommended):** Use [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and ~40% less VRAM ΓÇö critical for 10GB cards.

### GPU: RTX 3080 (10240 MiB)

QLoRA is mandatory. 4-bit quantization reduces model footprint to ~5GB, leaving room for gradients and optimizer states.

### Base Model

Training requires HuggingFace format (not GGUF). Download `mistralai/Ministral-8B-Instruct-2410` (~16GB bf16). Tokenizer config already exists at `Q:\models\ministral-8b-config\`. May require `huggingface-cli login` if gated.

---

## 4. Training Configuration

### QLoRA Parameters

| Parameter              | Value                            | Notes                   |
| ---------------------- | -------------------------------- | ----------------------- |
| Quantization           | 4-bit NF4, double quant          | via bitsandbytes        |
| LoRA rank              | 8                                | ~0.1% params trainable  |
| LoRA alpha             | 16                               | alpha/r = 2 scaling     |
| Target modules         | `q_proj`, `v_proj`               | Attention projections   |
| Dropout                | 0.05                             |                         |
| Epochs                 | 3                                |                         |
| Batch size             | 4 (x8 grad accum = 32 effective) |                         |
| Learning rate          | 2e-4, cosine schedule            |                         |
| Warmup                 | 3% of steps                      |                         |
| Optimizer              | paged_adamw_8bit                 | Memory-efficient        |
| Gradient checkpointing | Yes                              | Trades compute for VRAM |
| Max sequence length    | 2048                             |                         |
| bf16                   | Yes                              |                         |

### Training Script

Save as `Q:\finance-analyzer\scripts\train_lora.py`. Key steps:

1. Load base model with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
2. `prepare_model_for_kbit_training()` + apply `LoraConfig`
3. Load JSONL dataset, tokenize with `[INST]{user}[/INST]{assistant}` format
4. 90/10 train/eval split
5. Train with HuggingFace `Trainer`, save to `Q:\finance-analyzer\lora_output\final`

### Estimates

- ~281 optimizer steps (3000 examples x 3 epochs / 32 effective batch)
- ~30-45 minutes on RTX 3080
- ~8-9GB peak VRAM

---

## 5. GGUF Conversion

### Option A: LoRA-Only GGUF (Preferred)

Keep LoRA separate so we can swap between CryptoTrader-LM and custom without touching the base model:

```powershell
cd Q:\models
python convert_lora_to_gguf.py Q:\finance-analyzer\lora_output\final --outfile Q:\models\custom-trading-lora.gguf
```

Produces a small (~50MB) file that loads via the existing `lora_path` parameter.

### Option B: Merged GGUF (Fallback)

If LoRA-only conversion fails, merge into base and convert the whole model:

1. `PeftModel.from_pretrained()` + `merge_and_unload()` -> save merged HF model
2. `convert_hf_to_gguf.py merged/ --outtype q4_k_m --outfile Q:\models\ministral-8b-custom-trading.gguf`

### Verification

Load with `llama-cpp-python`, run a test prompt, confirm output format matches expected `DECISION: [BUY/SELL/HOLD] - [reason]`.

---

## 6. A/B Testing Framework

### Shadow Mode

Modify `ministral_signal.py` to run both LoRAs per cycle:

- Original (CryptoTrader-LM) remains the **active** signal
- Custom LoRA runs in shadow, result logged but not acted on
- Both predictions + timestamp + ticker + price logged to `Q:\finance-analyzer\data\ab_test_log.jsonl`

Requires small change to `ministral_trader.py`: accept `_lora_override` field in input JSON to select LoRA path.

### Metrics (evaluated after 1+ week)

1. **Accuracy:** BUY correct if price rose >1% in 12h, SELL correct if fell >1%, HOLD correct if <1% move
2. **Signal flip rate:** How often consecutive signals change for same ticker (lower = more stable)
3. **Agreement rate:** How often models agree (disagreements are interesting cases)
4. **Reasoning quality:** Manual review of ~50 samples per model

### Decision Criteria

Custom LoRA replaces CryptoTrader-LM if:

- Accuracy >= 5% higher (absolute) on BUY/SELL signals
- Flip rate does not increase by >10%
- No systematic failure modes (e.g., always predicting one class)

Evaluation script: `Q:\finance-analyzer\scripts\evaluate_ab_test.py`

---

## 7. Timeline

| Phase    | Tasks                                                               | Duration |
| -------- | ------------------------------------------------------------------- | -------- |
| Day 1    | Generate training data, inspect class balance, sanity-check prompts | 2-3 hrs  |
| Day 2    | Set up `.venv-train`, download HF model, train, convert to GGUF     | 3-4 hrs  |
| Day 3    | Verify GGUF, deploy shadow mode, begin A/B logging                  | 1-2 hrs  |
| Week 1-2 | Shadow testing runs automatically with live pipeline                | Passive  |
| Decision | Run evaluation, review results, swap or iterate                     | 1 hr     |

**Total active effort:** ~8-10 hours, then passive monitoring.

### Iteration Path (if results unsatisfactory)

1. Increase rank to 16, add `k_proj`/`o_proj` to targets
2. More data: 5000+ examples, include 4h candles
3. Use Unsloth for faster iteration, try lr=1e-4 or 5e-5
4. Add real historical Fear & Greed data (alternative-me API) instead of neutral placeholders

---

## 8. File Inventory

| Path                                                    | Description                   |
| ------------------------------------------------------- | ----------------------------- |
| `Q:\finance-analyzer\scripts\generate_lora_data.py`     | Training data generation      |
| `Q:\finance-analyzer\training_data\lora_training.jsonl` | Generated training data       |
| `Q:\finance-analyzer\scripts\train_lora.py`             | QLoRA training script         |
| `Q:\finance-analyzer\lora_output\`                      | Checkpoints and final adapter |
| `Q:\models\custom-trading-lora.gguf`                    | Converted LoRA for deployment |
| `Q:\finance-analyzer\data\ab_test_log.jsonl`            | A/B test log                  |
| `Q:\finance-analyzer\scripts\evaluate_ab_test.py`       | Evaluation metrics            |
| `Q:\finance-analyzer\.venv-train\`                      | Training-only venv            |

## 9. Risks and Mitigations

| Risk                                | Mitigation                                          |
| ----------------------------------- | --------------------------------------------------- |
| 10GB VRAM OOM                       | Gradient checkpointing + Unsloth; reduce batch to 2 |
| Gated HF model                      | Pre-accept license on huggingface.co                |
| Overfitting on 3000 examples        | 90/10 split, monitor eval loss, early stopping      |
| Templated reasoning lacks diversity | 10+ templates per class, random wording variation   |
| GGUF LoRA conversion fails          | Fallback: merge into base, convert entire model     |
| Custom worse than CryptoTrader-LM   | Shadow mode = zero risk; original stays active      |
