# Session Plan — Feb 23, 2026

## Task 2: Monday Market Analysis (PRIORITY — time-sensitive)

### What
Comprehensive analysis of all 31 instruments for Monday Feb 24 open. Categorized into
long-term holds, short-term swing plays, and day-trade bets with specific price levels.

### Data Sources (already read)
- `agent_summary_compact.json` — all 25 signals, 7 timeframes, macro context (00:47 UTC snapshot)
- `portfolio_state.json` — Patient: 425K cash, MU 19.45sh @ $423.42
- `portfolio_state_bold.json` — Bold: 227.6K cash, MU 36.13sh + NVDA 56.56sh
- `layer2_context.md` + recent journal entries — theses, watchlist, regime
- Macro: DXY 97.42 (above SMA20), 10Y 4.086% (rising), FOMC 22d, F&G crypto=5/stocks=55

### Key Findings (pre-analysis)
- **TSM**: Strongest setup — BUY 100%, 3B/0S, 5/5 short TFs BUY. $370.77
- **GRRR**: 5B/0S BUY but long TFs all SELL — day-trade only
- **PLTR**: Below lower BB, BUY 100%, 3B/0S — mean-reversion bounce
- **VRT**: Thesis FLIPPED from 7/7 TFs BUY → Now SELL, 3mo+6mo SELL. Skip watchlist.
- **NVDA** (held): SELL signal on Now TF. RSI approaching overbought. Watch closely.
- **MU** (held): HOLD with 1B/0S. 4/7 mid-term TFs BUY. Stable.

### Deliverables
1. `docs/MARKET_ANALYSIS_MONDAY.md` — full analysis with price levels
2. Telegram message — condensed phone-scannable version
3. Commit

### Risk
- Weekend signal data is stale for stocks (last traded Fri close)
- Crypto signals are live but volatile (BTC below lower BB, F&G=5)
- VRT thesis flip needs Monday confirmation before acting

---

## Task 1: CUDA GPU Acceleration

### What
Research and potentially implement GPU acceleration using the RTX 3080.

### Hardware Found
- **GPU**: NVIDIA GeForce RTX 3080, 10GB VRAM, 8704 CUDA cores
- **Driver**: 591.74, CUDA 13.1
- **Toolkit**: nvcc 13.1 installed at `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/`
- **cuDNN**: Not found in lib directory

### Python Environment Status
- **PyTorch**: 2.10.0+cpu — **CPU-only build, no CUDA support**
- **CuPy**: Not installed
- **cuDF/RAPIDS**: Not installed

### Viability Assessment
The system processes ~31 tickers × 25 signals × 7 timeframes per minute.

**Bottleneck analysis:**
- **I/O (API calls)**: ~80% of cycle time. Fetching from Binance, Alpaca, news APIs.
  GPU cannot help here.
- **LLM inference (Ministral)**: ✅ CONFIRMED using RTX 3080 via llama-cpp-python CUDA.
  Runs from separate venv (`Q:/models/.venv-llm`). `n_gpu_layers=-1` = full GPU offload.
  This is the one component where GPU matters, and it's already accelerated.
- **TA calculations**: numpy-based indicator math. ~31 tickers × ~20 indicators each.
  Small dataset (~600 calculations). GPU overhead > compute savings at this scale.
- **ML classifier**: sklearn HistGradientBoosting. No CUDA backend. Would need
  cuML (RAPIDS) or XGBoost-GPU to accelerate. Training is weekly, inference is fast.

### Honest Assessment
**GPU acceleration is NOT worth the effort for the current system scale.**
- Data volume is too small (31 tickers, ~600 data points) for GPU to outperform CPU
- The real bottleneck is I/O (API latency), not compute
- Installing PyTorch-CUDA would add ~4GB to the venv for minimal benefit
- CuPy/RAPIDS ecosystem on Windows is fragile and poorly supported

### What IS Worth Doing
1. ✅ **Verified Ministral inference uses GPU** — Confirmed: llama-cpp-python CUDA loads
   Ministral-8B onto RTX 3080. `ggml_cuda_init: found 1 CUDA devices: Device 0: NVIDIA
   GeForce RTX 3080, compute capability 8.6, VMM: yes`
2. **Optional: Upgrade main venv PyTorch to CUDA** — Sentiment models (CryptoBERT,
   TradingHero, FinBERT) use `torch 2.10.0+cpu`. Upgrading would save ~1-2s per cycle
   but adds 4GB disk and compatibility risk. Low priority.
3. **Future trigger** — GPU would become more valuable at ~500+ tickers or if we add
   real-time backtesting with historical data replay

### Deliverables
1. GPU availability and viability documented in this plan
2. Check Ministral inference GPU usage
3. Update MEMORY.md with GPU findings

---

## Execution Order
1. ✅ Write this plan → commit
2. Write `docs/MARKET_ANALYSIS_MONDAY.md`
3. Send Telegram market report
4. Verify Ministral GPU usage
5. Update memory with GPU findings
6. Commit all deliverables
