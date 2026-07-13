# Local LLM Audit + Replacement Research — Progress

## Part 1 — herc2 LLM stack audit (DONE 2026-07-12 evening)

Dry run: live BTC $64,108 from Binance → production prompt builders
(`ministral_trader._build_prompt` / `qwen3_trader._build_prompt`) →
llama-server :8787 GPU inference → production parsers. Gate flag was
temporarily lifted (`ren data\local_llm.disabled ...dryrun-lifted`) and
RESTORED after. Test scripts deleted from Q:\.

| Model (slot)                                            | Result                                       | Latency |
| ------------------------------------------------------- | -------------------------------------------- | ------- |
| Ministral-3-8B Q5 (`ministral3`)                        | HOLD conf 0.55, clean parse                  | 6 s     |
| Qwen3-8B Q4 (`qwen3`)                                   | HOLD, thinking-mode, conf None on parse      | 15 s    |
| Ministral-8B + CryptoTrader LoRA v2 (`ministral8_lora`) | HOLD conf 0.6                                | 7 s     |
| Phi-4-mini-reasoning 3.8B (`phi4_mini`)                 | HOLD conf 0.65 — fastest, cleanest JSON      | 4 s     |
| finance-llama-8b (sentiment slot)                       | loads + generates (wrong-template test only) | 11 s    |

- llama-server build 8391 CUDA (Clang 19.1.5), sees full 10 GB VRAM.
- Raw prompts WITHOUT per-model chat template → empty output from
  Ministral variants. Always go through production builders.
- `portfolio/llama_server.py` already contains Linux paths
  (`/home/deck/models/...`) — Deck LLM residency pre-planned upstream.
- Model files: ministral3/qwen3 GGUFs Mar 17, LoRA v2 May 18, phi4 Jun 1,
  finance-llama-8b + LFM2-1.2B-FinGPT Feb 24. BERT sentiment models
  (CryptoBERT/FinBERT/Trading-Hero) Feb 2026 — consuming signal disabled.

## Part 2 — replacement-model deep research (PARTIAL — resumable)

Workflow run `wf_517568ad-5cc` hit the Claude session token limit mid-verify
(55/106 agents done, 9 claims confirmed 3-0, synthesis failed).

**Resume:** `Workflow({scriptPath: "<session>/workflows/scripts/deep-research-wf_517568ad-5cc.js", resumeFromRunId: "wf_517568ad-5cc", args: <same question>})`
— completed agents replay from cache. Journal:
`~/.claude/projects/-home-deck/94e546e4-.../subagents/workflows/wf_517568ad-5cc/journal.jsonl`

### Confirmed so far (3-0 adversarial votes)

- **Fin-R1 (SUFE-AIFLM-Lab)** — 7B, Qwen2.5-7B-Instruct base, SFT+GRPO on
  verifiable financial questions. FinQA 76.0 (1st), ConvFinQA 85.0 (1st),
  75.2 avg over 5 finance benchmarks — beats DeepSeek-R1-Distill-Llama-70B
  (69.2) at 10x smaller; trails full 671B R1 by 3 pts. GGUF exists
  (Mungert/Fin-R1-GGUF): Q4_K_M 4.68 GB, Q6_K 7.54 GB — fits RTX 3080.
  **Top candidate.**
- **Trading-R1 (TauricResearch)** — Qwen3-4B backbone, purpose-built
  trading reasoning, but weights NOT released (repo = "Terminal coming
  soon"). Watch, can't deploy.
- **FinLLaMA suite** — 8B LLaMA3 base, 52B-token finance corpus + 573K
  instruction tune. Candidate; GGUF availability unverified.

### Unverified leads (verify agents died at limit — rerun on resume)

- Qwen3-8B strong zero/few-shot on finance benchmarks (supports KEEP).
- DeepSeek-R1-Distill-Qwen-14B beats LLaMA-3.3-70B under RAG on financial
  reasoning (14B Q4 ≈ 8.5 GB — tight on 10 GB but possible).
- FinTradeBench: LLMs can't derive indicators from raw prices — validates
  our feed-precomputed-indicators design.
- FinBERT-class encoder claims; "models <70B can't X" claim; FinMA/Fin-o1.

### Deliverable when research completes

Top 5-10 ranked list + keep/replace verdicts on Ministral-8B, Qwen3-8B,
CryptoBERT, Trading-Hero-LLM, Chronos-2. Then: benchmark shortlist on herc2
via the same dry-run harness (production prompts, live data), shadow-slot
the winner like phi4_mini before promoting to voter.

## FINAL: Top local-LLM candidates (research 2026-07-12/13)

Verification key: [3-0] = adversarially confirmed, [unv] = claim died at
token limit, sanity-checked against training knowledge only.

### Ranked list (RTX 3080 10GB, llama.cpp, JSON-output trading duty)

1. **Fin-R1 7B (SUFE-AIFLM-Lab)** — REPLACE-candidate for the ministral slots.
   Finance-specific SFT+GRPO on Qwen2.5-7B, trained on 60K verified financial
   CoT samples [3-0]. 75.2 avg over 5 finance benchmarks — 2nd only to 671B
   DeepSeek-R1, beats the 70B R1-distill [3-0]. Apache 2.0 [3-0]. GGUF ready:
   Q4_K_M 4.68GB / Q6_K 7.54GB [3-0]. Purpose-matched: structured financial
   reasoning → JSON.
2. **Phi-4-mini-reasoning 3.8B** — ALREADY IN STACK (shadow since Jun 1).
   Our own dry run: fastest (4s), cleanest JSON of all 5. MIT. Action: pull
   its shadow accuracy stats; promote to voter if >= incumbents.
3. **Qwen3-8B** — KEEP. Best incumbent (59.7% 1d). One study claims best
   zero/few-shot financial benchmark scores in the <=8B class [unv]. Apache 2.0.
4. **DeepSeek-R1-Distill-Qwen-7B** — shadow candidate. Reasoning distill,
   MIT, ubiquitous GGUF. Related study: R1-Distill-Qwen-14B beat LLaMA-3.3-70B
   on financial reasoning under RAG [unv] — the 7B is the safe VRAM fit.
5. **FinLLaMA-Instruct 8B (Open-FinLLMs)** — watch. 52B-token finance
   pretrain + 573K instruction tune [3-0]; authors claim it beats GPT-4 on
   financial NLP/decision benchmarks [3-0 that they CLAIM it]. No official
   GGUF found — would need self-conversion. License unclear.
6. **wiroai-finance-llama-8b** — already on disk (sentiment slot). Keep as-is.
7. **Trading-R1 (Qwen3-4B)** — WATCHLIST ONLY. Ideal on paper; weights NOT
   released, repo is a placeholder [3-0]. Its headline Sharpe claims failed
   adversarial review anyway [refuted 1-2].

Rejected: FinGPT v3.x family (LoRA on 2023 Llama-2, stale, 13B doesn't fit
well), Palmyra-Fin-70B (VRAM), FinMA/FinGPT-Forecaster (2023-era).

### Time-series (Chronos-2 replacement — secondary)

Forecast signal is DISABLED since 2026-05-12; nothing actionable. If ever
revived: TimesFM 2.x / Moirai are the successors to evaluate. No change now.

### Keep/replace verdicts

| Current | Verdict |
|---|---|
| Ministral-3-8B + Ministral-8B+LoRA | KEEP for now; Fin-R1 is the direct upgrade candidate — shadow A/B first |
| Qwen3-8B | KEEP (best incumbent) |
| Phi-4-mini | PROMOTE if shadow stats confirm |
| CryptoBERT / Trading-Hero-LLM | RETIRE-in-place (signal disabled, 2023-era) — no action |
| Chronos-2 bolt | dormant, leave |

### Next actions

1. Download Fin-R1 Q5_K_M GGUF to herc2 Q:\models\fin-r1-gguf\.
2. Add "fin_r1" slot to portfolio/llama_server.py _MODEL_CONFIGS.
3. Dry-run via production harness (same BTC test).
4. Wire into llm_batch shadow rotation like phi4_mini; compare accuracy
   over 2-4 weeks; promote/drop on evidence.

Refuted-claims note: adversarial verify killed 2 overreach claims (Fin-R1
"tops ALL models incl. 671B R1 on FinQA/ConvFinQA"; Trading-R1 Sharpe-2.72
transferability). Treat vendor benchmark tables with that skepticism.
