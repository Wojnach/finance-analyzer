# PLAN — Fix fingpt parser defaulting to neutral 0.7 (task #9)

## Root cause (confirmed via live probe 2026-04-09 post-compact)

`wiroai-finance-llama-8b-q4_k_m.gguf` is a **completion/continuation model**, not a
Llama-3 chat-tuned instruct model. Fed the current `PROMPT_TEMPLATES["finance-llama-8b"]`
chat-format prompt (with `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>` tokens),
the model treats everything as plain text and continues by echoing the user question
and inventing new financial headlines. It never emits a classification word.

Parser falls through `_parse_sentiment` → `return "neutral"` (line 104), then
`_estimate_confidence` → `return 0.70` (line 116). This explains the exact signature
seen in `data/sentiment_ab_log.jsonl`:
```
{"sentiment": "neutral", "confidence": 0.7, "avg_scores": {"positive": 0.15, "negative": 0.15, "neutral": 0.7}}
```

The bug is NOT that the model thinks everything is neutral — it's that every
response is unparseable, so the fallback fires.

### Probe results (2026-04-09)

Raw response to `"Bitcoin crashes 20% in one hour"` with the current chat template:
```
'What is the sentiment of this financial news headline?'
```

Tested 5 alternative formats against the same model:

| Format | Accuracy on 4 test headlines |
|---|---|
| A. Plain `Sentiment:` suffix (plain text) | 3/4 |
| B. Instruct-style plain text | 3/4 |
| **C. Few-shot (3 examples) plain text** | **4/4** ✓ |
| D. create_chat_completion (model's embedded template) | 3/4 |
| E. One-shot minimalist | 4/4 but with continuation bleed |

**Decision**: Use Format C (few-shot plain text). Highest accuracy, clean single-word
outputs, stops cleanly on `\n\n`, compatible with llama-server's `/completion` endpoint.

The model *does* have an embedded chat template (DeepSeek-style, discovered via
`model.metadata["tokenizer.chat_template"]`) — but using it requires `create_chat_completion`
which is not how `query_llama_server_batch` talks to llama-server. The plain-text
few-shot route is architecturally cleaner.

## Scope

| File | Change |
|---|---|
| `/mnt/q/models/fingpt_infer.py` | Rewrite `PROMPT_TEMPLATES["finance-llama-8b"]` to few-shot plain-text format. Rewrite `CUMULATIVE_PROMPT` to matching multi-headline plain-text format. Tighten `_parse_sentiment` + `_estimate_confidence` to handle first-line extraction. Add a dated comment block explaining the 2026-04-09 incident + why. |
| `/mnt/q/finance-analyzer/portfolio/llm_batch.py` | Update stop tokens in `_flush_fingpt_phase`: headlines path `["\n", "<|eot_id|>", "[INST]"]` → `["\n\n"]`; cumulative path `["\n", "<|eot_id|>"]` → `["\n\n"]`. Remove the "parser bug" warning comment on `_parse_fingpt_completion` (no longer relevant after this fix). Add dated comment explaining the few-shot format requirement. |
| `tests/test_llm_batch.py` | Update tests if any assertions check the old stop tokens. Mock fingpt_infer stubs are unaffected since tests inject their own module. |
| `memory/project_fingpt_parser_defaulting_neutral.md` | Update with SHIPPED status + root cause + fix summary. |

**Not in scope**: CryptoBERT/Trading-Hero-LLM CPU → GPU migration (separate follow-up).

## Execution order

1. **Plan commit** — commit this PLAN.md so the investigation trail is preserved.
2. **Create worktree** — `git worktree add /mnt/q/finance-analyzer-fingpt-parser -b fix/fingpt-parser-prompt`.
3. **Edit `/mnt/q/models/fingpt_infer.py` in place** — this file is NOT in the git repo
   (lives in a different directory), so edits go directly on the live copy. Dated
   comment block documents the before/after for future sessions.
4. **Run the probe script again** with the new templates to confirm diverse outputs.
   Must pass on all 4 test headlines.
5. **Edit `portfolio/llm_batch.py` in the worktree** — update stop tokens, clean up comments.
6. **Update `tests/test_llm_batch.py`** — adjust stop token assertions if any.
7. **Run targeted tests**: `.venv/Scripts/python.exe -m pytest tests/test_llm_batch.py -v`
8. **Run sentiment tests**: `.venv/Scripts/python.exe -m pytest tests/test_sentiment*.py -v`
9. **Commit in worktree** — "fix(fingpt): rewrite prompts for wiroai-finance-llama-8b base model"
10. **Merge to main + push** — standard /fgl flow.
11. **Restart PF-DataLoop** — `schtasks /run /tn PF-DataLoop`.
12. **Monitor**: tail `data/sentiment_ab_log.jsonl` for a few cycles, confirm fingpt
    entries show varied sentiment + realistic confidence distribution.
13. **Update memory file** to SHIPPED status.

## Verification

- **Probe script**: 4/4 correct sentiments on the test headlines with the new template
- **Tests**: `tests/test_llm_batch.py` all pass (10 tests)
- **Live log**: within 3 post-restart cycles, `sentiment_ab_log.jsonl` shows at least
  3 distinct `fingpt:finance-llama-8b` sentiment values (not all "neutral 0.7")
- **No regression**: ministral + qwen3 phases still run, primary sentiment unchanged

## Rollback

If the new template breaks more than it fixes:
```bash
git -C /mnt/q/finance-analyzer revert <merge-sha>
cmd.exe /c "cd /d Q:\finance-analyzer && git push"
# Manually revert /mnt/q/models/fingpt_infer.py from the backup copy written
# alongside the edit (the old template text is also preserved in
# memory/project_fingpt_parser_defaulting_neutral.md)
schtasks /run /tn PF-DataLoop
```

Since fingpt is a shadow signal, "broken fingpt" still means "correct voting via
CryptoBERT + Trading-Hero-LLM" — the rollback safety net is already in place.
