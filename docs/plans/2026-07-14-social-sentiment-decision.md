# Social sentiment sources — decision (2026-07-14)

Verdict from deep research (run wf_97a06673-365, 105 agents, unanimous
high-confidence): **do not chase Reddit; park social sentiment behind the
headlines A/B verdict.**

## Reddit is closed + legally fraught

- Self-serve API access ended Nov 2025; since the June 2026 Responsible
  Builder Policy, ALL Reddit Data API access requires filing a support
  ticket and manual approval. Personal scripts explicitly don't qualify.
  The `/prefs/apps` dead-end at the policy page IS the current system.
- Keyless `reddit.com/.json` trick: confirmed dead (403, even with a
  descriptive User-Agent).
- **Legal:** Reddit policy explicitly prohibits using Reddit data to
  train/feed ML/AI models without written approval; commercial use needs
  separate approval. Reddit sued Anthropic + Perplexity (2024-25) over
  exactly this. We are an Anthropic-model trading system — scraping Reddit
  (incl. via Pushshift successors PullPush / Arctic Shift) sits in the
  litigated grey zone. **Decision: do NOT scrape Reddit.**

## Clean alternatives (all either paid or already-held)

| Source                          | Covers                                                                          | Cost / status                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| LunarCrush                      | Reddit+X+YouTube+TikTok+News, 4000 crypto / 2000 stocks, Galaxy Score / AltRank | $90/mo Individual, $300/mo Builder (100 req/min). Purpose-built, legal. NOT free. |
| Alternative.me F&G              | crypto fear & greed index                                                       | free, keyless — **already used** (signal #5). No new signal.                      |
| Finnhub /stock/social-sentiment | Reddit+Twitter mention counts                                                   | Premium only — NOT on our free tier.                                              |
| StockTwits                      | stock chatter                                                                   | official API closed to new registrations; public JSON = grey.                     |

## Decision

Social sentiment is **unproven** for this system (the signal that
consumed it was disabled). No clean free path exists in 2026. Therefore:

1. **Now:** park it. Let the Finnhub **headlines A/B** (running) decide
   whether _news_ signal helps at all.
2. **If news helps** → LunarCrush ($90/mo) becomes a rational paid bet for
   social, gated through the same A/B harness before any live wiring.
3. **If news doesn't help** → social almost certainly won't either; save
   the money.

No user action needed. Reddit tab closed.
