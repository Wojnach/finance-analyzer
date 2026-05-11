# Codex prose extract: 7-data-external-codex.md

_(Codex exec emitted a transcript instead of a final P0/P1/P2 report. This is the model's actual narrative observations, extracted by stripping tool calls and raw file dumps.)_

---

Reviewing the in-scope data-external modules for concrete failure modes around rate limits, time handling, retries, fallback chains, and state persistence. I’ll read the target files first, map the call paths, then extract only findings that are demonstrably supported by the code.
  → Inspect in-scope files and map data-external call paths
  • Validate adversarial focus areas against concrete code and line numbers
  • Draft severity-ordered review with missing-test gaps

---

I’ve narrowed the likely risk areas to `alpha_vantage`, `http_retry`, `fx_rates`, `price_source`, `microstructure`, `onchain_data`, `session_calendar`, and the sentiment stack. Reading those in full now to confirm whether the suspicious patterns are real defects or just implementation choices.

---

The shell policy is rejecting the `Get-Content` form, so I’m switching to `rg -n "^"` for full-file reads with stable line numbers. That keeps the review grounded in exact locations without fighting the shell wrapper.

---

The first pass already surfaced one likely budget-accounting defect and one direct retry bypass in `social_sentiment`. I’m reading the smaller modules in full next, then I’ll go back for targeted context inside the larger ones.

---

I’ve reached the point where the remaining value is in call-site validation, not more blind file reads. Next I’m checking who calls the calendar, indicator, and microstructure helpers so I can separate real production paths from theoretical edge cases.