# Codex prose extract: 7-data-external

_(Codex run hit the OpenAI usage limit before emitting its final report. This is the model's narrative observations extracted from the live transcript.)_

---

Reviewing the `data-external` files in scope only. I’ll read each file fully, map the data flow and state handling, then produce a findings-first adversarial report with exact line references.
  → Read every in-scope file fully and capture line-numbered contents
  • Analyze for the requested adversarial bug classes and project-rule drift
  • Assemble findings with exact file:line citations and missing-test coverage

---

I’m enumerating the files and their sizes first so I can batch the review without missing any sections. After that I’ll read the contents in full with line numbers and start classifying risks by severity.

---

The shell wrapper is stricter than usual here, so I’m switching to smaller read-only commands and will keep the review inside those constraints. Next step is to pull the file contents directly and inspect them in batches.

---

I have the first file content flowing now. To keep the review complete and line-accurate, I’m going file-by-file rather than grep-only pattern scans, then I’ll do a second pass for cross-cutting rule drift like atomic I/O and stale-cache handling.

---

I’ve loaded the full review set into an in-memory workspace so I can cross-reference line-accurate findings without re-reading partial snippets. Next pass is focused on silent fallbacks, time math, caching, and file I/O patterns.

---

I’ve moved from ingestion wrappers into the decision-support math and calendar pieces. The issues clustering so far are around stale fallback paths, cache semantics that treat fetch failures as fresh data, and a few places where hard-coded timing/consensus rules drift from the project contract.

---

The review is into the larger sentiment and scheduler modules now. I’m chunking those in full so I can separate intentional fallbacks from the genuinely dangerous ones like stale-cache poisoning, silent report duplication, and accuracy-rule violations.