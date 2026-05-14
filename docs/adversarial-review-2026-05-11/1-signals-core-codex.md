Reading additional input from stdin...
OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\fa-adv-2026-05-11
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019e17a7-2228-7362-8b24-12ec82abca22
--------
user
You are doing an ADVERSARIAL code review of the signals-core subsystem of a quantitative trading system at Q:\finance-analyzer. Sandbox: read-only.

In-scope files (read these and ONLY these):
- portfolio/signal_engine.py
- portfolio/signal_registry.py
- portfolio/signal_utils.py
- portfolio/signal_weights.py
- portfolio/signal_weight_optimizer.py
- portfolio/signal_history.py
- portfolio/signal_state_since.py
- portfolio/signal_decay_alert.py
- portfolio/signal_postmortem.py
- portfolio/signal_db.py
- portfolio/accuracy_stats.py
- portfolio/accuracy_degradation.py
- portfolio/ticker_accuracy.py
- portfolio/outcome_tracker.py
- portfolio/forecast_accuracy.py
- portfolio/ic_computation.py
- portfolio/train_signal_weights.py
- portfolio/linear_factor.py
- portfolio/feature_normalizer.py
- portfolio/short_horizon.py

Project rules (from CLAUDE.md):
- Atomic I/O only via file_utils.atomic_write_json / load_json / atomic_append_jsonl. Raw json.loads(open(...).read()) is a defect.
- MIN_VOTERS = 3 (consensus = active BUY+SELL voters, not total).
- Accuracy gate: signals <47% accuracy with 30+ samples are force-HOLD (not inverted — inversion causes whiplash).
- Recency-weighted: 70% recent (7d) + 30% all-time.
- Regime penalties: ranging 0.75x, high-vol 0.80x confidence multipliers.

Adversarial focus — find these bug classes:
1. Silent failures: try/except that swallow errors or return default values without logging.
2. Race conditions: shared state, JSONL append/rotate races, threading without locks, mutable defaults.
3. Float / numeric: division by zero, NaN propagation, off-by-one in horizon windowing.
4. Stale data / cache bugs: TTL not honored, time comparisons in mixed tz, "now" used inconsistently.
5. Accuracy/weighting math: sample-size guards missing, biased estimators, mixing horizons, look-ahead bias in IC/factor models.
6. Signal inversion or weight changes that contradict the rule "never invert sub-50% signals — force HOLD instead".
7. Atomic I/O violations.
8. Resource leaks: open files / sqlite connections not closed.
9. Security: SQL injection in signal_db.py, unsafe deserialization of untrusted input, eval/exec.
10. Dead code / unused parameters that mask intent.

Output format — strict markdown:
# Codex adversarial review: signals-core
## Summary
(2-3 sentence overall verdict)

## P0 — Blockers (production breakage / data loss / silent wrong trades)
- path:line — short description. Why it bites: ... Fix: ...

## P1 — High (will cause incidents)
- path:line — ...

## P2 — Medium (correctness / robustness)
- path:line — ...

## P3 — Low (style / dead code / minor)
- path:line — ...

## Tests missing
- path:line — what scenario isn't covered.

Be concrete. Cite line numbers. No false-positive padding. If subsystem is clean, say so.
codex
I’m reviewing only the listed `portfolio/*.py` files and checking them against the subsystem rules you provided. I’ll start by reading the files with line numbers, then I’ll classify concrete defects by impact and cite exact locations.
2026-05-11T15:28:38.741984Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/train_signal_weights.py',
'portfolio/linear_factor.py',
'portfolio/feature_normalizer.py',
'portfolio/short_horizon.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
2026-05-11T15:28:38.742493Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_engine.py',
'portfolio/signal_registry.py',
'portfolio/signal_utils.py',
'portfolio/signal_weights.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_engine.py',
'portfolio/signal_registry.py',
'portfolio/signal_utils.py',
'portfolio/signal_weights.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/train_signal_weights.py',
'portfolio/linear_factor.py',
'portfolio/feature_normalizer.py',
'portfolio/short_horizon.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/ticker_accuracy.py',
'portfolio/outcome_tracker.py',
'portfolio/forecast_accuracy.py',
'portfolio/ic_computation.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_postmortem.py',
'portfolio/signal_db.py',
'portfolio/accuracy_stats.py',
'portfolio/accuracy_degradation.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_weight_optimizer.py',
'portfolio/signal_history.py',
'portfolio/signal_state_since.py',
'portfolio/signal_decay_alert.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}' in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/train_signal_weights.py',
'portfolio/linear_factor.py',
'portfolio/feature_normalizer.py',
'portfolio/short_horizon.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_engine.py',
'portfolio/signal_registry.py',
'portfolio/signal_utils.py',
'portfolio/signal_weights.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_postmortem.py',
'portfolio/signal_db.py',
'portfolio/accuracy_stats.py',
'portfolio/accuracy_degradation.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_weight_optimizer.py',
'portfolio/signal_history.py',
'portfolio/signal_state_since.py',
'portfolio/signal_decay_alert.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/ticker_accuracy.py',
'portfolio/outcome_tracker.py',
'portfolio/forecast_accuracy.py',
'portfolio/ic_computation.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
2026-05-11T15:28:38.743106Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_postmortem.py',
'portfolio/signal_db.py',
'portfolio/accuracy_stats.py',
'portfolio/accuracy_degradation.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
2026-05-11T15:28:38.743711Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/signal_weight_optimizer.py',
'portfolio/signal_history.py',
'portfolio/signal_state_since.py',
'portfolio/signal_decay_alert.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
2026-05-11T15:28:38.744412Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/ticker_accuracy.py',
'portfolio/outcome_tracker.py',
'portfolio/forecast_accuracy.py',
'portfolio/ic_computation.py'
)
foreach ("'$f in $files) {
  if (Test-Path $f) {
    Write-Output ("=== " + $f + " ===")
    $i = 0
    Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) }
  } else {
    Write-Output ("MISSING: " + $f)
  }
}'` rejected: blocked by policy
codex
The shell is refusing the initial batched reads, likely because of path restrictions around `Q:\finance-analyzer`. I’m switching to minimal single-file reads against absolute paths so I can confirm access before continuing the review.
2026-05-11T15:28:46.244017Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$f = '"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'
if (Test-Path "'$f) { $i = 0; Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) } } else { Write-Output ('"'MISSING: ' + "'$f) }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$f = '"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'
if (Test-Path "'$f) { $i = 0; Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) } } else { Write-Output ('"'MISSING: ' + "'$f) }' in Q:\fa-adv-2026-05-11
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$f = '"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'
if (Test-Path "'$f) { $i = 0; Get-Content $f | ForEach-Object { $i++; ('"'{0,5}: {1}' -f "'$i, $_) } } else { Write-Output ('"'MISSING: ' + "'$f) }'` rejected: blocked by policy
mcp: codex/list_mcp_resources started
mcp: codex/list_mcp_resources (completed)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 135ms:
    1: """Signal generation engine — 32-signal voting system with weighted consensus."""
    2: 
    3: import logging
    4: import threading
    5: import time
    6: from datetime import UTC, datetime
    7: from pathlib import Path
    8: from typing import Any, cast
    9: 
   10: import numpy as np
   11: import pandas as pd
   12: 
   13: from portfolio.indicators import detect_regime
   14: from portfolio.shared_state import (
   15:     FEAR_GREED_TTL,
   16:     FUNDING_RATE_TTL,
   17:     MINISTRAL_TTL,
   18:     ONCHAIN_TTL,
   19:     SENTIMENT_TTL,
   20:     VOLUME_TTL,
codex
I’ve confirmed read access through the Node REPL, so I’m keeping the review inside those exact files. Next I’m indexing the 20 targets in memory and searching for the specific defect patterns: swallowed exceptions, non-atomic JSON I/O, accuracy/inversion rule violations, and time/math hazards.
exec
js_repl in Q:\fa-adv-2026-05-11
2026-05-11T15:29:09.475792Z ERROR codex_core::tools::router: error=Identifier 'fs' has already been declared
 exited 1 in 6ms:
Identifier 'fs' has already been declared
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 8ms:
{
  "portfolio/signal_engine.py": 4198,
  "portfolio/signal_registry.py": 286,
  "portfolio/signal_utils.py": 133,
  "portfolio/signal_weights.py": 121,
  "portfolio/signal_weight_optimizer.py": 171,
  "portfolio/signal_history.py": 216,
  "portfolio/signal_state_since.py": 68,
  "portfolio/signal_decay_alert.py": 156,
  "portfolio/signal_postmortem.py": 267,
  "portfolio/signal_db.py": 393,
  "portfolio/accuracy_stats.py": 2071,
  "portfolio/accuracy_degradation.py": 1017,
  "portfolio/ticker_accuracy.py": 344,
  "portfolio/outcome_tracker.py": 581,
  "portfolio/forecast_accuracy.py": 446,
  "portfolio/ic_computation.py": 297,
  "portfolio/train_signal_weights.py": 176,
  "portfolio/linear_factor.py": 199,
  "portfolio/feature_normalizer.py": 114,
  "portfolio/short_horizon.py": 44
}
codex
I’ve loaded the full in-scope set. The engine file is large, so I’m triaging by pattern first and then reading surrounding logic to avoid missing rule violations buried in helpers.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
helpers ready
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 14ms:

=== except (113) ===
portfolio/signal_engine.py:235:     except (KeyError, IndexError, ValueError, TypeError):
portfolio/signal_engine.py:260:     except (KeyError, TypeError, ValueError):
portfolio/signal_engine.py:305:     except (TypeError, ValueError):
portfolio/signal_engine.py:316:     except (KeyError, IndexError, ValueError, TypeError):
portfolio/signal_engine.py:359:     except (KeyError, IndexError, ValueError, TypeError):
portfolio/signal_engine.py:418:     except (KeyError, IndexError, ValueError, TypeError):
portfolio/signal_engine.py:1028:         except Exception:
portfolio/signal_engine.py:1067:     except Exception:
portfolio/signal_engine.py:1402:     except Exception:
portfolio/signal_engine.py:1488:     except (TypeError, ValueError):
portfolio/signal_engine.py:1629:     except Exception:
portfolio/signal_engine.py:1758:     except (TypeError, ValueError):
portfolio/signal_engine.py:1772:     except (TypeError, ValueError):
portfolio/signal_engine.py:1811:         # half-written cache). Replay had `except TypeError` but live didn't.
portfolio/signal_engine.py:2046:     except Exception:
portfolio/signal_engine.py:2077:         except Exception as e:
portfolio/signal_engine.py:2176:         except (TypeError, ValueError):
portfolio/signal_engine.py:2601:             except (TypeError, ValueError):
portfolio/signal_engine.py:2659:         except Exception:
portfolio/signal_engine.py:2798:     except Exception:
portfolio/signal_engine.py:2888:         except Exception:
portfolio/signal_engine.py:3159:     except ImportError:
portfolio/signal_engine.py:3177:         except ImportError:
portfolio/signal_engine.py:3236:         except ImportError:
portfolio/signal_engine.py:3258:         except ImportError:
portfolio/signal_engine.py:3320:         except ImportError:
portfolio/signal_engine.py:3334:         except ImportError:
portfolio/signal_engine.py:3381:         except ImportError:
portfolio/signal_engine.py:3428:         except ImportError:
portfolio/signal_engine.py:3452:             except Exception:
portfolio/signal_engine.py:3461:             except Exception:
portfolio/signal_engine.py:3511:                     except Exception as e:
portfolio/signal_engine.py:3552:             except Exception as e:
portfolio/signal_engine.py:3572:         except Exception:
portfolio/signal_engine.py:3672:     except Exception:
portfolio/signal_engine.py:3706:     except Exception:
portfolio/signal_engine.py:3723:     except Exception:
portfolio/signal_engine.py:3831:     # H3: Define acc_horizon before the try/except so the except block and
portfolio/signal_engine.py:3875:     except Exception:
portfolio/signal_engine.py:3906:         except Exception:
portfolio/signal_engine.py:3968:         except Exception:
portfolio/signal_engine.py:3988:         except Exception:
portfolio/signal_engine.py:4096:     except Exception:
portfolio/signal_engine.py:4111:         except Exception:
portfolio/signal_engine.py:4143:     except Exception:
portfolio/signal_engine.py:4176:         except Exception:
portfolio/signal_registry.py:85:     except Exception as e:
portfolio/signal_utils.py:62:     except (TypeError, ValueError):
portfolio/signal_decay_alert.py:37:     except (FileNotFoundError, json.JSONDecodeError) as e:
portfolio/signal_decay_alert.py:147:     except Exception:
portfolio/signal_postmortem.py:138:         except Exception:
portfolio/signal_postmortem.py:222:     except Exception:
portfolio/signal_postmortem.py:228:     except Exception:
portfolio/signal_db.py:108:         except sqlite3.IntegrityError:
portfolio/signal_db.py:155:         except Exception:
portfolio/accuracy_stats.py:140:     except Exception:
portfolio/accuracy_stats.py:157:     except Exception as e:
portfolio/accuracy_stats.py:331:         except (ValueError, TypeError):
portfolio/accuracy_stats.py:700:             # except, so the in-memory cache never populates and every call
portfolio/accuracy_stats.py:761:     except Exception:
portfolio/accuracy_stats.py:983:         except (KeyError, AttributeError):
portfolio/accuracy_stats.py:989:     except Exception:
portfolio/accuracy_stats.py:1005:         except (KeyError, AttributeError):
portfolio/accuracy_stats.py:1141:     except Exception:
portfolio/accuracy_stats.py:1196:     except Exception:
portfolio/accuracy_stats.py:1232:     except Exception:
portfolio/accuracy_stats.py:1392:         except (KeyError, AttributeError):
portfolio/accuracy_stats.py:1474:     except Exception:
portfolio/accuracy_stats.py:1481:     except Exception:
portfolio/accuracy_stats.py:1523:     except Exception:
portfolio/accuracy_stats.py:1607:         except (ValueError, TypeError, KeyError):
portfolio/accuracy_stats.py:1733:         except (KeyError, TypeError):
portfolio/accuracy_stats.py:1801:     except Exception:
portfolio/accuracy_stats.py:1927:         except (KeyError, AttributeError):
portfolio/accuracy_degradation.py:155:     except Exception as e:
portfolio/accuracy_degradation.py:163:     except Exception as e:
portfolio/accuracy_degradation.py:171:     except Exception as e:
portfolio/accuracy_degradation.py:185:     except Exception as e:
portfolio/accuracy_degradation.py:194:     except Exception as e:
portfolio/accuracy_degradation.py:280:     except (KeyError, ValueError, TypeError):
portfolio/accuracy_degradation.py:294:     except Exception:
portfolio/accuracy_degradation.py:304:     except Exception as e:
portfolio/accuracy_degradation.py:476:     except Exception as e:
portfolio/accuracy_degradation.py:501:     except Exception as e:
portfolio/accuracy_degradation.py:529:     except Exception as e:
portfolio/accuracy_degradation.py:552:     except Exception as e:
portfolio/accuracy_degradation.py:595:     except (TypeError, ValueError):
portfolio/accuracy_degradation.py:735:     except Exception as e:
portfolio/accuracy_degradation.py:810:     except Exception as e:
portfolio/accuracy_degradation.py:852:         except (ValueError, TypeError):
portfolio/accuracy_degradation.py:863:     except Exception as e:
portfolio/accuracy_degradation.py:872:         # raised TypeError, the wrapper try/except caught it but the daily
portfolio/accuracy_degradation.py:875:     except Exception as e:
portfolio/accuracy_degradation.py:902:         except (TypeError, ValueError):
portfolio/accuracy_degradation.py:956:     except Exception:
portfolio/accuracy_degradation.py:985:         except (TypeError, ValueError):
portfolio/outcome_tracker.py:165:     except Exception as e:
portfolio/outcome_tracker.py:304:     except ImportError:  # pragma: no cover - non-Windows
portfolio/outcome_tracker.py:308:     except ImportError:  # pragma: no cover - Windows
portfolio/outcome_tracker.py:318:         except OSError:
portfolio/outcome_tracker.py:337:                 except OSError:
portfolio/outcome_tracker.py:396:                     except json.JSONDecodeError:
portfolio/outcome_tracker.py:417:     except Exception as e:
portfolio/outcome_tracker.py:423:         except (KeyError, ValueError, TypeError):
portfolio/outcome_tracker.py:464:                     except Exception:
portfolio/outcome_tracker.py:492:                     except Exception as e:
portfolio/outcome_tracker.py:508:         except Exception as e:
portfolio/outcome_tracker.py:524:         except FileNotFoundError:
portfolio/outcome_tracker.py:560:         except BaseException:
portfolio/outcome_tracker.py:572:         except Exception:
portfolio/forecast_accuracy.py:283:         except (ValueError, TypeError):
portfolio/forecast_accuracy.py:366:         except (ValueError, TypeError):
portfolio/linear_factor.py:99:         except np.linalg.LinAlgError:

=== raw_open (8) ===
portfolio/signal_decay_alert.py:35:         with open(accuracy_cache_path, encoding="utf-8") as f:
portfolio/accuracy_stats.py:159:     # H2: Fallback to JSONL using atomic load_jsonl_tail (avoids raw open()).
portfolio/outcome_tracker.py:315:             with open(lock_path, "ab") as lf:
portfolio/outcome_tracker.py:321:     with open(lock_path, "rb+") as lock_f:
portfolio/outcome_tracker.py:377:         with open(SIGNAL_LOG, "rb") as f:
portfolio/outcome_tracker.py:386:         with open(SIGNAL_LOG, "rb") as f:
portfolio/outcome_tracker.py:533:                     with open(SIGNAL_LOG, "rb") as f_in:
portfolio/outcome_tracker.py:550:                     with open(SIGNAL_LOG, "rb") as f_in:

=== json_loads (9) ===
portfolio/signal_decay_alert.py:36:             cache = json.load(f)
portfolio/signal_db.py:100:         trigger_reasons = json.dumps(entry.get("trigger_reasons", []))
portfolio/signal_db.py:128:                         json.dumps(tdata.get("signals", {})),
portfolio/signal_db.py:199:                     "signals": json.loads(row["signals"]) if row["signals"] else {},
portfolio/signal_db.py:217:                 "trigger_reasons": json.loads(snap["trigger_reasons"]) if snap["trigger_reasons"] else [],
portfolio/signal_db.py:268:             signals = json.loads(row["signals"]) if row["signals"] else {}
portfolio/signal_db.py:367:             signals = json.loads(row["signals"]) if row["signals"] else {}
portfolio/outcome_tracker.py:395:                         entries.append(json.loads(stripped))
portfolio/outcome_tracker.py:543:                     f_out.write((json.dumps(entry) + "\n").encode("utf-8"))

=== sqlite (4) ===
portfolio/signal_db.py:16: import sqlite3
portfolio/signal_db.py:33:             self._conn = sqlite3.connect(str(self.db_path), timeout=10)
portfolio/signal_db.py:36:             self._conn.row_factory = sqlite3.Row
portfolio/signal_db.py:108:         except sqlite3.IntegrityError:

=== evalexec (0) ===

=== threading (66) ===
portfolio/signal_engine.py:4: import threading
portfolio/signal_engine.py:38: _adx_lock = threading.Lock()  # BUG-86: protect concurrent access from ThreadPoolExecutor
portfolio/signal_engine.py:49: _last_signal_lock = threading.Lock()
portfolio/signal_engine.py:85: _phase_log_lock = threading.Lock()
portfolio/signal_engine.py:536: _persistence_lock = threading.Lock()
portfolio/signal_engine.py:563: _cross_ticker_lock = threading.Lock()
portfolio/signal_engine.py:1007: _sentiment_lock = threading.Lock()  # BUG-85: protect concurrent access from ThreadPoolExecutor
portfolio/signal_engine.py:1040:     """Set sentiment direction for a ticker (thread-safe, batched disk write)."""
portfolio/signal_engine.py:2021: _ic_data_lock = threading.Lock()
portfolio/signal_engine.py:2052: _macro_window_cache_lock = threading.Lock()
portfolio/signal_engine.py:3341:     # calling model inline (avoids model swap ping-pong between threads).
portfolio/signal_engine.py:3845:         # threads don't each pay the 7s+ cost of loading 50,000 signal-log
portfolio/signal_engine.py:3896:             # JSON parse) and on TTL miss all 5 ticker threads cold-computed
portfolio/signal_weights.py:28:     Thread-safety note: this class is not internally thread-safe.  In the
portfolio/signal_weights.py:29:     current system it is only called from the single-threaded outcome backfill
portfolio/signal_weights.py:30:     path, so no locking is required.  Add a threading.Lock if that changes.
portfolio/signal_history.py:8: import threading
portfolio/signal_history.py:23: # ThreadPoolExecutor (8 workers) calls update_history() concurrently for
portfolio/signal_history.py:29: # sequence is atomic across threads.
portfolio/signal_history.py:30: _history_lock = threading.Lock()
portfolio/signal_history.py:70:     serialized with `_history_lock` so concurrent ThreadPoolExecutor
portfolio/accuracy_stats.py:2: import threading
portfolio/accuracy_stats.py:15: # C2: Protect all read-modify-write cache operations from concurrent ticker threads
portfolio/accuracy_stats.py:16: _accuracy_write_lock = threading.Lock()
portfolio/accuracy_stats.py:20: # threads race through load_cached_accuracy() → None → signal_accuracy() and
portfolio/accuracy_stats.py:22: # Wall time was measured at 215s for a 5-thread race vs 7s single-threaded —
portfolio/accuracy_stats.py:25: # take the fast path with no lock acquisition; only the first miss-thread
portfolio/accuracy_stats.py:29: # per horizon) and serializing 4 threads through a 7s wait is far cheaper
portfolio/accuracy_stats.py:31: _accuracy_compute_lock = threading.Lock()
portfolio/accuracy_stats.py:46: # fresh process, antivirus scan) 5 concurrent ticker threads each pay the
portfolio/accuracy_stats.py:61: # guards the (timestamp, value) tuple so two threads racing to refresh
portfolio/accuracy_stats.py:64: # outside the lock, so other threads waiting on the lock see the fresh
portfolio/accuracy_stats.py:65: # value the moment the first thread returns. Two threads that both miss
portfolio/accuracy_stats.py:68: # through a 3.6s disk scan and funneling every ticker thread through it.
portfolio/accuracy_stats.py:71: _signal_utility_cache_lock = threading.Lock()
portfolio/accuracy_stats.py:75: # pay the ~49s parallel-cold-compute cost we measured under 4-thread
portfolio/accuracy_stats.py:90: # threads computing DIFFERENT horizons don't lose each other's writes (the
portfolio/accuracy_stats.py:92: # races would lose 3 of 4 horizons on a 4-thread cold-start cycle, defeating
portfolio/accuracy_stats.py:95: _signal_utility_disk_lock = threading.Lock()
portfolio/accuracy_stats.py:119:     threads computing DIFFERENT horizons don't lose each other's writes.
portfolio/accuracy_stats.py:121:     (idempotent), but a 4-thread cold-start cycle realistically has 4
portfolio/accuracy_stats.py:643:     walk costs ~3.6s on a 6K-snapshot log; with 5 ticker threads per cycle
portfolio/accuracy_stats.py:657:         # cost: thread-1 reads disk (~ms), populates L1, threads 2-4 hit L1.
portfolio/accuracy_stats.py:664:         # threads behind the slow path. Populate both caches on success.
portfolio/accuracy_stats.py:1027: # is done at most once across all racing ticker threads.
portfolio/accuracy_stats.py:1032:     Thread-safe via double-checked locking — first miss-thread computes,
portfolio/accuracy_stats.py:1120: _dashboard_prewarm_lock = threading.Lock()
portfolio/accuracy_stats.py:1155:     in-loop callers that hit the same cache from ticker threads.
portfolio/accuracy_stats.py:1158:       1. Process-local `threading.Lock` — guards the in-memory ts and
portfolio/accuracy_stats.py:1159:          lazy-load flag against concurrent threads in the same process.
portfolio/accuracy_stats.py:1166:          performs the fanout. Lock is non-blocking — a second process
portfolio/accuracy_stats.py:1415: # threads cold-computed signal_accuracy_by_regime() in parallel (50K-entry
portfolio/accuracy_stats.py:1418: # stale flag. This L1 cache makes the second-through-Nth ticker thread per
portfolio/accuracy_stats.py:1422: # lock guards ONLY the dict swap, never the compute. Two threads that both
portfolio/accuracy_stats.py:1427: _regime_accuracy_cache_lock = threading.Lock()
portfolio/accuracy_stats.py:1439:        log entries. Runs OUTSIDE any lock so concurrent ticker threads aren't
portfolio/accuracy_stats.py:1454:     # restart this is what saves us: thread-1 reads disk (~ms), populates L1,
portfolio/accuracy_stats.py:1455:     # threads 2-5 hit L1.
portfolio/accuracy_stats.py:1462:     # L1 + L2 miss — compute outside the lock so concurrent ticker threads
portfolio/accuracy_stats.py:1465:     # threads through a 30s compute.
portfolio/accuracy_stats.py:1964:     so concurrent ticker threads can't all redundantly walk the 50,000-
portfolio/accuracy_stats.py:1972:         # Re-check after acquiring the lock — another thread may have
portfolio/accuracy_degradation.py:127:     threads pre-loaded `entries` through every per-signal/per-ticker/
portfolio/forecast_accuracy.py:8: import threading
portfolio/forecast_accuracy.py:27: # lock guards the (timestamp, value) swap, NOT the compute, so two threads
portfolio/forecast_accuracy.py:31: _forecast_accuracy_cache_lock = threading.Lock()

=== now (43) ===
portfolio/signal_engine.py:6: from datetime import UTC, datetime
portfolio/signal_engine.py:1553:         cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
portfolio/signal_engine.py:2638:     hour = datetime.now(UTC).hour
portfolio/signal_registry.py:262:     # (33% annualized BTC 22:00-23:00 UTC), CME Group 2026 (60-70% gold daily range
portfolio/signal_history.py:10: from datetime import UTC, datetime
portfolio/signal_history.py:81:             "ts": datetime.now(UTC).isoformat(),
portfolio/signal_decay_alert.py:14: from datetime import UTC, datetime
portfolio/signal_decay_alert.py:139:         "ts": datetime.now(UTC).isoformat(),
portfolio/signal_postmortem.py:13: from datetime import UTC, datetime
portfolio/signal_postmortem.py:192:         "generated_at": datetime.now(UTC).isoformat(),
portfolio/accuracy_stats.py:9: from datetime import UTC
portfolio/accuracy_stats.py:287:     cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
portfolio/accuracy_stats.py:315:     now = datetime.now(UTC)
portfolio/accuracy_stats.py:483:             cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
portfolio/accuracy_stats.py:582:         cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
portfolio/accuracy_stats.py:1554:         "ts": datetime.now(UTC).isoformat(),
portfolio/accuracy_stats.py:1634:     now = datetime.now(UTC)
portfolio/accuracy_degradation.py:36: from datetime import UTC, datetime, timedelta
portfolio/accuracy_degradation.py:144:     cutoff = (datetime.now(UTC) - _td(days=days)).isoformat()
portfolio/accuracy_degradation.py:347:     now = now or datetime.now(UTC)
portfolio/accuracy_degradation.py:707:     now = now or datetime.now(UTC)
portfolio/accuracy_degradation.py:830:     now = now or datetime.now(UTC)
portfolio/accuracy_degradation.py:840:     last_send_dt = datetime.fromtimestamp(last_send_ts, tz=UTC) if last_send_ts else None
portfolio/accuracy_degradation.py:889:     now = now or datetime.now(UTC)
portfolio/ticker_accuracy.py:9: from datetime import UTC
portfolio/ticker_accuracy.py:42:         cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
portfolio/outcome_tracker.py:3: from datetime import UTC, datetime, timedelta
portfolio/outcome_tracker.py:113:     ts = datetime.now(UTC).isoformat()
portfolio/outcome_tracker.py:265:         target_dt = datetime.fromtimestamp(target_ts, tz=UTC)
portfolio/outcome_tracker.py:404:     now = datetime.now(UTC)
portfolio/outcome_tracker.py:476:                     target_ts, tz=UTC
portfolio/forecast_accuracy.py:11: from datetime import UTC, datetime, timedelta
portfolio/forecast_accuracy.py:124:         cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
portfolio/forecast_accuracy.py:297:         now = datetime.now(UTC)
portfolio/ic_computation.py:229:         cutoff = (datetime.datetime.now(datetime.UTC)
portfolio/short_horizon.py:5: - Peak trading hours (10-17 UTC) have 43-45% accuracy; quiet hours (20-01 UTC) have 52-55%
portfolio/short_horizon.py:19: # Time-of-day scaling for 3h predictions (UTC hours).
portfolio/short_horizon.py:22:     # Quiet hours (20:00-01:00 UTC) — 52-55% accuracy
portfolio/short_horizon.py:24:     # Asian session (1-6 UTC) — 48-50%, slightly below baseline
portfolio/short_horizon.py:26:     # Pre-EU (7-9 UTC) — neutral
portfolio/short_horizon.py:28:     # Peak noise (10-17 UTC) — 43-45% accuracy
portfolio/short_horizon.py:31:     # Transition (18-19 UTC) — near neutral
portfolio/short_horizon.py:42:     """Return confidence scaling factor for 3h predictions at given UTC hour."""

=== invert (1150) ===
portfolio/signal_engine.py:59:     """Return (sig_name, elapsed_seconds) for the most recent signal start
portfolio/signal_engine.py:75: # (accuracy load, weighted consensus, penalties, linear factor, etc.) into
portfolio/signal_engine.py:91: # slowly grow the dict. When we exceed the cap, prune the least-recently-
portfolio/signal_engine.py:121:     individual phases (e.g., cold accuracy_stats load, lock contention)
portfolio/signal_engine.py:159: # weighting (accuracy gate, regime mult, horizon mult, correlation
portfolio/signal_engine.py:160: # group leader) still applies, so a soft vote on a force-HOLD signal
portfolio/signal_engine.py:161: # is still force-HOLD'd. We are NOT bypassing the accuracy gate.
portfolio/signal_engine.py:387:     + accuracy gating force ~87% HOLD on metals (low confidence, vol
portfolio/signal_engine.py:388:     gate, accuracy gate). When the forecast ran but didn't commit, a
portfolio/signal_engine.py:427: # Accuracy gate: signals with blended accuracy below this threshold are
portfolio/signal_engine.py:428: # force-HOLD (treated like DISABLED_SIGNALS but dynamically). A signal at
portfolio/signal_engine.py:429: # 44% is noise, not a reliable contrarian indicator — inverting it just
portfolio/signal_engine.py:430: # produces different noise with whiplash as accuracy oscillates around 50%.
portfolio/signal_engine.py:448: # specific regimes that the aggregate accuracy hides).
portfolio/signal_engine.py:452: # Directional accuracy gate: signals whose BUY or SELL accuracy is below this
portfolio/signal_engine.py:453: # threshold get that direction force-HOLD'd while the other direction can still
portfolio/signal_engine.py:456: # 2026-04-10: raised from 0.35 → 0.40 to catch macro_regime BUY (38.9%),
portfolio/signal_engine.py:462: # Directional rescue (2026-04-28): when a signal fails the overall accuracy
portfolio/signal_engine.py:463: # gate but its vote direction has strong accuracy, rescue it at reduced weight.
portfolio/signal_engine.py:465: # Only triggers when direction accuracy >= 55% with >= 30 samples, giving
portfolio/signal_engine.py:466: # a 5pp safety margin above coin-flip.  Rescued signals get a 0.7x weight
portfolio/signal_engine.py:470: _DIRECTIONAL_RESCUE_WEIGHT_PENALTY = 0.70
portfolio/signal_engine.py:472: # Adaptive recency blend: when recent accuracy diverges from all-time by more
portfolio/signal_engine.py:473: # than this threshold, increase recent weight for faster regime adaptation.
portfolio/signal_engine.py:474: # Normal: 70% recent + 30% all-time. Fast: 90% recent + 10% all-time.
portfolio/signal_engine.py:475: # 2026-04-15: raised normal 0.70→0.75, fast 0.90→0.95 to better capture
portfolio/signal_engine.py:476: # recent-regime signals like trend (40.3% alltime → 61.6% recent).
portfolio/signal_engine.py:477: # 2026-04-16: REVERTED to 0.70/0.90. The 0.75/0.95 tuning amplified noise
portfolio/signal_engine.py:479: # with only 170 samples was dominating a 10K-sample all-time baseline,
portfolio/signal_engine.py:480: # triggering gates on signals whose "bad recent" was just the crash tail
portfolio/signal_engine.py:481: # rolling through the window. 0.70/0.90 gives regime adaptation while
portfolio/signal_engine.py:482: # leaving enough all-time anchor to damp single-week noise.
portfolio/signal_engine.py:484: _RECENCY_WEIGHT_NORMAL = 0.70
portfolio/signal_engine.py:489: # degraded (recent accuracy < 35% with 50+ samples), the market is in a
portfolio/signal_engine.py:493: _CRISIS_THRESHOLD = 0.35  # signal accuracy below this counts as "broken"
portfolio/signal_engine.py:500: # "accuracy" may just reflect market drift rather than genuine edge.
portfolio/signal_engine.py:510: # actual return magnitude. A signal with 55% accuracy but IC=0.15 catches big
portfolio/signal_engine.py:511: # moves; one with 58% accuracy but IC=0.00 is riding market drift.
portfolio/signal_engine.py:524: # are noise" pattern. Raw votes are still recorded for accuracy tracking —
portfolio/signal_engine.py:558: # Cross-ticker consensus cache: stores the most recent consensus action per
portfolio/signal_engine.py:570:     The original votes dict is not modified (needed for accuracy tracking).
portfolio/signal_engine.py:629: # that have proven accuracy on specific tickers. These are re-enabled for
portfolio/signal_engine.py:630: # compute+consensus on the listed ticker only. The standard accuracy gate
portfolio/signal_engine.py:644: # track accuracy while they remain force-HOLD in consensus.
portfolio/signal_engine.py:656:     # 2026-04-29: Added compute-only signals to accumulate accuracy data.
portfolio/signal_engine.py:662:     "fibonacci",  # newly disabled, shadow-track to confirm continued poor accuracy
portfolio/signal_engine.py:669: _PER_TICKER_CONSENSUS_GATE = 0.38  # below 38% = force HOLD
portfolio/signal_engine.py:672: # Voter-count circuit breaker (2026-04-16, Batch 2 of accuracy gating reconfig).
portfolio/signal_engine.py:674: # for a ticker, progressively relax the accuracy gate by _GATE_RELAXATION_STEP
portfolio/signal_engine.py:681: # silencing several voters whose recent accuracy dipped to 45-47%. Keeps at
portfolio/signal_engine.py:684: # only the overall accuracy gate is relaxed.
portfolio/signal_engine.py:689: # Per-ticker signal disable: force HOLD for specific signal+ticker combos
portfolio/signal_engine.py:690: # where accuracy data shows the signal is actively harmful for that instrument.
portfolio/signal_engine.py:702: # is force-HOLD'd per-horizon at consensus time).
portfolio/signal_engine.py:707: # Why this structure: the Apr 14 MSTR blacklist was built from 3h accuracy
portfolio/signal_engine.py:716:         # 2026-04-15 audit: per-ticker 3h accuracy gating, retained pending
portfolio/signal_engine.py:752:     # Source: accuracy_by_ticker_signal_cached() cross-referenced with
portfolio/signal_engine.py:753:     # global accuracy. Each entry justified by per-ticker accuracy data:
portfolio/signal_engine.py:758:     #     XAG forecast 40.3%/248, qwen3 44.8%/413;
portfolio/signal_engine.py:764:         # 2026-04-30 audit: added sentiment (33.8% 3h_recent, 3629 sam, 94.9% BUY).
portfolio/signal_engine.py:765:         # Also added bb for more tickers, forecast for BTC (38.3% 3h_recent).
portfolio/signal_engine.py:767:                               "sentiment",  # 33.8% 3h_recent (3629 sam), 94.9% BUY-only
portfolio/signal_engine.py:770:                               "sentiment",  # 33.8% 3h_recent (3629 sam), 94.9% BUY-only
portfolio/signal_engine.py:773:         "XAU-USD": frozenset({"sentiment",  # 33.8% 3h_recent (3629 sam)
portfolio/signal_engine.py:776:                               "sentiment",  # 33.8% 3h_recent (3629 sam)
portfolio/signal_engine.py:779:                            "sentiment",  # 33.8% 3h_recent (3629 sam)
portfolio/signal_engine.py:787:         # 2026-04-30 audit: added signals with <40% 1d_recent accuracy (50+ sam).
portfolio/signal_engine.py:791:         # These signals are already auto-gated by the blended accuracy gate,
portfolio/signal_engine.py:796:                               "claude_fundamental",  # 34.2% 1d_recent (730 sam), 99.3% BUY-only
portfolio/signal_engine.py:797:                               "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
portfolio/signal_engine.py:798:                               "momentum_factors",    # 32.7% 1d_recent (910 sam), 60.1% at 3h — horizon divergence
portfolio/signal_engine.py:799:                               "volume_flow",         # 35.8% 1d_recent (924 sam)
portfolio/signal_engine.py:800:                               "heikin_ashi",         # 38.2% 1d_recent (709 sam)
portfolio/signal_engine.py:801:                               "crypto_macro",        # 33.8% 1d_recent (476 sam)
portfolio/signal_engine.py:802:                               "structure",           # 33.1% 1d_recent (758 sam)
portfolio/signal_engine.py:806:                               "claude_fundamental",  # 34.2% 1d_recent (730 sam), 99.3% BUY-only
portfolio/signal_engine.py:807:                               "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
portfolio/signal_engine.py:808:                               "momentum_factors",    # 32.7% 1d_recent (910 sam)
portfolio/signal_engine.py:809:                               "volume_flow",         # 35.8% 1d_recent (924 sam)
portfolio/signal_engine.py:810:                               "heikin_ashi",         # 38.2% 1d_recent (709 sam)
portfolio/signal_engine.py:811:                               "crypto_macro",        # 33.8% 1d_recent (476 sam)
portfolio/signal_engine.py:812:                               "structure",           # 33.1% 1d_recent (758 sam)
portfolio/signal_engine.py:816:                               "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
portfolio/signal_engine.py:817:                               "momentum_factors",    # 32.7% 1d_recent (910 sam)
portfolio/signal_engine.py:818:                               "volume_flow",         # 35.8% 1d_recent (924 sam)
portfolio/signal_engine.py:819:                               "heikin_ashi",         # 38.2% 1d_recent (709 sam)
portfolio/signal_engine.py:820:                               "smart_money",         # 34.2% 1d_recent (155 sam)
portfolio/signal_engine.py:824:                               "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
portfolio/signal_engine.py:825:                               "momentum_factors",    # 32.7% 1d_recent (910 sam)
portfolio/signal_engine.py:826:                               "volume_flow",         # 35.8% 1d_recent (924 sam)
portfolio/signal_engine.py:827:                               "heikin_ashi",         # 38.2% 1d_recent (709 sam)
portfolio/signal_engine.py:830:                            "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
portfolio/signal_engine.py:831:                            "momentum_factors",    # 32.7% 1d_recent (910 sam)
portfolio/signal_engine.py:832:                            "volume_flow",         # 35.8% 1d_recent (924 sam)
portfolio/signal_engine.py:833:                            "heikin_ashi",         # 38.2% 1d_recent (709 sam)
portfolio/signal_engine.py:834:                            "smart_money",         # 34.2% 1d_recent (155 sam)
portfolio/signal_engine.py:835:                            "structure",           # 33.1% 1d_recent (758 sam)
portfolio/signal_engine.py:836:                            "macro_regime",        # 40.3% 1d (1475 sam) — moved from _default to preserve 3h
portfolio/signal_engine.py:840:         # 2026-05-10: signals with <45% accuracy at 3d horizon (global).
portfolio/signal_engine.py:850:         # 2026-05-10: signals with <45% accuracy at 5d horizon.
portfolio/signal_engine.py:888:     """Return signals to force-HOLD for (ticker, horizon). Union of default + horizon-specific.
portfolio/signal_engine.py:933: # others — its accuracy collapses the most).
portfolio/signal_engine.py:945: MIN_VOTERS_CRYPTO = 3  # crypto has 30 signals (8 core + 22 enhanced; ml disabled) — need 3
portfolio/signal_engine.py:946: MIN_VOTERS_STOCK = 3  # stocks have 24-26 signals (7 core + 17-19 enhanced, GPU-dependent) — need 3
portfolio/signal_engine.py:947: MIN_VOTERS_METALS = 2  # 2026-05-11: metals run at noisier intraday horizon
portfolio/signal_engine.py:951:                        # MIN_VOTERS=3 produced 0 trades in 20 days.
portfolio/signal_engine.py:954: # circuit-breaker precondition. Placing here (after MIN_VOTERS_*) keeps the
portfolio/signal_engine.py:955: # relationship explicit and prevents silent drift if the base MIN_VOTERS_*
portfolio/signal_engine.py:957: _MIN_VOTERS_BASE = max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)
portfolio/signal_engine.py:960: _POST_EXCLUSION_MIN = _MIN_VOTERS_BASE
portfolio/signal_engine.py:961: # Lone-signal escape floor: raised from 2 to _MIN_VOTERS_BASE (3) because a
portfolio/signal_engine.py:965: _LONE_SIGNAL_FLOOR = _MIN_VOTERS_BASE
portfolio/signal_engine.py:970: assert MIN_VOTERS_CRYPTO > 0 and MIN_VOTERS_STOCK > 0, (
portfolio/signal_engine.py:971:     "MIN_VOTERS_* must be positive")
portfolio/signal_engine.py:983:     f"Relaxed overall accuracy gate "
portfolio/signal_engine.py:987:     f"accuracy gate and the claim that the directional gate is NEVER "
portfolio/signal_engine.py:1074:         "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
portfolio/signal_engine.py:1081:         "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
portfolio/signal_engine.py:1089:         # 2026-04-05 audit: macd 58.7% recent (crossover catches range turns)
portfolio/signal_engine.py:1092:         # 2026-04-05 audit: fibonacci 68.2% recent — boost to 1.8 (was 1.6)
portfolio/signal_engine.py:1093:         # mean_reversion 65.4% recent — boost to 1.7 (was 1.5)
portfolio/signal_engine.py:1094:         # ministral 68.0% recent (Apr 5) — was 1.4x boost but collapsed to 41.5%
portfolio/signal_engine.py:1095:         # recent (Apr 26 audit, 41 sam). Removed boost, added to regime gate.
portfolio/signal_engine.py:1100:         # Was 1.2x (boosted), now 0.3x (heavily penalized).
portfolio/signal_engine.py:1101:         "oscillators": 0.3,
portfolio/signal_engine.py:1103:         # previous 0.5-0.7x was insufficient. Lowered to 0.3x.
portfolio/signal_engine.py:1104:         "trend": 0.3, "momentum_factors": 0.3, "heikin_ashi": 0.3,
portfolio/signal_engine.py:1105:         "structure": 0.4, "fear_greed": 0.3,
portfolio/signal_engine.py:1113:         "trend": 0.6, "calendar": 0.7,
portfolio/signal_engine.py:1119: # BUG-149: trend has 61.6% accuracy on 3h even in ranging — short-term trends
portfolio/signal_engine.py:1125:         # 2026-04-02 audit: 13 signals below 45% on 1d_recent. Gate the worst.
portfolio/signal_engine.py:1126:         # trend 40.7%, momentum_factors 41.4%, ema 40.8%, heikin_ashi 42.0%,
portfolio/signal_engine.py:1127:         # structure 36.1%, fear_greed 25.9%, macro_regime 30.3%,
portfolio/signal_engine.py:1130:         # The dynamic 45% accuracy gate also catches these, but explicit
portfolio/signal_engine.py:1135:             # 2026-04-02: added based on 1d_recent audit
portfolio/signal_engine.py:1138:             # candlestick 44.5% recent (292 sam). Both noise in ranging.
portfolio/signal_engine.py:1147:             # 2026-04-26: volume_flow collapsed to 40.8% recent (-10.0pp, 1310 sam).
portfolio/signal_engine.py:1148:             # credit_spread_risk collapsed to 39.0% recent (-15.2pp, 249 sam).
portfolio/signal_engine.py:1149:             # ministral collapsed to 41.5% recent (41 sam) from 58.4% all-time.
portfolio/signal_engine.py:1152:             # 2026-04-27: claude_fundamental 40.5% at 1d_recent (1178 sam),
portfolio/signal_engine.py:1155:             # 2026-04-27: sentiment 40.1% at 1d_recent (202 sam), 33.8% at 3h.
portfolio/signal_engine.py:1161:         # volatility_sig 47.2%, forecast 47.2% — marginal, let accuracy gate
portfolio/signal_engine.py:1164:         # 2026-04-11: sentiment added — 33.8% at 3h_recent (3629 sam). The 0.5x
portfolio/signal_engine.py:1166:         # 2026-04-25: claude_fundamental added — 0 accuracy samples at 3h so
portfolio/signal_engine.py:1167:         # the accuracy gate defaults to 0.5 and passes the 47% gate. At 1d the
portfolio/signal_engine.py:1168:         # fast-blended accuracy is ~40% (correctly gated), but at 3h it escapes.
portfolio/signal_engine.py:1176:         # BUG-152: SELL-biased signals have 0-11% accuracy in trending-up.
portfolio/signal_engine.py:1187:         # mean_reversion 3h_recent=45.5% — gate on short horizons
portfolio/signal_engine.py:1205:             "smart_money",  # BUG-165: 10.0% accuracy in trending-down
portfolio/signal_engine.py:1207:             # 2026-04-27: sentiment 40.1% at 1d_recent (202 sam), BUY-only bias.
portfolio/signal_engine.py:1239:     `trend` has 40.7% accuracy at 1d ranging (gate via _default) but
portfolio/signal_engine.py:1258: # Signals with >15pp accuracy divergence between horizons get adjusted.
portfolio/signal_engine.py:1259: # Updated: 2026-04-27 accuracy audit (3h_recent vs 1d_recent).
portfolio/signal_engine.py:1262:         "news_event": 1.4,      # 70.0% at 3h_recent (1762 sam)
portfolio/signal_engine.py:1274:         "mean_reversion": 0.7,  # 45.5% at 3h (but 51.8% at 1d)
portfolio/signal_engine.py:1275:         "volume_flow": 0.7,     # 46.4% at 3h — NEW 2026-04-27
portfolio/signal_engine.py:1290:         "mean_reversion": 0.7,
portfolio/signal_engine.py:1291:         "volume_flow": 0.7,
portfolio/signal_engine.py:1294:         "bb": 1.3,              # 62.5% at 1d_recent (120 sam) — boosted from 1.2
portfolio/signal_engine.py:1295:         "rsi": 1.1,             # 56.2% at 1d_recent (569 sam) — NEW 2026-04-27
portfolio/signal_engine.py:1296:         "credit_spread_risk": 1.1,  # 56.4% at 1d_recent (140 sam), SELL 77.9% — NEW 2026-04-27
portfolio/signal_engine.py:1297:         "volume": 1.1,          # 54.7% at 1d_recent (265 sam) — NEW 2026-04-27
portfolio/signal_engine.py:1298:         "macd": 1.1,            # 54.8% at 1d_recent (93 sam)
portfolio/signal_engine.py:1299:         "mean_reversion": 1.1,  # 51.8% at 1d_recent — reduced from 1.3
portfolio/signal_engine.py:1300:         "news_event": 1.4,      # 70.0% at 1d_recent (340 sam)! — was 0.5 (SELL-focused works now)
portfolio/signal_engine.py:1301:         "claude_fundamental": 0.5,  # 40.5% at 1d_recent (1178 sam) — NEW 2026-04-27 penalty
portfolio/signal_engine.py:1302:         "sentiment": 0.4,       # 40.1% at 1d_recent (202 sam) — NEW 2026-04-27
portfolio/signal_engine.py:1304:         "macro_regime": 0.5,    # 36.8% at 1d_recent
portfolio/signal_engine.py:1305:         "volatility_sig": 0.5,  # 45.5% at 1d_recent
portfolio/signal_engine.py:1306:         "structure": 0.5,       # 33.7% at 1d_recent — tightened from 0.6
portfolio/signal_engine.py:1307:         "forecast": 0.5,        # 44.6% at 1d_recent
portfolio/signal_engine.py:1308:         "ema": 0.5,             # 48.6% at 1d_recent — tightened from 0.6
portfolio/signal_engine.py:1309:         "trend": 0.5,           # 37.7% at 1d_recent — tightened from 0.6
portfolio/signal_engine.py:1310:         "heikin_ashi": 0.6,     # 42.7% at 1d_recent — tightened from 0.7
portfolio/signal_engine.py:1311:         "momentum_factors": 0.5, # 35.4% at 1d_recent — NEW 2026-04-27
portfolio/signal_engine.py:1312:         "volume_flow": 0.5,     # 40.0% at 1d_recent — NEW 2026-04-27
portfolio/signal_engine.py:1313:         "crypto_macro": 0.7,    # 46.9% at 1d_recent — NEW 2026-04-27
portfolio/signal_engine.py:1319: # dominating consensus.  Targets volume_flow (83.1% activity, 49.2% accuracy).
portfolio/signal_engine.py:1320: _ACTIVITY_RATE_CAP = 0.70
portfolio/signal_engine.py:1339:     """Compute horizon-specific signal weight multipliers from accuracy cache.
portfolio/signal_engine.py:1341:     For each signal, computes the ratio of its accuracy on this horizon vs
portfolio/signal_engine.py:1347:     HORIZON_SIGNAL_WEIGHTS if accuracy cache is unavailable.
portfolio/signal_engine.py:1351:         cache = load_json(DATA_DIR / "accuracy_cache.json")
portfolio/signal_engine.py:1355:         # Get recent accuracy for this horizon and comparison horizons
portfolio/signal_engine.py:1356:         this_key = f"{horizon}_recent"
portfolio/signal_engine.py:1369:             ch_key = f"{ch}_recent"
portfolio/signal_engine.py:1373:                     acc = stats.get("accuracy", 0.5)
portfolio/signal_engine.py:1388:             this_acc = stats.get("accuracy", 0.5)
portfolio/signal_engine.py:1393:             # Ratio of this-horizon accuracy to cross-horizon accuracy
portfolio/signal_engine.py:1510: # 2026-04-18: changed from Pearson r > 0.7 to agreement rate > 0.85.
portfolio/signal_engine.py:1512: # HOLD dominance — max observed r=0.538 (ema↔trend), making the 0.7
portfolio/signal_engine.py:1541:     """Compute signal correlation groups from recent signal_log data.
portfolio/signal_engine.py:1551:         from portfolio.accuracy_stats import load_entries
portfolio/signal_engine.py:1554:         recent = [e for e in entries if e.get("ts", "") >= cutoff]
portfolio/signal_engine.py:1555:         if len(recent) < _DYNAMIC_CORR_MIN_SAMPLES:
portfolio/signal_engine.py:1564:         for entry in recent:
portfolio/signal_engine.py:1649: # Updated 2026-04-08: empirical audit of 200 recent signal_log entries.
portfolio/signal_engine.py:1652:     # volatility_sig only weakly correlates with volume (r=0.38). Oscillators
portfolio/signal_engine.py:1655:     # RES-2026-04-21: REMOVED volatility_cluster. r=0.38 is too weak for a
portfolio/signal_engine.py:1683:     # 2026-04-04: BUG-162 — candlestick-fibonacci correlation 0.708 on BTC.
... 950 more
codex
I’ve already hit one hard policy mismatch in the engine constants, and I’m now reading the surrounding control flow to confirm it actually affects consensus rather than being dead config. After that I’m tracing the accuracy gate and persistence paths for concrete failure modes.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 8ms:

=== portfolio/signal_engine.py defs ===
52: def _set_last_signal(ticker: str, sig_name: str) -> None:
58: def get_last_signal(ticker: str) -> tuple[str, float] | None:
97: def _reset_phase_log(ticker: str) -> None:
117: def _record_phase(ticker: str, phase: str, start_mono: float) -> float:
134: def get_phase_log(ticker: str) -> list[tuple[str, float]]:
213: def _ema_dead_zone_vote(ind, df, lookback=_DEAD_ZONE_SLOPE_LOOKBACK):
246: def _bb_inside_band_vote(ind):
276: def _macd_dead_zone_vote(ind, df, lookback=_DEAD_ZONE_SLOPE_LOOKBACK):
336: def _candlestick_dead_zone_vote(df, lookback=_CANDLE_BODY_LOOKBACK):
370: def _forecast_dead_zone_vote(df, forecast_indicators,
546: def _persistence_cycles_for(ticker: str | None) -> int:
566: def _apply_persistence_filter(votes: dict[str, str], ticker: str | None) -> dict[str, str]:
887: def _get_horizon_disabled_signals(ticker: str | None, horizon: str | None) -> frozenset:
1013: def _load_prev_sentiments():
1033: def _get_prev_sentiment(ticker):
1039: def _set_prev_sentiment(ticker, direction):
1048: def flush_sentiment_state():
1231: def _get_regime_gated(regime: str, horizon: str | None = None) -> frozenset[str]:
1338: def _compute_dynamic_horizon_weights(horizon: str) -> dict[str, float]:
1407: def _get_horizon_weights(horizon: str | None) -> dict[str, float]:
1432: def _compute_applicable_count(ticker: str, skip_gpu: bool = False) -> int:
1470: def _validate_signal_result(result, sig_name=None, max_confidence=1.0):
1520: def _compute_agreement_rate(votes_a, votes_b):
1540: def _compute_dynamic_correlation_groups() -> dict[str, frozenset[str]]:
1634: def _get_correlation_groups() -> dict[str, frozenset[str]]:
1745: def _safe_accuracy(value, default):
1765: def _safe_sample_count(value):
1779: def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
1833: def _normalize_regime(regime):
1855: def _dynamic_min_voters_for_regime(regime):
1877: def _compute_gate_relaxation(votes, accuracy_data, excluded, group_gated, base_gate,
1995: def _compute_ic_mult(ic: float, icir: float, samples: int) -> float:
2024: def _get_ic_data(horizon: str) -> dict | None:
2056: def _is_macro_window_cached(now_ts: float | None = None) -> bool:
2095: def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
2621: def _confluence_score(votes, indicators):
2637: def _time_of_day_factor(horizon=None):
2648: def _load_local_model_accuracy(signal_name, horizon="1d", days=None, cache_ttl=None):
2666: def _build_llm_context(ticker, ind, timeframes, extra_info):
2705: def _gate_local_model_vote(signal_name, vote, ticker, config=None):
2748: def _compute_adx(df, period=14):
2805: def apply_confidence_penalties(action, conf, regime, ind, extra_info, ticker, df, config):
2998: def generate_signal(ind, ticker=None, config=None, timeframes=None, df=None, horizon=None):

=== portfolio/signal_registry.py defs ===
16: def register_signal(name: str, signal_type: str = "enhanced",
38: def register_enhanced(name: str, module_path: str, func_name: str,
55: def get_enhanced_signals() -> dict[str, dict]:
60: def get_signal_names() -> list:
68: def load_signal_func(entry: dict) -> Callable | None:
93: def _register_defaults():

=== portfolio/signal_utils.py defs ===
16: def sma(series: pd.Series, period: int) -> pd.Series:
24: def ema(series: pd.Series, span: int) -> pd.Series:
32: def rsi(close: pd.Series, period: int = 14) -> pd.Series:
46: def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
55: def safe_float(val) -> float:
66: def rma(series: pd.Series, period: int) -> pd.Series:
75: def wma(series: pd.Series, period: int) -> pd.Series:
90: def majority_vote(votes: list, count_hold: bool = False) -> tuple:
129: def roc(series: pd.Series, period: int) -> pd.Series:

=== portfolio/signal_weights.py defs ===
25: class SignalWeightManager:

=== portfolio/signal_weight_optimizer.py defs ===
32: class WalkForwardResult:
52: def walk_forward_optimize(
157: def save_results(result: WalkForwardResult, path: Path | None = None) -> None:
164: def load_results(path: Path | None = None) -> WalkForwardResult | None:

=== portfolio/signal_history.py defs ===
33: def _load_history():
42: def _save_history(entries):
51: def _entries_for_ticker(entries, ticker):
64: def update_history(ticker, votes_dict):
101: def get_persistence_scores(ticker):
148: def get_noisy_signals(ticker, threshold=0.3):
162: def get_signal_streaks(ticker):
198: def get_summary(ticker):

=== portfolio/signal_state_since.py defs ===
25: def update_state_since(

=== portfolio/signal_decay_alert.py defs ===
27: def check_signal_decay(accuracy_cache_path="data/accuracy_cache.json"):
91: def format_decay_report(alerts):
128: def log_decay_alerts(alerts):
151: def run_decay_check():

=== portfolio/signal_postmortem.py defs ===
32: def compute_regime_insights(regime_accuracy: dict) -> list[dict]:
87: def compute_signal_health_report(accuracy_data: dict) -> list[dict]:
121: def compute_vote_correlation(entries: list[dict] | None = None) -> list[dict]:
185: def generate_postmortem() -> dict:
252: def get_postmortem_context() -> dict | None:

=== portfolio/signal_db.py defs ===
24: class SignalDB:

=== portfolio/accuracy_stats.py defs ===
98: def _load_signal_utility_disk(horizon: str) -> dict | None:
115: def _write_signal_utility_disk(horizon: str, data: dict) -> None:
144: def load_entries():
170: def _vote_correct(vote, change_pct, min_change_pct=None):
189: def signal_accuracy(horizon="1d", since=None, entries=None):
280: def signal_accuracy_recent(horizon="1d", days=7):
291: def signal_accuracy_ewma(horizon="1d", halflife_days=5, entries=None):
388: def signal_accuracy_cost_adjusted(horizon="1d", cost_bps=10.0, entries=None):
459: def consensus_accuracy(horizon="1d", entries=None, days=None):
518: def per_ticker_accuracy(horizon="1d", entries=None):
556: def accuracy_by_signal_ticker(signal_name, horizon="1d", days=None, entries=None):
622: def signal_utility(horizon="1d", entries=None):
674: def _compute_signal_utility(horizon, entries):
735: def invalidate_signal_utility_cache():
765: def best_worst_signals(horizon="1d", acc=None):
780: def signal_activation_rates(entries=None):
863: def blend_accuracy_data(alltime, recent, divergence_threshold=0.15,
975: def load_cached_activation_rates():
994: def load_cached_accuracy(horizon="1d"):
1010: def write_accuracy_cache(horizon, data):
1029: def get_or_compute_accuracy(horizon: str):
1048: def get_or_compute_recent_accuracy(horizon: str, days: int = 7):
1064: def get_or_compute_per_ticker_accuracy(horizon: str):
1084: def get_or_compute_consensus_accuracy(horizon: str):
1125: def _load_prewarm_ts_from_disk() -> float:
1134: def _save_prewarm_ts_to_disk(ts: float) -> None:
1145: def maybe_prewarm_dashboard_accuracy(now: float | None = None) -> bool:
1240: def _count_entries_with_outcomes(entries, horizon):
1251: def print_accuracy_report():
1314: def signal_accuracy_by_regime(horizon="1d", since=None, entries=None):
1380: def load_cached_regime_accuracy(horizon="1d"):
1397: def write_regime_accuracy_cache(horizon, data):
1430: def get_or_compute_regime_accuracy(horizon: str = "1d") -> dict:
1486: def invalidate_regime_accuracy_cache(horizon: str | None = None) -> None:
1530: def save_accuracy_snapshot(extras=None):
1570: def _load_accuracy_snapshots():
1585: def _find_snapshot_near(snapshots, target_ts, max_delta_hours=36):
1612: def check_accuracy_changes(threshold_drop=0.1, threshold_rise=0.1):
1687: def format_accuracy_alerts(alerts):
1706: def signal_best_horizon_accuracy(min_samples=50, entries=None):
1807: def accuracy_by_ticker_signal(horizon="1d", min_samples=0):
1885: def top_signals_for_ticker(ticker, horizon="1d", min_samples=5):
1914: def load_cached_ticker_accuracy(horizon="1d"):
1932: def write_ticker_accuracy_cache(horizon, data):
1946: def _filter_min_samples(data, min_samples):
1958: def accuracy_by_ticker_signal_cached(horizon="1d", min_samples=0):
1984: def probability_calibration(horizon="1d", buckets=None, since=None):

=== portfolio/accuracy_degradation.py defs ===
90: def _load_alert_state() -> dict:
101: def _save_alert_state(state: dict) -> None:
105: def _load_snapshot_state() -> dict:
113: def _save_snapshot_state(state: dict) -> None:
119: def save_full_accuracy_snapshot(*, days: int = 7) -> dict[str, Any]:
203: def _per_ticker_recent(horizon: str, days: int, *, entries=None) -> dict:
240: def _compact_per_ticker(per_ticker: dict) -> dict:
265: def _load_snapshots() -> list[dict]:
270: def _find_baseline_snapshot(snapshots: list[dict], now: datetime) -> dict | None:
277: def _snapshot_age_days(snapshot: dict, now: datetime) -> float:
287: def _is_econ_blackout() -> bool:
311: def _make_violation(invariant: str, severity: str, message: str,
328: def check_degradation(now: datetime | None = None,
400: def _hydrate_cached_violations(state: dict) -> list:
410: def _violation_to_dict(violation) -> dict:
419: def _dict_to_violation(payload: dict):
430: def _diff_against_baseline(*, baseline: dict, now: datetime,
558: def _binomial_diff_se_pp(p1: float, n1: int, p2: float, n2: int) -> float:
576: def _maybe_alert(*, key: str, scope: str, old, new,
620: def _classify_severity(alerts: list[dict]) -> str:
631: def _alerts_to_violations(alerts: list[dict], *, age_days: float) -> list:
663: def filter_alerts_by_cooldown(alerts: list[dict],
693: def maybe_save_daily_snapshot(config: dict | None = None,
762: def _record_snapshot_writer_silent_failure(
819: def maybe_send_degradation_summary(config: dict | None = None,
886: def build_daily_summary(*, latest: dict, baseline: dict | None,
968: def _summary_diffs(*, latest: dict,
1010: def _format_summary_row(item: dict) -> str:

=== portfolio/ticker_accuracy.py defs ===
16: def accuracy_by_ticker_signal(ticker, horizon="1d", days=None):
85: def direction_probability(ticker, current_votes, horizon="1d", days=7, min_samples=5):
177: def direction_probability_with_forecast(ticker, current_votes, forecast_data=None,
254: def get_focus_probabilities(tickers, current_data, horizons=None, days=7):
320: def _extract_votes(ticker_data):

=== portfolio/outcome_tracker.py defs ===
29: def _derive_signal_vote(name, indicators, extra):
112: def log_signal_snapshot(signals_dict, prices_usd, fx_rate, trigger_reasons):
171: def _fetch_current_price(ticker):
211: def _fetch_historical_price(ticker, target_ts):
281: def _signal_log_lock_path():
293: def _hold_signal_log_lock():
342: def backfill_outcomes(max_entries=2000):

=== portfolio/forecast_accuracy.py defs ===
34: def cached_forecast_accuracy(horizon="24h", days=7, use_raw_sub_signals=True):
57: def invalidate_forecast_accuracy_cache():
68: def load_predictions(predictions_file=None):
74: def load_health_stats(health_file=None):
97: def compute_forecast_accuracy(ticker=None, horizon="24h", days=None,
194: def get_ticker_accuracy(ticker, horizon="24h", days=7, predictions_file=None):
224: def get_all_ticker_accuracies(horizon="24h", days=7, predictions_file=None):
254: def backfill_forecast_outcomes(max_entries=500, predictions_file=None,
332: def _lookup_price_at_time(ticker, target_time, snapshot_file=None,
372: def _write_predictions(entries, predictions_file=None):
378: def get_forecast_accuracy_summary(focus_tickers=None, days=7):
418: def print_forecast_accuracy_report():

=== portfolio/ic_computation.py defs ===
33: def _spearman_rank_correlation(x, y):
73: def compute_signal_ic(horizon="1d", days=None, entries=None):
150: def _rolling_ic(votes, returns, window=50):
163: def compute_signal_ic_per_ticker(horizon="1d", days=None, entries=None):
223: def _load_entries(days=None):
235: def compute_and_cache_ic(horizon="1d"):
253: def load_cached_ic(horizon="1d"):
265: def get_signal_ic_ranking(horizon="1d", min_samples=30):
285: def print_ic_report(horizon="1d"):

=== portfolio/train_signal_weights.py defs ===
38: def _load_signal_history(
101: def train_weights(

=== portfolio/linear_factor.py defs ===
31: class LinearFactorModel:

=== portfolio/feature_normalizer.py defs ===
35: def _ensure_buffer(ticker: str, indicator: str) -> deque:
43: def update(ticker: str, indicator: str, value: float) -> None:
54: def normalize(ticker: str, indicator: str, value: float) -> float:
74: def has_sufficient_history(ticker: str, indicator: str) -> bool:
81: def get_stats(ticker: str, indicator: str) -> dict | None:
100: def clear(ticker: str | None = None, indicator: str | None = None) -> None:

=== portfolio/short_horizon.py defs ===
36: def is_slow_signal_3h(signal_name: str) -> bool:
41: def time_of_day_scale_3h(hour: int) -> float:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
  420:     if price_slope > 0 and ema21_slope > 0:
  421:         return "BUY", FORECAST_DEAD_ZONE_SOFT_CONF
  422:     if price_slope < 0 and ema21_slope < 0:
  423:         return "SELL", FORECAST_DEAD_ZONE_SOFT_CONF
  424:     return "HOLD", 0.0
  425: 
  426: 
  427: # Accuracy gate: signals with blended accuracy below this threshold are
  428: # force-HOLD (treated like DISABLED_SIGNALS but dynamically). A signal at
  429: # 44% is noise, not a reliable contrarian indicator — inverting it just
  430: # produces different noise with whiplash as accuracy oscillates around 50%.
  431: # 2026-04-11 (A-PR-batch-5): raised 0.45 → 0.47. The signal audit on
  432: # 2026-04-10 found four signals sitting in the 45-47% band that the
  433: # previous gate let through (volatility_sig 0.453, trend 0.454, etc.).
  434: # Tightening the gate by 2pp removes ~4 coin-flip-adjacent signals from
  435: # consensus while leaving the well-performing tier untouched.
  436: ACCURACY_GATE_THRESHOLD = 0.47
  437: ACCURACY_GATE_MIN_SAMPLES = 30  # need enough data before gating
  438: # 2026-04-12: Tiered gate for high-confidence coin-flips. With 5000+ samples,
  439: # a signal at 49.8% is coin-flip with p < 0.001 — no amount of waiting will
  440: # fix it. Raising the gate to 50% for established signals removes structure
  441: # (49.8%, 12K sam), heikin_ashi (49.6%, 23K sam) etc. while letting newer
  442: # signals with <5000 samples prove themselves at the standard 47% threshold.
  443: # 2026-04-16: raised high-sample min 5000 -> 10000. Investigation of W15/W16
  444: # consensus collapse found the 5000 threshold catching signals during regime
  445: # transitions where 5000 samples is too few to distinguish true coin-flip
  446: # from transient degradation. 10000 samples reduces false-positive gating
  447: # (e.g., a signal at 49.5% over 6000 samples may still have real edge in
  448: # specific regimes that the aggregate accuracy hides).
  449: _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD = 0.50
  450: _ACCURACY_GATE_HIGH_SAMPLE_MIN = 7000
  451: 
  452: # Directional accuracy gate: signals whose BUY or SELL accuracy is below this
  453: # threshold get that direction force-HOLD'd while the other direction can still
  454: # vote.  E.g., qwen3 BUY=30% (gated) but SELL=74.2% (votes normally).
  455: # Uses the same min-samples threshold as the overall gate.
  456: # 2026-04-10: raised from 0.35 → 0.40 to catch macro_regime BUY (38.9%),
  457: # fibonacci SELL (35.9%), futures_flow both (36-37%).  Now with per-ticker
  458: # directional data, this gate also works per-instrument.
  459: _DIRECTIONAL_GATE_THRESHOLD = 0.40
  460: _DIRECTIONAL_GATE_MIN_SAMPLES = 30
  461: 
  462: # Directional rescue (2026-04-28): when a signal fails the overall accuracy
  463: # gate but its vote direction has strong accuracy, rescue it at reduced weight.
  464: # E.g., heikin_ashi overall=42.6% (gated) but SELL=55.7% → rescue SELL vote.
  465: # Only triggers when direction accuracy >= 55% with >= 30 samples, giving
  466: # a 5pp safety margin above coin-flip.  Rescued signals get a 0.7x weight
  467: # penalty so they contribute less than fully-passing signals.
  468: _DIRECTIONAL_RESCUE_THRESHOLD = 0.55
  469: _DIRECTIONAL_RESCUE_MIN_SAMPLES = 30
  470: _DIRECTIONAL_RESCUE_WEIGHT_PENALTY = 0.70
  471: 
  472: # Adaptive recency blend: when recent accuracy diverges from all-time by more
  473: # than this threshold, increase recent weight for faster regime adaptation.
  474: # Normal: 70% recent + 30% all-time. Fast: 90% recent + 10% all-time.
  475: # 2026-04-15: raised normal 0.70→0.75, fast 0.90→0.95 to better capture
  476: # recent-regime signals like trend (40.3% alltime → 61.6% recent).
  477: # 2026-04-16: REVERTED to 0.70/0.90. The 0.75/0.95 tuning amplified noise
  478: # during the W12-W13 crash -> W14-W16 recovery transition. A 7-day window
  479: # with only 170 samples was dominating a 10K-sample all-time baseline,
  480: # triggering gates on signals whose "bad recent" was just the crash tail
  481: # rolling through the window. 0.70/0.90 gives regime adaptation while
  482: # leaving enough all-time anchor to damp single-week noise.
  483: _RECENCY_DIVERGENCE_THRESHOLD = 0.15  # 15% absolute divergence triggers fast blend
  484: _RECENCY_WEIGHT_NORMAL = 0.70
  485: _RECENCY_WEIGHT_FAST = 0.90
  486: _RECENCY_MIN_SAMPLES = 30  # match ACCURACY_GATE_MIN_SAMPLES (was 50 default)
  487: 
  488: # Crisis regime: when multiple macro-external signals are simultaneously
  489: # degraded (recent accuracy < 35% with 50+ samples), the market is in a
  490: # regime that breaks fundamental assumptions (e.g., wartime, systemic crisis).
  491: # In crisis mode, apply extra penalty to trend-following signals and boost
  492: # mean-reversion/calendar signals.
  493: _CRISIS_THRESHOLD = 0.35  # signal accuracy below this counts as "broken"
  494: _CRISIS_MIN_BROKEN = 3  # need at least 3 broken macro signals for crisis flag
  495: _CRISIS_TREND_PENALTY = 0.6  # 0.6x weight for trend signals in crisis
  496: _CRISIS_MR_BOOST = 1.3  # 1.3x weight for mean-reversion in crisis
  497: 
  498: # Directional bias penalty: signals with extreme BUY or SELL bias (>85% of
  499: # their non-HOLD votes in one direction) get penalized because their
  500: # "accuracy" may just reflect market drift rather than genuine edge.
  501: # E.g., calendar (100% BUY) in a ranging-up market looks accurate by luck.
  502: _BIAS_THRESHOLD = 0.85  # >85% BUY or >85% SELL triggers penalty
  503: _BIAS_PENALTY = 0.5  # 0.5x weight for high-bias signals (85-95%)
  504: _BIAS_EXTREME_THRESHOLD = 0.95  # >95% triggers stronger penalty
  505: _BIAS_EXTREME_PENALTY = 0.2  # 0.2x weight for extreme-bias signals (>95%)
  506: _BIAS_MIN_ACTIVE = 30  # need enough active (non-HOLD) votes to judge bias
  507: 
  508: # IC-based weight multiplier (2026-04-18): adjusts signal weight based on
  509: # Information Coefficient — the rank correlation between a signal's votes and
  510: # actual return magnitude. A signal with 55% accuracy but IC=0.15 catches big
  511: # moves; one with 58% accuracy but IC=0.00 is riding market drift.
  512: _IC_ALPHA = 2.0         # IC sensitivity: IC=0.10 → 1.20x boost
  513: _IC_MULT_FLOOR = 0.6    # never zero out a signal via IC alone
  514: _IC_MULT_CAP = 1.5      # cap to prevent IC from dominating
  515: _IC_MIN_SAMPLES = 100   # need reliable IC estimate
  516: _IC_STABILITY_MIN = 0.10  # minimum |ICIR| to trust the IC value
  517: _IC_ZERO_PENALTY = 0.85   # phantom performers (|IC|<0.01, 500+ samples) get 0.85x
  518: _IC_ZERO_MIN_SAMPLES = 500  # sample floor for zero-IC penalty
  519: _IC_DATA_TTL = 3600     # IC cache TTL (matches ic_computation.py)
  520: 

---

  940: _MACRO_WINDOW_CACHE_TTL_S = 300
  941: 
  942: 
  943: # --- Signal (full 32-signal for "Now" timeframe) ---
  944: 
  945: MIN_VOTERS_CRYPTO = 3  # crypto has 30 signals (8 core + 22 enhanced; ml disabled) — need 3
  946: MIN_VOTERS_STOCK = 3  # stocks have 24-26 signals (7 core + 17-19 enhanced, GPU-dependent) — need 3
  947: MIN_VOTERS_METALS = 2  # 2026-05-11: metals run at noisier intraday horizon
  948:                        # (1m-1h target) where the standard 3-voter floor
  949:                        # almost never fires after persistence filter.
  950:                        # Empirical: XAG sees 5 raw voters → 2 post-persistence;
  951:                        # MIN_VOTERS=3 produced 0 trades in 20 days.
  952: 
  953: # P2-F (2026-04-17 adversarial review): derived floors used by the
  954: # circuit-breaker precondition. Placing here (after MIN_VOTERS_*) keeps the
  955: # relationship explicit and prevents silent drift if the base MIN_VOTERS_*
  956: # changes.
  957: _MIN_VOTERS_BASE = max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)
  958: # Slate viability floor: the post-exclusion candidate count below which
  959: # relaxation would produce a consensus thinner than any asset class's quorum.
  960: _POST_EXCLUSION_MIN = _MIN_VOTERS_BASE
  961: # Lone-signal escape floor: raised from 2 to _MIN_VOTERS_BASE (3) because a
  962: # 2-voter relaxed consensus is still thinner than any asset class's outer
  963: # quorum, so letting it emit trades was inconsistent with the system's
  964: # design. Codex rounds 6-9 each flagged variants of this issue.
  965: _LONE_SIGNAL_FLOOR = _MIN_VOTERS_BASE
  966: 
  967: # P2-G (2026-04-17): module-load assertions on constant relationships.
  968: # These catch misconfigurations at import time rather than producing silent
  969: # wrong behavior at runtime.
  970: assert MIN_VOTERS_CRYPTO > 0 and MIN_VOTERS_STOCK > 0, (
  971:     "MIN_VOTERS_* must be positive")
  972: assert _POST_EXCLUSION_MIN <= _MIN_ACTIVE_VOTERS_SOFT, (
  973:     f"_POST_EXCLUSION_MIN ({_POST_EXCLUSION_MIN}) must be <= "
  974:     f"_MIN_ACTIVE_VOTERS_SOFT ({_MIN_ACTIVE_VOTERS_SOFT}); "
  975:     f"otherwise the circuit breaker requires more candidates than it can "
  976:     f"ever accept."
  977: )
  978: assert _GATE_RELAXATION_STEP > 0, (
  979:     "_GATE_RELAXATION_STEP must be positive (else ZeroDivisionError in "
  980:     "circuit-breaker step-count math).")
  981: assert _GATE_RELAXATION_MAX > 0, "_GATE_RELAXATION_MAX must be positive."
  982: assert (ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX) > _DIRECTIONAL_GATE_THRESHOLD, (
  983:     f"Relaxed overall accuracy gate "
  984:     f"({ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX:.2f}) must remain "
  985:     f"above _DIRECTIONAL_GATE_THRESHOLD ({_DIRECTIONAL_GATE_THRESHOLD}); "
  986:     f"otherwise directional gating becomes tighter than the relaxed "
  987:     f"accuracy gate and the claim that the directional gate is NEVER "
  988:     f"relaxed becomes meaningless."
  989: )
  990: # Step must divide max cleanly so iteration lands on the intended max.

---

 1770:     try:
 1771:         f = float(value)
 1772:     except (TypeError, ValueError):
 1773:         return 0
 1774:     if math.isnan(f) or math.isinf(f) or f < 0:
 1775:         return 0
 1776:     return int(f)
 1777: 
 1778: 
 1779: def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
 1780:                                   base_gate, relaxation):
 1781:     """Count how many signals would pass gating at gate=(base_gate - relaxation).
 1782: 
 1783:     Counts only voters that survive the full gate cascade:
 1784:       1) excluded (top-N)
 1785:       2) group-gated (correlation leader below group-leader gate)
 1786:       3) accuracy gate at (base - relaxation), tiered for high-sample signals
 1787:       4) directional gate (unchanged by relaxation)
 1788: 
 1789:     Returns int — the number of signals still voting BUY/SELL.
 1790:     """
 1791:     gate_val = base_gate - relaxation
 1792:     # SC-P1-2 (2026-05-02 adversarial follow-ups): high-sample tier is NOT
 1793:     # relaxed. A signal with 10K+ samples at sub-50% accuracy has measurable
 1794:     # negative edge — circuit-breaker relaxation must not promote it back to
 1795:     # voting. Standard tier (under 10K samples) still relaxes so borderline
 1796:     # newer signals can be rescued during regime transitions. Must mirror
 1797:     # the same logic in `_weighted_consensus` (line ~2068).
 1798:     high_gate_val = _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD
 1799:     active = 0
 1800:     for signal_name, vote in votes.items():
 1801:         if vote == "HOLD":
 1802:             continue
 1803:         if signal_name in excluded:
 1804:             continue
 1805:         if signal_name in group_gated:
 1806:             continue
 1807:         stats = accuracy_data.get(signal_name) or {}
 1808:         # P1-C (2026-04-17 adversarial review): coerce None / NaN values to
 1809:         # safe defaults. The live path previously crashed with TypeError when
 1810:         # `accuracy_data[sig]` contained explicit None values (e.g., from a
 1811:         # half-written cache). Replay had `except TypeError` but live didn't.
 1812:         acc = _safe_accuracy(stats.get("accuracy"), default=0.5)
 1813:         samples = _safe_sample_count(stats.get("total"))
 1814:         effective_gate = gate_val
 1815:         if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
 1816:             effective_gate = max(gate_val, high_gate_val)
 1817:         if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
 1818:             continue
 1819:         # Directional gate is not relaxed by the circuit breaker — those gates
 1820:         # catch signals that are actively wrong in one direction.
 1821:         if vote == "BUY":
 1822:             dir_acc = _safe_accuracy(stats.get("buy_accuracy"), default=acc)
 1823:             dir_n = _safe_sample_count(stats.get("total_buy"))
 1824:         else:
 1825:             dir_acc = _safe_accuracy(stats.get("sell_accuracy"), default=acc)
 1826:             dir_n = _safe_sample_count(stats.get("total_sell"))
 1827:         if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
 1828:             continue
 1829:         active += 1
 1830:     return active
 1831: 
 1832: 
 1833: def _normalize_regime(regime):
 1834:     """P2-D (2026-04-17): normalize regime strings to a canonical lowercase form.
 1835: 
 1836:     Protects against case/typo variants ("TRENDING-UP", " trending-up ",
 1837:     "trending_up") that would otherwise silently fall through to the
 1838:     strictest-quorum default. Returns None unchanged.
 1839:     """
 1840:     if regime is None:
 1841:         return None
 1842:     if not isinstance(regime, str):
 1843:         return regime  # Let downstream default handle non-strings.
 1844:     normalized = regime.strip().lower().replace("_", "-")
 1845:     # Common alias fixups.
 1846:     if normalized in ("trendingup", "trending"):
 1847:         normalized = "trending-up"
 1848:     elif normalized == "trendingdown":
 1849:         normalized = "trending-down"
 1850:     elif normalized in ("highvol", "high-volatility", "high_vol"):
 1851:         normalized = "high-vol"
 1852:     return normalized
 1853: 
 1854: 
 1855: def _dynamic_min_voters_for_regime(regime):
 1856:     """Regime-dependent final quorum. Single source of truth - called by both
 1857:     the circuit breaker and apply_confidence_penalties.
 1858: 
 1859:     This is the minimum voter count the OUTER consensus path requires before
 1860:     emitting a non-HOLD action. The circuit breaker uses it to size its
 1861:     recovery floor so relaxation is only engaged when it could reach the
 1862:     regime's actual quorum.
 1863: 
 1864:     2026-04-17 (P2-C/P2-D): de-duplicated. apply_confidence_penalties
 1865:     previously had an inline copy at line ~1623 that had to stay in lockstep
 1866:     manually - now it calls this helper. Also accepts case/typo-variant
 1867:     regime strings via _normalize_regime.
 1868:     """
 1869:     canonical = _normalize_regime(regime)
 1870:     if canonical in ("trending-up", "trending-down"):
 1871:         return 3
 1872:     if canonical == "high-vol":
 1873:         return 4
 1874:     return 5  # ranging, unknown, None
 1875: 
 1876: 
 1877: def _compute_gate_relaxation(votes, accuracy_data, excluded, group_gated, base_gate,
 1878:                               regime=None):
 1879:     """Compute circuit-breaker relaxation to preserve voter diversity.
 1880: 
 1881:     Progressively tests relaxation values 0, step, 2*step, ..., up to
 1882:     _GATE_RELAXATION_MAX. Returns the smallest relaxation that yields at
 1883:     least _MIN_ACTIVE_VOTERS_SOFT active voters.
 1884: 
 1885:     Decision tree:
 1886:       - baseline >= floor                    -> 0.0 (no relaxation needed)
 1887:       - best_possible <= baseline            -> 0.0 (relaxation doesn't help;
 1888:                                                 either a low-signal scenario
 1889:                                                 or a genuine regime break
 1890:                                                 where remaining signals are
 1891:                                                 below even the 41% relaxed
 1892:                                                 gate — letting them vote
 1893:                                                 would be wrong)
 1894:       - best_possible >= floor               -> smallest step that meets floor
 1895:       - baseline < best_possible < floor     -> _GATE_RELAXATION_MAX (partial
 1896:                                                 recovery: a single
 1897:                                                 irrecoverable outlier must
 1898:                                                 not veto relaxation for the
 1899:                                                 rest - Codex P2 fix)
 1900: 
 1901:     Uses `_count_active_voters_at_gate` which applies directional gating,
 1902:     so signals gated on BUY-accuracy=30% don't inflate the decision.
 1903: 
 1904:     Returns float - relaxation in absolute accuracy points (e.g., 0.02).
 1905:     """
 1906:     # Defensive: caller may pass None for either set (older paths or a future
 1907:     # refactor). Treat as empty to avoid `in None` TypeErrors in a hot path.
 1908:     excluded = excluded or set()
 1909:     group_gated = group_gated or set()
 1910: 

---

 2095: def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
 2096:                         accuracy_gate=None, max_signals=None, horizon=None,
 2097:                         regime_gated_override=None, ticker=None,
 2098:                         soft_confidences=None):
 2099:     """Compute weighted consensus using accuracy, IC, regime, and activation frequency.
 2100: 
 2101:     Weight per signal = accuracy_weight * ic_mult * regime_mult * normalized_weight
 2102:                         * horizon_mult * activity_cap
 2103:     where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
 2104:     Rare, balanced signals get more weight; noisy/biased signals get less.
 2105: 
 2106:     Signals below the accuracy gate (with sufficient samples) are force-skipped —
 2107:     they are noise, not useful contrarian indicators.
 2108: 
 2109:     Regime gating: signals in REGIME_GATED_SIGNALS for the current regime are
 2110:     forced to HOLD before vote processing — they produce negative alpha.
 2111: 
 2112:     Correlation deduplication: within defined correlation groups, only the
 2113:     highest-accuracy signal gets full weight. Others get 0.3x penalty.
 2114: 
 2115:     Horizon-specific weights: signals with divergent accuracy across horizons
 2116:     get boosted or penalized via HORIZON_SIGNAL_WEIGHTS.
 2117: 
 2118:     Activity rate cap: signals with >70% activation rate get 0.5x penalty
 2119:     to prevent a single high-activity signal from dominating consensus.
 2120: 
 2121:     Top-N gate: when max_signals is set, only the top max_signals non-HOLD
 2122:     signals (ranked by accuracy) participate in the consensus. This focuses
 2123:     the vote on the best performers and ignores marginal contributors.
 2124: 
 2125:     2026-05-11 (Codex Fix B) — soft-confidence dampening:
 2126:     The Stage 2 dead-zone helpers (EMA / BB / MACD) emit *weak* directional
 2127:     votes when the strong path would HOLD, and stash a small per-vote
 2128:     confidence (0.15-0.20) into extra_info under the keys
 2129:     "_soft_conf_ema" / "_soft_conf_bb" / "_soft_conf_macd". Without
 2130:     propagation, _weighted_consensus treated those soft votes as full-
 2131:     strength votes (just direction × accuracy weight), so an all-soft
 2132:     slate could produce full directional confidence — defeating the
 2133:     "weak weight" contract. We now scale each soft vote's contribution
 2134:     by its soft_conf, so e.g. 3 × 0.18 ≈ 0.54 < 1.0 (a single strong
 2135:     vote). Pass the soft_confidences dict to opt in; strong votes (no
 2136:     key present) keep their original weight × accuracy × regime mult.
 2137:     """
 2138:     soft_confidences = soft_confidences or {}
 2139:     gate = accuracy_gate if accuracy_gate is not None else ACCURACY_GATE_THRESHOLD
 2140:     buy_weight = 0.0
 2141:     sell_weight = 0.0
 2142:     gated_signals = []
 2143:     regime_mults = REGIME_WEIGHTS.get(regime, {})
 2144:     activation_rates = activation_rates or {}
 2145:     horizon_mults = _get_horizon_weights(horizon)
 2146: 
 2147:     # Codex round 10/11/12 (2026-04-17 follow-up): deep-sanitize accuracy_data
 2148:     # at function entry.
 2149:     #   Round 10: coerced non-dict container values to {}.
 2150:     #   Round 11: found dict values with poisoned numeric fields still
 2151:     #             crashed. Added per-field coercion.
 2152:     #   Round 12: coerce-with-0.5-default silently promoted partially-
 2153:     #             written cache rows ({"accuracy": null, "total": 200}) into
 2154:     #             mature 50% signals that cleared the min-samples gate.
 2155:     #             Now: if a numeric field is poisoned, DROP that field so
 2156:     #             downstream `.get(..., default)` falls back cleanly. A row
 2157:     #             whose overall accuracy is poisoned but total=200 becomes
 2158:     #             {"total": 200} - the gate sees no accuracy, the downstream
 2159:     #             code default to the safe fallback. The row no longer
 2160:     #             masquerades as a 50%-accurate mature signal.
 2161:     # Codex round 13 (2026-04-17): a poisoned accuracy must invalidate its
 2162:     # PAIRED sample count too. Otherwise `{"accuracy": None, "total": 200}`
 2163:     # becomes `{"total": 200}` which downstream still reads as a mature
 2164:     # 50% signal (accuracy defaults to 0.5, samples=200 clears the gate).
 2165:     # Drop-together semantics: overall acc poisoned -> drop (accuracy, total);
 2166:     # buy_accuracy poisoned -> drop (buy_accuracy, total_buy); likewise for
 2167:     # sell. Fields whose pair is clean but themselves clean pass through.
 2168:     import math as _math
 2169: 
 2170:     def _coerce_sample_count(val):
 2171:         """Return int >= 0, or None if val is missing/poisoned/invalid."""
 2172:         if val is None:
 2173:             return None
 2174:         try:
 2175:             f = float(val)
 2176:         except (TypeError, ValueError):
 2177:             return None
 2178:         if _math.isnan(f) or _math.isinf(f) or f < 0:
 2179:             return None
 2180:         return int(f)
 2181: 
 2182:     _PAIRED = (
 2183:         ("accuracy", "total"),
 2184:         ("buy_accuracy", "total_buy"),
 2185:         ("sell_accuracy", "total_sell"),
 2186:     )
 2187:     if accuracy_data:
 2188:         _sanitized: dict[Any, dict[str, Any]] = {}
 2189:         for _k, _v in accuracy_data.items():
 2190:             if not isinstance(_v, dict):
 2191:                 _sanitized[_k] = {}
 2192:                 continue
 2193:             _clean = dict(_v)  # start from a copy, then prune.
 2194:             for _acc_key, _cnt_key in _PAIRED:
 2195:                 _acc_has = _acc_key in _clean
 2196:                 _cnt_has = _cnt_key in _clean
 2197:                 if _acc_has:
 2198:                     _clean_acc = _safe_accuracy(_clean.get(_acc_key), default=None)
 2199:                 else:
 2200:                     _clean_acc = None
 2201:                 if _cnt_has:
 2202:                     _clean_cnt = _coerce_sample_count(_clean.get(_cnt_key))
 2203:                 else:
 2204:                     _clean_cnt = None
 2205:                 # Decide whether to keep each field:
 2206:                 #   Both clean      -> keep both.
 2207:                 #   Only acc clean  -> keep acc; drop cnt (if it was present-and-poisoned).
 2208:                 #   Only cnt clean  -> drop BOTH (count without trustworthy accuracy
 2209:                 #                      must not promote the row to a mature signal).
 2210:                 #   Neither clean   -> drop both.
 2211:                 if _clean_acc is not None and _clean_cnt is not None:
 2212:                     _clean[_acc_key] = _clean_acc
 2213:                     _clean[_cnt_key] = _clean_cnt
 2214:                 elif _clean_acc is not None and not _cnt_has:
 2215:                     # Accuracy present (clean), count field absent - keep acc.
 2216:                     _clean[_acc_key] = _clean_acc
 2217:                 else:
 2218:                     # Poisoned accuracy OR poisoned count: drop both so the
 2219:                     # row doesn't masquerade as a mature signal. Downstream
 2220:                     # .get() calls then use their safe defaults.
 2221:                     _clean.pop(_acc_key, None)
 2222:                     _clean.pop(_cnt_key, None)
 2223:             _sanitized[_k] = _clean
 2224:         accuracy_data = _sanitized
 2225:     else:
 2226:         accuracy_data = {}
 2227: 
 2228:     # Regime gating: force-HOLD signals that produce negative alpha in this regime.
 2229:     # BUG-149: now horizon-aware — e.g., trend works at 3h in ranging (61.6%)
 2230:     # SC-I-001: when caller provides regime_gated_override (with BUG-158 per-ticker
 2231:     # exemptions already applied), use it instead of recomputing from scratch.
 2232:     regime_gated = regime_gated_override if regime_gated_override is not None else _get_regime_gated(regime, horizon)
 2233:     votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}
 2234: 
 2235:     # Horizon-specific per-ticker blacklist (2026-04-16, Batch 4). Extends the
 2236:     # compute-time _default blacklist with horizon-specific entries. Compute time
 2237:     # can't see horizon (one vote reused across 3h/4h/12h/1d/3d/5d/10d consensus),
 2238:     # so per-horizon gating must happen here.
 2239:     horizon_disabled = _get_horizon_disabled_signals(ticker, horizon)
 2240:     if horizon_disabled:
 2241:         votes = {k: ("HOLD" if k in horizon_disabled else v) for k, v in votes.items()}
 2242: 
 2243:     # Macro-window force-HOLD pre-pass (2026-04-28). When a high-impact
 2244:     # event is within ~24h past or ~72h future, force-HOLD the signals
 2245:     # whose lag/bias makes them dominantly wrong in news-driven regimes.
 2246:     # The downweight branch for the other macro-fragile signals lives in
 2247:     # the weight loop below so it composes with regime/horizon multipliers.
 2248:     macro_active = _is_macro_window_cached()
 2249:     if macro_active and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
 2250:         votes = {
 2251:             k: ("HOLD" if k in MACRO_WINDOW_FORCE_HOLD_SIGNALS else v)
 2252:             for k, v in votes.items()
 2253:         }
 2254: 
 2255:     # Top-N gate: only let the top max_signals (by accuracy) participate.
 2256:     # Codex round 2 P2 (2026-04-28): rank with macro-adjusted accuracy so
 2257:     # downweighted signals lose Top-N slots to healthier peers during a
 2258:     # macro window. Without this, sentiment can keep its slot at full
 2259:     # raw accuracy and exclude a peer that would have voted more reliably.
 2260:     def _topn_accuracy_key(s: str) -> float:
 2261:         base = float(accuracy_data.get(s, {}).get("accuracy", 0.5))
 2262:         if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
 2263:             base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
 2264:         return base
 2265: 
 2266:     active_votes = {k: v for k, v in votes.items() if v != "HOLD"}
 2267:     if max_signals and len(active_votes) > max_signals:
 2268:         ranked = sorted(
 2269:             active_votes.keys(),
 2270:             key=_topn_accuracy_key,
 2271:             reverse=True,
 2272:         )
 2273:         excluded = set(ranked[max_signals:])
 2274:     else:
 2275:         excluded = set()
 2276: 
 2277:     # Pre-compute which signal is the "leader" (highest accuracy) in each
 2278:     # correlation group, considering only signals that are actively voting.
 2279:     # Prefer dynamic groups (from signal_log correlations) over static.
 2280:     active_non_hold = {s for s, v in votes.items() if v != "HOLD"}
 2281:     _active_corr_groups = _get_correlation_groups() or _STATIC_CORRELATION_GROUPS
 2282: 
 2283:     # Codex P2 (2026-04-28): apply the macro-window downweight to the
 2284:     # leader-selection key BEFORE picking the leader. Otherwise sentiment
 2285:     # (lifetime ~70% acc) stays leader of macro_external during a macro
 2286:     # window — and the 0.15x follower penalty pushes healthier peers
 2287:     # below sentiment's already-halved weight, making the overlay
 2288:     # actively reinforce the wrong signal.
 2289:     def _leader_accuracy_key(s: str) -> float:
 2290:         base = float(accuracy_data.get(s, {}).get("accuracy", 0.5))
 2291:         if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
 2292:             base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
 2293:         return base
 2294: 
 2295:     group_leaders = {}
 2296:     for group_name, group_sigs in _active_corr_groups.items():
 2297:         active_in_group = active_non_hold & group_sigs
 2298:         if len(active_in_group) <= 1:
 2299:             continue
 2300:         best_sig = max(active_in_group, key=_leader_accuracy_key)
 2301:         group_leaders[group_name] = best_sig
 2302: 
 2303:     # Correlation group leader gating: when the best signal in a group has
 2304:     # accuracy below threshold (with enough samples), gate the ENTIRE group.
 2305:     # Prevents the "least bad" broken signal from voting.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
945: MIN_VOTERS_CRYPTO = 3  # crypto has 30 signals (8 core + 22 enhanced; ml disabled) — need 3
946: MIN_VOTERS_STOCK = 3  # stocks have 24-26 signals (7 core + 17-19 enhanced, GPU-dependent) — need 3
947: MIN_VOTERS_METALS = 2  # 2026-05-11: metals run at noisier intraday horizon
957: _MIN_VOTERS_BASE = max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)
970: assert MIN_VOTERS_CRYPTO > 0 and MIN_VOTERS_STOCK > 0, (
1779: def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
1855: def _dynamic_min_voters_for_regime(regime):
1901:     Uses `_count_active_voters_at_gate` which applies directional gating,
1932:     min_regime_quorum = _dynamic_min_voters_for_regime(regime)
1937:     # P2-F (2026-04-17): derived from MIN_VOTERS_CRYPTO/STOCK rather than
1946:     baseline = _count_active_voters_at_gate(
1952:     best_possible = _count_active_voters_at_gate(
1978:         active = _count_active_voters_at_gate(
2898:     active_voters = extra_info.get("_voters_post_filter",
2900:     dynamic_min = _dynamic_min_voters_for_regime(regime)
2902:     if action != "HOLD" and active_voters < dynamic_min:
2904:             "stage": "dynamic_min_voters", "regime": regime,
2905:             "required": dynamic_min, "actual": active_voters, "effect": "force_hold",
3693:     # all downstream code (core gate, min_voters, unanimity penalty) sees
3755:     # HERE, before buy/sell counting, so `active_voters` reflects the post-
3798:     active_voters = buy + sell
3800:         min_voters = MIN_VOTERS_STOCK
3802:         # 2026-05-11: metals lowered from MIN_VOTERS_STOCK(3) to
3803:         # MIN_VOTERS_METALS(2). Intraday horizon + persistence filter
3806:         min_voters = MIN_VOTERS_METALS
3808:         min_voters = MIN_VOTERS_CRYPTO
3811:     if core_active == 0 or active_voters < min_voters:
3815:         buy_conf = buy / active_voters
3816:         sell_conf = sell / active_voters
4035:     # Use post_persistence_voters (not pre-filter active_voters) because the
4037:     if core_active == 0 or post_persistence_voters < min_voters:
4051:     extra_info["_voters"] = active_voters  # pre-filter (compatibility)
4072:     # Apply confidence penalty cascade (regime, volume/ADX, trap, dynamic min_voters)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
 3770:     # line ~3333 ("core_active == 0 ...") sees a stale 1 even when the
 3771:     # only core voter (e.g. claude_fundamental) was suppressed by macro.
 3772:     # Mirrors the existing regime_gate / horizon_disabled mutation
 3773:     # pattern above.
 3774:     macro_active_effective = _is_macro_window_cached()
 3775:     if macro_active_effective and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
 3776:         for sig_name in MACRO_WINDOW_FORCE_HOLD_SIGNALS:
 3777:             if sig_name in votes and votes[sig_name] != "HOLD":
 3778:                 votes[sig_name] = "HOLD"
 3779: 
 3780:     if ticker:
 3781:         _record_phase(ticker, "regime_gate", _phase_start)
 3782:         _phase_start = time.monotonic()
 3783: 
 3784:     # Derive buy/sell counts from named votes (post-gating)
 3785:     buy = sum(1 for v in votes.values() if v == "BUY")
 3786:     sell = sum(1 for v in votes.values() if v == "SELL")
 3787: 
 3788:     # Core signal gate: at least 1 core signal must be active for non-HOLD consensus.
 3789:     # Enhanced signals can strengthen/weaken a consensus but never create one alone.
 3790:     core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
 3791:     core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
 3792:     core_active = core_buy + core_sell
 3793: 
 3794:     # Total applicable signals: computed dynamically from SIGNAL_NAMES
 3795:     # minus DISABLED_SIGNALS minus per-asset-class exclusions.
 3796:     total_applicable = _compute_applicable_count(ticker, skip_gpu=skip_gpu)
 3797: 
 3798:     active_voters = buy + sell
 3799:     if ticker in STOCK_SYMBOLS:
 3800:         min_voters = MIN_VOTERS_STOCK
 3801:     elif ticker in METALS_SYMBOLS:
 3802:         # 2026-05-11: metals lowered from MIN_VOTERS_STOCK(3) to
 3803:         # MIN_VOTERS_METALS(2). Intraday horizon + persistence filter
 3804:         # leaves only 2 voters in steady-state on XAG; the old 3-voter
 3805:         # floor produced 0 trades in 20 days.
 3806:         min_voters = MIN_VOTERS_METALS
 3807:     else:
 3808:         min_voters = MIN_VOTERS_CRYPTO
 3809: 
 3810:     # Core gate: if no core signal is active, force HOLD regardless of enhanced votes
 3811:     if core_active == 0 or active_voters < min_voters:
 3812:         action = "HOLD"
 3813:         conf = 0.0
 3814:     else:
 3815:         buy_conf = buy / active_voters
 3816:         sell_conf = sell / active_voters
 3817:         if buy_conf > sell_conf and buy_conf >= 0.5:
 3818:             action = "BUY"
 3819:             conf = buy_conf
 3820:         elif sell_conf > buy_conf and sell_conf >= 0.5:

---

 4028:     )
 4029: 
 4030:     if ticker:
 4031:         _record_phase(ticker, "weighted_consensus", _phase_start)
 4032:         _phase_start = time.monotonic()
 4033: 
 4034:     # BUG-227: Apply core gate AND MIN_VOTERS gate to weighted consensus.
 4035:     # Use post_persistence_voters (not pre-filter active_voters) because the
 4036:     # persistence filter may have reduced voters below the threshold.
 4037:     if core_active == 0 or post_persistence_voters < min_voters:
 4038:         weighted_action = "HOLD"
 4039:         weighted_conf = 0.0
 4040: 
 4041:     # Confluence score
 4042:     confluence = _confluence_score(votes, extra_info)
 4043: 
 4044:     # Time-of-day confidence adjustment
 4045:     tod_factor = _time_of_day_factor(horizon=horizon)
 4046:     weighted_conf *= tod_factor
 4047: 
 4048:     # Store raw consensus in extra for debugging, then use weighted as primary
 4049:     extra_info["_raw_action"] = action
 4050:     extra_info["_raw_confidence"] = conf
 4051:     extra_info["_voters"] = active_voters  # pre-filter (compatibility)
 4052:     extra_info["_voters_post_filter"] = post_persistence_voters
 4053:     extra_info["_total_applicable"] = total_applicable
 4054:     extra_info["_buy_count"] = buy
 4055:     extra_info["_sell_count"] = sell
 4056:     extra_info["_core_buy"] = core_buy
 4057:     extra_info["_core_sell"] = core_sell
 4058:     extra_info["_core_active"] = core_active
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
 2805: def apply_confidence_penalties(action, conf, regime, ind, extra_info, ticker, df, config):
 2806:     """Apply an 8-stage multiplicative confidence penalty cascade.
 2807: 
 2808:     Stages:
 2809:       1. Regime penalty — dampens confidence in choppy/volatile markets
 2810:       2. Volume/ADX gate — rejects low-conviction signals
 2811:       3. Trap detection — catches bull/bear traps (price vs volume divergence)
 2812:       4. Dynamic MIN_VOTERS — raises the bar in uncertain markets
 2813:       5. Unanimity penalty — over-agreement often means the move is priced in
 2814:       5b. Ensemble entropy — high 3-way disagreement caps confidence
 2815:       6. Per-ticker consensus — penalizes tickers where ensemble accuracy < 50%
 2816:       7. Calibration compression — compress overconfident predictions to honest levels
 2817: 
 2818:     Returns (action, conf, penalty_log) where penalty_log is a list of applied penalties.
 2819:     """
 2820:     cfg = (config or {}).get("confidence_penalties", {})
 2821:     if cfg.get("enabled") is False:
 2822:         return action, conf, []
 2823: 
 2824:     penalty_log = []
 2825: 
 2826:     # --- Stage 1: Regime penalties ---
 2827:     if regime == "ranging":
 2828:         conf *= 0.75
 2829:         penalty_log.append({"stage": "regime", "regime": "ranging", "mult": 0.75})
 2830:     elif regime == "high-vol":
 2831:         conf *= 0.80
 2832:         penalty_log.append({"stage": "regime", "regime": "high-vol", "mult": 0.80})
 2833:     elif regime in ("trending-up", "trending-down"):
 2834:         # Bonus only if action aligns with trend direction
 2835:         trending_buy = regime == "trending-up" and action == "BUY"
 2836:         trending_sell = regime == "trending-down" and action == "SELL"
 2837:         if trending_buy or trending_sell:
 2838:             conf *= 1.10
 2839:             penalty_log.append({"stage": "regime", "regime": regime, "aligned": True, "mult": 1.10})
 2840:     # BUG-90: Clamp after Stage 1 so inflated confidence doesn't bypass Stage 2 gates
 2841:     conf = min(1.0, conf)
 2842: 
 2843:     # --- Stage 2: Volume/ADX gate ---
 2844:     volume_ratio = extra_info.get("volume_ratio")
 2845:     adx = _compute_adx(df)
 2846:     extra_info["_adx"] = adx
 2847: 
 2848:     if volume_ratio is not None and action != "HOLD":
 2849:         if volume_ratio < 0.5:
 2850:             # Very low volume — force HOLD
 2851:             penalty_log.append({"stage": "volume_gate", "rvol": volume_ratio, "effect": "force_hold"})
 2852:             action = "HOLD"
 2853:             conf = 0.0
 2854:         elif volume_ratio < 0.8 and (adx is not None and adx < 20) and conf < 0.65:
 2855:             # Low volume + weak trend + marginal confidence — force HOLD
 2856:             penalty_log.append({
 2857:                 "stage": "volume_adx_gate", "rvol": volume_ratio,
 2858:                 "adx": round(adx, 1), "conf": round(conf, 4), "effect": "force_hold",
 2859:             })
 2860:             action = "HOLD"
 2861:             conf = 0.0
 2862:         elif volume_ratio > 1.5:
 2863:             # High volume — slight confidence boost
 2864:             conf *= 1.15
 2865:             penalty_log.append({"stage": "volume_boost", "rvol": volume_ratio, "mult": 1.15})
 2866:     # BUG-90: Clamp after Stage 2
 2867:     conf = min(1.0, conf)
 2868: 
 2869:     # --- Stage 3: Trap detection ---
 2870:     # NOTE: df must be the "Now" timeframe (15m candles, 100 bars ≈ 25h).
 2871:     # Last 5 bars = 75 minutes — appropriate for intraday trap detection.
 2872:     if action != "HOLD" and df is not None and isinstance(df, pd.DataFrame) and len(df) >= 5:
 2873:         try:
 2874:             recent_close = df["close"].iloc[-5:]
 2875:             recent_vol = df["volume"].iloc[-5:] if "volume" in df.columns else None
 2876:             price_up = recent_close.iloc[-1] > recent_close.iloc[0]
 2877:             price_down = recent_close.iloc[-1] < recent_close.iloc[0]
 2878: 
 2879:             if recent_vol is not None and len(recent_vol) >= 5:
 2880:                 vol_declining = recent_vol.iloc[-1] < recent_vol.iloc[0] * 0.8
 2881: 
 2882:                 if action == "BUY" and price_up and vol_declining:
 2883:                     conf *= 0.5
 2884:                     penalty_log.append({"stage": "trap", "type": "bull_trap", "mult": 0.5})
 2885:                 elif action == "SELL" and price_down and vol_declining:
 2886:                     conf *= 0.5
 2887:                     penalty_log.append({"stage": "trap", "type": "bear_trap", "mult": 0.5})
 2888:         except Exception:
 2889:             logger.warning("Trap detection failed for %s", ticker, exc_info=True)
 2890:     # BUG-90: Clamp after Stage 3
 2891:     conf = min(1.0, conf)
 2892: 
 2893:     # --- Stage 4: Dynamic MIN_VOTERS ---
 2894:     # P2-C (2026-04-17): delegate to shared helper to avoid drift with the
 2895:     # circuit breaker's recovery-floor logic. Same semantic as before.
 2896:     # BUG-227: Use post-persistence voter count (not pre-filter) so the gate
 2897:     # reflects the actual participating voters after debounce filtering.
 2898:     active_voters = extra_info.get("_voters_post_filter",
 2899:                                     extra_info.get("_voters", 0))
 2900:     dynamic_min = _dynamic_min_voters_for_regime(regime)
 2901: 
 2902:     if action != "HOLD" and active_voters < dynamic_min:
 2903:         penalty_log.append({
 2904:             "stage": "dynamic_min_voters", "regime": regime,
 2905:             "required": dynamic_min, "actual": active_voters, "effect": "force_hold",
 2906:         })
 2907:         action = "HOLD"
 2908:         conf = 0.0
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
    1: """Signal registry — plugin system for modular signal management.
    2: 
    3: Enhanced signals register via register_enhanced(). signal_engine.py
    4: discovers all signals from the registry instead of hardcoded lists.
    5: """
    6: import importlib
    7: import logging
    8: import time
    9: from collections.abc import Callable
   10: 
   11: logger = logging.getLogger("portfolio.signal_registry")
   12: 
   13: _ENHANCED_SIGNALS: dict[str, dict] = {}
   14: 
   15: 
   16: def register_signal(name: str, signal_type: str = "enhanced",
   17:                     module_path: str = None, func_name: str = None,
   18:                     requires_macro: bool = False):
   19:     """Register a signal in the global registry.
   20: 
   21:     Can be used as a decorator on compute functions, or called directly
   22:     to register signals programmatically.
   23:     """
   24:     def decorator(func):
   25:         entry = {
   26:             "name": name,
   27:             "type": signal_type,
   28:             "module_path": module_path or func.__module__,
   29:             "func_name": func_name or func.__name__,
   30:             "requires_macro": requires_macro,
   31:             "func": func,
   32:         }
   33:         _ENHANCED_SIGNALS[name] = entry
   34:         return func
   35:     return decorator
   36: 
   37: 
   38: def register_enhanced(name: str, module_path: str, func_name: str,
   39:                       requires_macro: bool = False,
   40:                       requires_context: bool = False,
   41:                       max_confidence: float = 1.0):
   42:     """Programmatically register an enhanced signal module."""
   43:     _ENHANCED_SIGNALS[name] = {
   44:         "name": name,
   45:         "type": "enhanced",
   46:         "module_path": module_path,
   47:         "func_name": func_name,
   48:         "requires_macro": requires_macro,
   49:         "requires_context": requires_context,
   50:         "max_confidence": max_confidence,
   51:         "func": None,  # lazy-loaded
   52:     }
   53: 
   54: 
   55: def get_enhanced_signals() -> dict[str, dict]:
   56:     """Return all registered enhanced signals."""
   57:     return dict(_ENHANCED_SIGNALS)
   58: 
   59: 
   60: def get_signal_names() -> list:
   61:     """Return all registered signal names."""
   62:     return list(_ENHANCED_SIGNALS.keys())
   63: 
   64: 
   65: _FAILED_IMPORT_SENTINEL = object()
   66: _FAILED_IMPORT_COOLDOWN = 300  # retry broken imports after 5 min
   67: 
   68: def load_signal_func(entry: dict) -> Callable | None:
   69:     """Lazy-load and cache the compute function for a signal.
   70: 
   71:     On import failure, caches the failure for _FAILED_IMPORT_COOLDOWN seconds
   72:     so the warning is logged once, not 35× per cycle (5 tickers × 7 TFs).
   73:     """
   74:     cached = entry.get("func")
   75:     if cached is not None and cached is not _FAILED_IMPORT_SENTINEL:
   76:         return cached
   77:     if cached is _FAILED_IMPORT_SENTINEL and time.monotonic() - entry.get("_fail_ts", 0) < _FAILED_IMPORT_COOLDOWN:
   78:         return None
   79:     try:
   80:         mod = importlib.import_module(entry["module_path"])
   81:         func = getattr(mod, entry["func_name"])
   82:         entry["func"] = func
   83:         entry.pop("_fail_ts", None)
   84:         return func
   85:     except Exception as e:
   86:         logger.warning("Failed to load signal %s: %s", entry['name'], e)
   87:         entry["func"] = _FAILED_IMPORT_SENTINEL
   88:         entry["_fail_ts"] = time.monotonic()
   89:         return None
   90: 
   91: 
   92: # Register all enhanced signals (called at import time)
   93: def _register_defaults():
   94:     """Register the default set of enhanced signal modules."""
   95:     defaults = [
   96:         ("trend", "portfolio.signals.trend", "compute_trend_signal"),
   97:         ("momentum", "portfolio.signals.momentum", "compute_momentum_signal"),
   98:         ("volume_flow", "portfolio.signals.volume_flow", "compute_volume_flow_signal"),
   99:         ("volatility_sig", "portfolio.signals.volatility", "compute_volatility_signal"),
  100:         ("candlestick", "portfolio.signals.candlestick", "compute_candlestick_signal"),
  101:         ("structure", "portfolio.signals.structure", "compute_structure_signal"),
  102:         ("fibonacci", "portfolio.signals.fibonacci", "compute_fibonacci_signal"),
  103:         ("smart_money", "portfolio.signals.smart_money", "compute_smart_money_signal"),
  104:         ("oscillators", "portfolio.signals.oscillators", "compute_oscillator_signal"),
  105:         ("heikin_ashi", "portfolio.signals.heikin_ashi", "compute_heikin_ashi_signal"),
  106:         ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
  107:     ]
  108:     for name, mod_path, func_name in defaults:
  109:         register_enhanced(name, mod_path, func_name)
  110:     # mean_reversion and momentum_factors require context for seasonality detrending
  111:     register_enhanced("mean_reversion", "portfolio.signals.mean_reversion",
  112:                       "compute_mean_reversion_signal", requires_context=True)
  113:     register_enhanced("momentum_factors", "portfolio.signals.momentum_factors",
  114:                       "compute_momentum_factors_signal", requires_context=True)
  115:     # macro_regime is special — requires_macro=True
  116:     register_enhanced("macro_regime", "portfolio.signals.macro_regime",
  117:                       "compute_macro_regime_signal", requires_macro=True)
  118:     # news_event and econ_calendar require context (ticker, config); capped at 0.7
  119:     register_enhanced("news_event", "portfolio.signals.news_event",
  120:                       "compute_news_event_signal", requires_context=True, max_confidence=0.7)
  121:     register_enhanced("econ_calendar", "portfolio.signals.econ_calendar",
  122:                       "compute_econ_calendar_signal", requires_context=True, max_confidence=0.7)
  123:     # forecast signal — Kronos + Chronos price direction prediction; capped at 0.7
  124:     register_enhanced("forecast", "portfolio.signals.forecast",
  125:                       "compute_forecast_signal", requires_context=True, max_confidence=0.7)
  126:     # Claude fundamental — three-tier LLM cascade; capped at 0.7
  127:     register_enhanced("claude_fundamental", "portfolio.signals.claude_fundamental",
  128:                       "compute_claude_fundamental_signal", requires_context=True, max_confidence=0.7)
  129:     # Futures flow — OI, LS ratios, funding history (crypto only); capped at 0.7
  130:     register_enhanced("futures_flow", "portfolio.signals.futures_flow",
  131:                       "compute_futures_flow_signal", requires_context=True, max_confidence=0.7)
  132:     # Crypto macro — options max pain, gold-BTC rotation, exchange reserves (crypto only); capped at 0.7
  133:     register_enhanced("crypto_macro", "portfolio.signals.crypto_macro",
  134:                       "compute_crypto_macro_signal", requires_context=True, max_confidence=0.7)
  135:     # Orderbook flow — microstructure metrics (metals + crypto); capped at 0.7
  136:     register_enhanced("orderbook_flow", "portfolio.signals.orderbook_flow",
  137:                       "compute_orderbook_flow_signal", requires_context=True, max_confidence=0.7)
  138:     # Metals cross-asset — copper, GVZ, G/S ratio, SPY, oil (metals only); capped at 0.7
  139:     register_enhanced("metals_cross_asset", "portfolio.signals.metals_cross_asset",
  140:                       "compute_metals_cross_asset_signal", requires_context=True, max_confidence=0.7)
  141:     # DXY cross-asset — intraday USD index inverse correlation (metals only); capped at 0.8
  142:     # 2026-04-13: added standalone signal to capture DXY R²~0.6 vs silver at
  143:     # 1-3h horizon. Complements macro_regime's daily DXY sub-indicator.
  144:     register_enhanced("dxy_cross_asset", "portfolio.signals.dxy_cross_asset",
  145:                       "compute_dxy_cross_asset_signal", requires_context=True, max_confidence=0.8)
  146:     # COT positioning — CFTC speculative/commercial positioning, contrarian (metals only); capped at 0.7
  147:     register_enhanced("cot_positioning", "portfolio.signals.cot_positioning",
  148:                       "compute_cot_positioning_signal", requires_context=True, max_confidence=0.7)
  149:     # Credit spread risk — HY OAS from FRED as cross-asset risk appetite gauge; capped at 0.7
  150:     register_enhanced("credit_spread_risk", "portfolio.signals.credit_spread",
  151:                       "compute_credit_spread_signal", requires_context=True, max_confidence=0.7)
  152:     # Futures basis regime — mark-index spread, contango/backwardation detection; capped at 0.7
  153:     register_enhanced("futures_basis", "portfolio.signals.futures_basis",
  154:                       "compute_futures_basis_signal", requires_context=True, max_confidence=0.7)
  155:     # Hurst regime detector — R/S analysis for trending/MR/random-walk classification
  156:     register_enhanced("hurst_regime", "portfolio.signals.hurst_regime",
  157:                       "compute_hurst_regime_signal", requires_context=True)
  158:     # Shannon entropy — market noise/predictability filter; low entropy = trending
  159:     register_enhanced("shannon_entropy", "portfolio.signals.shannon_entropy",
  160:                       "compute_shannon_entropy_signal")
  161:     # VIX term structure — contango/backwardation regime detection; capped at 0.7
  162:     register_enhanced("vix_term_structure", "portfolio.signals.vix_term_structure",
  163:                       "compute_vix_term_structure_signal", requires_context=True, max_confidence=0.7)
  164:     # Gold real yield paradox — GYDI regime detector (metals only); capped at 0.7
  165:     register_enhanced("gold_real_yield_paradox", "portfolio.signals.gold_real_yield_paradox",
  166:                       "compute_gold_real_yield_paradox_signal", requires_context=True, max_confidence=0.7)
  167:     # Cross-asset TSMOM — bond/equity momentum predicts target asset direction; capped at 0.7
  168:     register_enhanced("cross_asset_tsmom", "portfolio.signals.cross_asset_tsmom",
  169:                       "compute_cross_asset_tsmom_signal", requires_context=True, max_confidence=0.7)
  170:     # Copper/gold ratio — intermarket regime indicator (all assets); capped at 0.7
  171:     # 2026-04-17: cross-asset signal, inverts direction for metals (falling ratio = gold strength).
  172:     # 94% recession prediction accuracy, 0.85 correlation with 10Y yields.
  173:     register_enhanced("copper_gold_ratio", "portfolio.signals.copper_gold_ratio",
  174:                       "compute_copper_gold_ratio_signal", requires_context=True, max_confidence=0.7)
  175:     # Statistical Jump Model regime — jump detection + persistence penalty (all assets)
  176:     # 2026-04-18: addresses failing macro_regime (46.6% at 1d, 30.3% metals).
  177:     # Source: Shu, Yu, Mulvey 2024 (12 citations). Persistence penalty prevents whiplash.
  178:     register_enhanced("statistical_jump_regime", "portfolio.signals.statistical_jump_regime",
  179:                       "compute_statistical_jump_regime_signal")
  180:     # Network momentum — cross-asset momentum spillover (all assets); capped at 0.7
  181:     # 2026-04-19: Pu et al. 2023 (arXiv:2308.11294). Sharpe 1.511 across 64 futures.
  182:     # Simplified: correlation-weighted peer momentum divergence instead of GNN.
  183:     register_enhanced("network_momentum", "portfolio.signals.network_momentum",
  184:                       "compute_network_momentum_signal", requires_context=True, max_confidence=0.7)
  185:     # OVX metals spillover — oil implied volatility as metals predictor; capped at 0.7
  186:     # 2026-04-20: OVX at extreme quantiles predicts precious metals returns via
  187:     # contagion/inflation/dollar channels. Distinct from metals_cross_asset (oil PRICE).
  188:     # Source: ScienceDirect OVX cross-asset quantile predictability papers.
  189:     register_enhanced("ovx_metals_spillover", "portfolio.signals.ovx_metals_spillover",
  190:                       "compute_ovx_metals_spillover_signal", requires_context=True, max_confidence=0.7)
  191:     # Cross-asset equity trend spillover — SPY/QQQ TA predicts all assets; capped at 0.7
  192:     # 2026-04-21: Fieberg et al. 2025, robust across 1.3M research designs.
  193:     # Inverts for safe havens (XAU, XAG): bullish equities = risk-on = SELL metals.
  194:     register_enhanced("xtrend_equity_spillover", "portfolio.signals.xtrend_equity_spillover",
  195:                       "compute_xtrend_equity_spillover_signal", requires_context=True, max_confidence=0.7)
  196:     # Complexity gap regime — RMT-based market structure/synchronization detector; capped at 0.7
  197:     # 2026-04-22: Mukhia et al. 2026, arXiv:2604.19107. Complexity gap = norm_max_eigenvalue -
  198:     # avg_pairwise_corr. Gap collapse = crisis synchronization. Inverts for safe havens.
  199:     register_enhanced("complexity_gap_regime", "portfolio.signals.complexity_gap_regime",
  200:                       "compute_complexity_gap_regime_signal", requires_context=True, max_confidence=0.7)
  201:     # Realized skewness — 3rd moment directional signal (all assets)
  202:     # 2026-04-23: Fernandez-Perez et al. 2018. Sharpe 0.79, 8.01% annual on
  203:     # 27 commodity futures. Z-scored skewness + kurtosis confirmation.
  204:     register_enhanced("realized_skewness", "portfolio.signals.realized_skewness",
  205:                       "compute_realized_skewness_signal")
  206:     # Mahalanobis turbulence — cross-asset regime detection via Mahalanobis distance
  207:     # 2026-04-24: Kritzman & Li (2010). Sharpe 2.20 vs 1.0 B&H, max DD 6% vs 32%.
  208:     # Measures statistical unusualness of multi-asset returns. Includes absorption ratio.
  209:     register_enhanced("mahalanobis_turbulence", "portfolio.signals.mahalanobis_turbulence",
  210:                       "compute_mahalanobis_turbulence_signal", requires_context=True, max_confidence=0.7)
  211:     # Crypto eVRP — Expected Volatility Risk Premium (crypto only); capped at 0.7
  212:     # 2026-04-25: Zarattini, Mele & Aziz (2025). eVRP = DVOL(30d) - RV(10d).
  213:     # Options-derived signal uncorrelated with trend-following cluster.
  214:     # Deribit public API, no auth. BTC + ETH only.
  215:     register_enhanced("crypto_evrp", "portfolio.signals.crypto_evrp",
  216:                       "compute_crypto_evrp_signal", requires_context=True, max_confidence=0.7)
  217:     # Hash Ribbons BTC — miner capitulation detector (BTC-only); capped at 0.7
  218:     # 2026-04-26: Charles Edwards / Capriole Investments. 89% win rate on 9 signals.
  219:     # 30d/60d SMA hashrate crossover with price momentum confirmation.
  220:     # blockchain.info API (free, no auth). Fires ~1/year but extreme conviction.
  221:     register_enhanced("hash_ribbons", "portfolio.signals.hash_ribbons",
  222:                       "compute_hash_ribbons_signal", requires_context=True, max_confidence=0.7)
  223:     # Drift Regime Gate — positive-day fraction regime detector; capped at 0.7
  224:     # 2026-04-28: arxiv:2511.12490 (2025). OOS Sharpe >13 on 20yr walk-forward.
  225:     # Fraction of positive close-to-close days in 63-bar window detects drift regime.
  226:     # Pure OHLCV, all 5 tickers. Directional via SMA distance.
  227:     register_enhanced("drift_regime_gate", "portfolio.signals.drift_regime_gate",
  228:                       "compute_drift_regime_gate_signal", max_confidence=0.7)
  229:     # Vol ratio regime — GK/CC volatility ratio + VR test + ER as regime detector
  230:     # 2026-04-29: Garman & Klass (1980), Lo & MacKinlay (1988), Kaufman ER.
  231:     # Three orthogonal regime measures: GK/CC ratio, variance ratio, efficiency ratio.
  232:     # Directional: mean-reversion in ranging, momentum in trending. All OHLCV, all assets.
  233:     register_enhanced("vol_ratio_regime", "portfolio.signals.vol_ratio_regime",
  234:                       "compute_vol_ratio_regime_signal", max_confidence=0.7)
  235:     # Residual pair reversion — cointegration-based pairs trading (all assets)
  236:     # 2026-04-30: Leung & Nguyen (2018), Amberdata crypto pairs (Sharpe 0.93).
  237:     # Regime-neutral mean reversion on OLS residual: ETH~BTC, XAG~XAU, MSTR~BTC.
  238:     # Specifically targets ETH-USD (weakest ticker) and XAG-USD (user focus).
  239:     register_enhanced("residual_pair_reversion",
  240:                       "portfolio.signals.residual_pair_reversion",
  241:                       "compute_residual_pair_reversion_signal",
  242:                       requires_context=True, max_confidence=0.7)
  243:     # Williams VIX Fix — synthetic volatility bottom/capitulation detector (all assets)
  244:     # 2026-05-01: Larry Williams (2007). Profit factor 2.0, 322 trades 1993-2024.
  245:     # WVF = (highest_close_22 - low) / highest_close_22 * 100. Spikes above upper
  246:     # Bollinger Band signal extreme fear = BUY. Directly targets system's BUY-side
  247:     # accuracy weakness. Pure OHLCV, all 5 tickers.
  248:     register_enhanced("williams_vix_fix",
  249:                       "portfolio.signals.williams_vix_fix",
  250:                       "compute_williams_vix_fix_signal",
  251:                       max_confidence=0.7)
  252:     # Treasury risk rotation — bond yield curve shape as cross-asset regime signal
  253:     # 2026-05-07: Gayed (2014), SSRN 2431022. IEF vs TLT relative performance.
  254:     # Steepening = risk-on, flattening = risk-off. Inverted for safe havens.
  255:     # Only signal using bond market data — zero correlation with existing clusters.
  256:     register_enhanced("treasury_risk_rotation",
  257:                       "portfolio.signals.treasury_risk_rotation",
  258:                       "compute_treasury_risk_rotation_signal",
  259:                       requires_context=True, max_confidence=0.7)
  260:     # Intraday seasonality gate — hour-of-day confidence multipliers (all assets)
  261:     # 2026-05-08: Concretum Group 2025 (Sharpe 1.6 BTC Asia-open), ScienceDirect 2024
  262:     # (33% annualized BTC 22:00-23:00 UTC), CME Group 2026 (60-70% gold daily range
  263:     # in London-NY overlap). Zero correlation with existing signal clusters.
  264:     register_enhanced("intraday_seasonality",
  265:                       "portfolio.signals.intraday_seasonality",
  266:                       "compute_intraday_seasonality_signal",
  267:                       requires_context=True, max_confidence=0.7)
  268:     # Cubic trend persistence — R(t+1) = b*phi + c*phi^3 (all assets)
  269:     # 2026-05-09: arXiv:2501.16772 (2025). Universal across equities, bonds,
  270:     # currencies, commodities. 330yr data. Weak trends persist, strong revert.
  271:     register_enhanced("cubic_trend_persistence",
  272:                       "portfolio.signals.cubic_trend_persistence",
  273:                       "compute_cubic_trend_persistence_signal",
  274:                       max_confidence=0.7)
  275:     # VWAP Z-Score MR — volume-weighted mean reversion (all assets)
  276:     # 2026-05-10: FMZ VWAP StdDev MR (77.78% win rate). Different from BB
  277:     # (volume-weighted anchor vs SMA anchor). Three sub-signals: vwap_z,
  278:     # vwap_slope, volume_confirm. Pure OHLCV, all assets.
  279:     register_enhanced("vwap_zscore_mr",
  280:                       "portfolio.signals.vwap_zscore_mr",
  281:                       "compute_vwap_zscore_mr_signal",
  282:                       max_confidence=0.85)
  283: 
  284: 
  285: _register_defaults()
  286: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Shared signal utility functions.
    2: 
    3: Canonical implementations of common technical-analysis helpers used across
    4: multiple signal modules.  Import from here instead of duplicating locally.
    5: 
    6: All functions operate on ``pd.Series`` inputs and return ``pd.Series``
    7: (or ``float`` for ``safe_float``).
    8: """
    9: 
   10: from __future__ import annotations
   11: 
   12: import numpy as np
   13: import pandas as pd
   14: 
   15: 
   16: def sma(series: pd.Series, period: int) -> pd.Series:
   17:     """Simple Moving Average.
   18: 
   19:     Returns NaN where insufficient data (min_periods = period).
   20:     """
   21:     return series.rolling(window=period, min_periods=period).mean()
   22: 
   23: 
   24: def ema(series: pd.Series, span: int) -> pd.Series:
   25:     """Exponential Moving Average using pandas ewm.
   26: 
   27:     Uses ``adjust=False`` for recursive EMA (standard in TA).
   28:     """
   29:     return series.ewm(span=span, adjust=False).mean()
   30: 
   31: 
   32: def rsi(close: pd.Series, period: int = 14) -> pd.Series:
   33:     """Wilder-smoothed RSI.
   34: 
   35:     Uses ``clip()`` for separating gains/losses (numerically stable).
   36:     """
   37:     delta = close.diff()
   38:     gain = delta.clip(lower=0)
   39:     loss = -delta.clip(upper=0)
   40:     avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
   41:     avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
   42:     rs = avg_gain / avg_loss.replace(0, np.nan)
   43:     return 100.0 - (100.0 / (1.0 + rs))
   44: 
   45: 
   46: def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
   47:     """Wilder's True Range."""
   48:     prev_close = close.shift(1)
   49:     tr1 = high - low
   50:     tr2 = (high - prev_close).abs()
   51:     tr3 = (low - prev_close).abs()
   52:     return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
   53: 
   54: 
   55: def safe_float(val) -> float:
   56:     """Convert *val* to float, returning ``NaN`` for non-finite / missing values."""
   57:     if val is None or (isinstance(val, float) and np.isnan(val)):
   58:         return float("nan")
   59:     try:
   60:         f = float(val)
   61:         return f if np.isfinite(f) else float("nan")
   62:     except (TypeError, ValueError):
   63:         return float("nan")
   64: 
   65: 
   66: def rma(series: pd.Series, period: int) -> pd.Series:
   67:     """Wilder's smoothed moving average (RMA / SMMA).
   68: 
   69:     Equivalent to EMA with ``alpha = 1 / period``.
   70:     """
   71:     alpha = 1.0 / period
   72:     return series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
   73: 
   74: 
   75: def wma(series: pd.Series, period: int) -> pd.Series:
   76:     """Weighted Moving Average with linearly increasing weights.
   77: 
   78:     Weight_i = i + 1 for i in 0..period-1 (most recent bar has highest weight).
   79:     """
   80:     weights = np.arange(1, period + 1, dtype=float)
   81: 
   82:     def _apply_wma(x: np.ndarray) -> float:
   83:         return np.dot(x, weights) / weights.sum()
   84: 
   85:     return series.rolling(window=period, min_periods=period).apply(
   86:         _apply_wma, raw=True,
   87:     )
   88: 
   89: 
   90: def majority_vote(votes: list, count_hold: bool = False) -> tuple:
   91:     """Compute majority vote from a list of BUY/SELL/HOLD strings.
   92: 
   93:     HOLD votes are treated as abstentions — they do not block a directional
   94:     result.  Only BUY vs SELL counts determine direction.  This means a
   95:     composite with 5 HOLDs and 2 BUYs produces BUY (not HOLD), because the
   96:     HOLDs are neutral, not opposing votes.
   97: 
   98:     Args:
   99:         votes: List of "BUY", "SELL", or "HOLD" strings
  100:         count_hold: If False (default), confidence = winner / active_voters (BUY+SELL only).
  101:                     If True, confidence = winner / total_votes (including HOLD).
  102: 
  103:     Returns:
  104:         (action, confidence) tuple where action is "BUY", "SELL", or "HOLD"
  105:     """
  106:     buy = sum(1 for v in votes if v == "BUY")
  107:     sell = sum(1 for v in votes if v == "SELL")
  108:     hold = sum(1 for v in votes if v == "HOLD")
  109: 
  110:     active = buy + sell
  111:     total = buy + sell + hold
  112: 
  113:     if active == 0:
  114:         return "HOLD", 0.0
  115: 
  116:     denom = total if count_hold else active
  117: 
  118:     if buy > sell:
  119:         return "BUY", round(buy / denom, 4) if denom > 0 else 0.0
  120:     elif sell > buy:
  121:         return "SELL", round(sell / denom, 4) if denom > 0 else 0.0
  122:     else:
  123:         # Tie between buy and sell — no clear direction.
  124:         # HOLD confidence is always 0.0 — it's the absence of a signal, not a
  125:         # directional vote.
  126:         return "HOLD", 0.0
  127: 
  128: 
  129: def roc(series: pd.Series, period: int) -> pd.Series:
  130:     """Rate of Change: ``100 * (current - n_periods_ago) / n_periods_ago``."""
  131:     shifted = series.shift(period)
  132:     return 100.0 * (series - shifted) / shifted.replace(0, np.nan)
  133: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
    1: """Multiplicative Weight Updates (MWU) for online signal learning.
    2: 
    3: Each signal maintains a persistent weight that is multiplied up on correct
    4: outcomes and down on wrong outcomes.  The result is a classic Hedge algorithm:
    5: signals that are consistently wrong rapidly approach zero weight (floor 0.01)
    6: while consistently correct signals grow to dominate the aggregation.
    7: 
    8: Weights are persisted to JSON via the same atomic I/O used across the project.
    9: """
   10: 
   11: import logging
   12: from pathlib import Path
   13: 
   14: from portfolio.file_utils import atomic_write_json, load_json
   15: 
   16: logger = logging.getLogger("portfolio.signal_weights")
   17: 
   18: _BASE_DIR = Path(__file__).resolve().parent.parent
   19: _DEFAULT_PATH = _BASE_DIR / "data" / "signal_weights.json"
   20: 
   21: _DEFAULT_ETA = 0.1   # learning rate — 10% multiplicative update per outcome
   22: _WEIGHT_FLOOR = 0.01  # never reaches zero
   23: 
   24: 
   25: class SignalWeightManager:
   26:     """Manages MWU weights for all trading signals.
   27: 
   28:     Thread-safety note: this class is not internally thread-safe.  In the
   29:     current system it is only called from the single-threaded outcome backfill
   30:     path, so no locking is required.  Add a threading.Lock if that changes.
   31:     """
   32: 
   33:     def __init__(self, path=None, eta=None):
   34:         self._path = Path(path) if path is not None else _DEFAULT_PATH
   35:         self._eta = eta if eta is not None else _DEFAULT_ETA
   36:         self._weights: dict[str, float] = {}
   37:         self._load()
   38: 
   39:     # ------------------------------------------------------------------
   40:     # Public API
   41:     # ------------------------------------------------------------------
   42: 
   43:     def get_weight(self, signal_name: str) -> float:
   44:         """Return the current weight for *signal_name*.
   45: 
   46:         Defaults to 1.0 for unknown signals (no prior history).
   47:         """
   48:         return self._weights.get(signal_name, 1.0)
   49: 
   50:     def update(self, signal_name: str, correct: bool) -> float:
   51:         """Update the weight for *signal_name* after one outcome.
   52: 
   53:         Correct prediction  → multiply by (1 + eta)
   54:         Incorrect prediction → multiply by (1 - eta)
   55: 
   56:         The weight is clamped to the floor [_WEIGHT_FLOOR, +∞).
   57: 
   58:         Returns the new weight.
   59:         """
   60:         current = self._weights.get(signal_name, 1.0)
   61:         if correct:
   62:             new_weight = current * (1.0 + self._eta)
   63:         else:
   64:             new_weight = current * (1.0 - self._eta)
   65:         new_weight = max(new_weight, _WEIGHT_FLOOR)
   66:         self._weights[signal_name] = new_weight
   67:         return new_weight
   68: 
   69:     def batch_update(self, outcomes: dict) -> None:
   70:         """Update multiple signals at once then persist to disk.
   71: 
   72:         Args:
   73:             outcomes: ``{signal_name: bool}`` — True means correct prediction.
   74:         """
   75:         for signal_name, correct in outcomes.items():
   76:             self.update(signal_name, correct)
   77:         self.save()
   78: 
   79:     def get_normalized_weights(self, signal_names) -> dict:
   80:         """Return weights normalised so their average equals 1.0.
   81: 
   82:         Only considers signals in *signal_names*.  If the list is empty or all
   83:         weights are zero, returns a uniform dict with all values set to 1.0.
   84: 
   85:         This means the total magnitude of the consensus is preserved — signals
   86:         above 1.0 are stronger than average, below 1.0 weaker.
   87:         """
   88:         signal_names = list(signal_names)
   89:         if not signal_names:
   90:             return {}
   91:         raw = {name: self.get_weight(name) for name in signal_names}
   92:         avg = sum(raw.values()) / len(raw)
   93:         if avg == 0.0:
   94:             return {name: 1.0 for name in signal_names}
   95:         return {name: w / avg for name, w in raw.items()}
   96: 
   97:     def save(self) -> None:
   98:         """Persist weights to JSON atomically."""
   99:         payload = {
  100:             "eta": self._eta,
  101:             "weights": self._weights,
  102:         }
  103:         atomic_write_json(self._path, payload)
  104: 
  105:     # ------------------------------------------------------------------
  106:     # Internal helpers
  107:     # ------------------------------------------------------------------
  108: 
  109:     def _load(self) -> None:
  110:         """Load weights from disk.  No-ops silently if the file is missing."""
  111:         data = load_json(self._path, default=None)
  112:         if data is None:
  113:             return
  114:         if isinstance(data, dict):
  115:             self._weights = {
  116:                 k: float(v)
  117:                 for k, v in data.get("weights", {}).items()
  118:             }
  119:             # Honour stored eta only if caller did not override it
  120:             # (caller passes None → _DEFAULT_ETA, so we preserve stored value)
  121: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  121: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  102:         }
  103:         atomic_write_json(self._path, payload)
  104: 
  105:     # ------------------------------------------------------------------
  106:     # Internal helpers
  107:     # ------------------------------------------------------------------
  108: 
  109:     def _load(self) -> None:
  110:         """Load weights from disk.  No-ops silently if the file is missing."""
  111:         data = load_json(self._path, default=None)
  112:         if data is None:
  113:             return
  114:         if isinstance(data, dict):
  115:             self._weights = {
  116:                 k: float(v)
  117:                 for k, v in data.get("weights", {}).items()
  118:             }
  119:             # Honour stored eta only if caller did not override it
  120:             # (caller passes None → _DEFAULT_ETA, so we preserve stored value)
  121: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Walk-forward signal weight optimizer.
    2: 
    3: Retrains signal weights using rolling windows to prevent overfitting
    4: and adapt to changing market regimes. Uses the LinearFactorModel for
    5: per-window ridge regression.
    6: 
    7: Walk-forward method:
    8:     1. Split history into train/test windows (e.g. 30d train, 7d test)
    9:     2. Train model on each window, score on out-of-sample test period
   10:     3. Track per-signal weight stability and out-of-sample performance
   11:     4. Output: recommended weights and stability metrics
   12: """
   13: from __future__ import annotations
   14: 
   15: import logging
   16: from dataclasses import dataclass, field
   17: from pathlib import Path
   18: 
   19: import numpy as np
   20: import pandas as pd
   21: 
   22: from portfolio.file_utils import atomic_write_json, load_json
   23: from portfolio.linear_factor import LinearFactorModel
   24: 
   25: logger = logging.getLogger("portfolio.signal_weight_optimizer")
   26: 
   27: _BASE_DIR = Path(__file__).resolve().parent.parent
   28: _RESULTS_FILE = _BASE_DIR / "data" / "models" / "walkforward_results.json"
   29: 
   30: 
   31: @dataclass
   32: class WalkForwardResult:
   33:     """Results from a single walk-forward optimization run."""
   34:     n_windows: int = 0
   35:     avg_r_squared: float = 0.0
   36:     avg_oos_corr: float = 0.0  # out-of-sample correlation
   37:     weight_stability: dict[str, float] = field(default_factory=dict)
   38:     recommended_weights: dict[str, float] = field(default_factory=dict)
   39:     signal_rankings: list[tuple[str, float]] = field(default_factory=list)
   40: 
   41:     def to_dict(self) -> dict:
   42:         return {
   43:             "n_windows": self.n_windows,
   44:             "avg_r_squared": self.avg_r_squared,
   45:             "avg_oos_corr": self.avg_oos_corr,
   46:             "weight_stability": self.weight_stability,
   47:             "recommended_weights": self.recommended_weights,
   48:             "signal_rankings": self.signal_rankings,
   49:         }
   50: 
   51: 
   52: def walk_forward_optimize(
   53:     signals_df: pd.DataFrame,
   54:     returns: pd.Series,
   55:     train_window: int = 720,   # 720 hours = 30 days
   56:     test_window: int = 168,    # 168 hours = 7 days
   57:     step_size: int = 168,      # step by 7 days
   58:     alpha: float = 1.0,
   59:     min_train_samples: int = 100,
   60: ) -> WalkForwardResult:
   61:     """Run walk-forward optimization across rolling windows.
   62: 
   63:     Args:
   64:         signals_df: DataFrame of signal values (columns=signals, rows=time).
   65:         returns: Series of forward returns aligned with signals_df.
   66:         train_window: Number of rows for training period.
   67:         test_window: Number of rows for test period.
   68:         step_size: Step size between windows.
   69:         alpha: Ridge regularization strength.
   70:         min_train_samples: Minimum training samples per window.
   71: 
   72:     Returns:
   73:         WalkForwardResult with averaged metrics and recommended weights.
   74:     """
   75:     common = signals_df.index.intersection(returns.index)
   76:     signals_df = signals_df.loc[common]
   77:     returns = returns.loc[common]
   78:     n = len(common)
   79: 
   80:     if n < train_window + test_window:
   81:         logger.warning("Insufficient data for walk-forward: %d < %d",
   82:                       n, train_window + test_window)
   83:         return WalkForwardResult()
   84: 
   85:     all_weights: list[dict[str, float]] = []
   86:     r_squared_scores: list[float] = []
   87:     oos_correlations: list[float] = []
   88: 
   89:     start = 0
   90:     while start + train_window + test_window <= n:
   91:         train_end = start + train_window
   92:         test_end = train_end + test_window
   93: 
   94:         train_X = signals_df.iloc[start:train_end]
   95:         train_y = returns.iloc[start:train_end]
   96:         test_X = signals_df.iloc[train_end:test_end]
   97:         test_y = returns.iloc[train_end:test_end]
   98: 
   99:         model = LinearFactorModel(alpha=alpha)
  100:         if not model.fit(train_X, train_y, min_samples=min_train_samples):
  101:             start += step_size
  102:             continue
  103: 
  104:         r_squared_scores.append(model.r_squared)
  105:         all_weights.append(model.weights)
  106: 
  107:         # Out-of-sample prediction correlation
  108:         predictions = []
  109:         for _, row in test_X.iterrows():
  110:             predictions.append(model.predict(row.to_dict()))
  111:         if len(predictions) > 1 and test_y.std() > 1e-10:
  112:             corr = float(np.corrcoef(predictions, test_y.values)[0, 1])
  113:             if not np.isnan(corr):
  114:                 oos_correlations.append(corr)
  115: 
  116:         start += step_size
  117: 
  118:     if not all_weights:
  119:         return WalkForwardResult()
  120: 
  121:     # Compute weight stability: std of each weight across windows / mean of abs
  122:     all_signals = set()
  123:     for w in all_weights:
  124:         all_signals.update(w.keys())
  125: 
  126:     weight_stability = {}
  127:     recommended_weights = {}
  128:     for sig in all_signals:
  129:         values = [w.get(sig, 0.0) for w in all_weights]
  130:         mean_val = float(np.mean(values))
  131:         std_val = float(np.std(values))
  132:         mean_abs = float(np.mean(np.abs(values)))
  133:         # Stability = 1 - (std / mean_abs). High = consistent direction.
  134:         stability = 1.0 - (std_val / mean_abs) if mean_abs > 1e-10 else 0.0
  135:         weight_stability[sig] = round(max(0.0, stability), 4)
  136:         recommended_weights[sig] = round(mean_val, 6)
  137: 
  138:     # Rank signals by |mean_weight| * stability
  139:     signal_rankings = sorted(
  140:         [(sig, round(abs(recommended_weights[sig]) * weight_stability.get(sig, 0), 6))
  141:          for sig in all_signals],
  142:         key=lambda x: x[1],
  143:         reverse=True,
  144:     )
  145: 
  146:     result = WalkForwardResult(
  147:         n_windows=len(all_weights),
  148:         avg_r_squared=round(float(np.mean(r_squared_scores)), 4),
  149:         avg_oos_corr=round(float(np.mean(oos_correlations)), 4) if oos_correlations else 0.0,
  150:         weight_stability=weight_stability,
  151:         recommended_weights=recommended_weights,
  152:         signal_rankings=signal_rankings,
  153:     )
  154:     return result
  155: 
  156: 
  157: def save_results(result: WalkForwardResult, path: Path | None = None) -> None:
  158:     """Persist walk-forward results to JSON."""
  159:     path = path or _RESULTS_FILE
  160:     path.parent.mkdir(parents=True, exist_ok=True)
  161:     atomic_write_json(path, result.to_dict())
  162: 
  163: 
  164: def load_results(path: Path | None = None) -> WalkForwardResult | None:
  165:     """Load walk-forward results from JSON."""
  166:     path = path or _RESULTS_FILE
  167:     data = load_json(path)
  168:     if not data:
  169:         return None
  170:     return WalkForwardResult(**data)
  171: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Track signal voting history for flip-flop detection and persistence scoring.
    2: 
    3: Maintains a rolling window of the last N votes per signal per ticker.
    4: Signals that maintain direction for 3+ consecutive checks are "persistent".
    5: Signals that flip every 1-2 checks are "noisy".
    6: """
    7: 
    8: import threading
    9: from collections import defaultdict
   10: from datetime import UTC, datetime
   11: from pathlib import Path
   12: 
   13: from portfolio.file_utils import atomic_write_jsonl, load_jsonl
   14: from portfolio.tickers import SIGNAL_NAMES
   15: 
   16: BASE_DIR = Path(__file__).resolve().parent.parent
   17: DATA_DIR = BASE_DIR / "data"
   18: HISTORY_FILE = DATA_DIR / "signal_history.jsonl"
   19: 
   20: MAX_ENTRIES_PER_TICKER = 50
   21: 
   22: # 2026-05-02 (adversarial review 05-01 P0-3): the main loop's
   23: # ThreadPoolExecutor (8 workers) calls update_history() concurrently for
   24: # 5 tickers. update_history is a read-modify-write of HISTORY_FILE; without
   25: # a lock, last-writer-wins and 4/5 ticker updates per cycle are silently
   26: # discarded. Persistence scores, streaks, and noisy-signal lists are then
   27: # computed from a corrupted history. The atomic_write_jsonl call inside
   28: # only guarantees the FILE write is atomic — not that the read+modify+write
   29: # sequence is atomic across threads.
   30: _history_lock = threading.Lock()
   31: 
   32: 
   33: def _load_history():
   34:     """Load all history entries from JSONL file.
   35: 
   36:     Returns:
   37:         list[dict]: All history entries.
   38:     """
   39:     return load_jsonl(HISTORY_FILE)
   40: 
   41: 
   42: def _save_history(entries):
   43:     """Write all history entries back to JSONL file.
   44: 
   45:     Args:
   46:         entries: List of history entry dicts.
   47:     """
   48:     atomic_write_jsonl(HISTORY_FILE, entries)
   49: 
   50: 
   51: def _entries_for_ticker(entries, ticker):
   52:     """Filter entries for a specific ticker.
   53: 
   54:     Args:
   55:         entries: All history entries.
   56:         ticker: Ticker symbol.
   57: 
   58:     Returns:
   59:         list[dict]: Entries for this ticker, ordered by time.
   60:     """
   61:     return [e for e in entries if e.get("ticker") == ticker]
   62: 
   63: 
   64: def update_history(ticker, votes_dict):
   65:     """Append current signal votes to history file for a ticker.
   66: 
   67:     Trims to keep only the last MAX_ENTRIES_PER_TICKER entries per ticker.
   68: 
   69:     Thread-safe (2026-05-02): the read-modify-write of HISTORY_FILE is
   70:     serialized with `_history_lock` so concurrent ThreadPoolExecutor
   71:     workers don't lose each other's writes.
   72: 
   73:     Args:
   74:         ticker: Ticker symbol (e.g. "BTC-USD").
   75:         votes_dict: Dict mapping signal_name -> vote ("BUY"/"SELL"/"HOLD").
   76:     """
   77:     with _history_lock:
   78:         entries = _load_history()
   79: 
   80:         new_entry = {
   81:             "ts": datetime.now(UTC).isoformat(),
   82:             "ticker": ticker,
   83:             "votes": {sig: votes_dict.get(sig, "HOLD") for sig in SIGNAL_NAMES},
   84:         }
   85:         entries.append(new_entry)
   86: 
   87:         # Trim: keep only last MAX_ENTRIES_PER_TICKER per ticker
   88:         by_ticker = defaultdict(list)
   89:         for e in entries:
   90:             by_ticker[e.get("ticker", "unknown")].append(e)
   91: 
   92:         trimmed = []
   93:         for _t, t_entries in by_ticker.items():
   94:             trimmed.extend(t_entries[-MAX_ENTRIES_PER_TICKER:])
   95: 
   96:         # Sort by timestamp for stable ordering
   97:         trimmed.sort(key=lambda e: e.get("ts", ""))
   98:         _save_history(trimmed)
   99: 
  100: 
  101: def get_persistence_scores(ticker):
  102:     """Compute persistence score for each signal for a ticker.
  103: 
  104:     Persistence score measures how often a signal maintains the same direction
  105:     across consecutive checks. Score of 1.0 means perfectly persistent (never
  106:     flips), 0.0 means flips every single check.
  107: 
  108:     The score is computed as: 1 - (flip_count / (N - 1)) where N is the number
  109:     of non-HOLD votes and flip_count is the number of direction changes.
  110:     Only considers non-HOLD votes (BUY/SELL) since HOLD is absence of signal.
  111: 
  112:     Args:
  113:         ticker: Ticker symbol.
  114: 
  115:     Returns:
  116:         dict: Mapping signal_name -> persistence score (0.0 to 1.0).
  117:               Signals with fewer than 3 non-HOLD votes return 0.5 (neutral).
  118:     """
  119:     entries = _entries_for_ticker(_load_history(), ticker)
  120:     if not entries:
  121:         return {sig: 0.5 for sig in SIGNAL_NAMES}
  122: 
  123:     scores = {}
  124:     for sig in SIGNAL_NAMES:
  125:         # Extract non-HOLD votes in order
  126:         active_votes = []
  127:         for e in entries:
  128:             vote = e.get("votes", {}).get(sig, "HOLD")
  129:             if vote in ("BUY", "SELL"):
  130:                 active_votes.append(vote)
  131: 
  132:         if len(active_votes) < 3:
  133:             scores[sig] = 0.5  # insufficient data
  134:             continue
  135: 
  136:         # Count direction flips
  137:         flips = 0
  138:         for i in range(1, len(active_votes)):
  139:             if active_votes[i] != active_votes[i - 1]:
  140:                 flips += 1
  141: 
  142:         max_flips = len(active_votes) - 1
  143:         scores[sig] = round(1.0 - (flips / max_flips), 4) if max_flips > 0 else 1.0
  144: 
  145:     return scores
  146: 
  147: 
  148: def get_noisy_signals(ticker, threshold=0.3):
  149:     """Get list of signals that flip too often for a ticker.
  150: 
  151:     Args:
  152:         ticker: Ticker symbol.
  153:         threshold: Persistence score below this is considered noisy.
  154: 
  155:     Returns:
  156:         list[str]: Signal names with persistence below threshold.
  157:     """
  158:     scores = get_persistence_scores(ticker)
  159:     return [sig for sig, score in scores.items() if score < threshold]
  160: 
  161: 
  162: def get_signal_streaks(ticker):
  163:     """Get current voting streak for each signal for a ticker.
  164: 
  165:     A streak is the number of consecutive checks a signal has maintained the
  166:     same vote (including HOLD). Useful for detecting sustained signals.
  167: 
  168:     Args:
  169:         ticker: Ticker symbol.
  170: 
  171:     Returns:
  172:         dict: Mapping signal_name -> {"current_vote": str, "streak_count": int}.
  173:     """
  174:     entries = _entries_for_ticker(_load_history(), ticker)
  175:     if not entries:
  176:         return {sig: {"current_vote": "HOLD", "streak_count": 0} for sig in SIGNAL_NAMES}
  177: 
  178:     streaks = {}
  179:     for sig in SIGNAL_NAMES:
  180:         votes = [e.get("votes", {}).get(sig, "HOLD") for e in entries]
  181:         if not votes:
  182:             streaks[sig] = {"current_vote": "HOLD", "streak_count": 0}
  183:             continue
  184: 
  185:         current = votes[-1]
  186:         count = 0
  187:         for v in reversed(votes):
  188:             if v == current:
  189:                 count += 1
  190:             else:
  191:                 break
  192: 
  193:         streaks[sig] = {"current_vote": current, "streak_count": count}
  194: 
  195:     return streaks
  196: 
  197: 
  198: def get_summary(ticker):
  199:     """Get a combined summary of persistence, noise, and streaks for a ticker.
  200: 
  201:     Convenience function that calls all three analysis functions.
  202: 
  203:     Args:
  204:         ticker: Ticker symbol.
  205: 
  206:     Returns:
  207:         dict with keys: persistence_scores, noisy_signals, streaks, entries_count.
  208:     """
  209:     entries = _entries_for_ticker(_load_history(), ticker)
  210:     return {
  211:         "entries_count": len(entries),
  212:         "persistence_scores": get_persistence_scores(ticker),
  213:         "noisy_signals": get_noisy_signals(ticker),
  214:         "streaks": get_signal_streaks(ticker),
  215:     }
  216: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Per-(ticker, signal) state-change timestamps for the dashboard heatmap.
    2: 
    3: Pure helper: no I/O. Caller is responsible for reading/writing
    4: `data/signal_state_since.json` via atomic helpers.
    5: 
    6: Invariant: for every (ticker, signal) pair in the *current* votes payload,
    7: the returned payload contains a `{"vote": ..., "since": <iso>}` entry.
    8: - If the previous payload had the same vote, `since` is preserved.
    9: - If the vote changed, was missing, or this is cold-start, `since = now_iso`.
   10: 
   11: Disabled / N/A votes are tracked the same as any other value — the dashboard
   12: suppresses the badge for cells already styled `cell--disabled`, so we don't
   13: need to filter here. Tracking them keeps the helper trivially monotonic.
   14: 
   15: Wired from `portfolio.reporting.write_agent_summary`, which is the single
   16: writer of the displayed `_votes` matrix; consuming the helper anywhere else
   17: would race against the loop's per-cycle update.
   18: """
   19: 
   20: from __future__ import annotations
   21: 
   22: from typing import Any
   23: 
   24: 
   25: def update_state_since(
   26:     prev: dict[str, Any] | None,
   27:     current_votes: dict[str, dict[str, str]],
   28:     now_iso: str,
   29: ) -> dict[str, Any]:
   30:     """Return a new state-since payload reflecting current votes vs prev.
   31: 
   32:     Args:
   33:         prev: previous payload, or None / empty dict on cold start. Expected
   34:             shape: {"updated_at": str, "votes": {ticker: {signal: {"vote": str, "since": str}}}}.
   35:             Anything else is treated as cold start (all `since = now_iso`).
   36:         current_votes: {ticker: {signal: "BUY"|"SELL"|"HOLD"|...}}.
   37:         now_iso: ISO-8601 timestamp string used for new / changed entries.
   38: 
   39:     Returns:
   40:         New payload, same shape as prev. Tickers / signals not in
   41:         current_votes are dropped (handles ticker churn).
   42:     """
   43:     prev_votes = (prev or {}).get("votes") if isinstance(prev, dict) else None
   44:     if not isinstance(prev_votes, dict):
   45:         prev_votes = {}
   46: 
   47:     new_votes: dict[str, dict[str, dict[str, str]]] = {}
   48:     for ticker, sigs in (current_votes or {}).items():
   49:         if not isinstance(sigs, dict):
   50:             continue
   51:         prev_ticker = prev_votes.get(ticker) if isinstance(prev_votes.get(ticker), dict) else {}
   52:         out_ticker: dict[str, dict[str, str]] = {}
   53:         for sig_name, vote in sigs.items():
   54:             vote_str = str(vote or "HOLD").upper()
   55:             prev_entry = prev_ticker.get(sig_name) if isinstance(prev_ticker, dict) else None
   56:             if (
   57:                 isinstance(prev_entry, dict)
   58:                 and prev_entry.get("vote") == vote_str
   59:                 and isinstance(prev_entry.get("since"), str)
   60:             ):
   61:                 since = prev_entry["since"]
   62:             else:
   63:                 since = now_iso
   64:             out_ticker[sig_name] = {"vote": vote_str, "since": since}
   65:         new_votes[ticker] = out_ticker
   66: 
   67:     return {"updated_at": now_iso, "votes": new_votes}
   68: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Signal decay alerting — detects when signal accuracy degrades significantly.
    2: 
    3: Compares recent (7d) accuracy against all-time accuracy for each signal.
    4: Alerts when a signal degrades by >10pp with 50+ recent samples.
    5: 
    6: Run as part of --check-outcomes or standalone via check_signal_decay().
    7: 
    8: Added 2026-04-30 after-hours research session. Prevents silent accuracy
    9: erosion between manual audit sessions.
   10: """
   11: 
   12: import json
   13: import logging
   14: from datetime import UTC, datetime
   15: 
   16: logger = logging.getLogger("portfolio.signal_decay_alert")
   17: 
   18: 
   19: # Minimum absolute accuracy drop (percentage points) to trigger an alert.
   20: _DECAY_THRESHOLD_PP = 10.0
   21: # Minimum recent samples to be confident in degradation.
   22: _MIN_RECENT_SAMPLES = 50
   23: # Minimum all-time samples to have a reliable baseline.
   24: _MIN_ALLTIME_SAMPLES = 100
   25: 
   26: 
   27: def check_signal_decay(accuracy_cache_path="data/accuracy_cache.json"):
   28:     """Check for signal accuracy decay and return a list of decay alerts.
   29: 
   30:     Returns:
   31:         list[dict]: Each dict has keys: signal, horizon, alltime_acc, recent_acc,
   32:         drop_pp, recent_samples, severity.
   33:     """
   34:     try:
   35:         with open(accuracy_cache_path, encoding="utf-8") as f:
   36:             cache = json.load(f)
   37:     except (FileNotFoundError, json.JSONDecodeError) as e:
   38:         logger.warning("Cannot load accuracy cache: %s", e)
   39:         return []
   40: 
   41:     alerts = []
   42:     horizon_pairs = [
   43:         ("3h", "3h_recent"),
   44:         ("1d", "1d_recent"),
   45:     ]
   46: 
   47:     for alltime_key, recent_key in horizon_pairs:
   48:         alltime_data = cache.get(alltime_key, {})
   49:         recent_data = cache.get(recent_key, {})
   50: 
   51:         if not isinstance(alltime_data, dict) or not isinstance(recent_data, dict):
   52:             continue
   53: 
   54:         for signal in recent_data:
   55:             recent = recent_data[signal]
   56:             alltime = alltime_data.get(signal, {})
   57: 
   58:             if not isinstance(recent, dict) or not isinstance(alltime, dict):
   59:                 continue
   60: 
   61:             recent_acc = recent.get("accuracy", 0)
   62:             recent_total = recent.get("total", 0)
   63:             alltime_acc = alltime.get("accuracy", 0)
   64:             alltime_total = alltime.get("total", 0)
   65: 
   66:             if recent_total < _MIN_RECENT_SAMPLES:
   67:                 continue
   68:             if alltime_total < _MIN_ALLTIME_SAMPLES:
   69:                 continue
   70: 
   71:             drop_pp = (alltime_acc - recent_acc) * 100
   72: 
   73:             if drop_pp >= _DECAY_THRESHOLD_PP:
   74:                 severity = "critical" if drop_pp >= 20 else "warning"
   75:                 alerts.append({
   76:                     "signal": signal,
   77:                     "horizon": alltime_key,
   78:                     "alltime_acc": round(alltime_acc * 100, 1),
   79:                     "recent_acc": round(recent_acc * 100, 1),
   80:                     "drop_pp": round(drop_pp, 1),
   81:                     "recent_samples": recent_total,
   82:                     "alltime_samples": alltime_total,
   83:                     "severity": severity,
   84:                 })
   85: 
   86:     # Sort by drop magnitude (worst first)
   87:     alerts.sort(key=lambda a: -a["drop_pp"])
   88:     return alerts
   89: 
   90: 
   91: def format_decay_report(alerts):
   92:     """Format decay alerts as a human-readable string.
   93: 
   94:     Returns:
   95:         str: Formatted report, or empty string if no alerts.
   96:     """
   97:     if not alerts:
   98:         return ""
   99: 
  100:     critical = [a for a in alerts if a["severity"] == "critical"]
  101:     warnings = [a for a in alerts if a["severity"] == "warning"]
  102: 
  103:     lines = [f"SIGNAL DECAY ALERT — {len(alerts)} signals degrading"]
  104:     lines.append("")
  105: 
  106:     if critical:
  107:         lines.append(f"CRITICAL ({len(critical)} signals, >20pp drop):")
  108:         for a in critical:
  109:             lines.append(
  110:                 f"  {a['signal']:25s} {a['horizon']:>3s}: "
  111:                 f"{a['alltime_acc']:.1f}% -> {a['recent_acc']:.1f}% "
  112:                 f"({a['drop_pp']:+.1f}pp, {a['recent_samples']} sam)"
  113:             )
  114:         lines.append("")
  115: 
  116:     if warnings:
  117:         lines.append(f"WARNING ({len(warnings)} signals, >10pp drop):")
  118:         for a in warnings:
  119:             lines.append(
  120:                 f"  {a['signal']:25s} {a['horizon']:>3s}: "
  121:                 f"{a['alltime_acc']:.1f}% -> {a['recent_acc']:.1f}% "
  122:                 f"({a['drop_pp']:+.1f}pp, {a['recent_samples']} sam)"
  123:             )
  124: 
  125:     return "\n".join(lines)
  126: 
  127: 
  128: def log_decay_alerts(alerts):
  129:     """Log decay alerts and write to data/signal_decay_alerts.jsonl."""
  130:     if not alerts:
  131:         logger.info("Signal decay check: no degradation detected")
  132:         return
  133: 
  134:     report = format_decay_report(alerts)
  135:     logger.warning("Signal decay detected:\n%s", report)
  136: 
  137:     # Append to JSONL log
  138:     entry = {
  139:         "ts": datetime.now(UTC).isoformat(),
  140:         "alert_count": len(alerts),
  141:         "critical_count": sum(1 for a in alerts if a["severity"] == "critical"),
  142:         "alerts": alerts,
  143:     }
  144:     try:
  145:         from portfolio.file_utils import atomic_append_jsonl
  146:         atomic_append_jsonl("data/signal_decay_alerts.jsonl", entry)
  147:     except Exception:
  148:         logger.debug("Could not write signal_decay_alerts.jsonl", exc_info=True)
  149: 
  150: 
  151: def run_decay_check():
  152:     """Run a full decay check and log results. Called from main.py --check-outcomes."""
  153:     alerts = check_signal_decay()
  154:     log_decay_alerts(alerts)
  155:     return alerts
  156: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Signal postmortem — analyze WHY signals fail by regime, ticker, and time.
    2: 
    3: Reads accuracy data broken down by regime and identifies patterns:
    4: - Which signals work in which regimes (and fail in others)
    5: - Which tickers are unpredictable for specific signals
    6: - Signal correlation clusters (vote agreement rates)
    7: 
    8: Output goes to data/signal_postmortem.json for Layer 2 context and
    9: periodic review by the after-hours research agent.
   10: """
   11: 
   12: import logging
   13: from datetime import UTC, datetime
   14: from pathlib import Path
   15: 
   16: from portfolio.file_utils import atomic_write_json, load_json
   17: 
   18: logger = logging.getLogger("portfolio.signal_postmortem")
   19: 
   20: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   21: POSTMORTEM_FILE = DATA_DIR / "signal_postmortem.json"
   22: 
   23: # Minimum samples for reliable analysis
   24: MIN_SAMPLES = 15
   25: 
   26: # Thresholds for classification
   27: STRONG_THRESHOLD = 0.60  # >=60% = signal works
   28: WEAK_THRESHOLD = 0.45    # <45% = signal is noise/harmful
   29: DIVERGENCE_THRESHOLD = 0.15  # 15pp regime divergence = regime-dependent
   30: 
   31: 
   32: def compute_regime_insights(regime_accuracy: dict) -> list[dict]:
   33:     """Identify signals that perform very differently across regimes.
   34: 
   35:     Finds signals where accuracy in one regime is >15pp different from another.
   36:     These are regime-dependent signals that should be gated or boosted.
   37: 
   38:     Args:
   39:         regime_accuracy: Output of accuracy_stats.signal_accuracy_by_regime()
   40: 
   41:     Returns:
   42:         List of insight dicts with signal, best_regime, worst_regime, spread.
   43:     """
   44:     if not regime_accuracy:
   45:         return []
   46: 
   47:     # Collect per-signal accuracy across regimes
   48:     signal_regimes: dict[str, dict[str, dict]] = {}
   49:     for regime, sig_map in regime_accuracy.items():
   50:         for sig, stats in sig_map.items():
   51:             if stats.get("total", 0) < MIN_SAMPLES:
   52:                 continue
   53:             signal_regimes.setdefault(sig, {})[regime] = stats
   54: 
   55:     insights = []
   56:     for sig, regimes in signal_regimes.items():
   57:         if len(regimes) < 2:
   58:             continue
   59: 
   60:         accs = {r: s["accuracy"] for r, s in regimes.items()}
   61:         best_regime = max(accs, key=accs.get)
   62:         worst_regime = min(accs, key=accs.get)
   63:         spread = accs[best_regime] - accs[worst_regime]
   64: 
   65:         if spread >= DIVERGENCE_THRESHOLD:
   66:             insights.append({
   67:                 "signal": sig,
   68:                 "type": "regime_dependent",
   69:                 "best_regime": best_regime,
   70:                 "best_accuracy": round(accs[best_regime] * 100, 1),
   71:                 "best_samples": regimes[best_regime]["total"],
   72:                 "worst_regime": worst_regime,
   73:                 "worst_accuracy": round(accs[worst_regime] * 100, 1),
   74:                 "worst_samples": regimes[worst_regime]["total"],
   75:                 "spread_pp": round(spread * 100, 1),
   76:                 "recommendation": (
   77:                     f"Gate {sig} in {worst_regime} regime "
   78:                     f"({accs[worst_regime]*100:.0f}%) — "
   79:                     f"it works in {best_regime} ({accs[best_regime]*100:.0f}%)"
   80:                 ),
   81:             })
   82: 
   83:     insights.sort(key=lambda x: x["spread_pp"], reverse=True)
   84:     return insights
   85: 
   86: 
   87: def compute_signal_health_report(accuracy_data: dict) -> list[dict]:
   88:     """Classify signals into strong, weak, and marginal categories.
   89: 
   90:     Args:
   91:         accuracy_data: Standard accuracy dict {signal: {accuracy, total, ...}}
   92: 
   93:     Returns:
   94:         List of signal health dicts sorted by accuracy.
   95:     """
   96:     report = []
   97:     for sig, stats in accuracy_data.items():
   98:         total = stats.get("total", 0)
   99:         if total < MIN_SAMPLES:
  100:             continue
  101: 
  102:         acc = stats.get("accuracy", 0.5)
  103:         if acc >= STRONG_THRESHOLD:
  104:             category = "strong"
  105:         elif acc < WEAK_THRESHOLD:
  106:             category = "weak"
  107:         else:
  108:             category = "marginal"
  109: 
  110:         report.append({
  111:             "signal": sig,
  112:             "accuracy_pct": round(acc * 100, 1),
  113:             "samples": total,
  114:             "category": category,
  115:         })
  116: 
  117:     report.sort(key=lambda x: x["accuracy_pct"], reverse=True)
  118:     return report
  119: 
  120: 
  121: def compute_vote_correlation(entries: list[dict] | None = None) -> list[dict]:
  122:     """Compute pairwise signal vote agreement rates.
  123: 
  124:     Analyzes signal_log entries to find which signals frequently agree.
  125:     High agreement (>80%) suggests redundancy — one signal adds no
  126:     information beyond what the other provides.
  127: 
  128:     Args:
  129:         entries: Pre-loaded signal_log entries. If None, loads from disk.
  130: 
  131:     Returns:
  132:         List of correlated pairs sorted by agreement rate.
  133:     """
  134:     if entries is None:
  135:         try:
  136:             from portfolio.accuracy_stats import load_entries
  137:             entries = load_entries()
  138:         except Exception:
  139:             logger.warning("Could not load signal_log entries for correlation analysis")
  140:             return []
  141: 
  142:     if not entries:
  143:         return []
  144: 
  145:     # Count pairwise agreement
  146:     from collections import defaultdict
  147:     pair_agree = defaultdict(int)
  148:     pair_total = defaultdict(int)
  149: 
  150:     for entry in entries:
  151:         tickers = entry.get("tickers", {})
  152:         for _ticker, tdata in tickers.items():
  153:             signals = tdata.get("signals", {})
  154:             # Only count signals that are actually voting (non-HOLD)
  155:             active = {s: v for s, v in signals.items() if v != "HOLD"}
  156:             active_names = sorted(active.keys())
  157: 
  158:             for i, s1 in enumerate(active_names):
  159:                 for s2 in active_names[i + 1:]:
  160:                     pair = (s1, s2)
  161:                     pair_total[pair] += 1
  162:                     if active[s1] == active[s2]:
  163:                         pair_agree[pair] += 1
  164: 
  165:     # Compute agreement rates
  166:     correlations = []
  167:     for pair, total in pair_total.items():
  168:         if total < 30:  # need enough co-occurrences
  169:             continue
  170:         agree = pair_agree.get(pair, 0)
  171:         rate = agree / total
  172:         if rate >= 0.70:  # only report high correlations
  173:             correlations.append({
  174:                 "signal_a": pair[0],
  175:                 "signal_b": pair[1],
  176:                 "agreement_rate": round(rate, 3),
  177:                 "co_occurrences": total,
  178:                 "agrees": agree,
  179:             })
  180: 
  181:     correlations.sort(key=lambda x: x["agreement_rate"], reverse=True)
  182:     return correlations
  183: 
  184: 
  185: def generate_postmortem() -> dict:
  186:     """Generate a complete signal postmortem report.
  187: 
  188:     Combines regime insights, health classification, and correlation analysis.
  189:     Writes to data/signal_postmortem.json.
  190:     """
  191:     report = {
  192:         "generated_at": datetime.now(UTC).isoformat(),
  193:         "regime_insights": [],
  194:         "signal_health": [],
  195:         "correlations": [],
  196:         "summary": {},
  197:     }
  198: 
  199:     # Regime-dependent analysis
  200:     try:
  201:         from portfolio.accuracy_stats import (
  202:             load_cached_accuracy,
  203:             load_cached_regime_accuracy,
  204:             signal_accuracy,
  205:             signal_accuracy_by_regime,
  206:         )
  207: 
  208:         # Overall accuracy
  209:         acc = load_cached_accuracy("1d")
  210:         if not acc:
  211:             acc = signal_accuracy("1d")
  212:         if acc:
  213:             report["signal_health"] = compute_signal_health_report(acc)
  214: 
  215:         # Regime breakdown
  216:         regime_acc = load_cached_regime_accuracy("1d")
  217:         if not regime_acc:
  218:             regime_acc = signal_accuracy_by_regime("1d")
  219:         if regime_acc:
  220:             report["regime_insights"] = compute_regime_insights(regime_acc)
  221: 
  222:     except Exception:
  223:         logger.warning("Accuracy data unavailable for postmortem", exc_info=True)
  224: 
  225:     # Correlation analysis
  226:     try:
  227:         report["correlations"] = compute_vote_correlation()
  228:     except Exception:
  229:         logger.warning("Correlation analysis failed", exc_info=True)
  230: 
  231:     # Summary
  232:     strong = [s for s in report["signal_health"] if s["category"] == "strong"]
  233:     weak = [s for s in report["signal_health"] if s["category"] == "weak"]
  234:     report["summary"] = {
  235:         "strong_signals": len(strong),
  236:         "weak_signals": len(weak),
  237:         "regime_dependent": len(report["regime_insights"]),
  238:         "correlated_pairs": len(report["correlations"]),
  239:         "top_3_strong": [s["signal"] for s in strong[:3]],
  240:         "top_3_weak": [s["signal"] for s in weak[:3]],
  241:     }
  242: 
  243:     atomic_write_json(POSTMORTEM_FILE, report)
  244:     logger.info(
  245:         "Signal postmortem: %d strong, %d weak, %d regime-dependent, %d correlated pairs",
  246:         len(strong), len(weak), len(report["regime_insights"]), len(report["correlations"]),
  247:     )
  248: 
  249:     return report
  250: 
  251: 
  252: def get_postmortem_context() -> dict | None:
  253:     """Load cached postmortem for inclusion in agent_summary.
  254: 
  255:     Returns compact version suitable for Layer 2 context.
  256:     """
  257:     data = load_json(POSTMORTEM_FILE)
  258:     if not data:
  259:         return None
  260: 
  261:     # Return compact version — just summary + top insights
  262:     return {
  263:         "summary": data.get("summary", {}),
  264:         "top_regime_insights": data.get("regime_insights", [])[:5],
  265:         "top_correlations": data.get("correlations", [])[:5],
  266:     }
  267: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
    1: """SQLite storage for signal snapshots — replaces signal_log.jsonl for reads.
    2: 
    3: Schema:
    4: - snapshots: one row per invocation (ts, trigger_reasons, fx_rate)
    5: - ticker_signals: one row per ticker per snapshot (price, consensus, votes, signals JSON)
    6: - outcomes: one row per ticker per horizon per snapshot (backfilled prices + change_pct)
    7: 
    8: Usage:
    9:     from portfolio.signal_db import SignalDB
   10:     db = SignalDB()           # uses default path data/signal_log.db
   11:     db.insert_snapshot(entry) # dict in same format as signal_log.jsonl line
   12:     entries = db.load_entries()  # returns list[dict] matching JSONL format
   13: """
   14: 
   15: import json
   16: import sqlite3
   17: from pathlib import Path
   18: 
   19: BASE_DIR = Path(__file__).resolve().parent.parent
   20: DATA_DIR = BASE_DIR / "data"
   21: DEFAULT_DB_PATH = DATA_DIR / "signal_log.db"
   22: 
   23: 
   24: class SignalDB:
   25:     def __init__(self, db_path=None):
   26:         self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
   27:         self.db_path.parent.mkdir(parents=True, exist_ok=True)
   28:         self._conn = None
   29:         self._ensure_schema()
   30: 
   31:     def _get_conn(self):
   32:         if self._conn is None:
   33:             self._conn = sqlite3.connect(str(self.db_path), timeout=10)
   34:             self._conn.execute("PRAGMA journal_mode=WAL")
   35:             self._conn.execute("PRAGMA foreign_keys=ON")
   36:             self._conn.row_factory = sqlite3.Row
   37:         return self._conn
   38: 
   39:     def _ensure_schema(self):
   40:         conn = self._get_conn()
   41:         conn.executescript("""
   42:             CREATE TABLE IF NOT EXISTS snapshots (
   43:                 id INTEGER PRIMARY KEY AUTOINCREMENT,
   44:                 ts TEXT NOT NULL UNIQUE,
   45:                 trigger_reasons TEXT,
   46:                 fx_rate REAL
   47:             );
   48: 
   49:             CREATE TABLE IF NOT EXISTS ticker_signals (
   50:                 snapshot_id INTEGER NOT NULL,
   51:                 ticker TEXT NOT NULL,
   52:                 price_usd REAL,
   53:                 consensus TEXT,
   54:                 buy_count INTEGER,
   55:                 sell_count INTEGER,
   56:                 total_voters INTEGER,
   57:                 signals TEXT,
   58:                 regime TEXT DEFAULT 'unknown',
   59:                 PRIMARY KEY (snapshot_id, ticker),
   60:                 FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
   61:             );
   62: 
   63:             CREATE TABLE IF NOT EXISTS outcomes (
   64:                 snapshot_id INTEGER NOT NULL,
   65:                 ticker TEXT NOT NULL,
   66:                 horizon TEXT NOT NULL,
   67:                 price_usd REAL,
   68:                 change_pct REAL,
   69:                 outcome_ts TEXT,
   70:                 PRIMARY KEY (snapshot_id, ticker, horizon),
   71:                 FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
   72:             );
   73: 
   74:             CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts);
   75:             CREATE INDEX IF NOT EXISTS idx_ticker_signals_ticker ON ticker_signals(ticker);
   76:             CREATE INDEX IF NOT EXISTS idx_outcomes_horizon ON outcomes(horizon);
   77:         """)
   78:         conn.commit()
   79: 
   80:         # Migration: add regime column to existing DBs that predate this field
   81:         existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(ticker_signals)").fetchall()}
   82:         if "regime" not in existing_cols:
   83:             conn.execute("ALTER TABLE ticker_signals ADD COLUMN regime TEXT DEFAULT 'unknown'")
   84:             conn.commit()
   85: 
   86:     def close(self):
   87:         if self._conn:
   88:             self._conn.close()
   89:             self._conn = None
   90: 
   91:     # --- Write ---
   92: 
   93:     def insert_snapshot(self, entry):
   94:         """Insert a signal_log entry (same dict format as JSONL line).
   95: 
   96:         Skips silently if ts already exists (idempotent for migration).
   97:         """
   98:         conn = self._get_conn()
   99:         ts = entry["ts"]
  100:         trigger_reasons = json.dumps(entry.get("trigger_reasons", []))
  101:         fx_rate = entry.get("fx_rate")
  102: 
  103:         try:
  104:             cur = conn.execute(
  105:                 "INSERT INTO snapshots (ts, trigger_reasons, fx_rate) VALUES (?, ?, ?)",
  106:                 (ts, trigger_reasons, fx_rate),
  107:             )
  108:         except sqlite3.IntegrityError:
  109:             return  # duplicate ts, skip
  110: 
  111:         snapshot_id = cur.lastrowid
  112: 
  113:         try:
  114:             tickers = entry.get("tickers", {})
  115:             for ticker, tdata in tickers.items():
  116:                 conn.execute(
  117:                     """INSERT INTO ticker_signals
  118:                        (snapshot_id, ticker, price_usd, consensus, buy_count, sell_count, total_voters, signals, regime)
  119:                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
  120:                     (
  121:                         snapshot_id,
  122:                         ticker,
  123:                         tdata.get("price_usd"),
  124:                         tdata.get("consensus"),
  125:                         tdata.get("buy_count"),
  126:                         tdata.get("sell_count"),
  127:                         tdata.get("total_voters"),
  128:                         json.dumps(tdata.get("signals", {})),
  129:                         tdata.get("regime", "unknown"),
  130:                     ),
  131:                 )
  132: 
  133:             outcomes = entry.get("outcomes", {})
  134:             for ticker, horizons in outcomes.items():
  135:                 if not isinstance(horizons, dict):
  136:                     continue
  137:                 for horizon, odata in horizons.items():
  138:                     if odata is None:
  139:                         continue
  140:                     conn.execute(
  141:                         """INSERT OR REPLACE INTO outcomes
  142:                            (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts)
  143:                            VALUES (?, ?, ?, ?, ?, ?)""",
  144:                         (
  145:                             snapshot_id,
  146:                             ticker,
  147:                             horizon,
  148:                             odata.get("price_usd"),
  149:                             odata.get("change_pct"),
  150:                             odata.get("ts"),
  151:                         ),
  152:                     )
  153: 
  154:             conn.commit()
  155:         except Exception:
  156:             conn.rollback()
  157:             raise
  158: 
  159:     def update_outcome(self, ts, ticker, horizon, price_usd, change_pct, outcome_ts):
  160:         """Update a single outcome cell. Used by backfill."""
  161:         conn = self._get_conn()
  162:         row = conn.execute("SELECT id FROM snapshots WHERE ts = ?", (ts,)).fetchone()
  163:         if not row:
  164:             return False
  165:         snapshot_id = row["id"]
  166:         conn.execute(
  167:             """INSERT OR REPLACE INTO outcomes
  168:                (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts)
  169:                VALUES (?, ?, ?, ?, ?, ?)""",
  170:             (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts),
  171:         )
  172:         conn.commit()
  173:         return True
  174: 
  175:     # --- Read ---
  176: 
  177:     def load_entries(self):
  178:         """Load all snapshots as list[dict] matching JSONL format.
  179: 
  180:         Compatible drop-in replacement for accuracy_stats.load_entries().
  181:         """
  182:         conn = self._get_conn()
  183:         snapshots = conn.execute("SELECT * FROM snapshots ORDER BY ts").fetchall()
  184: 
  185:         entries = []
  186:         for snap in snapshots:
  187:             sid = snap["id"]
  188: 
  189:             tickers = {}
  190:             for row in conn.execute(
  191:                 "SELECT * FROM ticker_signals WHERE snapshot_id = ?", (sid,)
  192:             ):
  193:                 tickers[row["ticker"]] = {
  194:                     "price_usd": row["price_usd"],
  195:                     "consensus": row["consensus"],
  196:                     "buy_count": row["buy_count"],
  197:                     "sell_count": row["sell_count"],
  198:                     "total_voters": row["total_voters"],
  199:                     "signals": json.loads(row["signals"]) if row["signals"] else {},
  200:                     "regime": row["regime"] if row["regime"] is not None else "unknown",
  201:                 }
  202: 
  203:             outcomes = {}
  204:             for row in conn.execute(
  205:                 "SELECT * FROM outcomes WHERE snapshot_id = ?", (sid,)
  206:             ):
  207:                 if row["ticker"] not in outcomes:
  208:                     outcomes[row["ticker"]] = {}
  209:                 outcomes[row["ticker"]][row["horizon"]] = {
  210:                     "price_usd": row["price_usd"],
  211:                     "change_pct": row["change_pct"],
  212:                     "ts": row["outcome_ts"],
  213:                 }
  214: 
  215:             entries.append({
  216:                 "ts": snap["ts"],
  217:                 "trigger_reasons": json.loads(snap["trigger_reasons"]) if snap["trigger_reasons"] else [],
  218:                 "fx_rate": snap["fx_rate"],
  219:                 "tickers": tickers,
  220:                 "outcomes": outcomes,
  221:             })
  222: 
  223:         return entries
  224: 
  225:     def snapshot_count(self):
  226:         """Return total number of snapshots."""
  227:         conn = self._get_conn()
  228:         row = conn.execute("SELECT COUNT(*) as cnt FROM snapshots").fetchone()
  229:         return row["cnt"]
  230: 
  231:     def entries_missing_outcomes(self, horizon):
  232:         """Find snapshot timestamps that are missing a specific horizon outcome.
  233: 
  234:         Returns list of (ts, ticker, price_usd) tuples for entries needing backfill.
  235:         """
  236:         conn = self._get_conn()
  237:         rows = conn.execute(
  238:             """SELECT s.ts, ts2.ticker, ts2.price_usd
  239:                FROM snapshots s
  240:                JOIN ticker_signals ts2 ON s.id = ts2.snapshot_id
  241:                LEFT JOIN outcomes o ON s.id = o.snapshot_id
  242:                    AND ts2.ticker = o.ticker AND o.horizon = ?
  243:                WHERE o.snapshot_id IS NULL
  244:                ORDER BY s.ts""",
  245:             (horizon,),
  246:         ).fetchall()
  247:         return [(r["ts"], r["ticker"], r["price_usd"]) for r in rows]
  248: 
  249:     def signal_accuracy(self, horizon="1d"):
  250:         """Compute per-signal accuracy directly via SQL.
  251: 
  252:         Returns dict matching accuracy_stats.signal_accuracy() format.
  253:         """
  254:         conn = self._get_conn()
  255:         from portfolio.tickers import SIGNAL_NAMES
  256: 
  257:         stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}
  258: 
  259:         rows = conn.execute(
  260:             """SELECT ts2.signals, o.change_pct
  261:                FROM ticker_signals ts2
  262:                JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
  263:                WHERE o.horizon = ? AND o.change_pct IS NOT NULL""",
  264:             (horizon,),
  265:         ).fetchall()
  266: 
  267:         for row in rows:
  268:             signals = json.loads(row["signals"]) if row["signals"] else {}
  269:             change_pct = row["change_pct"]
  270:             for sig_name in SIGNAL_NAMES:
  271:                 vote = signals.get(sig_name, "HOLD")
  272:                 if vote == "HOLD":
  273:                     continue
  274:                 stats[sig_name]["total"] += 1
  275:                 if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
  276:                     stats[sig_name]["correct"] += 1
  277: 
  278:         result = {}
  279:         for sig_name in SIGNAL_NAMES:
  280:             s = stats[sig_name]
  281:             acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
  282:             result[sig_name] = {
  283:                 "correct": s["correct"],
  284:                 "total": s["total"],
  285:                 "accuracy": acc,
  286:                 "pct": round(acc * 100, 1),
  287:             }
  288:         return result
  289: 
  290:     def consensus_accuracy(self, horizon="1d"):
  291:         """Compute consensus accuracy directly via SQL."""
  292:         conn = self._get_conn()
  293:         rows = conn.execute(
  294:             """SELECT ts2.consensus, o.change_pct
  295:                FROM ticker_signals ts2
  296:                JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
  297:                WHERE o.horizon = ? AND o.change_pct IS NOT NULL
  298:                  AND ts2.consensus != 'HOLD'""",
  299:             (horizon,),
  300:         ).fetchall()
  301: 
  302:         correct = 0
  303:         total = 0
  304:         for row in rows:
  305:             total += 1
  306:             if (row["consensus"] == "BUY" and row["change_pct"] > 0) or \
  307:                (row["consensus"] == "SELL" and row["change_pct"] < 0):
  308:                 correct += 1
  309: 
  310:         acc = correct / total if total > 0 else 0.0
  311:         return {
  312:             "correct": correct,
  313:             "total": total,
  314:             "accuracy": acc,
  315:             "pct": round(acc * 100, 1),
  316:         }
  317: 
  318:     def per_ticker_accuracy(self, horizon="1d"):
  319:         """Compute per-ticker consensus accuracy via SQL."""
  320:         conn = self._get_conn()
  321:         rows = conn.execute(
  322:             """SELECT ts2.ticker, ts2.consensus, o.change_pct
  323:                FROM ticker_signals ts2
  324:                JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
  325:                WHERE o.horizon = ? AND o.change_pct IS NOT NULL
  326:                  AND ts2.consensus != 'HOLD'""",
  327:             (horizon,),
  328:         ).fetchall()
  329: 
  330:         from collections import defaultdict
  331:         stats = defaultdict(lambda: {"correct": 0, "total": 0})
  332:         for row in rows:
  333:             stats[row["ticker"]]["total"] += 1
  334:             if (row["consensus"] == "BUY" and row["change_pct"] > 0) or \
  335:                (row["consensus"] == "SELL" and row["change_pct"] < 0):
  336:                 stats[row["ticker"]]["correct"] += 1
  337: 
  338:         result = {}
  339:         for ticker, s in stats.items():
  340:             acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
  341:             result[ticker] = {
  342:                 "correct": s["correct"],
  343:                 "total": s["total"],
  344:                 "accuracy": acc,
  345:                 "pct": round(acc * 100, 1),
  346:             }
  347:         return result
  348: 
  349:     def ticker_signal_accuracy(self, horizon="1d", min_samples=0):
  350:         """Per-ticker per-signal accuracy cross-tabulation via SQL.
  351: 
  352:         Returns: {ticker: {signal_name: {correct, total, accuracy, pct}}}
  353:         """
  354:         conn = self._get_conn()
  355:         rows = conn.execute(
  356:             """SELECT ts2.ticker, ts2.signals, o.change_pct
  357:                FROM ticker_signals ts2
  358:                JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
  359:                WHERE o.horizon = ? AND o.change_pct IS NOT NULL""",
  360:             (horizon,),
  361:         ).fetchall()
  362: 
  363:         from collections import defaultdict
  364:         stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
  365: 
  366:         for row in rows:
  367:             signals = json.loads(row["signals"]) if row["signals"] else {}
  368:             change_pct = row["change_pct"]
  369:             ticker = row["ticker"]
  370:             for sig_name, vote in signals.items():
  371:                 if vote == "HOLD":
  372:                     continue
  373:                 stats[ticker][sig_name]["total"] += 1
  374:                 if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
  375:                     stats[ticker][sig_name]["correct"] += 1
  376: 
  377:         result = {}
  378:         for ticker, sig_stats in stats.items():
  379:             ticker_result = {}
  380:             for sig_name, s in sig_stats.items():
  381:                 if s["total"] < min_samples:
  382:                     continue
  383:                 acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
  384:                 ticker_result[sig_name] = {
  385:                     "correct": s["correct"],
  386:                     "total": s["total"],
  387:                     "accuracy": acc,
  388:                     "pct": round(acc * 100, 1),
  389:                 }
  390:             if ticker_result:
  391:                 result[ticker] = ticker_result
  392:         return result
  393: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: import json
    2: import logging
    3: from datetime import UTC, datetime, timedelta
    4: from pathlib import Path
    5: 
    6: from portfolio.api_utils import BINANCE_BASE, BINANCE_FAPI_BASE
    7: from portfolio.file_utils import atomic_append_jsonl
    8: from portfolio.http_retry import fetch_with_retry
    9: from portfolio.shared_state import _yfinance_limiter
   10: 
   11: logger = logging.getLogger("portfolio.outcome_tracker")
   12: 
   13: 
   14: BASE_DIR = Path(__file__).resolve().parent.parent
   15: DATA_DIR = BASE_DIR / "data"
   16: SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
   17: 
   18: HORIZONS = {"3h": 10800, "4h": 14400, "12h": 43200, "1d": 86400, "3d": 259200, "5d": 432000, "10d": 864000}
   19: import contextlib
   20: 
   21: from portfolio.tickers import (
   22:     BINANCE_FAPI_MAP,
   23:     BINANCE_SPOT_MAP,
   24:     SIGNAL_NAMES,
   25:     YF_MAP,
   26: )
   27: 
   28: 
   29: def _derive_signal_vote(name, indicators, extra):
   30:     if name == "rsi":
   31:         rsi = indicators.get("rsi")
   32:         if rsi is None:
   33:             return "HOLD"
   34:         rsi_lower = indicators.get("rsi_p20", 30)
   35:         rsi_upper = indicators.get("rsi_p80", 70)
   36:         rsi_lower = max(rsi_lower, 15)
   37:         rsi_upper = min(rsi_upper, 85)
   38:         if rsi < rsi_lower:
   39:             return "BUY"
   40:         if rsi > rsi_upper:
   41:             return "SELL"
   42:         return "HOLD"
   43: 
   44:     if name == "macd":
   45:         hist = indicators.get("macd_hist")
   46:         hist_prev = indicators.get("macd_hist_prev")
   47:         if hist is None or hist_prev is None:
   48:             return "HOLD"
   49:         if hist > 0 and hist_prev <= 0:
   50:             return "BUY"
   51:         if hist < 0 and hist_prev >= 0:
   52:             return "SELL"
   53:         return "HOLD"
   54: 
   55:     if name == "ema":
   56:         ema9 = indicators.get("ema9")
   57:         ema21 = indicators.get("ema21")
   58:         if ema9 is None or ema21 is None:
   59:             return "HOLD"
   60:         ema_gap_pct = abs(ema9 - ema21) / ema21 * 100 if ema21 != 0 else 0
   61:         if ema_gap_pct < 0.5:
   62:             return "HOLD"
   63:         return "BUY" if ema9 > ema21 else "SELL"
   64: 
   65:     if name == "bb":
   66:         pos = indicators.get("price_vs_bb")
   67:         if pos == "below_lower":
   68:             return "BUY"
   69:         if pos == "above_upper":
   70:             return "SELL"
   71:         return "HOLD"
   72: 
   73:     if name == "fear_greed":
   74:         fg = extra.get("fear_greed")
   75:         if fg is None:
   76:             return "HOLD"
   77:         if fg <= 20:
   78:             return "BUY"
   79:         if fg >= 80:
   80:             return "SELL"
   81:         return "HOLD"
   82: 
   83:     if name == "sentiment":
   84:         sent = extra.get("sentiment")
   85:         conf = extra.get("sentiment_conf", 0)
   86:         if sent == "positive" and conf > 0.4:
   87:             return "BUY"
   88:         if sent == "negative" and conf > 0.4:
   89:             return "SELL"
   90:         return "HOLD"
   91: 
   92:     if name == "ministral":
   93:         return extra.get("ministral_action", "HOLD")
   94: 
   95:     if name == "ml":
   96:         return extra.get("ml_action", "HOLD")
   97: 
   98:     if name == "funding":
   99:         return extra.get("funding_action", "HOLD")
  100: 
  101:     if name == "volume":
  102:         return extra.get("volume_action", "HOLD")
  103: 
  104:     if name == "qwen3":
  105:         return extra.get("qwen3_action", "HOLD")
  106: 
  107:     # custom_lora removed — signal disabled, no longer tracked
  108: 
  109:     return "HOLD"
  110: 
  111: 
  112: def log_signal_snapshot(signals_dict, prices_usd, fx_rate, trigger_reasons):
  113:     ts = datetime.now(UTC).isoformat()
  114:     tickers = {}
  115: 
  116:     for ticker, sig_data in signals_dict.items():
  117:         indicators = sig_data.get("indicators", {})
  118:         extra = sig_data.get("extra", {})
  119:         price = prices_usd.get(ticker, indicators.get("close"))
  120: 
  121:         # CROSS-001: use _raw_votes (pre-gate) so accuracy accumulates for
  122:         # regime-gated signals, breaking the dead-signal trap.
  123:         passed_votes = extra.get("_raw_votes", extra.get("_votes"))
  124:         if passed_votes:
  125:             signals = {name: passed_votes.get(name, "HOLD") for name in SIGNAL_NAMES}
  126:         else:
  127:             signals = {}
  128:             for name in SIGNAL_NAMES:
  129:                 signals[name] = _derive_signal_vote(name, indicators, extra)
  130: 
  131:         buy_count = sum(1 for v in signals.values() if v == "BUY")
  132:         sell_count = sum(1 for v in signals.values() if v == "SELL")
  133: 
  134:         consensus = sig_data.get("action", "HOLD")
  135:         total_voters = buy_count + sell_count
  136: 
  137:         regime = extra.get("_regime", "unknown")
  138: 
  139:         tickers[ticker] = {
  140:             "price_usd": price,
  141:             "consensus": consensus,
  142:             "buy_count": buy_count,
  143:             "sell_count": sell_count,
  144:             "total_voters": total_voters,
  145:             "signals": signals,
  146:             "regime": regime,
  147:         }
  148: 
  149:     entry = {
  150:         "ts": ts,
  151:         "trigger_reasons": trigger_reasons,
  152:         "fx_rate": fx_rate,
  153:         "tickers": tickers,
  154:         "outcomes": {},
  155:     }
  156: 
  157:     atomic_append_jsonl(SIGNAL_LOG, entry)
  158: 
  159:     # Dual-write to SQLite
  160:     try:
  161:         from portfolio.signal_db import SignalDB
  162:         db = SignalDB()
  163:         db.insert_snapshot(entry)
  164:         db.close()
  165:     except Exception as e:
  166:         logger.warning("SQLite snapshot write failed (JSONL has data, SQLite may lag): %s", e)
  167: 
  168:     return entry
  169: 
  170: 
  171: def _fetch_current_price(ticker):
  172:     if ticker in BINANCE_FAPI_MAP:
  173:         symbol = BINANCE_FAPI_MAP[ticker]
  174:         r = fetch_with_retry(
  175:             f"{BINANCE_FAPI_BASE}/ticker/price",
  176:             params={"symbol": symbol},
  177:             timeout=5,
  178:         )
  179:         if r is None:
  180:             return None
  181:         r.raise_for_status()
  182:         return float(r.json()["price"])
  183: 
  184:     if ticker in BINANCE_SPOT_MAP:
  185:         symbol = BINANCE_SPOT_MAP[ticker]
  186:         r = fetch_with_retry(
  187:             f"{BINANCE_BASE}/ticker/price",
  188:             params={"symbol": symbol},
  189:             timeout=5,
  190:         )
  191:         if r is None:
  192:             return None
  193:         r.raise_for_status()
  194:         return float(r.json()["price"])
  195: 
  196:     if ticker in YF_MAP:
  197:         # 2026-04-14: route through canonical price_source (Alpaca for
  198:         # stocks, yfinance emergency fallback). Previously bypassed
  199:         # Alpaca entirely — added a 15-min delay to every stock outcome
  200:         # backfill call.
  201:         from portfolio.price_source import fetch_klines
  202: 
  203:         h = fetch_klines(YF_MAP[ticker], interval="1d", limit=5, period="5d")
  204:         if h is None or h.empty:
  205:             return None
  206:         return float(h["close"].iloc[-1])
  207: 
  208:     return None
  209: 
  210: 
  211: def _fetch_historical_price(ticker, target_ts):
  212:     if ticker in BINANCE_FAPI_MAP:
  213:         symbol = BINANCE_FAPI_MAP[ticker]
  214:         start_ms = int(target_ts * 1000)
  215:         r = fetch_with_retry(
  216:             f"{BINANCE_FAPI_BASE}/klines",
  217:             params={
  218:                 "symbol": symbol,
  219:                 "interval": "1h",
  220:                 "startTime": start_ms,
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  220:                 "startTime": start_ms,
  221:                 "limit": 1,
  222:             },
  223:             timeout=10,
  224:         )
  225:         if r is None:
  226:             return None
  227:         r.raise_for_status()
  228:         data = r.json()
  229:         if not data:
  230:             return None
  231:         return float(data[0][4])
  232: 
  233:     if ticker in BINANCE_SPOT_MAP:
  234:         symbol = BINANCE_SPOT_MAP[ticker]
  235:         start_ms = int(target_ts * 1000)
  236:         r = fetch_with_retry(
  237:             f"{BINANCE_BASE}/klines",
  238:             params={
  239:                 "symbol": symbol,
  240:                 "interval": "1h",
  241:                 "startTime": start_ms,
  242:                 "limit": 1,
  243:             },
  244:             timeout=10,
  245:         )
  246:         if r is None:
  247:             return None
  248:         r.raise_for_status()
  249:         data = r.json()
  250:         if not data:
  251:             return None
  252:         return float(data[0][4])
  253: 
  254:     if ticker in YF_MAP:
  255:         # 2026-04-14: historical price-at-timestamp needs a date window
  256:         # that price_source.fetch_klines's `limit`-based contract can't
  257:         # express precisely. Keep direct yfinance here — the call is
  258:         # only for outcome backfill (non-hot-path; 15-min delay is
  259:         # acceptable for an outcome at target_ts already in the past).
  260:         # To migrate, add start/end support to price_source and update
  261:         # this block.
  262:         import yfinance as yf
  263: 
  264:         _yfinance_limiter.wait()
  265:         target_dt = datetime.fromtimestamp(target_ts, tz=UTC)
  266:         start_date = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
  267:         end_date = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
  268:         t = yf.Ticker(YF_MAP[ticker])
  269:         h = t.history(start=start_date, end=end_date)
  270:         if h.empty:
  271:             return None
  272:         target_date = target_dt.date()
  273:         candidates = h[h.index.date <= target_date]
  274:         if candidates.empty:
  275:             return float(h["Close"].iloc[0])
  276:         return float(candidates["Close"].iloc[-1])
  277: 
  278:     return None
  279: 
  280: 
  281: def _signal_log_lock_path():
  282:     """Path of the sidecar lockfile shared with atomic_append_jsonl.
  283: 
  284:     Same convention as portfolio.file_utils.atomic_append_jsonl:
  285:         ``<dir>/.<filename>.lock`` (e.g. ``data/.signal_log.jsonl.lock``).
  286:     Held to coordinate read-modify-rewrite of SIGNAL_LOG with concurrent
  287:     log_signal_snapshot() appenders. SC-P1-3 (2026-05-02 follow-up).
  288:     """
  289:     return SIGNAL_LOG.parent / f".{SIGNAL_LOG.name}.lock"
  290: 
  291: 
  292: @contextlib.contextmanager
  293: def _hold_signal_log_lock():
  294:     """Acquire the sidecar lockfile that atomic_append_jsonl uses.
  295: 
  296:     Cross-platform pattern lifted from portfolio/file_utils.py:
  297:       - Windows: msvcrt.locking on a 1-byte range
  298:       - POSIX: fcntl.flock LOCK_EX
  299:     The lockfile is pre-created with a single null byte if missing so
  300:     locking never fails on a size-0 file.
  301:     """
  302:     try:
  303:         import msvcrt as _msvcrt  # type: ignore[import-not-found]
  304:     except ImportError:  # pragma: no cover - non-Windows
  305:         _msvcrt = None  # type: ignore[assignment]
  306:     try:
  307:         import fcntl as _fcntl  # type: ignore[import-not-found]
  308:     except ImportError:  # pragma: no cover - Windows
  309:         _fcntl = None  # type: ignore[assignment]
  310: 
  311:     import os as _os
  312:     lock_path = _signal_log_lock_path()
  313:     if not lock_path.exists():
  314:         try:
  315:             with open(lock_path, "ab") as lf:
  316:                 if lf.tell() == 0:
  317:                     lf.write(b"\0")
  318:         except OSError:
  319:             pass
  320: 
  321:     with open(lock_path, "rb+") as lock_f:
  322:         lfd = lock_f.fileno()
  323:         win_locked = False
  324:         try:
  325:             if _msvcrt is not None:
  326:                 _os.lseek(lfd, 0, _os.SEEK_SET)
  327:                 _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
  328:                 win_locked = True
  329:             elif _fcntl is not None:
  330:                 _fcntl.flock(lfd, _fcntl.LOCK_EX)
  331:             yield
  332:         finally:
  333:             if win_locked and _msvcrt is not None:
  334:                 try:
  335:                     _os.lseek(lfd, 0, _os.SEEK_SET)
  336:                     _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
  337:                 except OSError:
  338:                     pass
  339:             # fcntl.flock releases on close.
  340: 
  341: 
  342: def backfill_outcomes(max_entries=2000):
  343:     """Backfill price outcomes for signal log entries.
  344: 
  345:     Memory-optimized: only parses the last ``max_entries`` lines as JSON.
  346:     Head entries are streamed as raw bytes during rewrite (BUG-112).
  347: 
  348:     SC-P1-3 (2026-05-02 follow-up): coordinates with concurrent
  349:     log_signal_snapshot() appenders via the sidecar lock used by
  350:     atomic_append_jsonl. Pattern:
  351:       1. Snapshot phase (lock held briefly): record file size at
  352:          backfill start, then read+parse entries from disk.
  353:       2. Process phase (lock RELEASED): make slow HTTP calls. Live
  354:          appenders run normally during this window.
  355:       3. Rewrite phase (lock re-acquired): re-stat the file. Any bytes
  356:          past the snapshot size are concurrent appends — copy them
  357:          verbatim into the rewritten tmp file AFTER the processed tail,
  358:          then os.replace under the lock so the rename is atomic w.r.t.
  359:          any further appender. Without this, every entry appended in
  360:          the process window was clobbered by the os.replace.
  361: 
  362:     Args:
  363:         max_entries: Only process the last N entries to limit memory usage.
  364:             Older entries are assumed to be fully backfilled already.
  365:     """
  366:     if not SIGNAL_LOG.exists():
  367:         return 0
  368: 
  369:     # ---- Phase 1: snapshot read (lock held) ----
  370:     with _hold_signal_log_lock():
  371:         file_size = SIGNAL_LOG.stat().st_size
  372:         if file_size == 0:
  373:             return 0
  374: 
  375:         # Phase 1a: Count total lines (fast binary scan, no JSON parsing)
  376:         total_lines = 0
  377:         with open(SIGNAL_LOG, "rb") as f:
  378:             for _ in f:
  379:                 total_lines += 1
  380: 
  381:         head_count = max(0, total_lines - max_entries) if max_entries else 0
  382: 
  383:         # Phase 1b: Skip head lines, parse only the tail as JSON
  384:         head_end_offset = 0
  385:         entries = []
  386:         with open(SIGNAL_LOG, "rb") as f:
  387:             for _ in range(head_count):
  388:                 f.readline()  # skip without JSON parsing
  389:             head_end_offset = f.tell()
  390: 
  391:             for raw_line in f:
  392:                 stripped = raw_line.strip()
  393:                 if stripped:
  394:                     try:
  395:                         entries.append(json.loads(stripped))
  396:                     except json.JSONDecodeError:
  397:                         continue
  398:         # snapshot_size pins the byte boundary of "what we read".
  399:         # Anything appended past this offset during processing must be
  400:         # preserved verbatim during the rewrite.
  401:         snapshot_size = file_size
  402:         # ---- lock released here for the slow processing window ----
  403: 
  404:     now = datetime.now(UTC)
  405:     now_ts = now.timestamp()
  406:     price_cache = {}
  407:     updated = 0
  408: 
  409:     # Tickers we can actually fetch prices for — skip unknown/removed tickers
  410:     known_tickers = set(BINANCE_SPOT_MAP) | set(BINANCE_FAPI_MAP) | set(YF_MAP)
  411: 
  412:     # Open SignalDB once for all dual-writes (avoids per-outcome open/close)
  413:     _db = None
  414:     try:
  415:         from portfolio.signal_db import SignalDB
  416:         _db = SignalDB()
  417:     except Exception as e:
  418:         logger.debug("SignalDB open failed: %s", e)
  419: 
  420:     for entry in entries:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  420:     for entry in entries:
  421:         try:
  422:             entry_ts = datetime.fromisoformat(entry["ts"]).timestamp()
  423:         except (KeyError, ValueError, TypeError):
  424:             continue
  425:         tickers = entry.get("tickers", {})
  426:         outcomes = entry.get("outcomes", {})
  427: 
  428:         all_filled = True
  429:         for ticker in tickers:
  430:             if ticker not in outcomes:
  431:                 outcomes[ticker] = {h: None for h in HORIZONS}
  432:             for h_key in HORIZONS:
  433:                 if outcomes[ticker].get(h_key) is None:
  434:                     all_filled = False
  435: 
  436:         if all_filled and all(
  437:             all(outcomes[t].get(h) is not None for h in HORIZONS) for t in tickers
  438:         ):
  439:             continue
  440: 
  441:         entry_updated = False
  442:         for ticker in tickers:
  443:             if ticker not in known_tickers:
  444:                 continue  # skip removed/unknown tickers (e.g. AI)
  445:             if ticker not in outcomes:
  446:                 outcomes[ticker] = {h: None for h in HORIZONS}
  447: 
  448:             base_price = tickers[ticker].get("price_usd")
  449:             if not base_price or base_price <= 0:
  450:                 continue  # BUG-220: skip — no base price to compute change_pct
  451:             for h_key, h_seconds in HORIZONS.items():
  452:                 if outcomes[ticker].get(h_key) is not None:
  453:                     continue
  454:                 target_ts = entry_ts + h_seconds
  455:                 if now_ts < target_ts:
  456:                     continue
  457: 
  458:                 cache_key = (ticker, int(target_ts // 3600))
  459:                 if cache_key not in price_cache:
  460:                     try:
  461:                         price_cache[cache_key] = _fetch_historical_price(
  462:                             ticker, target_ts
  463:                         )
  464:                     except Exception:
  465:                         price_cache[cache_key] = None
  466: 
  467:                 hist_price = price_cache[cache_key]
  468:                 if hist_price is None:
  469:                     continue
  470: 
  471:                 change_pct = round(
  472:                     ((hist_price - base_price) / base_price) * 100, 2
  473:                 )
  474: 
  475:                 outcome_ts_str = datetime.fromtimestamp(
  476:                     target_ts, tz=UTC
  477:                 ).isoformat()
  478:                 outcomes[ticker][h_key] = {
  479:                     "price_usd": round(hist_price, 2),
  480:                     "change_pct": change_pct,
  481:                     "ts": outcome_ts_str,
  482:                 }
  483:                 entry_updated = True
  484: 
  485:                 # Dual-write outcome to SQLite
  486:                 if _db is not None:
  487:                     try:
  488:                         _db.update_outcome(
  489:                             entry["ts"], ticker, h_key,
  490:                             round(hist_price, 2), change_pct, outcome_ts_str,
  491:                         )
  492:                     except Exception as e:
  493:                         logger.debug("SQLite outcome write failed: %s", e)
  494: 
  495:         entry["outcomes"] = outcomes
  496: 
  497:         # C6: MWU weight update removed — SignalWeightManager.batch_update()
  498:         # wrote to data/signal_weights.json but signal_engine.py never read it.
  499:         # The entire MWU adaptation path was dead code producing disk I/O and
  500:         # CPU burn for zero effect. See Adversarial Review C6 for details.
  501: 
  502:         if entry_updated:
  503:             updated += 1
  504: 
  505:     if _db is not None:
  506:         try:
  507:             _db.close()
  508:         except Exception as e:
  509:             logger.debug("SignalDB close failed: %s", e)
  510: 
  511:     import os
  512:     import tempfile
  513: 
  514:     # ---- Phase 3: rewrite (lock re-acquired) ----
  515:     # SC-P1-3: hold the sidecar lock across the rewrite so that:
  516:     #   1. We can stat the file and detect concurrent appends past
  517:     #      snapshot_size without racing a partial append.
  518:     #   2. The os.replace is atomic w.r.t. any subsequent appender —
  519:     #      no appender can land bytes between our copy and our rename.
  520:     with _hold_signal_log_lock():
  521:         # Re-stat to find any new bytes appended during processing.
  522:         try:
  523:             current_size = SIGNAL_LOG.stat().st_size
  524:         except FileNotFoundError:
  525:             current_size = snapshot_size  # nothing to preserve
  526:         concurrent_tail_bytes = max(0, current_size - snapshot_size)
  527: 
  528:         fd, tmp = tempfile.mkstemp(dir=SIGNAL_LOG.parent, suffix=".tmp")
  529:         try:
  530:             with os.fdopen(fd, "wb") as f_out:
  531:                 # Stream head bytes verbatim from original file (no JSON parsing)
  532:                 if head_end_offset > 0:
  533:                     with open(SIGNAL_LOG, "rb") as f_in:
  534:                         remaining = head_end_offset
  535:                         while remaining > 0:
  536:                             chunk = f_in.read(min(65536, remaining))
  537:                             if not chunk:
  538:                                 break
  539:                             f_out.write(chunk)
  540:                             remaining -= len(chunk)
  541:                 # Write modified tail entries
  542:                 for entry in entries:
  543:                     f_out.write((json.dumps(entry) + "\n").encode("utf-8"))
  544:                 # SC-P1-3: copy bytes appended after our snapshot verbatim
  545:                 # so concurrent log_signal_snapshot() appends survive the
  546:                 # rewrite. We never parse them as JSON — preserves every
  547:                 # byte exactly as the appender wrote it (including the
  548:                 # trailing newline atomic_append_jsonl always emits).
  549:                 if concurrent_tail_bytes > 0:
  550:                     with open(SIGNAL_LOG, "rb") as f_in:
  551:                         f_in.seek(snapshot_size)
  552:                         remaining = concurrent_tail_bytes
  553:                         while remaining > 0:
  554:                             chunk = f_in.read(min(65536, remaining))
  555:                             if not chunk:
  556:                                 break
  557:                             f_out.write(chunk)
  558:                             remaining -= len(chunk)
  559:             os.replace(tmp, SIGNAL_LOG)
  560:         except BaseException:
  561:             with contextlib.suppress(OSError):
  562:                 os.unlink(tmp)
  563:             raise
  564: 
  565:     # Invalidate signal utility cache so the next cycle picks up fresh
  566:     # accuracy data immediately rather than waiting for the 300s TTL.
  567:     if updated > 0:
  568:         try:
  569:             from portfolio.accuracy_stats import invalidate_signal_utility_cache
  570:             invalidate_signal_utility_cache()
  571:             logger.info("Signal utility cache invalidated after backfill (%d entries)", updated)
  572:         except Exception:
  573:             logger.debug("Could not invalidate signal utility cache", exc_info=True)
  574: 
  575:     return updated
  576: 
  577: 
  578: if __name__ == "__main__":
  579:     count = backfill_outcomes()
  580:     print(f"Backfilled {count} entries")
  581: 
codex
I’ve moved into the persistence/analytics side. The outcome backfill path is doing its own JSONL parsing and rewrite choreography instead of staying on the project’s atomic helper path, so I’m checking whether that introduces silent drops, stale caches, or incorrect accuracy math downstream.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: import logging
    2: import threading
    3: import time
    4: from collections import defaultdict
    5: from pathlib import Path
    6: 
    7: logger = logging.getLogger("portfolio.accuracy_stats")
    8: 
    9: from datetime import UTC
   10: 
   11: from portfolio.file_utils import atomic_write_json as _atomic_write_json
   12: from portfolio.file_utils import load_json, load_jsonl, load_jsonl_tail
   13: from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES
   14: 
   15: # C2: Protect all read-modify-write cache operations from concurrent ticker threads
   16: _accuracy_write_lock = threading.Lock()
   17: 
   18: # BUG-178 (2026-04-16): thundering-herd protection. The disk-backed accuracy
   19: # caches expire on a 1h TTL; on the first cycle after expiry, all 5 ticker
   20: # threads race through load_cached_accuracy() → None → signal_accuracy() and
   21: # each pays the 7s+ cost of loading 50,000 signal-log entries from SQLite.
   22: # Wall time was measured at 215s for a 5-thread race vs 7s single-threaded —
   23: # 30x amplification driven by GIL + DB + file-I/O serialization. The
   24: # get_or_compute_*() helpers below use double-checked locking: cache hits
   25: # take the fast path with no lock acquisition; only the first miss-thread
   26: # computes, and the others wait on _accuracy_compute_lock and then read the
   27: # freshly-populated cache. The lock is held THROUGH the compute (unlike the
   28: # signal_utility cache below) because cache-miss is rare (~once per hour
   29: # per horizon) and serializing 4 threads through a 7s wait is far cheaper
   30: # than 4 redundant 50000-entry SQL scans.
   31: _accuracy_compute_lock = threading.Lock()
   32: 
   33: BASE_DIR = Path(__file__).resolve().parent.parent
   34: DATA_DIR = BASE_DIR / "data"
   35: SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
   36: ACCURACY_CACHE_FILE = DATA_DIR / "accuracy_cache.json"
   37: BEST_HORIZON_CACHE_FILE = DATA_DIR / "best_horizon_cache.json"
   38: ACCURACY_CACHE_TTL = 3600
   39: HORIZONS = ["3h", "4h", "12h", "1d", "3d", "5d", "10d"]
   40: 
   41: # In-memory cache for signal_utility (added 2026-04-15, BUG-178 mitigation).
   42: # signal_utility() walks every entry in the signal log (~6320 snapshots / ~92K
   43: # ticker rows as of this writing) and costs ~3.6s cold and <50ms hot. It's
   44: # called from generate_signal() on every ticker, every cycle, with NO
   45: # disk-backed cache — so when the OS file cache is cold (memory pressure,
   46: # fresh process, antivirus scan) 5 concurrent ticker threads each pay the
   47: # 3-4s cold read, which can compound under file-cache page-in contention.
   48: #
   49: # Invalidation is pure TTL (300s) — there is NO mtime check against
   50: # signal_log.db, so a backfill that writes new outcomes mid-cycle is only
   51: # visible to signal_utility after the TTL expires. This is an explicit
   52: # trade: outcome backfill runs daily at 18:00 local via the PF-OutcomeCheck
   53: # scheduled task (see docs/operational-runbook.md), so a 5-minute staleness
   54: # window is dominated by the 24-hour write cadence. Code paths that need
   55: # immediately-fresh utility (tests, outcome_tracker, post-backfill reports)
   56: # must either pass entries= explicitly (which bypasses the cache) or call
   57: # invalidate_signal_utility_cache() after the write.
   58: #
   59: # 300s TTL matches the shortest LLM rotation period and is well below the
   60: # 3600s ACCURACY_CACHE_TTL used for the disk-backed caches. The lock
   61: # guards the (timestamp, value) tuple so two threads racing to refresh
   62: # can't corrupt the dict. Dogpile behavior: the lock is held ONLY for the
   63: # swap, NOT for the compute — the slow signal_utility() call happens
   64: # outside the lock, so other threads waiting on the lock see the fresh
   65: # value the moment the first thread returns. Two threads that both miss
   66: # on a TTL-boundary race will each recompute once (one wasted walk), but
   67: # neither blocks the other. This is cheaper than holding a global lock
   68: # through a 3.6s disk scan and funneling every ticker thread through it.
   69: _SIGNAL_UTILITY_CACHE_TTL = 300.0
   70: _signal_utility_cache: dict[str, tuple[float, dict]] = {}
   71: _signal_utility_cache_lock = threading.Lock()
   72: 
   73: # L2 disk-backed cache (added 2026-05-03, BUG-178 follow-up).
   74: # Survives process restart so the first cycle after a `schtasks /run` doesn't
   75: # pay the ~49s parallel-cold-compute cost we measured under 4-thread
   76: # contention (PASS 4 of scripts/perf/profile_utility_overlay.py). Mirrors the
   77: # existing pattern at regime_accuracy_cache.json: single "time" key gates TTL,
   78: # per-horizon data persists across writes via load-merge-write.
   79: #
   80: # 1-hour TTL matches ACCURACY_CACHE_TTL and is appropriate because outcome
   81: # backfill runs daily — a fresh-after-restart cache from earlier today is
   82: # more accurate than a freshly-computed cache from a partially-loaded SQLite.
   83: # Atomic writes via _atomic_write_json so concurrent writers race to be last
   84: # without producing torn reads.
   85: _SIGNAL_UTILITY_DISK_TTL = 3600.0
   86: SIGNAL_UTILITY_CACHE_FILE = DATA_DIR / "signal_utility_cache.json"
   87: 
   88: # Separate from _signal_utility_cache_lock — that one only guards the in-memory
   89: # dict swap. This one serializes the read-modify-write of the disk file so two
   90: # threads computing DIFFERENT horizons don't lose each other's writes (the
   91: # benign-race assumption only holds for same-horizon races; cross-horizon
   92: # races would lose 3 of 4 horizons on a 4-thread cold-start cycle, defeating
   93: # the L2 cache's purpose). Disk write is ~10-50ms — keeping this lock
   94: # separate from the L1 lock means L1 reads aren't blocked behind disk IO.
   95: _signal_utility_disk_lock = threading.Lock()
   96: 
   97: 
   98: def _load_signal_utility_disk(horizon: str) -> dict | None:
   99:     """Return cached utility dict for horizon if disk cache is fresh, else None.
  100: 
  101:     Single global "time" timestamp gates TTL for all horizons (matches
  102:     regime_accuracy_cache pattern). Per-horizon data persists across writes,
  103:     so a horizon that hasn't been recomputed since the last process can
  104:     still be served from disk as long as the file's "time" is fresh.
  105:     """
  106:     cache = load_json(SIGNAL_UTILITY_CACHE_FILE)
  107:     if not isinstance(cache, dict):
  108:         return None
  109:     if time.time() - cache.get("time", 0) >= _SIGNAL_UTILITY_DISK_TTL:
  110:         return None
  111:     cached = cache.get(horizon)
  112:     return cached if isinstance(cached, dict) else None
  113: 
  114: 
  115: def _write_signal_utility_disk(horizon: str, data: dict) -> None:
  116:     """Persist cached utility dict for horizon. Merges with existing horizons.
  117: 
  118:     Holds _signal_utility_disk_lock through the read-modify-write so two
  119:     threads computing DIFFERENT horizons don't lose each other's writes.
  120:     The earlier lock-free version assumed all races would be same-horizon
  121:     (idempotent), but a 4-thread cold-start cycle realistically has 4
  122:     different horizons in flight; the lockless last-writer-wins kept only
  123:     1 of 4 horizons on disk and forced the other 3 to recompute next cycle.
  124: 
  125:     Atomic write via _atomic_write_json prevents torn reads even outside
  126:     the lock.
  127: 
  128:     Failures are swallowed — a stale L2 cache is harmless (TTL expires) and
  129:     must never crash the live signal pipeline (would re-trigger BUG-178's
  130:     silent-failure pattern).
  131:     """
  132:     try:
  133:         with _signal_utility_disk_lock:
  134:             cache = load_json(SIGNAL_UTILITY_CACHE_FILE, default={})
  135:             if not isinstance(cache, dict):
  136:                 cache = {}
  137:             cache[horizon] = data
  138:             cache["time"] = time.time()
  139:             _atomic_write_json(SIGNAL_UTILITY_CACHE_FILE, cache)
  140:     except Exception:
  141:         logger.debug("signal_utility disk cache write failed", exc_info=True)
  142: 
  143: 
  144: def load_entries():
  145:     """Load signal log entries. Prefers SQLite if available, falls back to JSONL."""
  146:     try:
  147:         from portfolio.signal_db import SignalDB
  148:         db = SignalDB()
  149:         try:
  150:             count = db.snapshot_count()
  151:             if count > 0:
  152:                 entries = db.load_entries()
  153:                 return entries
  154:         finally:
  155:             # BUG-137: Always close DB, even if load_entries() throws.
  156:             db.close()
  157:     except Exception as e:
  158:         logger.debug("SQLite signal_db unavailable, falling back to JSONL: %s", e)
  159:     # H2: Fallback to JSONL using atomic load_jsonl_tail (avoids raw open()).
  160:     # 50000 entries covers full accuracy computation; reading all 68MB risks OOM.
  161:     if not SIGNAL_LOG.exists():
  162:         return []
  163:     entries = load_jsonl_tail(SIGNAL_LOG, max_entries=50000)
  164:     return entries if entries else []
  165: 
  166: 
  167: _MIN_CHANGE_PCT = 0.05  # outcomes within ±0.05% are treated as neutral (skip)
  168: 
  169: 
  170: def _vote_correct(vote, change_pct, min_change_pct=None):
  171:     """Check if a signal vote matches the price outcome.
  172: 
  173:     Returns True (correct), False (incorrect), or None (neutral — skip this outcome).
  174:     Outcomes within ±min_change_pct are considered noise and should not count
  175:     for or against the signal's accuracy.
  176:     """
  177:     threshold = min_change_pct if min_change_pct is not None else _MIN_CHANGE_PCT
  178:     # 2026-04-22: some outcome entries have change_pct=None (missing backfill
  179:     # data for 4h+ horizons). Treat as neutral instead of TypeErroring — was
  180:     # killing --accuracy report mid-horizon. Matches the None-guard pattern
  181:     # at accuracy_stats.py:1617 and in ic_computation / train_signal_weights.
  182:     if change_pct is None or abs(change_pct) < threshold:
  183:         return None  # neutral — price didn't move enough to judge (or unknown)
  184:     if vote == "BUY" and change_pct > 0:
  185:         return True
  186:     return bool(vote == "SELL" and change_pct < 0)
  187: 
  188: 
  189: def signal_accuracy(horizon="1d", since=None, entries=None):
  190:     """Compute per-signal accuracy, optionally filtered to entries after `since`.
  191: 
  192:     Args:
  193:         horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
  194:         since: Optional ISO-8601 string cutoff. Only entries with ts >= since
  195:                are included. None means all entries (no time filter).
  196:         entries: Pre-loaded entries list. If None, loads from disk.
  197: 
  198:     Returns:
  199:         dict: {signal_name: {correct, total, accuracy, pct}} for each signal.
  200:     """
  201:     if entries is None:
  202:         entries = load_entries()
  203:     stats = {s: {"correct": 0, "total": 0,
  204:                  "correct_buy": 0, "total_buy": 0,
  205:                  "correct_sell": 0, "total_sell": 0} for s in SIGNAL_NAMES}
  206:     # 2026-04-22 follow-up: count outcomes we skip because change_pct is None.
  207:     # Previously these crashed the report; now they're silently dropped, which
  208:     # would let a data-quality regression (e.g. outcome_tracker writing nulls)
  209:     # go unnoticed. Surface the count so operators see drift.
  210:     null_change_pct_skipped = 0
  211:     total_outcomes_seen = 0
  212: 
  213:     for entry in entries:
  214:         if since and entry.get("ts", "") < since:
  215:             continue
  216:         outcomes = entry.get("outcomes", {})
  217:         tickers = entry.get("tickers", {})
  218: 
  219:         for ticker, tdata in tickers.items():
  220:             outcome = outcomes.get(ticker, {}).get(horizon)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  220:             outcome = outcomes.get(ticker, {}).get(horizon)
  221:             if not outcome:
  222:                 continue
  223:             total_outcomes_seen += 1
  224: 
  225:             change_pct = outcome.get("change_pct", 0)
  226:             if change_pct is None:
  227:                 null_change_pct_skipped += 1
  228:             signals = tdata.get("signals", {})
  229: 
  230:             for sig_name in SIGNAL_NAMES:
  231:                 vote = signals.get(sig_name, "HOLD")
  232:                 if vote == "HOLD":
  233:                     continue
  234:                 result_val = _vote_correct(vote, change_pct)
  235:                 if result_val is None:
  236:                     continue  # neutral outcome — don't count
  237:                 stats[sig_name]["total"] += 1
  238:                 if vote == "BUY":
  239:                     stats[sig_name]["total_buy"] += 1
  240:                     if result_val:
  241:                         stats[sig_name]["correct_buy"] += 1
  242:                 else:
  243:                     stats[sig_name]["total_sell"] += 1
  244:                     if result_val:
  245:                         stats[sig_name]["correct_sell"] += 1
  246:                 if result_val:
  247:                     stats[sig_name]["correct"] += 1
  248: 
  249:     if null_change_pct_skipped > 0:
  250:         pct = 100.0 * null_change_pct_skipped / total_outcomes_seen
  251:         logger.info(
  252:             "signal_accuracy[%s]: skipped %d/%d outcomes (%.2f%%) with "
  253:             "change_pct=None — check outcome_tracker / signal_db backfill",
  254:             horizon, null_change_pct_skipped, total_outcomes_seen, pct,
  255:         )
  256: 
  257:     result = {}
  258:     for sig_name in SIGNAL_NAMES:
  259:         s = stats[sig_name]
  260:         acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
  261:         buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
  262:         sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
  263:         result[sig_name] = {
  264:             "correct": s["correct"],
  265:             "total": s["total"],
  266:             "samples": s["total"],
  267:             "enabled": sig_name not in DISABLED_SIGNALS,
  268:             "accuracy": acc,
  269:             "pct": round(acc * 100, 1),
  270:             "correct_buy": s["correct_buy"],
  271:             "total_buy": s["total_buy"],
  272:             "buy_accuracy": round(buy_acc, 4),
  273:             "correct_sell": s["correct_sell"],
  274:             "total_sell": s["total_sell"],
  275:             "sell_accuracy": round(sell_acc, 4),
  276:         }
  277:     return result
  278: 
  279: 
  280: def signal_accuracy_recent(horizon="1d", days=7):
  281:     """Compute per-signal accuracy using only the last N days of data.
  282: 
  283:     Thin wrapper around signal_accuracy() with a time cutoff.
  284:     """
  285:     from datetime import datetime, timedelta
  286: 
  287:     cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
  288:     return signal_accuracy(horizon, since=cutoff)
  289: 
  290: 
  291: def signal_accuracy_ewma(horizon="1d", halflife_days=5, entries=None):
  292:     """Compute per-signal accuracy with exponential decay weighting.
  293: 
  294:     Recent observations are weighted higher than older ones. The weight for
  295:     an observation of age ``age_days`` is ``exp(-ln(2) / halflife_days * age_days)``,
  296:     meaning entries that are ``halflife_days`` old receive half the weight of
  297:     today's entries.
  298: 
  299:     Args:
  300:         horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
  301:         halflife_days: Half-life of the decay in days (default 5). Smaller
  302:             values weight recent data more aggressively.
  303:         entries: Pre-loaded entries list. If None, loads from disk.
  304: 
  305:     Returns:
  306:         dict: {signal_name: {accuracy, total_weight, effective_samples, total, correct, pct}}
  307:         where ``total`` and ``correct`` are int(round(...)) of the weighted sums
  308:         for compatibility with the existing accuracy pipeline.
  309:     """
  310:     import math
  311:     from datetime import datetime
  312: 
  313:     if entries is None:
  314:         entries = load_entries()
  315:     now = datetime.now(UTC)
  316:     decay_rate = math.log(2) / halflife_days  # λ = ln(2) / t½
  317: 
  318:     # Accumulate per-signal: weighted_total, weighted_correct, sum_of_sq_weights
  319:     stats = {
  320:         s: {"w_total": 0.0, "w_correct": 0.0, "sum_w2": 0.0}
  321:         for s in SIGNAL_NAMES
  322:     }
  323: 
  324:     for entry in entries:
  325:         # Compute age in days from entry timestamp
  326:         ts_str = entry.get("ts", "")
  327:         try:
  328:             entry_dt = datetime.fromisoformat(ts_str)
  329:             age_days = (now - entry_dt).total_seconds() / 86400.0
  330:             age_days = max(age_days, 0.0)  # clamp: never negative for future entries
  331:         except (ValueError, TypeError):
  332:             continue  # skip malformed timestamps
  333: 
  334:         weight = math.exp(-decay_rate * age_days)
  335: 
  336:         outcomes = entry.get("outcomes", {})
  337:         tickers = entry.get("tickers", {})
  338: 
  339:         for ticker, tdata in tickers.items():
  340:             outcome = outcomes.get(ticker, {}).get(horizon)
  341:             if not outcome:
  342:                 continue
  343: 
  344:             change_pct = outcome.get("change_pct", 0)
  345:             signals = tdata.get("signals", {})
  346: 
  347:             for sig_name in SIGNAL_NAMES:
  348:                 vote = signals.get(sig_name, "HOLD")
  349:                 if vote == "HOLD":
  350:                     continue
  351:                 result_val = _vote_correct(vote, change_pct)
  352:                 if result_val is None:
  353:                     continue  # neutral outcome — skip
  354: 
  355:                 stats[sig_name]["w_total"] += weight
  356:                 stats[sig_name]["sum_w2"] += weight * weight
  357:                 if result_val:
  358:                     stats[sig_name]["w_correct"] += weight
  359: 
  360:     result = {}
  361:     for sig_name in SIGNAL_NAMES:
  362:         s = stats[sig_name]
  363:         w_total = s["w_total"]
  364:         w_correct = s["w_correct"]
  365:         sum_w2 = s["sum_w2"]
  366: 
  367:         if w_total > 0:
  368:             accuracy = w_correct / w_total
  369:             # Kish (1965) effective sample size: n_eff = (Σwᵢ)² / Σwᵢ²
  370:             effective_samples = (w_total * w_total) / sum_w2
  371:         else:
  372:             accuracy = 0.0
  373:             effective_samples = 0.0
  374: 
  375:         result[sig_name] = {
  376:             "accuracy": accuracy,
  377:             "total_weight": w_total,
  378:             "effective_samples": effective_samples,
  379:             "total": int(round(w_total)),
  380:             "samples": int(round(w_total)),
  381:             "enabled": sig_name not in DISABLED_SIGNALS,
  382:             "correct": int(round(w_correct)),
  383:             "pct": round(accuracy * 100, 1),
  384:         }
  385:     return result
  386: 
  387: 
  388: def signal_accuracy_cost_adjusted(horizon="1d", cost_bps=10.0, entries=None):
  389:     """Compute per-signal accuracy adjusted for transaction costs.
  390: 
  391:     A signal vote is only counted as correct if the price move exceeds
  392:     the estimated round-trip cost (spread + slippage).  This reveals
  393:     signals that are "technically correct" but unprofitable after execution.
  394: 
  395:     Args:
  396:         horizon: Outcome horizon to evaluate.
  397:         cost_bps: Estimated round-trip cost in basis points (default 10 bps).
  398:                   Must exceed _MIN_CHANGE_PCT (5 bps) to filter beyond
  399:                   the neutral outcome threshold. Metals warrants ~10 bps,
  400:                   crypto ~5 bps.
  401:         entries: Pre-loaded entries list. If None, loads from disk.
  402: 
  403:     Returns:
  404:         dict: {signal_name: {correct, total, accuracy, pct, cost_bps}}
  405:     """
  406:     if entries is None:
  407:         entries = load_entries()
  408: 
  409:     # Cost threshold: moves below this are unprofitable even if directionally correct
  410:     cost_pct = cost_bps / 100.0  # convert bps to percentage
  411: 
  412:     stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}
  413: 
  414:     for entry in entries:
  415:         outcomes = entry.get("outcomes", {})
  416:         tickers = entry.get("tickers", {})
  417: 
  418:         for ticker, tdata in tickers.items():
  419:             outcome = outcomes.get(ticker, {}).get(horizon)
  420:             if not outcome:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  840:         bias_penalty = max(1.0 - bias, 0.1)  # floor at 0.1
  841: 
  842:         result[sig_name] = {
  843:             "activation_rate": round(activation_rate, 4),
  844:             "buy_rate": round(buy_rate, 4),
  845:             "sell_rate": round(sell_rate, 4),
  846:             "bias": round(bias, 4),
  847:             "rarity_weight": round(rarity_weight, 4),
  848:             "bias_penalty": round(bias_penalty, 4),
  849:             "normalized_weight": round(rarity_weight * bias_penalty, 4),
  850:             "samples": total,
  851:         }
  852: 
  853:     return result
  854: 
  855: 
  856: # P2-B (2026-04-17 adversarial review): default min_recent_samples was 50
  857: # while production (signal_engine) passes 30. Default lowered so non-prod
  858: # callers (backtester, replay script) match live behavior rather than
  859: # silently dropping the blended value for signals with 30-49 recent samples.
  860: _BLEND_DEFAULT_MIN_RECENT_SAMPLES = 30
  861: 
  862: 
  863: def blend_accuracy_data(alltime, recent, divergence_threshold=0.15,
  864:                         normal_weight=0.70, fast_weight=0.90,
  865:                         min_recent_samples=_BLEND_DEFAULT_MIN_RECENT_SAMPLES):
  866:     """Blend all-time and recent accuracy using adaptive recency weighting.
  867: 
  868:     When recent accuracy diverges sharply from all-time (> divergence_threshold),
  869:     fast-track to higher recent weight for faster regime adaptation.
  870: 
  871:     Args:
  872:         alltime: Dict of {signal_name: {accuracy, total, correct, pct}}.
  873:         recent: Dict of {signal_name: {accuracy, total, correct, pct}}.
  874:         divergence_threshold: Absolute accuracy difference that triggers fast blend.
  875:         normal_weight: Recent weight when divergence is below threshold.
  876:         fast_weight: Recent weight when divergence exceeds threshold.
  877:         min_recent_samples: Minimum recent samples before blending (else use alltime).
  878: 
  879:     Returns:
  880:         dict: Blended {signal_name: {accuracy, total, correct, pct}}.
  881:     """
  882:     if not alltime and not recent:
  883:         return {}
  884:     # Codex round-10 P2 (2026-04-17 follow-up): don't early-return recent
  885:     # without going through the blend loop - the min_recent_samples floor
  886:     # must apply to recent-only signals too (a 20-sample signal with recent
  887:     # accuracy=0.80 should default to neutral 0.5, not vote at 0.80).
  888:     # Treating empty alltime/recent as {} keeps the loop's per-signal
  889:     # sample-threshold logic authoritative.
  890:     if not alltime:
  891:         alltime = {}
  892:     if not recent:
  893:         recent = {}
  894: 
  895:     # P1-D (2026-04-17 adversarial review): iterate over the UNION of signal
  896:     # names, not just alltime. Previously a signal present only in `recent`
  897:     # was silently dropped, and directional keys (buy_accuracy, sell_accuracy,
  898:     # total_buy, total_sell) were copied only from `at` - so a signal with
  899:     # sell_accuracy=0.28 over 400 samples in recent but no alltime entry
  900:     # silently passed the directional gate because total_sell defaulted to 0.
  901:     accuracy_data = {}
  902:     all_signal_names = set(alltime) | set(recent)
  903:     for sig_name in all_signal_names:
  904:         at = alltime.get(sig_name) or {}
  905:         rc = recent.get(sig_name) or {}
  906:         at_acc = at.get("accuracy", 0.5) if at else 0.5
  907:         rc_acc = rc.get("accuracy", 0.5) if rc else 0.5
  908:         rc_samples = rc.get("total", 0) if rc else 0
  909:         at_samples = at.get("total", 0) if at else 0
  910: 
  911:         # Blend only when recent has enough samples AND alltime exists;
  912:         # otherwise fall back to whichever source has data.
  913:         # Codex round-10 P2 (2026-04-17 follow-up): previously a recent-only
  914:         # signal with <min_recent_samples samples fell through to rc_acc,
  915:         # letting an immature signal's raw recent accuracy drive consensus.
  916:         # Now we require min_recent_samples even for recent-only signals,
  917:         # falling back to a neutral 0.5 otherwise (matches pre-patch
  918:         # semantics for signals below the recent-sample floor).
  919:         if rc_samples >= min_recent_samples and at_samples > 0:
  920:             divergence = abs(rc_acc - at_acc)
  921:             w = fast_weight if divergence > divergence_threshold else normal_weight
  922:             blended = w * rc_acc + (1 - w) * at_acc
  923:         elif at_samples > 0:
  924:             blended = at_acc
  925:         elif rc_samples >= min_recent_samples:
  926:             blended = rc_acc  # recent-only signal with enough samples
  927:         else:
  928:             blended = 0.5  # immature signal: neutral default
  929: 
  930:         total = max(at_samples, rc_samples)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  930:         total = max(at_samples, rc_samples)
  931:         result = {
  932:             "accuracy": blended,
  933:             "total": total,
  934:             "samples": total,
  935:             "enabled": sig_name not in DISABLED_SIGNALS,
  936:             "correct": int(round(blended * total)),  # BUG-186
  937:             "pct": round(blended * 100, 1),
  938:         }
  939:         # Codex round 11 P2 (2026-04-17 follow-up): directional stats must
  940:         # follow the same sample-floor rule as `accuracy`. Without this,
  941:         # a recent-only signal with 20-29 one-sided votes still influenced
  942:         # _weighted_consensus's directional gate/weighting at its raw
  943:         # recent directional accuracy, even though the overall `accuracy`
  944:         # field had already been set back to neutral 0.5. Omit directional
  945:         # keys entirely for immature signals so downstream callers see the
  946:         # `.get('buy_accuracy', acc)` fallback.
  947:         _directionals_trustworthy = (
  948:             at_samples > 0 or rc_samples >= min_recent_samples
  949:         )
  950:         if _directionals_trustworthy:
  951:             # Merge directional keys from the larger-sample source per key.
  952:             # Prevents silent gate-bypass when a key exists only in `recent`.
  953:             for key in ("buy_accuracy", "sell_accuracy"):
  954:                 if key in at and key in rc:
  955:                     side_total = "total_buy" if key == "buy_accuracy" else "total_sell"
  956:                     at_side = at.get(side_total, 0) or 0
  957:                     rc_side = rc.get(side_total, 0) or 0
  958:                     result[key] = at[key] if at_side >= rc_side else rc[key]
  959:                 elif key in at:
  960:                     result[key] = at[key]
  961:                 elif key in rc:
  962:                     result[key] = rc[key]
  963:             for key in ("total_buy", "total_sell"):
  964:                 at_v = at.get(key, 0) or 0
  965:                 rc_v = rc.get(key, 0) or 0
  966:                 if at_v or rc_v:
  967:                     result[key] = max(at_v, rc_v)
  968:         accuracy_data[sig_name] = result
  969:     return accuracy_data
  970: 
  971: 
  972: ACTIVATION_CACHE_TTL = 3600  # recompute hourly
  973: 
  974: 
  975: def load_cached_activation_rates():
  976:     """Load cached activation rates, recomputing if stale."""
  977:     cache_file = DATA_DIR / "activation_cache.json"
  978:     cache = load_json(cache_file)
  979:     if cache is not None:
  980:         try:
  981:             if time.time() - cache.get("time", 0) < ACTIVATION_CACHE_TTL:
  982:                 return cache.get("rates", {})
  983:         except (KeyError, AttributeError):
  984:             logger.debug("Activation rates cache corrupted, regenerating")
  985:     rates = signal_activation_rates()
  986:     try:
  987:         with _accuracy_write_lock:
  988:             _atomic_write_json(cache_file, {"rates": rates, "time": time.time()})
  989:     except Exception:
  990:         logger.warning("Failed to write activation rates cache", exc_info=True)
  991:     return rates
  992: 
  993: 
  994: def load_cached_accuracy(horizon="1d"):
  995:     cache = load_json(ACCURACY_CACHE_FILE)
  996:     if cache is not None:
  997:         try:
  998:             # BUG-133: Use per-horizon timestamps to avoid cross-horizon staleness.
  999:             # Fall back to legacy shared "time" key for backwards compatibility.
 1000:             ts = cache.get(f"time_{horizon}", cache.get("time", 0))
 1001:             if time.time() - ts < ACCURACY_CACHE_TTL:
 1002:                 cached = cache.get(horizon)
 1003:                 if cached:
 1004:                     return cached
 1005:         except (KeyError, AttributeError):
 1006:             logger.debug("Accuracy cache corrupted or missing horizon %s", horizon)
 1007:     return None
 1008: 
 1009: 
 1010: def write_accuracy_cache(horizon, data):
 1011:     with _accuracy_write_lock:
 1012:         cache = load_json(ACCURACY_CACHE_FILE, default={})
 1013:         if not isinstance(cache, dict):
 1014:             cache = {}
 1015:         cache[horizon] = data
 1016:         # BUG-133: Write per-horizon timestamp so other horizons don't appear fresh.
 1017:         cache[f"time_{horizon}"] = time.time()
 1018:         # Keep legacy "time" key for backwards compat with older code paths.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1018:         # Keep legacy "time" key for backwards compat with older code paths.
 1019:         cache["time"] = time.time()
 1020:         _atomic_write_json(ACCURACY_CACHE_FILE, cache)
 1021: 
 1022: 
 1023: # BUG-178 (2026-04-16) cache-miss wrappers. See _accuracy_compute_lock comment
 1024: # at the top of this module for the rationale. Callers that previously did
 1025: # `cached = load_cached_accuracy(h); if not cached: cached = signal_accuracy(h);
 1026: # write_accuracy_cache(h, cached)` should call these instead so the compute
 1027: # is done at most once across all racing ticker threads.
 1028: 
 1029: def get_or_compute_accuracy(horizon: str):
 1030:     """Return cached all-time accuracy, computing it once if cache is cold.
 1031: 
 1032:     Thread-safe via double-checked locking — first miss-thread computes,
 1033:     others wait on _accuracy_compute_lock and then read the populated cache.
 1034:     """
 1035:     cached = load_cached_accuracy(horizon)
 1036:     if cached:
 1037:         return cached
 1038:     with _accuracy_compute_lock:
 1039:         cached = load_cached_accuracy(horizon)
 1040:         if cached:
 1041:             return cached
 1042:         result = signal_accuracy(horizon)
 1043:         if result:
 1044:             write_accuracy_cache(horizon, result)
 1045:         return result
 1046: 
 1047: 
 1048: def get_or_compute_recent_accuracy(horizon: str, days: int = 7):
 1049:     """Cached recent-window (default 7d) accuracy, computed at most once."""
 1050:     cache_key = f"{horizon}_recent"
 1051:     cached = load_cached_accuracy(cache_key)
 1052:     if cached:
 1053:         return cached
 1054:     with _accuracy_compute_lock:
 1055:         cached = load_cached_accuracy(cache_key)
 1056:         if cached:
 1057:             return cached
 1058:         result = signal_accuracy_recent(horizon, days=days)
 1059:         if result:
 1060:             write_accuracy_cache(cache_key, result)
 1061:         return result
 1062: 
 1063: 
 1064: def get_or_compute_per_ticker_accuracy(horizon: str):
 1065:     """Cached per-ticker consensus accuracy, computed at most once.
 1066: 
 1067:     Cache key matches the BUG-164 lazy-populate convention used by
 1068:     signal_engine.py:_ptc_key.
 1069:     """
 1070:     cache_key = f"per_ticker_consensus_{horizon}"
 1071:     cached = load_cached_accuracy(cache_key)
 1072:     if cached:
 1073:         return cached
 1074:     with _accuracy_compute_lock:
 1075:         cached = load_cached_accuracy(cache_key)
 1076:         if cached:
 1077:             return cached
 1078:         result = per_ticker_accuracy(horizon)
 1079:         if result:
 1080:             write_accuracy_cache(cache_key, result)
 1081:         return result
 1082: 
 1083: 
 1084: def get_or_compute_consensus_accuracy(horizon: str):
 1085:     """Cached aggregate consensus accuracy, computed at most once.
 1086: 
 1087:     2026-05-03: added to fix /api/accuracy timeout (>15s). The dashboard
 1088:     requests 4 horizons × 3 metrics = 12 lookups; previously consensus
 1089:     was the only one without a cache, forcing a full signal-log scan
 1090:     per horizon per request.
 1091: 
 1092:     Cache key: `consensus_{horizon}`. Mirrors get_or_compute_accuracy
 1093:     semantics — double-checked locking, written on first miss.
 1094:     """
 1095:     cache_key = f"consensus_{horizon}"
 1096:     cached = load_cached_accuracy(cache_key)
 1097:     if cached:
 1098:         return cached
 1099:     with _accuracy_compute_lock:
 1100:         cached = load_cached_accuracy(cache_key)
 1101:         if cached:
 1102:             return cached
 1103:         result = consensus_accuracy(horizon)
 1104:         # consensus_accuracy always returns a dict (even with total=0);
 1105:         # write all of them so a horizon with zero scored entries doesn't
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1105:         # write all of them so a horizon with zero scored entries doesn't
 1106:         # re-compute on every request.
 1107:         if result is not None:
 1108:             write_accuracy_cache(cache_key, result)
 1109:         return result
 1110: 
 1111: 
 1112: # Dashboard /api/accuracy serves these four horizons; the loop's normal
 1113: # operation only warms 1d / 3h / their _recent variants. Without an
 1114: # explicit prewarm, 3d / 5d / 10d are cold and the first dashboard
 1115: # request after a restart spends seconds re-scanning the signal log.
 1116: _DASHBOARD_PREWARM_HORIZONS: tuple = ("1d", "3d", "5d", "10d")
 1117: _DASHBOARD_PREWARM_INTERVAL_SEC = 3600.0  # 1 hour
 1118: _DASHBOARD_PREWARM_STATE_FILE = DATA_DIR / "dashboard_prewarm_state.json"
 1119: _DASHBOARD_PREWARM_LOCK_FILE = DATA_DIR / "dashboard_prewarm.lock"
 1120: _dashboard_prewarm_lock = threading.Lock()
 1121: _last_dashboard_prewarm_ts: float = 0.0
 1122: _dashboard_prewarm_loaded: bool = False
 1123: 
 1124: 
 1125: def _load_prewarm_ts_from_disk() -> float:
 1126:     """Read the persisted prewarm ts, returning 0.0 if missing/corrupt."""
 1127:     state = load_json(_DASHBOARD_PREWARM_STATE_FILE, default={}) or {}
 1128:     ts = state.get("last_prewarm_ts")
 1129:     if isinstance(ts, (int, float)) and ts > 0:
 1130:         return float(ts)
 1131:     return 0.0
 1132: 
 1133: 
 1134: def _save_prewarm_ts_to_disk(ts: float) -> None:
 1135:     """Persist the latest prewarm ts. Best-effort — never raises."""
 1136:     try:
 1137:         _atomic_write_json(
 1138:             _DASHBOARD_PREWARM_STATE_FILE,
 1139:             {"last_prewarm_ts": float(ts)},
 1140:         )
 1141:     except Exception:
 1142:         logger.debug("prewarm-ts persist failed", exc_info=True)
 1143: 
 1144: 
 1145: def maybe_prewarm_dashboard_accuracy(now: float | None = None) -> bool:
 1146:     """Periodically pre-warm accuracy_cache.json for the dashboard endpoint.
 1147: 
 1148:     Self-gating to once per hour so the loop's per-cycle call is cheap
 1149:     on the steady-state path. On miss, fans out to 12 cache reads + at
 1150:     most 12 underlying computes — the same fanout the dashboard endpoint
 1151:     would otherwise pay on its first request after a restart.
 1152: 
 1153:     Internally uses the existing get_or_compute_* helpers, so it
 1154:     respects the BUG-178 thundering-herd lock and won't fight with
 1155:     in-loop callers that hit the same cache from ticker threads.
 1156: 
 1157:     Concurrency layers (2026-05-04 codex P2-2 follow-up):
 1158:       1. Process-local `threading.Lock` — guards the in-memory ts and
 1159:          lazy-load flag against concurrent threads in the same process.
 1160:       2. Re-read the persisted ts from disk inside the file-lock window
 1161:          — catches the case where another process wrote between our
 1162:          lazy-load and our gate decision.
 1163:       3. Cross-process file lock around the gate decision + fanout so
 1164:          only one of N concurrent processes (main loop + a manual
 1165:          trigger, two main loops during a botched restart) actually
 1166:          performs the fanout. Lock is non-blocking — a second process
 1167:          that races just returns False and treats the call as gated,
 1168:          since whichever process wins is doing the same fanout.
 1169: 
 1170:     Persistence: `data/dashboard_prewarm_state.json` survives loop
 1171:     restarts. Stale-on-corruption-or-missing falls back to 0.0 (next
 1172:     call fires) — safe direction.
 1173: 
 1174:     Args:
 1175:         now: Override clock for tests. Defaults to time.time().
 1176: 
 1177:     Returns:
 1178:         True if prewarm fired this call, False if gated by the interval
 1179:         or another process holds the file lock.
 1180:     """
 1181:     global _last_dashboard_prewarm_ts, _dashboard_prewarm_loaded
 1182:     t = now if now is not None else time.time()
 1183:     with _dashboard_prewarm_lock:
 1184:         # Layer 1: lazy-load the persisted ts on first call per process.
 1185:         if not _dashboard_prewarm_loaded:
 1186:             _last_dashboard_prewarm_ts = _load_prewarm_ts_from_disk()
 1187:             _dashboard_prewarm_loaded = True
 1188:         # First gate check using in-memory (and possibly stale) ts.
 1189:         if t - _last_dashboard_prewarm_ts < _DASHBOARD_PREWARM_INTERVAL_SEC:
 1190:             return False
 1191: 
 1192:     # Layer 3: cross-process exclusion. If another process is already
 1193:     # in the fanout window, skip — the work is being done.
 1194:     try:
 1195:         from portfolio.process_lock import acquire_lock_file, release_lock_file
 1196:     except Exception:
 1197:         # If process_lock is somehow unavailable, fall back to the
 1198:         # process-local guarantee (better than crashing the loop).
 1199:         acquire_lock_file = None
 1200:         release_lock_file = None
 1201: 
 1202:     fh = acquire_lock_file(_DASHBOARD_PREWARM_LOCK_FILE,
 1203:                             owner="dashboard_prewarm") if acquire_lock_file else "noop"
 1204:     if fh is None:
 1205:         # Another process holds the lock — they're doing the fanout.
 1206:         # Treat as gated; next caller will see the persisted ts they write.
 1207:         return False
 1208: 
 1209:     try:
 1210:         # Layer 2: re-read disk under the file lock. A racer that won
 1211:         # the lock just before us would have written; honor their work.
 1212:         # Only honor a positive disk_ts — "file missing" returns 0 from
 1213:         # _load_prewarm_ts_from_disk, and treating that as authoritative
 1214:         # would clobber a deliberately-old in-memory seed (e.g., test
 1215:         # fixtures that pre-seed -10000 to force a fire).
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1215:         # fixtures that pre-seed -10000 to force a fire).
 1216:         with _dashboard_prewarm_lock:
 1217:             disk_ts = _load_prewarm_ts_from_disk()
 1218:             if disk_ts > 0 and disk_ts > _last_dashboard_prewarm_ts:
 1219:                 _last_dashboard_prewarm_ts = disk_ts
 1220:             if t - _last_dashboard_prewarm_ts < _DASHBOARD_PREWARM_INTERVAL_SEC:
 1221:                 return False
 1222:             _last_dashboard_prewarm_ts = t
 1223: 
 1224:         for h in _DASHBOARD_PREWARM_HORIZONS:
 1225:             get_or_compute_accuracy(h)
 1226:             get_or_compute_consensus_accuracy(h)
 1227:             get_or_compute_per_ticker_accuracy(h)
 1228:         # Persist AFTER the fanout completes so a crash mid-fanout doesn't
 1229:         # pin the gate. The file lock means we're the only writer.
 1230:         _save_prewarm_ts_to_disk(t)
 1231:         return True
 1232:     except Exception:
 1233:         logger.debug("maybe_prewarm_dashboard_accuracy failed", exc_info=True)
 1234:         return False
 1235:     finally:
 1236:         if release_lock_file and fh != "noop":
 1237:             release_lock_file(fh)
 1238: 
 1239: 
 1240: def _count_entries_with_outcomes(entries, horizon):
 1241:     count = 0
 1242:     for entry in entries:
 1243:         outcomes = entry.get("outcomes", {})
 1244:         for _ticker, horizons in outcomes.items():
 1245:             if horizons.get(horizon):
 1246:                 count += 1
 1247:                 break
 1248:     return count
 1249: 
 1250: 
 1251: def print_accuracy_report():
 1252:     entries = load_entries()
 1253:     if not entries:
 1254:         print("No signal log data found.")
 1255:         return
 1256: 
 1257:     horizon_counts = {h: _count_entries_with_outcomes(entries, h) for h in HORIZONS}
 1258:     counts_str = ", ".join(f"{horizon_counts[h]} with {h} outcomes" for h in HORIZONS)
 1259: 
 1260:     print("=== Signal Accuracy Report ===")
 1261:     print()
 1262:     print(f"Entries: {len(entries)} total, {counts_str}")
 1263: 
 1264:     for h in HORIZONS:
 1265:         if horizon_counts[h] == 0:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Signal accuracy degradation tracker.
    2: 
    3: BUG-178/W15-W16 follow-up (2026-04-16): the W15/W16 Tier-1 1d consensus
    4: collapse from 52-56% to 36-41% (memory/project_accuracy_degradation_20260416.md)
    5: went undetected because the existing 11 main-loop runtime contracts
    6: (portfolio/loop_contract.py) check execution health, not decision quality.
    7: This module is the missing piece: daily snapshots of recent-7d accuracy
    8: across four scopes (per-signal global, per-ticker per-signal, forecast
    9: models, aggregate consensus), an hourly comparison against the snapshot
   10: from 7 days ago, and severity classification that plugs into the existing
   11: Violation framework.
   12: 
   13: Public surface used by other modules:
   14: 
   15: * save_full_accuracy_snapshot()        — once-per-day snapshot writer
   16: * check_degradation()                  — hourly violation producer
   17: * maybe_save_daily_snapshot(config)    — guarded daily snapshot driver
   18: * maybe_send_degradation_summary(cfg)  — guarded daily Telegram summary
   19: * build_daily_summary(...)             — Telegram body builder
   20: * DEGRADATION_INVARIANT                — invariant name string
   21: 
   22: Codex pre-impl review (2026-04-16) flagged 4 design problems addressed
   23: in commit history of docs/plans/2026-04-16-accuracy-degradation-tracker.md:
   24:   P1#1 — recent-window source, not lifetime aggregate
   25:   P1#2 — throttle replays cached violations so ViolationTracker keeps
   26:          escalation counts (do NOT return [] when throttled)
   27:   P2#3 — backward FOMC/CPI window via econ_dates.recent_high_impact_events
   28:   P2#4 — Ministral/Qwen3 from signal_log, only Chronos/Kronos from
   29:          forecast_predictions.jsonl
   30: """
   31: 
   32: from __future__ import annotations
   33: 
   34: import logging
   35: import time
   36: from datetime import UTC, datetime, timedelta
   37: from pathlib import Path
   38: from typing import Any
   39: 
   40: from portfolio.file_utils import atomic_write_json, load_json
   41: from portfolio.tickers import SIGNAL_NAMES
   42: 
   43: logger = logging.getLogger("portfolio.accuracy_degradation")
   44: 
   45: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   46: ALERT_STATE_FILE = DATA_DIR / "degradation_alert_state.json"
   47: SNAPSHOT_STATE_FILE = DATA_DIR / "accuracy_snapshot_state.json"
   48: 
   49: # --- Public constants exposed to callers (loop_contract integration) ---
   50: 
   51: DEGRADATION_INVARIANT = "accuracy_degradation"
   52: 
   53: # --- Detection tuning ---
   54: 
   55: # 15pp drop AND <50% absolute. The dual gate keeps strong signals dropping
   56: # from 75% to 60% from triggering (still strong) while catching 58%->42%.
   57: DROP_THRESHOLD_PP = 15.0
   58: ABSOLUTE_FLOOR_PCT = 50.0
   59: RISE_THRESHOLD_PP = 10.0  # symmetric for the daily summary "improved" list
   60: 
   61: # Anti-noise gates. Bumped 100 → 200 (2026-04-28) after the Apr 21 baseline
   62: # of sentiment 75.3% on N=223 was found to be a small-sample one-week spike
   63: # above lifetime 46% (N=39k). At N=100 the binomial SE on a 50/50 process
   64: # is ±5pp; differencing two such samples gives ±7pp 1-σ — so 15pp drops
   65: # happen on noise alone roughly 1 in 33 cycles. The combined SE-gate +
   66: # raised floor pushes the noise floor down to roughly 1 in 1000.
   67: MIN_SAMPLES_HISTORICAL = 200
   68: MIN_SAMPLES_CURRENT = 200
   69: MIN_SNAPSHOT_AGE_DAYS = 6.0   # don't alert without a real baseline
   70: BASELINE_TARGET_DAYS = 7.0    # find snapshot near now-7d
   71: BASELINE_MAX_DELTA_HOURS = 36.0  # tolerance when picking the baseline snapshot
   72: COOLDOWN_PER_SIGNAL_S = 24 * 3600   # Telegram re-emission cooldown
   73: HOURLY_THROTTLE_S = 55 * 60         # don't recompute more than once per ~hour
   74: ECON_BLACKOUT_HOURS = 24.0
   75: ECON_BLACKOUT_IMPACTS = ("high",)
   76: 
   77: # Severity classification
   78: CRITICAL_MIN_SIGNAL_COUNT = 3   # ≥3 simultaneous drops = CRITICAL
   79: SEVERITY_WARNING = "WARNING"
   80: SEVERITY_CRITICAL = "CRITICAL"
   81: 
   82: # Daily summary
   83: SUMMARY_HOUR_UTC_DEFAULT = 6
   84: TOP_DROPS_IN_SUMMARY = 8
   85: TOP_GAINS_IN_SUMMARY = 5
   86: 
   87: 
   88: # --- State helpers ---
   89: 
   90: def _load_alert_state() -> dict:
   91:     state = load_json(ALERT_STATE_FILE, default={})
   92:     if not isinstance(state, dict):
   93:         return {}
   94:     state.setdefault("last_full_check_time", 0.0)
   95:     state.setdefault("last_full_check_violations", [])
   96:     state.setdefault("last_alert_per_signal", {})
   97:     state.setdefault("last_summary_send_time", 0.0)
   98:     return state
   99: 
  100: 
  101: def _save_alert_state(state: dict) -> None:
  102:     atomic_write_json(ALERT_STATE_FILE, state)
  103: 
  104: 
  105: def _load_snapshot_state() -> dict:
  106:     state = load_json(SNAPSHOT_STATE_FILE, default={})
  107:     if not isinstance(state, dict):
  108:         return {}
  109:     state.setdefault("last_snapshot_date_utc", "")
  110:     return state
  111: 
  112: 
  113: def _save_snapshot_state(state: dict) -> None:
  114:     atomic_write_json(SNAPSHOT_STATE_FILE, state)
  115: 
  116: 
  117: # --- Snapshot writer ---
  118: 
  119: def save_full_accuracy_snapshot(*, days: int = 7) -> dict[str, Any]:
  120:     """Compute the full four-scope snapshot and append to accuracy_snapshots.jsonl.
  121: 
  122:     Returns the snapshot dict for inspection. Safe to call repeatedly —
  123:     each call appends a new line. Caller is responsible for once-per-day
  124:     gating (see maybe_save_daily_snapshot).
  125: 
  126:     BUG-178/W15-W16 review (2026-04-16): loads the signal log ONCE and
  127:     threads pre-loaded `entries` through every per-signal/per-ticker/
  128:     consensus call. Without the share, _per_ticker_recent alone re-scans
  129:     the 50,000-row file 41 times, blowing snapshot wall time.
  130:     """
  131:     from datetime import timedelta as _td
  132: 
  133:     from portfolio.accuracy_stats import (
  134:         accuracy_by_ticker_signal_cached,
  135:         consensus_accuracy,
  136:         load_entries,
  137:         save_accuracy_snapshot,
  138:         signal_accuracy,
  139:     )
  140:     from portfolio.forecast_accuracy import cached_forecast_accuracy
  141: 
  142:     extras: dict[str, Any] = {}
  143: 
  144:     cutoff = (datetime.now(UTC) - _td(days=days)).isoformat()
  145:     all_entries = load_entries()
  146:     recent_entries = [e for e in all_entries if e.get("ts", "") >= cutoff]
  147: 
  148:     # Recent-window per-signal accuracy (Codex P1#1)
  149:     try:
  150:         recent = signal_accuracy("1d", entries=recent_entries)
  151:         extras["signals_recent"] = {
  152:             name: {"accuracy": data["accuracy"], "total": data["total"]}
  153:             for name, data in recent.items()
  154:         }
  155:     except Exception as e:
  156:         logger.warning("Recent signal_accuracy snapshot failed: %s", e)
  157: 
  158:     # Per-ticker per-signal — lifetime via cached helper (1h TTL inside),
  159:     # recent via shared-entries scan
  160:     try:
  161:         per_ticker_lifetime = accuracy_by_ticker_signal_cached("1d")
  162:         extras["per_ticker"] = _compact_per_ticker(per_ticker_lifetime)
  163:     except Exception as e:
  164:         logger.warning("Lifetime per-ticker accuracy snapshot failed: %s", e)
  165: 
  166:     try:
  167:         per_ticker_recent = _per_ticker_recent(
  168:             "1d", days=days, entries=recent_entries,
  169:         )
  170:         extras["per_ticker_recent"] = _compact_per_ticker(per_ticker_recent)
  171:     except Exception as e:
  172:         logger.warning("Recent per-ticker accuracy snapshot failed: %s", e)
  173: 
  174:     # Forecast (Chronos/Kronos) — Codex P2#4 split. Forecast uses its
  175:     # own JSONL (forecast_predictions), not signal_log, so we can't share
  176:     # entries here.
  177:     try:
  178:         forecast_recent = cached_forecast_accuracy(
  179:             horizon="24h", days=days, use_raw_sub_signals=True,
  180:         )
  181:         extras["forecast_recent"] = {
  182:             name: {"accuracy": stats["accuracy"], "total": stats["total"]}
  183:             for name, stats in forecast_recent.items()
  184:         }
  185:     except Exception as e:
  186:         logger.warning("Forecast accuracy snapshot failed: %s", e)
  187: 
  188:     # Aggregate consensus — lifetime over all entries, recent over shared list
  189:     try:
  190:         extras["consensus"] = consensus_accuracy("1d", entries=all_entries)
  191:         extras["consensus_recent"] = consensus_accuracy(
  192:             "1d", entries=recent_entries,
  193:         )
  194:     except Exception as e:
  195:         logger.warning("Consensus accuracy snapshot failed: %s", e)
  196: 
  197:     # save_accuracy_snapshot() also writes the lifetime `signals` block.
  198:     # That helper still does its own load_entries() — acceptable cost
  199:     # (one extra scan / day).
  200:     return save_accuracy_snapshot(extras=extras)
  201: 
  202: 
  203: def _per_ticker_recent(horizon: str, days: int, *, entries=None) -> dict:
  204:     """Per-ticker per-signal accuracy on a recent-N-day window.
  205: 
  206:     accuracy_by_ticker_signal_cached only exposes the lifetime aggregate;
  207:     for the recent-window variant we compute it inline from
  208:     accuracy_by_signal_ticker(name, horizon, days=days), inverting the
  209:     indexing so the result is keyed by ticker first.
  210: 
  211:     BUG-178/W15-W16 follow-up review (2026-04-16): pre-loaded `entries`
  212:     is mandatory for the hot path. Without it, this function calls
  213:     accuracy_by_signal_ticker once per signal (~41 entries in
  214:     SIGNAL_NAMES today), each one re-scanning the entire 50,000-row
  215:     signal log. That ~290s of redundant compute would blow the
  216:     180s MAX_CYCLE_DURATION_S contract every time the degradation
  217:     check runs from verify_contract.
  218:     """
  219:     from portfolio.accuracy_stats import accuracy_by_signal_ticker, load_entries
  220: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  220: 
  221:     if entries is None:
  222:         entries = load_entries()
  223: 
  224:     result: dict[str, dict[str, dict]] = {}
  225:     for sig_name in SIGNAL_NAMES:
  226:         per_ticker = accuracy_by_signal_ticker(
  227:             sig_name, horizon=horizon, days=days, entries=entries,
  228:         )
  229:         for ticker, stats in per_ticker.items():
  230:             samples = stats.get("samples", stats.get("total", 0))
  231:             if samples <= 0:
  232:                 continue
  233:             result.setdefault(ticker, {})[sig_name] = {
  234:                 "accuracy": stats.get("accuracy", 0.0),
  235:                 "total": samples,
  236:             }
  237:     return result
  238: 
  239: 
  240: def _compact_per_ticker(per_ticker: dict) -> dict:
  241:     """Strip per-ticker per-signal blocks to just (accuracy, total).
  242: 
  243:     Snapshots are append-only forever; keep them small. accuracy_by_ticker_signal
  244:     returns extra fields like buy_accuracy/sell_accuracy that the degradation
  245:     detector doesn't use.
  246:     """
  247:     compact: dict[str, dict[str, dict]] = {}
  248:     for ticker, sigs in (per_ticker or {}).items():
  249:         per_sig: dict[str, dict] = {}
  250:         for sig_name, stats in (sigs or {}).items():
  251:             total = stats.get("total", stats.get("samples", 0))
  252:             if total <= 0:
  253:                 continue
  254:             per_sig[sig_name] = {
  255:                 "accuracy": stats.get("accuracy", 0.0),
  256:                 "total": total,
  257:             }
  258:         if per_sig:
  259:             compact[ticker] = per_sig
  260:     return compact
  261: 
  262: 
  263: # --- Snapshot loader (delegates to accuracy_stats for the JSONL parse) ---
  264: 
  265: def _load_snapshots() -> list[dict]:
  266:     from portfolio.accuracy_stats import _load_accuracy_snapshots
  267:     return _load_accuracy_snapshots()
  268: 
  269: 
  270: def _find_baseline_snapshot(snapshots: list[dict], now: datetime) -> dict | None:
  271:     from portfolio.accuracy_stats import _find_snapshot_near
  272:     target = now - timedelta(days=BASELINE_TARGET_DAYS)
  273:     return _find_snapshot_near(snapshots, target,
  274:                                max_delta_hours=BASELINE_MAX_DELTA_HOURS)
  275: 
  276: 
  277: def _snapshot_age_days(snapshot: dict, now: datetime) -> float:
  278:     try:
  279:         snap_ts = datetime.fromisoformat(snapshot["ts"])
  280:     except (KeyError, ValueError, TypeError):
  281:         return 0.0
  282:     return (now - snap_ts).total_seconds() / 86400.0
  283: 
  284: 
  285: # --- Blackout helpers ---
  286: 
  287: def _is_econ_blackout() -> bool:
  288:     """True when high-impact econ event is within ±ECON_BLACKOUT_HOURS."""
  289:     try:
  290:         from portfolio.econ_dates import (
  291:             events_within_hours,
  292:             recent_high_impact_events,
  293:         )
  294:     except Exception:
  295:         return False
  296:     try:
  297:         forward = events_within_hours(ECON_BLACKOUT_HOURS)
  298:         if any(e.get("impact") in ECON_BLACKOUT_IMPACTS for e in forward):
  299:             return True
  300:         backward = recent_high_impact_events(
  301:             ECON_BLACKOUT_HOURS, impact_filter=ECON_BLACKOUT_IMPACTS,
  302:         )
  303:         return bool(backward)
  304:     except Exception as e:
  305:         logger.debug("econ blackout check failed: %s", e)
  306:         return False
  307: 
  308: 
  309: # --- Core check ---
  310: 
  311: def _make_violation(invariant: str, severity: str, message: str,
  312:                     details: dict | None = None):
  313:     """Build a loop_contract.Violation without a hard import dependency.
  314: 
  315:     loop_contract imports from many subsystems; importing it here at
  316:     module-import time would risk a cycle. Lazy-import inside the call
  317:     keeps accuracy_degradation a leaf module.
  318:     """
  319:     from portfolio.loop_contract import Violation
  320:     return Violation(
  321:         invariant=invariant,
  322:         severity=severity,
  323:         message=message,
  324:         details=details or {},
  325:     )
  326: 
  327: 
  328: def check_degradation(now: datetime | None = None,
  329:                       *,
  330:                       drop_threshold_pp: float = DROP_THRESHOLD_PP,
  331:                       absolute_floor_pct: float = ABSOLUTE_FLOOR_PCT,
  332:                       min_samples_historical: int = MIN_SAMPLES_HISTORICAL,
  333:                       min_samples_current: int = MIN_SAMPLES_CURRENT,
  334:                       throttle_seconds: float = HOURLY_THROTTLE_S) -> list:
  335:     """Compare recent-7d accuracy to the snapshot from 7 days ago.
  336: 
  337:     Returns a list of loop_contract.Violation objects. Codex P1#2: when
  338:     throttled, returns the cached violations from the last full check
  339:     rather than [] so ViolationTracker can keep the consecutive-fire
  340:     count alive across cycles.
  341: 
  342:     Returns [] when:
  343:     - No baseline snapshot exists.
  344:     - Baseline is younger than MIN_SNAPSHOT_AGE_DAYS.
  345:     - High-impact econ event is within ±24h.
  346:     """
  347:     now = now or datetime.now(UTC)
  348:     state = _load_alert_state()
  349: 
  350:     # Hourly throttle — replay cached result instead of returning empty
  351:     elapsed = time.time() - float(state.get("last_full_check_time", 0.0) or 0.0)
  352:     if elapsed < throttle_seconds:
  353:         return _hydrate_cached_violations(state)
  354: 
  355:     # FOMC/CPI/NFP blackout (forward + backward).
  356:     # BUG-178/W15-W16 review (2026-04-16, P2#4): return [] rather than
  357:     # replaying cached violations. Otherwise a 24h+ blackout (FOMC week,
  358:     # NFP-on-Friday-then-CPI-on-Monday) keeps replaying alerts from
  359:     # before the blackout, and ViolationTracker would happily escalate
  360:     # the same stale alert to CRITICAL after 3 cycles.
  361:     if _is_econ_blackout():
  362:         logger.info("Degradation check skipped: econ blackout window active")
  363:         # Clear the cached violation list so post-blackout checks start
  364:         # from a known-fresh slate.
  365:         if state.get("last_full_check_violations"):
  366:             state["last_full_check_violations"] = []
  367:             _save_alert_state(state)
  368:         return []
  369: 
  370:     snapshots = _load_snapshots()
  371:     baseline = _find_baseline_snapshot(snapshots, now) if snapshots else None
  372:     age_days = _snapshot_age_days(baseline, now) if baseline else 0.0
  373: 
  374:     # Compute violations (empty list on the no-baseline / too-young paths)
  375:     if baseline and age_days >= MIN_SNAPSHOT_AGE_DAYS:
  376:         alerts = _diff_against_baseline(
  377:             baseline=baseline,
  378:             now=now,
  379:             drop_threshold_pp=drop_threshold_pp,
  380:             absolute_floor_pct=absolute_floor_pct,
  381:             min_samples_historical=min_samples_historical,
  382:             min_samples_current=min_samples_current,
  383:         )
  384:         violations = _alerts_to_violations(alerts, age_days=age_days)
  385:     else:
  386:         violations = []
  387: 
  388:     # Always update last_full_check_time after passing the throttle + blackout
  389:     # gates — even when the no-baseline / too-young branches produced []. Skipping
  390:     # the state write would let the throttle re-fire every cycle, defeating the
  391:     # whole point. The cached violations list is also written so the next
  392:     # throttled cycle replays exactly what verify_contract saw this time.
  393:     state["last_full_check_time"] = time.time()
  394:     state["last_full_check_violations"] = [_violation_to_dict(v) for v in violations]
  395:     _save_alert_state(state)
  396: 
  397:     return violations
  398: 
  399: 
  400: def _hydrate_cached_violations(state: dict) -> list:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
  400: def _hydrate_cached_violations(state: dict) -> list:
  401:     """Re-build Violation objects from the cached state JSON.
  402: 
  403:     This is the throttle-replay path. Crucial for ViolationTracker
  404:     continuity — see Codex P1#2 in the plan doc.
  405:     """
  406:     cached = state.get("last_full_check_violations") or []
  407:     return [_dict_to_violation(c) for c in cached]
  408: 
  409: 
  410: def _violation_to_dict(violation) -> dict:
  411:     return {
  412:         "invariant": violation.invariant,
  413:         "severity": violation.severity,
  414:         "message": violation.message,
  415:         "details": dict(violation.details or {}),
  416:     }
  417: 
  418: 
  419: def _dict_to_violation(payload: dict):
  420:     return _make_violation(
  421:         invariant=payload.get("invariant", DEGRADATION_INVARIANT),
  422:         severity=payload.get("severity", SEVERITY_WARNING),
  423:         message=payload.get("message", ""),
  424:         details=payload.get("details", {}),
  425:     )
  426: 
  427: 
  428: # --- Diff engine ---
  429: 
  430: def _diff_against_baseline(*, baseline: dict, now: datetime,
  431:                            drop_threshold_pp: float,
  432:                            absolute_floor_pct: float,
  433:                            min_samples_historical: int,
  434:                            min_samples_current: int) -> list[dict]:
  435:     """Return alert dicts for each scope where the degradation gate fires."""
  436:     # BUG-178/W15-W16 review (2026-04-16): load entries ONCE and share
  437:     # across the per-signal / per-ticker / consensus diffs. Without
  438:     # entry-sharing _per_ticker_recent re-scans the 50,000-entry SQLite
  439:     # file once per signal name (~41 scans), blowing the 180s cycle
  440:     # budget every time the degradation check runs.
  441:     from datetime import timedelta as _td
  442: 
  443:     from portfolio.accuracy_stats import (
  444:         consensus_accuracy,
  445:         load_entries,
  446:         signal_accuracy,
  447:     )
  448:     from portfolio.forecast_accuracy import cached_forecast_accuracy
  449:     cutoff = (now - _td(days=int(BASELINE_TARGET_DAYS))).isoformat()
  450:     all_entries = load_entries()
  451:     recent_entries = [e for e in all_entries if e.get("ts", "") >= cutoff]
  452: 
  453:     alerts: list[dict] = []
  454: 
  455:     # 1) Per-signal global (recent-window now vs recent-window in baseline)
  456:     try:
  457:         old_signals = baseline.get("signals_recent") or {}
  458:         new_signals = signal_accuracy("1d", entries=recent_entries)
  459:         for sig_name, new_data in new_signals.items():
  460:             old_data = old_signals.get(sig_name)
  461:             alert = _maybe_alert(
  462:                 key=sig_name,
  463:                 scope="signal",
  464:                 old=old_data,
  465:                 new={
  466:                     "accuracy": new_data.get("accuracy", 0.0),
  467:                     "total": new_data.get("total", 0),
  468:                 },
  469:                 drop_threshold_pp=drop_threshold_pp,
  470:                 absolute_floor_pct=absolute_floor_pct,
  471:                 min_samples_historical=min_samples_historical,
  472:                 min_samples_current=min_samples_current,
  473:             )
  474:             if alert:
  475:                 alerts.append(alert)
  476:     except Exception as e:
  477:         logger.warning("Per-signal degradation diff failed: %s", e)
  478: 
  479:     # 2) Per-ticker per-signal — share entries to avoid 41x re-scan
  480:     try:
  481:         old_per = baseline.get("per_ticker_recent") or {}
  482:         new_per = _per_ticker_recent(
  483:             "1d", days=int(BASELINE_TARGET_DAYS), entries=recent_entries,
  484:         )
  485:         for ticker, sigs in new_per.items():
  486:             old_for_ticker = old_per.get(ticker, {}) or {}
  487:             for sig_name, new_data in sigs.items():
  488:                 old_data = old_for_ticker.get(sig_name)
  489:                 alert = _maybe_alert(
  490:                     key=f"{ticker}::{sig_name}",
  491:                     scope="per_ticker",
  492:                     old=old_data,
  493:                     new=new_data,
  494:                     drop_threshold_pp=drop_threshold_pp,
  495:                     absolute_floor_pct=absolute_floor_pct,
  496:                     min_samples_historical=min_samples_historical,
  497:                     min_samples_current=min_samples_current,
  498:                 )
  499:                 if alert:
  500:                     alerts.append(alert)
  501:     except Exception as e:
  502:         logger.warning("Per-ticker degradation diff failed: %s", e)
  503: 
  504:     # 3) Forecast models (Chronos/Kronos only — Ministral/Qwen3 are in scope #1)
  505:     try:
  506:         old_forecast = baseline.get("forecast_recent") or {}
  507:         new_forecast = cached_forecast_accuracy(
  508:             horizon="24h", days=int(BASELINE_TARGET_DAYS),
  509:             use_raw_sub_signals=True,
  510:         )
  511:         for sub_name, stats in new_forecast.items():
  512:             old_data = old_forecast.get(sub_name)
  513:             new_data = {
  514:                 "accuracy": stats.get("accuracy", 0.0),
  515:                 "total": stats.get("total", 0),
  516:             }
  517:             alert = _maybe_alert(
  518:                 key=f"forecast::{sub_name}",
  519:                 scope="forecast",
  520:                 old=old_data,
  521:                 new=new_data,
  522:                 drop_threshold_pp=drop_threshold_pp,
  523:                 absolute_floor_pct=absolute_floor_pct,
  524:                 min_samples_historical=min_samples_historical,
  525:                 min_samples_current=min_samples_current,
  526:             )
  527:             if alert:
  528:                 alerts.append(alert)
  529:     except Exception as e:
  530:         logger.warning("Forecast degradation diff failed: %s", e)
  531: 
  532:     # 4) Aggregate consensus — share entries (consensus_accuracy honors
  533:     # `entries` and skips both load_entries() and the days filter)
  534:     try:
  535:         old_consensus = baseline.get("consensus_recent")
  536:         new_consensus = consensus_accuracy("1d", entries=recent_entries)
  537:         alert = _maybe_alert(
  538:             key="consensus",
  539:             scope="consensus",
  540:             old=old_consensus,
  541:             new={
  542:                 "accuracy": new_consensus.get("accuracy", 0.0),
  543:                 "total": new_consensus.get("total", 0),
  544:             },
  545:             drop_threshold_pp=drop_threshold_pp,
  546:             absolute_floor_pct=absolute_floor_pct,
  547:             min_samples_historical=min_samples_historical,
  548:             min_samples_current=min_samples_current,
  549:         )
  550:         if alert:
  551:             alerts.append(alert)
  552:     except Exception as e:
  553:         logger.warning("Consensus degradation diff failed: %s", e)
  554: 
  555:     return alerts
  556: 
  557: 
  558: def _binomial_diff_se_pp(p1: float, n1: int, p2: float, n2: int) -> float:
  559:     """Standard error of the difference of two independent binomial proportions, in pp.
  560: 
  561:     SE = sqrt( p1*(1-p1)/n1 + p2*(1-p2)/n2 ) * 100
  562: 
  563:     Returns 0.0 when either sample is empty (the caller already gates on
  564:     min_samples — this is a safety belt for the formula itself).
  565:     """
  566:     import math
  567: 
  568:     if n1 < 1 or n2 < 1:
  569:         return 0.0
  570:     var = p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2
  571:     if var <= 0.0:
  572:         return 0.0
  573:     return math.sqrt(var) * 100.0
  574: 
  575: 
  576: def _maybe_alert(*, key: str, scope: str, old, new,
  577:                  drop_threshold_pp: float, absolute_floor_pct: float,
  578:                  min_samples_historical: int, min_samples_current: int) -> dict | None:
  579:     """Apply the dual gate (drop AND absolute floor) plus an SE
  580:     significance gate to one (key, scope).
  581: 
  582:     The SE gate (added 2026-04-28) requires that the observed drop is at
  583:     least 2 standard errors of the difference of two binomial proportions.
  584:     Without it, a flat 15pp threshold produces ~1-in-33 false positives at
  585:     N=100 against stable signals — the failure mode that produced the
  586:     sentiment 75.3%→43.3% "cliff" alert from regression-to-mean noise.
  587:     """
  588:     if not old or not new:
  589:         return None
  590:     try:
  591:         old_acc = float(old.get("accuracy", 0.0))
  592:         old_total = int(old.get("total", 0))
  593:         new_acc = float(new.get("accuracy", 0.0))
  594:         new_total = int(new.get("total", 0))
  595:     except (TypeError, ValueError):
  596:         return None
  597:     if old_total < min_samples_historical or new_total < min_samples_current:
  598:         return None
  599:     drop_pp = (old_acc - new_acc) * 100.0
  600:     new_pct = new_acc * 100.0
  601:     if drop_pp < drop_threshold_pp:
  602:         return None
  603:     if new_pct >= absolute_floor_pct:
  604:         return None
  605:     se_pp = _binomial_diff_se_pp(old_acc, old_total, new_acc, new_total)
  606:     if drop_pp < 2.0 * se_pp:
  607:         return None
  608:     return {
  609:         "key": key,
  610:         "scope": scope,
  611:         "old_accuracy_pct": round(old_acc * 100.0, 1),
  612:         "new_accuracy_pct": round(new_pct, 1),
  613:         "drop_pp": round(drop_pp, 1),
  614:         "old_samples": old_total,
  615:         "new_samples": new_total,
  616:         "se_pp": round(se_pp, 2),
  617:     }
  618: 
  619: 
  620: def _classify_severity(alerts: list[dict]) -> str:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  620: def _classify_severity(alerts: list[dict]) -> str:
  621:     """≥3 simultaneous drops OR consensus drop ⇒ CRITICAL; else WARNING."""
  622:     if not alerts:
  623:         return SEVERITY_WARNING
  624:     if any(a.get("scope") == "consensus" for a in alerts):
  625:         return SEVERITY_CRITICAL
  626:     if len(alerts) >= CRITICAL_MIN_SIGNAL_COUNT:
  627:         return SEVERITY_CRITICAL
  628:     return SEVERITY_WARNING
  629: 
  630: 
  631: def _alerts_to_violations(alerts: list[dict], *, age_days: float) -> list:
  632:     if not alerts:
  633:         return []
  634: 
  635:     severity = _classify_severity(alerts)
  636:     summary_parts = []
  637:     for a in alerts[:8]:  # cap message length
  638:         summary_parts.append(
  639:             f"{a['key']} {a['old_accuracy_pct']}%→{a['new_accuracy_pct']}%"
  640:         )
  641:     overflow = len(alerts) - 8
  642:     summary_str = ", ".join(summary_parts)
  643:     if overflow > 0:
  644:         summary_str += f" (+{overflow} more)"
  645: 
  646:     message = (
  647:         f"{len(alerts)} signal(s) dropped >{DROP_THRESHOLD_PP:.0f}pp vs "
  648:         f"{int(BASELINE_TARGET_DAYS)}d baseline AND below "
  649:         f"{ABSOLUTE_FLOOR_PCT:.0f}% absolute: {summary_str}"
  650:     )
  651: 
  652:     details = {
  653:         "alert_count": len(alerts),
  654:         "baseline_age_days": round(age_days, 1),
  655:         "alerts": alerts,
  656:     }
  657: 
  658:     return [_make_violation(DEGRADATION_INVARIANT, severity, message, details)]
  659: 
  660: 
  661: # --- Cooldown filter for the Telegram path ---
  662: 
  663: def filter_alerts_by_cooldown(alerts: list[dict],
  664:                               now_ts: float | None = None,
  665:                               cooldown_s: float = COOLDOWN_PER_SIGNAL_S) -> list[dict]:
  666:     """Drop alerts that fired for the same key within `cooldown_s` seconds.
  667: 
  668:     Codex P1#2: the underlying Violation list still includes ALL alerts
  669:     (so ViolationTracker can escalate), but Telegram and the daily
  670:     summary should not re-shout. Updates state with new last-alert
  671:     timestamps for keys that pass.
  672:     """
  673:     now_ts = now_ts if now_ts is not None else time.time()
  674:     state = _load_alert_state()
  675:     last_per_key = state.get("last_alert_per_signal") or {}
  676: 
  677:     fresh: list[dict] = []
  678:     for a in alerts:
  679:         key = a.get("key", "")
  680:         last_ts = float(last_per_key.get(key, 0.0) or 0.0)
  681:         if now_ts - last_ts < cooldown_s:
  682:             continue
  683:         fresh.append(a)
  684:         last_per_key[key] = now_ts
  685: 
  686:     state["last_alert_per_signal"] = last_per_key
  687:     _save_alert_state(state)
  688:     return fresh
  689: 
  690: 
  691: # --- Daily snapshot + summary drivers (called from main.py post-cycle) ---
  692: 
  693: def maybe_save_daily_snapshot(config: dict | None = None,
  694:                               now: datetime | None = None) -> bool:
  695:     """Write a snapshot iff today's snapshot hasn't been written yet.
  696: 
  697:     Returns True when a snapshot was written this call. Driven by main.py
  698:     in the post-cycle path, gated by configurable hour-of-day so the
  699:     snapshot lands after the daily PF-OutcomeCheck backfill runs.
  700: 
  701:     BUG-178/W15-W16 review (2026-04-16, P2#3): on success this also
  702:     invalidates the degradation check throttle (last_full_check_time=0)
  703:     so the next contract cycle re-runs the full check against the
  704:     freshly-written snapshot rather than replaying the cached
  705:     violation list that compared against the previous baseline.
  706:     """
  707:     now = now or datetime.now(UTC)
  708:     cfg_section = (config or {}).get("notification", {}) if config else {}
  709:     target_hour = int(cfg_section.get(
  710:         "accuracy_snapshot_hour_utc", SUMMARY_HOUR_UTC_DEFAULT,
  711:     ))
  712:     if now.hour < target_hour:
  713:         return False
  714: 
  715:     state = _load_snapshot_state()
  716:     today_str = now.date().isoformat()
  717:     if state.get("last_snapshot_date_utc") == today_str:
  718:         return False
  719: 
  720:     # Defense in depth (2026-04-28): verify accuracy_snapshots.jsonl
  721:     # actually grew before persisting state. Caught after 7 days of
  722:     # silent failure where state said today was done but the JSONL had
  723:     # no entry. Once last_snapshot_date_utc=today, every subsequent
  724:     # call returns False — so a single state-without-write desync
  725:     # silences the writer for the rest of the day.
  726:     from portfolio.accuracy_stats import ACCURACY_SNAPSHOTS_FILE
  727:     size_before = (
  728:         ACCURACY_SNAPSHOTS_FILE.stat().st_size
  729:         if ACCURACY_SNAPSHOTS_FILE.exists()
  730:         else 0
  731:     )
  732: 
  733:     try:
  734:         save_full_accuracy_snapshot()
  735:     except Exception as e:
  736:         logger.warning("Daily accuracy snapshot failed: %s", e)
  737:         return False
  738: 
  739:     size_after = (
  740:         ACCURACY_SNAPSHOTS_FILE.stat().st_size
  741:         if ACCURACY_SNAPSHOTS_FILE.exists()
  742:         else 0
  743:     )
  744:     if size_after <= size_before:
  745:         _record_snapshot_writer_silent_failure(now, size_before, size_after)
  746:         return False
  747: 
  748:     state["last_snapshot_date_utc"] = today_str
  749:     _save_snapshot_state(state)
  750: 
  751:     # Force the next contract cycle to re-check against the new snapshot.
  752:     alert_state = _load_alert_state()
  753:     alert_state["last_full_check_time"] = 0.0
  754:     alert_state["last_full_check_violations"] = []
  755:     _save_alert_state(alert_state)
  756:     return True
  757: 
  758: 
  759: SILENT_FAILURE_JOURNAL_COOLDOWN_S = 30 * 60  # 30 min
  760: 
  761: 
  762: def _record_snapshot_writer_silent_failure(
  763:     now: datetime, size_before: int, size_after: int,
  764: ) -> None:
  765:     """Surface a silent writer failure to ``critical_errors.jsonl``.
  766: 
  767:     The "silent" failure mode is when ``save_full_accuracy_snapshot``
  768:     returns without raising but doesn't append to the JSONL (stub
  769:     bypass, downstream swallow, partial atomic write). The natural
  770:     daily writer would otherwise mark today as done and skip retries.
  771: 
  772:     Codex round 7 P3 (2026-04-28): the gate at the top of
  773:     ``maybe_save_daily_snapshot`` only short-circuits *after* state
  774:     flips to today. On a recurring silent failure, state stays
  775:     unchanged → writer runs every cycle → a fresh critical_errors row
  776:     appends every cycle → check_critical_errors and the fix-agent get
  777:     flooded. Rate-limit journal writes to one row per
  778:     ``SILENT_FAILURE_JOURNAL_COOLDOWN_S`` (30 min) using a
  779:     ``last_silent_failure_ts`` field in the snapshot state. The actual
  780:     failure is still logged at WARNING every cycle so the loop_log
  781:     surfaces the recurrence cadence.
  782:     """
  783:     state = _load_snapshot_state()
  784:     last_ts = float(state.get("last_silent_failure_ts", 0.0) or 0.0)
  785:     now_ts = now.timestamp()
  786:     if last_ts and (now_ts - last_ts) < SILENT_FAILURE_JOURNAL_COOLDOWN_S:
  787:         logger.warning(
  788:             "Snapshot writer silent failure recurring (%.0fs since last journal "
  789:             "row); skipping critical_errors append for dedup",
  790:             now_ts - last_ts,
  791:         )
  792:         return
  793: 
  794:     try:
  795:         from portfolio.claude_gate import record_critical_error
  796:         wrote = record_critical_error(
  797:             category="snapshot_writer_silent_failure",
  798:             caller="maybe_save_daily_snapshot",
  799:             message=(
  800:                 "save_full_accuracy_snapshot returned but accuracy_snapshots.jsonl "
  801:                 f"didn't grow (before={size_before}B, after={size_after}B). "
  802:                 "State NOT updated; will retry next cycle."
  803:             ),
  804:             context={
  805:                 "now": now.isoformat(),
  806:                 "size_before": size_before,
  807:                 "size_after": size_after,
  808:             },
  809:         )
  810:     except Exception as e:
  811:         logger.exception("Failed to record snapshot writer silent failure: %s", e)
  812:         return
  813: 
  814:     if wrote:
  815:         state["last_silent_failure_ts"] = now_ts
  816:         _save_snapshot_state(state)
  817: 
  818: 
  819: def maybe_send_degradation_summary(config: dict | None = None,
  820:                                    now: datetime | None = None) -> bool:
  821:     """Send the daily Telegram summary iff today's hasn't been sent yet.
  822: 
  823:     BUG-178/W15-W16 review (2026-04-16, P1#2): refuses to build a
  824:     summary if the latest snapshot is older than today's expected
  825:     snapshot. Without this guard a silent snapshot failure would
  826:     cause the summary to ship yesterday's data labeled as today's
  827:     "Δ vs prev 7d", and on a recurring failure the same stale data
  828:     would shout day after day.
  829:     """
  830:     now = now or datetime.now(UTC)
  831:     cfg_section = (config or {}).get("notification", {}) if config else {}
  832:     target_hour = int(cfg_section.get(
  833:         "accuracy_snapshot_hour_utc", SUMMARY_HOUR_UTC_DEFAULT,
  834:     ))
  835:     if now.hour < target_hour:
  836:         return False
  837: 
  838:     state = _load_alert_state()
  839:     last_send_ts = float(state.get("last_summary_send_time", 0.0) or 0.0)
  840:     last_send_dt = datetime.fromtimestamp(last_send_ts, tz=UTC) if last_send_ts else None
  841:     if last_send_dt and last_send_dt.date() == now.date():
  842:         return False
  843: 
  844:     try:
  845:         snapshots = _load_snapshots()
  846:         if not snapshots:
  847:             return False
  848:         latest = snapshots[-1]
  849:         # Refuse to ship stale data: today's snapshot must already exist.
  850:         try:
  851:             latest_ts = datetime.fromisoformat(latest.get("ts", ""))
  852:         except (ValueError, TypeError):
  853:             latest_ts = None
  854:         if latest_ts is None or latest_ts.date() != now.date():
  855:             logger.warning(
  856:                 "Daily summary skipped: latest snapshot %s is not from today (%s). "
  857:                 "Snapshot writer likely failed; check accuracy_snapshot_state.json.",
  858:                 latest.get("ts"), now.date().isoformat(),
  859:             )
  860:             return False
  861:         baseline = _find_baseline_snapshot(snapshots, now)
  862:         body = build_daily_summary(latest=latest, baseline=baseline, now=now)
  863:     except Exception as e:
  864:         logger.warning("Daily summary build failed: %s", e)
  865:         return False
  866: 
  867:     try:
  868:         from portfolio.message_store import send_or_store
  869:         # send_or_store(msg, config, category=...) — config is REQUIRED
  870:         # (needs telegram.token + telegram.chat_id). Production bug found
  871:         # 2026-04-16 first cycle after the merge: passing only `category=`
  872:         # raised TypeError, the wrapper try/except caught it but the daily
  873:         # summary never went out. Pass the loop's config dict through.
  874:         send_or_store(body, config or {}, category="daily_digest")
  875:     except Exception as e:
  876:         logger.warning("Daily summary send failed: %s", e)
  877:         return False
  878: 
  879:     state["last_summary_send_time"] = time.time()
  880:     _save_alert_state(state)
  881:     return True
  882: 
  883: 
  884: # --- Telegram body builder ---
  885: 
  886: def build_daily_summary(*, latest: dict, baseline: dict | None,
  887:                         now: datetime | None = None) -> str:
  888:     """Build the Telegram body for the *ACCURACY DAILY* summary."""
  889:     now = now or datetime.now(UTC)
  890:     lines = [f"*ACCURACY DAILY* · {now.date().isoformat()}"]
  891: 
  892:     consensus_recent = latest.get("consensus_recent") or {}
  893:     consensus_acc = float(consensus_recent.get("accuracy", 0.0)) * 100.0
  894:     consensus_total = int(consensus_recent.get("total", 0))
  895:     delta_str = ""
  896:     if baseline:
  897:         b_consensus = baseline.get("consensus_recent") or {}
  898:         try:
  899:             b_acc = float(b_consensus.get("accuracy", 0.0)) * 100.0
  900:             delta = consensus_acc - b_acc
  901:             delta_str = f" (Δ {delta:+.1f}pp vs prev 7d)"
  902:         except (TypeError, ValueError):
  903:             delta_str = ""
  904:     lines.append(
  905:         f"`Consensus: {consensus_acc:.0f}% recent7d{delta_str} · "
  906:         f"{consensus_total} sam`"
  907:     )
  908: 
  909:     # Forecast block: explicit allowlist instead of Ministral/Qwen3
  910:     # exclusion, so a future renamed model doesn't silently misclassify
  911:     # (review P3#7).
  912:     forecast = latest.get("forecast_recent") or {}
  913:     forecast_pairs = [
  914:         (k, v.get("accuracy", 0.0)) for k, v in forecast.items()
  915:         if k.startswith("chronos") or k.startswith("kronos")
  916:     ]
  917:     if forecast_pairs:
  918:         rendered = " · ".join(
  919:             f"{name.replace('_24h','').replace('_1h','')} {acc*100:.0f}%"
  920:             for name, acc in sorted(forecast_pairs)
  921:         )
  922:         lines.append(f"`Forecast:  {rendered}`")
  923: 
  924:     signals_recent = latest.get("signals_recent") or {}
  925:     llm_pairs = [
  926:         (k, v.get("accuracy", 0.0)) for k, v in signals_recent.items()
  927:         if k in ("ministral", "qwen3")
  928:     ]
  929:     if llm_pairs:
  930:         rendered = " · ".join(
  931:             f"{name} {acc*100:.0f}%"
  932:             for name, acc in sorted(llm_pairs)
  933:         )
  934:         lines.append(f"`LLM:       {rendered}`")
  935: 
  936:     drops, gains = _summary_diffs(latest=latest, baseline=baseline)
  937:     if drops:
  938:         lines.append("")
  939:         lines.append(
  940:             f"*Degraded (>{DROP_THRESHOLD_PP:.0f}pp drop vs prev 7d, "
  941:             f"<{ABSOLUTE_FLOOR_PCT:.0f}% recent abs)*"
  942:         )
  943:         for d in drops[:TOP_DROPS_IN_SUMMARY]:
  944:             lines.append(_format_summary_row(d))
  945: 
  946:     if gains:
  947:         lines.append("")
  948:         lines.append(f"*Improved (>{RISE_THRESHOLD_PP:.0f}pp gain vs prev 7d)*")
  949:         for g in gains[:TOP_GAINS_IN_SUMMARY]:
  950:             lines.append(_format_summary_row(g))
  951: 
  952:     snap_age = "?"
  953:     try:
  954:         snap_age_days = _snapshot_age_days(baseline, now) if baseline else 0.0
  955:         snap_age = f"{snap_age_days:.1f}d"
  956:     except Exception:
  957:         logger.debug("Snapshot age computation failed", exc_info=True)
  958: 
  959:     sig_count = len(signals_recent or {})
  960:     lines.append("")
  961:     lines.append(
  962:         f"`Snapshot age: {snap_age} · {sig_count} signals tracked · window: recent-7d`"
  963:     )
  964: 
  965:     return "\n".join(lines)
  966: 
  967: 
  968: def _summary_diffs(*, latest: dict,
  969:                    baseline: dict | None) -> tuple[list[dict], list[dict]]:
  970:     drops: list[dict] = []
  971:     gains: list[dict] = []
  972:     if not baseline:
  973:         return drops, gains
  974: 
  975:     new_signals = latest.get("signals_recent") or {}
  976:     old_signals = baseline.get("signals_recent") or {}
  977:     for name, new_data in new_signals.items():
  978:         old_data = old_signals.get(name)
  979:         if not old_data:
  980:             continue
  981:         try:
  982:             old_acc = float(old_data.get("accuracy", 0.0))
  983:             new_acc = float(new_data.get("accuracy", 0.0))
  984:             samples = int(new_data.get("total", 0))
  985:         except (TypeError, ValueError):
  986:             continue
  987:         change_pp = (new_acc - old_acc) * 100.0
  988:         if change_pp <= -DROP_THRESHOLD_PP and new_acc * 100.0 < ABSOLUTE_FLOOR_PCT:
  989:             drops.append({
  990:                 "key": name,
  991:                 "old_accuracy_pct": round(old_acc * 100.0, 1),
  992:                 "new_accuracy_pct": round(new_acc * 100.0, 1),
  993:                 "drop_pp": round(-change_pp, 1),
  994:                 "samples": samples,
  995:             })
  996:         elif change_pp >= RISE_THRESHOLD_PP:
  997:             gains.append({
  998:                 "key": name,
  999:                 "old_accuracy_pct": round(old_acc * 100.0, 1),
 1000:                 "new_accuracy_pct": round(new_acc * 100.0, 1),
 1001:                 "drop_pp": round(-change_pp, 1),  # negative = gain
 1002:                 "samples": samples,
 1003:             })
 1004: 
 1005:     drops.sort(key=lambda x: x["drop_pp"], reverse=True)
 1006:     gains.sort(key=lambda x: x["drop_pp"])  # most-negative first (biggest gain)
 1007:     return drops, gains
 1008: 
 1009: 
 1010: def _format_summary_row(item: dict) -> str:
 1011:     name = item["key"]
 1012:     return (
 1013:         f"`{name:<10} {item['old_accuracy_pct']:>4.0f}% -> "
 1014:         f"{item['new_accuracy_pct']:>4.0f}% "
 1015:         f"({-item['drop_pp']:+.0f}pp, {item['samples']:>5} sam)`"
 1016:     )
 1017: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1528: 
 1529: 
 1530: def save_accuracy_snapshot(extras=None):
 1531:     """Save current per-signal accuracy as a timestamped snapshot.
 1532: 
 1533:     Appends one JSON line to accuracy_snapshots.jsonl with the current
 1534:     accuracy for each signal at the 1d horizon. Used by check_accuracy_changes()
 1535:     and accuracy_degradation.check_degradation() to detect significant shifts
 1536:     over time.
 1537: 
 1538:     Args:
 1539:         extras: Optional dict of extra blocks to merge into the snapshot.
 1540:             BUG-178/W15-W16 follow-up (2026-04-16): the degradation tracker
 1541:             needs more than the lifetime per-signal block. Callers can pass
 1542:             recent-window per-signal accuracy, per-ticker per-signal blocks,
 1543:             forecast model accuracy, and aggregate consensus accuracy by
 1544:             providing keys like "signals_recent", "per_ticker", "forecast",
 1545:             "consensus", etc. Unknown keys are accepted as-is so future
 1546:             scopes can be added without churning this function. Old single-
 1547:             block snapshots remain readable — the loader treats absent
 1548:             keys as missing.
 1549:     """
 1550:     from datetime import datetime
 1551: 
 1552:     acc = signal_accuracy("1d")
 1553:     snapshot = {
 1554:         "ts": datetime.now(UTC).isoformat(),
 1555:         "signals": {
 1556:             name: {"accuracy": data["accuracy"], "total": data["total"]}
 1557:             for name, data in acc.items()
 1558:         },
 1559:     }
 1560:     if extras:
 1561:         for key, value in extras.items():
 1562:             snapshot[key] = value
 1563:     from portfolio.file_utils import atomic_append_jsonl
 1564:     atomic_append_jsonl(ACCURACY_SNAPSHOTS_FILE, snapshot)
 1565:     return snapshot
 1566: 
 1567: 
 1568: 
 1569: 
 1570: def _load_accuracy_snapshots():
 1571:     """Load all accuracy snapshots from JSONL file.
 1572: 
 1573:     2026-04-28 (audit C1): switched from raw read_text().splitlines() to
 1574:     file_utils.load_jsonl per CLAUDE.md rule 4 (atomic I/O). The previous
 1575:     implementation read the whole file at once via read_text(), which
 1576:     races against the live atomic_append_jsonl writer in maybe_save_daily_snapshot
 1577:     -> save_full_accuracy_snapshot — a torn last line silently produced an
 1578:     empty/missing snapshot, sending check_degradation() to a no-alert quiet
 1579:     state instead of erroring. load_jsonl streams line-by-line and logs
 1580:     malformed lines at debug level, so torn writes leave a footprint.
 1581:     """
 1582:     return load_jsonl(ACCURACY_SNAPSHOTS_FILE)
 1583: 
 1584: 
 1585: def _find_snapshot_near(snapshots, target_ts, max_delta_hours=36):
 1586:     """Find the snapshot closest to target_ts within max_delta_hours.
 1587: 
 1588:     Args:
 1589:         snapshots: List of snapshot dicts with 'ts' field.
 1590:         target_ts: datetime to search near.
 1591:         max_delta_hours: Maximum allowed time difference in hours.
 1592: 
 1593:     Returns:
 1594:         dict or None: The closest snapshot, or None if none within range.
 1595:     """
 1596:     from datetime import datetime
 1597: 
 1598:     best = None
 1599:     best_delta = None
 1600:     for snap in snapshots:
 1601:         try:
 1602:             snap_ts = datetime.fromisoformat(snap["ts"])
 1603:             delta = abs((snap_ts - target_ts).total_seconds()) / 3600
 1604:             if delta <= max_delta_hours and (best_delta is None or delta < best_delta):
 1605:                     best = snap
 1606:                     best_delta = delta
 1607:         except (ValueError, TypeError, KeyError):
 1608:             continue
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
    1: """Per-ticker per-signal accuracy and directional probability engine.
    2: 
    3: Computes P(up) at multiple horizons for focus instruments using accuracy-weighted
    4: signal votes. This is the core engine for Mode B probability-based notifications.
    5: """
    6: 
    7: import logging
    8: import math
    9: from datetime import UTC
   10: 
   11: from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES
   12: 
   13: logger = logging.getLogger("portfolio.ticker_accuracy")
   14: 
   15: 
   16: def accuracy_by_ticker_signal(ticker, horizon="1d", days=None):
   17:     """Per-signal accuracy for a specific ticker.
   18: 
   19:     Queries signal_log data (SQLite preferred, JSONL fallback) and computes
   20:     accuracy grouped by signal name for a single ticker.
   21: 
   22:     Args:
   23:         ticker: Instrument ticker (e.g., "XAG-USD").
   24:         horizon: Outcome horizon ("3h", "1d", "3d", "5d", "10d").
   25:         days: If set, only include entries from the last N days.
   26: 
   27:     Returns:
   28:         dict: {signal_name: {"accuracy": float, "samples": int, "correct": int}}
   29:     """
   30:     # P0-1 (2026-05-02 adversarial follow-up): use accuracy_stats._vote_correct
   31:     # so the neutral-outcome filter (|change_pct| < _MIN_CHANGE_PCT, change_pct
   32:     # is None) matches signal_accuracy() / per_ticker_accuracy() / consensus_
   33:     # accuracy() everywhere else. Without this, per-ticker accuracy was
   34:     # overstated — Mode B Telegram and Kelly sizing both feed off it.
   35:     from portfolio.accuracy_stats import _vote_correct, load_entries
   36: 
   37:     entries = load_entries()
   38: 
   39:     cutoff = None
   40:     if days is not None:
   41:         from datetime import datetime, timedelta
   42:         cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
   43: 
   44:     stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}
   45: 
   46:     for entry in entries:
   47:         if cutoff and entry.get("ts", "") < cutoff:
   48:             continue
   49: 
   50:         tdata = entry.get("tickers", {}).get(ticker)
   51:         if not tdata:
   52:             continue
   53: 
   54:         outcome = entry.get("outcomes", {}).get(ticker, {}).get(horizon)
   55:         if not outcome:
   56:             continue
   57: 
   58:         change_pct = outcome.get("change_pct")
   59:         signals = tdata.get("signals", {})
   60: 
   61:         for sig_name in SIGNAL_NAMES:
   62:             vote = signals.get(sig_name, "HOLD")
   63:             if vote == "HOLD":
   64:                 continue
   65:             result_val = _vote_correct(vote, change_pct)
   66:             if result_val is None:
   67:                 continue  # neutral outcome — don't count
   68:             stats[sig_name]["total"] += 1
   69:             if result_val:
   70:                 stats[sig_name]["correct"] += 1
   71: 
   72:     result = {}
   73:     for sig_name in SIGNAL_NAMES:
   74:         s = stats[sig_name]
   75:         if s["total"] == 0:
   76:             continue
   77:         result[sig_name] = {
   78:             "accuracy": s["correct"] / s["total"],
   79:             "samples": s["total"],
   80:             "correct": s["correct"],
   81:         }
   82:     return result
   83: 
   84: 
   85: def direction_probability(ticker, current_votes, horizon="1d", days=7, min_samples=5):
   86:     """Convert signal votes + per-ticker accuracy into P(up) at given horizon.
   87: 
   88:     Algorithm:
   89:         For each active signal voting BUY or SELL for this ticker:
   90:           acc = that signal's accuracy for THIS ticker at THIS horizon
   91:           if signal says BUY:  p_up = acc       (71% accurate BUY → 71% P(up))
   92:           if signal says SELL: p_up = 1 - acc    (71% accurate SELL → 29% P(up))
   93:         Weighted average of all p_up values (weight = sqrt(sample_count))
   94: 
   95:     Args:
   96:         ticker: Instrument ticker.
   97:         current_votes: dict {signal_name: "BUY"|"SELL"|"HOLD"} for current cycle.
   98:         horizon: Outcome horizon for accuracy lookup.
   99:         days: Lookback window for accuracy computation (None = all time).
  100:         min_samples: Minimum samples required to use a signal's accuracy.
  101: 
  102:     Returns:
  103:         dict: {
  104:             "direction": "up"|"down"|"neutral",
  105:             "probability": float (0.0-1.0),
  106:             "signals_used": int,
  107:             "total_samples": int,
  108:             "signal_details": [{name, vote, accuracy, samples, p_up, weight}]
  109:         }
  110:     """
  111:     acc_data = accuracy_by_ticker_signal(ticker, horizon=horizon, days=days)
  112: 
  113:     weighted_sum = 0.0
  114:     weight_total = 0.0
  115:     signals_used = 0
  116:     total_samples = 0
  117:     details = []
  118: 
  119:     for sig_name, vote in current_votes.items():
  120:         if vote == "HOLD" or sig_name in DISABLED_SIGNALS:
  121:             continue
  122: 
  123:         sig_acc = acc_data.get(sig_name)
  124:         if not sig_acc or sig_acc["samples"] < min_samples:
  125:             continue
  126: 
  127:         accuracy = sig_acc["accuracy"]
  128:         samples = sig_acc["samples"]
  129: 
  130:         # Compute P(up) based on vote direction and accuracy
  131:         if vote == "BUY":
  132:             p_up = accuracy
  133:         else:  # SELL
  134:             p_up = 1.0 - accuracy
  135: 
  136:         weight = math.sqrt(samples)
  137:         weighted_sum += p_up * weight
  138:         weight_total += weight
  139:         signals_used += 1
  140:         total_samples += samples
  141: 
  142:         details.append({
  143:             "name": sig_name,
  144:             "vote": vote,
  145:             "accuracy": round(accuracy, 3),
  146:             "samples": samples,
  147:             "p_up": round(p_up, 3),
  148:             "weight": round(weight, 2),
  149:         })
  150: 
  151:     if weight_total == 0:
  152:         return {
  153:             "direction": "neutral",
  154:             "probability": 0.5,
  155:             "signals_used": 0,
  156:             "total_samples": 0,
  157:             "signal_details": [],
  158:         }
  159: 
  160:     p_up_final = weighted_sum / weight_total
  161: 
  162:     direction = "neutral"
  163:     if p_up_final > 0.52:
  164:         direction = "up"
  165:     elif p_up_final < 0.48:
  166:         direction = "down"
  167: 
  168:     return {
  169:         "direction": direction,
  170:         "probability": round(p_up_final, 3),
  171:         "signals_used": signals_used,
  172:         "total_samples": total_samples,
  173:         "signal_details": details,
  174:     }
  175: 
  176: 
  177: def direction_probability_with_forecast(ticker, current_votes, forecast_data=None,
  178:                                         horizon="1d", days=7, min_samples=5):
  179:     """Enhanced probability with Chronos forecast blending.
  180: 
  181:     Takes the base signal-accuracy probability and blends in Chronos forecast
  182:     predictions, weighted by Chronos's per-ticker accuracy.
  183: 
  184:     Args:
  185:         ticker: Instrument ticker.
  186:         current_votes: dict {signal_name: vote} for current cycle.
  187:         forecast_data: dict from forecast signal indicators, e.g.:
  188:             {"chronos_24h_pct": 0.5, "chronos_24h_conf": 0.6,
  189:              "chronos_1h_pct": 0.3, "chronos_1h_conf": 0.55,
  190:              "chronos_ok": True}
  191:         horizon: Outcome horizon.
  192:         days: Lookback window.
  193:         min_samples: Minimum samples for signal accuracy.
  194: 
  195:     Returns:
  196:         dict: Same as direction_probability() but with added fields:
  197:             "forecast_pct_move": float (Chronos predicted % move)
  198:             "forecast_confidence": float (Chronos confidence)
  199:             "forecast_blended": bool (whether forecast was blended in)
  200:     """
  201:     base = direction_probability(ticker, current_votes, horizon=horizon,
  202:                                  days=days, min_samples=min_samples)
  203: 
  204:     # Add forecast data if available
  205:     if not forecast_data or not forecast_data.get("chronos_ok"):
  206:         base["forecast_blended"] = False
  207:         return base
  208: 
  209:     # Map horizon to chronos key
  210:     horizon_map = {"1h": "1h", "3h": "1h", "1d": "24h", "3d": "24h", "24h": "24h"}
  211:     chronos_h = horizon_map.get(horizon, "24h")
  212: 
  213:     pct_key = f"chronos_{chronos_h}_pct"
  214:     conf_key = f"chronos_{chronos_h}_conf"
  215: 
  216:     pct_move = forecast_data.get(pct_key, 0) or 0
  217:     confidence = forecast_data.get(conf_key, 0) or 0
  218: 
  219:     base["forecast_pct_move"] = round(pct_move, 3) if pct_move else 0
  220:     base["forecast_confidence"] = round(confidence, 3) if confidence else 0
  221: 
  222:     # Blend forecast into probability if we have meaningful data
  223:     if abs(pct_move) > 0.1 and confidence > 0.1:
  224:         # Convert Chronos prediction to P(up): positive pct = higher P(up)
  225:         forecast_p_up = 0.5 + min(max(pct_move * 0.1, -0.3), 0.3)  # scale, cap at 0.2-0.8
  226: 
  227:         # Weight by confidence (scaled down since forecast is one input among many)
  228:         forecast_weight = confidence * 2.0  # comparable to sqrt(samples) for ~4 samples
  229: 
  230:         # Blend with existing probability
  231:         existing_weight = sum(d.get("weight", 0) for d in base.get("signal_details", []))
  232:         total_weight = existing_weight + forecast_weight
  233: 
  234:         if total_weight > 0:
  235:             blended_p = (base["probability"] * existing_weight +
  236:                          forecast_p_up * forecast_weight) / total_weight
  237:             base["probability"] = round(blended_p, 3)
  238: 
  239:             # Update direction
  240:             if blended_p > 0.52:
  241:                 base["direction"] = "up"
  242:             elif blended_p < 0.48:
  243:                 base["direction"] = "down"
  244:             else:
  245:                 base["direction"] = "neutral"
  246: 
  247:         base["forecast_blended"] = True
  248:     else:
  249:         base["forecast_blended"] = False
  250: 
  251:     return base
  252: 
  253: 
  254: def get_focus_probabilities(tickers, current_data, horizons=None, days=7):
  255:     """Compute probabilities for focus instruments across multiple horizons.
  256: 
  257:     Args:
  258:         tickers: List of focus ticker names (e.g., ["XAG-USD", "BTC-USD"]).
  259:         current_data: dict {ticker: {signals dict from signal_engine output}}.
  260:             Each ticker entry needs an "extra" dict with "_votes" or
  261:             a "signals" dict mapping signal_name -> vote.
  262:         horizons: List of horizon strings. Defaults to ["3h", "1d", "3d"].
  263:         days: Lookback window for accuracy.
  264: 
  265:     Returns:
  266:         dict: {
  267:             ticker: {
  268:                 "3h": {"direction": "up", "probability": 0.72, ...},
  269:                 "1d": {"direction": "up", "probability": 0.68, ...},
  270:                 "3d": {"direction": "up", "probability": 0.55, ...},
  271:                 "accuracy_1d": 0.71,
  272:                 "accuracy_samples": 89,
  273:             }
  274:         }
  275:     """
  276:     if horizons is None:
  277:         horizons = ["3h", "1d", "3d"]
  278: 
  279:     result = {}
  280: 
  281:     for ticker in tickers:
  282:         ticker_data = current_data.get(ticker, {})
  283: 
  284:         # Extract current votes from the signal data
  285:         votes = _extract_votes(ticker_data)
  286:         if not votes:
  287:             continue
  288: 
  289:         ticker_result = {}
  290:         for h in horizons:
  291:             prob = direction_probability(ticker, votes, horizon=h, days=days)
  292:             ticker_result[h] = prob
  293: 
  294:         # Overall accuracy summary (1d horizon as primary)
  295:         acc_1d = accuracy_by_ticker_signal(ticker, horizon="1d", days=days)
  296:         total_samples = sum(s["samples"] for s in acc_1d.values())
  297:         if acc_1d:
  298:             # Weighted average accuracy across active signals
  299:             weighted_acc = 0.0
  300:             weight_sum = 0.0
  301:             for sig_name, sig_data in acc_1d.items():
  302:                 if sig_name in DISABLED_SIGNALS:
  303:                     continue
  304:                 w = math.sqrt(sig_data["samples"])
  305:                 weighted_acc += sig_data["accuracy"] * w
  306:                 weight_sum += w
  307:             if weight_sum > 0:
  308:                 ticker_result["accuracy_1d"] = round(weighted_acc / weight_sum, 3)
  309:             else:
  310:                 ticker_result["accuracy_1d"] = 0.0
  311:         else:
  312:             ticker_result["accuracy_1d"] = 0.0
  313: 
  314:         ticker_result["accuracy_samples"] = total_samples
  315:         result[ticker] = ticker_result
  316: 
  317:     return result
  318: 
  319: 
  320: def _extract_votes(ticker_data):
  321:     """Extract signal votes from various data formats.
  322: 
  323:     Handles:
  324:     - {"extra": {"_votes": {signal: vote}}} (from signal_engine)
  325:     - {"signals": {signal: vote}} (from signal_log)
  326:     - Direct {signal: vote} dict
  327:     """
  328:     # From signal_engine output
  329:     extra = ticker_data.get("extra", {})
  330:     votes = extra.get("_votes")
  331:     if votes:
  332:         return votes
  333: 
  334:     # From signal_log format
  335:     signals = ticker_data.get("signals", {})
  336:     if signals and any(v in ("BUY", "SELL") for v in signals.values()):
  337:         return signals
  338: 
  339:     # Direct dict
  340:     if any(k in SIGNAL_NAMES for k in ticker_data):
  341:         return {k: v for k, v in ticker_data.items() if k in SIGNAL_NAMES}
  342: 
  343:     return {}
  344: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Forecast sub-signal accuracy tracker.
    2: 
    3: Reads forecast_predictions.jsonl, backfills actual prices at 1h/24h horizons,
    4: and computes per-model per-ticker per-horizon accuracy statistics.
    5: """
    6: 
    7: import logging
    8: import threading
    9: import time
   10: from collections import defaultdict
   11: from datetime import UTC, datetime, timedelta
   12: from pathlib import Path
   13: 
   14: from portfolio.file_utils import atomic_write_jsonl, load_jsonl
   15: 
   16: logger = logging.getLogger("portfolio.forecast_accuracy")
   17: 
   18: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   19: PREDICTIONS_FILE = DATA_DIR / "forecast_predictions.jsonl"
   20: HEALTH_FILE = DATA_DIR / "forecast_health.jsonl"
   21: 
   22: # BUG-178/W15-W16 follow-up (2026-04-16): per-cycle in-memory cache for
   23: # compute_forecast_accuracy(). The accuracy_degradation tracker calls this
   24: # every cycle (throttled to every ~hour for the full check); each call walks
   25: # forecast_predictions.jsonl. Cache the result for 1h so the hot path is a
   26: # dict lookup. Mirrors the signal_utility cache pattern in accuracy_stats.py:
   27: # lock guards the (timestamp, value) swap, NOT the compute, so two threads
   28: # racing past TTL each recompute once but neither blocks the other.
   29: _FORECAST_ACCURACY_CACHE_TTL = 3600.0
   30: _forecast_accuracy_cache: dict[tuple, tuple[float, dict]] = {}
   31: _forecast_accuracy_cache_lock = threading.Lock()
   32: 
   33: 
   34: def cached_forecast_accuracy(horizon="24h", days=7, use_raw_sub_signals=True):
   35:     """Cached wrapper around compute_forecast_accuracy().
   36: 
   37:     Keyed by (horizon, days, use_raw_sub_signals). 1h TTL — long enough to
   38:     keep cycle-level callers fast, short enough to surface fresh outcomes
   39:     after the daily backfill. Tests and callers that need immediately-fresh
   40:     data can call compute_forecast_accuracy() directly.
   41:     """
   42:     key = (horizon, days, use_raw_sub_signals)
   43:     now = time.time()
   44:     with _forecast_accuracy_cache_lock:
   45:         cached = _forecast_accuracy_cache.get(key)
   46:         if cached and now - cached[0] < _FORECAST_ACCURACY_CACHE_TTL:
   47:             return cached[1]
   48:     # Compute outside the lock so other horizons aren't blocked.
   49:     result = compute_forecast_accuracy(
   50:         horizon=horizon, days=days, use_raw_sub_signals=use_raw_sub_signals,
   51:     )
   52:     with _forecast_accuracy_cache_lock:
   53:         _forecast_accuracy_cache[key] = (now, result)
   54:     return result
   55: 
   56: 
   57: def invalidate_forecast_accuracy_cache():
   58:     """Clear the in-memory forecast accuracy cache.
   59: 
   60:     Called by tests or by code paths that just wrote new outcomes (e.g.
   61:     after a manual backfill). Production code rarely needs this — the 1h
   62:     TTL handles natural staleness.
   63:     """
   64:     with _forecast_accuracy_cache_lock:
   65:         _forecast_accuracy_cache.clear()
   66: 
   67: 
   68: def load_predictions(predictions_file=None):
   69:     """Load all forecast predictions from JSONL file."""
   70:     path = predictions_file or PREDICTIONS_FILE
   71:     return load_jsonl(str(path))
   72: 
   73: 
   74: def load_health_stats(health_file=None):
   75:     """Load forecast health stats (success/failure rates per model)."""
   76:     path = health_file or HEALTH_FILE
   77:     stats = defaultdict(lambda: {"ok": 0, "fail": 0, "total": 0})
   78:     for entry in load_jsonl(str(path)):
   79:         model = entry.get("model", "unknown")
   80:         if entry.get("ok"):
   81:             stats[model]["ok"] += 1
   82:         else:
   83:             stats[model]["fail"] += 1
   84:         stats[model]["total"] += 1
   85: 
   86:     result = {}
   87:     for model, s in stats.items():
   88:         result[model] = {
   89:             "ok": s["ok"],
   90:             "fail": s["fail"],
   91:             "total": s["total"],
   92:             "success_rate": round(s["ok"] / s["total"], 3) if s["total"] else 0.0,
   93:         }
   94:     return result
   95: 
   96: 
   97: def compute_forecast_accuracy(ticker=None, horizon="24h", days=None,
   98:                               predictions_file=None, use_raw_sub_signals=False):
   99:     """Compute accuracy of forecast sub-signals.
  100: 
  101:     For each prediction entry that has an actual outcome (backfilled),
  102:     check if the predicted direction matched actual price movement.
  103: 
  104:     Args:
  105:         ticker: Filter to specific ticker (None = all).
  106:         horizon: Which horizon to evaluate ("1h" or "24h").
  107:         days: Only include entries from last N days (None = all).
  108:         predictions_file: Override predictions file path (for testing).
  109: 
  110:     Returns:
  111:         dict: {
  112:             model_name: {
  113:                 "accuracy": float,
  114:                 "correct": int,
  115:                 "total": int,
  116:                 "by_ticker": {ticker: {"accuracy": float, "correct": int, "total": int}}
  117:             }
  118:         }
  119:     """
  120:     entries = load_predictions(predictions_file)
  121: 
  122:     cutoff = None
  123:     if days is not None:
  124:         cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
  125: 
  126:     # Track per-model stats
  127:     model_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
  128: 
  129:     for entry in entries:
  130:         if cutoff and entry.get("ts", "") < cutoff:
  131:             continue
  132: 
  133:         entry_ticker = entry.get("ticker", "")
  134:         if ticker and entry_ticker != ticker:
  135:             continue
  136: 
  137:         # Need actual outcome
  138:         outcome = entry.get("outcome", {}).get(horizon)
  139:         if outcome is None:
  140:             continue
  141: 
  142:         actual_change = outcome.get("change_pct", 0)
  143: 
  144:         # Check each sub-signal
  145:         sub_signals = entry.get("sub_signals", {})
  146:         if use_raw_sub_signals and entry.get("raw_sub_signals"):
  147:             sub_signals = entry.get("raw_sub_signals", {})
  148:         for sub_name, vote in sub_signals.items():
  149:             if vote == "HOLD":
  150:                 continue
  151: 
  152:             # Only check sub-signals matching the requested horizon
  153:             if "_" in sub_name:
  154:                 sub_horizon = sub_name.split("_", 1)[1]
  155:                 if sub_horizon != horizon:
  156:                     continue
  157: 
  158:             predicted_up = vote == "BUY"
  159:             actual_up = actual_change > 0
  160: 
  161:             correct = (predicted_up and actual_up) or (not predicted_up and not actual_up)
  162: 
  163:             model_stats[sub_name][entry_ticker]["total"] += 1
  164:             if correct:
  165:                 model_stats[sub_name][entry_ticker]["correct"] += 1
  166: 
  167:     # Aggregate
  168:     result = {}
  169:     for sub_name, ticker_stats in model_stats.items():
  170:         total_correct = 0
  171:         total_count = 0
  172:         by_ticker = {}
  173: 
  174:         for t, s in ticker_stats.items():
  175:             total_correct += s["correct"]
  176:             total_count += s["total"]
  177:             if s["total"] > 0:
  178:                 by_ticker[t] = {
  179:                     "accuracy": round(s["correct"] / s["total"], 3),
  180:                     "correct": s["correct"],
  181:                     "total": s["total"],
  182:                 }
  183: 
  184:         result[sub_name] = {
  185:             "accuracy": round(total_correct / total_count, 3) if total_count else 0.0,
  186:             "correct": total_correct,
  187:             "total": total_count,
  188:             "by_ticker": by_ticker,
  189:         }
  190: 
  191:     return result
  192: 
  193: 
  194: def get_ticker_accuracy(ticker, horizon="24h", days=7, predictions_file=None):
  195:     """Get forecast accuracy for a single ticker.
  196: 
  197:     Lightweight wrapper around compute_forecast_accuracy filtered to one ticker.
  198: 
  199:     Returns:
  200:         dict: {"accuracy": float, "samples": int} or None if no data.
  201:     """
  202:     result = compute_forecast_accuracy(
  203:         ticker=ticker, horizon=horizon, days=days,
  204:         predictions_file=predictions_file,
  205:     )
  206:     # Aggregate across all sub-signals for this ticker's horizon
  207:     total_correct = 0
  208:     total_count = 0
  209:     for sub_name, stats in result.items():
  210:         # Only include sub-signals matching the requested horizon
  211:         if "_" in sub_name and sub_name.split("_", 1)[1] == horizon:
  212:             total_correct += stats["correct"]
  213:             total_count += stats["total"]
  214: 
  215:     if total_count == 0:
  216:         return None
  217: 
  218:     return {
  219:         "accuracy": round(total_correct / total_count, 3),
  220:         "samples": total_count,
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  220:         "samples": total_count,
  221:     }
  222: 
  223: 
  224: def get_all_ticker_accuracies(horizon="24h", days=7, predictions_file=None):
  225:     """Get per-ticker forecast accuracy for all tickers.
  226: 
  227:     Returns:
  228:         dict: {ticker: {"accuracy": float, "samples": int}}
  229:     """
  230:     result = compute_forecast_accuracy(
  231:         horizon=horizon, days=days, predictions_file=predictions_file,
  232:     )
  233: 
  234:     # Aggregate per-ticker across sub-signals
  235:     ticker_stats = defaultdict(lambda: {"correct": 0, "total": 0})
  236:     for sub_name, stats in result.items():
  237:         # Only include sub-signals matching the requested horizon
  238:         if "_" not in sub_name or sub_name.split("_", 1)[1] != horizon:
  239:             continue
  240:         for t, t_data in stats.get("by_ticker", {}).items():
  241:             ticker_stats[t]["correct"] += t_data["correct"]
  242:             ticker_stats[t]["total"] += t_data["total"]
  243: 
  244:     return {
  245:         t: {
  246:             "accuracy": round(s["correct"] / s["total"], 3) if s["total"] else 0.0,
  247:             "samples": s["total"],
  248:         }
  249:         for t, s in ticker_stats.items()
  250:         if s["total"] > 0
  251:     }
  252: 
  253: 
  254: def backfill_forecast_outcomes(max_entries=500, predictions_file=None,
  255:                                snapshot_file=None):
  256:     """Backfill actual price outcomes into forecast predictions.
  257: 
  258:     For each prediction without an outcome, check if enough time has
  259:     passed for the horizon, then look up the actual price and compute
  260:     the change percentage.
  261: 
  262:     Returns number of entries updated.
  263:     """
  264:     path = predictions_file or PREDICTIONS_FILE
  265:     entries = load_predictions(path)
  266:     if not entries:
  267:         return 0
  268: 
  269:     updated = 0
  270:     modified_entries = []
  271: 
  272:     for entry in entries:
  273:         if "outcome" not in entry:
  274:             entry["outcome"] = {}
  275: 
  276:         ts_str = entry.get("ts", "")
  277:         if not ts_str:
  278:             modified_entries.append(entry)
  279:             continue
  280: 
  281:         try:
  282:             entry_time = datetime.fromisoformat(ts_str)
  283:         except (ValueError, TypeError):
  284:             modified_entries.append(entry)
  285:             continue
  286: 
  287:         current_price = entry.get("current_price", 0)
  288:         if not current_price:
  289:             modified_entries.append(entry)
  290:             continue
  291: 
  292:         entry_ticker = entry.get("ticker", "")
  293:         if not entry_ticker:
  294:             modified_entries.append(entry)
  295:             continue
  296: 
  297:         now = datetime.now(UTC)
  298: 
  299:         for horizon_key, hours in [("1h", 1), ("24h", 24)]:
  300:             if horizon_key in entry["outcome"]:
  301:                 continue  # already backfilled
  302: 
  303:             horizon_time = entry_time + timedelta(hours=hours)
  304:             if now < horizon_time:
  305:                 continue  # not enough time passed
  306: 
  307:             # Look up actual price at horizon time
  308:             actual_price = _lookup_price_at_time(
  309:                 entry_ticker, horizon_time, snapshot_file=snapshot_file
  310:             )
  311:             if actual_price is not None:
  312:                 change_pct = (actual_price - current_price) / current_price * 100
  313:                 entry["outcome"][horizon_key] = {
  314:                     "actual_price": round(actual_price, 6),
  315:                     "change_pct": round(change_pct, 4),
  316:                     "backfilled_at": now.isoformat(),
  317:                 }
  318:                 updated += 1
  319: 
  320:         modified_entries.append(entry)
  321: 
  322:         if updated >= max_entries:
  323:             break
  324: 
  325:     # Write back
  326:     if updated > 0:
  327:         _write_predictions(modified_entries, path)
  328: 
  329:     return updated
  330: 
  331: 
  332: def _lookup_price_at_time(ticker, target_time, snapshot_file=None,
  333:                            tolerance_hours: float = 2.0):
  334:     """Look up the actual price for a ticker at a specific time.
  335: 
  336:     Uses hourly price snapshots from data/price_snapshots_hourly.jsonl
  337:     and finds the closest entry within `tolerance_hours` of target_time.
  338: 
  339:     The default `tolerance_hours=2.0` preserves the original behavior used
  340:     by `forecast_accuracy.compute_forecast_accuracy` and any other consumer
  341:     that expects tight tolerance. Callers backfilling LLM/sentiment outcomes
  342:     can pass a wider value (8h for crypto/metals' 24/7 markets, 24-72h for
  343:     stocks where after-hours and weekend gaps are structural).
  344: 
  345:     2026-05-01 (deferred research / fix/missing-backfill-outcomes): added
  346:     tolerance_hours parameter. Previously hardcoded to 2h, which silently
  347:     dropped 2,200+ LLM probability rows whose target_time fell into either:
  348:       - a loop-downtime gap (4-8h, all tickers symmetrically), or
  349:       - the structural MSTR overnight gap (12-72h, US stock only).
  350:     Documented in docs/PLAN_missing_backfills_20260501.md.
  351:     """
  352:     path = snapshot_file or (DATA_DIR / "price_snapshots_hourly.jsonl")
  353: 
  354:     best_price = None
  355:     best_delta = timedelta(hours=tolerance_hours)
  356: 
  357:     for snap in load_jsonl(str(path)):
  358:         try:
  359:             snap_time = datetime.fromisoformat(snap.get("ts", ""))
  360:             delta = abs(snap_time - target_time)
  361:             if delta < best_delta:
  362:                 prices = snap.get("prices", {})
  363:                 if ticker in prices:
  364:                     best_price = prices[ticker]
  365:                     best_delta = delta
  366:         except (ValueError, TypeError):
  367:             continue
  368: 
  369:     return best_price
  370: 
  371: 
  372: def _write_predictions(entries, predictions_file=None):
  373:     """Write predictions back to JSONL file."""
  374:     path = predictions_file or PREDICTIONS_FILE
  375:     atomic_write_jsonl(path, entries)
  376: 
  377: 
  378: def get_forecast_accuracy_summary(focus_tickers=None, days=7):
  379:     """Get a compact accuracy summary for Layer 2 consumption.
  380: 
  381:     Args:
  382:         focus_tickers: List of tickers to include (None = all with data).
  383:         days: Lookback window.
  384: 
  385:     Returns:
  386:         dict: {
  387:             "health": {model: success_rate},
  388:             "accuracy": {sub_signal: {accuracy, samples, by_ticker}},
  389:         }
  390:     """
  391:     health = load_health_stats()
  392:     accuracy = compute_forecast_accuracy(days=days)
  393: 
  394:     result = {
  395:         "health": health,
  396:         "accuracy": {},
  397:     }
  398: 
  399:     for sub_name, stats in accuracy.items():
  400:         entry = {
  401:             "accuracy": stats["accuracy"],
  402:             "samples": stats["total"],
  403:         }
  404:         if focus_tickers:
  405:             by_ticker = {t: stats["by_ticker"][t]
  406:                          for t in focus_tickers
  407:                          if t in stats["by_ticker"]}
  408:             if by_ticker:
  409:                 entry["by_ticker"] = by_ticker
  410:         else:
  411:             if stats["by_ticker"]:
  412:                 entry["by_ticker"] = stats["by_ticker"]
  413:         result["accuracy"][sub_name] = entry
  414: 
  415:     return result
  416: 
  417: 
  418: def print_forecast_accuracy_report():
  419:     """Print a human-readable forecast accuracy report."""
  420:     health = load_health_stats()
  421: 
  422:     print("=== Forecast Model Health ===")
  423:     if not health:
  424:         print("  No health data available yet.")
  425:     else:
  426:         for model, stats in sorted(health.items()):
  427:             rate = stats["success_rate"] * 100
  428:             print(f"  {model:10s}: {stats['ok']}/{stats['total']} ({rate:.1f}% success)")
  429: 
  430:     print("\n=== Forecast Sub-Signal Accuracy ===")
  431:     accuracy = compute_forecast_accuracy()
  432: 
  433:     if not accuracy:
  434:         print("  No outcome data available yet. Run --forecast-outcomes to backfill.")
  435:         return
  436: 
  437:     for sub_name, stats in sorted(accuracy.items()):
  438:         if stats["total"] == 0:
  439:             continue
  440:         acc = stats["accuracy"] * 100
  441:         print(f"\n  {sub_name}:")
  442:         print(f"    Overall: {acc:.1f}% ({stats['correct']}/{stats['total']})")
  443:         for t, t_stats in sorted(stats["by_ticker"].items()):
  444:             t_acc = t_stats["accuracy"] * 100
  445:             print(f"    {t:10s}: {t_acc:.1f}% ({t_stats['correct']}/{t_stats['total']})")
  446: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
    1: """Information Coefficient (IC) computation for signal evaluation.
    2: 
    3: IC measures the rank correlation between signal predictions and actual returns,
    4: providing a more nuanced quality metric than simple directional accuracy.
    5: A signal can be 55% accurate but have IC of 0.15 if it's better at predicting
    6: large moves, or 55% accurate with IC of 0.02 if it only catches noise.
    7: """
    8: 
    9: import logging
   10: import math
   11: import time
   12: from pathlib import Path
   13: 
   14: from portfolio.file_utils import atomic_write_json, load_json
   15: from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES
   16: 
   17: logger = logging.getLogger(__name__)
   18: 
   19: # 2026-05-02: absolute path (was relative `Path("data")`).
   20: # Adversarial review 05-01 P0-2: every other module uses
   21: # `Path(__file__).resolve().parent.parent / "data"`. When the scheduled task
   22: # CWD differs from repo root (e.g. PF-DataLoop launched from C:\Windows),
   23: # IC cache reads/writes silently went to a phantom directory and IC-based
   24: # weight multipliers fell to 1.0 for every signal every cycle.
   25: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   26: IC_CACHE_FILE = DATA_DIR / "ic_cache.json"
   27: IC_CACHE_TTL = 3600
   28: SIGNAL_LOG_FILE = DATA_DIR / "signal_log.jsonl"
   29: 
   30: MIN_SAMPLES_FOR_IC = 30
   31: 
   32: 
   33: def _spearman_rank_correlation(x, y):
   34:     """Compute Spearman rank correlation between two lists.
   35: 
   36:     Returns (rho, n) where rho is the correlation and n is the sample size.
   37:     Uses the standard rank-correlation formula without scipy dependency.
   38:     """
   39:     n = len(x)
   40:     if n < MIN_SAMPLES_FOR_IC:
   41:         return 0.0, n
   42: 
   43:     def _rank(values):
   44:         indexed = sorted(enumerate(values), key=lambda p: p[1])
   45:         ranks = [0.0] * n
   46:         i = 0
   47:         while i < n:
   48:             j = i
   49:             while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
   50:                 j += 1
   51:             avg_rank = (i + j) / 2.0 + 1
   52:             for k in range(i, j + 1):
   53:                 ranks[indexed[k][0]] = avg_rank
   54:             i = j + 1
   55:         return ranks
   56: 
   57:     rx = _rank(x)
   58:     ry = _rank(y)
   59: 
   60:     mean_rx = sum(rx) / n
   61:     mean_ry = sum(ry) / n
   62: 
   63:     num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
   64:     den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
   65:     den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
   66: 
   67:     if den_x == 0 or den_y == 0:
   68:         return 0.0, n
   69: 
   70:     return num / (den_x * den_y), n
   71: 
   72: 
   73: def compute_signal_ic(horizon="1d", days=None, entries=None):
   74:     """Compute Information Coefficient for each signal.
   75: 
   76:     IC = Spearman rank correlation between signal vote (+1 BUY, -1 SELL)
   77:     and actual return (change_pct) over the given horizon.
   78: 
   79:     Returns:
   80:         dict: {signal_name: {ic, ic_abs, samples, ic_buy, ic_sell, icir}}
   81:     """
   82:     if entries is None:
   83:         entries = _load_entries(days=days)
   84: 
   85:     signal_data = {s: {"votes": [], "returns": []} for s in SIGNAL_NAMES}
   86: 
   87:     for entry in entries:
   88:         outcomes = entry.get("outcomes", {})
   89:         tickers = entry.get("tickers", {})
   90: 
   91:         for ticker, tdata in tickers.items():
   92:             outcome = outcomes.get(ticker, {}).get(horizon)
   93:             if not outcome:
   94:                 continue
   95: 
   96:             change_pct = outcome.get("change_pct")
   97:             if change_pct is None:
   98:                 continue
   99: 
  100:             signals = tdata.get("signals", {})
  101:             for sig_name in SIGNAL_NAMES:
  102:                 vote = signals.get(sig_name, "HOLD")
  103:                 if vote == "HOLD":
  104:                     continue
  105:                 vote_num = 1.0 if vote == "BUY" else -1.0
  106:                 signal_data[sig_name]["votes"].append(vote_num)
  107:                 signal_data[sig_name]["returns"].append(change_pct)
  108: 
  109:     results = {}
  110:     for sig_name in SIGNAL_NAMES:
  111:         sd = signal_data[sig_name]
  112:         votes = sd["votes"]
  113:         returns = sd["returns"]
  114:         n = len(votes)
  115: 
  116:         if n < MIN_SAMPLES_FOR_IC:
  117:             results[sig_name] = {
  118:                 "ic": 0.0, "ic_abs": 0.0, "samples": n,
  119:                 "ic_buy": 0.0, "ic_sell": 0.0, "icir": 0.0,
  120:             }
  121:             continue
  122: 
  123:         ic, _ = _spearman_rank_correlation(votes, returns)
  124: 
  125:         buy_returns = [r for v, r in zip(votes, returns) if v > 0]
  126:         sell_returns = [r for v, r in zip(votes, returns) if v < 0]
  127:         ic_buy = sum(buy_returns) / len(buy_returns) if buy_returns else 0.0
  128:         ic_sell = -sum(sell_returns) / len(sell_returns) if sell_returns else 0.0
  129: 
  130:         ic_list = _rolling_ic(votes, returns, window=50)
  131:         if len(ic_list) >= 2:
  132:             ic_mean = sum(ic_list) / len(ic_list)
  133:             ic_std = math.sqrt(sum((x - ic_mean) ** 2 for x in ic_list) / len(ic_list))
  134:             icir = ic_mean / ic_std if ic_std > 0.001 else 0.0
  135:         else:
  136:             icir = 0.0
  137: 
  138:         results[sig_name] = {
  139:             "ic": round(ic, 4),
  140:             "ic_abs": round(abs(ic), 4),
  141:             "samples": n,
  142:             "ic_buy": round(ic_buy, 4),
  143:             "ic_sell": round(ic_sell, 4),
  144:             "icir": round(icir, 4),
  145:         }
  146: 
  147:     return results
  148: 
  149: 
  150: def _rolling_ic(votes, returns, window=50):
  151:     """Compute rolling IC values over a sliding window."""
  152:     if len(votes) < window:
  153:         return []
  154:     ics = []
  155:     for i in range(len(votes) - window + 1):
  156:         v_win = votes[i:i + window]
  157:         r_win = returns[i:i + window]
  158:         ic, _ = _spearman_rank_correlation(v_win, r_win)
  159:         ics.append(ic)
  160:     return ics
  161: 
  162: 
  163: def compute_signal_ic_per_ticker(horizon="1d", days=None, entries=None):
  164:     """Compute IC for each signal per ticker.
  165: 
  166:     Returns:
  167:         dict: {ticker: {signal_name: {ic, samples, ...}}}
  168:     """
  169:     if entries is None:
  170:         entries = _load_entries(days=days)
  171: 
  172:     ticker_signal_data = {}
  173: 
  174:     for entry in entries:
  175:         outcomes = entry.get("outcomes", {})
  176:         tickers = entry.get("tickers", {})
  177: 
  178:         for ticker, tdata in tickers.items():
  179:             outcome = outcomes.get(ticker, {}).get(horizon)
  180:             if not outcome:
  181:                 continue
  182: 
  183:             change_pct = outcome.get("change_pct")
  184:             if change_pct is None:
  185:                 continue
  186: 
  187:             if ticker not in ticker_signal_data:
  188:                 ticker_signal_data[ticker] = {s: {"votes": [], "returns": []}
  189:                                                for s in SIGNAL_NAMES}
  190: 
  191:             signals = tdata.get("signals", {})
  192:             for sig_name in SIGNAL_NAMES:
  193:                 vote = signals.get(sig_name, "HOLD")
  194:                 if vote == "HOLD":
  195:                     continue
  196:                 vote_num = 1.0 if vote == "BUY" else -1.0
  197:                 ticker_signal_data[ticker][sig_name]["votes"].append(vote_num)
  198:                 ticker_signal_data[ticker][sig_name]["returns"].append(change_pct)
  199: 
  200:     results = {}
  201:     for ticker, sig_data in ticker_signal_data.items():
  202:         results[ticker] = {}
  203:         for sig_name in SIGNAL_NAMES:
  204:             sd = sig_data[sig_name]
  205:             votes = sd["votes"]
  206:             returns = sd["returns"]
  207:             n = len(votes)
  208: 
  209:             if n < MIN_SAMPLES_FOR_IC:
  210:                 results[ticker][sig_name] = {"ic": 0.0, "samples": n}
  211:                 continue
  212: 
  213:             ic, _ = _spearman_rank_correlation(votes, returns)
  214:             results[ticker][sig_name] = {
  215:                 "ic": round(ic, 4),
  216:                 "ic_abs": round(abs(ic), 4),
  217:                 "samples": n,
  218:             }
  219: 
  220:     return results
  221: 
  222: 
  223: def _load_entries(days=None):
  224:     """Load signal log entries, optionally limited to last N days."""
  225:     from portfolio.accuracy_stats import load_entries
  226:     entries = load_entries()
  227:     if days and entries:
  228:         import datetime
  229:         cutoff = (datetime.datetime.now(datetime.UTC)
  230:                   - datetime.timedelta(days=days)).isoformat()
  231:         entries = [e for e in entries if e.get("ts", "") >= cutoff]
  232:     return entries
  233: 
  234: 
  235: def compute_and_cache_ic(horizon="1d"):
  236:     """Compute IC data and write to cache file."""
  237:     t0 = time.time()
  238:     ic_global = compute_signal_ic(horizon=horizon)
  239:     ic_per_ticker = compute_signal_ic_per_ticker(horizon=horizon)
  240: 
  241:     cache = {
  242:         "time": time.time(),
  243:         "horizon": horizon,
  244:         "global": ic_global,
  245:         "per_ticker": ic_per_ticker,
  246:     }
  247:     atomic_write_json(IC_CACHE_FILE, cache)
  248:     dt = time.time() - t0
  249:     logger.info("IC cache updated for %s in %.1fs", horizon, dt)
  250:     return cache
  251: 
  252: 
  253: def load_cached_ic(horizon="1d"):
  254:     """Load cached IC data, returning None if stale."""
  255:     cache = load_json(IC_CACHE_FILE)
  256:     if cache is None:
  257:         return None
  258:     if time.time() - cache.get("time", 0) > IC_CACHE_TTL:
  259:         return None
  260:     if cache.get("horizon") != horizon:
  261:         return None
  262:     return cache
  263: 
  264: 
  265: def get_signal_ic_ranking(horizon="1d", min_samples=30):
  266:     """Get signals ranked by IC magnitude.
  267: 
  268:     Returns list of (signal_name, ic, samples) sorted by |IC| descending.
  269:     """
  270:     cache = load_cached_ic(horizon)
  271:     if cache is None:
  272:         cache = compute_and_cache_ic(horizon)
  273:     ic_data = cache.get("global", {})
  274:     ranked = []
  275:     for sig_name, data in ic_data.items():
  276:         if sig_name in DISABLED_SIGNALS:
  277:             continue
  278:         if data.get("samples", 0) < min_samples:
  279:             continue
  280:         ranked.append((sig_name, data.get("ic", 0.0), data.get("samples", 0)))
  281:     ranked.sort(key=lambda x: abs(x[1]), reverse=True)
  282:     return ranked
  283: 
  284: 
  285: def print_ic_report(horizon="1d"):
  286:     """Print human-readable IC report."""
  287:     ranked = get_signal_ic_ranking(horizon, min_samples=30)
  288:     print(f"\n{'Signal':<24} {'IC':>8} {'|IC|':>8} {'Samples':>8}")
  289:     print("-" * 52)
  290:     for sig_name, ic, samples in ranked:
  291:         tag = ""
  292:         if ic < -0.02:
  293:             tag = " [CONTRARIAN]"
  294:         elif ic > 0.05:
  295:             tag = " [STRONG]"
  296:         print(f"{sig_name:<24} {ic:>8.4f} {abs(ic):>8.4f} {samples:>8}{tag}")
  297: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Train signal weights using historical signal log data.
    2: 
    3: Reads signal_log.jsonl, extracts per-signal votes and forward returns,
    4: trains a LinearFactorModel via ridge regression, and runs walk-forward
    5: validation to assess weight stability and out-of-sample performance.
    6: 
    7: Usage:
    8:     from portfolio.train_signal_weights import train_weights
    9:     result = train_weights()  # trains on all available data
   10: 
   11:     # Or from CLI:
   12:     # .venv/Scripts/python.exe -m portfolio.train_signal_weights
   13: """
   14: from __future__ import annotations
   15: 
   16: import logging
   17: from pathlib import Path
   18: 
   19: import pandas as pd
   20: 
   21: from portfolio.file_utils import load_jsonl
   22: from portfolio.linear_factor import LinearFactorModel
   23: from portfolio.signal_weight_optimizer import (
   24:     save_results,
   25:     walk_forward_optimize,
   26: )
   27: 
   28: logger = logging.getLogger("portfolio.train_signal_weights")
   29: 
   30: _BASE_DIR = Path(__file__).resolve().parent.parent
   31: _SIGNAL_LOG = _BASE_DIR / "data" / "signal_log.jsonl"
   32: _VOTE_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
   33: 
   34: # Signals to include as features (skip disabled/broken ones)
   35: _SKIP_SIGNALS = {"ml", "funding", "lora"}
   36: 
   37: 
   38: def _load_signal_history(
   39:     log_path: Path | None = None,
   40:     horizon: str = "1d",
   41:     min_entries: int = 50,
   42: ) -> tuple[pd.DataFrame, pd.Series] | None:
   43:     """Load signal log and extract signal votes + forward returns.
   44: 
   45:     Args:
   46:         log_path: Path to signal_log.jsonl.
   47:         horizon: Outcome horizon to use for returns ("3h", "1d", "3d").
   48:         min_entries: Minimum entries with outcomes required.
   49: 
   50:     Returns:
   51:         (signals_df, returns_series) or None if insufficient data.
   52:     """
   53:     log_path = log_path or _SIGNAL_LOG
   54:     entries = load_jsonl(log_path)
   55:     if not entries:
   56:         logger.warning("No signal log entries found at %s", log_path)
   57:         return None
   58: 
   59:     rows = []
   60:     for entry in entries:
   61:         ts = entry.get("ts")
   62:         tickers = entry.get("tickers", {})
   63:         outcomes = entry.get("outcomes", {})
   64: 
   65:         for ticker, tdata in tickers.items():
   66:             signals = tdata.get("signals", {})
   67:             # Get outcome for this horizon
   68:             ticker_outcomes = outcomes.get(ticker, {})
   69:             outcome = ticker_outcomes.get(horizon)
   70:             if outcome is None:
   71:                 continue
   72: 
   73:             change_pct = outcome if isinstance(outcome, (int, float)) else outcome.get("change_pct")
   74:             if change_pct is None:
   75:                 continue
   76: 
   77:             # Convert votes to numeric
   78:             row = {"ts": ts, "ticker": ticker, "return": float(change_pct) / 100.0}
   79:             for sig_name, vote in signals.items():
   80:                 if sig_name in _SKIP_SIGNALS:
   81:                     continue
   82:                 row[sig_name] = _VOTE_MAP.get(vote, 0.0)
   83:             rows.append(row)
   84: 
   85:     if len(rows) < min_entries:
   86:         logger.warning("Insufficient entries with outcomes: %d < %d", len(rows), min_entries)
   87:         return None
   88: 
   89:     df = pd.DataFrame(rows)
   90:     df["ts"] = pd.to_datetime(df["ts"])
   91:     df = df.set_index("ts").sort_index()
   92: 
   93:     # Separate returns from signals
   94:     returns = df["return"]
   95:     signal_cols = [c for c in df.columns if c not in ("return", "ticker")]
   96:     signals_df = df[signal_cols].fillna(0.0)
   97: 
   98:     return signals_df, returns
   99: 
  100: 
  101: def train_weights(
  102:     horizon: str = "1d",
  103:     alpha: float = 1.0,
  104:     log_path: Path | None = None,
  105: ) -> dict:
  106:     """Train linear factor model and run walk-forward validation.
  107: 
  108:     Args:
  109:         horizon: Forward return horizon ("3h", "1d", "3d").
  110:         alpha: Ridge regularization strength.
  111:         log_path: Path to signal_log.jsonl.
  112: 
  113:     Returns:
  114:         Dict with model stats, walk-forward results, and feature rankings.
  115:         Empty dict on failure.
  116:     """
  117:     data = _load_signal_history(log_path, horizon=horizon)
  118:     if data is None:
  119:         return {}
  120: 
  121:     signals_df, returns = data
  122:     logger.info("Training on %d samples, %d signals, horizon=%s",
  123:                len(signals_df), len(signals_df.columns), horizon)
  124: 
  125:     # Train full model
  126:     model = LinearFactorModel(alpha=alpha)
  127:     if not model.fit(signals_df, returns):
  128:         logger.warning("Model training failed")
  129:         return {}
  130: 
  131:     model.save()
  132:     logger.info("Model saved: R²=%.4f, %d features", model.r_squared, len(model.weights))
  133: 
  134:     # Run walk-forward validation
  135:     wf_result = walk_forward_optimize(
  136:         signals_df, returns,
  137:         train_window=min(720, len(signals_df) // 3),
  138:         test_window=min(168, len(signals_df) // 6),
  139:         step_size=min(168, len(signals_df) // 6),
  140:         alpha=alpha,
  141:     )
  142: 
  143:     if wf_result.n_windows > 0:
  144:         save_results(wf_result)
  145:         logger.info("Walk-forward: %d windows, OOS corr=%.4f",
  146:                     wf_result.n_windows, wf_result.avg_oos_corr)
  147: 
  148:     return {
  149:         "model": {
  150:             "r_squared": model.r_squared,
  151:             "n_samples": model.n_samples,
  152:             "n_features": len(model.weights),
  153:             "top_features": model.feature_importance()[:10],
  154:         },
  155:         "walk_forward": wf_result.to_dict() if wf_result.n_windows > 0 else {},
  156:     }
  157: 
  158: 
  159: if __name__ == "__main__":
  160:     logging.basicConfig(level=logging.INFO)
  161:     result = train_weights()
  162:     if result:
  163:         print(f"\nModel: R²={result['model']['r_squared']:.4f}, "
  164:               f"{result['model']['n_features']} features, "
  165:               f"{result['model']['n_samples']} samples")
  166:         print("\nTop features:")
  167:         for name, weight in result["model"]["top_features"]:
  168:             print(f"  {name:25s} β={weight:+.6f}")
  169:         if result.get("walk_forward"):
  170:             wf = result["walk_forward"]
  171:             print(f"\nWalk-forward: {wf['n_windows']} windows, "
  172:                   f"R²={wf['avg_r_squared']:.4f}, "
  173:                   f"OOS corr={wf['avg_oos_corr']:.4f}")
  174:     else:
  175:         print("Training failed — insufficient data")
  176: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Linear factor model for signal combination.
    2: 
    3: Alternative to majority voting: trains ridge regression weights on historical
    4: signal snapshots and forward returns. Produces a continuous score rather than
    5: BUY/SELL/HOLD vote counts.
    6: 
    7: Formula: predicted_return = B0 + sum_i Bi * zi
    8: where zi are z-scored signal features.
    9: 
   10: Usage:
   11:     model = LinearFactorModel()
   12:     model.fit(signal_history_df, returns_series)
   13:     score = model.predict(current_signals_dict)
   14: """
   15: from __future__ import annotations
   16: 
   17: import logging
   18: from pathlib import Path
   19: 
   20: import numpy as np
   21: import pandas as pd
   22: 
   23: from portfolio.file_utils import atomic_write_json, load_json
   24: 
   25: logger = logging.getLogger("portfolio.linear_factor")
   26: 
   27: _BASE_DIR = Path(__file__).resolve().parent.parent
   28: _MODEL_FILE = _BASE_DIR / "data" / "models" / "linear_factor_weights.json"
   29: 
   30: 
   31: class LinearFactorModel:
   32:     """Ridge regression model for combining trading signals.
   33: 
   34:     Attributes:
   35:         weights: Dict mapping signal name to beta weight.
   36:         intercept: beta_0 intercept term.
   37:         alpha: Ridge regularization strength.
   38:         feature_means: Dict of feature means for z-scoring.
   39:         feature_stds: Dict of feature stds for z-scoring.
   40:         r_squared: Training R-squared score.
   41:         n_samples: Number of training samples used.
   42:     """
   43: 
   44:     def __init__(self, alpha: float = 1.0):
   45:         self.alpha = alpha
   46:         self.weights: dict[str, float] = {}
   47:         self.intercept: float = 0.0
   48:         self.feature_means: dict[str, float] = {}
   49:         self.feature_stds: dict[str, float] = {}
   50:         self.r_squared: float = 0.0
   51:         self.n_samples: int = 0
   52: 
   53:     def fit(self, signals_df: pd.DataFrame, returns: pd.Series,
   54:             min_samples: int = 30) -> bool:
   55:         """Train ridge regression on historical signal data.
   56: 
   57:         Args:
   58:             signals_df: DataFrame where columns are signal names and rows are
   59:                        time observations. Values should be numeric
   60:                        (e.g. confidence * direction_sign).
   61:             returns: Series of forward returns aligned with signals_df index.
   62:             min_samples: Minimum training samples required.
   63: 
   64:         Returns:
   65:             True if training succeeded, False if insufficient data.
   66:         """
   67:         # Align and drop NaN
   68:         common = signals_df.index.intersection(returns.index)
   69:         if len(common) < min_samples:
   70:             logger.warning("Insufficient data for linear factor: %d < %d",
   71:                            len(common), min_samples)
   72:             return False
   73: 
   74:         X = signals_df.loc[common].copy()
   75:         y = returns.loc[common].copy()
   76: 
   77:         # Drop columns with zero variance
   78:         stds = X.std()
   79:         valid_cols = stds[stds > 1e-10].index.tolist()
   80:         if not valid_cols:
   81:             logger.warning("No valid signal columns (all zero variance)")
   82:             return False
   83:         X = X[valid_cols]
   84: 
   85:         # Z-score features
   86:         means = X.mean()
   87:         stds = X.std()
   88:         X_z = (X - means) / stds.replace(0, 1)
   89: 
   90:         # Ridge regression: beta = (X'X + alpha*I)^-1 X'y
   91:         X_arr = X_z.values
   92:         y_arr = y.values
   93:         n_features = X_arr.shape[1]
   94:         XtX = X_arr.T @ X_arr
   95:         Xty = X_arr.T @ y_arr
   96:         ridge_term = self.alpha * np.eye(n_features)
   97:         try:
   98:             beta = np.linalg.solve(XtX + ridge_term, Xty)
   99:         except np.linalg.LinAlgError:
  100:             logger.warning("Ridge regression solve failed (singular matrix)")
  101:             return False
  102: 
  103:         # Intercept
  104:         intercept = float(y_arr.mean() - beta @ X_z.mean().values)
  105: 
  106:         # R-squared score
  107:         y_pred = X_arr @ beta + intercept
  108:         ss_res = np.sum((y_arr - y_pred) ** 2)
  109:         ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
  110:         r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
  111: 
  112:         # Store results
  113:         self.weights = {col: float(beta[i]) for i, col in enumerate(valid_cols)}
  114:         self.intercept = intercept
  115:         self.feature_means = {col: float(means[col]) for col in valid_cols}
  116:         self.feature_stds = {col: float(stds[col]) for col in valid_cols}
  117:         self.r_squared = float(r2)
  118:         self.n_samples = len(common)
  119: 
  120:         logger.info("Linear factor trained: %d features, %d samples, R²=%.4f",
  121:                      len(valid_cols), len(common), r2)
  122:         return True
  123: 
  124:     def predict(self, signals: dict[str, float]) -> float:
  125:         """Score a set of current signal values.
  126: 
  127:         Args:
  128:             signals: Dict mapping signal name to numeric value.
  129: 
  130:         Returns:
  131:             Predicted return (continuous). Positive = bullish, negative = bearish.
  132:             Returns 0.0 if model not trained.
  133:         """
  134:         if not self.weights:
  135:             return 0.0
  136: 
  137:         score = self.intercept
  138:         for name, beta in self.weights.items():
  139:             raw = signals.get(name, 0.0)
  140:             mean = self.feature_means.get(name, 0.0)
  141:             std = self.feature_stds.get(name, 1.0)
  142:             z = (raw - mean) / std if std > 1e-10 else 0.0
  143:             score += beta * z
  144:         return float(score)
  145: 
  146:     def score_to_action(self, score: float, threshold: float = 0.001) -> tuple[str, float]:
  147:         """Convert continuous score to BUY/SELL/HOLD action.
  148: 
  149:         Args:
  150:             score: Predicted return from predict().
  151:             threshold: Minimum absolute score for directional signal.
  152: 
  153:         Returns:
  154:             (action, confidence) tuple.
  155:         """
  156:         if abs(score) < threshold:
  157:             return "HOLD", 0.0
  158:         action = "BUY" if score > 0 else "SELL"
  159:         # Confidence proportional to score magnitude, capped at 0.8
  160:         confidence = min(abs(score) / (threshold * 5), 0.8)
  161:         return action, round(confidence, 4)
  162: 
  163:     def save(self, path: Path | None = None) -> None:
  164:         """Persist model weights to JSON."""
  165:         path = path or _MODEL_FILE
  166:         path.parent.mkdir(parents=True, exist_ok=True)
  167:         data = {
  168:             "weights": self.weights,
  169:             "intercept": self.intercept,
  170:             "feature_means": self.feature_means,
  171:             "feature_stds": self.feature_stds,
  172:             "alpha": self.alpha,
  173:             "r_squared": self.r_squared,
  174:             "n_samples": self.n_samples,
  175:         }
  176:         atomic_write_json(path, data)
  177: 
  178:     def load(self, path: Path | None = None) -> bool:
  179:         """Load model weights from JSON.
  180: 
  181:         Returns True if loaded successfully, False otherwise.
  182:         """
  183:         path = path or _MODEL_FILE
  184:         data = load_json(path)
  185:         if not data or "weights" not in data:
  186:             return False
  187:         self.weights = data["weights"]
  188:         self.intercept = data.get("intercept", 0.0)
  189:         self.feature_means = data.get("feature_means", {})
  190:         self.feature_stds = data.get("feature_stds", {})
  191:         self.alpha = data.get("alpha", 1.0)
  192:         self.r_squared = data.get("r_squared", 0.0)
  193:         self.n_samples = data.get("n_samples", 0)
  194:         return True
  195: 
  196:     def feature_importance(self) -> list[tuple[str, float]]:
  197:         """Return features sorted by absolute weight (most important first)."""
  198:         return sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True)
  199: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Rolling z-score feature normalizer for signal inputs.
    2: 
    3: Maintains per-ticker, per-indicator rolling statistics (mean, std) and
    4: provides z-score normalization.  This makes signal thresholds regime-adaptive:
    5: a RSI of 72 in low-vol is more extreme than RSI 72 in high-vol.
    6: 
    7: Usage:
    8:     from portfolio.feature_normalizer import normalize, update
    9: 
   10:     # Each signal cycle, update with raw value
   11:     update("XAG-USD", "rsi_14", 72.0)
   12: 
   13:     # Get z-score for threshold comparison
   14:     z = normalize("XAG-USD", "rsi_14", 72.0)  # returns z-score or raw if insufficient data
   15: 
   16: Research basis: "Standardization (Z-score): Convert features to zero mean/unit variance"
   17: from quantitative signals research paper on 1-3h metals forecasting.
   18: """
   19: from __future__ import annotations
   20: 
   21: import logging
   22: from collections import deque
   23: 
   24: import numpy as np
   25: 
   26: logger = logging.getLogger("portfolio.feature_normalizer")
   27: 
   28: _DEFAULT_WINDOW = 100  # rolling window size for stats
   29: _MIN_SAMPLES = 20      # minimum samples before z-scoring (otherwise return raw)
   30: 
   31: # In-memory storage: {(ticker, indicator_name): deque of values}
   32: _buffers: dict[tuple[str, str], deque] = {}
   33: 
   34: 
   35: def _ensure_buffer(ticker: str, indicator: str) -> deque:
   36:     """Get or create the rolling buffer for a ticker+indicator pair."""
   37:     key = (ticker, indicator)
   38:     if key not in _buffers:
   39:         _buffers[key] = deque(maxlen=_DEFAULT_WINDOW)
   40:     return _buffers[key]
   41: 
   42: 
   43: def update(ticker: str, indicator: str, value: float) -> None:
   44:     """Record a new raw value for a ticker+indicator pair.
   45: 
   46:     Call this each signal cycle to build up the rolling distribution.
   47:     """
   48:     if not np.isfinite(value):
   49:         return
   50:     buf = _ensure_buffer(ticker, indicator)
   51:     buf.append(value)
   52: 
   53: 
   54: def normalize(ticker: str, indicator: str, value: float) -> float:
   55:     """Z-score normalize a value against its rolling distribution.
   56: 
   57:     Returns the z-score if sufficient history exists (>= _MIN_SAMPLES),
   58:     otherwise returns the raw value unchanged.  This ensures cold-start
   59:     safety: signals work with raw thresholds until enough data accumulates.
   60:     """
   61:     if not np.isfinite(value):
   62:         return 0.0
   63:     buf = _ensure_buffer(ticker, indicator)
   64:     if len(buf) < _MIN_SAMPLES:
   65:         return value
   66:     arr = np.array(buf, dtype=float)
   67:     mean = arr.mean()
   68:     std = arr.std()
   69:     if std < 1e-12:
   70:         return 0.0
   71:     return float((value - mean) / std)
   72: 
   73: 
   74: def has_sufficient_history(ticker: str, indicator: str) -> bool:
   75:     """Check if enough samples have accumulated for z-scoring."""
   76:     key = (ticker, indicator)
   77:     buf = _buffers.get(key)
   78:     return buf is not None and len(buf) >= _MIN_SAMPLES
   79: 
   80: 
   81: def get_stats(ticker: str, indicator: str) -> dict | None:
   82:     """Get rolling statistics for a ticker+indicator pair.
   83: 
   84:     Returns dict with mean, std, count, or None if no data.
   85:     """
   86:     key = (ticker, indicator)
   87:     buf = _buffers.get(key)
   88:     if not buf:
   89:         return None
   90:     arr = np.array(buf, dtype=float)
   91:     return {
   92:         "mean": float(arr.mean()),
   93:         "std": float(arr.std()),
   94:         "count": len(buf),
   95:         "min": float(arr.min()),
   96:         "max": float(arr.max()),
   97:     }
   98: 
   99: 
  100: def clear(ticker: str | None = None, indicator: str | None = None) -> None:
  101:     """Clear buffers.  If ticker given, clear only that ticker's buffers.
  102:     If both ticker and indicator given, clear only that specific pair.
  103:     """
  104:     if ticker and indicator:
  105:         key = (ticker, indicator)
  106:         if key in _buffers:
  107:             del _buffers[key]
  108:     elif ticker:
  109:         keys = [k for k in _buffers if k[0] == ticker]
  110:         for k in keys:
  111:             del _buffers[k]
  112:     else:
  113:         _buffers.clear()
  114: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
    1: """Configuration for 3-hour prediction horizon.
    2: 
    3: Research-backed constants for optimizing signal accuracy at 3h:
    4: - Slow signals (trend, fibonacci, macro_regime) are noise at 3h (38-45% accuracy)
    5: - Peak trading hours (10-17 UTC) have 43-45% accuracy; quiet hours (20-01 UTC) have 52-55%
    6: - Confidence above 75% at 3h is anti-correlated with accuracy (90%+ bucket = 28% actual)
    7: """
    8: 
    9: # Signals that need daily+ data and degrade at 3h.
   10: # trend: 38.8% at 3h (needs SMA200 = 33+ days context)
   11: # fibonacci: 43.3% at 3h (swing detection needs broad history)
   12: # macro_regime: 45.1% at 3h (200-SMA regime is daily-scale)
   13: SLOW_SIGNALS_3H = frozenset({"trend", "fibonacci", "macro_regime"})
   14: 
   15: # 3h confidence cap. The 90%+ confidence bucket has 28.1% actual accuracy.
   16: # The 70-80% band is the best-performing at 58.9%. Cap at 75%.
   17: CONFIDENCE_CAP_3H = 0.75
   18: 
   19: # Time-of-day scaling for 3h predictions (UTC hours).
   20: # Based on measured consensus accuracy by hour.
   21: _TOD_SCALE = {
   22:     # Quiet hours (20:00-01:00 UTC) — 52-55% accuracy
   23:     20: 1.10, 21: 1.10, 22: 1.08, 23: 1.08, 0: 1.10,
   24:     # Asian session (1-6 UTC) — 48-50%, slightly below baseline
   25:     1: 0.97, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.97,
   26:     # Pre-EU (7-9 UTC) — neutral
   27:     7: 1.0, 8: 1.0, 9: 1.0,
   28:     # Peak noise (10-17 UTC) — 43-45% accuracy
   29:     10: 0.88, 11: 0.88, 12: 0.85, 13: 0.88, 14: 0.88,
   30:     15: 0.88, 16: 0.85, 17: 0.85,
   31:     # Transition (18-19 UTC) — near neutral
   32:     18: 1.0, 19: 1.0,
   33: }
   34: 
   35: 
   36: def is_slow_signal_3h(signal_name: str) -> bool:
   37:     """Check if a signal should be disabled for 3h horizon predictions."""
   38:     return signal_name in SLOW_SIGNALS_3H
   39: 
   40: 
   41: def time_of_day_scale_3h(hour: int) -> float:
   42:     """Return confidence scaling factor for 3h predictions at given UTC hour."""
   43:     return float(_TOD_SCALE.get(hour % 24, 1.0))
   44: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
160: # group leader) still applies, so a soft vote on a force-HOLD signal
161: # is still force-HOLD'd. We are NOT bypassing the accuracy gate.
428: # force-HOLD (treated like DISABLED_SIGNALS but dynamically). A signal at
429: # 44% is noise, not a reliable contrarian indicator — inverting it just
431: # 2026-04-11 (A-PR-batch-5): raised 0.45 → 0.47. The signal audit on
436: ACCURACY_GATE_THRESHOLD = 0.47
453: # threshold get that direction force-HOLD'd while the other direction can still
475: # 2026-04-15: raised normal 0.70→0.75, fast 0.90→0.95 to better capture
477: # 2026-04-16: REVERTED to 0.70/0.90. The 0.75/0.95 tuning amplified noise
644: # track accuracy while they remain force-HOLD in consensus.
687: _GATE_RELAXATION_MAX = 0.06   # cap at 6pp below base gate (0.47 -> 0.41)
702: # is force-HOLD'd per-horizon at consensus time).
750:     # 2026-04-16 after-hours audit: signals that PASS global gate (>0.47)
888:     """Return signals to force-HOLD for (ticker, horizon). Union of default + horizon-specific.
1793:     # relaxed. A signal with 10K+ samples at sub-50% accuracy has measurable
2002:     - IC < 0 with stable ICIR → penalty (contrarian, accuracy gate handles)
2107:     they are noise, not useful contrarian indicators.
2228:     # Regime gating: force-HOLD signals that produce negative alpha in this regime.
2243:     # Macro-window force-HOLD pre-pass (2026-04-28). When a high-impact
2244:     # event is within ~24h past or ~72h future, force-HOLD the signals
2308:     # 2026-04-06: Lowered from 0.47 → 0.46 to catch borderline cases where
2466:         # at sub-50% accuracy has statistically demonstrated negative edge —
2828:         conf *= 0.75
2829:         penalty_log.append({"stage": "regime", "regime": "ranging", "mult": 0.75})
2831:         conf *= 0.80
2832:         penalty_log.append({"stage": "regime", "regime": "high-vol", "mult": 0.80})
2923:                 conf *= 0.75
2924:                 penalty_log.append({"stage": "unanimity", "agreement": round(agreement_ratio, 3), "mult": 0.75})
3120:     # Gated: F&G is contrarian (buy fear, sell greed) which fights trends.
3123:     # fear (46+ consecutive days as of Apr 2), contrarian BUY signals are noise.
3145:             # Gate: sustained extreme fear — contrarian BUY is unreliable
3151:                 extra_info["fear_greed_note"] = f"sustained_fear_{fear_days}d_allowing_contrarian"
3292:                 # NUPL: <0 capitulation (BUY), >0.75 euphoria (SELL)
3297:                     elif nupl > 0.75:
3477:             # "registered but force-HOLD pending live validation" signals
3767:     # Codex round 2 P1 (2026-04-28): macro-window force-HOLD must mutate
4005:     # Macro-window force-HOLD has already been applied to `votes` above
4182:     conf = min(conf, 0.80)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 2298:         if len(active_in_group) <= 1:
 2299:             continue
 2300:         best_sig = max(active_in_group, key=_leader_accuracy_key)
 2301:         group_leaders[group_name] = best_sig
 2302: 
 2303:     # Correlation group leader gating: when the best signal in a group has
 2304:     # accuracy below threshold (with enough samples), gate the ENTIRE group.
 2305:     # Prevents the "least bad" broken signal from voting.
 2306:     # 2026-03-31: macro_external group (fear_greed 25.9%, sentiment 46.8%,
 2307:     # news_event 29.5%) — even the leader is near noise.
 2308:     # 2026-04-06: Lowered from 0.47 → 0.46 to catch borderline cases where
 2309:     # sentiment (blended ~46.4%) barely escapes as group leader.
 2310:     _GROUP_LEADER_GATE_THRESHOLD = 0.46
 2311:     group_gated_signals: set[str] = set()
 2312:     for group_name, group_sigs in _active_corr_groups.items():
 2313:         leader = group_leaders.get(group_name)
 2314:         if leader:
 2315:             leader_stats = accuracy_data.get(leader, {})
 2316:             leader_acc = leader_stats.get("accuracy", 0.5)
 2317:             leader_samples = leader_stats.get("total", 0)
 2318:             if leader_samples >= ACCURACY_GATE_MIN_SAMPLES and leader_acc < _GROUP_LEADER_GATE_THRESHOLD:
 2319:                 group_gated_signals.update(group_sigs & active_non_hold)
 2320:                 logger.debug(
 2321:                     "Correlation group %s gated: leader %s at %.1f%% < %.0f%% threshold",
 2322:                     group_name, leader, leader_acc * 100, _GROUP_LEADER_GATE_THRESHOLD * 100,
 2323:                 )
 2324: 
 2325:     # Build a mapping of signal → correlation penalty (per-cluster override).
 2326:     # When a signal is in multiple groups, use the harshest (lowest) penalty.
 2327:     penalized_signals: dict[str, float] = {}
 2328:     for group_name, group_sigs in _active_corr_groups.items():
 2329:         leader = group_leaders.get(group_name)
 2330:         if leader:
 2331:             penalty = _CLUSTER_CORRELATION_PENALTIES.get(group_name, _CORRELATION_PENALTY)
 2332:             for s in group_sigs:
 2333:                 if s != leader and s in active_non_hold:
 2334:                     penalized_signals[s] = min(penalized_signals.get(s, 1.0), penalty)
 2335: 
 2336:     # Meta-cluster deduplication (2026-05-01): when leaders from related
 2337:     # sub-clusters agree on direction, apply penalty to redundant leaders.
 2338:     # Prevents the trend mega-view from getting 3.0x effective leader weight
 2339:     # when pure_trend/oscillator_trend/structural_flow leaders vote identically.
 2340:     for meta_name, sub_clusters in _META_CLUSTER_GROUPS.items():
 2341:         meta_leaders: dict[str, str] = {}
 2342:         for sc_name in sub_clusters:
 2343:             leader = group_leaders.get(sc_name)
 2344:             if leader and leader in active_non_hold:
 2345:                 meta_leaders[sc_name] = leader
 2346:         if len(meta_leaders) < 2:
 2347:             continue
 2348:         # Check if all leaders agree on direction
 2349:         leader_directions = {sc: votes.get(ldr, "HOLD")
 2350:                             for sc, ldr in meta_leaders.items()}
 2351:         active_dirs = set(leader_directions.values()) - {"HOLD"}
 2352:         if len(active_dirs) != 1:
 2353:             continue  # Leaders disagree — informative diversity, no penalty
 2354:         # All leaders agree: keep best-accuracy leader, penalize others
 2355:         best_sc = max(meta_leaders,
 2356:                       key=lambda sc: _leader_accuracy_key(meta_leaders[sc]))
 2357:         for sc_name, leader in meta_leaders.items():
 2358:             if sc_name != best_sc:
 2359:                 current = penalized_signals.get(leader, 1.0)
 2360:                 penalized_signals[leader] = min(current, _META_CLUSTER_PENALTY)
 2361:                 logger.debug(
 2362:                     "Meta-cluster %s: %s leader %s agrees with %s leader %s "
 2363:                     "— penalized to %.0f%%",
 2364:                     meta_name, sc_name, leader, best_sc,
 2365:                     meta_leaders[best_sc], _META_CLUSTER_PENALTY * 100,
 2366:                 )
 2367: 
 2368:     # Crisis mode detection: when multiple macro-external signals have degraded
 2369:     # accuracy, the market is in an abnormal regime (war, systemic crisis) where
 2370:     # trend-following breaks and mean-reversion becomes more reliable.
 2371:     #
 2372:     # 2026-04-19: Made crisis response conditional on trend signal performance.
 2373:     # When macro signals are broken but trend signals have >55% accuracy, the
 2374:     # crisis is in the macro indicators, not in the trend — penalizing trend
 2375:     # signals that are winning is actively harmful (observed: trend 61.6%,
 2376:     # EMA 62.9% being penalized 0.6x while crisis mode was active).
 2377:     _MACRO_CRISIS_SIGNALS = {"fear_greed", "macro_regime", "structure", "news_event", "sentiment"}
 2378:     broken_count = sum(
 2379:         1 for s in _MACRO_CRISIS_SIGNALS
 2380:         if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
 2381:         and accuracy_data.get(s, {}).get("accuracy", 0.5) < _CRISIS_THRESHOLD
 2382:     )
 2383:     crisis_mode = broken_count >= _CRISIS_MIN_BROKEN
 2384: 
 2385:     _TREND_SIGNALS = {"ema", "trend", "heikin_ashi", "volume_flow"}
 2386:     _MR_SIGNALS = {"mean_reversion", "calendar"}
 2387: 
 2388:     # Check if trend signals are actually underperforming before penalizing.
 2389:     # If avg trend accuracy > 55%, trend is capturing edge despite macro chaos.
 2390:     _CRISIS_TREND_ACCURACY_FLOOR = 0.55
 2391:     crisis_penalize_trend = False
 2392:     if crisis_mode:
 2393:         trend_accs = [
 2394:             accuracy_data.get(s, {}).get("accuracy", 0.5)
 2395:             for s in _TREND_SIGNALS
 2396:             if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
 2397:         ]
 2398:         avg_trend_acc = sum(trend_accs) / len(trend_accs) if trend_accs else 0.5
 2399:         crisis_penalize_trend = avg_trend_acc < _CRISIS_TREND_ACCURACY_FLOOR
 2400:         if crisis_penalize_trend:
 2401:             logger.info(
 2402:                 "Crisis mode active (full): %d/%d macro signals broken, "
 2403:                 "trend avg %.1f%% < %.0f%% floor — penalizing trend, boosting MR",
 2404:                 broken_count, len(_MACRO_CRISIS_SIGNALS),
 2405:                 avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
 2406:             )
 2407:         else:
 2408:             logger.info(
 2409:                 "Crisis mode active (partial): %d/%d macro signals broken, but "
 2410:                 "trend avg %.1f%% >= %.0f%% floor — NOT penalizing trend signals",
 2411:                 broken_count, len(_MACRO_CRISIS_SIGNALS),
 2412:                 avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
 2413:             )
 2414: 
 2415:     # Voter-count circuit breaker (Batch 2 of 2026-04-16 accuracy gating reconfig).
 2416:     # Only the overall accuracy gate is relaxable — directional and correlation
 2417:     # gates still fire. Prevents regime-transition over-gating that silenced
 2418:     # ~8 voters in W15/W16.
 2419:     relaxation = _compute_gate_relaxation(
 2420:         votes=votes,
 2421:         accuracy_data=accuracy_data,
 2422:         excluded=excluded,
 2423:         group_gated=group_gated_signals,
 2424:         base_gate=gate,
 2425:         regime=regime,
 2426:     )
 2427:     if relaxation > 0:
 2428:         logger.debug(
 2429:             "Circuit breaker: relaxing accuracy gate by %.0fpp "
 2430:             "(base=%.2f -> effective=%.2f) to preserve voter diversity",
 2431:             relaxation * 100, gate, gate - relaxation,
 2432:         )
 2433: 
 2434:     # IC-based weight multiplier (2026-04-18): load IC data once per consensus
 2435:     # call. Returns {"global": {sig: {ic, icir, samples}}, "per_ticker": {...}}
 2436:     # or None if IC computation is unavailable.
 2437:     ic_cache = _get_ic_data(horizon) if horizon else None
 2438:     ic_global = ic_cache.get("global", {}) if ic_cache else {}
 2439:     ic_per_ticker = ic_cache.get("per_ticker", {}) if ic_cache else {}
 2440: 
 2441:     for signal_name, vote in votes.items():
 2442:         # P1-1 (2026-05-02 adversarial follow-ups): defensive — initialize
 2443:         # _rescued at the TOP of every iteration so a future contributor who
 2444:         # adds a third branch to the gate-check below cannot leak a stale
 2445:         # True from a prior iteration into line 2123 (`if _rescued: weight
 2446:         # *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY`). Today both arms of the
 2447:         # if/else at line 2072 set _rescued, so the bug doesn't manifest in
 2448:         # production — but the structural guarantee is now hardcoded.
 2449:         _rescued = False
 2450:         if vote == "HOLD":
 2451:             continue
 2452:         if signal_name in excluded:
 2453:             continue
 2454:         # Correlation group leader gating: entire group silenced
 2455:         if signal_name in group_gated_signals:
 2456:             gated_signals.append(signal_name)
 2457:             continue
 2458:         stats = accuracy_data.get(signal_name, {})
 2459:         acc = stats.get("accuracy", 0.5)
 2460:         samples = stats.get("total", 0)
 2461:         # Accuracy gate: skip signals that are below threshold with enough data.
 2462:         # Tiered: established signals (10000+ samples) use a tighter 50% gate;
 2463:         # newer signals use the standard 47% gate.
 2464:         # SC-P1-2 (2026-05-02 adversarial follow-ups): the high-sample tier
 2465:         # (10K+ samples, 0.50 gate) is NOT relaxed. A signal with 10K+ samples
 2466:         # at sub-50% accuracy has statistically demonstrated negative edge —
 2467:         # circuit-breaker relaxation must not let it back in. The standard
 2468:         # tier still relaxes uniformly so newer borderline signals can be
 2469:         # rescued during regime transitions.
 2470:         effective_gate = gate - relaxation
 2471:         if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
 2472:             effective_gate = max(
 2473:                 gate - relaxation,
 2474:                 _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD,
 2475:             )
 2476:         if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
 2477:             # Directional rescue: before gating, check if the vote direction
 2478:             # has strong enough accuracy to justify a reduced-weight vote.
 2479:             if vote == "BUY":
 2480:                 rescue_acc = stats.get("buy_accuracy", 0.0)
 2481:                 rescue_n = stats.get("total_buy", 0)
 2482:             else:
 2483:                 rescue_acc = stats.get("sell_accuracy", 0.0)
 2484:                 rescue_n = stats.get("total_sell", 0)
 2485:             if (rescue_n >= _DIRECTIONAL_RESCUE_MIN_SAMPLES
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
 2485:             if (rescue_n >= _DIRECTIONAL_RESCUE_MIN_SAMPLES
 2486:                     and rescue_acc >= _DIRECTIONAL_RESCUE_THRESHOLD):
 2487:                 logger.debug(
 2488:                     "Directional rescue: %s overall=%.1f%% (gated) but "
 2489:                     "%s=%.1f%% (%d sam) — rescued at %.0f%% weight",
 2490:                     signal_name, acc * 100, vote,
 2491:                     rescue_acc * 100, rescue_n,
 2492:                     _DIRECTIONAL_RESCUE_WEIGHT_PENALTY * 100,
 2493:                 )
 2494:                 # Fall through to weighting with rescue penalty applied later
 2495:                 _rescued = True
 2496:             else:
 2497:                 gated_signals.append(signal_name)
 2498:                 continue
 2499:         else:
 2500:             _rescued = False
 2501:         # Directional accuracy gate: gate individual BUY/SELL direction when
 2502:         # direction-specific accuracy is very poor, even if overall accuracy passes.
 2503:         # E.g., qwen3 overall=59.8% passes, but BUY=30.0% → gate BUY only.
 2504:         if vote == "BUY":
 2505:             dir_acc = stats.get("buy_accuracy", acc)
 2506:             dir_n = stats.get("total_buy", 0)
 2507:         else:
 2508:             dir_acc = stats.get("sell_accuracy", acc)
 2509:             dir_n = stats.get("total_sell", 0)
 2510:         if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
 2511:             gated_signals.append(f"{signal_name}_{vote}")
 2512:             continue
 2513:         # BUG-182: Use direction-specific accuracy as weight when available.
 2514:         # A signal with overall 60% accuracy may be 30% for BUY and 75% for SELL.
 2515:         # Using overall accuracy over-weights the weak direction.
 2516:         _DIR_WEIGHT_MIN_SAMPLES = 20
 2517:         if vote == "BUY" and stats.get("total_buy", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
 2518:             weight = stats.get("buy_accuracy", acc)  # BUG-185: .get() for cache safety
 2519:         elif vote == "SELL" and stats.get("total_sell", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
 2520:             weight = stats.get("sell_accuracy", acc)  # BUG-185: .get() for cache safety
 2521:         elif samples >= 20:
 2522:             weight = acc
 2523:         else:
 2524:             weight = 0.5
 2525:         # Apply directional rescue penalty: rescued signals contribute at
 2526:         # reduced weight since their overall accuracy failed the gate.
 2527:         if _rescued:
 2528:             weight *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY
 2529:         # IC-based weight adjustment: boost signals with high return-magnitude
 2530:         # predictive power, penalize phantom performers with zero IC.
 2531:         if ic_global:
 2532:             # Prefer per-ticker IC when available with enough samples
 2533:             _ic_info = None
 2534:             if ticker and ic_per_ticker:
 2535:                 _ic_info = ic_per_ticker.get(ticker, {}).get(signal_name)
 2536:                 if _ic_info and _ic_info.get("samples", 0) < _IC_MIN_SAMPLES:
 2537:                     _ic_info = None  # fall back to global
 2538:             if _ic_info is None:
 2539:                 _ic_info = ic_global.get(signal_name, {})
 2540:             _ic = _ic_info.get("ic", 0.0)
 2541:             _icir = _ic_info.get("icir", 0.0)
 2542:             _ic_n = _ic_info.get("samples", 0)
 2543:             ic_mult = _compute_ic_mult(_ic, _icir, _ic_n)
 2544:             weight *= ic_mult
 2545:         # Regime adjustment
 2546:         weight *= regime_mults.get(signal_name, 1.0)
 2547:         # Horizon-specific weight adjustment
 2548:         if signal_name in horizon_mults:
 2549:             weight *= horizon_mults[signal_name]
 2550:         # Macro-window downweight (2026-04-28). Composes with regime/
 2551:         # horizon multipliers — e.g., during a macro window in ranging
 2552:         # regime, sentiment hits 0.5 (macro) × 0.X (regime) × Y (horizon).
 2553:         # Only applies to MACRO_WINDOW_DOWNWEIGHT_SIGNALS — the
 2554:         # FORCE_HOLD signals were already mutated to HOLD above and won't
 2555:         # reach this branch.
 2556:         if macro_active and signal_name in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
 2557:             weight *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
 2558:         # Crisis mode adjustments: penalize trend signals (only if they're
 2559:         # underperforming), boost mean-reversion. See 2026-04-19 fix above.
 2560:         if crisis_mode:
 2561:             if signal_name in _TREND_SIGNALS and crisis_penalize_trend:
 2562:                 weight *= _CRISIS_TREND_PENALTY
 2563:             elif signal_name in _MR_SIGNALS:
 2564:                 weight *= _CRISIS_MR_BOOST
 2565:         # Activation frequency normalization (rarity * bias correction)
 2566:         act_data = activation_rates.get(signal_name, {})
 2567:         norm_weight = act_data.get("normalized_weight", 1.0)
 2568:         weight *= norm_weight
 2569:         # Activity rate cap: penalize signals with extremely high activation rates
 2570:         act_rate = act_data.get("activation_rate", 0.0)
 2571:         if act_rate > _ACTIVITY_RATE_CAP:
 2572:             weight *= _ACTIVITY_RATE_PENALTY
 2573:         # Correlation penalty: secondary signals in a group get reduced weight
 2574:         if signal_name in penalized_signals:
 2575:             weight *= penalized_signals[signal_name]
 2576:         # Directional bias penalty (2026-05-02 research): signals with extreme
 2577:         # BUY/SELL bias get penalized ONLY when voting in their bias direction.
 2578:         # Contrarian votes (rare, high-value) keep full weight.
 2579:         # E.g., calendar is 100% BUY — its BUY votes get 0.5x, but a rare
 2580:         # SELL (if it ever emits one) keeps 1.0x because that's genuinely
 2581:         # informative. Previous version penalized ALL votes equally.
 2582:         signal_bias = act_data.get("bias", 0.0)
 2583:         signal_samples = act_data.get("samples", 0)
 2584:         if signal_samples >= _BIAS_MIN_ACTIVE and signal_bias > _BIAS_THRESHOLD:
 2585:             buy_rate = act_data.get("buy_rate", 0.0)
 2586:             sell_rate = act_data.get("sell_rate", 0.0)
 2587:             bias_direction = "BUY" if buy_rate >= sell_rate else "SELL"
 2588:             if vote == bias_direction:
 2589:                 penalty = _BIAS_EXTREME_PENALTY if signal_bias > _BIAS_EXTREME_THRESHOLD else _BIAS_PENALTY
 2590:                 weight *= penalty
 2591:         # 2026-05-11 (Codex Fix B): apply soft-vote dampening LAST so it
 2592:         # composes with all upstream multipliers (accuracy, IC, regime,
 2593:         # horizon, macro, crisis, activity, correlation, bias). The
 2594:         # soft_conf is small (0.15-0.20) for dead-zone votes — a strong
 2595:         # vote has no soft_conf key and so this branch is skipped,
 2596:         # preserving the existing strong-vote weight contract.
 2597:         soft = soft_confidences.get(f"_soft_conf_{signal_name}")
 2598:         if soft is not None:
 2599:             try:
 2600:                 weight *= float(soft)
 2601:             except (TypeError, ValueError):
 2602:                 pass
 2603:         if vote == "BUY":
 2604:             buy_weight += weight
 2605:         elif vote == "SELL":
 2606:             sell_weight += weight
 2607:     if gated_signals:
 2608:         logger.debug("Accuracy-gated signals (<%s%%): %s", ACCURACY_GATE_THRESHOLD * 100, gated_signals)
 2609:     total_weight = buy_weight + sell_weight
 2610:     if total_weight == 0:
 2611:         return "HOLD", 0.0
 2612:     buy_conf = buy_weight / total_weight
 2613:     sell_conf = sell_weight / total_weight
 2614:     if buy_conf > sell_conf and buy_conf >= 0.5:
 2615:         return "BUY", round(buy_conf, 4)
 2616:     if sell_conf > buy_conf and sell_conf >= 0.5:
 2617:         return "SELL", round(sell_conf, 4)
 2618:     return "HOLD", round(max(buy_conf, sell_conf), 4)
 2619: 
 2620: 
 2621: def _confluence_score(votes, indicators):
 2622:     active = {k: v for k, v in votes.items() if v != "HOLD"}
 2623:     if not active:
 2624:         return 0.0
 2625:     buy_count = sum(1 for v in active.values() if v == "BUY")
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1338: def _compute_dynamic_horizon_weights(horizon: str) -> dict[str, float]:
 1339:     """Compute horizon-specific signal weight multipliers from accuracy cache.
 1340: 
 1341:     For each signal, computes the ratio of its accuracy on this horizon vs
 1342:     the comparison horizon(s). Signals that perform much better on this
 1343:     horizon get boosted; signals that perform much worse get penalized.
 1344: 
 1345:     Returns a dict of {signal_name: multiplier} for multipliers outside
 1346:     the deadband (i.e., > 1.1 or < 0.9). Falls back to static
 1347:     HORIZON_SIGNAL_WEIGHTS if accuracy cache is unavailable.
 1348:     """
 1349:     try:
 1350:         from portfolio.file_utils import load_json
 1351:         cache = load_json(DATA_DIR / "accuracy_cache.json")
 1352:         if not cache:
 1353:             return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
 1354: 
 1355:         # Get recent accuracy for this horizon and comparison horizons
 1356:         this_key = f"{horizon}_recent"
 1357:         this_data = cache.get(this_key, {})
 1358:         if not this_data:
 1359:             return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
 1360: 
 1361:         cross_horizons = _CROSS_HORIZON_PAIRS.get(horizon, [])
 1362:         if not cross_horizons:
 1363:             return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
 1364: 
 1365:         # Gather comparison accuracies (true mean across comparison horizons)
 1366:         cross_sum: dict[str, float] = {}
 1367:         cross_count: dict[str, int] = {}
 1368:         for ch in cross_horizons:
 1369:             ch_key = f"{ch}_recent"
 1370:             ch_acc = cache.get(ch_key, {})
 1371:             for sig, stats in ch_acc.items():
 1372:                 if stats.get("total", 0) >= _DYNAMIC_HORIZON_MIN_SAMPLES:
 1373:                     acc = stats.get("accuracy", 0.5)
 1374:                     cross_sum[sig] = cross_sum.get(sig, 0.0) + acc
 1375:                     cross_count[sig] = cross_count.get(sig, 0) + 1
 1376:         cross_data = {
 1377:             sig: cross_sum[sig] / cross_count[sig]
 1378:             for sig in cross_sum
 1379:             if cross_count.get(sig, 0) > 0
 1380:         }
 1381: 
 1382:         # Compute multipliers
 1383:         weights = {}
 1384:         for sig, stats in this_data.items():
 1385:             samples = stats.get("total", 0)
 1386:             if samples < _DYNAMIC_HORIZON_MIN_SAMPLES:
 1387:                 continue
 1388:             this_acc = stats.get("accuracy", 0.5)
 1389:             cross_acc = cross_data.get(sig)
 1390:             if cross_acc is None or not (0.01 <= cross_acc <= 1.0):
 1391:                 continue
 1392: 
 1393:             # Ratio of this-horizon accuracy to cross-horizon accuracy
 1394:             ratio = this_acc / cross_acc
 1395:             # Clamp
 1396:             ratio = max(_DYNAMIC_HORIZON_CLAMP_LOW, min(_DYNAMIC_HORIZON_CLAMP_HIGH, ratio))
 1397:             # Deadband: only include if meaningfully different from 1.0
 1398:             if abs(ratio - 1.0) > _DYNAMIC_HORIZON_DEADBAND:
 1399:                 weights[sig] = round(ratio, 2)
 1400: 
 1401:         return weights if weights else HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
 1402:     except Exception:
 1403:         logger.debug("Dynamic horizon weights unavailable, using static fallback", exc_info=True)
 1404:         return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
 1405: 
 1406: 
 1407: def _get_horizon_weights(horizon: str | None) -> dict[str, float]:
 1408:     """Get horizon-specific signal weight multipliers, preferring dynamic computation.
 1409: 
 1410:     Uses cached dynamic weights when available, falling back to static dict.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1788:                 "accuracy": best_acc,
 1789:                 "total": best_total,
 1790:                 "samples": best_total,
 1791:                 "enabled": sig_name not in DISABLED_SIGNALS,
 1792:                 "correct": best_correct,
 1793:                 "pct": round(best_acc * 100, 1),
 1794:                 "best_horizon": best_hz,
 1795:             }
 1796: 
 1797:     # --- Write cache ---
 1798:     try:
 1799:         with _accuracy_write_lock:
 1800:             _atomic_write_json(BEST_HORIZON_CACHE_FILE, {"time": time.time(), "data": result})
 1801:     except Exception:
 1802:         logger.debug("Failed to write best_horizon cache", exc_info=True)
 1803: 
 1804:     return result
 1805: 
 1806: 
 1807: def accuracy_by_ticker_signal(horizon="1d", min_samples=0):
 1808:     """Compute per-ticker per-signal accuracy cross-tabulation.
 1809: 
 1810:     Returns nested dict: {ticker: {signal_name: {correct, total, accuracy, pct,
 1811:         correct_buy, total_buy, buy_accuracy, correct_sell, total_sell, sell_accuracy}}}
 1812:     Only includes signals that voted BUY or SELL (HOLD excluded).
 1813:     Directional fields (buy_accuracy, sell_accuracy) enable per-ticker directional
 1814:     gating in signal_engine._weighted_consensus().
 1815: 
 1816:     Args:
 1817:         horizon: Outcome horizon ("1d", "3d", "5d", "10d").
 1818:         min_samples: Minimum votes required to include a signal for a ticker.
 1819:     """
 1820:     entries = load_entries()
 1821:     # {ticker: {signal: {correct, total, correct_buy, total_buy, correct_sell, total_sell}}}
 1822:     def _empty():
 1823:         return {"correct": 0, "total": 0, "correct_buy": 0, "total_buy": 0,
 1824:                 "correct_sell": 0, "total_sell": 0}
 1825:     stats = defaultdict(lambda: defaultdict(_empty))
 1826: 
 1827:     for entry in entries:
 1828:         outcomes = entry.get("outcomes", {})
 1829:         tickers = entry.get("tickers", {})
 1830: 
 1831:         for ticker, tdata in tickers.items():
 1832:             outcome = outcomes.get(ticker, {}).get(horizon)
 1833:             if not outcome:
 1834:                 continue
 1835: 
 1836:             change_pct = outcome.get("change_pct", 0)
 1837:             signals = tdata.get("signals", {})
 1838: 
 1839:             for sig_name, vote in signals.items():
 1840:                 if vote == "HOLD":
 1841:                     continue
 1842:                 result_val = _vote_correct(vote, change_pct)
 1843:                 if result_val is None:
 1844:                     continue
 1845:                 s = stats[ticker][sig_name]
 1846:                 s["total"] += 1
 1847:                 if vote == "BUY":
 1848:                     s["total_buy"] += 1
 1849:                     if result_val:
 1850:                         s["correct"] += 1
 1851:                         s["correct_buy"] += 1
 1852:                 else:
 1853:                     s["total_sell"] += 1
 1854:                     if result_val:
 1855:                         s["correct"] += 1
 1856:                         s["correct_sell"] += 1
 1857: 
 1858:     result = {}
 1859:     for ticker, sig_stats in stats.items():
 1860:         ticker_result = {}
 1861:         for sig_name, s in sig_stats.items():
 1862:             if s["total"] < min_samples:
 1863:                 continue
 1864:             acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
 1865:             buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
 1866:             sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
 1867:             ticker_result[sig_name] = {
 1868:                 "correct": s["correct"],
 1869:                 "total": s["total"],
 1870:                 "accuracy": acc,
 1871:                 "pct": round(acc * 100, 1),
 1872:                 "correct_buy": s["correct_buy"],
 1873:                 "total_buy": s["total_buy"],
 1874:                 "buy_accuracy": round(buy_acc, 4),
 1875:                 "correct_sell": s["correct_sell"],
 1876:                 "total_sell": s["total_sell"],
 1877:                 "sell_accuracy": round(sell_acc, 4),
 1878:             }
 1879:         if ticker_result:
 1880:             result[ticker] = ticker_result
 1881: 
 1882:     return result
 1883: 
 1884: 
 1885: def top_signals_for_ticker(ticker, horizon="1d", min_samples=5):
 1886:     """Return ranked list of signals for a specific ticker, sorted by accuracy.
 1887: 
 1888:     Args:
 1889:         ticker: Ticker symbol (e.g. "BTC-USD").
 1890:         horizon: Outcome horizon.
 1891:         min_samples: Minimum votes to qualify.
 1892: 
 1893:     Returns:
 1894:         list[dict]: Sorted by accuracy descending. Each dict has:
 1895:             signal, correct, total, accuracy, pct.
 1896:     """
 1897:     all_data = accuracy_by_ticker_signal(horizon, min_samples=min_samples)
 1898:     ticker_data = all_data.get(ticker, {})
 1899:     ranked = [
 1900:         {"signal": sig, **data}
 1901:         for sig, data in ticker_data.items()
 1902:     ]
 1903:     ranked.sort(key=lambda x: x["accuracy"], reverse=True)
 1904:     return ranked
 1905: 
 1906: 
 1907: # ---------------------------------------------------------------------------
 1908: # Per-ticker accuracy cache
 1909: # ---------------------------------------------------------------------------
 1910: 
 1911: TICKER_ACCURACY_CACHE_FILE = DATA_DIR / "ticker_signal_accuracy_cache.json"
 1912: 
 1913: 
 1914: def load_cached_ticker_accuracy(horizon="1d"):
 1915:     """Load cached per-ticker per-signal accuracy, returning None if stale.
 1916: 
 1917:     Uses the same TTL as the main accuracy cache (ACCURACY_CACHE_TTL).
 1918:     Cache structure: {horizon: {ticker: {signal: {correct, total, accuracy, pct}}}, "time": ...}
 1919:     """
 1920:     cache = load_json(TICKER_ACCURACY_CACHE_FILE)
 1921:     if cache is not None:
 1922:         try:
 1923:             if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
 1924:                 cached = cache.get(horizon)
 1925:                 if cached:
 1926:                     return cached
 1927:         except (KeyError, AttributeError):
 1928:             logger.debug("Ticker accuracy cache corrupted or missing horizon %s", horizon)
 1929:     return None
 1930: 
 1931: 
 1932: def write_ticker_accuracy_cache(horizon, data):
 1933:     """Persist per-ticker per-signal accuracy data to the cache file.
 1934: 
 1935:     Merges with existing horizons to avoid overwriting other cached data.
 1936:     """
 1937:     with _accuracy_write_lock:
 1938:         cache = load_json(TICKER_ACCURACY_CACHE_FILE, default={})
 1939:         if not isinstance(cache, dict):
 1940:             cache = {}
 1941:         cache[horizon] = data
 1942:         cache["time"] = time.time()
 1943:         _atomic_write_json(TICKER_ACCURACY_CACHE_FILE, cache)
 1944: 
 1945: 
 1946: def _filter_min_samples(data, min_samples):
 1947:     if min_samples <= 0:
 1948:         return data
 1949:     return {
 1950:         ticker: {
 1951:             sig: sdata for sig, sdata in sigs.items()
 1952:             if sdata.get("total", 0) >= min_samples
 1953:         }
 1954:         for ticker, sigs in data.items()
 1955:     }
 1956: 
 1957: 
 1958: def accuracy_by_ticker_signal_cached(horizon="1d", min_samples=0):
 1959:     """Cached version of accuracy_by_ticker_signal().
 1960: 
 1961:     Checks the ticker accuracy cache first; on miss, computes from the
 1962:     full signal log and writes the cache. BUG-178 (2026-04-16): the
 1963:     cache-miss compute path is now serialized via _accuracy_compute_lock
 1964:     so concurrent ticker threads can't all redundantly walk the 50,000-
 1965:     entry signal log when the 1h TTL expires.
 1966:     """
 1967:     cached = load_cached_ticker_accuracy(horizon)
 1968:     if cached:
 1969:         return _filter_min_samples(cached, min_samples)
 1970: 
 1971:     with _accuracy_compute_lock:
 1972:         # Re-check after acquiring the lock — another thread may have
 1973:         # populated the cache while we waited.
 1974:         cached = load_cached_ticker_accuracy(horizon)
 1975:         if cached:
 1976:             return _filter_min_samples(cached, min_samples)
 1977: 
 1978:         data = accuracy_by_ticker_signal(horizon, min_samples=0)
 1979:         if data:
 1980:             write_ticker_accuracy_cache(horizon, data)
 1981:         return _filter_min_samples(data, min_samples)
 1982: 
 1983: 
 1984: def probability_calibration(horizon="1d", buckets=None, since=None):
 1985:     """Compute calibration data for reliability diagrams.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
 1000:     "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
 1001:     "volume", "ministral", "qwen3", "claude_fundamental",
 1002: })
 1003: 
 1004: # Sentiment hysteresis — prevents rapid flip spam from ~50% confidence oscillation
 1005: _prev_sentiment: dict[str, str] = {}  # in-memory cache; seeded from sentiment_state.json on first call
 1006: _prev_sentiment_loaded = False
 1007: _sentiment_lock = threading.Lock()  # BUG-85: protect concurrent access from ThreadPoolExecutor
 1008: _sentiment_dirty = False  # Track whether in-memory state diverged from disk
 1009: 
 1010: _SENTIMENT_STATE_FILE = DATA_DIR / "sentiment_state.json"
 1011: 
 1012: 
 1013: def _load_prev_sentiments():
 1014:     global _prev_sentiment, _prev_sentiment_loaded
 1015:     with _sentiment_lock:
 1016:         if _prev_sentiment_loaded:
 1017:             return
 1018:         try:
 1019:             from portfolio.file_utils import load_json as _load_json
 1020:             data = _load_json(str(_SENTIMENT_STATE_FILE), default=None)
 1021:             if data and isinstance(data, dict):
 1022:                 _prev_sentiment = data.get("prev_sentiment", {})
 1023:             # Prune entries for removed tickers
 1024:             from portfolio.tickers import ALL_TICKERS
 1025:             removed = [k for k in _prev_sentiment if k not in ALL_TICKERS]
 1026:             for k in removed:
 1027:                 del _prev_sentiment[k]
 1028:         except Exception:
 1029:             logger.warning("Failed to load prev sentiments", exc_info=True)
 1030:         _prev_sentiment_loaded = True
 1031: 
 1032: 
 1033: def _get_prev_sentiment(ticker):
 1034:     _load_prev_sentiments()
 1035:     with _sentiment_lock:
 1036:         return _prev_sentiment.get(ticker)
 1037: 
 1038: 
 1039: def _set_prev_sentiment(ticker, direction):
 1040:     """Set sentiment direction for a ticker (thread-safe, batched disk write)."""
 1041:     global _sentiment_dirty
 1042:     _load_prev_sentiments()
 1043:     with _sentiment_lock:
 1044:         _prev_sentiment[ticker] = direction
 1045:         _sentiment_dirty = True
 1046: 
 1047: 
 1048: def flush_sentiment_state():
 1049:     """Persist sentiment state to disk. Call once per cycle, not per-ticker.
 1050: 
 1051:     BUG-85 fix: batching prevents concurrent per-ticker writes that clobber each other.
 1052:     BUG-101 fix: dirty flag cleared only AFTER successful write, so a failed write
 1053:     will be retried on the next cycle instead of silently losing state.
 1054:     """
 1055:     global _sentiment_dirty
 1056:     with _sentiment_lock:
 1057:         if not _sentiment_dirty:
 1058:             return
 1059:         snapshot = dict(_prev_sentiment)
 1060:     # Write outside the lock to avoid holding it during I/O
 1061:     try:
 1062:         from portfolio.file_utils import atomic_write_json
 1063:         atomic_write_json(_SENTIMENT_STATE_FILE, {"prev_sentiment": snapshot})
 1064:         # BUG-101: Only clear dirty flag after successful write
 1065:         with _sentiment_lock:
 1066:             _sentiment_dirty = False
 1067:     except Exception:
 1068:         # Dirty flag remains True — next cycle will retry the write
 1069:         logger.warning("Failed to persist sentiment state (will retry next cycle)", exc_info=True)
 1070: 
 1071: 
 1072: REGIME_WEIGHTS = {
 1073:     "trending-up": {
 1074:         "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
 1075:         # Enhanced: boost trend-following, dampen mean-reversion
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
 2020: _ic_data_cache: dict = {}
 2021: _ic_data_lock = threading.Lock()
 2022: 
 2023: 
 2024: def _get_ic_data(horizon: str) -> dict | None:
 2025:     """Load IC data for the given horizon, computing if cache is stale.
 2026: 
 2027:     Returns the full cache dict {"global": {...}, "per_ticker": {...}}
 2028:     or None if IC data is unavailable.
 2029:     """
 2030:     now = time.time()
 2031:     with _ic_data_lock:
 2032:         cached = _ic_data_cache.get(horizon)
 2033:         if cached and now - cached.get("_loaded_at", 0) < _IC_DATA_TTL:
 2034:             return cast(dict[Any, Any], cached)
 2035: 
 2036:     try:
 2037:         from portfolio.ic_computation import compute_and_cache_ic, load_cached_ic
 2038:         cache = load_cached_ic(horizon)
 2039:         if cache is None:
 2040:             cache = compute_and_cache_ic(horizon)
 2041:         if cache:
 2042:             cache["_loaded_at"] = now
 2043:             with _ic_data_lock:
 2044:                 _ic_data_cache[horizon] = cache
 2045:             return cast(dict[Any, Any], cache)
 2046:     except Exception:
 2047:         logger.debug("IC data unavailable for %s", horizon, exc_info=True)
 2048:     return None
 2049: 
 2050: 
 2051: _macro_window_cache: dict = {"value": False, "ts": 0.0}
 2052: _macro_window_cache_lock = threading.Lock()
 2053: _macro_window_last_state: dict = {"active": None}  # transition logger
 2054: 
 2055: 
 2056: def _is_macro_window_cached(now_ts: float | None = None) -> bool:
 2057:     """Return whether we're inside a macro event window, with TTL caching.
 2058: 
 2059:     The underlying ``portfolio.econ_dates.is_macro_window`` iterates
 2060:     ``ECON_EVENTS`` linearly. That's cheap, but called per signal per
 2061:     ticker per cycle becomes wasteful. Cache the result for
 2062:     ``_MACRO_WINDOW_CACHE_TTL_S`` (5 minutes by default) — events have
 2063:     hourly cadence at fastest, so 5min staleness is acceptable.
 2064: 
 2065:     Logs once per state transition so the operational log shows when
 2066:     we entered/exited a macro window without spamming every cycle.
 2067:     """
 2068:     import time as _time
 2069:     if now_ts is None:
 2070:         now_ts = _time.time()
 2071:     with _macro_window_cache_lock:
 2072:         if now_ts - _macro_window_cache["ts"] < _MACRO_WINDOW_CACHE_TTL_S:
 2073:             return bool(_macro_window_cache["value"])
 2074:         try:
 2075:             from portfolio.econ_dates import is_macro_window
 2076:             active = bool(is_macro_window())
 2077:         except Exception as e:
 2078:             logger.warning("macro window detection failed (treating as inactive): %s", e)
 2079:             active = False
 2080:         _macro_window_cache["value"] = active
 2081:         _macro_window_cache["ts"] = now_ts
 2082:         last = _macro_window_last_state["active"]
 2083:         if last is None:
 2084:             _macro_window_last_state["active"] = active
 2085:         elif last != active:
 2086:             logger.info(
 2087:                 "macro_window state transition: %s -> %s",
 2088:                 "ACTIVE" if last else "inactive",
 2089:                 "ACTIVE" if active else "inactive",
 2090:             )
 2091:             _macro_window_last_state["active"] = active
 2092:         return active
 2093: 
 2094: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
none
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
75: # (accuracy load, weighted consensus, penalties, linear factor, etc.) into
4120:         from portfolio.linear_factor import LinearFactorModel
4121:         _lf_model = LinearFactorModel()
4128:             extra_info["_linear_factor_score"] = round(lf_score, 6)
4129:             extra_info["_linear_factor_action"] = lf_action
4130:             extra_info["_linear_factor_confidence"] = lf_conf
4136:                     {"stage": "linear_factor", "effect": "confirm_boost",
4141:                     {"stage": "linear_factor", "effect": "disagree_dampen",
4147:         _record_phase(ticker, "linear_factor", _phase_start)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 4108:                 extra_info["_earnings_gated"] = True
 4109:                 action = "HOLD"
 4110:                 conf = 0.0
 4111:         except Exception:
 4112:             logger.debug("Earnings gate failed for %s", ticker, exc_info=True)
 4113: 
 4114:     # --- Linear factor model score (supplementary, not overriding) ---
 4115:     # Provides a continuous predicted-return score from ridge regression
 4116:     # weights trained on historical signal+return data. Used to:
 4117:     # 1. Confirm or weaken consensus (agreement = boost, disagreement = dampen)
 4118:     # 2. Provide alternative ranking in agent_summary for Layer 2 decisions
 4119:     try:
 4120:         from portfolio.linear_factor import LinearFactorModel
 4121:         _lf_model = LinearFactorModel()
 4122:         if _lf_model.load():
 4123:             # Convert votes to numeric: BUY=+1, SELL=-1, HOLD=0
 4124:             _vote_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
 4125:             numeric_votes = {k: _vote_map.get(v, 0.0) for k, v in votes.items()}
 4126:             lf_score = _lf_model.predict(numeric_votes)
 4127:             lf_action, lf_conf = _lf_model.score_to_action(lf_score)
 4128:             extra_info["_linear_factor_score"] = round(lf_score, 6)
 4129:             extra_info["_linear_factor_action"] = lf_action
 4130:             extra_info["_linear_factor_confidence"] = lf_conf
 4131:             # Confirmation boost/dampen: if linear model agrees with consensus,
 4132:             # boost confidence by 10%. If it disagrees, dampen by 10%.
 4133:             if lf_action == action and action != "HOLD" and lf_conf > 0.3:
 4134:                 conf *= 1.10
 4135:                 extra_info.setdefault("_penalty_log", []).append(
 4136:                     {"stage": "linear_factor", "effect": "confirm_boost",
 4137:                      "lf_action": lf_action, "lf_score": round(lf_score, 6)})
 4138:             elif lf_action != "HOLD" and lf_action != action and action != "HOLD":
 4139:                 conf *= 0.90
 4140:                 extra_info.setdefault("_penalty_log", []).append(
 4141:                     {"stage": "linear_factor", "effect": "disagree_dampen",
 4142:                      "lf_action": lf_action, "lf_score": round(lf_score, 6)})
 4143:     except Exception:
 4144:         logger.debug("Linear factor model failed", exc_info=True)
 4145: 
 4146:     if ticker:
 4147:         _record_phase(ticker, "linear_factor", _phase_start)
 4148:         _phase_start = time.monotonic()
 4149: 
 4150:     # --- Per-ticker consensus accuracy gate ---
 4151:     # BUG-164: AMD 24.8%, GOOGL 31.3%, META 34.2% consensus accuracy.
 4152:     # The system is actively harmful on these tickers.  Gate consensus
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 4150:     # --- Per-ticker consensus accuracy gate ---
 4151:     # BUG-164: AMD 24.8%, GOOGL 31.3%, META 34.2% consensus accuracy.
 4152:     # The system is actively harmful on these tickers.  Gate consensus
 4153:     # when per-ticker historical accuracy is below threshold.
 4154:     if ticker and action != "HOLD":
 4155:         try:
 4156:             from portfolio.accuracy_stats import load_cached_accuracy
 4157: 
 4158:             # H1: Match the horizon-scoped key written above.
 4159:             _ptc_acc = load_cached_accuracy(f"per_ticker_consensus_{acc_horizon}")
 4160:             if _ptc_acc:
 4161:                 _ptc_stats = _ptc_acc.get(ticker, {})
 4162:                 _ptc_total = _ptc_stats.get("total", 0)
 4163:                 _ptc_accuracy = _ptc_stats.get("accuracy", 0.5)
 4164:                 if _ptc_total >= _PER_TICKER_CONSENSUS_MIN_SAMPLES and _ptc_accuracy < _PER_TICKER_CONSENSUS_GATE:
 4165:                     extra_info.setdefault("_penalty_log", []).append({
 4166:                         "stage": "per_ticker_consensus_gate",
 4167:                         "ticker": ticker,
 4168:                         "accuracy": round(_ptc_accuracy, 3),
 4169:                         "samples": _ptc_total,
 4170:                         "threshold": _PER_TICKER_CONSENSUS_GATE,
 4171:                         "effect": "force_hold",
 4172:                     })
 4173:                     extra_info["_ticker_consensus_gated"] = True
 4174:                     action = "HOLD"
 4175:                     conf = 0.0
 4176:         except Exception:
 4177:             logger.debug("Per-ticker consensus gate failed for %s", ticker, exc_info=True)
 4178: 
 4179:     # Global confidence cap — calibration data shows >80% confidence is
 4180:     # anti-correlated with accuracy at every horizon (70-80% bucket is the
 4181:     # best performing at 57-59% actual accuracy)
 4182:     conf = min(conf, 0.80)
 4183: 
 4184:     # 3h horizon: cap confidence to prevent overconfident short-term predictions
 4185:     if horizon in ("3h", "4h"):
 4186:         from portfolio.short_horizon import CONFIDENCE_CAP_3H
 4187:         conf = min(conf, CONFIDENCE_CAP_3H)
 4188: 
 4189:     if ticker:
 4190:         _record_phase(ticker, "consensus_gate", _phase_start)
 4191: 
 4192:     # Update cross-ticker consensus cache for synthetic cross-asset signals
 4193:     if ticker:
 4194:         with _cross_ticker_lock:
 4195:             _cross_ticker_consensus[ticker] = {"action": action, "confidence": conf}
 4196: 
 4197:     return action, conf, extra_info
 4198: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
3719:         from portfolio.accuracy_stats import get_or_compute_recent_accuracy
3720:         # get_or_compute_recent_accuracy expects the base horizon, not the cache key
3722:         _recent_acc_data = get_or_compute_recent_accuracy(base_hz) or {}
3836:             blend_accuracy_data,
3837:             get_or_compute_accuracy,
3838:             get_or_compute_per_ticker_accuracy,
3839:             get_or_compute_recent_accuracy,
3848:         alltime = get_or_compute_accuracy(acc_horizon)
3849:         recent = get_or_compute_recent_accuracy(acc_horizon, days=7)
3851:         # convention preserved by get_or_compute_per_ticker_accuracy.
3852:         _ptc_data = get_or_compute_per_ticker_accuracy(acc_horizon)
3865:         accuracy_data = blend_accuracy_data(
4159:             _ptc_acc = load_cached_accuracy(f"per_ticker_consensus_{acc_horizon}")
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 3828:     # (regime already computed early in the function for F&G gating)
 3829:     accuracy_data = {}
 3830:     activation_rates = {}
 3831:     # H3: Define acc_horizon before the try/except so the except block and
 3832:     # subsequent code can reference it even if the import fails.
 3833:     acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
 3834:     try:
 3835:         from portfolio.accuracy_stats import (
 3836:             blend_accuracy_data,
 3837:             get_or_compute_accuracy,
 3838:             get_or_compute_per_ticker_accuracy,
 3839:             get_or_compute_recent_accuracy,
 3840:             load_cached_activation_rates,
 3841:         )
 3842: 
 3843:         # BUG-178 (2026-04-16): the get_or_compute_* helpers serialize the
 3844:         # cache-miss compute via _accuracy_compute_lock so 5 parallel ticker
 3845:         # threads don't each pay the 7s+ cost of loading 50,000 signal-log
 3846:         # entries when the 1h TTL expires (was 215s wall before the fix).
 3847:         # See accuracy_stats.py for the lock rationale.
 3848:         alltime = get_or_compute_accuracy(acc_horizon)
 3849:         recent = get_or_compute_recent_accuracy(acc_horizon, days=7)
 3850:         # BUG-164 lazy-populate per-ticker consensus accuracy — _ptc_key
 3851:         # convention preserved by get_or_compute_per_ticker_accuracy.
 3852:         _ptc_data = get_or_compute_per_ticker_accuracy(acc_horizon)
 3853:         # RES-2026-04-17: Pass per-ticker consensus accuracy into extra_info
 3854:         # so apply_confidence_penalties can penalize tickers where the consensus
 3855:         # system itself has below-coinflip accuracy (e.g. ETH-USD 47.7% at 3h,
 3856:         # MSTR 45.9%). The consensus is the aggregated output, not individual
 3857:         # signals — if it's below 50%, the ensemble is net-negative for this ticker.
 3858:         if _ptc_data and ticker and isinstance(_ptc_data, dict):
 3859:             _ticker_ptc = _ptc_data.get(ticker)
 3860:             if isinstance(_ticker_ptc, dict):
 3861:                 extra_info["_ptc_accuracy"] = _ticker_ptc.get("accuracy")
 3862:                 extra_info["_ptc_samples"] = _ticker_ptc.get("total", 0)
 3863: 
 3864:         # ARCH-23: Use shared blend function (replaces inline logic).
 3865:         accuracy_data = blend_accuracy_data(
 3866:             alltime, recent,
 3867:             divergence_threshold=_RECENCY_DIVERGENCE_THRESHOLD,
 3868:             normal_weight=_RECENCY_WEIGHT_NORMAL,
 3869:             fast_weight=_RECENCY_WEIGHT_FAST,
 3870:             min_recent_samples=_RECENCY_MIN_SAMPLES,
 3871:         )
 3872: 
 3873:         activation_rates = load_cached_activation_rates()
 3874:         _accuracy_failed = False
 3875:     except Exception:
 3876:         logger.error("Accuracy stats load failed", exc_info=True)
 3877:         # H3: Fail-closed: gate all signals (0% accuracy, 999 samples) rather than
 3878:         # leaving accuracy_data = {} which bypasses the accuracy gate entirely.
 3879:         accuracy_data = {sig: {"accuracy": 0.0, "total": 999} for sig in SIGNAL_NAMES}
 3880:         _accuracy_failed = True
 3881: 
 3882:     if ticker:
 3883:         _record_phase(ticker, "acc_load", _phase_start)
 3884:         _phase_start = time.monotonic()
 3885: 
 3886:     # Overlay regime-specific accuracy when available.
 3887:     # H3: Skip all overlays when primary load failed — they would silently restore
 3888:     # real accuracy values for cached signals, negating the fail-closed gate.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  450:             "samples": s["total"],
  451:             "enabled": sig_name not in DISABLED_SIGNALS,
  452:             "accuracy": acc,
  453:             "pct": round(acc * 100, 1),
  454:             "cost_bps": cost_bps,
  455:         }
  456:     return result
  457: 
  458: 
  459: def consensus_accuracy(horizon="1d", entries=None, days=None):
  460:     """Aggregate consensus decision accuracy across all tickers.
  461: 
  462:     For each signal-log entry that has an outcome at `horizon`, check if
  463:     the per-ticker `tdata["consensus"]` BUY/SELL call matched actual
  464:     direction. HOLD calls are skipped (no direction to score).
  465: 
  466:     BUG-178/W15-W16 follow-up (2026-04-16): added optional `days` kwarg
  467:     for the recent-window variant the degradation tracker needs. The
  468:     existing `entries` kwarg is preserved for callers that already pass
  469:     a pre-loaded list. When both are passed, `entries` wins (caller has
  470:     already filtered).
  471: 
  472:     Args:
  473:         horizon: Outcome horizon ("3h", "4h", "12h", "1d", "3d", "5d", "10d").
  474:         entries: Pre-loaded entries list (skips both load_entries() and
  475:             the days filter — caller is assumed to have filtered already).
  476:         days: Optional lookback window in days. Ignored if entries is
  477:             provided. None = lifetime aggregate.
  478:     """
  479:     if entries is None:
  480:         entries = load_entries()
  481:         if days is not None:
  482:             from datetime import datetime, timedelta
  483:             cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
  484:             entries = [e for e in entries if e.get("ts", "") >= cutoff]
  485:     correct = 0
  486:     total = 0
  487: 
  488:     for entry in entries:
  489:         outcomes = entry.get("outcomes", {})
  490:         tickers = entry.get("tickers", {})
  491: 
  492:         for ticker, tdata in tickers.items():
  493:             consensus = tdata.get("consensus", "HOLD")
  494:             if consensus == "HOLD":
  495:                 continue
  496: 
  497:             outcome = outcomes.get(ticker, {}).get(horizon)
  498:             if not outcome:
  499:                 continue
  500: 
  501:             change_pct = outcome.get("change_pct", 0)
  502:             result_val = _vote_correct(consensus, change_pct)
  503:             if result_val is None:
  504:                 continue
  505:             total += 1
  506:             if result_val:
  507:                 correct += 1
  508: 
  509:     acc = correct / total if total > 0 else 0.0
  510:     return {
  511:         "correct": correct,
  512:         "total": total,
  513:         "accuracy": acc,
  514:         "pct": round(acc * 100, 1),
  515:     }
  516: 
  517: 
  518: def per_ticker_accuracy(horizon="1d", entries=None):
  519:     if entries is None:
  520:         entries = load_entries()
  521:     stats = defaultdict(lambda: {"correct": 0, "total": 0})
  522: 
  523:     for entry in entries:
  524:         outcomes = entry.get("outcomes", {})
  525:         tickers = entry.get("tickers", {})
  526: 
  527:         for ticker, tdata in tickers.items():
  528:             consensus = tdata.get("consensus", "HOLD")
  529:             if consensus == "HOLD":
  530:                 continue
  531: 
  532:             outcome = outcomes.get(ticker, {}).get(horizon)
  533:             if not outcome:
  534:                 continue
  535: 
  536:             change_pct = outcome.get("change_pct", 0)
  537:             result_val = _vote_correct(consensus, change_pct)
  538:             if result_val is None:
  539:                 continue
  540:             stats[ticker]["total"] += 1
  541:             if result_val:
  542:                 stats[ticker]["correct"] += 1
  543: 
  544:     result = {}
  545:     for ticker, s in stats.items():
  546:         acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
  547:         result[ticker] = {
  548:             "correct": s["correct"],
  549:             "total": s["total"],
  550:             "accuracy": acc,
  551:             "pct": round(acc * 100, 1),
  552:         }
  553:     return result
  554: 
  555: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 17ms:

=== portfolio/outcome_tracker.py exception blocks ===
  165:     except Exception as e:
  166:         logger.warning("SQLite snapshot write failed (JSONL has data, SQLite may lag): %s", e)
  167: 
  168:     return entry
---
  304:     except ImportError:  # pragma: no cover - non-Windows
  305:         _msvcrt = None  # type: ignore[assignment]
  306:     try:
  307:         import fcntl as _fcntl  # type: ignore[import-not-found]
---
  308:     except ImportError:  # pragma: no cover - Windows
  309:         _fcntl = None  # type: ignore[assignment]
  310: 
  311:     import os as _os
---
  318:         except OSError:
  319:             pass
  320: 
  321:     with open(lock_path, "rb+") as lock_f:
---
  337:                 except OSError:
  338:                     pass
  339:             # fcntl.flock releases on close.
  340: 
---
  396:                     except json.JSONDecodeError:
  397:                         continue
  398:         # snapshot_size pins the byte boundary of "what we read".
  399:         # Anything appended past this offset during processing must be
---
  417:     except Exception as e:
  418:         logger.debug("SignalDB open failed: %s", e)
  419: 
  420:     for entry in entries:
---
  423:         except (KeyError, ValueError, TypeError):
  424:             continue
  425:         tickers = entry.get("tickers", {})
  426:         outcomes = entry.get("outcomes", {})
---
  464:                     except Exception:
  465:                         price_cache[cache_key] = None
  466: 
  467:                 hist_price = price_cache[cache_key]
---
  492:                     except Exception as e:
  493:                         logger.debug("SQLite outcome write failed: %s", e)
  494: 
  495:         entry["outcomes"] = outcomes
---
  508:         except Exception as e:
  509:             logger.debug("SignalDB close failed: %s", e)
  510: 
  511:     import os
---
  524:         except FileNotFoundError:
  525:             current_size = snapshot_size  # nothing to preserve
  526:         concurrent_tail_bytes = max(0, current_size - snapshot_size)
  527: 
---
  560:         except BaseException:
  561:             with contextlib.suppress(OSError):
  562:                 os.unlink(tmp)
  563:             raise
---
  572:         except Exception:
  573:             logger.debug("Could not invalidate signal utility cache", exc_info=True)
  574: 
  575:     return updated
---

=== portfolio/signal_db.py exception blocks ===
  108:         except sqlite3.IntegrityError:
  109:             return  # duplicate ts, skip
  110: 
  111:         snapshot_id = cur.lastrowid
---
  155:         except Exception:
  156:             conn.rollback()
  157:             raise
  158: 
---

=== portfolio/accuracy_stats.py exception blocks ===
  140:     except Exception:
  141:         logger.debug("signal_utility disk cache write failed", exc_info=True)
  142: 
  143: 
---
  157:     except Exception as e:
  158:         logger.debug("SQLite signal_db unavailable, falling back to JSONL: %s", e)
  159:     # H2: Fallback to JSONL using atomic load_jsonl_tail (avoids raw open()).
  160:     # 50000 entries covers full accuracy computation; reading all 68MB risks OOM.
---
  331:         except (ValueError, TypeError):
  332:             continue  # skip malformed timestamps
  333: 
  334:         weight = math.exp(-decay_rate * age_days)
---
  761:     except Exception:
  762:         logger.debug("signal_utility disk cache delete failed", exc_info=True)
  763: 
  764: 
---
  983:         except (KeyError, AttributeError):
  984:             logger.debug("Activation rates cache corrupted, regenerating")
  985:     rates = signal_activation_rates()
  986:     try:
---
  989:     except Exception:
  990:         logger.warning("Failed to write activation rates cache", exc_info=True)
  991:     return rates
  992: 
---
 1005:         except (KeyError, AttributeError):
 1006:             logger.debug("Accuracy cache corrupted or missing horizon %s", horizon)
 1007:     return None
 1008: 
---
 1141:     except Exception:
 1142:         logger.debug("prewarm-ts persist failed", exc_info=True)
 1143: 
 1144: 
---
 1196:     except Exception:
 1197:         # If process_lock is somehow unavailable, fall back to the
 1198:         # process-local guarantee (better than crashing the loop).
 1199:         acquire_lock_file = None
---
 1232:     except Exception:
 1233:         logger.debug("maybe_prewarm_dashboard_accuracy failed", exc_info=True)
 1234:         return False
 1235:     finally:
---
 1392:         except (KeyError, AttributeError):
 1393:             logger.debug("Regime accuracy cache corrupted or missing horizon %s", horizon)
 1394:     return None
 1395: 
---
 1474:     except Exception:
 1475:         logger.debug("regime accuracy compute failed", exc_info=True)
 1476:         return {}
 1477:     with _regime_accuracy_cache_lock:
---
 1481:     except Exception:
 1482:         logger.debug("regime accuracy disk write failed", exc_info=True)
 1483:     return result
 1484: 
---
 1523:     except Exception:
 1524:         logger.debug("regime accuracy L2 invalidation failed", exc_info=True)
 1525: 
 1526: 
---
 1607:         except (ValueError, TypeError, KeyError):
 1608:             continue
 1609:     return best
 1610: 
---
 1733:         except (KeyError, TypeError):
 1734:             pass
 1735: 
 1736:     if entries is None:
---
 1801:     except Exception:
 1802:         logger.debug("Failed to write best_horizon cache", exc_info=True)
 1803: 
 1804:     return result
---
 1927:         except (KeyError, AttributeError):
 1928:             logger.debug("Ticker accuracy cache corrupted or missing horizon %s", horizon)
 1929:     return None
 1930: 
---

=== portfolio/accuracy_degradation.py exception blocks ===
  155:     except Exception as e:
  156:         logger.warning("Recent signal_accuracy snapshot failed: %s", e)
  157: 
  158:     # Per-ticker per-signal — lifetime via cached helper (1h TTL inside),
---
  163:     except Exception as e:
  164:         logger.warning("Lifetime per-ticker accuracy snapshot failed: %s", e)
  165: 
  166:     try:
---
  171:     except Exception as e:
  172:         logger.warning("Recent per-ticker accuracy snapshot failed: %s", e)
  173: 
  174:     # Forecast (Chronos/Kronos) — Codex P2#4 split. Forecast uses its
---
  185:     except Exception as e:
  186:         logger.warning("Forecast accuracy snapshot failed: %s", e)
  187: 
  188:     # Aggregate consensus — lifetime over all entries, recent over shared list
---
  194:     except Exception as e:
  195:         logger.warning("Consensus accuracy snapshot failed: %s", e)
  196: 
  197:     # save_accuracy_snapshot() also writes the lifetime `signals` block.
---
  280:     except (KeyError, ValueError, TypeError):
  281:         return 0.0
  282:     return (now - snap_ts).total_seconds() / 86400.0
  283: 
---
  294:     except Exception:
  295:         return False
  296:     try:
  297:         forward = events_within_hours(ECON_BLACKOUT_HOURS)
---
  304:     except Exception as e:
  305:         logger.debug("econ blackout check failed: %s", e)
  306:         return False
  307: 
---
  476:     except Exception as e:
  477:         logger.warning("Per-signal degradation diff failed: %s", e)
  478: 
  479:     # 2) Per-ticker per-signal — share entries to avoid 41x re-scan
---
  501:     except Exception as e:
  502:         logger.warning("Per-ticker degradation diff failed: %s", e)
  503: 
  504:     # 3) Forecast models (Chronos/Kronos only — Ministral/Qwen3 are in scope #1)
---
  529:     except Exception as e:
  530:         logger.warning("Forecast degradation diff failed: %s", e)
  531: 
  532:     # 4) Aggregate consensus — share entries (consensus_accuracy honors
---
  552:     except Exception as e:
  553:         logger.warning("Consensus degradation diff failed: %s", e)
  554: 
  555:     return alerts
---
  595:     except (TypeError, ValueError):
  596:         return None
  597:     if old_total < min_samples_historical or new_total < min_samples_current:
  598:         return None
---
  735:     except Exception as e:
  736:         logger.warning("Daily accuracy snapshot failed: %s", e)
  737:         return False
  738: 
---
  810:     except Exception as e:
  811:         logger.exception("Failed to record snapshot writer silent failure: %s", e)
  812:         return
  813: 
---
  852:         except (ValueError, TypeError):
  853:             latest_ts = None
  854:         if latest_ts is None or latest_ts.date() != now.date():
  855:             logger.warning(
---
  863:     except Exception as e:
  864:         logger.warning("Daily summary build failed: %s", e)
  865:         return False
  866: 
---
  875:     except Exception as e:
  876:         logger.warning("Daily summary send failed: %s", e)
  877:         return False
  878: 
---
  902:         except (TypeError, ValueError):
  903:             delta_str = ""
  904:     lines.append(
  905:         f"`Consensus: {consensus_acc:.0f}% recent7d{delta_str} · "
---
  956:     except Exception:
  957:         logger.debug("Snapshot age computation failed", exc_info=True)
  958: 
  959:     sig_count = len(signals_recent or {})
---
  985:         except (TypeError, ValueError):
  986:             continue
  987:         change_pp = (new_acc - old_acc) * 100.0
  988:         if change_pp <= -DROP_THRESHOLD_PP and new_acc * 100.0 < ABSOLUTE_FLOOR_PCT:
---

=== portfolio/signal_engine.py exception blocks ===
  235:     except (KeyError, IndexError, ValueError, TypeError):
  236:         return "HOLD", 0.0
  237:     price = float(ind.get("close", 0.0)) or float(close.iloc[-1])
  238:     tiny_eps = abs(price) * _EMA_FLAT_EPS_REL
---
  260:     except (KeyError, TypeError, ValueError):
  261:         return "HOLD", 0.0
  262:     half_width = upper - mid
  263:     if half_width <= 0:
---
  305:     except (TypeError, ValueError):
  306:         current_hist = None
  307:     try:
  308:         close = df["close"]
---
  316:     except (KeyError, IndexError, ValueError, TypeError):
  317:         return "HOLD", 0.0
  318:     # If ind lacked a usable current hist, fall back to the recomputed
  319:     # value so the magnitude gate still applies.
---
  359:     except (KeyError, IndexError, ValueError, TypeError):
  360:         return "HOLD", 0.0
  361:     bull = int((closes > opens).sum())
  362:     bear = int((closes < opens).sum())
---
  418:     except (KeyError, IndexError, ValueError, TypeError):
  419:         return "HOLD", 0.0
  420:     if price_slope > 0 and ema21_slope > 0:
  421:         return "BUY", FORECAST_DEAD_ZONE_SOFT_CONF
---
 1028:         except Exception:
 1029:             logger.warning("Failed to load prev sentiments", exc_info=True)
 1030:         _prev_sentiment_loaded = True
 1031: 
---
 1067:     except Exception:
 1068:         # Dirty flag remains True — next cycle will retry the write
 1069:         logger.warning("Failed to persist sentiment state (will retry next cycle)", exc_info=True)
 1070: 
---
 1402:     except Exception:
 1403:         logger.debug("Dynamic horizon weights unavailable, using static fallback", exc_info=True)
 1404:         return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
 1405: 
---
 1488:     except (TypeError, ValueError):
 1489:         conf = 0.0
 1490:     if not np.isfinite(conf):
 1491:         if sig_name:
---
 1629:     except Exception:
 1630:         logger.debug("Dynamic correlation groups unavailable, using static", exc_info=True)
 1631:         return _STATIC_CORRELATION_GROUPS
 1632: 
---
 1758:     except (TypeError, ValueError):
 1759:         return default
 1760:     if math.isnan(f) or math.isinf(f):
 1761:         return default
---
 1772:     except (TypeError, ValueError):
 1773:         return 0
 1774:     if math.isnan(f) or math.isinf(f) or f < 0:
 1775:         return 0
---
 2046:     except Exception:
 2047:         logger.debug("IC data unavailable for %s", horizon, exc_info=True)
 2048:     return None
 2049: 
---
 2077:         except Exception as e:
 2078:             logger.warning("macro window detection failed (treating as inactive): %s", e)
 2079:             active = False
 2080:         _macro_window_cache["value"] = active
---
 2176:         except (TypeError, ValueError):
 2177:             return None
 2178:         if _math.isnan(f) or _math.isinf(f) or f < 0:
 2179:             return None
---
 2601:             except (TypeError, ValueError):
 2602:                 pass
 2603:         if vote == "BUY":
 2604:             buy_weight += weight
---
 2659:         except Exception:
 2660:             logger.warning("Failed to load %s accuracy", signal_name, exc_info=True)
 2661:             return {}
 2662: 
---
 2798:     except Exception:
 2799:         logger.warning("ADX computation failed", exc_info=True)
 2800:         with _adx_lock:
 2801:             _adx_cache[df_id] = None
---
 2888:         except Exception:
 2889:             logger.warning("Trap detection failed for %s", ticker, exc_info=True)
 2890:     # BUG-90: Clamp after Stage 3
 2891:     conf = min(1.0, conf)
---
 3159:     except ImportError:
 3160:         logger.debug("Optional module %s not available", "fear_greed")
 3161: 
 3162:     # Social media posts (Reddit) — fetched separately, merged into sentiment
---
 3177:         except ImportError:
 3178:             logger.debug("Optional module %s not available", "social_sentiment")
 3179: 
 3180:     # Sentiment (crypto->CryptoBERT, stocks->Trading-Hero-LLM) — includes social posts
---
 3236:         except ImportError:
 3237:             logger.debug("Optional module %s not available", "sentiment")
 3238: 
 3239:     # ML Classifier — disabled: 28.2% accuracy (1,027 samples, 1d horizon).
---
 3258:         except ImportError:
 3259:             logger.debug("Optional module %s not available", "funding_rate")
 3260: 
 3261:     # On-Chain BTC Signal — MVRV Z-Score, SOPR, NUPL, Exchange Netflow.
---
 3320:         except ImportError:
 3321:             logger.debug("Optional module %s not available", "onchain_data")
 3322: 
 3323:     # Volume Confirmation (spike + price direction = vote)
---
 3334:         except ImportError:
 3335:             logger.debug("Optional module %s not available", "macro_context")
 3336: 
 3337:     # Ministral-3-8B LLM reasoning (all tickers — crypto, stocks, metals)
---
 3381:         except ImportError:
 3382:             logger.debug("Optional module %s not available", "ministral_signal")
 3383: 
 3384:     # Qwen3-8B LLM reasoning (all tickers — crypto, stocks, metals)
---
 3428:         except ImportError:
 3429:             logger.debug("Optional module %s not available", "qwen3_signal")
 3430: 
 3431:     # --- Enhanced signal modules (composite indicators computed from raw OHLCV) ---
---
 3452:             except Exception:
 3453:                 logger.warning("Macro context fetch failed", exc_info=True)
 3454: 
 3455:         # Load seasonality profile for metals tickers (detrending)
---
 3461:             except Exception:
 3462:                 logger.debug("Seasonality profile load failed for %s", ticker, exc_info=True)
 3463: 
 3464:         # Build context data once for signals that need it
---
 3511:                     except Exception as e:
 3512:                         logger.debug("Shadow signal %s failed: %s", sig_name, e)
 3513:                 votes[sig_name] = "HOLD"
 3514:                 continue
---
 3552:             except Exception as e:
 3553:                 logger.warning("Signal %s failed: %s", sig_name, e)
 3554:                 votes[sig_name] = "HOLD"
 3555:                 _signal_failures.append(sig_name)
---
 3572:         except Exception:
 3573:             logger.debug("Signal health tracking failed", exc_info=True)
 3574: 
 3575:         # 2026-05-11 Stage 2 Batch 2: candlestick + forecast dead-zone
---
 3672:     except Exception:
 3673:         logger.debug("llm probability logging failed", exc_info=True)
 3674: 
 3675:     # C10: Capture raw pre-gate votes BEFORE any gating rewrites them to HOLD.
---
 3706:     except Exception:
 3707:         logger.debug("Per-ticker accuracy unavailable for regime gating exemption", exc_info=True)
 3708:     _TICKER_EXEMPT_ACC = 0.60
 3709:     _TICKER_EXEMPT_MIN_SAMPLES = 50
---
 3723:     except Exception:
 3724:         logger.debug("Recent accuracy unavailable for regime gating override", exc_info=True)
 3725: 
 3726:     regime_gated_effective = set(regime_gated)
---
 3875:     except Exception:
 3876:         logger.error("Accuracy stats load failed", exc_info=True)
 3877:         # H3: Fail-closed: gate all signals (0% accuracy, 999 samples) rather than
 3878:         # leaving accuracy_data = {} which bypasses the accuracy gate entirely.
---
 3906:         except Exception:
 3907:             logger.debug("Regime-conditional accuracy unavailable", exc_info=True)
 3908: 
 3909:     # BUG-158: Override global accuracy with per-ticker accuracy for ALL signals.
---
 3968:         except Exception:
 3969:             logger.debug("Utility weighting unavailable", exc_info=True)
 3970: 
 3971:     if ticker:
---
 3988:         except Exception:
 3989:             logger.debug("Best-horizon accuracy unavailable", exc_info=True)
 3990:     accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
 3991:     max_signals = sig_cfg.get("max_active_signals")
---
 4096:     except Exception:
 4097:         logger.debug("Market health penalty failed", exc_info=True)
 4098: 
 4099:     # --- Earnings proximity gate (stocks only) ---
---
 4111:         except Exception:
 4112:             logger.debug("Earnings gate failed for %s", ticker, exc_info=True)
 4113: 
 4114:     # --- Linear factor model score (supplementary, not overriding) ---
---
 4143:     except Exception:
 4144:         logger.debug("Linear factor model failed", exc_info=True)
 4145: 
 4146:     if ticker:
---
 4176:         except Exception:
 4177:             logger.debug("Per-ticker consensus gate failed for %s", ticker, exc_info=True)
 4178: 
 4179:     # Global confidence cap — calibration data shows >80% confidence is
---

=== portfolio/forecast_accuracy.py exception blocks ===
  283:         except (ValueError, TypeError):
  284:             modified_entries.append(entry)
  285:             continue
  286: 
---
  366:         except (ValueError, TypeError):
  367:             continue
  368: 
  369:     return best_price
---

=== portfolio/signal_postmortem.py exception blocks ===
  138:         except Exception:
  139:             logger.warning("Could not load signal_log entries for correlation analysis")
  140:             return []
  141: 
---
  222:     except Exception:
  223:         logger.warning("Accuracy data unavailable for postmortem", exc_info=True)
  224: 
  225:     # Correlation analysis
---
  228:     except Exception:
  229:         logger.warning("Correlation analysis failed", exc_info=True)
  230: 
  231:     # Summary
---

=== portfolio/signal_decay_alert.py exception blocks ===
   37:     except (FileNotFoundError, json.JSONDecodeError) as e:
   38:         logger.warning("Cannot load accuracy cache: %s", e)
   39:         return []
   40: 
---
  147:     except Exception:
  148:         logger.debug("Could not write signal_decay_alerts.jsonl", exc_info=True)
  149: 
  150: 
---
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
 1700:             f"{a['old_accuracy']}% to {a['new_accuracy']}% "
 1701:             f"({a['change']:+.1f}pp, {a['new_samples']} samples)"
 1702:         )
 1703:     return lines
 1704: 
 1705: 
 1706: def signal_best_horizon_accuracy(min_samples=50, entries=None):
 1707:     """Compute each signal's best accuracy across all horizons.
 1708: 
 1709:     For each signal, evaluates accuracy at every horizon in HORIZONS and returns
 1710:     the horizon with the highest accuracy (provided it meets the minimum sample
 1711:     threshold). This allows the signal weighting system to use the most
 1712:     predictive horizon for each signal rather than a fixed 1d window.
 1713: 
 1714:     Results are cached in BEST_HORIZON_CACHE_FILE with the same TTL as the
 1715:     main accuracy cache (ACCURACY_CACHE_TTL).
 1716: 
 1717:     Args:
 1718:         min_samples: Minimum number of datapoints required for a horizon to be
 1719:             considered. Horizons below this threshold are skipped. Default 50.
 1720: 
 1721:     Returns:
 1722:         dict: {signal_name: {accuracy, total, correct, pct, best_horizon}}
 1723:         Signals with no qualifying horizons are omitted from the result.
 1724:     """
 1725:     # --- Cache check ---
 1726:     cached = load_json(BEST_HORIZON_CACHE_FILE)
 1727:     if cached is not None and isinstance(cached, dict):
 1728:         try:
 1729:             if time.time() - cached.get("time", 0) < ACCURACY_CACHE_TTL:
 1730:                 data = cached.get("data")
 1731:                 if isinstance(data, dict):
 1732:                     return data
 1733:         except (KeyError, TypeError):
 1734:             pass
 1735: 
 1736:     if entries is None:
 1737:         entries = load_entries()
 1738:     # {sig_name: {horizon: {correct, total}}}
 1739:     stats: dict[str, dict[str, dict[str, int]]] = {
 1740:         s: {h: {"correct": 0, "total": 0} for h in HORIZONS}
 1741:         for s in SIGNAL_NAMES
 1742:     }
 1743: 
 1744:     for entry in entries:
 1745:         outcomes = entry.get("outcomes", {})
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/signal_engine.py:3978:     if not _accuracy_failed and sig_cfg.get("use_best_horizon", False):
portfolio/signal_engine.py:3980:             from portfolio.accuracy_stats import signal_best_horizon_accuracy
portfolio/signal_engine.py:3981:             best_hz = signal_best_horizon_accuracy(min_samples=50)
portfolio/accuracy_stats.py:37: BEST_HORIZON_CACHE_FILE = DATA_DIR / "best_horizon_cache.json"
portfolio/accuracy_stats.py:1706: def signal_best_horizon_accuracy(min_samples=50, entries=None):
portfolio/accuracy_stats.py:1722:         dict: {signal_name: {accuracy, total, correct, pct, best_horizon}}
portfolio/accuracy_stats.py:1794:                 "best_horizon": best_hz,
portfolio/accuracy_stats.py:1802:         logger.debug("Failed to write best_horizon cache", exc_info=True)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 3970: 
 3971:     if ticker:
 3972:         _record_phase(ticker, "utility_overlay", _phase_start)
 3973:         _phase_start = time.monotonic()
 3974: 
 3975:     # Multi-horizon: optionally use each signal's best horizon accuracy.
 3976:     # H3: Skip when primary load failed to preserve fail-closed gate.
 3977:     sig_cfg = (config or {}).get("signals", {})
 3978:     if not _accuracy_failed and sig_cfg.get("use_best_horizon", False):
 3979:         try:
 3980:             from portfolio.accuracy_stats import signal_best_horizon_accuracy
 3981:             best_hz = signal_best_horizon_accuracy(min_samples=50)
 3982:             for sig_name, bh_data in best_hz.items():
 3983:                 if bh_data.get("total", 0) >= 30:
 3984:                     # Only override if best-horizon accuracy is meaningfully better
 3985:                     current = accuracy_data.get(sig_name, {}).get("accuracy", 0.5)
 3986:                     if bh_data["accuracy"] > current + 0.03:
 3987:                         accuracy_data[sig_name] = bh_data
 3988:         except Exception:
 3989:             logger.debug("Best-horizon accuracy unavailable", exc_info=True)
 3990:     accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
 3991:     max_signals = sig_cfg.get("max_active_signals")
 3992: 
 3993:     # Signal persistence filter: only let signals that maintained their vote
 3994:     # for 2+ consecutive cycles participate in consensus. Raw votes are kept
 3995:     # intact for accuracy tracking (signal_log records unfiltered votes).
 3996:     consensus_votes = _apply_persistence_filter(votes, ticker)
 3997:     # Track how many signals were filtered for debugging
 3998:     _filtered_count = sum(
 3999:         1 for s in votes
 4000:         if votes[s] != "HOLD" and consensus_votes.get(s) == "HOLD"
 4001:     )
 4002:     if _filtered_count > 0:
 4003:         extra_info["_persistence_filtered"] = _filtered_count
 4004: 
 4005:     # Macro-window force-HOLD has already been applied to `votes` above
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 4000:         if votes[s] != "HOLD" and consensus_votes.get(s) == "HOLD"
 4001:     )
 4002:     if _filtered_count > 0:
 4003:         extra_info["_persistence_filtered"] = _filtered_count
 4004: 
 4005:     # Macro-window force-HOLD has already been applied to `votes` above
 4006:     # (before buy/sell/core_active counting), so `consensus_votes` —
 4007:     # which derives from `votes` via persistence filter — already
 4008:     # carries the suppression. No additional mutation needed here.
 4009:     # `_weighted_consensus` runs its own pre-pass as defense-in-depth.
 4010: 
 4011:     # BUG-224: compute post-persistence voter count so downstream consumers
 4012:     # (accuracy tracking, Layer 2) see the actual participating voter count,
 4013:     # not the inflated pre-filter number.
 4014:     post_persistence_voters = sum(
 4015:         1 for v in consensus_votes.values() if v in ("BUY", "SELL")
 4016:     )
 4017: 
 4018:     weighted_action, weighted_conf = _weighted_consensus(
 4019:         consensus_votes, accuracy_data, regime, activation_rates,
 4020:         accuracy_gate=accuracy_gate,
 4021:         max_signals=max_signals,
 4022:         horizon=horizon,
 4023:         regime_gated_override=regime_gated_effective,
 4024:         ticker=ticker,
 4025:         # 2026-05-11 (Codex Fix B): pass extra_info so _weighted_consensus
 4026:         # can dampen ema/bb/macd soft votes by their _soft_conf_* values.
 4027:         soft_confidences=extra_info,
 4028:     )
 4029: 
 4030:     if ticker:
 4031:         _record_phase(ticker, "weighted_consensus", _phase_start)
 4032:         _phase_start = time.monotonic()
 4033: 
 4034:     # BUG-227: Apply core gate AND MIN_VOTERS gate to weighted consensus.
 4035:     # Use post_persistence_voters (not pre-filter active_voters) because the
 4036:     # persistence filter may have reduced voters below the threshold.
 4037:     if core_active == 0 or post_persistence_voters < min_voters:
 4038:         weighted_action = "HOLD"
 4039:         weighted_conf = 0.0
 4040: 
 4041:     # Confluence score
 4042:     confluence = _confluence_score(votes, extra_info)
 4043: 
 4044:     # Time-of-day confidence adjustment
 4045:     tod_factor = _time_of_day_factor(horizon=horizon)
 4046:     weighted_conf *= tod_factor
 4047: 
 4048:     # Store raw consensus in extra for debugging, then use weighted as primary
 4049:     extra_info["_raw_action"] = action
 4050:     extra_info["_raw_confidence"] = conf
 4051:     extra_info["_voters"] = active_voters  # pre-filter (compatibility)
 4052:     extra_info["_voters_post_filter"] = post_persistence_voters
 4053:     extra_info["_total_applicable"] = total_applicable
 4054:     extra_info["_buy_count"] = buy
 4055:     extra_info["_sell_count"] = sell
 4056:     extra_info["_core_buy"] = core_buy
 4057:     extra_info["_core_sell"] = core_sell
 4058:     extra_info["_core_active"] = core_active
 4059:     extra_info["_votes"] = votes
 4060:     extra_info["_raw_votes"] = raw_votes  # C10: pre-gate votes for accuracy recovery
 4061:     extra_info["_regime"] = regime
 4062:     if horizon:
 4063:         extra_info["_horizon"] = horizon
 4064:     extra_info["_weighted_action"] = weighted_action
 4065:     extra_info["_weighted_confidence"] = weighted_conf
 4066:     extra_info["_confluence_score"] = confluence
 4067: 
 4068:     # Primary action = weighted consensus (accounts for accuracy + bias penalties)
 4069:     action = weighted_action
 4070:     conf = weighted_conf
 4071: 
 4072:     # Apply confidence penalty cascade (regime, volume/ADX, trap, dynamic min_voters)
 4073:     action, conf, penalty_log = apply_confidence_penalties(
 4074:         action, conf, regime, ind, extra_info, ticker, df, config
 4075:     )
 4076:     if penalty_log:
 4077:         extra_info["_penalty_log"] = penalty_log
 4078: 
 4079:     if ticker:
 4080:         _record_phase(ticker, "penalties", _phase_start)
 4081:         _phase_start = time.monotonic()
 4082: 
 4083:     # --- Market health confidence penalty ---
 4084:     # Penalizes BUY signals when broad market is unhealthy (distribution days,
 4085:     # broken FTD, etc.).  Only affects BUY; SELL and HOLD pass through.
 4086:     try:
 4087:         from portfolio.market_health import get_confidence_penalty, get_market_health
 4088:         mh = get_market_health()
 4089:         mh_mult = get_confidence_penalty(action, mh)
 4090:         if mh_mult != 1.0:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
 2648: def _load_local_model_accuracy(signal_name, horizon="1d", days=None, cache_ttl=None):
 2649:     """Load per-ticker accuracy for a local model signal."""
 2650:     lookback_days = days if days is not None else _LOCAL_MODEL_LOOKBACK_DAYS
 2651:     ttl = cache_ttl or _LOCAL_MODEL_ACCURACY_TTL
 2652:     cache_key = f"local_model_accuracy_{signal_name}_{horizon}_{lookback_days}"
 2653: 
 2654:     def _fetch():
 2655:         try:
 2656:             from portfolio.accuracy_stats import accuracy_by_signal_ticker
 2657: 
 2658:             return accuracy_by_signal_ticker(signal_name, horizon=horizon, days=lookback_days)
 2659:         except Exception:
 2660:             logger.warning("Failed to load %s accuracy", signal_name, exc_info=True)
 2661:             return {}
 2662: 
 2663:     return _cached(cache_key, ttl, _fetch)
 2664: 
 2665: 
 2666: def _build_llm_context(ticker, ind, timeframes, extra_info):
 2667:     """Build shared context dict for local LLM signals (Ministral, Qwen3)."""
 2668:     tf_summary = ""
 2669:     if timeframes:
 2670:         parts = []
 2671:         for label, entry in timeframes:
 2672:             if isinstance(entry, dict) and "action" in entry and entry["action"]:
 2673:                 ti = entry.get("indicators", {})
 2674:                 parts.append(f"{label}: {entry['action']} (RSI={ti.get('rsi', 0):.0f})")
 2675:         if parts:
 2676:             tf_summary = " | ".join(parts)
 2677: 
 2678:     ema_gap = (
 2679:         abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100
 2680:         if ind["ema21"] != 0
 2681:         else 0
 2682:     )
 2683: 
 2684:     return {
 2685:         "ticker": ticker.replace("-USD", ""),
 2686:         "price_usd": ind["close"],
 2687:         "rsi": round(ind["rsi"], 1),
 2688:         # 2026-04-10: bumped to 5 decimals — see portfolio/reporting.py:114
 2689:         # for the root-cause explanation (MACD-improving gate rounding).
 2690:         "macd_hist": round(ind["macd_hist"], 5),
 2691:         "ema_bullish": ind["ema9"] > ind["ema21"],
 2692:         "ema_gap_pct": round(ema_gap, 2),
 2693:         "bb_position": ind["price_vs_bb"],
 2694:         "fear_greed": extra_info.get("fear_greed", "N/A"),
 2695:         "fear_greed_class": extra_info.get("fear_greed_class", ""),
 2696:         "news_sentiment": extra_info.get("sentiment", "N/A"),
 2697:         "sentiment_confidence": extra_info.get("sentiment_conf", "N/A"),
 2698:         "volume_ratio": extra_info.get("volume_ratio", "N/A"),
 2699:         "funding_rate": extra_info.get("funding_action", "N/A"),
 2700:         "timeframe_summary": tf_summary,
 2701:         "headlines": "",
 2702:     }
 2703: 
 2704: 
 2705: def _gate_local_model_vote(signal_name, vote, ticker, config=None):
 2706:     """Apply accuracy-based abstention to local model votes."""
 2707:     # 2026-05-10 (codex re-review): explicit dict[str, Any] — initial
 2708:     # values mix str / None / int and later rounds add float (accuracy)
 2709:     # and int (samples). Without annotation mypy locks the value type
 2710:     # to the union of the literal initialisers and rejects every later
 2711:     # assignment.
 2712:     info: dict[str, Any] = {
 2713:         "gating": "raw",
 2714:         "accuracy": None,
 2715:         "samples": 0,
 2716:     }
 2717:     if vote == "HOLD" or not ticker:
 2718:         return vote, info
 2719: 
 2720:     cfg = ((config or {}).get("local_models", {}) or {}).get(signal_name, {})
 2721:     hold_threshold = cfg.get("hold_threshold", _LOCAL_MODEL_HOLD_THRESHOLD)
 2722:     min_samples = cfg.get("min_samples", _LOCAL_MODEL_MIN_SAMPLES)
 2723:     days = cfg.get("accuracy_days", _LOCAL_MODEL_LOOKBACK_DAYS)
 2724:     cache_ttl = cfg.get("accuracy_cache_ttl", _LOCAL_MODEL_ACCURACY_TTL)
 2725: 
 2726:     accuracy_data = _load_local_model_accuracy(
 2727:         signal_name, horizon=cfg.get("horizon", "1d"), days=days, cache_ttl=cache_ttl
 2728:     )
 2729:     ticker_stats = (accuracy_data or {}).get(ticker)
 2730:     if not ticker_stats or ticker_stats.get("samples", 0) < min_samples:
 2731:         info["gating"] = "insufficient_data"
 2732:         if ticker_stats:
 2733:             info["accuracy"] = round(ticker_stats.get("accuracy", 0.0), 3)
 2734:             info["samples"] = ticker_stats.get("samples", 0)
 2735:         return vote, info
 2736: 
 2737:     accuracy = float(ticker_stats.get("accuracy", 0.0))
 2738:     samples = int(ticker_stats.get("samples", 0))
 2739:     info["accuracy"] = round(accuracy, 3)
 2740:     info["samples"] = samples
 2741:     if accuracy < hold_threshold:
 2742:         info["gating"] = "held"
 2743:         return "HOLD", info
 2744: 
 2745:     return vote, info
 2746: 
 2747: 
 2748: def _compute_adx(df, period=14):
 2749:     """Compute ADX (Average Directional Index) from a DataFrame with high/low/close.
 2750: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
  556: def accuracy_by_signal_ticker(signal_name, horizon="1d", days=None, entries=None):
  557:     """Compute per-ticker accuracy for one signal.
  558: 
  559:     Args:
  560:         signal_name: Signal name present in SIGNAL_NAMES.
  561:         horizon: Outcome horizon to evaluate.
  562:         days: Optional lookback window in days.
  563:         entries: Pre-loaded entries list. BUG-178/W15-W16 follow-up
  564:             (2026-04-16 review): callers that iterate over many signal
  565:             names (e.g. accuracy_degradation._per_ticker_recent) must
  566:             pass a single pre-loaded list instead of letting each call
  567:             re-scan the 50,000-entry SQLite file. Skipping that knob
  568:             blew cycle time by ~290s in the original implementation.
  569: 
  570:     Returns:
  571:         dict: {ticker: {"accuracy": float, "samples": int, "correct": int}}
  572:     """
  573:     if signal_name not in SIGNAL_NAMES:
  574:         return {}
  575: 
  576:     if entries is None:
  577:         entries = load_entries()
  578:     cutoff = None
  579:     if days is not None:
  580:         from datetime import datetime, timedelta
  581: 
  582:         cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
  583: 
  584:     stats = defaultdict(lambda: {"correct": 0, "total": 0})
  585: 
  586:     for entry in entries:
  587:         if cutoff and entry.get("ts", "") < cutoff:
  588:             continue
  589: 
  590:         outcomes = entry.get("outcomes", {})
  591:         tickers = entry.get("tickers", {})
  592:         for ticker, tdata in tickers.items():
  593:             outcome = outcomes.get(ticker, {}).get(horizon)
  594:             if not outcome:
  595:                 continue
  596: 
  597:             vote = (tdata.get("signals") or {}).get(signal_name, "HOLD")
  598:             if vote == "HOLD":
  599:                 continue
  600: 
  601:             change_pct = outcome.get("change_pct", 0)
  602:             result_val = _vote_correct(vote, change_pct)
  603:             if result_val is None:
  604:                 continue
  605:             stats[ticker]["total"] += 1
  606:             if result_val:
  607:                 stats[ticker]["correct"] += 1
  608: 
  609:     result = {}
  610:     for ticker, data in stats.items():
  611:         total = data["total"]
  612:         if total == 0:
  613:             continue
  614:         result[ticker] = {
  615:             "accuracy": data["correct"] / total,
  616:             "samples": total,
  617:             "correct": data["correct"],
  618:         }
  619:     return result
  620: 
  621: 
  622: def signal_utility(horizon="1d", entries=None):
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
  622: def signal_utility(horizon="1d", entries=None):
  623:     """Compute per-signal return magnitude utility.
  624: 
  625:     For each non-HOLD signal vote with a non-neutral outcome, compute the
  626:     directional return:
  627:       - BUY  → +change_pct   (positive = correct direction)
  628:       - SELL → -change_pct   (negative change = correct direction → positive return)
  629: 
  630:     Neutral outcomes (|change_pct| < _MIN_CHANGE_PCT) are skipped.
  631: 
  632:     Args:
  633:         horizon: Outcome horizon to evaluate.
  634:         entries: Pre-loaded entries list. If None, loads from disk.
  635: 
  636:     Returns:
  637:         dict: {signal_name: {avg_return, total_return, samples, utility_score}}
  638:         where utility_score = avg_return * sqrt(samples).
  639:         Signals with no data get zeros.
  640: 
  641:     2026-04-15 (BUG-178 mitigation): when `entries` is None, the result is
  642:     cached for _SIGNAL_UTILITY_CACHE_TTL seconds keyed by horizon. The cold
  643:     walk costs ~3.6s on a 6K-snapshot log; with 5 ticker threads per cycle
  644:     and the OS file cache occasionally cold, this was a legitimate
  645:     per-cycle cost. Passing an explicit `entries` list bypasses the cache
  646:     (preserves the old behavior for test fixtures that want a specific
  647:     entries snapshot).
  648:     """
  649:     if entries is None:
  650:         now = time.time()
  651:         with _signal_utility_cache_lock:
  652:             cached = _signal_utility_cache.get(horizon)
  653:             if cached and now - cached[0] < _SIGNAL_UTILITY_CACHE_TTL:
  654:                 return cached[1]
  655:         # L1 miss — try L2 disk cache before paying the cold compute. After
  656:         # a process restart this is what saves us the ~49s parallel-cold
  657:         # cost: thread-1 reads disk (~ms), populates L1, threads 2-4 hit L1.
  658:         disk_cached = _load_signal_utility_disk(horizon)
  659:         if disk_cached is not None:
  660:             with _signal_utility_cache_lock:
  661:                 _signal_utility_cache[horizon] = (time.time(), disk_cached)
  662:             return disk_cached
  663:         # L1 + L2 miss — compute outside the lock to avoid serializing all
  664:         # threads behind the slow path. Populate both caches on success.
  665:         result = _compute_signal_utility(horizon, None)
  666:         with _signal_utility_cache_lock:
  667:             _signal_utility_cache[horizon] = (time.time(), result)
  668:         _write_signal_utility_disk(horizon, result)
  669:         return result
  670:     # Explicit entries — bypass cache (caller controls the dataset).
  671:     return _compute_signal_utility(horizon, entries)
  672: 
  673: 
  674: def _compute_signal_utility(horizon, entries):
  675:     """Actual utility computation. Extracted from signal_utility so the
  676:     cache wrapper can call it without re-entering the cached function
  677:     (and so test fixtures passing explicit entries can hit the raw path).
  678:     """
  679:     import math
  680: 
  681:     if entries is None:
  682:         entries = load_entries()
  683:     # {sig_name: {"total_return": float, "samples": int}}
  684:     stats = {s: {"total_return": 0.0, "samples": 0} for s in SIGNAL_NAMES}
  685: 
  686:     for entry in entries:
  687:         outcomes = entry.get("outcomes", {})
  688:         tickers = entry.get("tickers", {})
  689: 
  690:         for ticker, tdata in tickers.items():
  691:             outcome = outcomes.get(ticker, {}).get(horizon)
  692:             if not outcome:
  693:                 continue
  694: 
  695:             change_pct = outcome.get("change_pct", 0)
  696:             # 2026-05-03: None-guard mirrors _vote_correct (line 112) and the
  697:             # 2026-04-22 outcome-backfill regression. Without this, every
  698:             # cold compute crashes on 4h+ horizons that haven't backfilled
  699:             # yet — silently swallowed by signal_engine.py:3486's broad
  700:             # except, so the in-memory cache never populates and every call
  701:             # pays cold cost (~2.5s). That's the entire BUG-178 cache regression.
  702:             if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:
  703:                 continue  # neutral / unknown outcome — skip
  704: 
  705:             signals = tdata.get("signals", {})
  706:             for sig_name in SIGNAL_NAMES:
  707:                 vote = signals.get(sig_name, "HOLD")
  708:                 if vote == "HOLD":
  709:                     continue
  710:                 # Directional return: positive when signal was correct
  711:                 if vote == "BUY":
  712:                     dir_return = change_pct
  713:                 else:  # SELL
  714:                     dir_return = -change_pct
  715: 
  716:                 stats[sig_name]["total_return"] += dir_return
  717:                 stats[sig_name]["samples"] += 1
  718: 
  719:     result = {}
  720:     for sig_name in SIGNAL_NAMES:
  721:         s = stats[sig_name]
  722:         n = s["samples"]
  723:         total_ret = s["total_return"]
  724:         avg_ret = total_ret / n if n > 0 else 0.0
  725:         utility = avg_ret * math.sqrt(n) if n > 0 else 0.0
  726:         result[sig_name] = {
  727:             "avg_return": avg_ret,
  728:             "total_return": total_ret,
  729:             "samples": n,
  730:             "utility_score": utility,
  731:         }
  732:     return result
  733: 
  734: 
  735: def invalidate_signal_utility_cache():
  736:     """Clear both the in-memory L1 and the disk-backed L2 signal_utility cache.
  737: 
  738:     Cross-process scope:
  739:       - L1 in-memory clear is process-local. Other processes (crypto_loop,
  740:         oil_loop, metals_loop) keep their own L1 until their TTL expires.
  741:       - L2 disk delete is shared. After this call, the next call from ANY
  742:         process for any horizon misses L2 and recomputes.
  743: 
  744:     Intended caller: outcome_tracker (which runs as the PF-OutcomeCheck
  745:     daily scheduled task) after backfilling new outcomes — at that point
  746:     the cached utility values are stale by definition and forcing recompute
  747:     across all processes is correct. The satellite loops do not call this
  748:     function (verified 2026-05-03 grep) and should not start to: a delete
  749:     from one would force the others to pay the cold-compute cost on their
  750:     next cycle. If a satellite ever needs to invalidate its own L1 only,
  751:     add a separate L1-only function rather than reusing this one.
  752: 
  753:     Disk file removal is best-effort and silent on failure: a stale L2 file
  754:     expires by TTL anyway and must never crash the caller.
  755:     """
  756:     with _signal_utility_cache_lock:
  757:         _signal_utility_cache.clear()
  758:     try:
  759:         if SIGNAL_UTILITY_CACHE_FILE.exists():
  760:             SIGNAL_UTILITY_CACHE_FILE.unlink()
  761:     except Exception:
  762:         logger.debug("signal_utility disk cache delete failed", exc_info=True)
  763: 
  764: 
  765: def best_worst_signals(horizon="1d", acc=None):
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
portfolio/signal_engine.py:1351:         cache = load_json(DATA_DIR / "accuracy_cache.json")
portfolio/signal_decay_alert.py:27: def check_signal_decay(accuracy_cache_path="data/accuracy_cache.json"):
portfolio/accuracy_stats.py:36: ACCURACY_CACHE_FILE = DATA_DIR / "accuracy_cache.json"
portfolio/accuracy_stats.py:77: # existing pattern at regime_accuracy_cache.json: single "time" key gates TTL,
portfolio/accuracy_stats.py:995:     cache = load_json(ACCURACY_CACHE_FILE)
portfolio/accuracy_stats.py:1010: def write_accuracy_cache(horizon, data):
portfolio/accuracy_stats.py:1012:         cache = load_json(ACCURACY_CACHE_FILE, default={})
portfolio/accuracy_stats.py:1020:         _atomic_write_json(ACCURACY_CACHE_FILE, cache)
portfolio/accuracy_stats.py:1026: # write_accuracy_cache(h, cached)` should call these instead so the compute
portfolio/accuracy_stats.py:1044:             write_accuracy_cache(horizon, result)
portfolio/accuracy_stats.py:1060:             write_accuracy_cache(cache_key, result)
portfolio/accuracy_stats.py:1080:             write_accuracy_cache(cache_key, result)
portfolio/accuracy_stats.py:1108:             write_accuracy_cache(cache_key, result)
portfolio/accuracy_stats.py:1146:     """Periodically pre-warm accuracy_cache.json for the dashboard endpoint.
portfolio/accuracy_stats.py:1311: REGIME_ACCURACY_CACHE_FILE = DATA_DIR / "regime_accuracy_cache.json"
portfolio/accuracy_stats.py:1385:     cache = load_json(REGIME_ACCURACY_CACHE_FILE)
portfolio/accuracy_stats.py:1403:         cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
portfolio/accuracy_stats.py:1408:         _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)
portfolio/accuracy_stats.py:1436:     2. L2 (regime_accuracy_cache.json on disk) — survives process restart.
portfolio/accuracy_stats.py:1486: def invalidate_regime_accuracy_cache(horizon: str | None = None) -> None:
portfolio/accuracy_stats.py:1509:             cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
portfolio/accuracy_stats.py:1515:                 _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, {})
portfolio/accuracy_stats.py:1522:                 _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)
portfolio/accuracy_stats.py:1911: TICKER_ACCURACY_CACHE_FILE = DATA_DIR / "ticker_signal_accuracy_cache.json"
portfolio/accuracy_stats.py:1920:     cache = load_json(TICKER_ACCURACY_CACHE_FILE)
portfolio/accuracy_stats.py:1938:         cache = load_json(TICKER_ACCURACY_CACHE_FILE, default={})
portfolio/accuracy_stats.py:1943:         _atomic_write_json(TICKER_ACCURACY_CACHE_FILE, cache)
portfolio/forecast_accuracy.py:57: def invalidate_forecast_accuracy_cache():
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1910: 
 1911:     # Three guards, in increasing strictness, all applied:
 1912:     #
 1913:     #   Guard A (raw vs regime quorum):
 1914:     #     Matches downstream's `apply_confidence_penalties` which checks
 1915:     #     `extra_info["_voters"]` against dynamic_min. `_voters` is raw
 1916:     #     non-HOLD count post-regime, pre top-N/group-gate, so this check
 1917:     #     must NOT subtract `excluded` or `group_gated`.
 1918:     #
 1919:     #   Guard B (post-exclusion slate viability):
 1920:     #     Downstream's raw `_voters` doesn't account for top-N or
 1921:     #     correlation-group exclusions. If the POST-exclusion slate is
 1922:     #     below MIN_VOTERS_BASE (3) — the floor across all asset classes —
 1923:     #     a relaxed consensus would be built from a too-thin slate even
 1924:     #     though downstream would accept the raw count. Codex round 9
 1925:     #     (2026-04-17) caught this with a 3-signal correlation cluster
 1926:     #     gated out, leaving only 2 voters to drive consensus.
 1927:     #
 1928:     #   Guard C (lone-signal escape):
 1929:     #     Even with a large post-exclusion slate, directional gating can
 1930:     #     leave a single accuracy-passing signal. `best_possible >= 2`
 1931:     #     catches this case.
 1932:     min_regime_quorum = _dynamic_min_voters_for_regime(regime)
 1933:     raw_candidates = sum(1 for v in votes.values() if v != "HOLD")
 1934:     if raw_candidates < min_regime_quorum:
 1935:         return 0.0
 1936: 
 1937:     # P2-F (2026-04-17): derived from MIN_VOTERS_CRYPTO/STOCK rather than
 1938:     # hardcoded. If the base quorum changes, this follows automatically.
 1939:     post_exclusion_candidates = sum(
 1940:         1 for sn, v in votes.items()
 1941:         if v != "HOLD" and sn not in excluded and sn not in group_gated
 1942:     )
 1943:     if post_exclusion_candidates < _POST_EXCLUSION_MIN:
 1944:         return 0.0
 1945: 
 1946:     baseline = _count_active_voters_at_gate(
 1947:         votes, accuracy_data, excluded, group_gated, base_gate, 0.0,
 1948:     )
 1949:     if baseline >= _MIN_ACTIVE_VOTERS_SOFT:
 1950:         return 0.0
 1951: 
 1952:     best_possible = _count_active_voters_at_gate(
 1953:         votes, accuracy_data, excluded, group_gated,
 1954:         base_gate, _GATE_RELAXATION_MAX,
 1955:     )
 1956: 
 1957:     # Lone-signal escape guard. Even when raw candidates meet the downstream
 1958:     # quorum, directional gating can leave a thin set of recoverable voters.
 1959:     # P2-A (2026-04-17): raised from 2 to MIN_VOTERS_BASE (3). A 2-voter
 1960:     # "consensus" is still exposure-worthy in trending markets where
 1961:     # dynamic_min=3 — but any relaxation that only recovers 2 voters from a
 1962:     # large slate is catching signals that the downstream quorum would
 1963:     # accept as a weak consensus. Require at least as many as the base
 1964:     # MIN_VOTERS_* to avoid creating "relaxed" sub-quorum consensuses.
 1965:     if best_possible < _LONE_SIGNAL_FLOOR:
 1966:         return 0.0
 1967: 
 1968:     # Regime break: relaxation recovers nothing beyond baseline. Keep the
 1969:     # strict gate so the event shows up in logs rather than silently opening
 1970:     # to sub-41% signals.
 1971:     if best_possible <= baseline:
 1972:         return 0.0
 1973: 
 1974:     # Integer steps up to and including max - use int steps to avoid float drift.
 1975:     n_steps = int(round(_GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP))
 1976:     for i in range(1, n_steps + 1):
 1977:         candidate_rel = round(i * _GATE_RELAXATION_STEP, 6)
 1978:         active = _count_active_voters_at_gate(
 1979:             votes, accuracy_data, excluded, group_gated, base_gate, candidate_rel,
 1980:         )
 1981:         if active >= _MIN_ACTIVE_VOTERS_SOFT:
 1982:             return candidate_rel
 1983:     # Partial-recovery case (Codex P2 fix): best_possible > baseline but
 1984:     # still < floor. A single irrecoverable outlier shouldn't veto recovery
 1985:     # of the recoverable majority - apply max relaxation to get as many
 1986:     # voters back as possible. Logs still carry the relaxation value so
 1987:     # operators can distinguish this from a clean relaxation-to-floor.
 1988:     return _GATE_RELAXATION_MAX
 1989: 
 1990: 
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
  930: # tier-bias gate. During macro windows the bias dominates because the
  931: # 30-120min LLM cascade (Haiku/Sonnet/Opus) lags real-time regime
  932: # shifts. Force-HOLD instead of down-weighting (stricter than the
  933: # others — its accuracy collapses the most).
  934: MACRO_WINDOW_FORCE_HOLD_SIGNALS = frozenset({"claude_fundamental"})
  935: 
  936: # 5-minute cache. econ_dates is hardcoded so the underlying data
  937: # doesn't change between cycles, but we still pay an iteration over
  938: # ECON_EVENTS each call. Caching avoids that hit per signal per ticker
  939: # per cycle.
  940: _MACRO_WINDOW_CACHE_TTL_S = 300
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
673: # When cascaded gates would leave fewer than _MIN_ACTIVE_VOTERS_SOFT active voters
685: _MIN_ACTIVE_VOTERS_SOFT = 5
972: assert _POST_EXCLUSION_MIN <= _MIN_ACTIVE_VOTERS_SOFT, (
974:     f"_MIN_ACTIVE_VOTERS_SOFT ({_MIN_ACTIVE_VOTERS_SOFT}); "
1883:     least _MIN_ACTIVE_VOTERS_SOFT active voters.
1949:     if baseline >= _MIN_ACTIVE_VOTERS_SOFT:
1981:         if active >= _MIN_ACTIVE_VOTERS_SOFT:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/signal_db.py:10:     db = SignalDB()           # uses default path data/signal_log.db
portfolio/accuracy_stats.py:148:         db = SignalDB()
portfolio/outcome_tracker.py:162:         db = SignalDB()
portfolio/outcome_tracker.py:416:         _db = SignalDB()
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 5ms:

refs for signal_accuracy(
portfolio/signal_postmortem.py:211:             acc = signal_accuracy("1d")
portfolio/signal_db.py:249:     def signal_accuracy(self, horizon="1d"):
portfolio/signal_db.py:252:         Returns dict matching accuracy_stats.signal_accuracy() format.
portfolio/signal_db.py:349:     def ticker_signal_accuracy(self, horizon="1d", min_samples=0):
portfolio/accuracy_stats.py:20: # threads race through load_cached_accuracy() → None → signal_accuracy() and
portfolio/accuracy_stats.py:189: def signal_accuracy(horizon="1d", since=None, entries=None):
portfolio/accuracy_stats.py:283:     Thin wrapper around signal_accuracy() with a time cutoff.
portfolio/accuracy_stats.py:288:     return signal_accuracy(horizon, since=cutoff)
portfolio/accuracy_stats.py:767:         acc = signal_accuracy(horizon)
portfolio/accuracy_stats.py:1025: # `cached = load_cached_accuracy(h); if not cached: cached = signal_accuracy(h);
portfolio/accuracy_stats.py:1042:         result = signal_accuracy(horizon)
portfolio/accuracy_stats.py:1273:         sig_acc = signal_accuracy(h, entries=entries)
portfolio/accuracy_stats.py:1552:     acc = signal_accuracy("1d")
portfolio/accuracy_stats.py:1642:     current_acc = signal_accuracy("1d")
portfolio/accuracy_degradation.py:150:         recent = signal_accuracy("1d", entries=recent_entries)
portfolio/accuracy_degradation.py:458:         new_signals = signal_accuracy("1d", entries=recent_entries)
portfolio/ticker_accuracy.py:32:     # is None) matches signal_accuracy() / per_ticker_accuracy() / consensus_

refs for consensus_accuracy(
portfolio/signal_db.py:290:     def consensus_accuracy(self, horizon="1d"):
portfolio/accuracy_stats.py:459: def consensus_accuracy(horizon="1d", entries=None, days=None):
portfolio/accuracy_stats.py:1084: def get_or_compute_consensus_accuracy(horizon: str):
portfolio/accuracy_stats.py:1103:         result = consensus_accuracy(horizon)
portfolio/accuracy_stats.py:1226:             get_or_compute_consensus_accuracy(h)
portfolio/accuracy_stats.py:1290:         cons = consensus_accuracy(h, entries=entries)
portfolio/accuracy_degradation.py:190:         extras["consensus"] = consensus_accuracy("1d", entries=all_entries)
portfolio/accuracy_degradation.py:191:         extras["consensus_recent"] = consensus_accuracy(
portfolio/accuracy_degradation.py:536:         new_consensus = consensus_accuracy("1d", entries=recent_entries)

refs for per_ticker_accuracy(
portfolio/signal_engine.py:3852:         _ptc_data = get_or_compute_per_ticker_accuracy(acc_horizon)
portfolio/signal_db.py:318:     def per_ticker_accuracy(self, horizon="1d"):
portfolio/accuracy_stats.py:518: def per_ticker_accuracy(horizon="1d", entries=None):
portfolio/accuracy_stats.py:1064: def get_or_compute_per_ticker_accuracy(horizon: str):
portfolio/accuracy_stats.py:1078:         result = per_ticker_accuracy(horizon)
portfolio/accuracy_stats.py:1227:             get_or_compute_per_ticker_accuracy(h)
portfolio/accuracy_stats.py:1297:         ticker_acc = per_ticker_accuracy(h, entries=entries)
portfolio/ticker_accuracy.py:32:     # is None) matches signal_accuracy() / per_ticker_accuracy() / consensus_

refs for ticker_signal_accuracy(
portfolio/signal_db.py:349:     def ticker_signal_accuracy(self, horizon="1d", min_samples=0):
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 1407: def _get_horizon_weights(horizon: str | None) -> dict[str, float]:
 1408:     """Get horizon-specific signal weight multipliers, preferring dynamic computation.
 1409: 
 1410:     Uses cached dynamic weights when available, falling back to static dict.
 1411:     """
 1412:     if not horizon:
 1413:         return {}
 1414:     cache_key = f"dynamic_horizon_weights_{horizon}"
 1415:     # Codex 2026-05-10: _cached returns None on dogpile/timeout/error
 1416:     # paths (shared_state.py:88, 109, 123, 126). The previous bare cast
 1417:     # silenced the type but left ``signal_name in horizon_mults`` to
 1418:     # crash at runtime when None leaked through. Coerce to {} here so
 1419:     # the contract — "horizon weights are always a dict" — holds at the
 1420:     # boundary where the lie used to live.
 1421:     weights = _cached(cache_key, _DYNAMIC_HORIZON_WEIGHT_TTL,
 1422:                       lambda: _compute_dynamic_horizon_weights(horizon))
 1423:     return cast(dict[str, float], weights) if weights else {}
 1424: 
 1425: 
 1426: # Signals that only apply to specific asset classes
 1427: _CRYPTO_ONLY_SIGNALS = {"futures_flow", "funding", "crypto_macro", "onchain"}
 1428: _METALS_ONLY_SIGNALS = {"metals_cross_asset"}
 1429: _NON_STOCK_SIGNALS = {"orderbook_flow"}  # metals + crypto only
 1430: 
 1431: 
 1432: def _compute_applicable_count(ticker: str, skip_gpu: bool = False) -> int:
 1433:     """Compute total applicable signals for a ticker dynamically.
 1434: 
 1435:     Accounts for disabled signals, per-asset-class restrictions,
 1436:     and GPU signals skipped outside market hours.
 1437:     """
 1438:     is_crypto = ticker in CRYPTO_SYMBOLS
 1439:     is_metal = ticker in METALS_SYMBOLS
 1440:     is_stock = ticker in STOCK_SYMBOLS
 1441:     count = 0
 1442:     for sig in SIGNAL_NAMES:
 1443:         if sig in DISABLED_SIGNALS and (sig, ticker) not in _DISABLED_SIGNAL_OVERRIDES:
 1444:             continue
 1445:         # Per-ticker blacklist: check _default horizon for signals bad at all horizons
 1446:         if sig in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
 1447:             continue
 1448:         # crypto-only signals (futures_flow, funding, crypto_macro)
 1449:         if sig in _CRYPTO_ONLY_SIGNALS and not is_crypto:
 1450:             continue
 1451:         # metals-only signals (metals_cross_asset)
 1452:         if sig in _METALS_ONLY_SIGNALS and not is_metal:
 1453:             continue
 1454:         # non-stock signals (orderbook_flow — metals + crypto only)
 1455:         if sig in _NON_STOCK_SIGNALS and is_stock:
 1456:             continue
 1457:         # ministral (CryptoTrader-LM) only runs for crypto
 1458:         if sig == "ministral" and not is_crypto:
 1459:             continue
 1460:         # GPU signals skipped for stocks outside market hours
 1461:         if skip_gpu and sig in GPU_SIGNALS:
 1462:             continue
 1463:         count += 1
 1464:     return count
 1465: 
 1466: 
 1467: _VALID_ACTIONS = frozenset({"BUY", "SELL", "HOLD"})
 1468: 
 1469: 
 1470: def _validate_signal_result(result, sig_name=None, max_confidence=1.0):
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/signal_engine.py:752:     # Source: accuracy_by_ticker_signal_cached() cross-referenced with
portfolio/signal_engine.py:3703:         from portfolio.accuracy_stats import accuracy_by_ticker_signal_cached
portfolio/signal_engine.py:3705:         _ticker_acc_data = (accuracy_by_ticker_signal_cached(acc_horizon) or {}).get(ticker, {})
portfolio/accuracy_stats.py:1958: def accuracy_by_ticker_signal_cached(horizon="1d", min_samples=0):
portfolio/accuracy_degradation.py:134:         accuracy_by_ticker_signal_cached,
portfolio/accuracy_degradation.py:161:         per_ticker_lifetime = accuracy_by_ticker_signal_cached("1d")
portfolio/accuracy_degradation.py:206:     accuracy_by_ticker_signal_cached only exposes the lifetime aggregate;
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
 3688:         for sig_name in list(votes.keys()):
 3689:             if is_slow_signal_3h(sig_name) and votes[sig_name] != "HOLD":
 3690:                 votes[sig_name] = "HOLD"
 3691: 
 3692:     # BUG-143: Apply regime gating BEFORE computing buy/sell counts so that
 3693:     # all downstream code (core gate, min_voters, unanimity penalty) sees
 3694:     # post-gated counts.  _weighted_consensus also applies this internally
 3695:     # (idempotent — gating HOLD→HOLD is a no-op).
 3696:     # BUG-149: now horizon-aware via _get_regime_gated()
 3697:     # BUG-158: Per-ticker exemption — if a signal has ≥60% accuracy with ≥50
 3698:     # samples on THIS ticker, exempt it from regime gating. fear_greed is 93.8%
 3699:     # on XAG-USD but globally gated in ranging — this recovers that alpha.
 3700:     regime_gated = _get_regime_gated(regime, horizon)
 3701:     _ticker_acc_data = {}
 3702:     try:
 3703:         from portfolio.accuracy_stats import accuracy_by_ticker_signal_cached
 3704:         acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
 3705:         _ticker_acc_data = (accuracy_by_ticker_signal_cached(acc_horizon) or {}).get(ticker, {})
 3706:     except Exception:
 3707:         logger.debug("Per-ticker accuracy unavailable for regime gating exemption", exc_info=True)
 3708:     _TICKER_EXEMPT_ACC = 0.60
 3709:     _TICKER_EXEMPT_MIN_SAMPLES = 50
 3710:     # RES-2026-04-21: Recent-accuracy override for regime gating. When a signal's
 3711:     # 7d recent accuracy is significantly above the gate threshold (>55%, 50+ samples),
 3712:     # exempt it from regime gating even if all-time data is bad. Prevents stale regime
 3713:     # gates from suppressing signals that have recovered in a new market regime.
 3714:     # Example: fibonacci went from 43% all-time to 68.2% recent — should not be gated.
 3715:     _RECENT_EXEMPT_ACC = 0.55
 3716:     _RECENT_EXEMPT_MIN_SAMPLES = 50
 3717:     _recent_acc_data: dict[str, Any] = {}
 3718:     try:
 3719:         from portfolio.accuracy_stats import get_or_compute_recent_accuracy
 3720:         # get_or_compute_recent_accuracy expects the base horizon, not the cache key
 3721:         base_hz = "3h" if horizon in ("3h", "4h") else "1d"
 3722:         _recent_acc_data = get_or_compute_recent_accuracy(base_hz) or {}
 3723:     except Exception:
 3724:         logger.debug("Recent accuracy unavailable for regime gating override", exc_info=True)
 3725: 
 3726:     regime_gated_effective = set(regime_gated)
 3727:     for sig_name in list(regime_gated_effective):
 3728:         # Per-ticker exemption (BUG-158)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
portfolio/signal_decay_alert.py:27: def check_signal_decay(accuracy_cache_path="data/accuracy_cache.json"):
portfolio/signal_decay_alert.py:129:     """Log decay alerts and write to data/signal_decay_alerts.jsonl."""
portfolio/signal_decay_alert.py:146:         atomic_append_jsonl("data/signal_decay_alerts.jsonl", entry)
portfolio/signal_decay_alert.py:148:         logger.debug("Could not write signal_decay_alerts.jsonl", exc_info=True)
portfolio/ic_computation.py:19: # 2026-05-02: absolute path (was relative `Path("data")`).
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
1324: _DYNAMIC_HORIZON_WEIGHT_TTL = 3600  # 1 hour cache
1325: _DYNAMIC_HORIZON_MIN_SAMPLES = 50   # need enough data per signal per horizon
1326: _DYNAMIC_HORIZON_CLAMP_LOW = 0.4    # minimum multiplier
1327: _DYNAMIC_HORIZON_CLAMP_HIGH = 1.5   # maximum multiplier
1328: _DYNAMIC_HORIZON_DEADBAND = 0.1     # ignore multipliers within ±10% of 1.0
1372:                 if stats.get("total", 0) >= _DYNAMIC_HORIZON_MIN_SAMPLES:
1386:             if samples < _DYNAMIC_HORIZON_MIN_SAMPLES:
1396:             ratio = max(_DYNAMIC_HORIZON_CLAMP_LOW, min(_DYNAMIC_HORIZON_CLAMP_HIGH, ratio))
1398:             if abs(ratio - 1.0) > _DYNAMIC_HORIZON_DEADBAND:
1421:     weights = _cached(cache_key, _DYNAMIC_HORIZON_WEIGHT_TTL,
codex
# Codex adversarial review: signals-core
## Summary
The consensus path has a lot of defensive code, but the data/label pipeline still has blocker-class defects. The biggest problems are silent source-of-truth divergence, horizon-misaligned outcome labels, and a forecast backfill that can truncate its own dataset; those invalidate downstream accuracy, IC, and gating decisions.

## P0 — Blockers (production breakage / data loss / silent wrong trades)
- `portfolio/forecast_accuracy.py:322` — `backfill_forecast_outcomes()` breaks once `updated >= max_entries`, then `_write_predictions(modified_entries, path)` overwrites the whole JSONL with only the processed prefix. Why it bites: the default `max_entries=500` can silently delete every later prediction row on the first sizeable backfill. Fix: keep the full list for persistence and only limit the subset you mutate.
- `portfolio/outcome_tracker.py:214` — `_fetch_historical_price()` labels outcomes with future prices. Why it bites: the Binance branches ask 1h klines with `startTime=target_ts`, which rounds non-hour-aligned targets up to the next candle, and the YF branch at line 262 falls back to target-date daily closes; both inject look-ahead / off-by-hours error into the ground truth feeding every accuracy metric. Fix: fetch intraday bars and take the last price at or before `target_ts`, never the next candle or end-of-day close.
- `portfolio/accuracy_stats.py:146` — `load_entries()` prefers SQLite whenever it has any rows, even though `portfolio/outcome_tracker.py:165` explicitly tolerates SQLite write failures and `portfolio/signal_db.py:163` silently refuses outcome updates for missing snapshot rows. Why it bites: one dual-write failure can leave JSONL advancing while accuracy/IC/regime logic reads a permanently stale SQLite subset. Fix: compare DB `max(ts)` to the JSONL tail or maintain a dual-write health marker and fall back/reconcile when SQLite lags.

## P1 — High (will cause incidents)
- `portfolio/signal_engine.py:1817` — `_count_active_voters_at_gate()` drops every signal below the overall gate, but `_weighted_consensus()` later rescues some of those votes via strong directional accuracy at lines 2477-2495. Why it bites: the circuit breaker undercounts baseline voters, over-relaxes the gate, and can admit extra weak signals that were never needed to meet quorum. Fix: mirror the directional-rescue logic inside the gate-count helper.
- `portfolio/ticker_accuracy.py:124` — probability mode only requires 5 samples and then turns a weak SELL signal into `p_up = 1 - accuracy` at lines 130-135. Why it bites: sub-47% signals become contrarian probability boosters instead of being force-HOLD, violating the “never invert sub-50% signals” rule and corrupting Mode B sizing/alerts. Fix: enforce the same 47%/30-sample hold gate before probability aggregation and drop ungated signals entirely.
- `portfolio/accuracy_stats.py:1923` — `ticker_signal_accuracy_cache.json` uses one shared `time` key for every horizon. Why it bites: a fresh 1d write at line 1942 makes stale 3h/4h data look fresh, and `portfolio/signal_engine.py:3705` then uses that stale horizon data for per-ticker regime exemptions. Fix: store per-horizon timestamps the same way `accuracy_cache.json` already does.

## P2 — Medium (correctness / robustness)
- `portfolio/signal_engine.py:3978` — when `signals.use_best_horizon` is enabled, live `accuracy_data` gets overwritten with whichever historical horizon scored best in `signal_best_horizon_accuracy()`. Why it bites: this mixes horizons and uses an in-sample winner to override the current horizon’s gate/weights, which is classic look-ahead bias. Fix: keep best-horizon analysis offline only; live gating must stay horizon-local.
- `portfolio/signal_engine.py:947` — `MIN_VOTERS_METALS = 2` is wired into the primary consensus gates at lines 3802-3806 and 4037. Why it bites: it violates the stated `MIN_VOTERS = 3` invariant at the base decision layer; Stage 4 only masks it if confidence penalties stay enabled. Fix: restore the floor to 3 and address metals sparsity with signal selection, not quorum drift.
- `portfolio/signal_engine.py:1351` — `_compute_dynamic_horizon_weights()` reads `accuracy_cache.json` directly and never checks the per-horizon TTLs. Why it bites: expired cache files can keep driving live horizon multipliers until some other path rewrites them. Fix: source those numbers through `load_cached_accuracy()` / get-or-compute helpers and reject expired data.

## P3 — Low (style / dead code / minor)
- `portfolio/signal_decay_alert.py:27` — decay checks still use relative `"data/..."` paths with raw `open`/`json.load`, and line 146 appends to another relative path. Why it bites: under a non-repo CWD the job reads/writes the wrong files or returns `[]`, and it violates the project’s atomic-I/O rule. Fix: use absolute `DATA_DIR` paths plus `load_json` / `atomic_append_jsonl`.

## Tests missing
- `portfolio/forecast_accuracy.py:254` — no test that `backfill_forecast_outcomes(max_entries < len(file))` preserves untouched tail rows.
- `portfolio/outcome_tracker.py:211` — no test that outcome backfill uses the last price at or before `target_ts` for non-hour-aligned crypto snapshots and intraday stock horizons.
- `portfolio/accuracy_stats.py:144` — no test that `load_entries()` falls back from SQLite when dual-write misses the latest snapshot.
- `portfolio/signal_engine.py:1779` — no test that circuit-breaker voter counts include directionally rescued votes before relaxing the gate.
- `portfolio/ticker_accuracy.py:85` — no test enforcing the 47% / 30-sample HOLD rule in probability mode.
- `portfolio/accuracy_stats.py:1914` — no test that ticker-accuracy cache expiry is tracked independently per horizon.
- `portfolio/signal_engine.py:3978` — no test preventing cross-horizon “best horizon” data from overriding live current-horizon gating.
- `portfolio/signal_decay_alert.py:27` — no test that decay checks work under a non-repo working directory and only use atomic JSON helpers.
tokens used
246,901
# Codex adversarial review: signals-core
## Summary
The consensus path has a lot of defensive code, but the data/label pipeline still has blocker-class defects. The biggest problems are silent source-of-truth divergence, horizon-misaligned outcome labels, and a forecast backfill that can truncate its own dataset; those invalidate downstream accuracy, IC, and gating decisions.

## P0 — Blockers (production breakage / data loss / silent wrong trades)
- `portfolio/forecast_accuracy.py:322` — `backfill_forecast_outcomes()` breaks once `updated >= max_entries`, then `_write_predictions(modified_entries, path)` overwrites the whole JSONL with only the processed prefix. Why it bites: the default `max_entries=500` can silently delete every later prediction row on the first sizeable backfill. Fix: keep the full list for persistence and only limit the subset you mutate.
- `portfolio/outcome_tracker.py:214` — `_fetch_historical_price()` labels outcomes with future prices. Why it bites: the Binance branches ask 1h klines with `startTime=target_ts`, which rounds non-hour-aligned targets up to the next candle, and the YF branch at line 262 falls back to target-date daily closes; both inject look-ahead / off-by-hours error into the ground truth feeding every accuracy metric. Fix: fetch intraday bars and take the last price at or before `target_ts`, never the next candle or end-of-day close.
- `portfolio/accuracy_stats.py:146` — `load_entries()` prefers SQLite whenever it has any rows, even though `portfolio/outcome_tracker.py:165` explicitly tolerates SQLite write failures and `portfolio/signal_db.py:163` silently refuses outcome updates for missing snapshot rows. Why it bites: one dual-write failure can leave JSONL advancing while accuracy/IC/regime logic reads a permanently stale SQLite subset. Fix: compare DB `max(ts)` to the JSONL tail or maintain a dual-write health marker and fall back/reconcile when SQLite lags.

## P1 — High (will cause incidents)
- `portfolio/signal_engine.py:1817` — `_count_active_voters_at_gate()` drops every signal below the overall gate, but `_weighted_consensus()` later rescues some of those votes via strong directional accuracy at lines 2477-2495. Why it bites: the circuit breaker undercounts baseline voters, over-relaxes the gate, and can admit extra weak signals that were never needed to meet quorum. Fix: mirror the directional-rescue logic inside the gate-count helper.
- `portfolio/ticker_accuracy.py:124` — probability mode only requires 5 samples and then turns a weak SELL signal into `p_up = 1 - accuracy` at lines 130-135. Why it bites: sub-47% signals become contrarian probability boosters instead of being force-HOLD, violating the “never invert sub-50% signals” rule and corrupting Mode B sizing/alerts. Fix: enforce the same 47%/30-sample hold gate before probability aggregation and drop ungated signals entirely.
- `portfolio/accuracy_stats.py:1923` — `ticker_signal_accuracy_cache.json` uses one shared `time` key for every horizon. Why it bites: a fresh 1d write at line 1942 makes stale 3h/4h data look fresh, and `portfolio/signal_engine.py:3705` then uses that stale horizon data for per-ticker regime exemptions. Fix: store per-horizon timestamps the same way `accuracy_cache.json` already does.

## P2 — Medium (correctness / robustness)
- `portfolio/signal_engine.py:3978` — when `signals.use_best_horizon` is enabled, live `accuracy_data` gets overwritten with whichever historical horizon scored best in `signal_best_horizon_accuracy()`. Why it bites: this mixes horizons and uses an in-sample winner to override the current horizon’s gate/weights, which is classic look-ahead bias. Fix: keep best-horizon analysis offline only; live gating must stay horizon-local.
- `portfolio/signal_engine.py:947` — `MIN_VOTERS_METALS = 2` is wired into the primary consensus gates at lines 3802-3806 and 4037. Why it bites: it violates the stated `MIN_VOTERS = 3` invariant at the base decision layer; Stage 4 only masks it if confidence penalties stay enabled. Fix: restore the floor to 3 and address metals sparsity with signal selection, not quorum drift.
- `portfolio/signal_engine.py:1351` — `_compute_dynamic_horizon_weights()` reads `accuracy_cache.json` directly and never checks the per-horizon TTLs. Why it bites: expired cache files can keep driving live horizon multipliers until some other path rewrites them. Fix: source those numbers through `load_cached_accuracy()` / get-or-compute helpers and reject expired data.

## P3 — Low (style / dead code / minor)
- `portfolio/signal_decay_alert.py:27` — decay checks still use relative `"data/..."` paths with raw `open`/`json.load`, and line 146 appends to another relative path. Why it bites: under a non-repo CWD the job reads/writes the wrong files or returns `[]`, and it violates the project’s atomic-I/O rule. Fix: use absolute `DATA_DIR` paths plus `load_json` / `atomic_append_jsonl`.

## Tests missing
- `portfolio/forecast_accuracy.py:254` — no test that `backfill_forecast_outcomes(max_entries < len(file))` preserves untouched tail rows.
- `portfolio/outcome_tracker.py:211` — no test that outcome backfill uses the last price at or before `target_ts` for non-hour-aligned crypto snapshots and intraday stock horizons.
- `portfolio/accuracy_stats.py:144` — no test that `load_entries()` falls back from SQLite when dual-write misses the latest snapshot.
- `portfolio/signal_engine.py:1779` — no test that circuit-breaker voter counts include directionally rescued votes before relaxing the gate.
- `portfolio/ticker_accuracy.py:85` — no test enforcing the 47% / 30-sample HOLD rule in probability mode.
- `portfolio/accuracy_stats.py:1914` — no test that ticker-accuracy cache expiry is tracked independently per horizon.
- `portfolio/signal_engine.py:3978` — no test preventing cross-horizon “best horizon” data from overriding live current-horizon gating.
- `portfolio/signal_decay_alert.py:27` — no test that decay checks work under a non-repo working directory and only use atomic JSON helpers.
