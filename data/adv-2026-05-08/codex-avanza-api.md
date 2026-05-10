OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-avanza-api
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e0834-03b7-7e11-bc83-6c10a25b909d
--------
user
changes against 'empty-baseline'
  → Inspect diff against merge base
  • Analyze changed files for actionable bugs
  • Produce JSON review findings
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-avanza-api
2026-05-08T15:28:16.673022Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
warning: Not a git repository. Use --no-index to compare two paths outside a working tree
usage: git diff --no-index [<options>] <path> <path> [<pathspec>...]

Diff output format options
    -p, --patch           generate patch
    -s, --no-patch        suppress diff output
    -u                    generate patch
    -U, --unified[=<n>]   generate diffs with <n> lines context
    -W, --[no-]function-context
                          generate diffs with <n> lines context
    --raw                 generate the diff in raw format
    --patch-with-raw      synonym for '-p --raw'
    --patch-with-stat     synonym for '-p --stat'
    --numstat             machine friendly --stat
    --shortstat           output only the last line of --stat
    -X, --dirstat[=<param1>,<param2>...]
                          output the distribution of relative amount of changes for each sub-directory
    --cumulative          synonym for --dirstat=cumulative
    --dirstat-by-file[=<param1>,<param2>...]
                          synonym for --dirstat=files,<param1>,<param2>...
    --check               warn if changes introduce conflict markers or whitespace errors
    --summary             condensed summary such as creations, renames and mode changes
    --name-only           show only names of changed files
    --name-status         show only names and status of changed files
    --stat[=<width>[,<name-width>[,<count>]]]
                          generate diffstat
    --stat-width <width>  generate diffstat with a given width
    --stat-name-width <width>
                          generate diffstat with a given name width
    --stat-graph-width <width>
                          generate diffstat with a given graph width
    --stat-count <count>  generate diffstat with limited lines
    --[no-]compact-summary
                          generate compact summary in diffstat
    --binary              output a binary diff that can be applied
    --[no-]full-index     show full pre- and post-image object names on the "index" lines
    --[no-]color[=<when>] show colored diff
    --ws-error-highlight <kind>
                          highlight whitespace errors in the 'context', 'old' or 'new' lines in the diff
    -z                    do not munge pathnames and use NULs as output field terminators in --raw or --numstat
    --[no-]abbrev[=<n>]   use <n> digits to display object names
    --src-prefix <prefix> show the given source prefix instead of "a/"
    --dst-prefix <prefix> show the given destination prefix instead of "b/"
    --line-prefix <prefix>
                          prepend an additional prefix to every line of output
    --no-prefix           do not show any source or destination prefix
    --default-prefix      use default prefixes a/ and b/
    --inter-hunk-context <n>
                          show context between diff hunks up to the specified number of lines
    --output-indicator-new <char>
                          specify the character to indicate a new line instead of '+'
    --output-indicator-old <char>
                          specify the character to indicate an old line instead of '-'
    --output-indicator-context <char>
                          specify the character to indicate a context instead of ' '

Diff rename options
    -B, --break-rewrites[=<n>[/<m>]]
                          break complete rewrite changes into pairs of delete and create
    -M, --find-renames[=<n>]
                          detect renames
    -D, --irreversible-delete
                          omit the preimage for deletes
    -C, --find-copies[=<n>]
                          detect copies
    --[no-]find-copies-harder
                          use unmodified files as source to find copies
    --no-renames          disable rename detection
    --[no-]rename-empty   use empty blobs as rename source
    --[no-]follow         continue listing the history of a file beyond renames
    -l <n>                prevent rename/copy detection if the number of rename/copy targets exceeds given limit

Diff algorithm options
    --minimal             produce the smallest possible diff
    -w, --ignore-all-space
                          ignore whitespace when comparing lines
    -b, --ignore-space-change
                          ignore changes in amount of whitespace
    --ignore-space-at-eol ignore changes in whitespace at EOL
    --ignore-cr-at-eol    ignore carrier-return at the end of line
    --ignore-blank-lines  ignore changes whose lines are all blank
    -I, --[no-]ignore-matching-lines <regex>
                          ignore changes whose all lines match <regex>
    --[no-]indent-heuristic
                          heuristic to shift diff hunk boundaries for easy reading
    --patience            generate diff using the "patience diff" algorithm
    --histogram           generate diff using the "histogram diff" algorithm
    --diff-algorithm <algorithm>
                          choose a diff algorithm
    --anchored <text>     generate diff using the "anchored diff" algorithm
    --word-diff[=<mode>]  show word diff, using <mode> to delimit changed words
    --word-diff-regex <regex>
                          use <regex> to decide what a word is
    --color-words[=<regex>]
                          equivalent to --word-diff=color --word-diff-regex=<regex>
    --[no-]color-moved[=<mode>]
                          moved lines of code are colored differently
    --[no-]color-moved-ws <mode>
                          how white spaces are ignored in --color-moved

Other diff options
    --[no-]relative[=<prefix>]
                          when run from subdir, exclude changes outside and show relative paths
    -a, --[no-]text       treat all files as text
    -R                    swap two inputs, reverse the diff
    --[no-]exit-code      exit with 1 if there were differences, 0 otherwise
    --[no-]quiet          disable all output of the program
    --[no-]ext-diff       allow an external diff helper to be executed
    --[no-]textconv       run external text conversion filters when comparing binary files
    --ignore-submodules[=<when>]
                          ignore changes to submodules in the diff generation
    --submodule[=<format>]
                          specify how differences in submodules are shown
    --ita-invisible-in-index
                          hide 'git add -N' entries from the index
    --ita-visible-in-index
                          treat 'git add -N' entries as real in the index
    -S <string>           look for differences that change the number of occurrences of the specified string
    -G <regex>            look for differences that change the number of occurrences of the specified regex
    --pickaxe-all         show all changes in the changeset with -S or -G
    --pickaxe-regex       treat <string> in -S as extended POSIX regular expression
    -O <file>             control the order in which files appear in the output
    --rotate-to <path>    show the change in the specified path first
    --skip-to <path>      skip the output to the specified path
    --find-object <object-id>
                          look for differences that change the number of occurrences of the specified object
    --diff-filter [(A|C|D|M|R|T|U|X|B)...[*]]
                          select files by diff type
    --max-depth <depth>   maximum tree depth to recurse
    --output <file>       output to a specific file


 exited 1 in 538ms:
warning: Not a git repository. Use --no-index to compare two paths outside a working tree
usage: git diff --no-index [<options>] <path> <path> [<pathspec>...]

Diff output format options
    -p, --patch           generate patch
    -s, --no-patch        suppress diff output
    -u                    generate patch
    -U, --unified[=<n>]   generate diffs with <n> lines context
    -W, --[no-]function-context
                          generate diffs with <n> lines context
    --raw                 generate the diff in raw format
    --patch-with-raw      synonym for '-p --raw'
    --patch-with-stat     synonym for '-p --stat'
    --numstat             machine friendly --stat
    --shortstat           output only the last line of --stat
    -X, --dirstat[=<param1>,<param2>...]
                          output the distribution of relative amount of changes for each sub-directory
    --cumulative          synonym for --dirstat=cumulative
    --dirstat-by-file[=<param1>,<param2>...]
                          synonym for --dirstat=files,<param1>,<param2>...
    --check               warn if changes introduce conflict markers or whitespace errors
    --summary             condensed summary such as creations, renames and mode changes
    --name-only           show only names of changed files
    --name-status         show only names and status of changed files
    --stat[=<width>[,<name-width>[,<count>]]]
                          generate diffstat
    --stat-width <width>  generate diffstat with a given width
    --stat-name-width <width>
                          generate diffstat with a given name width
    --stat-graph-width <width>
                          generate diffstat with a given graph width
    --stat-count <count>  generate diffstat with limited lines
    --[no-]compact-summary
                          generate compact summary in diffstat
    --binary              output a binary diff that can be applied
    --[no-]full-index     show full pre- and post-image object names on the "index" lines
    --[no-]color[=<when>] show colored diff
    --ws-error-highlight <kind>
                          highlight whitespace errors in the 'context', 'old' or 'new' lines in the diff
    -z                    do not munge pathnames and use NULs as output field terminators in --raw or --numstat
    --[no-]abbrev[=<n>]   use <n> digits to display object names
    --src-prefix <prefix> show the given source prefix instead of "a/"
    --dst-prefix <prefix> show the given destination prefix instead of "b/"
    --line-prefix <prefix>
                          prepend an additional prefix to every line of output
    --no-prefix           do not show any source or destination prefix
    --default-prefix      use default prefixes a/ and b/
    --inter-hunk-context <n>
                          show context between diff hunks up to the specified number of lines
    --output-indicator-new <char>
                          specify the character to indicate a new line instead of '+'
    --output-indicator-old <char>
                          specify the character to indicate an old line instead of '-'
    --output-indicator-context <char>
                          specify the character to indicate a context instead of ' '

Diff rename options
    -B, --break-rewrites[=<n>[/<m>]]
                          break complete rewrite changes into pairs of delete and create
    -M, --find-renames[=<n>]
                          detect renames
    -D, --irreversible-delete
                          omit the preimage for deletes
    -C, --find-copies[=<n>]
                          detect copies
    --[no-]find-copies-harder
                          use unmodified files as source to find copies
    --no-renames          disable rename detection
    --[no-]rename-empty   use empty blobs as rename source
    --[no-]follow         continue listing the history of a file beyond renames
    -l <n>                prevent rename/copy detection if the number of rename/copy targets exceeds given limit

Diff algorithm options
    --minimal             produce the smallest possible diff
    -w, --ignore-all-space
                          ignore whitespace when comparing lines
    -b, --ignore-space-change
                          ignore changes in amount of whitespace
    --ignore-space-at-eol ignore changes in whitespace at EOL
    --ignore-cr-at-eol    ignore carrier-return at the end of line
    --ignore-blank-lines  ignore changes whose lines are all blank
    -I, --[no-]ignore-matching-lines <regex>
                          ignore changes whose all lines match <regex>
    --[no-]indent-heuristic
                          heuristic to shift diff hunk boundaries for easy reading
    --patience            generate diff using the "patience diff" algorithm
    --histogram           generate diff using the "histogram diff" algorithm
    --diff-algorithm <algorithm>
                          choose a diff algorithm
    --anchored <text>     generate diff using the "anchored diff" algorithm
    --word-diff[=<mode>]  show word diff, using <mode> to delimit changed words
    --word-diff-regex <regex>
                          use <regex> to decide what a word is
    --color-words[=<regex>]
                          equivalent to --word-diff=color --word-diff-regex=<regex>
    --[no-]color-moved[=<mode>]
                          moved lines of code are colored differently
    --[no-]color-moved-ws <mode>
                          how white spaces are ignored in --color-moved

Other diff options
    --[no-]relative[=<prefix>]
                          when run from subdir, exclude changes outside and show relative paths
    -a, --[no-]text       treat all files as text
    -R                    swap two inputs, reverse the diff
    --[no-]exit-code      exit with 1 if there were differences, 0 otherwise
    --[no-]quiet          disable all output of the program
    --[no-]ext-diff       allow an external diff helper to be executed
    --[no-]textconv       run external text conversion filters when comparing binary files
    --ignore-submodules[=<when>]
                          ignore changes to submodules in the diff generation
    --submodule[=<format>]
                          specify how differences in submodules are shown
    --ita-invisible-in-index
                          hide 'git add -N' entries from the index
    --ita-visible-in-index
                          treat 'git add -N' entries as real in the index
    -S <string>           look for differences that change the number of occurrences of the specified string
    -G <regex>            look for differences that change the number of occurrences of the specified regex
    --pickaxe-all         show all changes in the changeset with -S or -G
    --pickaxe-regex       treat <string> in -S as extended POSIX regular expression
    -O <file>             control the order in which files appear in the output
    --rotate-to <path>    show the change in the specified path first
    --skip-to <path>      skip the output to the specified path
    --find-object <object-id>
                          look for differences that change the number of occurrences of the specified object
    --diff-filter [(A|C|D|M|R|T|U|X|B)...[*]]
                          select files by diff type
    --max-depth <depth>   maximum tree depth to recurse
    --output <file>       output to a specific file


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Force' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 726ms:

    Directory: C:\Users\CodexSandboxOffline\.codex\.sandbox\cwd\15780932ee6abedd

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    17:25                [44;1mportfolio[0m
-a-h-          2026-05-08    17:25             58 .git
-a---          2026-05-08    17:25             46 BASELINE.md


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content .git; Get-Location; Get-ChildItem -Force portfolio' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 654ms:
gitdir: Q:/finance-analyzer/.git/worktrees/adv-avanza-api

[32;1mPath[0m
[32;1m----[0m
Q:\finance-analyzer\.worktrees\adv-avanza-api

[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\ava
                      nza
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio
[32;1mPSChildName         : [0mavanza
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mavanza
[32;1mParent              : [0mQ:\finance-analyzer\.worktrees\adv-avanza-api\portfolio
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-08 17:25:55
[32;1mCreationTimeUtc     : [0m2026-05-08 15:25:55
[32;1mLastAccessTime      : [0m2026-05-08 17:28:07
[32;1mLastAccessTimeUtc   : [0m2026-05-08 15:28:07
[32;1mLastWriteTime       : [0m2026-05-08 17:25:55
[32;1mLastWriteTimeUtc    : [0m2026-05-08 15:25:55
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mavanza
[32;1mResolvedTarget      : [0mQ:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m13438
[32;1mName          : [0mavanza_client.py


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m15330
[32;1mName          : [0mavanza_control.py


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m3643
[32;1mName          : [0mavanza_order_lock.py


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m16708
[32;1mName          : [0mavanza_orders.py


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m7995
[32;1mName          : [0mavanza_resilient_page.py


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m47835
[32;1mName          : [0mavanza_session.py


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m4184
[32;1mName          : [0mavanza_tracker.py



exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel' in Q:\finance-analyzer\.worktrees\adv-avanza-api
2026-05-08T15:28:25.351000Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-avanza-api'
'Q:/finance-analyzer/.worktrees/adv-avanza-api/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-avanza-api

 exited 1 in 613ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-avanza-api'
'Q:/finance-analyzer/.worktrees/adv-avanza-api/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-avanza-api

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-avanza-api diff 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 546ms:
diff --git a/portfolio/avanza/__init__.py b/portfolio/avanza/__init__.py
new file mode 100644
index 00000000..e4ce30a2
--- /dev/null
+++ b/portfolio/avanza/__init__.py
@@ -0,0 +1,44 @@
+"""Unified Avanza API package.
+
+    from portfolio.avanza import get_quote, place_order, get_positions
+"""
+
+# Auth & client
+from portfolio.avanza.auth import AvanzaAuth, AuthError
+from portfolio.avanza.client import AvanzaClient
+
+# Account
+from portfolio.avanza.account import get_buying_power, get_positions, get_transactions
+
+# Market data
+from portfolio.avanza.market_data import get_instrument_info, get_market_data, get_news, get_ohlc, get_quote
+
+# Search
+from portfolio.avanza.search import find_certificates, find_warrants, search
+
+# Tick rules
+from portfolio.avanza.tick_rules import clear_cache as clear_tick_cache, get_tick_rules, round_to_tick
+
+# Scanner
+from portfolio.avanza.scanner import ScannedInstrument, format_scan_results, scan_instruments
+
+# Streaming
+from portfolio.avanza.streaming import AvanzaStream
+
+# Trading
+from portfolio.avanza.trading import (
+    cancel_order, delete_stop_loss, get_deals, get_orders, get_stop_losses,
+    modify_order, place_order, place_stop_loss, place_trailing_stop,
+)
+
+__all__ = [
+    "AvanzaAuth", "AuthError", "AvanzaClient", "AvanzaStream", "ScannedInstrument",
+    "scan_instruments", "format_scan_results",
+    "get_positions", "get_buying_power", "get_transactions",
+    "get_quote", "get_market_data", "get_ohlc", "get_instrument_info", "get_news",
+    "search", "find_warrants", "find_certificates",
+    "get_tick_rules", "round_to_tick", "clear_tick_cache",
+    "place_order", "modify_order", "cancel_order",
+    "get_orders", "get_deals",
+    "place_stop_loss", "place_trailing_stop", "get_stop_losses", "delete_stop_loss",
+]
diff --git a/portfolio/avanza/account.py b/portfolio/avanza/account.py
new file mode 100644
index 00000000..170403b0
--- /dev/null
+++ b/portfolio/avanza/account.py
@@ -0,0 +1,149 @@
+"""Account data — positions, buying power, transactions.
+
+Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
+raw delegators for account-level queries.
+"""
+
+from __future__ import annotations
+
+import logging
+from collections.abc import Sequence
+from datetime import date
+from typing import Any
+
+from avanza.constants import TransactionsDetailsType
+
+from portfolio.avanza.client import AvanzaClient
+from portfolio.avanza.types import AccountCash, Position, Transaction
+
+logger = logging.getLogger("portfolio.avanza.account")
+
+
+# ---------------------------------------------------------------------------
+# Public API
+# ---------------------------------------------------------------------------
+
+
+def get_positions(account_id: str | None = None) -> list[Position]:
+    """Fetch all positions, optionally filtered to a single account.
+
+    Args:
+        account_id: When provided only positions for this account are
+            returned.  Otherwise all positions across all accounts.
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.Position`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: Any = client.get_positions_raw()
+
+    # The API may return a dict with a nested positions list, or a list.
+    positions_raw: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        # Prefer "withOrderbook" (newer API), fall back to "positions"
+        positions_raw = raw.get("withOrderbook", raw.get("positions", []))
+    elif isinstance(raw, list):
+        positions_raw = raw
+    else:
+        positions_raw = []
+
+    positions = [Position.from_api(p) for p in positions_raw]
+
+    if account_id is not None:
+        positions = [p for p in positions if p.account_id == str(account_id)]
+
+    logger.debug(
+        "get_positions account_id=%s total=%d filtered=%d",
+        account_id,
+        len(positions_raw),
+        len(positions),
+    )
+    return positions
+
+
+def get_buying_power(account_id: str | None = None) -> AccountCash:
+    """Fetch buying power / cash info for a specific account.
+
+    Args:
+        account_id: Account to query.  Defaults to the client's configured
+            account.
+
+    Returns:
+        :class:`~portfolio.avanza.types.AccountCash`.
+    """
+    client = AvanzaClient.get_instance()
+    acct = str(account_id) if account_id else client.account_id
+    raw: Any = client.get_overview_raw()
+
+    # The overview contains a list of accounts — find the right one.
+    accounts: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        accounts = raw.get("accounts", [])
+    elif isinstance(raw, list):
+        accounts = raw
+    else:
+        accounts = []
+
+    for account in accounts:
+        if str(account.get("accountId", account.get("id", ""))) == acct:
+            logger.debug("get_buying_power account_id=%s found", acct)
+            return AccountCash.from_api(account)
+
+    # Account not found — return zeroes
+    logger.warning("get_buying_power account_id=%s not found in overview", acct)
+    return AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)
+
+
+def get_transactions(
+    from_date: str,
+    to_date: str,
+    types: Sequence[str] | None = None,
+    account_id: str | None = None,
+) -> list[Transaction]:
+    """Fetch historical transactions.
+
+    Args:
+        from_date: Start date (ISO-8601, e.g. ``"2026-01-01"``).
+        to_date: End date (ISO-8601).
+        types: Transaction type filters (e.g. ``["BUY", "SELL"]``).
+            When *None* all types are returned.
+        account_id: Unused by the library call but kept for future
+            server-side filtering.
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.Transaction`.
+    """
+    client = AvanzaClient.get_instance()
+
+    tx_types: list[TransactionsDetailsType] | None = None
+    if types:
+        tx_types = [TransactionsDetailsType(t) for t in types]
+
+    raw: Any = client.avanza.get_transactions_details(
+        transaction_details_types=tx_types or [],
+        transactions_from=date.fromisoformat(from_date),
+        transactions_to=date.fromisoformat(to_date),
+    )
+
+    # The API may return a dict with a "transactions" key, or a list.
+    tx_list: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        tx_list = raw.get("transactions", [])
+    elif isinstance(raw, list):
+        tx_list = raw
+    else:
+        tx_list = []
+
+    transactions = [Transaction.from_api(t) for t in tx_list]
+
+    if account_id is not None:
+        transactions = [t for t in transactions if t.account_id == str(account_id)]
+
+    logger.debug(
+        "get_transactions from=%s to=%s types=%s count=%d",
+        from_date,
+        to_date,
+        types,
+        len(transactions),
+    )
+    return transactions
diff --git a/portfolio/avanza/auth.py b/portfolio/avanza/auth.py
new file mode 100644
index 00000000..17376906
--- /dev/null
+++ b/portfolio/avanza/auth.py
@@ -0,0 +1,121 @@
+"""Thread-safe TOTP authentication singleton for Avanza.
+
+Wraps the ``avanza-api`` library's ``Avanza`` class with a double-checked
+locking singleton so that the entire application shares one authenticated
+session regardless of how many threads call ``get_instance()``.
+"""
+
+from __future__ import annotations
+
+import logging
+import threading
+from typing import Any
+
+logger = logging.getLogger("portfolio.avanza.auth")
+
+
+class AuthError(Exception):
+    """Raised when Avanza authentication fails."""
+
+
+def _create_avanza_client(credentials: dict[str, str]) -> Any:
+    """Create and return an authenticated ``avanza.Avanza`` instance.
+
+    Separated from :class:`AvanzaAuth` to allow easy mocking in tests
+    (patch ``portfolio.avanza.auth._create_avanza_client``).
+
+    Args:
+        credentials: Dict with keys ``username``, ``password``, ``totpSecret``.
+
+    Returns:
+        An authenticated ``avanza.Avanza`` instance.
+
+    Raises:
+        AuthError: If authentication fails.
+    """
+    try:
+        from avanza import Avanza  # noqa: WPS433 — late import
+
+        client = Avanza(credentials, quiet=True)
+        return client
+    except Exception as exc:
+        raise AuthError(f"Avanza authentication failed: {exc}") from exc
+
+
+class AvanzaAuth:
+    """Thread-safe singleton managing Avanza TOTP authentication.
+
+    Usage::
+
+        auth = AvanzaAuth.get_instance(username, password, totp_secret)
+        auth.client  # -> avanza.Avanza instance
+
+    Call ``AvanzaAuth.reset()`` to tear down the singleton (e.g. in tests or
+    on session expiry).
+    """
+
+    _instance: AvanzaAuth | None = None
+    _lock = threading.Lock()
+
+    def __init__(
+        self,
+        client: Any,
+        push_subscription_id: str,
+        csrf_token: str,
+        authentication_session: str,
+        customer_id: str,
+    ) -> None:
+        self.client = client
+        self.push_subscription_id = push_subscription_id
+        self.csrf_token = csrf_token
+        self.authentication_session = authentication_session
+        self.customer_id = customer_id
+
+    @classmethod
+    def get_instance(
+        cls,
+        username: str,
+        password: str,
+        totp_secret: str,
+    ) -> AvanzaAuth:
+        """Return the singleton, creating it on first call.
+
+        Uses double-checked locking so that only the first caller pays the
+        cost of TOTP authentication; subsequent callers return immediately.
+        """
+        if cls._instance is not None:
+            return cls._instance
+
+        with cls._lock:
+            # Double-check after acquiring the lock
+            if cls._instance is not None:
+                return cls._instance
+
+            credentials = {
+                "username": username,
+                "password": password,
+                "totpSecret": totp_secret,
+            }
+
+            client = _create_avanza_client(credentials)
+
+            instance = cls(
+                client=client,
+                push_subscription_id=getattr(client, "_push_subscription_id", ""),
+                csrf_token=getattr(client, "_security_token", ""),
+                authentication_session=getattr(client, "_authentication_session", ""),
+                customer_id=getattr(client, "_customer_id", ""),
+            )
+            cls._instance = instance
+            logger.info(
+                "AvanzaAuth singleton created (customer_id=%s)",
+                instance.customer_id,
+            )
+            return instance
+
+    @classmethod
+    def reset(cls) -> None:
+        """Tear down the singleton — useful for tests or re-auth."""
+        with cls._lock:
+            cls._instance = None
+            logger.info("AvanzaAuth singleton reset")
diff --git a/portfolio/avanza/client.py b/portfolio/avanza/client.py
new file mode 100644
index 00000000..a860e833
--- /dev/null
+++ b/portfolio/avanza/client.py
@@ -0,0 +1,133 @@
+"""Singleton HTTP client wrapping the avanza-api library.
+
+Provides raw delegator methods that return whatever the underlying
+``avanza.Avanza`` instance returns.  Typed higher-level modules (market
+data, trading, account, etc.) will wrap these delegators and return our
+own dataclasses from :mod:`portfolio.avanza.types`.
+"""
+
+from __future__ import annotations
+
+import logging
+import threading
+from typing import Any
+
+from portfolio.avanza.auth import AvanzaAuth
+
+logger = logging.getLogger("portfolio.avanza.client")
+
+DEFAULT_ACCOUNT_ID = "1625505"
+
+
+class AvanzaClient:
+    """Singleton client wrapping the avanza-api library.
+
+    Usage::
+
+        client = AvanzaClient.get_instance(config)
+        raw = client.get_market_data_raw("2213050")
+    """
+
+    _instance: AvanzaClient | None = None
+    _lock = threading.Lock()
+
+    def __init__(self, auth: AvanzaAuth, account_id: str) -> None:
+        self._auth = auth
+        self._account_id = account_id
+
+    @classmethod
+    def get_instance(cls, config: dict[str, Any] | None = None) -> AvanzaClient:
+        """Return the singleton, creating it on first call.
+
+        Args:
+            config: Application config dict.  Must contain an ``"avanza"`` key
+                with ``"username"``, ``"password"``, and ``"totp_secret"`` when
+                creating for the first time.  Ignored on subsequent calls.
+        """
+        if cls._instance is not None:
+            return cls._instance
+
+        with cls._lock:
+            if cls._instance is not None:
+                return cls._instance
+
+            if config is None:
+                raise ValueError(
+                    "AvanzaClient.get_instance() requires config on first call"
+                )
+
+            avanza_cfg = config.get("avanza", {})
+            auth = AvanzaAuth.get_instance(
+                username=avanza_cfg["username"],
+                password=avanza_cfg["password"],
+                totp_secret=avanza_cfg["totp_secret"],
+            )
+            account_id = str(avanza_cfg.get("account_id", DEFAULT_ACCOUNT_ID))
+
+            instance = cls(auth=auth, account_id=account_id)
+            cls._instance = instance
+            logger.info(
+                "AvanzaClient singleton created (account_id=%s)", account_id
+            )
+            return instance
+
+    @classmethod
+    def reset(cls) -> None:
+        """Tear down the singleton."""
+        with cls._lock:
+            cls._instance = None
+            logger.info("AvanzaClient singleton reset")
+
+    # ------------------------------------------------------------------
+    # Convenience properties
+    # ------------------------------------------------------------------
+
+    @property
+    def account_id(self) -> str:
+        return self._account_id
+
+    @property
+    def avanza(self) -> Any:
+        """The underlying ``avanza.Avanza`` instance."""
+        return self._auth.client
+
+    @property
+    def push_subscription_id(self) -> str:
+        return self._auth.push_subscription_id
+
+    @property
+    def csrf_token(self) -> str:
+        return self._auth.csrf_token
+
+    @property
+    def session(self) -> Any:
+        """The underlying ``requests.Session`` used by the avanza-api library."""
+        return self._auth.client._session
+
+    # ------------------------------------------------------------------
+    # Raw delegators — return whatever the library returns
+    # ------------------------------------------------------------------
+
+    def get_positions_raw(self) -> Any:
+        return self.avanza.get_accounts_positions()
+
+    def get_overview_raw(self) -> Any:
+        return self.avanza.get_overview()
+
+    def get_market_data_raw(self, ob_id: str) -> Any:
+        return self.avanza.get_market_data(ob_id)
+
+    def get_order_book_raw(self, ob_id: str) -> Any:
+        return self.avanza.get_order_book(ob_id)
+
+    def get_deals_raw(self) -> Any:
+        return self.avanza.get_deals()
+
+    def get_orders_raw(self) -> Any:
+        return self.avanza.get_orders()
+
+    def get_all_stop_losses_raw(self) -> Any:
+        return self.avanza.get_all_stop_losses()
+
+    def get_news_raw(self, ob_id: str) -> Any:
+        return self.avanza.get_news(ob_id)
diff --git a/portfolio/avanza/market_data.py b/portfolio/avanza/market_data.py
new file mode 100644
index 00000000..b23f8eea
--- /dev/null
+++ b/portfolio/avanza/market_data.py
@@ -0,0 +1,151 @@
+"""Market data retrieval — quotes, depth, OHLC, instrument info, news.
+
+Thin typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
+raw delegators.  Every function returns our own dataclasses from
+:mod:`portfolio.avanza.types`.
+"""
+
+from __future__ import annotations
+
+import logging
+from typing import Any
+
+from avanza.constants import Resolution, TimePeriod
+
+from portfolio.avanza.client import AvanzaClient
+from portfolio.avanza.types import (
+    OHLC,
+    InstrumentInfo,
+    MarketData,
+    NewsArticle,
+    Quote,
+)
+
+logger = logging.getLogger("portfolio.avanza.market_data")
+
+# ---------------------------------------------------------------------------
+# Resolution lookup (period -> sensible default resolution)
+# ---------------------------------------------------------------------------
+
+_DEFAULT_RESOLUTION: dict[str, Resolution] = {
+    "TODAY": Resolution.THIRTY_MINUTES,
+    "ONE_WEEK": Resolution.THIRTY_MINUTES,
+    "ONE_MONTH": Resolution.DAY,
+    "THREE_MONTHS": Resolution.DAY,
+    "THIS_YEAR": Resolution.WEEK,
+    "ONE_YEAR": Resolution.WEEK,
+    "THREE_YEARS": Resolution.MONTH,
+    "FIVE_YEARS": Resolution.MONTH,
+    "INFINITY": Resolution.MONTH,
+}
+
+
+# ---------------------------------------------------------------------------
+# Public API
+# ---------------------------------------------------------------------------
+
+
+def get_quote(ob_id: str, instrument_type: str = "certificate") -> Quote:
+    """Fetch a live quote for the given orderbook ID.
+
+    Calls ``client.avanza.get_instrument(type, id)`` and parses the
+    result into a :class:`~portfolio.avanza.types.Quote`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: dict[str, Any] = client.avanza.get_instrument(instrument_type, ob_id)
+    logger.debug("get_quote ob_id=%s raw_keys=%s", ob_id, list(raw.keys()))
+    return Quote.from_api(raw)
+
+
+def get_market_data(ob_id: str) -> MarketData:
+    """Fetch full market data (quote + depth + recent trades).
+
+    Calls ``client.get_market_data_raw(id)`` and parses the result
+    into a :class:`~portfolio.avanza.types.MarketData`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: dict[str, Any] = client.get_market_data_raw(ob_id)
+    logger.debug("get_market_data ob_id=%s raw_keys=%s", ob_id, list(raw.keys()))
+    return MarketData.from_api(raw)
+
+
+def get_ohlc(
+    ob_id: str,
+    period: str = "ONE_MONTH",
+    resolution: str | None = None,
+) -> list[OHLC]:
+    """Fetch OHLCV candles for the given orderbook ID.
+
+    Args:
+        ob_id: Avanza orderbook ID.
+        period: Time period string (e.g. ``"ONE_MONTH"``, ``"ONE_WEEK"``).
+        resolution: Optional resolution override.  When *None* a sensible
+            default is chosen based on *period*.
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.OHLC` candles.
+    """
+    client = AvanzaClient.get_instance()
+
+    tp = TimePeriod[period]
+    if resolution is not None:
+        res = Resolution[resolution]
+    else:
+        res = _DEFAULT_RESOLUTION.get(period, Resolution.DAY)
+
+    raw: Any = client.avanza.get_chart_data(ob_id, tp, res)
+    logger.debug(
+        "get_ohlc ob_id=%s period=%s resolution=%s candles=%d",
+        ob_id,
+        period,
+        res.name,
+        len(raw) if isinstance(raw, list) else 0,
+    )
+
+    # The API may return a dict with an "ohlc" key or a plain list.
+    candles: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        candles = raw.get("ohlc", raw.get("dataPoints", []))
+    elif isinstance(raw, list):
+        candles = raw
+    else:
+        candles = []
+
+    return [OHLC.from_api(c) for c in candles]
+
+
+def get_instrument_info(
+    ob_id: str,
+    instrument_type: str = "certificate",
+) -> InstrumentInfo:
+    """Fetch instrument metadata (leverage, barrier, underlying, etc.).
+
+    Calls ``client.avanza.get_instrument(type, id)`` and parses the
+    result into a :class:`~portfolio.avanza.types.InstrumentInfo`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: dict[str, Any] = client.avanza.get_instrument(instrument_type, ob_id)
+    logger.debug("get_instrument_info ob_id=%s name=%s", ob_id, raw.get("name"))
+    return InstrumentInfo.from_api(raw)
+
+
+def get_news(ob_id: str) -> list[NewsArticle]:
+    """Fetch news articles linked to the given orderbook ID.
+
+    Calls ``client.get_news_raw(id)`` and parses the result into a
+    list of :class:`~portfolio.avanza.types.NewsArticle`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: Any = client.get_news_raw(ob_id)
+    logger.debug("get_news ob_id=%s", ob_id)
+
+    # The API may return a list directly or a dict with an "articles" key.
+    articles: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        articles = raw.get("articles", raw.get("news", []))
+    elif isinstance(raw, list):
+        articles = raw
+    else:
+        articles = []
+
+    return [NewsArticle.from_api(a) for a in articles]
diff --git a/portfolio/avanza/scanner.py b/portfolio/avanza/scanner.py
new file mode 100644
index 00000000..8260340c
--- /dev/null
+++ b/portfolio/avanza/scanner.py
@@ -0,0 +1,375 @@
+"""Instrument scanner — find and rank the best warrants/certificates.
+
+Chains search → detail fetch → ranking to answer questions like:
+"Find the best bull mini-future for oil right now"
+
+Works with EITHER auth method:
+- TOTP (AvanzaClient) — preferred, faster
+- BankID session (avanza_session.api_get/api_post) — fallback
+
+Usage:
+    from portfolio.avanza.scanner import scan_instruments
+
+    results = scan_instruments(
+        query="OLJA",           # underlying asset keyword
+        direction="BULL",       # BULL or BEAR
+        instrument_type="certificate",  # certificate, warrant, or None for both
+        sort_by="spread",       # spread, leverage, price, barrier_distance
+        limit=10,
+    )
+    for r in results:
+        print(f"{r['name']:40s} lev={r['leverage']:5.1f}x  spread={r['spread_pct']:.2f}%  bid={r['bid']}")
+"""
+
+from __future__ import annotations
+
+import logging
+import time
+from concurrent.futures import ThreadPoolExecutor, as_completed
+from contextlib import suppress
+from dataclasses import dataclass
+
+from portfolio.avanza.types import _val
+
+logger = logging.getLogger("portfolio.avanza.scanner")
+
+
+# ---------------------------------------------------------------------------
+# Dual-auth API helpers — try TOTP first, fall back to BankID session
+# ---------------------------------------------------------------------------
+
+def _get_api():
+    """Return (search_fn, instrument_fn, marketdata_fn, thread_safe) that work
+    with whichever auth is currently available.
+
+    Returns:
+        Tuple of four:
+        - search(instrument_type_str, query, limit) -> dict or list
+        - get_instrument(api_type, ob_id) -> dict
+        - get_market_data(ob_id) -> dict
+        - thread_safe: bool — True for TOTP (requests.Session), False for BankID (Playwright)
+    """
+    # Try TOTP client first (thread-safe, supports parallel fetching)
+    try:
+        from portfolio.avanza.client import AvanzaClient
+        client = AvanzaClient.get_instance()
+        avanza = client.avanza
+
+        def _search(itype_str, query, limit):
+            from avanza.constants import InstrumentType
+            return avanza.search_for_instrument(InstrumentType(itype_str), query, limit)
+
+        def _instrument(api_type, ob_id):
+            return avanza.get_instrument(api_type, ob_id)
+
+        def _marketdata(ob_id):
+            return avanza.get_market_data(ob_id)
+
+        logger.debug("Scanner using TOTP client (thread-safe)")
+        return _search, _instrument, _marketdata, True
+    except Exception:
+        logger.debug("TOTP client unavailable, falling back to BankID session")
+
+    # Fall back to BankID session (Playwright — NOT thread-safe, must be sequential)
+    try:
+        from portfolio.avanza_session import api_get, api_post
+
+        def _search(itype_str, query, limit):
+            return api_post("/_api/search/filtered-search", {"query": query, "limit": limit})
+
+        def _instrument(api_type, ob_id):
+            return api_get(f"/_api/market-guide/{api_type}/{ob_id}")
+
+        def _marketdata(ob_id):
+            try:
+                return api_get(f"/_api/trading-critical/rest/marketdata/{ob_id}")
+            except Exception:
+                return {}
+
+        logger.debug("Scanner using BankID session (sequential only)")
+        return _search, _instrument, _marketdata, False
+    except Exception as e:
+        raise RuntimeError(
+            "No Avanza auth available. Either configure TOTP credentials "
+            "or run scripts/avanza_login.py for BankID session."
+        ) from e
+
+
+@dataclass
+class ScannedInstrument:
+    """Rich instrument data combining search + market-guide + marketdata."""
+
+    orderbook_id: str
+    name: str
+    instrument_type: str  # CERTIFICATE, WARRANT, etc.
+    direction: str  # BULL, BEAR, LONG, SHORT, or ""
+
+    # Price
+    bid: float | None
+    ask: float | None
+    last: float | None
+    spread_pct: float | None  # (ask-bid)/bid * 100
+
+    # Instrument details
+    leverage: float | None
+    barrier: float | None
+    barrier_distance_pct: float | None  # distance from last to barrier
+
+    # Underlying
+    underlying_name: str
+    underlying_price: float | None
+
+    # Market quality
+    volume_today: int
+    turnover_today: float
+    market_maker: bool
+
+    # Order depth (best level)
+    bid_volume: int
+    ask_volume: int
+
+    # Computed score (lower = better for spread, higher = better for leverage)
+    score: float
+
+
+def scan_instruments(
+    query: str,
+    direction: str = "",
+    instrument_type: str | None = None,
+    sort_by: str = "spread",
+    limit: int = 10,
+    max_search: int = 30,
+    min_leverage: float = 0,
+    max_spread_pct: float = 100,
+    workers: int = 6,
+) -> list[ScannedInstrument]:
+    """Search Avanza and fetch details for the best instruments.
+
+    Args:
+        query: Search keyword (e.g. "OLJA", "SILVER", "GULD", "TSMC").
+        direction: "BULL" or "BEAR" (filters results by name). Empty = both.
+        instrument_type: "certificate", "warrant", or None for both.
+        sort_by: Ranking criterion — "spread", "leverage", "price", "barrier_distance".
+        limit: Max results to return (after filtering and ranking).
+        max_search: How many search results to fetch before filtering.
+        min_leverage: Minimum leverage to include.
+        max_spread_pct: Maximum spread % to include (filters illiquid instruments).
+        workers: Thread pool size for parallel detail fetching.
+
+    Returns:
+        List of ScannedInstrument, sorted by the chosen criterion.
+    """
+    search_fn, instrument_fn, marketdata_fn, thread_safe = _get_api()
+
+    # --- Step 1: Search ---
+    search_query = f"{direction} {query}".strip() if direction else query
+    types_to_search = []
+    if instrument_type:
+        types_to_search.append(instrument_type)
+    else:
+        types_to_search.extend(["certificate", "warrant"])
+
+    all_hits: list[dict] = []
+    for itype in types_to_search:
+        try:
+            raw = search_fn(itype, search_query, max_search)
+            hits = raw.get("hits", raw) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
+            all_hits.extend(hits)
+        except Exception as e:
+            logger.warning("Search failed for type=%s query=%r: %s", itype, search_query, e)
+
+    if not all_hits:
+        logger.info("No search results for query=%r direction=%s", query, direction)
+        return []
+
+    # Filter by direction if specified
+    dir_upper = direction.upper()
+    if dir_upper:
+        all_hits = [h for h in all_hits if dir_upper in (h.get("title", "") or "").upper()]
+
+    # Filter tradeable only
+    all_hits = [h for h in all_hits if h.get("tradeable", h.get("tradable", True))]
+
+    # Deduplicate by orderbook ID
+    seen = set()
+    unique_hits = []
+    for h in all_hits:
+        ob_id = str(h.get("orderBookId", h.get("id", "")))
+        if ob_id and ob_id not in seen:
+            seen.add(ob_id)
+            unique_hits.append(h)
+    all_hits = unique_hits[:max_search]
+
+    logger.info("Scanner: %d candidates after search+filter for %r %s", len(all_hits), query, direction)
+
+    # --- Step 2: Fetch details in parallel ---
+    def fetch_detail(hit: dict) -> ScannedInstrument | None:
+        ob_id = str(hit.get("orderBookId", hit.get("id", "")))
+        name = hit.get("title", hit.get("name", ""))
+        itype = hit.get("type", hit.get("instrumentType", ""))
+
+        # Determine API type for market-guide
+        api_type = "certificate"
+        type_lower = itype.lower() if itype else ""
+        if "warrant" in type_lower or "mini" in name.upper():
+            api_type = "warrant"
+        elif "stock" in type_lower:
+            api_type = "stock"
+
+        try:
+            # Fetch instrument details (leverage, barrier, underlying)
+            info = instrument_fn(api_type, ob_id)
+            if not info or not isinstance(info, dict):
+                return None
+
+            # Extract quote
+            quote = info.get("quote", {})
+            bid = _val(quote.get("buy"))
+            ask = _val(quote.get("sell"))
+            last = _val(quote.get("last"))
+
+            # Compute spread
+            spread_pct = None
+            if bid and ask and bid > 0:
+                spread_pct = round((ask - bid) / bid * 100, 3)
+
+            # Extract leverage and barrier
+            ki = info.get("keyIndicators", {})
+            leverage = _val(ki.get("leverage"))
+            barrier = _val(ki.get("barrierLevel"))
+
+            # Barrier distance
+            barrier_dist_pct = None
+            if barrier and last and last > 0:
+                barrier_dist_pct = round(abs(last - barrier) / last * 100, 2)
+
+            # Underlying
+            underlying = info.get("underlying", {})
+            underlying_name = underlying.get("name", "")
+            underlying_quote = underlying.get("quote", {})
+            underlying_price = _val(underlying_quote.get("last"))
+
+            # Volume/turnover
+            volume = _val(quote.get("totalVolumeTraded"), 0) or 0
+            turnover = _val(quote.get("totalValueTraded"), 0) or 0
+
+            # Detect direction from name
+            name_upper = name.upper()
+            detected_dir = ""
+            for d in ("BULL", "BEAR", "MINI L", "MINI S"):
+                if d in name_upper:
+                    detected_dir = "BULL" if d in ("BULL", "MINI L") else "BEAR"
+                    break
+
+            # Also try market data for order depth (fast call)
+            bid_vol = 0
+            ask_vol = 0
+            mm = False
+            with suppress(Exception):
+                md = marketdata_fn(ob_id)
+                if isinstance(md, dict):
+                    od = md.get("orderDepth", md.get("orderDepthLevels", {}))
+                    levels = od.get("levels", od) if isinstance(od, dict) else od
+                    if isinstance(levels, list) and levels:
+                        first = levels[0]
+                        bid_side = first.get("buySide", first.get("buy", {}))
+                        ask_side = first.get("sellSide", first.get("sell", {}))
+                        bid_vol = int(bid_side.get("volume", 0))
+                        ask_vol = int(ask_side.get("volume", 0))
+                    mm = md.get("marketMakerExpected", False)
+
+            return ScannedInstrument(
+                orderbook_id=ob_id,
+                name=name,
+                instrument_type=itype,
+                direction=detected_dir,
+                bid=bid,
+                ask=ask,
+                last=last,
+                spread_pct=spread_pct,
+                leverage=leverage,
+                barrier=barrier,
+                barrier_distance_pct=barrier_dist_pct,
+                underlying_name=underlying_name,
+                underlying_price=underlying_price,
+                volume_today=int(volume),
+                turnover_today=float(turnover),
+                market_maker=mm,
+                bid_volume=bid_vol,
+                ask_volume=ask_vol,
+                score=0.0,
+            )
+        except Exception as e:
+            logger.debug("Detail fetch failed for %s (%s): %s", ob_id, name, e)
+            return None
+
+    results: list[ScannedInstrument] = []
+    t0 = time.perf_counter()
+    if thread_safe and workers > 1:
+        # TOTP: parallel fetch via thread pool
+        with ThreadPoolExecutor(max_workers=workers) as pool:
+            futures = {pool.submit(fetch_detail, h): h for h in all_hits}
+            for future in as_completed(futures):
+                result = future.result()
+                if result is not None:
+                    results.append(result)
+    else:
+        # BankID/Playwright: sequential (not thread-safe)
+        for h in all_hits:
+            result = fetch_detail(h)
+            if result is not None:
+                results.append(result)
+    dt = (time.perf_counter() - t0) * 1000
+    logger.info("Scanner: fetched %d instrument details in %.0fms (%s)",
+                len(results), dt, "parallel" if thread_safe else "sequential")
+
+    # --- Step 3: Filter ---
+    if min_leverage > 0:
+        results = [r for r in results if r.leverage and r.leverage >= min_leverage]
+    if max_spread_pct < 100:
+        results = [r for r in results if r.spread_pct is not None and r.spread_pct <= max_spread_pct]
+
+    # Filter out instruments with no bid/ask (not tradeable right now)
+    results = [r for r in results if r.bid is not None and r.ask is not None]
+
+    # --- Step 4: Score and sort ---
+    for r in results:
+        if sort_by == "spread":
+            r.score = r.spread_pct if r.spread_pct is not None else 999
+        elif sort_by == "leverage":
+            r.score = -(r.leverage or 0)  # negative so higher leverage sorts first
+        elif sort_by == "price":
+            r.score = r.last or 999
+        elif sort_by == "barrier_distance":
+            r.score = -(r.barrier_distance_pct or 0)  # negative = larger distance first
+        else:
+            r.score = r.spread_pct if r.spread_pct is not None else 999
+
+    results.sort(key=lambda r: r.score)
+    return results[:limit]
+
+
+def format_scan_results(results: list[ScannedInstrument]) -> str:
+    """Format scan results as a readable table string."""
+    if not results:
+        return "No instruments found."
+
+    lines = []
+    lines.append(f"{'Name':45s} {'ID':>8s} {'Lev':>5s} {'Bid':>8s} {'Ask':>8s} "
+                 f"{'Spread':>7s} {'Barrier':>8s} {'Dist%':>6s} {'Vol':>6s} {'MM':>3s}")
+    lines.append("-" * 115)
+
+    for r in results:
+        lev = f"{r.leverage:.1f}x" if r.leverage else "  -  "
+        bid = f"{r.bid:.2f}" if r.bid else "   -   "
+        ask = f"{r.ask:.2f}" if r.ask else "   -   "
+        spread = f"{r.spread_pct:.2f}%" if r.spread_pct is not None else "  -  "
+        barrier = f"{r.barrier:.1f}" if r.barrier else "   -   "
+        dist = f"{r.barrier_distance_pct:.1f}%" if r.barrier_distance_pct is not None else "  -  "
+        vol = f"{r.volume_today:,}" if r.volume_today else "  0"
+        mm = "Yes" if r.market_maker else " No"
+
+        lines.append(f"{r.name[:45]:45s} {r.orderbook_id:>8s} {lev:>5s} {bid:>8s} {ask:>8s} "
+                     f"{spread:>7s} {barrier:>8s} {dist:>6s} {vol:>6s} {mm:>3s}")
+
+    return "\n".join(lines)
diff --git a/portfolio/avanza/search.py b/portfolio/avanza/search.py
new file mode 100644
index 00000000..85fb9ecc
--- /dev/null
+++ b/portfolio/avanza/search.py
@@ -0,0 +1,83 @@
+"""Instrument search — find stocks, certificates, warrants, etc.
+
+Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
+for instrument discovery.
+"""
+
+from __future__ import annotations
+
+import logging
+from typing import Any
+
+from avanza.constants import InstrumentType
+
+from portfolio.avanza.client import AvanzaClient
+from portfolio.avanza.types import SearchHit
+
+logger = logging.getLogger("portfolio.avanza.search")
+
+
+# ---------------------------------------------------------------------------
+# Public API
+# ---------------------------------------------------------------------------
+
+
+def search(
+    query: str,
+    limit: int = 10,
+    instrument_type: str | None = None,
+) -> list[SearchHit]:
+    """Search for instruments on Avanza.
+
+    Args:
+        query: Search string (ISIN, ticker, name fragment, etc.).
+        limit: Maximum number of results.
+        instrument_type: Optional filter (e.g. ``"certificate"``,
+            ``"stock"``, ``"warrant"``).  When *None*, all types are
+            searched.
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.SearchHit`.
+    """
+    client = AvanzaClient.get_instance()
+
+    inst_type = (
+        InstrumentType(instrument_type) if instrument_type else InstrumentType.ANY
+    )
+
+    raw: Any = client.avanza.search_for_instrument(inst_type, query, limit)
+    logger.debug(
+        "search query=%r type=%s limit=%d hits=%d",
+        query,
+        inst_type.name,
+        limit,
+        len(raw) if isinstance(raw, list) else 0,
+    )
+
+    hits: list[dict[str, Any]]
+    if isinstance(raw, list):
+        hits = raw
+    elif isinstance(raw, dict):
+        hits = raw.get("hits", raw.get("results", []))
+    else:
+        hits = []
+
+    return [SearchHit.from_api(h) for h in hits]
+
+
+def find_warrants(query: str = "", limit: int = 20) -> list[SearchHit]:
+    """Search specifically for warrants.
+
+    Convenience wrapper around :func:`search` with
+    ``instrument_type="warrant"``.
+    """
+    return search(query=query, limit=limit, instrument_type="warrant")
+
+
+def find_certificates(query: str = "", limit: int = 20) -> list[SearchHit]:
+    """Search specifically for certificates.
+
+    Convenience wrapper around :func:`search` with
+    ``instrument_type="certificate"``.
+    """
+    return search(query=query, limit=limit, instrument_type="certificate")
diff --git a/portfolio/avanza/streaming.py b/portfolio/avanza/streaming.py
new file mode 100644
index 00000000..39fd5628
--- /dev/null
+++ b/portfolio/avanza/streaming.py
@@ -0,0 +1,251 @@
+"""CometD/Bayeux WebSocket streaming client for Avanza push data.
+
+Connects to ``wss://www.avanza.se/_push/cometd`` and subscribes to
+real-time channels for quotes, order depths, trades, orders, and deals.
+Runs a background daemon thread with automatic reconnection.
+
+Usage::
+
+    stream = AvanzaStream(push_subscription_id="abc123")
+    stream.on_quote("856394", lambda msg: print(msg))
+    stream.start()
+    # ... later ...
+    stream.stop()
+"""
+
+from __future__ import annotations
+
+import contextlib
+import json
+import logging
+import threading
+import time
+from collections.abc import Callable
+from typing import Any
+
+import websocket
+
+logger = logging.getLogger("portfolio.avanza.streaming")
+
+WS_URL = "wss://www.avanza.se/_push/cometd"
+
+# Reconnect backoff
+_MIN_BACKOFF = 1.0
+_MAX_BACKOFF = 60.0
+_BACKOFF_FACTOR = 2.0
+
+# CometD heartbeat interval (seconds)
+_HEARTBEAT_INTERVAL = 30.0
+
+
+class AvanzaStream:
+    """CometD/Bayeux WebSocket client for Avanza push data.
+
+    Register callbacks with :meth:`on_quote`, :meth:`on_order_depth`, etc.
+    before calling :meth:`start`.  The read loop runs in a daemon thread
+    and dispatches messages to registered callbacks by channel.
+    """
+
+    def __init__(self, push_subscription_id: str) -> None:
+        self._push_sub_id = push_subscription_id
+        self._callbacks: dict[str, list[Callable[[dict], None]]] = {}
+        self._client_id: str | None = None
+        self._ws: websocket.WebSocket | None = None
+        self._thread: threading.Thread | None = None
+        self._stop_event = threading.Event()
+        self._backoff = _MIN_BACKOFF
+
+    # ------------------------------------------------------------------
+    # Public registration (before start)
+    # ------------------------------------------------------------------
+
+    def on_quote(self, ob_id: str, callback: Callable[[dict], None]) -> None:
+        """Register a callback for quote updates on *ob_id*."""
+        channel = f"/quotes/{ob_id}"
+        self._callbacks.setdefault(channel, []).append(callback)
+
+    def on_order_depth(self, ob_id: str, callback: Callable[[dict], None]) -> None:
+        """Register a callback for order depth updates on *ob_id*."""
+        channel = f"/orderdepths/{ob_id}"
+        self._callbacks.setdefault(channel, []).append(callback)
+
+    def on_trades(self, ob_id: str, callback: Callable[[dict], None]) -> None:
+        """Register a callback for trade updates on *ob_id*."""
+        channel = f"/trades/{ob_id}"
+        self._callbacks.setdefault(channel, []).append(callback)
+
+    def on_orders(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
+        """Register a callback for order updates on the given accounts."""
+        channel = "/orders/_" + ",".join(account_ids)
+        self._callbacks.setdefault(channel, []).append(callback)
+
+    def on_deals(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
+        """Register a callback for deal updates on the given accounts."""
+        channel = "/deals/_" + ",".join(account_ids)
+        self._callbacks.setdefault(channel, []).append(callback)
+
+    # ------------------------------------------------------------------
+    # Lifecycle
+    # ------------------------------------------------------------------
+
+    def start(self) -> None:
+        """Start the background daemon thread."""
+        if self._thread is not None and self._thread.is_alive():
+            logger.warning("AvanzaStream already running")
+            return
+
+        self._stop_event.clear()
+        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="avanza-stream")
+        self._thread.start()
+        logger.info("AvanzaStream started (subscriptions=%d)", len(self._callbacks))
+
+    def stop(self) -> None:
+        """Close the WebSocket and join the background thread."""
+        self._stop_event.set()
+        if self._ws is not None:
+            with contextlib.suppress(Exception):
+                self._ws.close()
+        if self._thread is not None:
+            self._thread.join(timeout=5.0)
+            self._thread = None
+        self._client_id = None
+        logger.info("AvanzaStream stopped")
+
+    # ------------------------------------------------------------------
+    # Internal: run loop with reconnection
+    # ------------------------------------------------------------------
+
+    def _run_loop(self) -> None:
+        """Connect, handshake, subscribe, and read — with reconnection."""
+        while not self._stop_event.is_set():
+            try:
+                self._connect()
+                self._do_handshake()
+                for channel in self._callbacks:
+                    self._subscribe_channel(channel)
+                self._backoff = _MIN_BACKOFF  # Reset on successful connect
+                self._read_loop()
+            except Exception as exc:
+                if self._stop_event.is_set():
+                    break
+                logger.warning(
+                    "AvanzaStream connection error: %s — reconnecting in %.0fs",
+                    exc,
+                    self._backoff,
+                )
+                self._stop_event.wait(self._backoff)
+                self._backoff = min(self._backoff * _BACKOFF_FACTOR, _MAX_BACKOFF)
+            finally:
+                if self._ws is not None:
+                    with contextlib.suppress(Exception):
+                        self._ws.close()
+                    self._ws = None
+
+    def _connect(self) -> None:
+        """Open WebSocket connection to Avanza push endpoint."""
+        self._ws = websocket.create_connection(
+            WS_URL,
+            timeout=_HEARTBEAT_INTERVAL + 10,
+        )
+        logger.debug("WebSocket connected to %s", WS_URL)
+
+    def _do_handshake(self) -> None:
+        """Perform CometD/Bayeux handshake and extract clientId."""
+        handshake_msg = [{
+            "channel": "/meta/handshake",
+            "ext": {"subscriptionId": self._push_sub_id},
+            "version": "1.0",
+            "supportedConnectionTypes": ["websocket"],
+        }]
+        self._ws.send(json.dumps(handshake_msg))  # type: ignore[union-attr]
+        response = self._ws.recv()  # type: ignore[union-attr]
+        msgs = json.loads(response)
+
+        if not isinstance(msgs, list) or len(msgs) == 0:
+            raise RuntimeError(f"Invalid handshake response: {response}")
+
+        handshake_resp = msgs[0]
+        if not handshake_resp.get("successful", False):
+            raise RuntimeError(f"Handshake failed: {handshake_resp}")
+
+        self._client_id = handshake_resp["clientId"]
+        logger.debug("Handshake successful, clientId=%s", self._client_id)
+
+        # Send initial connect message
+        connect_msg = [{
+            "channel": "/meta/connect",
+            "clientId": self._client_id,
+            "connectionType": "websocket",
+        }]
+        self._ws.send(json.dumps(connect_msg))  # type: ignore[union-attr]
+
+    def _subscribe_channel(self, channel: str) -> None:
+        """Subscribe to a single CometD channel."""
+        sub_msg = [{
+            "channel": "/meta/subscribe",
+            "subscription": channel,
+            "clientId": self._client_id,
+        }]
+        self._ws.send(json.dumps(sub_msg))  # type: ignore[union-attr]
+        logger.debug("Subscribed to %s", channel)
+
+    def _read_loop(self) -> None:
+        """Read messages from WebSocket, dispatch, and send heartbeats."""
+        last_heartbeat = time.monotonic()
+
+        while not self._stop_event.is_set():
+            # Send heartbeat if needed
+            now = time.monotonic()
+            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
+                heartbeat_msg = [{
+                    "channel": "/meta/connect",
+                    "clientId": self._client_id,
+                    "connectionType": "websocket",
+                }]
+                self._ws.send(json.dumps(heartbeat_msg))  # type: ignore[union-attr]
+                last_heartbeat = now
+
+            try:
+                raw = self._ws.recv()  # type: ignore[union-attr]
+            except websocket.WebSocketTimeoutException:
+                continue  # Timeout is expected, just loop and heartbeat
+            except websocket.WebSocketConnectionClosedException:
+                logger.info("WebSocket connection closed")
+                return  # Will reconnect in _run_loop
+
+            if not raw:
+                continue
+
+            try:
+                msgs = json.loads(raw)
+            except json.JSONDecodeError:
+                logger.warning("Failed to parse WebSocket message: %s", raw[:200])
+                continue
+
+            if not isinstance(msgs, list):
+                msgs = [msgs]
+
+            for msg in msgs:
+                self._dispatch_message(msg)
+
+    def _dispatch_message(self, msg: dict[str, Any]) -> None:
+        """Route a CometD message to registered callbacks by channel."""
+        channel = msg.get("channel", "")
+
+        # Ignore meta channels (handshake, connect, subscribe responses)
+        if channel.startswith("/meta/"):
+            return
+
+        callbacks = self._callbacks.get(channel, [])
+        data = msg.get("data", msg)
+
+        for cb in callbacks:
+            try:
+                cb(data)
+            except Exception as exc:
+                logger.error(
+                    "Callback error on channel %s: %s",
+                    channel,
+                    exc,
+                    exc_info=True,
+                )
diff --git a/portfolio/avanza/tick_rules.py b/portfolio/avanza/tick_rules.py
new file mode 100644
index 00000000..33318705
--- /dev/null
+++ b/portfolio/avanza/tick_rules.py
@@ -0,0 +1,135 @@
+"""Tick-size rules — price rounding for Avanza order books.
+
+Caches tick tables per orderbook ID so repeated rounding calls do not
+hit the API.  Uses integer arithmetic internally to avoid floating-point
+drift.
+"""
+
+from __future__ import annotations
+
+import logging
+import math
+from typing import Any
+
+from portfolio.avanza.client import AvanzaClient
+from portfolio.avanza.types import TickEntry
+
+logger = logging.getLogger("portfolio.avanza.tick_rules")
+
+# Module-level cache: ob_id -> list of TickEntry
+_cache: dict[str, list[TickEntry]] = {}
+
+
+# ---------------------------------------------------------------------------
+# Public API
+# ---------------------------------------------------------------------------
+
+
+def get_tick_rules(ob_id: str) -> list[TickEntry]:
+    """Fetch (and cache) the tick-size table for an orderbook.
+
+    Args:
+        ob_id: Avanza orderbook ID.
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.TickEntry` sorted by
+        ``min_price``.
+    """
+    if ob_id in _cache:
+        return _cache[ob_id]
+
+    client = AvanzaClient.get_instance()
+    raw: dict[str, Any] = client.get_order_book_raw(ob_id)
+
+    tick_list_raw: list[dict[str, Any]] = raw.get("tickSizeList", raw.get("tickSizes", []))
+    entries = [TickEntry.from_api(t) for t in tick_list_raw]
+    entries.sort(key=lambda e: e.min_price)
+
+    _cache[ob_id] = entries
+    logger.debug("get_tick_rules ob_id=%s entries=%d (cached)", ob_id, len(entries))
+    return entries
+
+
+def round_to_tick(price: float, ob_id: str, direction: str = "down") -> float:
+    """Round a price to the nearest valid tick.
+
+    Uses integer arithmetic (multiply -> floor/ceil -> divide) to avoid
+    floating-point drift.
+
+    Args:
+        price: The price to round.
+        ob_id: Avanza orderbook ID (needed to fetch the tick table).
+        direction: ``"down"`` (floor) or ``"up"`` (ceil).
+
+    Returns:
+        The rounded price.
+
+    Raises:
+        ValueError: If *direction* is not ``"down"`` or ``"up"``.
+        ValueError: If no tick rule matches *price*.
+    """
+    if direction not in ("down", "up"):
+        raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
+
+    entries = get_tick_rules(ob_id)
+    tick = _find_tick_for_price(price, entries)
+
+    if tick is None:
+        raise ValueError(f"No tick rule found for price {price} (ob_id={ob_id})")
+
+    # Integer arithmetic to avoid float drift:
+    # steps = price / tick  ->  round to int  ->  result = steps * tick
+    # We use a precision multiplier derived from the tick's decimal places.
+    precision = _decimal_places(tick)
+    multiplier = 10 ** precision
+
+    # Convert to integer domain
+    price_int = price * multiplier
+    tick_int = round(tick * multiplier)
+
+    if tick_int == 0:
+        return price  # degenerate tick; return unchanged
+
+    if direction == "down":
+        steps = math.floor(price_int / tick_int)
+    else:
+        steps = math.ceil(price_int / tick_int)
+
+    result = (steps * tick_int) / multiplier
+    return round(result, precision)
+
+
+def clear_cache() -> None:
+    """Clear the module-level tick-rule cache."""
+    _cache.clear()
+    logger.debug("tick_rules cache cleared")
+
+
+# ---------------------------------------------------------------------------
+# Internals
+# ---------------------------------------------------------------------------
+
+
+def _find_tick_for_price(price: float, entries: list[TickEntry]) -> float | None:
+    """Find the tick size applicable for *price*.
+
+    Returns ``None`` if no entry matches.
+    """
+    for entry in entries:
+        if entry.min_price <= price <= entry.max_price:
+            return entry.tick_size
+        # Handle unbounded upper range (max_price == 0 means infinity)
+        if entry.min_price <= price and entry.max_price == 0:
+            return entry.tick_size
+    # Fallback: if price exceeds all ranges, use the last entry
+    if entries:
+        return entries[-1].tick_size
+    return None
+
+
+def _decimal_places(value: float) -> int:
+    """Count the number of significant decimal places in *value*."""
+    s = f"{value:.10f}".rstrip("0")
+    if "." in s:
+        return len(s.split(".")[1])
+    return 0
diff --git a/portfolio/avanza/trading.py b/portfolio/avanza/trading.py
new file mode 100644
index 00000000..55196a21
--- /dev/null
+++ b/portfolio/avanza/trading.py
@@ -0,0 +1,364 @@
+"""Trading operations — orders, stop-losses, deals.
+
+Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
+for placing, modifying, and cancelling orders and stop-losses.
+"""
+
+from __future__ import annotations
+
+import logging
+from datetime import date, timedelta
+from typing import Any
+
+from avanza.constants import (
+    Condition,
+    OrderType,
+    StopLossPriceType,
+    StopLossTriggerType,
+)
+from avanza.entities import StopLossOrderEvent, StopLossTrigger
+
+from portfolio.avanza.client import AvanzaClient
+from portfolio.avanza.types import (
+    Deal,
+    Order,
+    OrderResult,
+    StopLoss,
+    StopLossResult,
+)
+
+logger = logging.getLogger("portfolio.avanza.trading")
+
+
+# ---------------------------------------------------------------------------
+# Orders
+# ---------------------------------------------------------------------------
+
+
+def place_order(
+    side: str,
+    ob_id: str,
+    price: float,
+    volume: int,
+    condition: str = "NORMAL",
+    valid_until: str | None = None,
+    account_id: str | None = None,
+) -> OrderResult:
+    """Place a BUY or SELL order.
+
+    Args:
+        side: ``"BUY"`` or ``"SELL"``.
+        ob_id: Avanza orderbook ID.
+        price: Limit price.
+        volume: Number of units.
+        condition: Order condition (``"NORMAL"``, ``"FILL_OR_KILL"``,
+            ``"FILL_AND_KILL"``).
+        valid_until: ISO date string (default: today).
+        account_id: Override default account.
+
+    Returns:
+        :class:`~portfolio.avanza.types.OrderResult`.
+
+    Raises:
+        ValueError: If volume < 1, price <= 0, or order total < 1000 SEK.
+    """
+    if volume < 1:
+        raise ValueError(f"volume must be >= 1, got {volume}")
+    if price <= 0:
+        raise ValueError(f"price must be > 0, got {price}")
+
+    # 2026-04-17: match portfolio/avanza_session.py:590 convention — orders
+    # below 1000 SEK pay the Avanza courtage minimum and are almost always
+    # a caller bug. Unified-package callers should hit the same guard as
+    # the legacy path.
+    order_total = round(volume * price, 2)
+    if order_total < 1000.0:
+        raise ValueError(
+            f"Order total {order_total:.2f} SEK below minimum 1000 SEK"
+        )
+
+    client = AvanzaClient.get_instance()
+    acct = account_id or client.account_id
+    valid = date.fromisoformat(valid_until) if valid_until else date.today()
+
+    raw: dict[str, Any] = client.avanza.place_order(
+        acct,
+        ob_id,
+        OrderType(side),
+        price,
+        valid,
+        volume,
+        condition=Condition(condition),
+    )
+
+    logger.info(
+        "place_order side=%s ob_id=%s price=%s vol=%d -> %s",
+        side,
+        ob_id,
+        price,
+        volume,
+        raw.get("orderRequestStatus"),
+    )
+    return OrderResult.from_api(raw)
+
+
+def modify_order(
+    order_id: str,
+    ob_id: str,
+    price: float,
+    volume: int,
+    condition: str = "NORMAL",
+    valid_until: str | None = None,
+    account_id: str | None = None,
+) -> OrderResult:
+    """Modify an existing order.
+
+    Args:
+        order_id: Existing order ID to modify.
+        ob_id: Avanza orderbook ID (unused by API but kept for consistency).
+        price: New limit price.
+        volume: New volume.
+        condition: Order condition (unused by edit_order API).
+        valid_until: ISO date string (default: today).
+        account_id: Override default account.
+
+    Returns:
+        :class:`~portfolio.avanza.types.OrderResult`.
+    """
+    client = AvanzaClient.get_instance()
+    acct = account_id or client.account_id
+    valid = date.fromisoformat(valid_until) if valid_until else date.today()
+
+    raw: dict[str, Any] = client.avanza.edit_order(
+        order_id,
+        acct,
+        price,
+        valid,
+        volume,
+    )
+
+    logger.info(
+        "modify_order order_id=%s price=%s vol=%d -> %s",
+        order_id,
+        price,
+        volume,
+        raw.get("orderRequestStatus"),
+    )
+    return OrderResult.from_api(raw)
+
+
+def cancel_order(
+    order_id: str,
+    account_id: str | None = None,
+) -> bool:
+    """Cancel an existing order.
+
+    Returns:
+        ``True`` if the cancellation was accepted.
+    """
+    client = AvanzaClient.get_instance()
+    acct = account_id or client.account_id
+    raw: dict[str, Any] = client.avanza.delete_order(acct, order_id)
+    status = str(raw.get("orderRequestStatus", "")).upper()
+    success = status == "SUCCESS"
+    logger.info("cancel_order order_id=%s -> %s", order_id, status)
+    return success
+
+
+def get_orders() -> list[Order]:
+    """Fetch all open/recent orders.
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.Order`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: Any = client.get_orders_raw()
+
+    orders_list: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        orders_list = raw.get("orders", [])
+    elif isinstance(raw, list):
+        orders_list = raw
+    else:
+        orders_list = []
+
+    return [Order.from_api(o) for o in orders_list]
+
+
+def get_deals() -> list[Deal]:
+    """Fetch recent deals (executions).
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.Deal`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: Any = client.get_deals_raw()
+
+    deals_list: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        deals_list = raw.get("deals", [])
+    elif isinstance(raw, list):
+        deals_list = raw
+    else:
+        deals_list = []
+
+    return [Deal.from_api(d) for d in deals_list]
+
+
+# ---------------------------------------------------------------------------
+# Stop-losses
+# ---------------------------------------------------------------------------
+
+
+def place_stop_loss(
+    ob_id: str,
+    trigger_price: float,
+    sell_price: float,
+    volume: int,
+    valid_days: int = 8,
+    trigger_type: str = "LESS_OR_EQUAL",
+    value_type: str = "MONETARY",
+    account_id: str | None = None,
+) -> StopLossResult:
+    """Place a stop-loss order.
+
+    Args:
+        ob_id: Avanza orderbook ID.
+        trigger_price: Price that triggers the stop.
+        sell_price: Limit price for the sell order when triggered.
+        volume: Number of units to sell.
+        valid_days: Days until the stop-loss expires (default 8).
+        trigger_type: Trigger direction (``"LESS_OR_EQUAL"``,
+            ``"FOLLOW_DOWNWARDS"``, etc.).
+        value_type: Price type (``"MONETARY"`` or ``"PERCENTAGE"``).
+        account_id: Override default account.
+
+    Returns:
+        :class:`~portfolio.avanza.types.StopLossResult`.
+    """
+    client = AvanzaClient.get_instance()
+    acct = account_id or client.account_id
+    valid_until = date.today() + timedelta(days=valid_days)
+
+    # 2026-04-17: warn (don't raise) on sub-1000 SEK stop legs. Metals-loop
+    # cascaded stops split a position into ≤3 legs; per-leg value may
+    # legitimately fall below the courtage threshold. Surfacing via log
+    # lets callers audit fee impact without breaking live cascading logic.
+    if value_type == "MONETARY" and sell_price > 0:
+        leg_total = round(volume * sell_price, 2)
+        if leg_total < 1000.0:
+            logger.warning(
+                "place_stop_loss leg %.2f SEK below 1000 SEK courtage threshold "
+                "(vol=%d sell=%.3f ob=%s)",
+                leg_total, volume, sell_price, ob_id,
+            )
+
+    trigger = StopLossTrigger(
+        type=StopLossTriggerType(trigger_type),
+        value=trigger_price,
+        valid_until=valid_until,
+        value_type=StopLossPriceType(value_type),
+    )
+
+    order_event = StopLossOrderEvent(
+        type=OrderType.SELL,
+        price=sell_price,
+        volume=volume,
+        valid_days=valid_days,
+        price_type=StopLossPriceType(value_type),
+        short_selling_allowed=False,
+    )
+
+    raw: dict[str, Any] = client.avanza.place_stop_loss_order(
+        "0",  # parent_stop_loss_id — "0" for new stop-loss
+        acct,
+        ob_id,
+        trigger,
+        order_event,
+    )
+
+    logger.info(
+        "place_stop_loss ob_id=%s trigger=%.4f sell=%.4f vol=%d -> %s",
+        ob_id,
+        trigger_price,
+        sell_price,
+        volume,
+        raw.get("status", raw.get("orderRequestStatus")),
+    )
+    return StopLossResult.from_api(raw)
+
+
+def place_trailing_stop(
+    ob_id: str,
+    trail_percent: float,
+    volume: int,
+    valid_days: int = 8,
+    account_id: str | None = None,
+) -> StopLossResult:
+    """Place a trailing stop-loss (follows price downwards by percentage).
+
+    Args:
+        ob_id: Avanza orderbook ID.
+        trail_percent: Trailing distance as percentage (e.g. ``5.0`` for 5%).
+        volume: Number of units to sell.
+        valid_days: Days until the stop-loss expires.
+        account_id: Override default account.
+
+    Returns:
+        :class:`~portfolio.avanza.types.StopLossResult`.
+    """
+    return place_stop_loss(
+        ob_id=ob_id,
+        trigger_price=trail_percent,
+        sell_price=0.0,  # Not applicable for trailing stops
+        volume=volume,
+        valid_days=valid_days,
+        trigger_type="FOLLOW_DOWNWARDS",
+        value_type="PERCENTAGE",
+        account_id=account_id,
+    )
+
+
+def get_stop_losses() -> list[StopLoss]:
+    """Fetch all active stop-losses.
+
+    Returns:
+        List of :class:`~portfolio.avanza.types.StopLoss`.
+    """
+    client = AvanzaClient.get_instance()
+    raw: Any = client.get_all_stop_losses_raw()
+
+    sl_list: list[dict[str, Any]]
+    if isinstance(raw, dict):
+        sl_list = raw.get("stopLosses", raw.get("stopLossOrders", []))
+    elif isinstance(raw, list):
+        sl_list = raw
+    else:
+        sl_list = []
+
+    return [StopLoss.from_api(sl) for sl in sl_list]
+
+
+def delete_stop_loss(
+    stop_id: str,
+    account_id: str | None = None,
+) -> bool:
+    """Delete a stop-loss order.  Idempotent — 404 is treated as success.
+
+    Returns:
+        ``True`` if the deletion succeeded (or the stop-loss was already gone).
+    """
+    client = AvanzaClient.get_instance()
+    acct = account_id or client.account_id
+    try:
+        client.avanza.delete_stop_loss_order(acct, stop_id)
+        logger.info("delete_stop_loss stop_id=%s -> OK", stop_id)
+        return True
+    except Exception as exc:
+        # 404 means already deleted — treat as success
+        exc_str = str(exc).lower()
+        if "404" in exc_str or "not found" in exc_str:
+            logger.info("delete_stop_loss stop_id=%s -> already gone (404)", stop_id)
+            return True
+        logger.error("delete_stop_loss stop_id=%s -> FAILED: %s", stop_id, exc)
+        return False
diff --git a/portfolio/avanza/types.py b/portfolio/avanza/types.py
new file mode 100644
index 00000000..521e015f
--- /dev/null
+++ b/portfolio/avanza/types.py
@@ -0,0 +1,530 @@
+"""Typed response dataclasses for Avanza API data.
+
+Avanza wraps many numeric values in ``{"value": X, "unit": "SEK", ...}``
+objects.  The ``_val`` helper unwraps these transparently so callers always
+get plain Python scalars.
+"""
+
+from __future__ import annotations
+
+from dataclasses import dataclass
+from datetime import UTC, datetime
+from typing import Any
+
+# ---------------------------------------------------------------------------
+# Helpers
+# ---------------------------------------------------------------------------
+
+def _val(obj: Any, default: Any = None) -> Any:
+    """Unwrap Avanza ``{value: X}`` wrappers, or return *obj* as-is.
+
+    Handles:
+      - ``{"value": 1.23, "unit": "SEK", ...}`` -> ``1.23``
+      - plain scalars -> passed through
+      - ``None`` / missing -> *default*
+    """
+    if obj is None:
+        return default
+    if isinstance(obj, dict):
+        if "value" in obj:
+            return obj["value"]
+        return default
+    return obj
+
+
+def _ts(millis: Any) -> str:
+    """Convert a millisecond Unix timestamp to an ISO-8601 string."""
+    if millis is None:
+        return ""
+    if isinstance(millis, str):
+        return millis
+    try:
+        return datetime.fromtimestamp(int(millis) / 1000, tz=UTC).isoformat()
+    except (ValueError, TypeError, OSError):
+        return str(millis)
+
+
+# ---------------------------------------------------------------------------
+# Quote & Market Data
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class Quote:
+    """Parsed quote snapshot."""
+
+    bid: float
+    ask: float
+    last: float
+    spread: float
+    change_percent: float
+    high: float
+    low: float
+    volume: float
+    updated: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> Quote:
+        bid = _val(raw.get("buy"), _val(raw.get("bid"), 0.0))
+        ask = _val(raw.get("sell"), _val(raw.get("ask"), 0.0))
+        last = _val(raw.get("last"), _val(raw.get("latest"), 0.0))
+        spread = _val(raw.get("spread"))
+        if spread is None:
+            spread = round(ask - bid, 6) if (ask and bid) else 0.0
+        change_percent = _val(raw.get("changePercent"), _val(raw.get("change_percent"), 0.0))
+        high = _val(raw.get("highest"), _val(raw.get("high"), 0.0))
+        low = _val(raw.get("lowest"), _val(raw.get("low"), 0.0))
+        volume = _val(raw.get("totalVolumeTraded"), _val(raw.get("volume"), 0.0))
+        updated = _ts(raw.get("updated", ""))
+        return cls(
+            bid=float(bid),
+            ask=float(ask),
+            last=float(last),
+            spread=float(spread),
+            change_percent=float(change_percent),
+            high=float(high),
+            low=float(low),
+            volume=float(volume),
+            updated=updated,
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class OrderDepthLevel:
+    """One price level in the order book."""
+
+    price: float
+    volume: int
+
+    @classmethod
+    def from_api(cls, raw: dict) -> OrderDepthLevel:
+        return cls(
+            price=float(_val(raw.get("price"), 0.0)),
+            volume=int(_val(raw.get("volume"), 0)),
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class Trade:
+    """A single executed trade from the market-data feed."""
+
+    price: float
+    volume: int
+    buyer: str
+    seller: str
+    time: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> Trade:
+        return cls(
+            price=float(_val(raw.get("price"), 0.0)),
+            volume=int(_val(raw.get("volume"), 0)),
+            buyer=str(raw.get("buyer", "")),
+            seller=str(raw.get("seller", "")),
+            time=str(raw.get("dealTime", raw.get("time", ""))),
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class MarketData:
+    """Aggregated market data: quote + depth + recent trades."""
+
+    quote: Quote
+    bid_levels: tuple[OrderDepthLevel, ...]
+    ask_levels: tuple[OrderDepthLevel, ...]
+    recent_trades: tuple[Trade, ...]
+    market_maker_expected: bool
+
+    @classmethod
+    def from_api(cls, raw: dict) -> MarketData:
+        # Quote
+        quote_raw = raw.get("quote", {})
+        quote = Quote.from_api(quote_raw)
+
+        # Order depth
+        depth = raw.get("orderDepth", {})
+        levels = depth.get("levels", [])
+        bid_levels: list[OrderDepthLevel] = []
+        ask_levels: list[OrderDepthLevel] = []
+        for lvl in levels:
+            buy_side = lvl.get("buySide", {})
+            sell_side = lvl.get("sellSide", {})
+            if buy_side:
+                bid_levels.append(OrderDepthLevel.from_api(buy_side))
+            if sell_side:
+                ask_levels.append(OrderDepthLevel.from_api(sell_side))
+
+        # Trades
+        trades_raw = raw.get("trades", [])
+        trades = tuple(Trade.from_api(t) for t in trades_raw)
+
+        mm = depth.get("marketMakerExpected", False)
+
+        return cls(
+            quote=quote,
+            bid_levels=tuple(bid_levels),
+            ask_levels=tuple(ask_levels),
+            recent_trades=trades,
+            market_maker_expected=bool(mm),
+        )
+
+
+# ---------------------------------------------------------------------------
+# Order / StopLoss results
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class OrderResult:
+    """Result of placing or deleting an order."""
+
+    success: bool
+    order_id: str
+    status: str
+    message: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> OrderResult:
+        status = raw.get("orderRequestStatus", raw.get("status", ""))
+        success = str(status).upper() == "SUCCESS"
+        order_id = str(raw.get("orderId", raw.get("order_id", "")))
+        message = raw.get("message", raw.get("messages", ""))
+        if isinstance(message, list):
+            message = "; ".join(str(m) for m in message)
+        return cls(
+            success=success,
+            order_id=order_id,
+            status=str(status),
+            message=str(message),
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class StopLossResult:
+    """Result of placing or modifying a stop-loss."""
+
+    success: bool
+    stop_id: str
+    status: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> StopLossResult:
+        status = raw.get("status", raw.get("orderRequestStatus", ""))
+        success = str(status).upper() in ("SUCCESS", "OK", "ACTIVE")
+        stop_id = str(raw.get("stoplossOrderId", raw.get("stopLossId", raw.get("stop_id", raw.get("id", "")))))
+        return cls(
+            success=success,
+            stop_id=stop_id,
+            status=str(status),
+        )
+
+
+# ---------------------------------------------------------------------------
+# Account & Portfolio
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class Position:
+    """A single instrument position within an account."""
+
+    name: str
+    orderbook_id: str
+    instrument_type: str
+    volume: float
+    value: float
+    acquired_value: float
+    profit: float
+    profit_percent: float
+    last_price: float
+    change_percent: float
+    account_id: str
+    currency: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> Position:
+        instrument = raw.get("instrument", {})
+        orderbook = instrument.get("orderbook", {})
+        account = raw.get("account", {})
+        perf = raw.get("lastTradingDayPerformance", {})
+
+        # Quote values from the orderbook sub-object
+        ob_quote = orderbook.get("quote", {})
+        latest = _val(ob_quote.get("latest"), _val(ob_quote.get("last"), 0.0))
+        change_pct = _val(ob_quote.get("changePercent"), _val(ob_quote.get("change_percent"), 0.0))
+
+        return cls(
+            name=orderbook.get("name", instrument.get("name", "")),
+            orderbook_id=str(orderbook.get("id", raw.get("id", ""))),
+            instrument_type=instrument.get("type", orderbook.get("type", "")),
+            volume=float(_val(raw.get("volume"), 0.0)),
+            value=float(_val(raw.get("value"), 0.0)),
+            acquired_value=float(_val(raw.get("acquiredValue"), 0.0)),
+            profit=float(_val(perf.get("absolute"), 0.0)),
+            profit_percent=float(_val(perf.get("relative"), 0.0)),
+            last_price=float(latest),
+            change_percent=float(change_pct),
+            account_id=str(account.get("id", "")),
+            currency=instrument.get("currency", ""),
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class Order:
+    """An open or filled order."""
+
+    order_id: str
+    orderbook_id: str
+    side: str
+    price: float
+    volume: int
+    status: str
+    account_id: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> Order:
+        return cls(
+            order_id=str(raw.get("orderId", raw.get("id", ""))),
+            orderbook_id=str(raw.get("orderBookId", raw.get("orderbookId", raw.get("orderbook_id", "")))),
+            side=str(raw.get("orderType", raw.get("side", ""))),
+            price=float(_val(raw.get("price"), 0.0)),
+            volume=int(_val(raw.get("volume"), 0)),
+            status=str(raw.get("status", raw.get("statusDescription", ""))),
+            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class Deal:
+    """A completed deal (execution)."""
+
+    deal_id: str
+    orderbook_id: str
+    side: str
+    price: float
+    volume: int
+    time: str
+    account_id: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> Deal:
+        return cls(
+            deal_id=str(raw.get("dealId", raw.get("id", ""))),
+            orderbook_id=str(raw.get("orderBookId", raw.get("orderbookId", ""))),
+            side=str(raw.get("orderType", raw.get("side", ""))),
+            price=float(_val(raw.get("price"), 0.0)),
+            volume=int(_val(raw.get("volume"), 0)),
+            time=str(raw.get("dealTime", raw.get("time", ""))),
+            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class StopLoss:
+    """An active stop-loss order."""
+
+    stop_id: str
+    orderbook_id: str
+    trigger_price: float
+    trigger_type: str
+    sell_price: float
+    volume: int
+    status: str
+    account_id: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> StopLoss:
+        trigger = raw.get("trigger", {})
+        order_event = raw.get("orderEvent", raw.get("order", {}))
+        return cls(
+            stop_id=str(raw.get("id", raw.get("stopLossId", ""))),
+            orderbook_id=str((raw.get("orderbook") or {}).get("id", raw.get("orderBookId", raw.get("orderbookId", "")))),
+            trigger_price=float(_val(trigger.get("value"), _val(raw.get("triggerPrice"), 0.0))),
+            trigger_type=str(trigger.get("type", raw.get("triggerType", "LAST_PRICE"))),
+            sell_price=float(_val(order_event.get("price"), _val(raw.get("sellPrice"), 0.0))),
+            volume=int(_val(order_event.get("volume"), _val(raw.get("volume"), 0))),
+            status=str(raw.get("status", "")),
+            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
+        )
+
+
+# ---------------------------------------------------------------------------
+# Search
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class SearchHit:
+    """Instrument search result."""
+
+    orderbook_id: str
+    name: str
+    instrument_type: str
+    tradeable: bool
+    last_price: float
+    change_percent: float
+
+    @classmethod
+    def from_api(cls, raw: dict) -> SearchHit:
+        return cls(
+            orderbook_id=str(raw.get("id", raw.get("orderbookId", ""))),
+            name=str(raw.get("name", "")),
+            instrument_type=str(raw.get("instrumentType", raw.get("type", ""))),
+            tradeable=bool(raw.get("tradable", raw.get("tradeable", False))),
+            last_price=float(_val(raw.get("lastPrice"), _val(raw.get("last_price"), 0.0))),
+            change_percent=float(_val(raw.get("changePercent"), _val(raw.get("change_percent"), 0.0))),
+        )
+
+
+# ---------------------------------------------------------------------------
+# Tick Table
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class TickEntry:
+    """One row from the tick-size table."""
+
+    min_price: float
+    max_price: float
+    tick_size: float
+
+    @classmethod
+    def from_api(cls, raw: dict) -> TickEntry:
+        return cls(
+            min_price=float(raw.get("min", raw.get("minPrice", 0.0))),
+            max_price=float(raw.get("max", raw.get("maxPrice", 0.0))),
+            tick_size=float(raw.get("tick", raw.get("tickSize", raw.get("tick_size", 0.0)))),
+        )
+
+
+# ---------------------------------------------------------------------------
+# OHLC
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class OHLC:
+    """A single OHLCV candle."""
+
+    timestamp: str
+    open: float
+    high: float
+    low: float
+    close: float
+    volume: int
+
+    @classmethod
+    def from_api(cls, raw: dict) -> OHLC:
+        return cls(
+            timestamp=_ts(raw.get("timestamp")),
+            open=float(raw.get("open", 0.0)),
+            high=float(raw.get("high", 0.0)),
+            low=float(raw.get("low", 0.0)),
+            close=float(raw.get("close", 0.0)),
+            volume=int(raw.get("totalVolumeTraded", raw.get("volume", 0))),
+        )
+
+
+# ---------------------------------------------------------------------------
+# Account
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class AccountCash:
+    """Account-level cash info."""
+
+    buying_power: float
+    total_value: float
+    own_capital: float
+
+    @classmethod
+    def from_api(cls, raw: dict) -> AccountCash:
+        return cls(
+            buying_power=float(_val(raw.get("buyingPower"), 0.0)),
+            total_value=float(_val(raw.get("totalValue"), 0.0)),
+            own_capital=float(_val(raw.get("ownCapital"), _val(raw.get("buyingPowerWithoutCredit"), 0.0))),
+        )
+
+
+# ---------------------------------------------------------------------------
+# Transaction
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class Transaction:
+    """A historical account transaction."""
+
+    transaction_id: str
+    transaction_type: str
+    instrument_name: str
+    amount: float
+    price: float
+    volume: float
+    date: str
+    account_id: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> Transaction:
+        account = raw.get("account", {})
+        return cls(
+            transaction_id=str(raw.get("id", "")),
+            transaction_type=str(raw.get("type", "")),
+            instrument_name=str(raw.get("instrumentName", raw.get("description", ""))),
+            amount=float(_val(raw.get("amount"), 0.0)),
+            price=float(_val(raw.get("priceInTradedCurrency"), _val(raw.get("price"), 0.0))),
+            volume=float(_val(raw.get("volume"), 0.0)),
+            date=str(raw.get("date", raw.get("tradeDate", ""))),
+            account_id=str(account.get("id", raw.get("accountId", ""))),
+        )
+
+
+# ---------------------------------------------------------------------------
+# Instrument Info
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class InstrumentInfo:
+    """Core instrument metadata (works for certificates, warrants, stocks)."""
+
+    orderbook_id: str
+    name: str
+    instrument_type: str
+    currency: str
+    leverage: float
+    barrier: float
+    underlying_name: str
+    underlying_price: float
+
+    @classmethod
+    def from_api(cls, raw: dict) -> InstrumentInfo:
+        return cls(
+            orderbook_id=str(raw.get("id", raw.get("orderbookId", ""))),
+            name=str(raw.get("name", "")),
+            instrument_type=str(raw.get("instrumentType", raw.get("type", ""))),
+            currency=str(raw.get("currency", "")),
+            leverage=float(_val(raw.get("leverage"), 0.0)),
+            barrier=float(_val(raw.get("barrier"), _val(raw.get("barrierLevel"), 0.0))),
+            underlying_name=str(raw.get("underlyingName", raw.get("underlying", {}).get("name", ""))),
+            underlying_price=float(_val(
+                raw.get("underlyingPrice"),
+                _val(raw.get("underlying", {}).get("price"), 0.0),
+            )),
+        )
+
+
+# ---------------------------------------------------------------------------
+# News
+# ---------------------------------------------------------------------------
+
+@dataclass(frozen=True, slots=True)
+class NewsArticle:
+    """A single news article linked to an instrument."""
+
+    article_id: str
+    headline: str
+    date: str
+    source: str
+
+    @classmethod
+    def from_api(cls, raw: dict) -> NewsArticle:
+        return cls(
+            article_id=str(raw.get("id", raw.get("articleId", ""))),
+            headline=str(raw.get("headline", "")),
+            date=_ts(raw.get("timePublishedMillis", raw.get("date", raw.get("timePublished", "")))),
+            source=str(raw.get("newsSource", raw.get("source", ""))),
+        )
diff --git a/portfolio/avanza_client.py b/portfolio/avanza_client.py
new file mode 100644
index 00000000..bdd6cc9c
--- /dev/null
+++ b/portfolio/avanza_client.py
@@ -0,0 +1,397 @@
+"""Avanza API client for portfolio monitoring and trading.
+
+Supports two authentication methods:
+1. BankID session (preferred) — captured by scripts/avanza_login.py, stored in
+   data/avanza_session.json. No credentials needed, valid ~24h.
+2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
+   from config.json.
+
+The client transparently tries BankID session first, then falls back to TOTP.
+"""
+
+import logging
+from datetime import date
+from pathlib import Path
+from typing import Any
+
+from portfolio.avanza_order_lock import avanza_order_lock
+from portfolio.file_utils import load_json
+
+logger = logging.getLogger("portfolio.avanza_client")
+
+BASE_DIR = Path(__file__).resolve().parent.parent
+CONFIG_FILE = BASE_DIR / "config.json"
+
+# A-AV-2 (2026-04-11): Hardcoded account whitelist. The TOTP path scans for
+# any account whose accountType contains "ISK", which means a future Avanza
+# response containing a *new* ISK-shaped account (e.g. a child's ISK or a
+# corporate ISK) could be picked up as the trading account. Worse, the
+# pension account (2674244) was previously reachable through some code paths
+# without filtering. Mirror the ALLOWED_ACCOUNT_IDS pattern from
+# avanza_session.py: anything not in this set is rejected, period.
+ALLOWED_ACCOUNT_IDS: set[str] = {"1625505"}
+
+# Singleton client instance (avanza-api library)
+_client = None
+# Cached signal that a BankID Playwright session has already been verified.
+_session_client = None
+
+
+def _load_credentials() -> dict:
+    """Load Avanza credentials from config.json.
+
+    Returns:
+        dict with keys: username, password, totp_secret
+
+    Raises:
+        FileNotFoundError: if config.json does not exist
+        KeyError: if 'avanza' section is missing or credentials incomplete
+    """
+    config = load_json(CONFIG_FILE)
+    if config is None:
+        raise FileNotFoundError(f"Config file not found or unreadable: {CONFIG_FILE}")
+    if "avanza" not in config:
+        raise KeyError(
+            "Missing 'avanza' section in config.json. "
+            "Add: {\"avanza\": {\"username\": \"...\", \"password\": \"...\", \"totp_secret\": \"...\"}}"
+        )
+    creds = config["avanza"]
+    for key in ("username", "password", "totp_secret"):
+        if key not in creds or not creds[key]:
+            raise KeyError(f"Missing or empty 'avanza.{key}' in config.json")
+    return creds
+
+
+def _try_session_auth() -> bool:
+    """Return True when a BankID-backed Playwright session is available."""
+    global _session_client
+    if _session_client is True:
+        return True
+    try:
+        from portfolio.avanza_session import verify_session
+        if verify_session():
+            _session_client = True
+            logger.info("Using BankID session for Avanza API")
+            return True
+        logger.info("BankID session exists but verification failed")
+    except Exception as e:
+        logger.debug("BankID session not available: %s", e)
+    return False
+
+
+def get_client():
+    """Get or create a singleton Avanza client.
+
+    Tries BankID session first, then falls back to TOTP credentials.
+
+    Returns:
+        Authenticated Avanza client instance (avanza-api library)
+
+    Raises:
+        Exception: if neither auth method works
+    """
+    global _client
+    if _client is not None:
+        return _client
+    try:
+        from avanza import Avanza
+    except ImportError:
+        raise ImportError(
+            "avanza-api package not installed. Run: pip install avanza-api"
+        ) from None
+    creds = _load_credentials()
+    _client = Avanza({
+        "username": creds["username"],
+        "password": creds["password"],
+        "totpSecret": creds["totp_secret"],
+    })
+    return _client
+
+
+def reset_client() -> None:
+    """Reset the singleton TOTP client (useful for re-authentication)."""
+    global _client
+    _client = None
+
+
+def reset_session() -> None:
+    """Reset the cached BankID session verification flag."""
+    global _session_client
+    _session_client = None
+
+
+def find_instrument(query: str) -> list[dict]:
+    """Search for instruments by name or ticker.
+
+    Args:
+        query: Search string (e.g., 'Bitcoin', 'NVDA', 'Silver')
+
+    Returns:
+        List of matching instruments with id, name, and type
+    """
+    client = get_client()
+    results = client.search_for_stock(query)
+    return results
+
+
+def get_price(orderbook_id: str) -> dict[str, Any]:
+    """Get current price and info for an instrument.
+
+    Tries BankID session first, then falls back to TOTP client.
+
+    Args:
+        orderbook_id: Avanza orderbook ID (numeric string)
+
+    Returns:
+        Dict with price info including lastPrice, change, changePercent, etc.
+    """
+    # Try session-based auth first
+    if _try_session_auth():
+        try:
+            from portfolio.avanza_session import get_instrument_price
+            return get_instrument_price(orderbook_id)
+        except Exception as e:
+            logger.warning("Session-based price fetch failed, trying TOTP: %s", e)
+            reset_session()
+
+    client = get_client()
+    info = client.get_stock_info(orderbook_id)
+    return info
+
+
+def get_positions() -> list[dict]:
+    """Get all current positions from the Avanza account.
+
+    Tries BankID session first, then falls back to TOTP client.
+
+    Returns:
+        List of position dicts, each with name, value, profit, etc.
+        Returns empty list if no positions or on error.
+    """
+    # Try session-based auth first
+    if _try_session_auth():
+        try:
+            from portfolio.avanza_session import get_positions as session_get_positions
+            return session_get_positions()
+        except Exception as e:
+            logger.warning("Session-based positions fetch failed, trying TOTP: %s", e)
+            reset_session()
+
+    client = get_client()
+    overview = client.get_overview()
+    positions = []
+    # A-AV-2: only return positions from whitelisted accounts so the pension
+    # account (or any future-added account) never leaks into the trading view.
+    for account in overview.get("accounts", []):
+        if str(account.get("accountId", "")) not in ALLOWED_ACCOUNT_IDS:
+            continue
+        for pos in account.get("positions", []):
+            positions.append({
+                "account": account.get("name", ""),
+                "account_id": account.get("accountId", ""),
+                "name": pos.get("name", ""),
+                "ticker": pos.get("orderbookId", ""),
+                "volume": pos.get("volume", 0),
+                "value": pos.get("value", 0),
+                "profit": pos.get("profit", 0),
+                "profit_percent": pos.get("profitPercent", 0),
+                "currency": pos.get("currency", "SEK"),
+            })
+    return positions
+
+
+def get_portfolio_value() -> float:
+    """Get total portfolio value in SEK for whitelisted Avanza accounts only.
+
+    A-AV-2: Filters to ALLOWED_ACCOUNT_IDS so pension/other-account values
+    never inflate the "trading" portfolio value used for sizing.
+
+    Returns:
+        Total portfolio value in SEK across whitelisted accounts
+    """
+    client = get_client()
+    overview = client.get_overview()
+    total = 0.0
+    for account in overview.get("accounts", []):
+        if str(account.get("accountId", "")) not in ALLOWED_ACCOUNT_IDS:
+            continue
+        total += account.get("totalValue", 0)
+    return total
+
+
+def get_open_orders() -> list:
+    """Return open orders for the ISK account (read-only).
+
+    Uses the authenticated Avanza client; does not place or cancel orders.
+    """
+    client = get_client()
+    account_id = get_account_id()
+    try:
+        orders = client.get_orders(account_id)
+    except Exception as e:
+        logger.error("Failed to fetch open orders: %s", e)
+        raise
+    return orders or []
+
+
+# --- Account ID ---
+
+_account_id: str | None = None
+
+
+def get_account_id() -> str:
+    """Get the trading account ID from Avanza overview (cached).
+
+    Scans accounts of type ISK and returns the first one *that is also in
+    ALLOWED_ACCOUNT_IDS*. Any ISK account not in the whitelist is rejected.
+    This prevents accidental trades on a future-added child ISK, corporate
+    ISK, or any account Avanza re-orders into the response.
+
+    Returns:
+        Account ID string (guaranteed to be in ALLOWED_ACCOUNT_IDS)
+
+    Raises:
+        RuntimeError: if no whitelisted ISK account is found
+    """
+    global _account_id
+    if _account_id is not None:
+        return _account_id
+    client = get_client()
+    overview = client.get_overview()
+    seen_ids: list[str] = []
+    for account in overview.get("accounts", []):
+        atype = account.get("accountType", "")
+        if "ISK" not in atype.upper():
+            continue
+        candidate = str(account.get("accountId", ""))
+        seen_ids.append(candidate)
+        # A-AV-2: enforce whitelist BEFORE caching, so a rogue first-call
+        # cannot poison the singleton with a non-whitelisted account.
+        if candidate in ALLOWED_ACCOUNT_IDS:
+            _account_id = candidate
+            logger.info("Found whitelisted ISK account: %s", _account_id)
+            return _account_id
+    raise RuntimeError(
+        "No whitelisted ISK account found in Avanza overview. "
+        f"ISK account IDs seen: {seen_ids}. Whitelist: {sorted(ALLOWED_ACCOUNT_IDS)}. "
+        "If this is a legitimate new account, update ALLOWED_ACCOUNT_IDS in "
+        "portfolio/avanza_client.py — never trade on auto-discovered accounts."
+    )
+
+
+# --- Trading functions ---
+
+
+def place_buy_order(
+    orderbook_id: str,
+    price: float,
+    volume: int,
+    valid_until: date | None = None,
+) -> dict:
+    """Place a limit BUY order on Avanza.
+
+    Args:
+        orderbook_id: Avanza orderbook ID for the instrument
+        price: Limit price in SEK
+        volume: Number of shares (must be int >= 1)
+        valid_until: Order expiry date. Defaults to today (day order).
+
+    Returns:
+        Dict with orderId, orderRequestStatus, message
+    """
+    from avanza.constants import OrderType
+    return _place_order(orderbook_id, OrderType.BUY, price, volume, valid_until)
+
+
+def place_sell_order(
+    orderbook_id: str,
+    price: float,
+    volume: int,
+    valid_until: date | None = None,
+) -> dict:
+    """Place a limit SELL order on Avanza.
+
+    Args:
+        orderbook_id: Avanza orderbook ID for the instrument
+        price: Limit price in SEK
+        volume: Number of shares (must be int >= 1)
+        valid_until: Order expiry date. Defaults to today (day order).
+
+    Returns:
+        Dict with orderId, orderRequestStatus, message
+    """
+    from avanza.constants import OrderType
+    return _place_order(orderbook_id, OrderType.SELL, price, volume, valid_until)
+
+
+def _place_order(orderbook_id, order_type, price, volume, valid_until):
+    """Internal: place an order via the Avanza API.
+
+    P0-4 (2026-05-02): Wrapped in ``avanza_order_lock`` so the TOTP path
+    cannot race against the page-session paths in
+    ``data/metals_avanza_helpers.place_order``,
+    ``portfolio/avanza_session.place_order``, or
+    ``portfolio/avanza_control.place_order`` (all of which already lock).
+    Without the lock, two callers observing the same ``buying_power``
+    snapshot could both fire orders and overdraw the ISK.
+
+    The op label is distinct from page-session labels
+    (``place_order_totp/...``) so the rate-limit diagnostic ("which loop
+    hit the busy lock") still works. ``OrderLockBusyError`` is allowed
+    to propagate so callers can decide whether to retry next cycle —
+    matches the existing semantics in
+    ``data/metals_avanza_helpers.place_order``.
+    """
+    if volume < 1:
+        raise ValueError(f"Volume must be >= 1, got {volume}")
+    if price <= 0:
+        raise ValueError(f"Price must be > 0, got {price}")
+
+    client = get_client()
+    account_id = get_account_id()
+    expiry = valid_until or date.today()
+
+    logger.info(
+        "Placing %s order: orderbook=%s price=%.2f vol=%d until=%s account=%s",
+        order_type.value, orderbook_id, price, volume, expiry, account_id,
+    )
+    with avanza_order_lock(op=f"place_order_totp/{order_type.value}/{orderbook_id}"):
+        result = client.place_order(
+            account_id=account_id,
+            order_book_id=orderbook_id,
+            order_type=order_type,
+            price=price,
+            valid_until=expiry,
+            volume=volume,
+        )
+    logger.info("Order result: %s", result)
+    return result
+
+
+def get_order_status(order_id: str) -> dict:
+    """Check the status of an order by ID.
+
+    Returns:
+        Order dict with state, price, volume, etc.
+    """
+    client = get_client()
+    account_id = get_account_id()
+    return client.get_order(account_id, order_id)
+
+
+def delete_order(order_id: str) -> dict:
+    """Cancel a pending order.
+
+    P0-4 (2026-05-02): Wrapped in ``avanza_order_lock`` for the same
+    reason as ``_place_order`` — cancelling an order is mutating and must
+    serialize against new orders / stop-loss adjustments to avoid races
+    where the position view is read between cancel-and-place.
+
+    Returns:
+        Dict with orderId, orderRequestStatus, messages
+    """
+    client = get_client()
+    account_id = get_account_id()
+    logger.info("Deleting order %s on account %s", order_id, account_id)
+    with avanza_order_lock(op=f"delete_order_totp/{order_id}"):
+        return client.delete_order(account_id, order_id)
diff --git a/portfolio/avanza_control.py b/portfolio/avanza_control.py
new file mode 100644
index 00000000..51f61f19
--- /dev/null
+++ b/portfolio/avanza_control.py
@@ -0,0 +1,439 @@
+"""Canonical Avanza control facade for reads, quotes, and browser-session trades.
+
+Use this module as the shared import path for Avanza operations in strategy code.
+It keeps the currently working Playwright-page execution path for metals/gold
+while exposing the broader account/session helpers from ``portfolio.avanza_*``.
+"""
+
+from __future__ import annotations
+
+import json
+import logging
+
+from portfolio.avanza_order_lock import avanza_order_lock
+
+logger = logging.getLogger("portfolio.avanza_control")
+
+from data.metals_avanza_helpers import (
+    check_session_alive,
+    get_csrf,
+)
+from data.metals_avanza_helpers import (
+    fetch_account_cash as _fetch_account_cash,
+)
+from data.metals_avanza_helpers import (
+    fetch_positions as _fetch_page_positions,
+)
+from data.metals_avanza_helpers import (
+    fetch_price as _fetch_page_price,
+)
+from data.metals_avanza_helpers import (
+    place_order as _place_page_order,
+)
+from data.metals_avanza_helpers import (
+    place_stop_loss as _place_page_stop_loss,
+)
+from portfolio.avanza_client import (
+    delete_order,
+    find_instrument,
+    get_account_id,
+    get_open_orders,
+    get_portfolio_value,
+    get_positions,
+    place_buy_order,
+    place_sell_order,
+)
+from portfolio.avanza_client import (
+    get_price as get_price_info,
+)
+
+_TYPE_ALIASES = {
+    "cert": "certificate",
+    "certifikat": "certificate",
+    "certificate": "certificate",
+    "warrant": "warrant",
+    "mini": "warrant",
+    "mini-future": "warrant",
+    "mini_future": "warrant",
+    "stock": "stock",
+    "share": "stock",
+    "fund": "fund",
+    "etf": "exchange_traded_fund",
+    "exchange_traded_fund": "exchange_traded_fund",
+}
+
+_PRICE_FALLBACK_TYPES = (
+    "certificate",
+    "warrant",
+    "stock",
+    "exchange_traded_fund",
+    "fund",
+)
+
+
+def normalize_api_type(api_type: str | None, default: str = "certificate") -> str:
+    """Normalize Avanza instrument type names for market-guide lookups."""
+    normalized = (api_type or "").strip().lower()
+    if not normalized:
+        return default
+    return _TYPE_ALIASES.get(normalized, normalized)
+
+
+def fetch_price(page, orderbook_id: str, api_type: str = "certificate"):
+    """Fetch a quote from the market-guide API using an authenticated page."""
+    return _fetch_page_price(page, orderbook_id, normalize_api_type(api_type))
+
+
+def fetch_price_with_fallback(page, orderbook_id: str, api_type: str | None = None):
+    """Try the preferred market-guide type and then the common fallback types."""
+    if not orderbook_id:
+        return None
+
+    candidates: list[str] = []
+    preferred = normalize_api_type(api_type) if api_type else ""
+    if preferred:
+        candidates.append(preferred)
+    for fallback in _PRICE_FALLBACK_TYPES:
+        if fallback not in candidates:
+            candidates.append(fallback)
+
+    for candidate in candidates:
+        data = fetch_price(page, orderbook_id, candidate)
+        if not data:
+            continue
+        if data.get("bid") is None and data.get("ask") is None and data.get("last") is None:
+            continue
+        payload = dict(data)
+        payload["api_type"] = candidate
+        return payload
+    return None
+
+
+def fetch_account_cash(page, account_id: str | None = None):
+    """Fetch buying power for an account via the authenticated browser session."""
+    resolved_account_id = str(account_id or get_account_id())
+    return _fetch_account_cash(page, resolved_account_id)
+
+
+def fetch_page_positions(page, account_id: str | None = None):
+    """Fetch current positions keyed by orderbook id via the page session.
+
+    Returns dict[ob_id -> {name, units, value, avg_price, api_type}] on
+    success, or None on transient failure. An empty dict `{}` is a valid
+    response meaning the account is flat — callers should distinguish it
+    from None.
+    """
+    resolved_account_id = str(account_id or get_account_id())
+    return _fetch_page_positions(page, resolved_account_id)
+
+
+def place_order(page, account_id: str | None, ob_id: str, side: str, price: float, volume: int):
+    """Place a BUY/SELL order via the authenticated browser session."""
+    resolved_account_id = str(account_id or get_account_id())
+    normalized_side = (side or "").strip().upper()
+    return _place_page_order(page, resolved_account_id, ob_id, normalized_side, price, volume)
+
+
+def place_stop_loss(
+    page,
+    account_id: str | None,
+    ob_id: str,
+    trigger_price: float,
+    sell_price: float,
+    volume: int,
+    valid_days: int = 8,
+):
+    """Place a hardware stop-loss order via the authenticated browser session."""
+    resolved_account_id = str(account_id or get_account_id())
+    return _place_page_stop_loss(
+        page,
+        resolved_account_id,
+        ob_id,
+        trigger_price,
+        sell_price,
+        volume,
+        valid_days=valid_days,
+    )
+
+
+def delete_order_live(page, account_id: str | None, order_id: str):
+    """Cancel an open order via the authenticated page session.
+
+    IMPORTANT: Uses POST to /_api/trading-critical/rest/order/delete with
+    JSON body {accountId, orderId}. The DELETE HTTP verb to
+    /_api/trading-critical/rest/order/{accountId}/{orderId} returns 404
+    (Avanza API change discovered 2026-03-24).
+    """
+    csrf = get_csrf(page)
+    if not csrf:
+        return False, {"error": "no CSRF token"}
+
+    resolved_account_id = str(account_id or get_account_id())
+    try:
+        # 2026-04-13: cross-process order lock (see metals_avanza_helpers.place_order).
+        with avanza_order_lock(op=f"delete_order_live/{order_id}"):
+            result = page.evaluate(
+                """async (args) => {
+                    const [accountId, orderId, token] = args;
+                    const resp = await fetch(
+                        'https://www.avanza.se/_api/trading-critical/rest/order/delete',
+                        {
+                            method: 'POST',
+                            headers: {
+                                'Content-Type': 'application/json',
+                                'X-SecurityToken': token,
+                            },
+                            credentials: 'include',
+                            body: JSON.stringify({accountId: accountId, orderId: orderId}),
+                        }
+                    );
+                    return {status: resp.status, body: await resp.text()};
+                }""",
+                [resolved_account_id, order_id, csrf],
+            )
+        http_status = int(result.get("status") or 0)
+        body_text = result.get("body", "")
+        parsed = {}
+        try:
+            if body_text:
+                parsed = json.loads(body_text)
+        except (TypeError, json.JSONDecodeError):
+            parsed = {}
+        success = parsed.get("orderRequestStatus") == "SUCCESS"
+        return success, {
+            "http_status": http_status,
+            "parsed": parsed,
+            "body": body_text,
+        }
+    except Exception as exc:
+        logger.error("Delete order failed for order %s: %s", order_id, exc, exc_info=True)
+        return False, {"error": str(exc)}
+
+
+def delete_stop_loss(page, account_id: str | None, stop_id: str):
+    """Delete an existing Avanza stop-loss order via the authenticated page."""
+    csrf = get_csrf(page)
+    if not csrf:
+        return False, {"error": "no CSRF token"}
+
+    resolved_account_id = str(account_id or get_account_id())
+    try:
+        # 2026-04-13: cross-process order lock. SL delete is mutating.
+        with avanza_order_lock(op=f"delete_stop_loss/{stop_id}"):
+            result = page.evaluate(
+                """async (args) => {
+                    const [accountId, stopId, token] = args;
+                    const resp = await fetch(
+                        'https://www.avanza.se/_api/trading/stoploss/' + accountId + '/' + stopId,
+                        {
+                            method: 'DELETE',
+                            headers: {'X-SecurityToken': token},
+                            credentials: 'include',
+                        }
+                    );
+                    return {status: resp.status, body: await resp.text()};
+                }""",
+                [resolved_account_id, stop_id, csrf],
+            )
+        http_status = int(result.get("status") or 0)
+        # 2xx = deleted successfully.  404 = stop already gone (triggered/expired/cancelled).
+        # Both mean the stop no longer exists, which is the goal of a cancel.
+        success = (200 <= http_status < 300) or http_status == 404
+        body_text = result.get("body", "")
+        parsed = {}
+        try:
+            if body_text:
+                parsed = json.loads(body_text)
+        except (TypeError, json.JSONDecodeError):
+            parsed = {}
+        return success, {
+            "http_status": http_status,
+            "parsed": parsed,
+            "body": body_text,
+        }
+    except Exception as exc:
+        logger.error("Delete stop-loss failed for stop %s: %s", stop_id, exc, exc_info=True)
+        return False, {"error": str(exc)}
+
+
+
+# --- Page-free API (uses BankID session, no Playwright page needed) ---
+
+from portfolio.avanza_session import (
+    api_delete as _api_delete,
+)
+from portfolio.avanza_session import (
+    api_get as _api_get,
+)
+from portfolio.avanza_session import (
+    cancel_order as _cancel_order,
+)
+from portfolio.avanza_session import (
+    place_buy_order as _place_buy_order,
+)
+from portfolio.avanza_session import (
+    place_sell_order as _place_sell_order,
+)
+from portfolio.avanza_session import (
+    place_stop_loss as _place_stop_loss_session,
+)
+from portfolio.avanza_session import (
+    place_trailing_stop as _place_trailing_stop_session,
+)
+from portfolio.avanza_session import (
+    verify_session,
+)
+
+
+def fetch_price_no_page(orderbook_id: str, api_type: str = "certificate"):
+    """Fetch a quote without a Playwright page — uses BankID session API."""
+    normalized = normalize_api_type(api_type)
+    try:
+        data = _api_get(f"/_api/market-guide/{normalized}/{orderbook_id}")
+        quote = data.get("quote", {})
+        ki = data.get("keyIndicators", {})
+        underlying = data.get("underlying", {})
+        def _v(obj):
+            return obj.get("value") if isinstance(obj, dict) else obj
+        return {
+            "bid": _v(quote.get("buy")),
+            "ask": _v(quote.get("sell")),
+            "last": _v(quote.get("last")),
+            "change_pct": _v(quote.get("changePercent")),
+            "high": _v(quote.get("highest")),
+            "low": _v(quote.get("lowest")),
+            "underlying": _v(underlying.get("quote", {}).get("last")),
+            "underlying_name": underlying.get("name"),
+            "leverage": _v(ki.get("leverage")),
+            "barrier": _v(ki.get("barrierLevel")),
+            "api_type": normalized,
+        }
+    except Exception as e:
+        logger.error("Warrant price fetch failed for orderbook %s: %s", orderbook_id, e, exc_info=True)
+        return None
+
+
+def fetch_price_no_page_with_fallback(orderbook_id: str, api_type: str | None = None):
+    """Try preferred type then fallback chain — no Playwright page needed."""
+    if not orderbook_id:
+        return None
+    candidates = []
+    preferred = normalize_api_type(api_type) if api_type else ""
+    if preferred:
+        candidates.append(preferred)
+    for fb in _PRICE_FALLBACK_TYPES:
+        if fb not in candidates:
+            candidates.append(fb)
+    for candidate in candidates:
+        data = fetch_price_no_page(orderbook_id, candidate)
+        if data and (data.get("bid") is not None or data.get("ask") is not None or data.get("last") is not None):
+            return data
+    return None
+
+
+def place_order_no_page(account_id, ob_id, side, price, volume):
+    """Place BUY/SELL via BankID session — no Playwright page needed.
+
+    Returns:
+        Tuple (ok: bool, result: dict) matching the page-based interface.
+
+    Raises:
+        ValueError: If *side* is not "BUY" or "SELL" (C2 fail-safe).
+    """
+    normalized_side = (side or "").strip().upper()
+    if normalized_side not in ("BUY", "SELL"):
+        raise ValueError(
+            f"Invalid order side: {side!r} (must be 'BUY' or 'SELL')"
+        )
+    if normalized_side == "BUY":
+        result = _place_buy_order(ob_id, price, volume, account_id)
+    else:
+        result = _place_sell_order(ob_id, price, volume, account_id)
+    ok = result.get("orderRequestStatus") == "SUCCESS"
+    return ok, result
+
+
+def place_stop_loss_no_page(account_id, ob_id, trigger_price, sell_price, volume, valid_days=8):
+    """Hardware stop-loss via BankID session — no Playwright page needed.
+
+    Returns:
+        Tuple (ok: bool, result: dict) matching the page-based interface.
+    """
+    result = _place_stop_loss_session(ob_id, trigger_price, sell_price, volume, account_id, valid_days)
+    ok = result.get("status") == "SUCCESS"
+    return ok, result
+
+
+def place_trailing_stop_no_page(account_id, ob_id, trail_percent, volume, valid_days=8):
+    """Hardware trailing stop via BankID session — no Playwright page needed.
+
+    Returns:
+        Tuple (ok: bool, result: dict) matching the page-based interface.
+    """
+    result = _place_trailing_stop_session(ob_id, trail_percent, volume, account_id, valid_days)
+    ok = result.get("status") == "SUCCESS"
+    return ok, result
+
+
+def delete_order_no_page(account_id, order_id):
+    """Cancel order via BankID session — no Playwright page needed.
+
+    Returns:
+        Tuple (ok: bool, result: dict) matching the page-based interface.
+    """
+    result = _cancel_order(order_id, account_id)
+    ok = result.get("orderRequestStatus") == "SUCCESS"
+    return ok, result
+
+
+def delete_stop_loss_no_page(account_id, stop_id):
+    """Delete stop-loss via BankID session — no Playwright page needed.
+
+    Returns:
+        Tuple (ok: bool, result: dict) matching the page-based interface.
+    """
+    resolved_account_id = str(account_id or get_account_id())
+    try:
+        result = _api_delete(f"/_api/trading/stoploss/{resolved_account_id}/{stop_id}")
+        # H18: Check for error indicators in the response.
+        # API returns {} on success (200 with empty body).
+        # A non-empty response with error keys indicates failure.
+        if isinstance(result, dict) and result.get("errorCode"):
+            logger.warning("Delete stop-loss returned error for stop %s: %s", stop_id, result)
+            return False, result
+        return True, result
+    except Exception as e:
+        logger.error("Delete stop-loss (no page) failed for stop %s: %s", stop_id, e, exc_info=True)
+        return False, {"error": str(e)}
+
+
+__all__ = [
+    "check_session_alive",
+    "delete_order",
+    "delete_order_live",
+    "delete_order_no_page",
+    "delete_stop_loss",
+    "delete_stop_loss_no_page",
+    "fetch_account_cash",
+    "fetch_page_positions",
+    "fetch_price",
+    "fetch_price_no_page",
+    "fetch_price_no_page_with_fallback",
+    "fetch_price_with_fallback",
+    "find_instrument",
+    "get_account_id",
+    "get_csrf",
+    "get_open_orders",
+    "get_portfolio_value",
+    "get_positions",
+    "get_price_info",
+    "normalize_api_type",
+    "place_buy_order",
+    "place_order",
+    "place_order_no_page",
+    "place_sell_order",
+    "place_stop_loss",
+    "place_stop_loss_no_page",
+    "place_trailing_stop_no_page",
+    "verify_session",
+]
diff --git a/portfolio/avanza_order_lock.py b/portfolio/avanza_order_lock.py
new file mode 100644
index 00000000..e2baf31a
--- /dev/null
+++ b/portfolio/avanza_order_lock.py
@@ -0,0 +1,100 @@
+"""Cross-process advisory file lock guarding Avanza order placement.
+
+Used across metals_loop, golddigger, fin_snipe_manager (and any future
+Avanza-bound loop) to prevent two processes submitting overlapping orders
+when they observe the same ``buying_power`` simultaneously. Without this,
+the following sequence is possible:
+
+    t=0.00  metals_loop reads buying_power=6000 SEK
+    t=0.05  golddigger reads buying_power=6000 SEK
+    t=0.10  metals_loop POSTs order for 5000 SEK
+    t=0.15  golddigger POSTs order for 5000 SEK
+    t=0.20  Avanza rejects one / or both fill and settlement overdraws
+
+READ paths (``fetch_price``, ``fetch_positions``, ``buying_power``) are
+NOT guarded — they're safe to run concurrently, and the whole point of
+the resilience refactor is to keep those fast.
+
+Usage:
+
+    from portfolio.avanza_order_lock import avanza_order_lock
+
+    with avanza_order_lock(op="place_order"):
+        resp = api_post("/_api/trading-critical/rest/order/new", payload)
+
+Design notes:
+
+* ``filelock.FileLock`` is already in requirements (3.20.3 as of 2026-04-13).
+* 2-second fail-fast default — long enough to ride through a normal order
+  round-trip (~300ms), short enough that a hung peer doesn't block trading.
+* Raises ``OrderLockBusyError`` on timeout so callers can log + retry next
+  cycle instead of blocking the whole loop.
+* Caller-provided ``op`` label threads through to log messages for
+  diagnostics ("which loop hit the busy lock").
+"""
+
+from __future__ import annotations
+
+import logging
+from collections.abc import Iterator
+from contextlib import contextmanager
+from pathlib import Path
+
+import filelock
+
+logger = logging.getLogger("portfolio.avanza_order_lock")
+
+BASE_DIR = Path(__file__).resolve().parent.parent
+DATA_DIR = BASE_DIR / "data"
+LOCK_FILE = DATA_DIR / "avanza_order.lock"
+
+DEFAULT_TIMEOUT_S = 2.0
+
+
+class OrderLockBusyError(Exception):
+    """Another process held the lock longer than the configured timeout."""
+
+
+@contextmanager
+def avanza_order_lock(
+    *,
+    timeout_s: float = DEFAULT_TIMEOUT_S,
+    op: str = "order",
+    lock_file: Path | None = None,
+) -> Iterator[filelock.FileLock]:
+    """Acquire the cross-process Avanza order lock for a short critical section.
+
+    Fail-fast after ``timeout_s``. The lock is released automatically on exit.
+
+    Args:
+        timeout_s: Seconds to wait for the lock before raising
+            :class:`OrderLockBusyError`. Default 2.0 — short enough to abort
+            a stuck caller, long enough to ride through a normal order RTT.
+        op: Short label for the operation, threaded into log messages.
+        lock_file: Override the lock path (tests only). Defaults to
+            ``data/avanza_order.lock``.
+
+    Raises:
+        OrderLockBusyError: If another process held the lock longer than
+            ``timeout_s`` seconds.
+    """
+    target = Path(lock_file) if lock_file is not None else LOCK_FILE
+    target.parent.mkdir(parents=True, exist_ok=True)
+    lock = filelock.FileLock(str(target), timeout=timeout_s)
+    try:
+        lock.acquire()
+    except filelock.Timeout as exc:
+        logger.warning(
+            "avanza_order_lock(%s): busy after %.1fs — another process holds the lock",
+            op, timeout_s,
+        )
+        raise OrderLockBusyError(f"lock busy after {timeout_s}s (op={op})") from exc
+    try:
+        logger.debug("avanza_order_lock(%s): acquired", op)
+        yield lock
+    finally:
+        try:
+            lock.release()
+            logger.debug("avanza_order_lock(%s): released", op)
+        except Exception as exc:
+            logger.warning("avanza_order_lock(%s): release failed: %s", op, exc)
diff --git a/portfolio/avanza_orders.py b/portfolio/avanza_orders.py
new file mode 100644
index 00000000..3425c443
--- /dev/null
+++ b/portfolio/avanza_orders.py
@@ -0,0 +1,416 @@
+"""Avanza order confirmation flow — human-in-the-loop for real money.
+
+Workflow:
+1. Layer 2 calls request_order() → saves intent to pending orders, returns details
+   (including a unique 6-hex `confirm_token`).
+2. Layer 2 sends Telegram message with order details + "Reply CONFIRM <token>
+   to execute".
+3. Main loop calls check_pending_orders() each cycle.
+4. On CONFIRM <token> reply → execute the order whose token matches, notify
+   via Telegram.
+5. On timeout (5 min) → expire the pending order, notify.
+
+P1-10 (2026-05-02): per-order `confirm_token` eliminates three races the
+old bare-CONFIRM design suffered from (see test class docstrings):
+- stale-CONFIRM race (replayed CONFIRM confirms a NEWER order)
+- wrong-order race (sort-by-time-DESC matches the wrong order)
+- no-pending-yet race (CONFIRM lands before the order it was for)
+
+Bare CONFIRM (no token) is still accepted but ONLY matches LEGACY orders
+that have no `confirm_token` field — i.e. orders that were already in
+flight when this code was deployed. New orders MUST be confirmed by token.
+"""
+
+import contextlib
+import logging
+import re
+import secrets
+import uuid
+from datetime import UTC, datetime, timedelta
+from pathlib import Path
+
+from portfolio.avanza_control import place_buy_order, place_sell_order
+from portfolio.file_utils import atomic_write_json, load_json
+from portfolio.http_retry import fetch_with_retry
+from portfolio.telegram_notifications import send_telegram
+
+logger = logging.getLogger("portfolio.avanza_orders")
+
+DATA_DIR = Path(__file__).resolve().parent.parent / "data"
+PENDING_FILE = DATA_DIR / "avanza_pending_orders.json"
+EXPIRY_MINUTES = 5
+
+# P1-10 (2026-05-02): per-order confirmation nonce. 6 hex chars = 24 bits
+# of entropy ≈ ~16M possible tokens. Collision probability across the at-most
+# ~5 in-flight pending orders is effectively zero (birthday bound:
+# ~5^2/(2*16M) ≈ 7.5e-7). Long enough to survive typos, short enough that
+# users will actually type it on a phone keyboard.
+_CONFIRM_TOKEN_HEX_CHARS = 6
+# Token validation: anything outside [0-9a-f] is silently dropped rather
+# than confirmed against an unknown order. This prevents 'CONFIRM xyz' (a
+# typo) from accidentally confirming any order via the legacy bare-CONFIRM
+# path or matching a token-holding order.
+_HEX_TOKEN_RE = re.compile(r"^[0-9a-f]+$")
+# CONFIRM prefix matcher. Word boundary required because "confirmed" /
+# "confirms" / "confirmation" parse to "confirm" + a hex-valid suffix
+# ("ed", "s", "ation") which would silently match against legacy orders
+# or non-existent tokens. Anchored at start since the user is asked to
+# reply with "CONFIRM <token>" as the entire message.
+_CONFIRM_PREFIX_RE = re.compile(r"^confirm(?:\s+|$)")
+
+
+def _generate_confirm_token() -> str:
+    """Return a fresh hex token for a new pending order. Module-level
+    indirection keeps tests deterministic via patch.object if ever needed."""
+    return secrets.token_hex(_CONFIRM_TOKEN_HEX_CHARS // 2)
+
+
+def _load_pending() -> list[dict]:
+    """Load pending orders from disk."""
+    result = load_json(PENDING_FILE, default=[])
+    if result is None:
+        logger.warning("Failed to read pending orders, returning empty")
+        return []
+    return result
+
+
+def _save_pending(orders: list[dict]) -> None:
+    """Save pending orders to disk atomically."""
+    atomic_write_json(PENDING_FILE, orders)
+
+
+def request_order(
+    action: str,
+    orderbook_id: str,
+    instrument_name: str,
+    config_key: str,
+    volume: int,
+    price: float,
+) -> dict:
+    """Create a pending order awaiting Telegram confirmation.
+
+    Args:
+        action: "BUY" or "SELL"
+        orderbook_id: Avanza orderbook ID
+        instrument_name: Human-readable name (e.g. "SAAB B")
+        config_key: Config key (e.g. "SAAB-B")
+        volume: Number of shares
+        price: Limit price in SEK
+
+    Returns:
+        The pending order dict (includes id, total_sek, expires, and
+        ``confirm_token``). The caller MUST include ``confirm_token`` in
+        the Telegram notification asking the user to reply
+        ``CONFIRM <token>``. Without that, the user sees the prompt but
+        has no way to confirm — bare CONFIRM only matches legacy orders
+        without a token.
+    """
+    if action not in ("BUY", "SELL"):
+        raise ValueError(f"action must be BUY or SELL, got {action!r}")
+    if volume < 1:
+        raise ValueError(f"volume must be >= 1, got {volume}")
+    if price <= 0:
+        raise ValueError(f"price must be > 0, got {price}")
+
+    now = datetime.now(UTC)
+    confirm_token = _generate_confirm_token()
+    order = {
+        "id": str(uuid.uuid4()),
+        "timestamp": now.isoformat(),
+        "action": action,
+        "orderbook_id": str(orderbook_id),
+        "instrument_name": instrument_name,
+        "config_key": config_key,
+        "volume": volume,
+        "price": price,
+        "total_sek": round(volume * price, 2),
+        "status": "pending_confirmation",
+        "expires": (now + timedelta(minutes=EXPIRY_MINUTES)).isoformat(),
+        "confirm_token": confirm_token,
+    }
+
+    pending = _load_pending()
+    pending.append(order)
+    _save_pending(pending)
+    # Log the token at INFO so an operator reading agent.log can read it
+    # if they need to confirm out-of-band (e.g. the agent's Telegram message
+    # got truncated). The token is per-order, expires in 5 min, and only
+    # confirms one specific order — leak surface is minimal.
+    logger.info(
+        "Order requested: %s %dx %s @ %.2f SEK (id=%s, confirm_token=%s)",
+        action, volume, instrument_name, price, order["id"], confirm_token,
+    )
+    return order
+
+
+def get_pending_orders() -> list[dict]:
+    """Get all orders with status 'pending_confirmation'."""
+    return [o for o in _load_pending() if o["status"] == "pending_confirmation"]
+
+
+def check_pending_orders(config: dict) -> list[dict]:
+    """Check for Telegram confirmations and expire stale orders.
+
+    Called by the main loop each cycle. Polls Telegram getUpdates for
+    CONFIRM <token> replies. Executes confirmed orders (matched by token)
+    and expires timed-out ones.
+
+    P1-10 (2026-05-02): a CONFIRM <token> reply confirms ONLY the order
+    whose ``confirm_token`` matches. Bare CONFIRM (no token) still works
+    but ONLY matches LEGACY orders without a token field — so freshly
+    created orders cannot be silently confirmed by a stale CONFIRM that
+    was replayed by a getUpdates offset bug.
+
+    Args:
+        config: App config dict (with telegram.token, telegram.chat_id,
+            and optionally telegram.allowed_user_id for sender auth).
+
+    Returns:
+        List of orders that were acted on (confirmed or expired) this cycle.
+    """
+    pending = _load_pending()
+    if not pending:
+        return []
+
+    acted_on = []
+    now = datetime.now(UTC)
+
+    # Set of tokens that arrived this cycle. Bare CONFIRM is "" (empty
+    # string) — only matches legacy orders without a token.
+    confirmed_tokens = _check_telegram_confirm(config)
+
+    for order in pending:
+        if order["status"] != "pending_confirmation":
+            continue
+
+        expires = datetime.fromisoformat(order["expires"])
+        order_token = order.get("confirm_token", "")
+
+        # P1-10: matching rules.
+        # 1. Order has a token AND that token is in confirmed_tokens → confirm.
+        # 2. Order has NO token (legacy in-flight order) AND bare CONFIRM
+        #    arrived ("" in the set) → confirm. This is the backwards-compat
+        #    path for orders that existed before the deploy.
+        # 3. Otherwise → no confirmation this cycle (may still expire).
+        confirmed_by_token = bool(order_token) and order_token in confirmed_tokens
+        confirmed_legacy = (not order_token) and ("" in confirmed_tokens)
+
+        if confirmed_by_token or confirmed_legacy:
+            order["status"] = "confirmed"
+            acted_on.append(order)
+            # Remove the matched token so the same CONFIRM can't double-fire
+            # against another order in the same cycle.
+            if confirmed_by_token:
+                confirmed_tokens.discard(order_token)
+            else:
+                # Legacy bare CONFIRM only matches one legacy order per cycle.
+                confirmed_tokens.discard("")
+            _execute_confirmed_order(order, config)
+        elif now > expires:
+            order["status"] = "expired"
+            acted_on.append(order)
+            _notify_expired(order, config)
+
+    _save_pending(pending)
+    return acted_on
+
+
+def _check_telegram_confirm(config: dict) -> set[str]:
+    """Poll Telegram for CONFIRM <token> replies from the configured chat.
+
+    Returns ``set[str]`` of matched tokens (lowercase hex). Bare CONFIRM
+    (with no token) is represented as ``""`` and matches only LEGACY
+    pending orders without a ``confirm_token`` field. Anything that's not
+    valid hex after CONFIRM (e.g. ``CONFIRM xyz`` typo) is silently
+    dropped — never matched against an order — so a typo doesn't
+    accidentally confirm via the legacy path.
+
+    Uses getUpdates with a stored offset to avoid reprocessing old messages.
+
+    AV-P1-3 (2026-05-02): Sender-authenticated when
+    ``telegram.allowed_user_id`` is set. Without sender auth, the chat-only
+    filter is bypassable in two ways:
+      - Group chats: anyone admitted can send CONFIRM and execute the
+        pending order.
+      - Bot-token compromise: an attacker who has the bot token can
+        deliver fake updates with the right ``chat_id`` and execute orders.
+    When ``allowed_user_id`` is unset the chat-only check is preserved
+    (backwards-compatible). The offset still advances on dropped messages
+    so we don't re-process the rejected update every cycle.
+
+    P1-10 (2026-05-02): return type changed from ``bool`` to ``set[str]``
+    so each pending order can match its own token. Bare CONFIRM is still
+    captured (as ``""``) for the legacy backwards-compat path.
+    """
+    token = config.get("telegram", {}).get("token", "")
+    chat_id = str(config.get("telegram", {}).get("chat_id", ""))
+    if not token or not chat_id:
+        return set()
+
+    # AV-P1-3 (2026-05-02): optional sender allow-list. Accept either int
+    # or string in config — Telegram's `from.id` is always int, so coerce
+    # both sides to str for comparison so format mistakes don't accidentally
+    # admit/reject a real user.
+    raw_allowed_user = config.get("telegram", {}).get("allowed_user_id")
+    allowed_user = str(raw_allowed_user) if raw_allowed_user is not None else None
+
+    # Load stored offset (BUG-128: now atomic JSON; handles legacy plain-text format)
+    offset_file = DATA_DIR / "avanza_telegram_offset.txt"
+    offset = 0
+    offset_data = load_json(offset_file)
+    if isinstance(offset_data, dict):
+        offset = int(offset_data.get("offset", 0))
+    elif offset_file.exists():
+        with contextlib.suppress(ValueError, OSError):
+            offset = int(offset_file.read_text().strip())
+
+    params = {"timeout": 1, "allowed_updates": ["message"]}
+    if offset:
+        params["offset"] = offset
+
+    try:
+        r = fetch_with_retry(
+            f"https://api.telegram.org/bot{token}/getUpdates",
+            params=params,
+            timeout=5,
+        )
+        if r is None or not r.ok:
+            return set()
+        data = r.json()
+        if not data.get("ok"):
+            return set()
+    except Exception as e:
+        logger.warning("Telegram getUpdates failed: %s", e)
+        return set()
+
+    found_tokens: set[str] = set()
+    for update in data.get("result", []):
+        update_id = update.get("update_id", 0)
+        # Always advance offset (AV-P1-3: applies to dropped messages too —
+        # otherwise a single rejected CONFIRM would replay every cycle).
+        if update_id >= offset:
+            offset = update_id + 1
+
+        msg = update.get("message", {})
+        if str(msg.get("chat", {}).get("id")) != chat_id:
+            continue
+
+        # AV-P1-3 (2026-05-02): sender authentication. Fail-closed:
+        # missing `from` field with auth enabled drops the message.
+        if allowed_user is not None:
+            sender = msg.get("from") or {}
+            sender_id = sender.get("id")
+            if sender_id is None or str(sender_id) != allowed_user:
+                logger.warning(
+                    "Dropping Telegram message from unauthorized sender id=%r "
+                    "(allowed=%s, chat=%s)",
+                    sender_id, allowed_user, chat_id,
+                )
+                continue
+
+        # P1-10 (2026-05-02): parse "CONFIRM <token>" or bare "CONFIRM".
+        # Lowercase + collapse whitespace so user-typed variants normalize.
+        # Word-boundary match is critical here — without it, "confirmed"
+        # parses as "confirm" + "ed" and "ed" IS valid hex (defense vs an
+        # accidental "confirmed by my broker" message in the chat).
+        text = (msg.get("text") or "").strip().lower()
+        m = _CONFIRM_PREFIX_RE.match(text)
+        if not m:
+            continue
+        # Anything after the matched prefix (which includes the word
+        # "confirm" + whitespace OR end-of-string) is the candidate.
+        rest = text[m.end():].strip()
+        if not rest:
+            # Bare CONFIRM — legacy backwards-compat path.
+            found_tokens.add("")
+            continue
+        # Take the first whitespace-separated token. Anything trailing is
+        # ignored (lets the user paste extra text without breaking the match).
+        candidate = rest.split()[0]
+        if _HEX_TOKEN_RE.match(candidate):
+            found_tokens.add(candidate)
+        else:
+            logger.warning(
+                "Dropping CONFIRM with non-hex token %r (must be lowercase "
+                "[0-9a-f] from request_order's confirm_token)",
+                candidate,
+            )
+
+    # Save offset atomically to prevent corruption on crash (BUG-128)
+    try:
+        atomic_write_json(offset_file, {"offset": offset})
+    except OSError as e:
+        logger.warning("Failed to save Telegram offset: %s", e)
+
+    return found_tokens
+
+
+def _execute_confirmed_order(order: dict, config: dict) -> None:
+    """Execute a confirmed order on Avanza and notify via Telegram."""
+    action = order["action"]
+    try:
+        if action == "BUY":
+            result = place_buy_order(
+                orderbook_id=order["orderbook_id"],
+                price=order["price"],
+                volume=order["volume"],
+            )
+        else:
+            result = place_sell_order(
+                orderbook_id=order["orderbook_id"],
+                price=order["price"],
+                volume=order["volume"],
+            )
+
+        status = result.get("orderRequestStatus", "UNKNOWN")
+        order_id = result.get("orderId", "?")
+        msg_text = result.get("message", "")
+
+        if status == "SUCCESS":
+            order["status"] = "executed"
+            order["avanza_order_id"] = order_id
+            msg = (
+                f"AVANZA {action} EXECUTED\n"
+                f"{order['instrument_name']}: {order['volume']}x @ {order['price']:.2f} SEK\n"
+                f"Total: {order['total_sek']:,.0f} SEK\n"
+                f"Order ID: {order_id}"
+            )
+            logger.info("Order executed: %s (avanza_id=%s)", order["id"], order_id)
+        else:
+            order["status"] = "failed"
+            order["error"] = msg_text
+            msg = (
+                f"AVANZA {action} FAILED\n"
+                f"{order['instrument_name']}: {order['volume']}x @ {order['price']:.2f} SEK\n"
+                f"Error: {msg_text}"
+            )
+            logger.error("Order failed: %s — %s", order["id"], msg_text)
+
+        send_telegram(msg, config)
+
+    except Exception as e:
+        order["status"] = "error"
+        order["error"] = str(e)
+        logger.error("Order execution error: %s — %s", order["id"], e)
+        try:
+            send_telegram(
+                f"AVANZA ORDER ERROR\n{order['instrument_name']}: {e}",
+                config,
+            )
+        except Exception as e:
+            logger.warning("Order error notification failed: %s", e)
+
+
+def _notify_expired(order: dict, config: dict) -> None:
+    """Notify via Telegram that a pending order expired."""
+    msg = (
+        f"AVANZA ORDER EXPIRED\n"
+        f"{order['action']} {order['instrument_name']}: "
+        f"{order['volume']}x @ {order['price']:.2f} SEK\n"
+        f"No confirmation received within {EXPIRY_MINUTES} min."
+    )
+    logger.info("Order expired: %s", order["id"])
+    try:
+        send_telegram(msg, config)
+    except Exception as e:
+        logger.warning("Failed to send expiry notification: %s", e)
diff --git a/portfolio/avanza_resilient_page.py b/portfolio/avanza_resilient_page.py
new file mode 100644
index 00000000..30162056
--- /dev/null
+++ b/portfolio/avanza_resilient_page.py
@@ -0,0 +1,231 @@
+"""Auto-recovering Playwright Page wrapper for Avanza-bound loops.
+
+Problem: long-running loops (`data/metals_loop.py`, `portfolio/golddigger/`,
+`portfolio/main.py` via `avanza_session.py`) open a headless Chromium at
+startup and hold the `page` reference for days. When the browser dies
+(OS sleep, memory pressure, WSL ping hiccup, external BankID re-auth) the
+Python process keeps running but every `page.evaluate()` throws
+`playwright._impl._errors.TargetClosedError` — silently, for days. The
+bug discovered 2026-04-13: metals loop was emitting this error 662 times
+between 2026-04-09 and 2026-04-13, making zero trades.
+
+Solution: pass a `ResilientPage` instead of a raw Playwright `Page`. On
+`TargetClosedError` (or equivalent browser-dead message) the wrapper tears
+down the dead browser+context, relaunches Chromium, reloads the saved
+`avanza_storage_state.json`, and retries the failing call once. Only then
+does it propagate the error.
+
+This is the minimal surface — `evaluate()` and `context.cookies()` — that
+the existing helpers use. Other Page methods pass through unchanged via
+`__getattr__`; they get no auto-recovery (good enough: they're only used
+during startup, where crash-and-bat-restart is acceptable).
+
+Usage:
+
+    with sync_playwright() as pw:
+        page = ResilientPage.open(pw, "data/avanza_storage_state.json")
+        # pass `page` to existing helpers — zero call-site changes needed
+        fetch_price(page, ob_id, api_type)
+        fetch_account_cash(page, account_id)
+        # on shutdown:
+        page.close()
+"""
+
+from __future__ import annotations
+
+import datetime
+import logging
+from typing import Any
+
+logger = logging.getLogger("portfolio.avanza_resilient_page")
+
+_INITIAL_URL = "https://www.avanza.se/min-ekonomi/oversikt.html"
+_INITIAL_URL_WAIT_MS = 2000
+
+
+def is_browser_dead_error(exc: BaseException) -> bool:
+    """True if ``exc`` signals a dead Playwright browser/context.
+
+    Checks both the exception class name (Playwright's
+    ``TargetClosedError`` name changed across versions) and the message
+    (the stable cross-version signal). Exposed for tests and for
+    ``avanza_session.py`` which wants the same classifier without
+    importing Playwright internals.
+    """
+    name = type(exc).__name__
+    if name == "TargetClosedError":
+        return True
+    msg = str(exc)
+    for marker in (
+        "Target page, context or browser has been closed",
+        "Target closed",
+        "Browser has been closed",
+        "has been closed",
+    ):
+        if marker in msg:
+            return True
+    return False
+
+
+class ResilientPage:
+    """Playwright ``Page`` wrapper that auto-relaunches the browser on death."""
+
+    def __init__(
+        self,
+        pw: Any,
+        storage_state_path: str,
+        *,
+        headless: bool = True,
+        locale: str = "sv-SE",
+        initial_url: str | None = _INITIAL_URL,
+        initial_url_wait_ms: int = _INITIAL_URL_WAIT_MS,
+    ) -> None:
+        self._pw = pw
+        self._storage_state_path = storage_state_path
+        self._headless = headless
+        self._locale = locale
+        self._initial_url = initial_url
+        self._initial_url_wait_ms = initial_url_wait_ms
+        self._browser = None
+        self._ctx = None
+        self._page = None
+        self._relaunch_count = 0
+        self._last_relaunch_ts: str | None = None
+
+    @classmethod
+    def open(
+        cls,
+        pw: Any,
+        storage_state_path: str,
+        *,
+        headless: bool = True,
+        locale: str = "sv-SE",
+        initial_url: str | None = _INITIAL_URL,
+        initial_url_wait_ms: int = _INITIAL_URL_WAIT_MS,
+    ) -> ResilientPage:
+        """Construct and open the browser. Preferred entry point."""
+        rp = cls(
+            pw,
+            storage_state_path,
+            headless=headless,
+            locale=locale,
+            initial_url=initial_url,
+            initial_url_wait_ms=initial_url_wait_ms,
+        )
+        rp._open()
+        return rp
+
+    def _open(self) -> None:
+        self._browser = self._pw.chromium.launch(headless=self._headless)
+        self._ctx = self._browser.new_context(
+            storage_state=self._storage_state_path,
+            locale=self._locale,
+        )
+        self._page = self._ctx.new_page()
+        if self._initial_url:
+            self._page.goto(self._initial_url, wait_until="domcontentloaded")
+            if self._initial_url_wait_ms:
+                self._page.wait_for_timeout(self._initial_url_wait_ms)
+
+    def _close_quietly(self) -> None:
+        for closer in (self._ctx, self._browser):
+            if closer is None:
+                continue
+            try:
+                closer.close()
+            except Exception as exc:
+                logger.debug("ResilientPage teardown: %s", exc)
+        self._ctx = None
+        self._browser = None
+        self._page = None
+
+    def _relaunch(self, *, reason: str) -> None:
+        self._relaunch_count += 1
+        self._last_relaunch_ts = datetime.datetime.now(datetime.UTC).isoformat()
+        logger.warning(
+            "ResilientPage: browser dead (%s) — relaunching (count=%d)",
+            reason, self._relaunch_count,
+        )
+        self._close_quietly()
+        self._open()
+
+    def close(self) -> None:
+        """Teardown browser. Safe to call multiple times."""
+        self._close_quietly()
+
+    # --- Recovery-aware proxy API ---
+
+    def evaluate(self, script: str, arg: Any = None) -> Any:
+        """``page.evaluate(script, arg)`` with one-shot auto-recovery.
+
+        On ``TargetClosedError`` (or equivalent), teardown + relaunch +
+        retry. If the retry also fails with a browser-dead error, propagate.
+        """
+        try:
+            if arg is None:
+                return self._page.evaluate(script)
+            return self._page.evaluate(script, arg)
+        except Exception as exc:
+            if not is_browser_dead_error(exc):
+                raise
+            self._relaunch(reason="evaluate")
+            if arg is None:
+                return self._page.evaluate(script)
+            return self._page.evaluate(script, arg)
+
+    def goto(self, *args, **kwargs) -> Any:
+        """``page.goto()`` with one-shot auto-recovery."""
+        try:
+            return self._page.goto(*args, **kwargs)
+        except Exception as exc:
+            if not is_browser_dead_error(exc):
+                raise
+            self._relaunch(reason="goto")
+            return self._page.goto(*args, **kwargs)
+
+    @property
+    def context(self):
+        """Return a context proxy whose ``cookies()`` auto-recovers."""
+        return _ResilientContextProxy(self)
+
+    # Passthrough for everything else (wait_for_timeout, locator, on, etc.)
+    def __getattr__(self, name: str) -> Any:
+        # __getattr__ only fires when normal lookup fails — so this does
+        # NOT shadow evaluate/goto/context/close defined above.
+        if name.startswith("_"):
+            raise AttributeError(name)
+        target = self.__dict__.get("_page")
+        if target is None:
+            raise AttributeError(name)
+        return getattr(target, name)
+
+    # --- Observability ---
+
+    @property
+    def relaunch_count(self) -> int:
+        return self._relaunch_count
+
+    @property
+    def last_relaunch_ts(self) -> str | None:
+        return self._last_relaunch_ts
+
+
+class _ResilientContextProxy:
+    """Proxy for ``BrowserContext`` that auto-recovers ``cookies()`` calls."""
+
+    def __init__(self, resilient_page: ResilientPage) -> None:
+        self._rp = resilient_page
+
+    def cookies(self) -> list[dict]:
+        try:
+            return self._rp._ctx.cookies()
+        except Exception as exc:
+            if not is_browser_dead_error(exc):
+                raise
+            self._rp._relaunch(reason="context.cookies")
+            return self._rp._ctx.cookies()
+
+    def __getattr__(self, name: str) -> Any:
+        if name.startswith("_"):
+            raise AttributeError(name)
+        return getattr(self._rp._ctx, name)
diff --git a/portfolio/avanza_session.py b/portfolio/avanza_session.py
new file mode 100644
index 00000000..039298fa
--- /dev/null
+++ b/portfolio/avanza_session.py
@@ -0,0 +1,1243 @@
+"""Avanza session management — load, validate, and use BankID-captured sessions.
+
+Uses Playwright's saved storage state to make authenticated API calls via a
+headless browser context. This ensures cookies and TLS session match what
+Avanza expects (replaying cookies via requests library causes 401s).
+
+This is the preferred auth method until TOTP credentials are configured.
+"""
+
+import json
+import logging
+import threading
+import time
+from collections.abc import Callable
+from datetime import UTC, date, datetime, timedelta
+from pathlib import Path
+from typing import Any
+
+from portfolio.avanza_order_lock import avanza_order_lock
+from portfolio.avanza_resilient_page import is_browser_dead_error
+from portfolio.file_utils import load_json
+
+logger = logging.getLogger("portfolio.avanza_session")
+
+BASE_DIR = Path(__file__).resolve().parent.parent
+DATA_DIR = BASE_DIR / "data"
+SESSION_FILE = DATA_DIR / "avanza_session.json"
+STORAGE_STATE_FILE = DATA_DIR / "avanza_storage_state.json"
+API_BASE = "https://www.avanza.se"
+
+# Minimum remaining session life before we consider it expired (minutes)
+EXPIRY_BUFFER_MINUTES = 30
+
+# Default trading account
+DEFAULT_ACCOUNT_ID = "1625505"
+
+# Whitelist of permitted account IDs — never trade outside these
+ALLOWED_ACCOUNT_IDS = {"1625505"}
+
+# Module-level Playwright context (lazy-initialized, reused across calls)
+# BUG-129: Protected by _pw_lock to prevent concurrent access corruption
+# A-AV-1 (2026-04-11): Upgraded to RLock so api_get/api_post/api_delete can
+# wrap their *entire* request flow under the lock — they call
+# _get_playwright_context() (which itself acquires the lock) inside the
+# critical section. The previous Lock would deadlock; RLock is reentrant
+# for the same thread. Without this, Playwright's sync_api was being used
+# concurrently from main loop's 8-worker pool + metals 10s fast-tick,
+# corrupting trade responses (e.g. CONFIRM stolen by wrong request).
+_pw_lock = threading.RLock()
+_pw_instance = None
+_pw_browser = None
+_pw_context = None
+
+
+class AvanzaSessionError(Exception):
+    """Raised when session is missing, expired, or invalid."""
+
+
+def load_session() -> dict:
+    """Load saved BankID session metadata from disk.
+
+    Returns:
+        Session dict with expiry info, customer_id, etc.
+
+    Raises:
+        AvanzaSessionError: if file missing, unreadable, or expired.
+    """
+    if not SESSION_FILE.exists():
+        raise AvanzaSessionError(
+            f"No session file found at {SESSION_FILE}. "
+            "Run: python scripts/avanza_login.py"
+        )
+
+    data = load_json(SESSION_FILE)
+    if data is None:
+        raise AvanzaSessionError(f"Failed to read session file: {SESSION_FILE}")
+
+    # Check expiry
+    expires_at = data.get("expires_at")
+    if expires_at:
+        try:
+            exp = datetime.fromisoformat(expires_at)
+            now = datetime.now(UTC)
+            if exp <= now:
+                raise AvanzaSessionError(
+                    f"Session expired at {expires_at}. "
+                    "Run: python scripts/avanza_login.py"
+                )
+        except ValueError:
+            logger.warning("Cannot parse expires_at %r — cannot verify expiry, proceeding with caution", expires_at)
+
+    if not STORAGE_STATE_FILE.exists():
+        raise AvanzaSessionError(
+            f"No storage state file at {STORAGE_STATE_FILE}. "
+            "Run: python scripts/avanza_login.py"
+        )
+
+    return data
+
+
+def session_remaining_minutes() -> float | None:
+    """Get minutes remaining on the current session, or None if no session."""
+    try:
+        data = load_json(SESSION_FILE)
+        if data is None:
+            return None
+        expires_at = data.get("expires_at")
+        if not expires_at:
+            return None
+        exp = datetime.fromisoformat(expires_at)
+        now = datetime.now(UTC)
+        return (exp - now).total_seconds() / 60.0
+    except Exception as e:
+        logger.warning("Failed to compute session minutes remaining: %s", e)
+        return None
+
+
+def is_session_expiring_soon(threshold_minutes: float = 60.0) -> bool:
+    """Check if session will expire within the given threshold.
+
+    Returns True if session is expired, expiring soon, or doesn't exist.
+    """
+    remaining = session_remaining_minutes()
+    if remaining is None:
+        return True
+    return remaining < threshold_minutes
+
+
+def _get_playwright_context():
+    """Get or create a headless Playwright browser context with saved auth state."""
+    global _pw_instance, _pw_browser, _pw_context
+
+    with _pw_lock:
+        if _pw_context is not None:
+            return _pw_context
+
+        # Validate session first
+        load_session()
+
+        from playwright.sync_api import sync_playwright
+
+        _pw_instance = sync_playwright().start()
+        _pw_browser = _pw_instance.chromium.launch(headless=True)
+        _pw_context = _pw_browser.new_context(
+            storage_state=str(STORAGE_STATE_FILE),
+            locale="sv-SE",
+        )
+        return _pw_context
+
+
+def close_playwright():
+    """Clean up Playwright resources."""
+    global _pw_instance, _pw_browser, _pw_context
+    with _pw_lock:
+        if _pw_context:
+            try:
+                _pw_context.close()
+            except Exception as e:
+                logger.debug("Context close failed: %s", e)
+            _pw_context = None
+        if _pw_browser:
+            try:
+                _pw_browser.close()
+            except Exception as e:
+                logger.debug("Browser close failed: %s", e)
+            _pw_browser = None
+        if _pw_instance:
+            try:
+                _pw_instance.stop()
+            except Exception as e:
+                logger.debug("Playwright stop failed: %s", e)
+            _pw_instance = None
+
+
+def verify_session() -> bool:
+    """Verify that the session is valid by making a lightweight API call.
+
+    Returns:
+        True if session is valid, False otherwise.
+    """
+    # A-AV-1: Hold _pw_lock for the entire context+request flow.
+    # ctx.request.* is NOT thread-safe; concurrent callers must serialize.
+    try:
+        with _pw_lock:
+            ctx = _get_playwright_context()
+            resp = ctx.request.get(f"{API_BASE}/_api/position-data/positions")
+            return resp.ok
+    except Exception as e:
+        logger.warning("Session verification failed: %s", e)
+        close_playwright()
+        return False
+
+
+# 2026-04-13: Auto-recovery wrapper for api_get/api_post/api_delete.
+# The singleton Playwright browser held in _pw_context occasionally dies
+# mid-flight (OS sleep, memory pressure, external BankID re-auth by the
+# user, cookie-jar corruption under heavy concurrency). When that happens
+# every subsequent ctx.request.* call throws TargetClosedError until the
+# process restarts. The pre-existing 401/403 path already knows to call
+# close_playwright() so the next request re-launches; we extend the same
+# pattern to browser-dead errors.
+#
+# Keeps the singleton + _pw_lock (BUG-129 / A-AV-1). The whole retry runs
+# under the RLock so a concurrent thread cannot partially observe the
+# teardown/relaunch. _get_playwright_context also acquires the lock but
+# it's reentrant for the same thread.
+def _with_browser_recovery(op: Callable[[Any], Any], *, op_name: str) -> Any:
+    """Run ``op(ctx)`` under ``_pw_lock``; on browser-dead error, teardown +
+    relaunch + retry once. Propagate all other exceptions unchanged.
+
+    ``op`` is called with the current Playwright context. The op is responsible
+    for making the actual ctx.request.* call and handling HTTP-level errors.
+    """
+    with _pw_lock:
+        ctx = _get_playwright_context()
+        try:
+            return op(ctx)
+        except Exception as exc:
+            if not is_browser_dead_error(exc):
+                raise
+            logger.warning(
+                "avanza_session: browser dead on %s (%r) — teardown + relaunch + retry",
+                op_name, exc,
+            )
+            close_playwright()
+            ctx = _get_playwright_context()
+            return op(ctx)
+
+
+# --- API convenience functions ---
+
+
+def api_get(path: str, **kwargs) -> Any:
+    """Make an authenticated GET request to Avanza API.
+
+    Args:
+        path: API path (e.g., "/_api/position-data/positions")
+
+    Returns:
+        Parsed JSON response.
+
+    Raises:
+        AvanzaSessionError: if session is invalid.
+    """
+    # A-AV-1: Hold _pw_lock for the entire request. Playwright's sync_api
+    # is NOT thread-safe and the metals fast-tick + main 8-worker pool race.
+    # 2026-04-13: Wrapped in _with_browser_recovery so TargetClosedError
+    # (browser died mid-flight) triggers a teardown + relaunch + retry.
+    url = f"{API_BASE}{path}" if path.startswith("/") else path
+
+    def _op(ctx):
+        resp = ctx.request.get(url)
+        if resp.status == 401:
+            close_playwright()
+            raise AvanzaSessionError(
+                "Session returned 401 Unauthorized. "
+                "Run: python scripts/avanza_login.py"
+            )
+        if not resp.ok:
+            raise RuntimeError(f"Avanza API error {resp.status}: {resp.text()[:500]}")
+        return resp.json()
+
+    return _with_browser_recovery(_op, op_name=f"GET {path}")
+
+
+def _get_csrf(ctx=None) -> str:
+    """Extract CSRF token from Playwright context cookies.
+
+    If ``ctx`` is provided (e.g. from inside an already-locked _with_recovery
+    block) it is used directly — avoids re-entering the RLock and avoids a
+    stale context reference after a relaunch. Otherwise acquires the lock
+    and fetches a fresh context.
+    """
+    if ctx is not None:
+        for c in ctx.cookies():
+            if c["name"] == "AZACSRF":
+                return c["value"]
+        raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")
+
+    # A-AV-1: ctx.cookies() reads Playwright internal state — needs lock.
+    with _pw_lock:
+        ctx = _get_playwright_context()
+        for c in ctx.cookies():
+            if c["name"] == "AZACSRF":
+                return c["value"]
+        raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")
+
+
+def api_post(path: str, payload: dict) -> Any:
+    """Make an authenticated POST request to Avanza API.
+
+    Automatically includes the X-SecurityToken (CSRF) header.
+
+    Args:
+        path: API path (e.g., "/_api/trading-critical/rest/order/new")
+        payload: Request body dict.
+
+    Returns:
+        Parsed JSON response.
+    """
+    # A-AV-1: Hold lock across CSRF read + POST so a concurrent request
+    # cannot rotate the cookie jar mid-flight.
+    # 2026-04-13: Wrapped in _with_browser_recovery. CSRF is read from the
+    # same ctx used for the POST, so a relaunch picks up fresh cookies in
+    # both places atomically (no stale-CSRF-against-fresh-context mismatch).
+    url = f"{API_BASE}{path}" if path.startswith("/") else path
+    body_data = json.dumps(payload)
+
+    def _op(ctx):
+        csrf = _get_csrf(ctx)
+        resp = ctx.request.post(
+            url,
+            data=body_data,
+            headers={
+                "Content-Type": "application/json",
+                "X-SecurityToken": csrf,
+            },
+        )
+        if resp.status == 401:
+            close_playwright()
+            raise AvanzaSessionError(
+                "Session returned 401 Unauthorized. "
+                "Run: python scripts/avanza_login.py"
+            )
+        if resp.status == 403:
+            close_playwright()
+            raise AvanzaSessionError(
+                "Session returned 403 Forbidden — CSRF token may be stale. "
+                "Run: python scripts/avanza_login.py"
+            )
+        body = resp.text()
+        try:
+            return json.loads(body)
+        except (json.JSONDecodeError, TypeError):
+            if not resp.ok:
+                raise RuntimeError(f"Avanza API error {resp.status}: {body[:500]}") from None
+            return {"raw": body}
+
+    return _with_browser_recovery(_op, op_name=f"POST {path}")
+
+
+def api_delete(path: str) -> Any:
+    """Make an authenticated DELETE request to Avanza API.
+
+    Automatically includes the X-SecurityToken (CSRF) header.
+
+    Args:
+        path: API path (e.g., "/_api/trading/stoploss/{stop_id}")
+
+    Returns:
+        Dict with ``http_status`` and ``ok`` keys.
+    """
+    # A-AV-1: Hold lock across CSRF read + DELETE.
+    # 2026-04-13: Wrapped in _with_browser_recovery (see api_get/api_post).
+    url = f"{API_BASE}{path}" if path.startswith("/") else path
+
+    def _op(ctx):
+        csrf = _get_csrf(ctx)
+        resp = ctx.request.delete(
+            url,
+            headers={
+                "Content-Type": "application/json",
+                "X-SecurityToken": csrf,
+            },
+        )
+        if resp.status == 401:
+            close_playwright()
+            raise AvanzaSessionError(
+                "Session returned 401 Unauthorized. "
+                "Run: python scripts/avanza_login.py"
+            )
+        return {"http_status": resp.status, "ok": 200 <= resp.status < 300 or resp.status == 404}
+
+    return _with_browser_recovery(_op, op_name=f"DELETE {path}")
+
+
+# --- Trading convenience functions ---
+
+
+def get_buying_power(account_id: str | None = None) -> dict | None:
+    """Get buying power and account value for an account.
+
+    2026-04-09 (Bug C7 fix): ported the multi-shape + multi-field-ID fallback
+    pattern from ``data/metals_avanza_helpers.fetch_account_cash`` after Avanza
+    changed the ``/_api/account-overview/overview/categorizedAccounts`` response
+    shape mid-day. The endpoint used to return a single top-level key
+    ``categorizedAccounts`` (an array of categories each with an ``accounts``
+    child). The new shape exposes three top-level keys simultaneously:
+    ``categories`` (new categorized path), ``accounts`` (flat list of all user
+    accounts), and ``loans``. At the same time, the per-account ID field
+    renamed from ``accountId`` to ``id`` (the other Avanza endpoints such as
+    ``position-data/positions`` already use ``id`` — see ``get_positions``).
+
+    Previously this function assumed the legacy shape + legacy ID field, so on
+    the new shape the iteration walked an empty list, then hit ``cats[0]`` on
+    an empty list (IndexError) or — if the shape still exposed the legacy key
+    but with no matches — silently returned fake numbers derived from the
+    first category's totalValue. That made callers like ``fish_straddle`` and
+    ``fish_monitor_live`` size positions off wrong cash balances.
+
+    We now try all three shapes (legacy categorized → flat → new categorized)
+    and all four known ID fields (``accountId``, ``id``, ``accountNumber``,
+    ``number``), taking whichever finds the target account first. On any
+    failure path we return ``None`` so callers can distinguish "API call failed"
+    from "balance is legitimately zero" — callers must now explicitly handle
+    the ``None`` case (previously they silently got ``buying_power=0``, which
+    was a dangerous silent failure).
+
+    Args:
+        account_id: Avanza account ID (default: ``DEFAULT_ACCOUNT_ID``).
+
+    Returns:
+        Dict with ``buying_power``, ``total_value``, ``own_capital`` (all SEK)
+        on success. ``None`` on any failure (HTTP error, account not found,
+        shape drift, etc.). Failures are logged with enough diagnostic context
+        (sample keys, counts per shape) to identify the next shape drift
+        without guessing.
+    """
+    aid = str(account_id or DEFAULT_ACCOUNT_ID)
+
+    try:
+        data = api_get("/_api/account-overview/overview/categorizedAccounts")
+    except Exception as e:
+        logger.warning(
+            "get_buying_power: api_get raised account_id=%s exception=%r",
+            aid, e,
+        )
+        return None
+
+    if not isinstance(data, dict):
+        logger.warning(
+            "get_buying_power: unexpected response type account_id=%s type=%s",
+            aid, type(data).__name__,
+        )
+        return None
+
+    def _v(obj):
+        """Unwrap Avanza {value: N} wrappers → N, else return obj as-is."""
+        if isinstance(obj, dict) and "value" in obj:
+            return obj["value"]
+        return obj
+
+    def _get_acc_id(acc: dict) -> str | None:
+        """Try every known ID field in order — matches fetch_account_cash.
+
+        Order preserved from the reference JS implementation so a regression
+        hitting one file makes the other equally easy to diagnose.
+        """
+        if not isinstance(acc, dict):
+            return None
+        for key in ("accountId", "id", "accountNumber", "number"):
+            val = acc.get(key)
+            if val is not None:
+                return str(val)
+        return None
+
+    def _get_balance(acc: dict, primary: str, alternates: tuple[str, ...]):
+        """Try primary balance field, fall back to alternates.
+
+        2026-04-09: we haven't confirmed whether `buyingPower` survived the
+        shape change, so we try common alternates if the primary is missing.
+        Mirrors getBalance() in fetch_account_cash.
+        """
+        p = _v(acc.get(primary))
+        if p is not None:
+            return p
+        for alt in alternates:
+            x = _v(acc.get(alt))
+            if x is not None:
+                return x
+        return None
+
+    def _make_result(acc: dict) -> dict:
+        return {
+            "buying_power": _get_balance(
+                acc, "buyingPower",
+                ("buyingPowerAvailable", "availableCash", "availableFunds"),
+            ),
+            "total_value": _get_balance(
+                acc, "totalValue",
+                ("accountTotalValue", "totalHoldings"),
+            ),
+            "own_capital": _get_balance(
+                acc, "ownCapital",
+                ("netDeposit", "selfOwnedCapital"),
+            ),
+        }
+
+    ids_seen: list[str] = []
+    sample_account_keys: list[str] | None = None
+
+    def _check_account(acc: dict) -> dict | None:
+        nonlocal sample_account_keys
+        if sample_account_keys is None and isinstance(acc, dict):
+            sample_account_keys = list(acc.keys())
+        acc_id = _get_acc_id(acc)
+        if acc_id is not None:
+            ids_seen.append(acc_id)
+        if acc_id == aid:
+            return _make_result(acc)
+        return None
+
+    # Path A (legacy, pre-2026-04-09): data.categorizedAccounts[].accounts[]
+    legacy_cats = data.get("categorizedAccounts") or []
+    for cat in legacy_cats:
+        for acc in (cat.get("accounts") or []):
+            r = _check_account(acc)
+            if r is not None:
+                return r
+
+    # Path B (new flat shape, 2026-04-09): data.accounts[]
+    flat_accounts = data.get("accounts") or []
+    for acc in flat_accounts:
+        r = _check_account(acc)
+        if r is not None:
+            return r
+
+    # Path C (new categorized shape, 2026-04-09): data.categories[].accounts[]
+    new_cats = data.get("categories") or []
+    for cat in new_cats:
+        for acc in (cat.get("accounts") or []):
+            r = _check_account(acc)
+            if r is not None:
+                return r
+
+    # Total miss — log the full diagnostic so the next shape drift is obvious.
+    logger.warning(
+        "get_buying_power: no_account_match account_id=%s "
+        "legacy_category_count=%d flat_account_count=%d new_category_count=%d "
+        "ids_seen=%s sample_account_keys=%s top_level_keys=%s",
+        aid, len(legacy_cats), len(flat_accounts), len(new_cats),
+        ids_seen, sample_account_keys, list(data.keys()),
+    )
+    return None
+
+
+def place_buy_order(
+    orderbook_id: str,
+    price: float,
+    volume: int,
+    account_id: str | None = None,
+    valid_until: str | None = None,
+) -> dict:
+    """Place a limit BUY order on Avanza.
+
+    Args:
+        orderbook_id: Avanza orderbook ID.
+        price: Limit price in SEK.
+        volume: Number of units (int >= 1).
+        account_id: Defaults to DEFAULT_ACCOUNT_ID.
+        valid_until: ISO date string. Defaults to today (day order).
+
+    Returns:
+        Dict with orderRequestStatus, orderId, message.
+    """
+    return _place_order("BUY", orderbook_id, price, volume, account_id, valid_until)
+
+
+def place_sell_order(
+    orderbook_id: str,
+    price: float,
+    volume: int,
+    account_id: str | None = None,
+    valid_until: str | None = None,
+) -> dict:
+    """Place a limit SELL order on Avanza."""
+    return _place_order("SELL", orderbook_id, price, volume, account_id, valid_until)
+
+
+def _place_order(
+    side: str,
+    orderbook_id: str,
+    price: float,
+    volume: int,
+    account_id: str | None = None,
+    valid_until: str | None = None,
+) -> dict:
+    """Internal: place a BUY or SELL limit order."""
+    if volume < 1:
+        raise ValueError(f"volume must be >= 1, got {volume}")
+    if price <= 0:
+        raise ValueError(f"price must be > 0, got {price}")
+
+    # H7: account whitelist guard
+    effective_account_id = str(account_id or DEFAULT_ACCOUNT_ID)
+    if effective_account_id not in ALLOWED_ACCOUNT_IDS:
+        raise ValueError(f"Refusing to trade on non-whitelisted account {effective_account_id!r}")
+
+    # H8: minimum order size guard
+    order_total = round(volume * price, 2)
+    if order_total < 1000.0:
+        raise ValueError(f"Order total {order_total:.2f} SEK below minimum 1000 SEK")
+
+    # BUG-211: maximum order size guard — prevents full-account exposure from
+    # a single malformed call (LLM hallucination, unit error, runaway loop).
+    # 50K SEK is ~25% of a 200K ISK account; adjust via config if needed.
+    MAX_ORDER_TOTAL_SEK = 50_000.0
+    if order_total > MAX_ORDER_TOTAL_SEK:
+        raise ValueError(
+            f"Order total {order_total:.2f} SEK exceeds maximum {MAX_ORDER_TOTAL_SEK:.0f} SEK"
+        )
+
+    payload = {
+        "accountId": effective_account_id,
+        "orderbookId": str(orderbook_id),
+        "side": side,
+        "condition": "NORMAL",
+        "price": price,
+        "validUntil": valid_until or date.today().isoformat(),
+        "volume": volume,
+    }
+    # 2026-04-13: cross-process lock — metals_loop + golddigger + fin_snipe
+    # must not race on buying_power. 2s fail-fast; busy peer aborts the order
+    # (caller retries next cycle).
+    with avanza_order_lock(op=f"place_order/{side}/{orderbook_id}"):
+        result = api_post("/_api/trading-critical/rest/order/new", payload)
+    status = result.get("orderRequestStatus", "UNKNOWN")
+    if status != "SUCCESS":
+        logger.warning("Order %s failed: %s — %s", side, status, result.get("message", ""))
+    else:
+        logger.info(
+            "Order %s placed: %dx @ %.3f SEK (id=%s)",
+            side, volume, price, result.get("orderId", "?"),
+        )
+    return result
+
+
+def cancel_order(order_id: str, account_id: str | None = None) -> dict:
+    """Cancel an open order.
+
+    IMPORTANT: Uses POST (not DELETE verb) — Avanza API change 2026-03-24.
+    """
+    payload = {
+        "accountId": str(account_id or DEFAULT_ACCOUNT_ID),
+        "orderId": str(order_id),
+    }
+    # 2026-04-13: cross-process order lock — cancel is a mutation, same
+    # concurrency concern as place_order (don't want two cancels racing).
+    with avanza_order_lock(op=f"cancel_order/{order_id}"):
+        return api_post("/_api/trading-critical/rest/order/delete", payload)
+
+
+def get_open_orders(account_id: str | None = None) -> list[dict]:
+    """Get all open (unfilled) orders for an account."""
+    aid = str(account_id or DEFAULT_ACCOUNT_ID)
+    try:
+        data = api_get(f"/_api/trading/rest/order/account/{aid}")
+        if isinstance(data, list):
+            return data
+        return data.get("orders", data.get("openOrders", []))
+    except RuntimeError:
+        # Endpoint may vary — fallback to deal endpoint
+        try:
+            data = api_get("/_api/trading/rest/deals-and-orders")
+            orders = data.get("orders", [])
+            return [o for o in orders if str(o.get("accountId", "")) == aid]
+        except RuntimeError:
+            logger.warning("Could not fetch open orders")
+            return []
+
+
+def get_quote(orderbook_id: str) -> dict:
+    """Get bid/ask/last quote for an instrument. Fast single-endpoint call.
+
+    Returns:
+        Dict with buy, sell, last, changePercent, highest, lowest.
+    """
+    return api_get(f"/_api/market-guide/stock/{orderbook_id}/quote")
+
+
+def get_positions() -> list[dict]:
+    """Get all positions via session-based auth.
+
+    Returns:
+        List of position dicts with name, value, profit, etc.
+    """
+    data = api_get("/_api/position-data/positions")
+    positions = []
+    for entry in data.get("withOrderbook", []):
+        inst = entry.get("instrument", {})
+        orderbook = inst.get("orderbook", {})
+        quote = orderbook.get("quote", {})
+        volume_obj = entry.get("volume", {})
+        value_obj = entry.get("value", {})
+        acquired_obj = entry.get("acquiredValue", {})
+        account = entry.get("account", {})
+
+        vol = volume_obj.get("value", 0) if isinstance(volume_obj, dict) else volume_obj
+        val = value_obj.get("value", 0) if isinstance(value_obj, dict) else value_obj
+        acq = acquired_obj.get("value", 0) if isinstance(acquired_obj, dict) else acquired_obj
+        latest = quote.get("latest", {})
+        last_price = latest.get("value", 0) if isinstance(latest, dict) else latest
+        change_pct_obj = quote.get("changePercent", {})
+        change_pct = change_pct_obj.get("value", 0) if isinstance(change_pct_obj, dict) else change_pct_obj
+
+        positions.append({
+            "name": inst.get("name", orderbook.get("name", "")),
+            "orderbook_id": str(orderbook.get("id", "")),
+            "instrument_id": str(inst.get("id", "")),
+            "type": inst.get("type", orderbook.get("type", "")),
+            "volume": vol,
+            "value": val,
+            "acquired_value": acq,
+            "profit": val - acq if val and acq else 0,
+            "profit_percent": ((val - acq) / acq * 100) if acq else 0,
+            "currency": inst.get("currency", "SEK"),
+            "last_price": last_price,
+            "change_percent": change_pct,
+            "account_id": account.get("id", ""),
+            "account_type": account.get("type", ""),
+        })
+    return positions
+
+
+def place_stop_loss(
+    orderbook_id: str,
+    trigger_price: float,
+    sell_price: float,
+    volume: int,
+    account_id: str | None = None,
+    valid_days: int = 8,
+    trigger_type: str = "LESS_OR_EQUAL",
+    value_type: str = "MONETARY",
+) -> dict:
+    """Place a hardware stop-loss order on Avanza.
+
+    IMPORTANT: Uses /_api/trading/stoploss/new, NOT the regular order API.
+
+    Args:
+        orderbook_id: Avanza orderbook ID.
+        trigger_price: Price at which to trigger the stop-loss.
+            For FOLLOW_DOWNWARDS with PERCENTAGE, this is the trail %.
+        sell_price: Price to sell at when triggered.
+            For trailing stops (FOLLOW_DOWNWARDS), set to 0 (market).
+        volume: Number of units to sell.
+        account_id: Defaults to DEFAULT_ACCOUNT_ID.
+        valid_days: Days until the stop-loss expires (default 8).
+        trigger_type: LESS_OR_EQUAL, MORE_OR_EQUAL, FOLLOW_DOWNWARDS, FOLLOW_UPWARDS.
+        value_type: MONETARY (absolute price) or PERCENTAGE.
+
+    Returns:
+        Dict with status, stoplossOrderId.
+    """
+    acct = str(account_id or DEFAULT_ACCOUNT_ID)
+    if acct not in ALLOWED_ACCOUNT_IDS:
+        raise ValueError(f"Refusing to place stop-loss on non-whitelisted account {acct!r}")
+    valid_until = (date.today() + timedelta(days=valid_days)).isoformat()
+
+    # BUG-223: trailing stops (FOLLOW_DOWNWARDS/UPWARDS) legitimately use
+    # sell_price=0 (market order on trigger). Non-trailing MONETARY stops
+    # must have sell_price > 0 — a zero sell_price would execute as a market
+    # sell at whatever price exists, potentially the worst available price.
+    _TRAILING_TYPES = {"FOLLOW_DOWNWARDS", "FOLLOW_UPWARDS"}
+    if trigger_type not in _TRAILING_TYPES and value_type == "MONETARY" and sell_price <= 0:
+        raise ValueError(
+            f"Non-trailing stop-loss requires sell_price > 0, got {sell_price}"
+        )
+
+    # 2026-04-17: stops below Avanza's 1000 SEK min-courtage threshold still
+    # succeed at the API but carry outsized fees. Cascaded-stop callers
+    # (metals_loop) can legitimately produce sub-1000 legs, so warn rather
+    # than raise — surface fee inefficiency without breaking live stops.
+    if value_type == "MONETARY" and sell_price > 0:
+        leg_total = round(volume * sell_price, 2)
+        if leg_total < 1000.0:
+            logger.warning(
+                "place_stop_loss leg %.2f SEK below 1000 SEK courtage threshold "
+                "(vol=%d sell=%.3f ob=%s)",
+                leg_total, volume, sell_price, orderbook_id,
+            )
+
+    payload = {
+        "parentStopLossId": "0",
+        "accountId": acct,
+        "orderBookId": str(orderbook_id),
+        "stopLossTrigger": {
+            "type": trigger_type,
+            "value": trigger_price,
+            "validUntil": valid_until,
+            "valueType": value_type,
+            "triggerOnMarketMakerQuote": True,
+        },
+        "stopLossOrderEvent": {
+            "type": "SELL",
+            "price": sell_price,
+            "volume": volume,
+            "validDays": valid_days,
+            "priceType": value_type,
+            "shortSellingAllowed": False,
+        },
+    }
+    # 2026-04-13: cross-process order lock. Stop-loss placement is
+    # especially race-sensitive because cancel-before-place flows are
+    # common (see user memory: cancel existing stop BEFORE placing new sell).
+    with avanza_order_lock(op=f"place_stop_loss/{orderbook_id}"):
+        result = api_post("/_api/trading/stoploss/new", payload)
+    status = result.get("status", "UNKNOWN")
+    if status == "SUCCESS":
+        logger.info(
+            "Stop-loss placed: %s trigger=%.3f sell=%.3f vol=%d (id=%s)",
+            trigger_type, trigger_price, sell_price, volume,
+            result.get("stoplossOrderId", "?"),
+        )
+    else:
+        logger.warning("Stop-loss failed: %s — %s", status, result)
+    return result
+
+
+def place_trailing_stop(
+    orderbook_id: str,
+    trail_percent: float,
+    volume: int,
+    account_id: str | None = None,
+    valid_days: int = 8,
+) -> dict:
+    """Place a hardware trailing stop-loss that Avanza manages automatically.
+
+    The stop follows the price downward by trail_percent%. If the instrument
+    drops trail_percent% from its peak since placement, the stop triggers a
+    market sell.
+
+    Args:
+        orderbook_id: Avanza orderbook ID.
+        trail_percent: Trailing distance as percentage (e.g. 5.0 for 5%).
+        volume: Number of units to sell.
+        account_id: Defaults to DEFAULT_ACCOUNT_ID.
+        valid_days: Days until the stop expires (default 8).
+
+    Returns:
+        Dict with status, stoplossOrderId.
+    """
+    return place_stop_loss(
+        orderbook_id=orderbook_id,
+        trigger_price=trail_percent,
+        sell_price=0,
+        volume=volume,
+        account_id=account_id,
+        valid_days=valid_days,
+        trigger_type="FOLLOW_DOWNWARDS",
+        value_type="PERCENTAGE",
+    )
+
+
+def get_stop_losses() -> list[dict]:
+    """Get all active stop-loss orders.
+
+    Returns ``[]`` on read failure for backward compatibility with
+    callers that treat empty as "nothing to monitor". Code that needs
+    to distinguish "no stops" from "could not read stops" must use
+    :func:`get_stop_losses_strict` instead — or a False return from
+    that function will leave the caller unable to make safety
+    decisions like cancel-before-sell.
+    """
+    try:
+        data = api_get("/_api/trading/stoploss")
+        return data if isinstance(data, list) else []
+    except RuntimeError:
+        logger.warning("Could not fetch stop-losses")
+        return []
+
+
+def get_stop_losses_strict() -> list[dict]:
+    """Get all active stop-loss orders, raising on any read failure.
+
+    Use this in safety-critical paths (e.g., before a sell) where
+    "could not read" must NOT be silently treated as "no stops exist".
+    A swallowed read error there would let the dependent sell proceed
+    against still-encumbered volume, producing the very
+    ``short.sell.not.allowed`` error this module exists to prevent.
+
+    Raises:
+        RuntimeError: if the underlying ``api_get`` call fails or
+            returns a non-list shape.
+    """
+    data = api_get("/_api/trading/stoploss")
+    if not isinstance(data, list):
+        raise RuntimeError(
+            f"Unexpected stop-loss response shape: {type(data).__name__}"
+        )
+    return data
+
+
+def cancel_stop_loss(stop_id: str, account_id: str | None = None) -> dict:
+    """Cancel a single stop-loss order by ID.
+
+    Idempotent: HTTP 404 (already gone) is treated as success since the
+    end-state is identical from the caller's perspective.
+
+    Uses DELETE /_api/trading/stoploss/{accountId}/{stopId}, which is the
+    correct endpoint per portfolio/avanza_control.py:206. Do NOT use the
+    regular order cancel API — it returns "crossing prices" errors for
+    stop-losses (March 3 incident).
+
+    Args:
+        stop_id: Avanza stop-loss ID (e.g. "A2^1773297348702^1346781").
+        account_id: Avanza account ID. Defaults to ``DEFAULT_ACCOUNT_ID``.
+
+    Returns:
+        Dict with keys ``status`` ("SUCCESS"/"FAILED"), ``http_status`` (int),
+        and ``stop_id`` (str). Errors that prevent the call from running
+        (network, missing CSRF, etc.) yield ``status="FAILED"`` with
+        ``http_status=0`` and an ``error`` key describing the cause.
+    """
+    if not stop_id:
+        return {"status": "FAILED", "http_status": 0, "stop_id": "", "error": "empty stop_id"}
+    acct = str(account_id or DEFAULT_ACCOUNT_ID)
+    try:
+        # 2026-04-13: cross-process order lock — SL cancel is mutating.
+        # See cancel_order / place_stop_loss for rationale.
+        with avanza_order_lock(op=f"cancel_stop_loss/{stop_id}"):
+            result = api_delete(f"/_api/trading/stoploss/{acct}/{stop_id}")
+    except Exception as exc:  # noqa: BLE001 — propagate as structured failure
+        logger.error("cancel_stop_loss(%s) raised: %s", stop_id, exc, exc_info=True)
+        return {"status": "FAILED", "http_status": 0, "stop_id": stop_id, "error": str(exc)}
+    http_status = int(result.get("http_status", 0)) if isinstance(result, dict) else 0
+    # 2xx = deleted; 404 = already gone (triggered/expired/cancelled). Both succeed.
+    ok = (200 <= http_status < 300) or http_status == 404
+    if ok:
+        logger.info("cancel_stop_loss(%s) -> %s", stop_id, http_status)
+    else:
+        logger.warning("cancel_stop_loss(%s) failed: http=%s result=%s", stop_id, http_status, result)
+    return {
+        "status": "SUCCESS" if ok else "FAILED",
+        "http_status": http_status,
+        "stop_id": stop_id,
+    }
+
+
+def cancel_all_stop_losses_for(
+    orderbook_id: str,
+    account_id: str | None = None,
+    max_wait: float = 3.0,
+    poll_interval: float = 0.5,
+) -> dict:
+    """Cancel every active stop-loss for ``orderbook_id`` and verify clearance.
+
+    The "verify" step is the critical part: Avanza's DELETE returns 200 OK
+    immediately, but the encumbered volume on the position is not released
+    until the SL actually disappears from the position view. Without polling,
+    a follow-up SELL still gets ``short.sell.not.allowed``. We therefore
+    re-query ``get_stop_losses_strict()`` every ``poll_interval`` seconds
+    until none remain for the target orderbook (or ``max_wait`` is exceeded).
+
+    **Fail-closed semantics**: if the stop-loss list cannot be read (network
+    error, 5xx, malformed response), the function returns ``status="FAILED"``
+    rather than silently treating "could not read" as "no stops exist".
+    A safety-critical caller deciding whether to proceed with a sell MUST
+    NOT be misled into believing the path is clear when reality is unknown.
+
+    The function is idempotent and safe to call when no SLs exist — it
+    short-circuits to ``status="SUCCESS"`` without any DELETE calls.
+
+    Args:
+        orderbook_id: Avanza orderbook ID to clear.
+        account_id: Account filter. ``None`` means accept any account.
+        max_wait: Maximum total wall-clock seconds to wait for clearance.
+        poll_interval: Seconds between re-query attempts.
+
+    Returns:
+        Dict with:
+            - ``status``: "SUCCESS" (cleared), "PARTIAL" (some cancelled, some
+              still showing after timeout), or "FAILED" (no cancels succeeded
+              and stops still present, OR the SL list could not be read).
+            - ``cancelled``: list of stop_ids the DELETE call accepted.
+            - ``remaining``: list of stop_ids still present after the wait.
+            - ``snapshot``: list of full stop-loss dicts that were present at
+              the start of the cancel sequence. Callers can use this to
+              **re-arm** identical stops if the dependent sell fails — the
+              cancel/sell sequence is otherwise rollbackable but leaves the
+              position naked on partial-completion failure.
+            - ``elapsed_seconds``: float, total time spent in this call.
+            - ``error``: optional, present only when ``status="FAILED"`` due
+              to a read error rather than cancel failures.
+    """
+    started = time.monotonic()
+    target_ob = str(orderbook_id)
+    aid_filter = str(account_id) if account_id is not None else None
+
+    def _filter_for_ob(stops: list[dict]) -> list[dict]:
+        out = []
+        for sl in stops:
+            if not isinstance(sl, dict):
+                continue
+            ob = (sl.get("orderbook") or {}).get("id")
+            if str(ob) != target_ob:
+                continue
+            if aid_filter is not None:
+                acct = (sl.get("account") or {}).get("id")
+                if str(acct) != aid_filter:
+                    continue
+            out.append(sl)
+        return out
+
+    # Initial fetch — fail closed on read errors. A safety-critical caller
+    # cannot tell "no stops" apart from "API down" without this distinction.
+    try:
+        all_stops = get_stop_losses_strict()
+    except Exception as exc:  # noqa: BLE001 — convert to structured failure
+        elapsed = time.monotonic() - started
+        logger.error(
+            "cancel_all_stop_losses_for(%s): cannot read stop-loss list: %s",
+            target_ob, exc,
+        )
+        return {
+            "status": "FAILED",
+            "cancelled": [],
+            "remaining": [],
+            "snapshot": [],
+            "elapsed_seconds": elapsed,
+            "error": f"read_error: {exc}",
+        }
+
+    initial = _filter_for_ob(all_stops)
+    if not initial:
+        return {
+            "status": "SUCCESS",
+            "cancelled": [],
+            "remaining": [],
+            "snapshot": [],
+            "elapsed_seconds": time.monotonic() - started,
+        }
+
+    # Snapshot full dicts before cancelling so a caller can re-arm if the
+    # dependent sell fails downstream. We deep-copy to insulate against any
+    # downstream mutation of the returned structure.
+    import copy as _copy
+    snapshot = [_copy.deepcopy(sl) for sl in initial]
+
+    # Issue cancels for every matching stop. Use the SL's own account id when
+    # available — Avanza's DELETE endpoint requires the account that owns the
+    # stop, which may differ from DEFAULT_ACCOUNT_ID for multi-account users.
+    cancelled: list[str] = []
+    for sl in initial:
+        sid = sl.get("id") or ""
+        if not sid:
+            continue
+        sl_acct = (sl.get("account") or {}).get("id") or account_id
+        result = cancel_stop_loss(sid, account_id=sl_acct)
+        if result.get("status") == "SUCCESS":
+            cancelled.append(sid)
+
+    # Poll until cleared or timeout. Re-query is also fail-closed — if the
+    # API stops responding mid-poll, treat the orderbook as "may still have
+    # stops" rather than declaring victory.
+    remaining: list[str] = []
+    poll_read_failed = False
+    while True:
+        try:
+            poll_stops = get_stop_losses_strict()
+        except Exception as exc:  # noqa: BLE001
+            logger.warning(
+                "cancel_all_stop_losses_for(%s): poll read failed: %s",
+                target_ob, exc,
+            )
+            poll_read_failed = True
+            # We don't know if the stops are gone. Fail closed.
+            remaining = [sl.get("id", "") for sl in initial if sl.get("id") and sl.get("id") not in cancelled]
+            break
+        still = _filter_for_ob(poll_stops)
+        remaining = [s.get("id", "") for s in still if s.get("id")]
+        if not remaining:
+            break
+        if (time.monotonic() - started) >= max_wait:
+            break
+        time.sleep(poll_interval)
+
+    elapsed = time.monotonic() - started
+
+    # CODEX-7 finding 1: critical filter — a DELETE-accepted id can still
+    # be in `remaining` if the verification poll observed it alive
+    # (broker rejected the cancel late, or the DELETE was acknowledged
+    # but never propagated). The set we expose to callers as the
+    # rollback set MUST be the VERIFIED-cleared set:
+    #     verified = cancelled - remaining
+    # Re-arming a stop that is still alive would create a duplicate
+    # at the broker, recreating the exact over-encumbered failure mode
+    # this whole module exists to prevent.
+    remaining_set = set(remaining)
+    cancelled = [c for c in cancelled if c not in remaining_set]
+
+    if not remaining and not poll_read_failed:
+        status = "SUCCESS"
+        logger.info(
+            "cancel_all_stop_losses_for(%s): cleared %d stops in %.2fs",
+            target_ob, len(cancelled), elapsed,
+        )
+    elif cancelled and not poll_read_failed:
+        status = "PARTIAL"
+        logger.warning(
+            "cancel_all_stop_losses_for(%s): PARTIAL — verified_cancelled=%s remaining=%s elapsed=%.2fs",
+            target_ob, cancelled, remaining, elapsed,
+        )
+    else:
+        status = "FAILED"
+        logger.error(
+            "cancel_all_stop_losses_for(%s): FAILED — cancelled=%s remaining=%s read_failed=%s",
+            target_ob, cancelled, remaining, poll_read_failed,
+        )
+        # When the verification poll failed, we don't actually know which
+        # DELETEs took effect. The list of DELETE-accepted ids is
+        # broker-acknowledged but NOT verified-cleared. Drop them all to
+        # be safe on the rollback side.
+        if poll_read_failed:
+            cancelled = []
+    return {
+        "status": status,
+        "cancelled": cancelled,
+        "remaining": remaining,
+        "snapshot": snapshot,
+        "elapsed_seconds": elapsed,
+    }
+
+
+def rearm_stop_losses_from_snapshot(snapshot: list[dict]) -> dict:
+    """Re-place stop-losses from the snapshot returned by
+    :func:`cancel_all_stop_losses_for`.
+
+    Used to roll back a cancel-then-sell sequence when the sell fails:
+    we cancelled the stops to clear the volume, the sell didn't go through,
+    and the position is now naked. Re-arming restores the original
+    protection so we are no worse off than before the attempt.
+
+    Notes on best-effort behavior:
+
+    - Each re-arm is independent. If one fails, the others still try.
+    - The new stop-loss IDs differ from the originals — Avanza issues
+      fresh IDs on every place. Callers tracking IDs in local state must
+      replace, not deduplicate.
+    - ``valid_days`` is computed from the snapshot's ``trigger.validUntil``
+      field where present, falling back to 8 days. The trigger semantics
+      and price/volume are preserved exactly.
+
+    Args:
+        snapshot: List of stop-loss dicts as returned in
+            ``cancel_all_stop_losses_for(...)["snapshot"]``.
+
+    Returns:
+        Dict with:
+            - ``status``: "SUCCESS" (all re-armed), "PARTIAL" (some failed),
+              "FAILED" (none succeeded), or "SUCCESS" (snapshot was empty).
+            - ``rearmed``: list of new stop_ids placed.
+            - ``failed``: list of original stop_ids that could not be re-armed.
+    """
+    if not snapshot:
+        return {"status": "SUCCESS", "rearmed": [], "failed": []}
+
+    rearmed: list[str] = []
+    failed: list[str] = []
+    today_iso = date.today()
+
+    for sl in snapshot:
+        if not isinstance(sl, dict):
+            continue
+        original_id = sl.get("id", "")
+        try:
+            ob_id = (sl.get("orderbook") or {}).get("id")
+            account = (sl.get("account") or {}).get("id")
+            trigger = sl.get("trigger") or {}
+            order = sl.get("order") or {}
+            trigger_value = trigger.get("value")
+            trigger_type = trigger.get("type", "LESS_OR_EQUAL")
+            value_type = trigger.get("valueType", "MONETARY")
+            sell_price = order.get("price")
+            volume = order.get("volume")
+
+            # Compute valid_days from validUntil if present, else default 8.
+            valid_days = 8
+            valid_until = trigger.get("validUntil")
+            if valid_until:
+                try:
+                    parsed = datetime.strptime(valid_until, "%Y-%m-%d").date()
+                    delta = (parsed - today_iso).days
+                    if delta > 0:
+                        valid_days = delta
+                except (ValueError, TypeError):
+                    pass
+
+            if not (ob_id and trigger_value is not None and sell_price is not None and volume):
+                logger.warning("rearm_stop_losses: snapshot entry missing fields: %s", sl)
+                failed.append(original_id)
+                continue
+
+            result = place_stop_loss(
+                orderbook_id=str(ob_id),
+                trigger_price=float(trigger_value),
+                sell_price=float(sell_price),
+                volume=int(volume),
+                account_id=account,
+                valid_days=valid_days,
+                trigger_type=str(trigger_type),
+                value_type=str(value_type),
+            )
+            if result.get("status") == "SUCCESS":
+                new_id = result.get("stoplossOrderId", "")
+                rearmed.append(new_id)
+                logger.info(
+                    "rearm_stop_losses: replaced %s -> %s (ob=%s vol=%s)",
+                    original_id, new_id, ob_id, volume,
+                )
+            else:
+                logger.warning(
+                    "rearm_stop_losses: place_stop_loss failed for original %s: %s",
+                    original_id, result,
+                )
+                failed.append(original_id)
+        except Exception as exc:  # noqa: BLE001
+            logger.error(
+                "rearm_stop_losses: exception for original %s: %s",
+                original_id, exc, exc_info=True,
+            )
+            failed.append(original_id)
+
+    if not failed:
+        status = "SUCCESS"
+    elif rearmed:
+        status = "PARTIAL"
+    else:
+        status = "FAILED"
+    return {"status": status, "rearmed": rearmed, "failed": failed}
+
+
+def get_instrument_price(orderbook_id: str) -> dict[str, Any]:
+    """Get price info for a specific instrument.
+
+    Args:
+        orderbook_id: Avanza orderbook ID (numeric string)
+
+    Returns:
+        Dict with lastPrice, changePercent, etc.
+    """
+    # Try stock first, then fund, then certificate/warrant
+    for instrument_type in ("stock", "certificate", "fund", "exchange_traded_fund"):
+        try:
+            data = api_get(
+                f"/_api/market-guide/{instrument_type}/{orderbook_id}",
+            )
+            return data
+        except Exception as e:
+            logger.warning("Market guide lookup failed for %s/%s: %s", instrument_type, orderbook_id, e)
+            continue
+
+    # Fallback: generic orderbook endpoint
+    return api_get(f"/_api/orderbook/{orderbook_id}")
diff --git a/portfolio/avanza_tracker.py b/portfolio/avanza_tracker.py
new file mode 100644
index 00000000..f0fabf26
--- /dev/null
+++ b/portfolio/avanza_tracker.py
@@ -0,0 +1,132 @@
+"""Avanza-tracked instruments: Nordic stocks (price-only) and warrants (underlying signals).
+
+Tier 2 (Nordic equities): Price + P&L only via Avanza API. No technical signals.
+Tier 3 (Warrants): Warrant price via Avanza + underlying ticker's signals for decisions.
+
+Configuration lives in config.json under "avanza.instruments":
+    {
+        "avanza": {
+            "instruments": {
+                "SAAB-B": {"orderbook_id": "5533", "type": "equity", "name": "SAAB B"},
+                "MINI-SILVER": {"orderbook_id": "2345", "type": "warrant", "name": "MINI L SILVER AVA 140", "underlying": "XAG-USD"}
+            }
+        }
+    }
+"""
+
+import logging
+from pathlib import Path
+from typing import Any
+
+from portfolio.file_utils import load_json
+
+logger = logging.getLogger("portfolio.avanza_tracker")
+
+BASE_DIR = Path(__file__).resolve().parent.parent
+CONFIG_FILE = BASE_DIR / "config.json"
+
+
+def load_avanza_instruments() -> dict[str, dict]:
+    """Load Avanza instrument config from config.json.
+
+    Returns:
+        Dict of {config_key: instrument_config} or empty dict if not configured.
+    """
+    config = load_json(CONFIG_FILE, default={})
+    if not config:
+        return {}
+    return config.get("avanza", {}).get("instruments", {})
+
+
+def fetch_avanza_prices() -> dict[str, dict[str, Any]]:
+    """Fetch current prices for all configured Avanza instruments.
+
+    Returns:
+        Dict of {config_key: {"name": str, "price_sek": float, "change_pct": float, "type": str}}
+        Skips instruments with missing or empty orderbook_id.
+    """
+    instruments = load_avanza_instruments()
+    if not instruments:
+        return {}
+
+    try:
+        from portfolio.avanza_client import get_price
+    except Exception as e:
+        logger.debug("avanza_client not available: %s", e)
+        return {}
+
+    results = {}
+    for key, cfg in instruments.items():
+        ob_id = cfg.get("orderbook_id", "")
+        if not ob_id:
+            continue
+        try:
+            info = get_price(ob_id)
+            results[key] = {
+                "name": cfg.get("name", key),
+                "price_sek": float(info.get("lastPrice", 0)),
+                "change_pct": float(info.get("changePercent", 0)),
+                "type": cfg.get("type", "equity"),
+                "underlying": cfg.get("underlying"),
+            }
+        except Exception as e:
+            logger.warning("Price fetch failed for %s: %s", key, e)
+    return results
+
+
+def get_warrant_underlying(config_key: str) -> str | None:
+    """Get the underlying ticker for a warrant instrument.
+
+    Args:
+        config_key: The config key (e.g., "MINI-SILVER")
+
+    Returns:
+        Underlying ticker (e.g., "XAG-USD") or None if not a warrant.
+    """
+    instruments = load_avanza_instruments()
+    cfg = instruments.get(config_key, {})
+    if cfg.get("type") != "warrant":
+        return None
+    return cfg.get("underlying")
+
+
+def get_all_underlyings() -> dict[str, str]:
+    """Get mapping of all warrant config keys to their underlying tickers.
+
+    Returns:
+        Dict of {config_key: underlying_ticker} for all warrants.
+    """
+    instruments = load_avanza_instruments()
+    return {
+        key: cfg["underlying"]
+        for key, cfg in instruments.items()
+        if cfg.get("type") == "warrant" and cfg.get("underlying")
+    }
+
+
+def check_session_expiry() -> str | None:
+    """Check if Avanza BankID session is expired or expiring soon.
+
+    Returns:
+        Warning message string if session needs refresh, None if OK.
+    """
+    try:
+        from portfolio.avanza_session import (
+            is_session_expiring_soon,
+            session_remaining_minutes,
+        )
+    except ImportError:
+        return None
+
+    remaining = session_remaining_minutes()
+    if remaining is None:
+        return "Avanza session not found. Run: python scripts/avanza_login.py"
+    if remaining <= 0:
+        return "Avanza session expired. Run: python scripts/avanza_login.py"
+    if is_session_expiring_soon(threshold_minutes=60.0):
+        mins = int(remaining)
+        return (
+            f"Avanza session expires in {mins}min. "
+            "Run: python scripts/avanza_login.py"
+        )
+    return None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-avanza-api diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 580ms:
portfolio/avanza/__init__.py
portfolio/avanza/account.py
portfolio/avanza/auth.py
portfolio/avanza/client.py
portfolio/avanza/market_data.py
portfolio/avanza/scanner.py
portfolio/avanza/search.py
portfolio/avanza/streaming.py
portfolio/avanza/tick_rules.py
portfolio/avanza/trading.py
portfolio/avanza/types.py
portfolio/avanza_client.py
portfolio/avanza_control.py
portfolio/avanza_order_lock.py
portfolio/avanza_orders.py
portfolio/avanza_resilient_page.py
portfolio/avanza_session.py
portfolio/avanza_tracker.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@('portfolio/avanza/__init__.py','portfolio/avanza/account.py','portfolio/avanza/auth.py','portfolio/avanza/client.py','portfolio/avanza/market_data.py','portfolio/avanza/scanner.py','portfolio/avanza/search.py','portfolio/avanza/streaming.py','portfolio/avanza/tick_rules.py','portfolio/avanza/trading.py','portfolio/avanza/types.py','portfolio/avanza_client.py','portfolio/avanza_control.py','portfolio/avanza_order_lock.py','portfolio/avanza_orders.py','portfolio/avanza_resilient_page.py','portfolio/avanza_session.py','portfolio/avanza_tracker.py') | ForEach-Object { "'$count=(Get-Content $_).Count; Write-Output "$count`t$_" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 686ms:
44	portfolio/avanza/__init__.py
149	portfolio/avanza/account.py
121	portfolio/avanza/auth.py
133	portfolio/avanza/client.py
151	portfolio/avanza/market_data.py
375	portfolio/avanza/scanner.py
83	portfolio/avanza/search.py
251	portfolio/avanza/streaming.py
135	portfolio/avanza/tick_rules.py
364	portfolio/avanza/trading.py
530	portfolio/avanza/types.py
397	portfolio/avanza_client.py
439	portfolio/avanza_control.py
100	portfolio/avanza_order_lock.py
416	portfolio/avanza_orders.py
231	portfolio/avanza_resilient_page.py
1243	portfolio/avanza_session.py
132	portfolio/avanza_tracker.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_client.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_control.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_order_lock.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_orders.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/market_data.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/trading.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/types.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/streaming.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 972ms:
"""Canonical Avanza control facade for reads, quotes, and browser-session trades.

Use this module as the shared import path for Avanza operations in strategy code.
It keeps the currently working Playwright-page execution path for metals/gold
while exposing the broader account/session helpers from ``portfolio.avanza_*``.
"""

from __future__ import annotations

import json
import logging

from portfolio.avanza_order_lock import avanza_order_lock

logger = logging.getLogger("portfolio.avanza_control")

from data.metals_avanza_helpers import (
    check_session_alive,
    get_csrf,
)
from data.metals_avanza_helpers import (
    fetch_account_cash as _fetch_account_cash,
)
from data.metals_avanza_helpers import (
    fetch_positions as _fetch_page_positions,
)
from data.metals_avanza_helpers import (
    fetch_price as _fetch_page_price,
)
from data.metals_avanza_helpers import (
    place_order as _place_page_order,
)
from data.metals_avanza_helpers import (
    place_stop_loss as _place_page_stop_loss,
)
from portfolio.avanza_client import (
    delete_order,
    find_instrument,
    get_account_id,
    get_open_orders,
    get_portfolio_value,
    get_positions,
    place_buy_order,
    place_sell_order,
)
from portfolio.avanza_client import (
    get_price as get_price_info,
)

_TYPE_ALIASES = {
    "cert": "certificate",
    "certifikat": "certificate",
    "certificate": "certificate",
    "warrant": "warrant",
    "mini": "warrant",
    "mini-future": "warrant",
    "mini_future": "warrant",
    "stock": "stock",
    "share": "stock",
    "fund": "fund",
    "etf": "exchange_traded_fund",
    "exchange_traded_fund": "exchange_traded_fund",
}

_PRICE_FALLBACK_TYPES = (
    "certificate",
    "warrant",
    "stock",
    "exchange_traded_fund",
    "fund",
)


def normalize_api_type(api_type: str | None, default: str = "certificate") -> str:
    """Normalize Avanza instrument type names for market-guide lookups."""
    normalized = (api_type or "").strip().lower()
    if not normalized:
        return default
    return _TYPE_ALIASES.get(normalized, normalized)


def fetch_price(page, orderbook_id: str, api_type: str = "certificate"):
    """Fetch a quote from the market-guide API using an authenticated page."""
    return _fetch_page_price(page, orderbook_id, normalize_api_type(api_type))


def fetch_price_with_fallback(page, orderbook_id: str, api_type: str | None = None):
    """Try the preferred market-guide type and then the common fallback types."""
    if not orderbook_id:
        return None

    candidates: list[str] = []
    preferred = normalize_api_type(api_type) if api_type else ""
    if preferred:
        candidates.append(preferred)
    for fallback in _PRICE_FALLBACK_TYPES:
        if fallback not in candidates:
            candidates.append(fallback)

    for candidate in candidates:
        data = fetch_price(page, orderbook_id, candidate)
        if not data:
            continue
        if data.get("bid") is None and data.get("ask") is None and data.get("last") is None:
            continue
        payload = dict(data)
        payload["api_type"] = candidate
        return payload
    return None


def fetch_account_cash(page, account_id: str | None = None):
    """Fetch buying power for an account via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    return _fetch_account_cash(page, resolved_account_id)


def fetch_page_positions(page, account_id: str | None = None):
    """Fetch current positions keyed by orderbook id via the page session.

    Returns dict[ob_id -> {name, units, value, avg_price, api_type}] on
    success, or None on transient failure. An empty dict `{}` is a valid
    response meaning the account is flat — callers should distinguish it
    from None.
    """
    resolved_account_id = str(account_id or get_account_id())
    return _fetch_page_positions(page, resolved_account_id)


def place_order(page, account_id: str | None, ob_id: str, side: str, price: float, volume: int):
    """Place a BUY/SELL order via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    normalized_side = (side or "").strip().upper()
    return _place_page_order(page, resolved_account_id, ob_id, normalized_side, price, volume)


def place_stop_loss(
    page,
    account_id: str | None,
    ob_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    valid_days: int = 8,
):
    """Place a hardware stop-loss order via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    return _place_page_stop_loss(
        page,
        resolved_account_id,
        ob_id,
        trigger_price,
        sell_price,
        volume,
        valid_days=valid_days,
    )


def delete_order_live(page, account_id: str | None, order_id: str):
    """Cancel an open order via the authenticated page session.

    IMPORTANT: Uses POST to /_api/trading-critical/rest/order/delete with
    JSON body {accountId, orderId}. The DELETE HTTP verb to
    /_api/trading-critical/rest/order/{accountId}/{orderId} returns 404
    (Avanza API change discovered 2026-03-24).
    """
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    resolved_account_id = str(account_id or get_account_id())
    try:
        # 2026-04-13: cross-process order lock (see metals_avanza_helpers.place_order).
        with avanza_order_lock(op=f"delete_order_live/{order_id}"):
            result = page.evaluate(
                """async (args) => {
                    const [accountId, orderId, token] = args;
                    const resp = await fetch(
                        'https://www.avanza.se/_api/trading-critical/rest/order/delete',
                        {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-SecurityToken': token,
                            },
                            credentials: 'include',
                            body: JSON.stringify({accountId: accountId, orderId: orderId}),
                        }
                    );
                    return {status: resp.status, body: await resp.text()};
                }""",
                [resolved_account_id, order_id, csrf],
            )
        http_status = int(result.get("status") or 0)
        body_text = result.get("body", "")
        parsed = {}
        try:
            if body_text:
                parsed = json.loads(body_text)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        success = parsed.get("orderRequestStatus") == "SUCCESS"
        return success, {
            "http_status": http_status,
            "parsed": parsed,
            "body": body_text,
        }
    except Exception as exc:
        logger.error("Delete order failed for order %s: %s", order_id, exc, exc_info=True)
        return False, {"error": str(exc)}


def delete_stop_loss(page, account_id: str | None, stop_id: str):
    """Delete an existing Avanza stop-loss order via the authenticated page."""
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    resolved_account_id = str(account_id or get_account_id())
    try:
        # 2026-04-13: cross-process order lock. SL delete is mutating.
        with avanza_order_lock(op=f"delete_stop_loss/{stop_id}"):
            result = page.evaluate(
                """async (args) => {
                    const [accountId, stopId, token] = args;
                    const resp = await fetch(
                        'https://www.avanza.se/_api/trading/stoploss/' + accountId + '/' + stopId,
                        {
                            method: 'DELETE',
                            headers: {'X-SecurityToken': token},
                            credentials: 'include',
                        }
                    );
                    return {status: resp.status, body: await resp.text()};
                }""",
                [resolved_account_id, stop_id, csrf],
            )
        http_status = int(result.get("status") or 0)
        # 2xx = deleted successfully.  404 = stop already gone (triggered/expired/cancelled).
        # Both mean the stop no longer exists, which is the goal of a cancel.
        success = (200 <= http_status < 300) or http_status == 404
        body_text = result.get("body", "")
        parsed = {}
        try:
            if body_text:
                parsed = json.loads(body_text)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        return success, {
            "http_status": http_status,
            "parsed": parsed,
            "body": body_text,
        }
    except Exception as exc:
        logger.error("Delete stop-loss failed for stop %s: %s", stop_id, exc, exc_info=True)
        return False, {"error": str(exc)}



# --- Page-free API (uses BankID session, no Playwright page needed) ---

from portfolio.avanza_session import (
    api_delete as _api_delete,
)
from portfolio.avanza_session import (
    api_get as _api_get,
)
from portfolio.avanza_session import (
    cancel_order as _cancel_order,
)
from portfolio.avanza_session import (
    place_buy_order as _place_buy_order,
)
from portfolio.avanza_session import (
    place_sell_order as _place_sell_order,
)
from portfolio.avanza_session import (
    place_stop_loss as _place_stop_loss_session,
)
from portfolio.avanza_session import (
    place_trailing_stop as _place_trailing_stop_session,
)
from portfolio.avanza_session import (
    verify_session,
)


def fetch_price_no_page(orderbook_id: str, api_type: str = "certificate"):
    """Fetch a quote without a Playwright page — uses BankID session API."""
    normalized = normalize_api_type(api_type)
    try:
        data = _api_get(f"/_api/market-guide/{normalized}/{orderbook_id}")
        quote = data.get("quote", {})
        ki = data.get("keyIndicators", {})
        underlying = data.get("underlying", {})
        def _v(obj):
            return obj.get("value") if isinstance(obj, dict) else obj
        return {
            "bid": _v(quote.get("buy")),
            "ask": _v(quote.get("sell")),
            "last": _v(quote.get("last")),
            "change_pct": _v(quote.get("changePercent")),
            "high": _v(quote.get("highest")),
            "low": _v(quote.get("lowest")),
            "underlying": _v(underlying.get("quote", {}).get("last")),
            "underlying_name": underlying.get("name"),
            "leverage": _v(ki.get("leverage")),
            "barrier": _v(ki.get("barrierLevel")),
            "api_type": normalized,
        }
    except Exception as e:
        logger.error("Warrant price fetch failed for orderbook %s: %s", orderbook_id, e, exc_info=True)
        return None


def fetch_price_no_page_with_fallback(orderbook_id: str, api_type: str | None = None):
    """Try preferred type then fallback chain — no Playwright page needed."""
    if not orderbook_id:
        return None
    candidates = []
    preferred = normalize_api_type(api_type) if api_type else ""
    if preferred:
        candidates.append(preferred)
    for fb in _PRICE_FALLBACK_TYPES:
        if fb not in candidates:
            candidates.append(fb)
    for candidate in candidates:
        data = fetch_price_no_page(orderbook_id, candidate)
        if data and (data.get("bid") is not None or data.get("ask") is not None or data.get("last") is not None):
            return data
    return None


def place_order_no_page(account_id, ob_id, side, price, volume):
    """Place BUY/SELL via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.

    Raises:
        ValueError: If *side* is not "BUY" or "SELL" (C2 fail-safe).
    """
    normalized_side = (side or "").strip().upper()
    if normalized_side not in ("BUY", "SELL"):
        raise ValueError(
            f"Invalid order side: {side!r} (must be 'BUY' or 'SELL')"
        )
    if normalized_side == "BUY":
        result = _place_buy_order(ob_id, price, volume, account_id)
    else:
        result = _place_sell_order(ob_id, price, volume, account_id)
    ok = result.get("orderRequestStatus") == "SUCCESS"
    return ok, result


def place_stop_loss_no_page(account_id, ob_id, trigger_price, sell_price, volume, valid_days=8):
    """Hardware stop-loss via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _place_stop_loss_session(ob_id, trigger_price, sell_price, volume, account_id, valid_days)
    ok = result.get("status") == "SUCCESS"
    return ok, result


def place_trailing_stop_no_page(account_id, ob_id, trail_percent, volume, valid_days=8):
    """Hardware trailing stop via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _place_trailing_stop_session(ob_id, trail_percent, volume, account_id, valid_days)
    ok = result.get("status") == "SUCCESS"
    return ok, result


def delete_order_no_page(account_id, order_id):
    """Cancel order via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _cancel_order(order_id, account_id)
    ok = result.get("orderRequestStatus") == "SUCCESS"
    return ok, result


def delete_stop_loss_no_page(account_id, stop_id):
    """Delete stop-loss via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    resolved_account_id = str(account_id or get_account_id())
    try:
        result = _api_delete(f"/_api/trading/stoploss/{resolved_account_id}/{stop_id}")
        # H18: Check for error indicators in the response.
        # API returns {} on success (200 with empty body).
        # A non-empty response with error keys indicates failure.
        if isinstance(result, dict) and result.get("errorCode"):
            logger.warning("Delete stop-loss returned error for stop %s: %s", stop_id, result)
            return False, result
        return True, result
    except Exception as e:
        logger.error("Delete stop-loss (no page) failed for stop %s: %s", stop_id, e, exc_info=True)
        return False, {"error": str(e)}


__all__ = [
    "check_session_alive",
    "delete_order",
    "delete_order_live",
    "delete_order_no_page",
    "delete_stop_loss",
    "delete_stop_loss_no_page",
    "fetch_account_cash",
    "fetch_page_positions",
    "fetch_price",
    "fetch_price_no_page",
    "fetch_price_no_page_with_fallback",
    "fetch_price_with_fallback",
    "find_instrument",
    "get_account_id",
    "get_csrf",
    "get_open_orders",
    "get_portfolio_value",
    "get_positions",
    "get_price_info",
    "normalize_api_type",
    "place_buy_order",
    "place_order",
    "place_order_no_page",
    "place_sell_order",
    "place_stop_loss",
    "place_stop_loss_no_page",
    "place_trailing_stop_no_page",
    "verify_session",
]

 succeeded in 943ms:
"""Market data retrieval — quotes, depth, OHLC, instrument info, news.

Thin typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
raw delegators.  Every function returns our own dataclasses from
:mod:`portfolio.avanza.types`.
"""

from __future__ import annotations

import logging
from typing import Any

from avanza.constants import Resolution, TimePeriod

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import (
    OHLC,
    InstrumentInfo,
    MarketData,
    NewsArticle,
    Quote,
)

logger = logging.getLogger("portfolio.avanza.market_data")

# ---------------------------------------------------------------------------
# Resolution lookup (period -> sensible default resolution)
# ---------------------------------------------------------------------------

_DEFAULT_RESOLUTION: dict[str, Resolution] = {
    "TODAY": Resolution.THIRTY_MINUTES,
    "ONE_WEEK": Resolution.THIRTY_MINUTES,
    "ONE_MONTH": Resolution.DAY,
    "THREE_MONTHS": Resolution.DAY,
    "THIS_YEAR": Resolution.WEEK,
    "ONE_YEAR": Resolution.WEEK,
    "THREE_YEARS": Resolution.MONTH,
    "FIVE_YEARS": Resolution.MONTH,
    "INFINITY": Resolution.MONTH,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_quote(ob_id: str, instrument_type: str = "certificate") -> Quote:
    """Fetch a live quote for the given orderbook ID.

    Calls ``client.avanza.get_instrument(type, id)`` and parses the
    result into a :class:`~portfolio.avanza.types.Quote`.
    """
    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.avanza.get_instrument(instrument_type, ob_id)
    logger.debug("get_quote ob_id=%s raw_keys=%s", ob_id, list(raw.keys()))
    return Quote.from_api(raw)


def get_market_data(ob_id: str) -> MarketData:
    """Fetch full market data (quote + depth + recent trades).

    Calls ``client.get_market_data_raw(id)`` and parses the result
    into a :class:`~portfolio.avanza.types.MarketData`.
    """
    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.get_market_data_raw(ob_id)
    logger.debug("get_market_data ob_id=%s raw_keys=%s", ob_id, list(raw.keys()))
    return MarketData.from_api(raw)


def get_ohlc(
    ob_id: str,
    period: str = "ONE_MONTH",
    resolution: str | None = None,
) -> list[OHLC]:
    """Fetch OHLCV candles for the given orderbook ID.

    Args:
        ob_id: Avanza orderbook ID.
        period: Time period string (e.g. ``"ONE_MONTH"``, ``"ONE_WEEK"``).
        resolution: Optional resolution override.  When *None* a sensible
            default is chosen based on *period*.

    Returns:
        List of :class:`~portfolio.avanza.types.OHLC` candles.
    """
    client = AvanzaClient.get_instance()

    tp = TimePeriod[period]
    if resolution is not None:
        res = Resolution[resolution]
    else:
        res = _DEFAULT_RESOLUTION.get(period, Resolution.DAY)

    raw: Any = client.avanza.get_chart_data(ob_id, tp, res)
    logger.debug(
        "get_ohlc ob_id=%s period=%s resolution=%s candles=%d",
        ob_id,
        period,
        res.name,
        len(raw) if isinstance(raw, list) else 0,
    )

    # The API may return a dict with an "ohlc" key or a plain list.
    candles: list[dict[str, Any]]
    if isinstance(raw, dict):
        candles = raw.get("ohlc", raw.get("dataPoints", []))
    elif isinstance(raw, list):
        candles = raw
    else:
        candles = []

    return [OHLC.from_api(c) for c in candles]


def get_instrument_info(
    ob_id: str,
    instrument_type: str = "certificate",
) -> InstrumentInfo:
    """Fetch instrument metadata (leverage, barrier, underlying, etc.).

    Calls ``client.avanza.get_instrument(type, id)`` and parses the
    result into a :class:`~portfolio.avanza.types.InstrumentInfo`.
    """
    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.avanza.get_instrument(instrument_type, ob_id)
    logger.debug("get_instrument_info ob_id=%s name=%s", ob_id, raw.get("name"))
    return InstrumentInfo.from_api(raw)


def get_news(ob_id: str) -> list[NewsArticle]:
    """Fetch news articles linked to the given orderbook ID.

    Calls ``client.get_news_raw(id)`` and parses the result into a
    list of :class:`~portfolio.avanza.types.NewsArticle`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_news_raw(ob_id)
    logger.debug("get_news ob_id=%s", ob_id)

    # The API may return a list directly or a dict with an "articles" key.
    articles: list[dict[str, Any]]
    if isinstance(raw, dict):
        articles = raw.get("articles", raw.get("news", []))
    elif isinstance(raw, list):
        articles = raw
    else:
        articles = []

    return [NewsArticle.from_api(a) for a in articles]

 succeeded in 977ms:
"""Avanza API client for portfolio monitoring and trading.

Supports two authentication methods:
1. BankID session (preferred) — captured by scripts/avanza_login.py, stored in
   data/avanza_session.json. No credentials needed, valid ~24h.
2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
   from config.json.

The client transparently tries BankID session first, then falls back to TOTP.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any

from portfolio.avanza_order_lock import avanza_order_lock
from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.avanza_client")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"

# A-AV-2 (2026-04-11): Hardcoded account whitelist. The TOTP path scans for
# any account whose accountType contains "ISK", which means a future Avanza
# response containing a *new* ISK-shaped account (e.g. a child's ISK or a
# corporate ISK) could be picked up as the trading account. Worse, the
# pension account (2674244) was previously reachable through some code paths
# without filtering. Mirror the ALLOWED_ACCOUNT_IDS pattern from
# avanza_session.py: anything not in this set is rejected, period.
ALLOWED_ACCOUNT_IDS: set[str] = {"1625505"}

# Singleton client instance (avanza-api library)
_client = None
# Cached signal that a BankID Playwright session has already been verified.
_session_client = None


def _load_credentials() -> dict:
    """Load Avanza credentials from config.json.

    Returns:
        dict with keys: username, password, totp_secret

    Raises:
        FileNotFoundError: if config.json does not exist
        KeyError: if 'avanza' section is missing or credentials incomplete
    """
    config = load_json(CONFIG_FILE)
    if config is None:
        raise FileNotFoundError(f"Config file not found or unreadable: {CONFIG_FILE}")
    if "avanza" not in config:
        raise KeyError(
            "Missing 'avanza' section in config.json. "
            "Add: {\"avanza\": {\"username\": \"...\", \"password\": \"...\", \"totp_secret\": \"...\"}}"
        )
    creds = config["avanza"]
    for key in ("username", "password", "totp_secret"):
        if key not in creds or not creds[key]:
            raise KeyError(f"Missing or empty 'avanza.{key}' in config.json")
    return creds


def _try_session_auth() -> bool:
    """Return True when a BankID-backed Playwright session is available."""
    global _session_client
    if _session_client is True:
        return True
    try:
        from portfolio.avanza_session import verify_session
        if verify_session():
            _session_client = True
            logger.info("Using BankID session for Avanza API")
            return True
        logger.info("BankID session exists but verification failed")
    except Exception as e:
        logger.debug("BankID session not available: %s", e)
    return False


def get_client():
    """Get or create a singleton Avanza client.

    Tries BankID session first, then falls back to TOTP credentials.

    Returns:
        Authenticated Avanza client instance (avanza-api library)

    Raises:
        Exception: if neither auth method works
    """
    global _client
    if _client is not None:
        return _client
    try:
        from avanza import Avanza
    except ImportError:
        raise ImportError(
            "avanza-api package not installed. Run: pip install avanza-api"
        ) from None
    creds = _load_credentials()
    _client = Avanza({
        "username": creds["username"],
        "password": creds["password"],
        "totpSecret": creds["totp_secret"],
    })
    return _client


def reset_client() -> None:
    """Reset the singleton TOTP client (useful for re-authentication)."""
    global _client
    _client = None


def reset_session() -> None:
    """Reset the cached BankID session verification flag."""
    global _session_client
    _session_client = None


def find_instrument(query: str) -> list[dict]:
    """Search for instruments by name or ticker.

    Args:
        query: Search string (e.g., 'Bitcoin', 'NVDA', 'Silver')

    Returns:
        List of matching instruments with id, name, and type
    """
    client = get_client()
    results = client.search_for_stock(query)
    return results


def get_price(orderbook_id: str) -> dict[str, Any]:
    """Get current price and info for an instrument.

    Tries BankID session first, then falls back to TOTP client.

    Args:
        orderbook_id: Avanza orderbook ID (numeric string)

    Returns:
        Dict with price info including lastPrice, change, changePercent, etc.
    """
    # Try session-based auth first
    if _try_session_auth():
        try:
            from portfolio.avanza_session import get_instrument_price
            return get_instrument_price(orderbook_id)
        except Exception as e:
            logger.warning("Session-based price fetch failed, trying TOTP: %s", e)
            reset_session()

    client = get_client()
    info = client.get_stock_info(orderbook_id)
    return info


def get_positions() -> list[dict]:
    """Get all current positions from the Avanza account.

    Tries BankID session first, then falls back to TOTP client.

    Returns:
        List of position dicts, each with name, value, profit, etc.
        Returns empty list if no positions or on error.
    """
    # Try session-based auth first
    if _try_session_auth():
        try:
            from portfolio.avanza_session import get_positions as session_get_positions
            return session_get_positions()
        except Exception as e:
            logger.warning("Session-based positions fetch failed, trying TOTP: %s", e)
            reset_session()

    client = get_client()
    overview = client.get_overview()
    positions = []
    # A-AV-2: only return positions from whitelisted accounts so the pension
    # account (or any future-added account) never leaks into the trading view.
    for account in overview.get("accounts", []):
        if str(account.get("accountId", "")) not in ALLOWED_ACCOUNT_IDS:
            continue
        for pos in account.get("positions", []):
            positions.append({
                "account": account.get("name", ""),
                "account_id": account.get("accountId", ""),
                "name": pos.get("name", ""),
                "ticker": pos.get("orderbookId", ""),
                "volume": pos.get("volume", 0),
                "value": pos.get("value", 0),
                "profit": pos.get("profit", 0),
                "profit_percent": pos.get("profitPercent", 0),
                "currency": pos.get("currency", "SEK"),
            })
    return positions


def get_portfolio_value() -> float:
    """Get total portfolio value in SEK for whitelisted Avanza accounts only.

    A-AV-2: Filters to ALLOWED_ACCOUNT_IDS so pension/other-account values
    never inflate the "trading" portfolio value used for sizing.

    Returns:
        Total portfolio value in SEK across whitelisted accounts
    """
    client = get_client()
    overview = client.get_overview()
    total = 0.0
    for account in overview.get("accounts", []):
        if str(account.get("accountId", "")) not in ALLOWED_ACCOUNT_IDS:
            continue
        total += account.get("totalValue", 0)
    return total


def get_open_orders() -> list:
    """Return open orders for the ISK account (read-only).

    Uses the authenticated Avanza client; does not place or cancel orders.
    """
    client = get_client()
    account_id = get_account_id()
    try:
        orders = client.get_orders(account_id)
    except Exception as e:
        logger.error("Failed to fetch open orders: %s", e)
        raise
    return orders or []


# --- Account ID ---

_account_id: str | None = None


def get_account_id() -> str:
    """Get the trading account ID from Avanza overview (cached).

    Scans accounts of type ISK and returns the first one *that is also in
    ALLOWED_ACCOUNT_IDS*. Any ISK account not in the whitelist is rejected.
    This prevents accidental trades on a future-added child ISK, corporate
    ISK, or any account Avanza re-orders into the response.

    Returns:
        Account ID string (guaranteed to be in ALLOWED_ACCOUNT_IDS)

    Raises:
        RuntimeError: if no whitelisted ISK account is found
    """
    global _account_id
    if _account_id is not None:
        return _account_id
    client = get_client()
    overview = client.get_overview()
    seen_ids: list[str] = []
    for account in overview.get("accounts", []):
        atype = account.get("accountType", "")
        if "ISK" not in atype.upper():
            continue
        candidate = str(account.get("accountId", ""))
        seen_ids.append(candidate)
        # A-AV-2: enforce whitelist BEFORE caching, so a rogue first-call
        # cannot poison the singleton with a non-whitelisted account.
        if candidate in ALLOWED_ACCOUNT_IDS:
            _account_id = candidate
            logger.info("Found whitelisted ISK account: %s", _account_id)
            return _account_id
    raise RuntimeError(
        "No whitelisted ISK account found in Avanza overview. "
        f"ISK account IDs seen: {seen_ids}. Whitelist: {sorted(ALLOWED_ACCOUNT_IDS)}. "
        "If this is a legitimate new account, update ALLOWED_ACCOUNT_IDS in "
        "portfolio/avanza_client.py — never trade on auto-discovered accounts."
    )


# --- Trading functions ---


def place_buy_order(
    orderbook_id: str,
    price: float,
    volume: int,
    valid_until: date | None = None,
) -> dict:
    """Place a limit BUY order on Avanza.

    Args:
        orderbook_id: Avanza orderbook ID for the instrument
        price: Limit price in SEK
        volume: Number of shares (must be int >= 1)
        valid_until: Order expiry date. Defaults to today (day order).

    Returns:
        Dict with orderId, orderRequestStatus, message
    """
    from avanza.constants import OrderType
    return _place_order(orderbook_id, OrderType.BUY, price, volume, valid_until)


def place_sell_order(
    orderbook_id: str,
    price: float,
    volume: int,
    valid_until: date | None = None,
) -> dict:
    """Place a limit SELL order on Avanza.

    Args:
        orderbook_id: Avanza orderbook ID for the instrument
        price: Limit price in SEK
        volume: Number of shares (must be int >= 1)
        valid_until: Order expiry date. Defaults to today (day order).

    Returns:
        Dict with orderId, orderRequestStatus, message
    """
    from avanza.constants import OrderType
    return _place_order(orderbook_id, OrderType.SELL, price, volume, valid_until)


def _place_order(orderbook_id, order_type, price, volume, valid_until):
    """Internal: place an order via the Avanza API.

    P0-4 (2026-05-02): Wrapped in ``avanza_order_lock`` so the TOTP path
    cannot race against the page-session paths in
    ``data/metals_avanza_helpers.place_order``,
    ``portfolio/avanza_session.place_order``, or
    ``portfolio/avanza_control.place_order`` (all of which already lock).
    Without the lock, two callers observing the same ``buying_power``
    snapshot could both fire orders and overdraw the ISK.

    The op label is distinct from page-session labels
    (``place_order_totp/...``) so the rate-limit diagnostic ("which loop
    hit the busy lock") still works. ``OrderLockBusyError`` is allowed
    to propagate so callers can decide whether to retry next cycle —
    matches the existing semantics in
    ``data/metals_avanza_helpers.place_order``.
    """
    if volume < 1:
        raise ValueError(f"Volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"Price must be > 0, got {price}")

    client = get_client()
    account_id = get_account_id()
    expiry = valid_until or date.today()

    logger.info(
        "Placing %s order: orderbook=%s price=%.2f vol=%d until=%s account=%s",
        order_type.value, orderbook_id, price, volume, expiry, account_id,
    )
    with avanza_order_lock(op=f"place_order_totp/{order_type.value}/{orderbook_id}"):
        result = client.place_order(
            account_id=account_id,
            order_book_id=orderbook_id,
            order_type=order_type,
            price=price,
            valid_until=expiry,
            volume=volume,
        )
    logger.info("Order result: %s", result)
    return result


def get_order_status(order_id: str) -> dict:
    """Check the status of an order by ID.

    Returns:
        Order dict with state, price, volume, etc.
    """
    client = get_client()
    account_id = get_account_id()
    return client.get_order(account_id, order_id)


def delete_order(order_id: str) -> dict:
    """Cancel a pending order.

    P0-4 (2026-05-02): Wrapped in ``avanza_order_lock`` for the same
    reason as ``_place_order`` — cancelling an order is mutating and must
    serialize against new orders / stop-loss adjustments to avoid races
    where the position view is read between cancel-and-place.

    Returns:
        Dict with orderId, orderRequestStatus, messages
    """
    client = get_client()
    account_id = get_account_id()
    logger.info("Deleting order %s on account %s", order_id, account_id)
    with avanza_order_lock(op=f"delete_order_totp/{order_id}"):
        return client.delete_order(account_id, order_id)

 succeeded in 951ms:
"""Cross-process advisory file lock guarding Avanza order placement.

Used across metals_loop, golddigger, fin_snipe_manager (and any future
Avanza-bound loop) to prevent two processes submitting overlapping orders
when they observe the same ``buying_power`` simultaneously. Without this,
the following sequence is possible:

    t=0.00  metals_loop reads buying_power=6000 SEK
    t=0.05  golddigger reads buying_power=6000 SEK
    t=0.10  metals_loop POSTs order for 5000 SEK
    t=0.15  golddigger POSTs order for 5000 SEK
    t=0.20  Avanza rejects one / or both fill and settlement overdraws

READ paths (``fetch_price``, ``fetch_positions``, ``buying_power``) are
NOT guarded — they're safe to run concurrently, and the whole point of
the resilience refactor is to keep those fast.

Usage:

    from portfolio.avanza_order_lock import avanza_order_lock

    with avanza_order_lock(op="place_order"):
        resp = api_post("/_api/trading-critical/rest/order/new", payload)

Design notes:

* ``filelock.FileLock`` is already in requirements (3.20.3 as of 2026-04-13).
* 2-second fail-fast default — long enough to ride through a normal order
  round-trip (~300ms), short enough that a hung peer doesn't block trading.
* Raises ``OrderLockBusyError`` on timeout so callers can log + retry next
  cycle instead of blocking the whole loop.
* Caller-provided ``op`` label threads through to log messages for
  diagnostics ("which loop hit the busy lock").
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import filelock

logger = logging.getLogger("portfolio.avanza_order_lock")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOCK_FILE = DATA_DIR / "avanza_order.lock"

DEFAULT_TIMEOUT_S = 2.0


class OrderLockBusyError(Exception):
    """Another process held the lock longer than the configured timeout."""


@contextmanager
def avanza_order_lock(
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    op: str = "order",
    lock_file: Path | None = None,
) -> Iterator[filelock.FileLock]:
    """Acquire the cross-process Avanza order lock for a short critical section.

    Fail-fast after ``timeout_s``. The lock is released automatically on exit.

    Args:
        timeout_s: Seconds to wait for the lock before raising
            :class:`OrderLockBusyError`. Default 2.0 — short enough to abort
            a stuck caller, long enough to ride through a normal order RTT.
        op: Short label for the operation, threaded into log messages.
        lock_file: Override the lock path (tests only). Defaults to
            ``data/avanza_order.lock``.

    Raises:
        OrderLockBusyError: If another process held the lock longer than
            ``timeout_s`` seconds.
    """
    target = Path(lock_file) if lock_file is not None else LOCK_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(target), timeout=timeout_s)
    try:
        lock.acquire()
    except filelock.Timeout as exc:
        logger.warning(
            "avanza_order_lock(%s): busy after %.1fs — another process holds the lock",
            op, timeout_s,
        )
        raise OrderLockBusyError(f"lock busy after {timeout_s}s (op={op})") from exc
    try:
        logger.debug("avanza_order_lock(%s): acquired", op)
        yield lock
    finally:
        try:
            lock.release()
            logger.debug("avanza_order_lock(%s): released", op)
        except Exception as exc:
            logger.warning("avanza_order_lock(%s): release failed: %s", op, exc)

 succeeded in 967ms:
"""Avanza order confirmation flow — human-in-the-loop for real money.

Workflow:
1. Layer 2 calls request_order() → saves intent to pending orders, returns details
   (including a unique 6-hex `confirm_token`).
2. Layer 2 sends Telegram message with order details + "Reply CONFIRM <token>
   to execute".
3. Main loop calls check_pending_orders() each cycle.
4. On CONFIRM <token> reply → execute the order whose token matches, notify
   via Telegram.
5. On timeout (5 min) → expire the pending order, notify.

P1-10 (2026-05-02): per-order `confirm_token` eliminates three races the
old bare-CONFIRM design suffered from (see test class docstrings):
- stale-CONFIRM race (replayed CONFIRM confirms a NEWER order)
- wrong-order race (sort-by-time-DESC matches the wrong order)
- no-pending-yet race (CONFIRM lands before the order it was for)

Bare CONFIRM (no token) is still accepted but ONLY matches LEGACY orders
that have no `confirm_token` field — i.e. orders that were already in
flight when this code was deployed. New orders MUST be confirmed by token.
"""

import contextlib
import logging
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.avanza_control import place_buy_order, place_sell_order
from portfolio.file_utils import atomic_write_json, load_json
from portfolio.http_retry import fetch_with_retry
from portfolio.telegram_notifications import send_telegram

logger = logging.getLogger("portfolio.avanza_orders")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PENDING_FILE = DATA_DIR / "avanza_pending_orders.json"
EXPIRY_MINUTES = 5

# P1-10 (2026-05-02): per-order confirmation nonce. 6 hex chars = 24 bits
# of entropy ≈ ~16M possible tokens. Collision probability across the at-most
# ~5 in-flight pending orders is effectively zero (birthday bound:
# ~5^2/(2*16M) ≈ 7.5e-7). Long enough to survive typos, short enough that
# users will actually type it on a phone keyboard.
_CONFIRM_TOKEN_HEX_CHARS = 6
# Token validation: anything outside [0-9a-f] is silently dropped rather
# than confirmed against an unknown order. This prevents 'CONFIRM xyz' (a
# typo) from accidentally confirming any order via the legacy bare-CONFIRM
# path or matching a token-holding order.
_HEX_TOKEN_RE = re.compile(r"^[0-9a-f]+$")
# CONFIRM prefix matcher. Word boundary required because "confirmed" /
# "confirms" / "confirmation" parse to "confirm" + a hex-valid suffix
# ("ed", "s", "ation") which would silently match against legacy orders
# or non-existent tokens. Anchored at start since the user is asked to
# reply with "CONFIRM <token>" as the entire message.
_CONFIRM_PREFIX_RE = re.compile(r"^confirm(?:\s+|$)")


def _generate_confirm_token() -> str:
    """Return a fresh hex token for a new pending order. Module-level
    indirection keeps tests deterministic via patch.object if ever needed."""
    return secrets.token_hex(_CONFIRM_TOKEN_HEX_CHARS // 2)


def _load_pending() -> list[dict]:
    """Load pending orders from disk."""
    result = load_json(PENDING_FILE, default=[])
    if result is None:
        logger.warning("Failed to read pending orders, returning empty")
        return []
    return result


def _save_pending(orders: list[dict]) -> None:
    """Save pending orders to disk atomically."""
    atomic_write_json(PENDING_FILE, orders)


def request_order(
    action: str,
    orderbook_id: str,
    instrument_name: str,
    config_key: str,
    volume: int,
    price: float,
) -> dict:
    """Create a pending order awaiting Telegram confirmation.

    Args:
        action: "BUY" or "SELL"
        orderbook_id: Avanza orderbook ID
        instrument_name: Human-readable name (e.g. "SAAB B")
        config_key: Config key (e.g. "SAAB-B")
        volume: Number of shares
        price: Limit price in SEK

    Returns:
        The pending order dict (includes id, total_sek, expires, and
        ``confirm_token``). The caller MUST include ``confirm_token`` in
        the Telegram notification asking the user to reply
        ``CONFIRM <token>``. Without that, the user sees the prompt but
        has no way to confirm — bare CONFIRM only matches legacy orders
        without a token.
    """
    if action not in ("BUY", "SELL"):
        raise ValueError(f"action must be BUY or SELL, got {action!r}")
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    now = datetime.now(UTC)
    confirm_token = _generate_confirm_token()
    order = {
        "id": str(uuid.uuid4()),
        "timestamp": now.isoformat(),
        "action": action,
        "orderbook_id": str(orderbook_id),
        "instrument_name": instrument_name,
        "config_key": config_key,
        "volume": volume,
        "price": price,
        "total_sek": round(volume * price, 2),
        "status": "pending_confirmation",
        "expires": (now + timedelta(minutes=EXPIRY_MINUTES)).isoformat(),
        "confirm_token": confirm_token,
    }

    pending = _load_pending()
    pending.append(order)
    _save_pending(pending)
    # Log the token at INFO so an operator reading agent.log can read it
    # if they need to confirm out-of-band (e.g. the agent's Telegram message
    # got truncated). The token is per-order, expires in 5 min, and only
    # confirms one specific order — leak surface is minimal.
    logger.info(
        "Order requested: %s %dx %s @ %.2f SEK (id=%s, confirm_token=%s)",
        action, volume, instrument_name, price, order["id"], confirm_token,
    )
    return order


def get_pending_orders() -> list[dict]:
    """Get all orders with status 'pending_confirmation'."""
    return [o for o in _load_pending() if o["status"] == "pending_confirmation"]


def check_pending_orders(config: dict) -> list[dict]:
    """Check for Telegram confirmations and expire stale orders.

    Called by the main loop each cycle. Polls Telegram getUpdates for
    CONFIRM <token> replies. Executes confirmed orders (matched by token)
    and expires timed-out ones.

    P1-10 (2026-05-02): a CONFIRM <token> reply confirms ONLY the order
    whose ``confirm_token`` matches. Bare CONFIRM (no token) still works
    but ONLY matches LEGACY orders without a token field — so freshly
    created orders cannot be silently confirmed by a stale CONFIRM that
    was replayed by a getUpdates offset bug.

    Args:
        config: App config dict (with telegram.token, telegram.chat_id,
            and optionally telegram.allowed_user_id for sender auth).

    Returns:
        List of orders that were acted on (confirmed or expired) this cycle.
    """
    pending = _load_pending()
    if not pending:
        return []

    acted_on = []
    now = datetime.now(UTC)

    # Set of tokens that arrived this cycle. Bare CONFIRM is "" (empty
    # string) — only matches legacy orders without a token.
    confirmed_tokens = _check_telegram_confirm(config)

    for order in pending:
        if order["status"] != "pending_confirmation":
            continue

        expires = datetime.fromisoformat(order["expires"])
        order_token = order.get("confirm_token", "")

        # P1-10: matching rules.
        # 1. Order has a token AND that token is in confirmed_tokens → confirm.
        # 2. Order has NO token (legacy in-flight order) AND bare CONFIRM
        #    arrived ("" in the set) → confirm. This is the backwards-compat
        #    path for orders that existed before the deploy.
        # 3. Otherwise → no confirmation this cycle (may still expire).
        confirmed_by_token = bool(order_token) and order_token in confirmed_tokens
        confirmed_legacy = (not order_token) and ("" in confirmed_tokens)

        if confirmed_by_token or confirmed_legacy:
            order["status"] = "confirmed"
            acted_on.append(order)
            # Remove the matched token so the same CONFIRM can't double-fire
            # against another order in the same cycle.
            if confirmed_by_token:
                confirmed_tokens.discard(order_token)
            else:
                # Legacy bare CONFIRM only matches one legacy order per cycle.
                confirmed_tokens.discard("")
            _execute_confirmed_order(order, config)
        elif now > expires:
            order["status"] = "expired"
            acted_on.append(order)
            _notify_expired(order, config)

    _save_pending(pending)
    return acted_on


def _check_telegram_confirm(config: dict) -> set[str]:
    """Poll Telegram for CONFIRM <token> replies from the configured chat.

    Returns ``set[str]`` of matched tokens (lowercase hex). Bare CONFIRM
    (with no token) is represented as ``""`` and matches only LEGACY
    pending orders without a ``confirm_token`` field. Anything that's not
    valid hex after CONFIRM (e.g. ``CONFIRM xyz`` typo) is silently
    dropped — never matched against an order — so a typo doesn't
    accidentally confirm via the legacy path.

    Uses getUpdates with a stored offset to avoid reprocessing old messages.

    AV-P1-3 (2026-05-02): Sender-authenticated when
    ``telegram.allowed_user_id`` is set. Without sender auth, the chat-only
    filter is bypassable in two ways:
      - Group chats: anyone admitted can send CONFIRM and execute the
        pending order.
      - Bot-token compromise: an attacker who has the bot token can
        deliver fake updates with the right ``chat_id`` and execute orders.
    When ``allowed_user_id`` is unset the chat-only check is preserved
    (backwards-compatible). The offset still advances on dropped messages
    so we don't re-process the rejected update every cycle.

    P1-10 (2026-05-02): return type changed from ``bool`` to ``set[str]``
    so each pending order can match its own token. Bare CONFIRM is still
    captured (as ``""``) for the legacy backwards-compat path.
    """
    token = config.get("telegram", {}).get("token", "")
    chat_id = str(config.get("telegram", {}).get("chat_id", ""))
    if not token or not chat_id:
        return set()

    # AV-P1-3 (2026-05-02): optional sender allow-list. Accept either int
    # or string in config — Telegram's `from.id` is always int, so coerce
    # both sides to str for comparison so format mistakes don't accidentally
    # admit/reject a real user.
    raw_allowed_user = config.get("telegram", {}).get("allowed_user_id")
    allowed_user = str(raw_allowed_user) if raw_allowed_user is not None else None

    # Load stored offset (BUG-128: now atomic JSON; handles legacy plain-text format)
    offset_file = DATA_DIR / "avanza_telegram_offset.txt"
    offset = 0
    offset_data = load_json(offset_file)
    if isinstance(offset_data, dict):
        offset = int(offset_data.get("offset", 0))
    elif offset_file.exists():
        with contextlib.suppress(ValueError, OSError):
            offset = int(offset_file.read_text().strip())

    params = {"timeout": 1, "allowed_updates": ["message"]}
    if offset:
        params["offset"] = offset

    try:
        r = fetch_with_retry(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params=params,
            timeout=5,
        )
        if r is None or not r.ok:
            return set()
        data = r.json()
        if not data.get("ok"):
            return set()
    except Exception as e:
        logger.warning("Telegram getUpdates failed: %s", e)
        return set()

    found_tokens: set[str] = set()
    for update in data.get("result", []):
        update_id = update.get("update_id", 0)
        # Always advance offset (AV-P1-3: applies to dropped messages too —
        # otherwise a single rejected CONFIRM would replay every cycle).
        if update_id >= offset:
            offset = update_id + 1

        msg = update.get("message", {})
        if str(msg.get("chat", {}).get("id")) != chat_id:
            continue

        # AV-P1-3 (2026-05-02): sender authentication. Fail-closed:
        # missing `from` field with auth enabled drops the message.
        if allowed_user is not None:
            sender = msg.get("from") or {}
            sender_id = sender.get("id")
            if sender_id is None or str(sender_id) != allowed_user:
                logger.warning(
                    "Dropping Telegram message from unauthorized sender id=%r "
                    "(allowed=%s, chat=%s)",
                    sender_id, allowed_user, chat_id,
                )
                continue

        # P1-10 (2026-05-02): parse "CONFIRM <token>" or bare "CONFIRM".
        # Lowercase + collapse whitespace so user-typed variants normalize.
        # Word-boundary match is critical here — without it, "confirmed"
        # parses as "confirm" + "ed" and "ed" IS valid hex (defense vs an
        # accidental "confirmed by my broker" message in the chat).
        text = (msg.get("text") or "").strip().lower()
        m = _CONFIRM_PREFIX_RE.match(text)
        if not m:
            continue
        # Anything after the matched prefix (which includes the word
        # "confirm" + whitespace OR end-of-string) is the candidate.
        rest = text[m.end():].strip()
        if not rest:
            # Bare CONFIRM — legacy backwards-compat path.
            found_tokens.add("")
            continue
        # Take the first whitespace-separated token. Anything trailing is
        # ignored (lets the user paste extra text without breaking the match).
        candidate = rest.split()[0]
        if _HEX_TOKEN_RE.match(candidate):
            found_tokens.add(candidate)
        else:
            logger.warning(
                "Dropping CONFIRM with non-hex token %r (must be lowercase "
                "[0-9a-f] from request_order's confirm_token)",
                candidate,
            )

    # Save offset atomically to prevent corruption on crash (BUG-128)
    try:
        atomic_write_json(offset_file, {"offset": offset})
    except OSError as e:
        logger.warning("Failed to save Telegram offset: %s", e)

    return found_tokens


def _execute_confirmed_order(order: dict, config: dict) -> None:
    """Execute a confirmed order on Avanza and notify via Telegram."""
    action = order["action"]
    try:
        if action == "BUY":
            result = place_buy_order(
                orderbook_id=order["orderbook_id"],
                price=order["price"],
                volume=order["volume"],
            )
        else:
            result = place_sell_order(
                orderbook_id=order["orderbook_id"],
                price=order["price"],
                volume=order["volume"],
            )

        status = result.get("orderRequestStatus", "UNKNOWN")
        order_id = result.get("orderId", "?")
        msg_text = result.get("message", "")

        if status == "SUCCESS":
            order["status"] = "executed"
            order["avanza_order_id"] = order_id
            msg = (
                f"AVANZA {action} EXECUTED\n"
                f"{order['instrument_name']}: {order['volume']}x @ {order['price']:.2f} SEK\n"
                f"Total: {order['total_sek']:,.0f} SEK\n"
                f"Order ID: {order_id}"
            )
            logger.info("Order executed: %s (avanza_id=%s)", order["id"], order_id)
        else:
            order["status"] = "failed"
            order["error"] = msg_text
            msg = (
                f"AVANZA {action} FAILED\n"
                f"{order['instrument_name']}: {order['volume']}x @ {order['price']:.2f} SEK\n"
                f"Error: {msg_text}"
            )
            logger.error("Order failed: %s — %s", order["id"], msg_text)

        send_telegram(msg, config)

    except Exception as e:
        order["status"] = "error"
        order["error"] = str(e)
        logger.error("Order execution error: %s — %s", order["id"], e)
        try:
            send_telegram(
                f"AVANZA ORDER ERROR\n{order['instrument_name']}: {e}",
                config,
            )
        except Exception as e:
            logger.warning("Order error notification failed: %s", e)


def _notify_expired(order: dict, config: dict) -> None:
    """Notify via Telegram that a pending order expired."""
    msg = (
        f"AVANZA ORDER EXPIRED\n"
        f"{order['action']} {order['instrument_name']}: "
        f"{order['volume']}x @ {order['price']:.2f} SEK\n"
        f"No confirmation received within {EXPIRY_MINUTES} min."
    )
    logger.info("Order expired: %s", order["id"])
    try:
        send_telegram(msg, config)
    except Exception as e:
        logger.warning("Failed to send expiry notification: %s", e)

 succeeded in 980ms:
"""CometD/Bayeux WebSocket streaming client for Avanza push data.

Connects to ``wss://www.avanza.se/_push/cometd`` and subscribes to
real-time channels for quotes, order depths, trades, orders, and deals.
Runs a background daemon thread with automatic reconnection.

Usage::

    stream = AvanzaStream(push_subscription_id="abc123")
    stream.on_quote("856394", lambda msg: print(msg))
    stream.start()
    # ... later ...
    stream.stop()
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import websocket

logger = logging.getLogger("portfolio.avanza.streaming")

WS_URL = "wss://www.avanza.se/_push/cometd"

# Reconnect backoff
_MIN_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_BACKOFF_FACTOR = 2.0

# CometD heartbeat interval (seconds)
_HEARTBEAT_INTERVAL = 30.0


class AvanzaStream:
    """CometD/Bayeux WebSocket client for Avanza push data.

    Register callbacks with :meth:`on_quote`, :meth:`on_order_depth`, etc.
    before calling :meth:`start`.  The read loop runs in a daemon thread
    and dispatches messages to registered callbacks by channel.
    """

    def __init__(self, push_subscription_id: str) -> None:
        self._push_sub_id = push_subscription_id
        self._callbacks: dict[str, list[Callable[[dict], None]]] = {}
        self._client_id: str | None = None
        self._ws: websocket.WebSocket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._backoff = _MIN_BACKOFF

    # ------------------------------------------------------------------
    # Public registration (before start)
    # ------------------------------------------------------------------

    def on_quote(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register a callback for quote updates on *ob_id*."""
        channel = f"/quotes/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)

    def on_order_depth(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register a callback for order depth updates on *ob_id*."""
        channel = f"/orderdepths/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)

    def on_trades(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register a callback for trade updates on *ob_id*."""
        channel = f"/trades/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)

    def on_orders(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
        """Register a callback for order updates on the given accounts."""
        channel = "/orders/_" + ",".join(account_ids)
        self._callbacks.setdefault(channel, []).append(callback)

    def on_deals(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
        """Register a callback for deal updates on the given accounts."""
        channel = "/deals/_" + ",".join(account_ids)
        self._callbacks.setdefault(channel, []).append(callback)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("AvanzaStream already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="avanza-stream")
        self._thread.start()
        logger.info("AvanzaStream started (subscriptions=%d)", len(self._callbacks))

    def stop(self) -> None:
        """Close the WebSocket and join the background thread."""
        self._stop_event.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._client_id = None
        logger.info("AvanzaStream stopped")

    # ------------------------------------------------------------------
    # Internal: run loop with reconnection
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Connect, handshake, subscribe, and read — with reconnection."""
        while not self._stop_event.is_set():
            try:
                self._connect()
                self._do_handshake()
                for channel in self._callbacks:
                    self._subscribe_channel(channel)
                self._backoff = _MIN_BACKOFF  # Reset on successful connect
                self._read_loop()
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "AvanzaStream connection error: %s — reconnecting in %.0fs",
                    exc,
                    self._backoff,
                )
                self._stop_event.wait(self._backoff)
                self._backoff = min(self._backoff * _BACKOFF_FACTOR, _MAX_BACKOFF)
            finally:
                if self._ws is not None:
                    with contextlib.suppress(Exception):
                        self._ws.close()
                    self._ws = None

    def _connect(self) -> None:
        """Open WebSocket connection to Avanza push endpoint."""
        self._ws = websocket.create_connection(
            WS_URL,
            timeout=_HEARTBEAT_INTERVAL + 10,
        )
        logger.debug("WebSocket connected to %s", WS_URL)

    def _do_handshake(self) -> None:
        """Perform CometD/Bayeux handshake and extract clientId."""
        handshake_msg = [{
            "channel": "/meta/handshake",
            "ext": {"subscriptionId": self._push_sub_id},
            "version": "1.0",
            "supportedConnectionTypes": ["websocket"],
        }]
        self._ws.send(json.dumps(handshake_msg))  # type: ignore[union-attr]
        response = self._ws.recv()  # type: ignore[union-attr]
        msgs = json.loads(response)

        if not isinstance(msgs, list) or len(msgs) == 0:
            raise RuntimeError(f"Invalid handshake response: {response}")

        handshake_resp = msgs[0]
        if not handshake_resp.get("successful", False):
            raise RuntimeError(f"Handshake failed: {handshake_resp}")

        self._client_id = handshake_resp["clientId"]
        logger.debug("Handshake successful, clientId=%s", self._client_id)

        # Send initial connect message
        connect_msg = [{
            "channel": "/meta/connect",
            "clientId": self._client_id,
            "connectionType": "websocket",
        }]
        self._ws.send(json.dumps(connect_msg))  # type: ignore[union-attr]

    def _subscribe_channel(self, channel: str) -> None:
        """Subscribe to a single CometD channel."""
        sub_msg = [{
            "channel": "/meta/subscribe",
            "subscription": channel,
            "clientId": self._client_id,
        }]
        self._ws.send(json.dumps(sub_msg))  # type: ignore[union-attr]
        logger.debug("Subscribed to %s", channel)

    def _read_loop(self) -> None:
        """Read messages from WebSocket, dispatch, and send heartbeats."""
        last_heartbeat = time.monotonic()

        while not self._stop_event.is_set():
            # Send heartbeat if needed
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                heartbeat_msg = [{
                    "channel": "/meta/connect",
                    "clientId": self._client_id,
                    "connectionType": "websocket",
                }]
                self._ws.send(json.dumps(heartbeat_msg))  # type: ignore[union-attr]
                last_heartbeat = now

            try:
                raw = self._ws.recv()  # type: ignore[union-attr]
            except websocket.WebSocketTimeoutException:
                continue  # Timeout is expected, just loop and heartbeat
            except websocket.WebSocketConnectionClosedException:
                logger.info("WebSocket connection closed")
                return  # Will reconnect in _run_loop

            if not raw:
                continue

            try:
                msgs = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse WebSocket message: %s", raw[:200])
                continue

            if not isinstance(msgs, list):
                msgs = [msgs]

            for msg in msgs:
                self._dispatch_message(msg)

    def _dispatch_message(self, msg: dict[str, Any]) -> None:
        """Route a CometD message to registered callbacks by channel."""
        channel = msg.get("channel", "")

        # Ignore meta channels (handshake, connect, subscribe responses)
        if channel.startswith("/meta/"):
            return

        callbacks = self._callbacks.get(channel, [])
        data = msg.get("data", msg)

        for cb in callbacks:
            try:
                cb(data)
            except Exception as exc:
                logger.error(
                    "Callback error on channel %s: %s",
                    channel,
                    exc,
                    exc_info=True,
                )

 succeeded in 995ms:
"""Trading operations — orders, stop-losses, deals.

Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
for placing, modifying, and cancelling orders and stop-losses.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from avanza.constants import (
    Condition,
    OrderType,
    StopLossPriceType,
    StopLossTriggerType,
)
from avanza.entities import StopLossOrderEvent, StopLossTrigger

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import (
    Deal,
    Order,
    OrderResult,
    StopLoss,
    StopLossResult,
)

logger = logging.getLogger("portfolio.avanza.trading")


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def place_order(
    side: str,
    ob_id: str,
    price: float,
    volume: int,
    condition: str = "NORMAL",
    valid_until: str | None = None,
    account_id: str | None = None,
) -> OrderResult:
    """Place a BUY or SELL order.

    Args:
        side: ``"BUY"`` or ``"SELL"``.
        ob_id: Avanza orderbook ID.
        price: Limit price.
        volume: Number of units.
        condition: Order condition (``"NORMAL"``, ``"FILL_OR_KILL"``,
            ``"FILL_AND_KILL"``).
        valid_until: ISO date string (default: today).
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.OrderResult`.

    Raises:
        ValueError: If volume < 1, price <= 0, or order total < 1000 SEK.
    """
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    # 2026-04-17: match portfolio/avanza_session.py:590 convention — orders
    # below 1000 SEK pay the Avanza courtage minimum and are almost always
    # a caller bug. Unified-package callers should hit the same guard as
    # the legacy path.
    order_total = round(volume * price, 2)
    if order_total < 1000.0:
        raise ValueError(
            f"Order total {order_total:.2f} SEK below minimum 1000 SEK"
        )

    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid = date.fromisoformat(valid_until) if valid_until else date.today()

    raw: dict[str, Any] = client.avanza.place_order(
        acct,
        ob_id,
        OrderType(side),
        price,
        valid,
        volume,
        condition=Condition(condition),
    )

    logger.info(
        "place_order side=%s ob_id=%s price=%s vol=%d -> %s",
        side,
        ob_id,
        price,
        volume,
        raw.get("orderRequestStatus"),
    )
    return OrderResult.from_api(raw)


def modify_order(
    order_id: str,
    ob_id: str,
    price: float,
    volume: int,
    condition: str = "NORMAL",
    valid_until: str | None = None,
    account_id: str | None = None,
) -> OrderResult:
    """Modify an existing order.

    Args:
        order_id: Existing order ID to modify.
        ob_id: Avanza orderbook ID (unused by API but kept for consistency).
        price: New limit price.
        volume: New volume.
        condition: Order condition (unused by edit_order API).
        valid_until: ISO date string (default: today).
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.OrderResult`.
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid = date.fromisoformat(valid_until) if valid_until else date.today()

    raw: dict[str, Any] = client.avanza.edit_order(
        order_id,
        acct,
        price,
        valid,
        volume,
    )

    logger.info(
        "modify_order order_id=%s price=%s vol=%d -> %s",
        order_id,
        price,
        volume,
        raw.get("orderRequestStatus"),
    )
    return OrderResult.from_api(raw)


def cancel_order(
    order_id: str,
    account_id: str | None = None,
) -> bool:
    """Cancel an existing order.

    Returns:
        ``True`` if the cancellation was accepted.
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    raw: dict[str, Any] = client.avanza.delete_order(acct, order_id)
    status = str(raw.get("orderRequestStatus", "")).upper()
    success = status == "SUCCESS"
    logger.info("cancel_order order_id=%s -> %s", order_id, status)
    return success


def get_orders() -> list[Order]:
    """Fetch all open/recent orders.

    Returns:
        List of :class:`~portfolio.avanza.types.Order`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_orders_raw()

    orders_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        orders_list = raw.get("orders", [])
    elif isinstance(raw, list):
        orders_list = raw
    else:
        orders_list = []

    return [Order.from_api(o) for o in orders_list]


def get_deals() -> list[Deal]:
    """Fetch recent deals (executions).

    Returns:
        List of :class:`~portfolio.avanza.types.Deal`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_deals_raw()

    deals_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        deals_list = raw.get("deals", [])
    elif isinstance(raw, list):
        deals_list = raw
    else:
        deals_list = []

    return [Deal.from_api(d) for d in deals_list]


# ---------------------------------------------------------------------------
# Stop-losses
# ---------------------------------------------------------------------------


def place_stop_loss(
    ob_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    valid_days: int = 8,
    trigger_type: str = "LESS_OR_EQUAL",
    value_type: str = "MONETARY",
    account_id: str | None = None,
) -> StopLossResult:
    """Place a stop-loss order.

    Args:
        ob_id: Avanza orderbook ID.
        trigger_price: Price that triggers the stop.
        sell_price: Limit price for the sell order when triggered.
        volume: Number of units to sell.
        valid_days: Days until the stop-loss expires (default 8).
        trigger_type: Trigger direction (``"LESS_OR_EQUAL"``,
            ``"FOLLOW_DOWNWARDS"``, etc.).
        value_type: Price type (``"MONETARY"`` or ``"PERCENTAGE"``).
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.StopLossResult`.
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid_until = date.today() + timedelta(days=valid_days)

    # 2026-04-17: warn (don't raise) on sub-1000 SEK stop legs. Metals-loop
    # cascaded stops split a position into ≤3 legs; per-leg value may
    # legitimately fall below the courtage threshold. Surfacing via log
    # lets callers audit fee impact without breaking live cascading logic.
    if value_type == "MONETARY" and sell_price > 0:
        leg_total = round(volume * sell_price, 2)
        if leg_total < 1000.0:
            logger.warning(
                "place_stop_loss leg %.2f SEK below 1000 SEK courtage threshold "
                "(vol=%d sell=%.3f ob=%s)",
                leg_total, volume, sell_price, ob_id,
            )

    trigger = StopLossTrigger(
        type=StopLossTriggerType(trigger_type),
        value=trigger_price,
        valid_until=valid_until,
        value_type=StopLossPriceType(value_type),
    )

    order_event = StopLossOrderEvent(
        type=OrderType.SELL,
        price=sell_price,
        volume=volume,
        valid_days=valid_days,
        price_type=StopLossPriceType(value_type),
        short_selling_allowed=False,
    )

    raw: dict[str, Any] = client.avanza.place_stop_loss_order(
        "0",  # parent_stop_loss_id — "0" for new stop-loss
        acct,
        ob_id,
        trigger,
        order_event,
    )

    logger.info(
        "place_stop_loss ob_id=%s trigger=%.4f sell=%.4f vol=%d -> %s",
        ob_id,
        trigger_price,
        sell_price,
        volume,
        raw.get("status", raw.get("orderRequestStatus")),
    )
    return StopLossResult.from_api(raw)


def place_trailing_stop(
    ob_id: str,
    trail_percent: float,
    volume: int,
    valid_days: int = 8,
    account_id: str | None = None,
) -> StopLossResult:
    """Place a trailing stop-loss (follows price downwards by percentage).

    Args:
        ob_id: Avanza orderbook ID.
        trail_percent: Trailing distance as percentage (e.g. ``5.0`` for 5%).
        volume: Number of units to sell.
        valid_days: Days until the stop-loss expires.
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.StopLossResult`.
    """
    return place_stop_loss(
        ob_id=ob_id,
        trigger_price=trail_percent,
        sell_price=0.0,  # Not applicable for trailing stops
        volume=volume,
        valid_days=valid_days,
        trigger_type="FOLLOW_DOWNWARDS",
        value_type="PERCENTAGE",
        account_id=account_id,
    )


def get_stop_losses() -> list[StopLoss]:
    """Fetch all active stop-losses.

    Returns:
        List of :class:`~portfolio.avanza.types.StopLoss`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_all_stop_losses_raw()

    sl_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        sl_list = raw.get("stopLosses", raw.get("stopLossOrders", []))
    elif isinstance(raw, list):
        sl_list = raw
    else:
        sl_list = []

    return [StopLoss.from_api(sl) for sl in sl_list]


def delete_stop_loss(
    stop_id: str,
    account_id: str | None = None,
) -> bool:
    """Delete a stop-loss order.  Idempotent — 404 is treated as success.

    Returns:
        ``True`` if the deletion succeeded (or the stop-loss was already gone).
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    try:
        client.avanza.delete_stop_loss_order(acct, stop_id)
        logger.info("delete_stop_loss stop_id=%s -> OK", stop_id)
        return True
    except Exception as exc:
        # 404 means already deleted — treat as success
        exc_str = str(exc).lower()
        if "404" in exc_str or "not found" in exc_str:
            logger.info("delete_stop_loss stop_id=%s -> already gone (404)", stop_id)
            return True
        logger.error("delete_stop_loss stop_id=%s -> FAILED: %s", stop_id, exc)
        return False

 succeeded in 1027ms:
"""Typed response dataclasses for Avanza API data.

Avanza wraps many numeric values in ``{"value": X, "unit": "SEK", ...}``
objects.  The ``_val`` helper unwraps these transparently so callers always
get plain Python scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(obj: Any, default: Any = None) -> Any:
    """Unwrap Avanza ``{value: X}`` wrappers, or return *obj* as-is.

    Handles:
      - ``{"value": 1.23, "unit": "SEK", ...}`` -> ``1.23``
      - plain scalars -> passed through
      - ``None`` / missing -> *default*
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        if "value" in obj:
            return obj["value"]
        return default
    return obj


def _ts(millis: Any) -> str:
    """Convert a millisecond Unix timestamp to an ISO-8601 string."""
    if millis is None:
        return ""
    if isinstance(millis, str):
        return millis
    try:
        return datetime.fromtimestamp(int(millis) / 1000, tz=UTC).isoformat()
    except (ValueError, TypeError, OSError):
        return str(millis)


# ---------------------------------------------------------------------------
# Quote & Market Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Quote:
    """Parsed quote snapshot."""

    bid: float
    ask: float
    last: float
    spread: float
    change_percent: float
    high: float
    low: float
    volume: float
    updated: str

    @classmethod
    def from_api(cls, raw: dict) -> Quote:
        bid = _val(raw.get("buy"), _val(raw.get("bid"), 0.0))
        ask = _val(raw.get("sell"), _val(raw.get("ask"), 0.0))
        last = _val(raw.get("last"), _val(raw.get("latest"), 0.0))
        spread = _val(raw.get("spread"))
        if spread is None:
            spread = round(ask - bid, 6) if (ask and bid) else 0.0
        change_percent = _val(raw.get("changePercent"), _val(raw.get("change_percent"), 0.0))
        high = _val(raw.get("highest"), _val(raw.get("high"), 0.0))
        low = _val(raw.get("lowest"), _val(raw.get("low"), 0.0))
        volume = _val(raw.get("totalVolumeTraded"), _val(raw.get("volume"), 0.0))
        updated = _ts(raw.get("updated", ""))
        return cls(
            bid=float(bid),
            ask=float(ask),
            last=float(last),
            spread=float(spread),
            change_percent=float(change_percent),
            high=float(high),
            low=float(low),
            volume=float(volume),
            updated=updated,
        )


@dataclass(frozen=True, slots=True)
class OrderDepthLevel:
    """One price level in the order book."""

    price: float
    volume: int

    @classmethod
    def from_api(cls, raw: dict) -> OrderDepthLevel:
        return cls(
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """A single executed trade from the market-data feed."""

    price: float
    volume: int
    buyer: str
    seller: str
    time: str

    @classmethod
    def from_api(cls, raw: dict) -> Trade:
        return cls(
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
            buyer=str(raw.get("buyer", "")),
            seller=str(raw.get("seller", "")),
            time=str(raw.get("dealTime", raw.get("time", ""))),
        )


@dataclass(frozen=True, slots=True)
class MarketData:
    """Aggregated market data: quote + depth + recent trades."""

    quote: Quote
    bid_levels: tuple[OrderDepthLevel, ...]
    ask_levels: tuple[OrderDepthLevel, ...]
    recent_trades: tuple[Trade, ...]
    market_maker_expected: bool

    @classmethod
    def from_api(cls, raw: dict) -> MarketData:
        # Quote
        quote_raw = raw.get("quote", {})
        quote = Quote.from_api(quote_raw)

        # Order depth
        depth = raw.get("orderDepth", {})
        levels = depth.get("levels", [])
        bid_levels: list[OrderDepthLevel] = []
        ask_levels: list[OrderDepthLevel] = []
        for lvl in levels:
            buy_side = lvl.get("buySide", {})
            sell_side = lvl.get("sellSide", {})
            if buy_side:
                bid_levels.append(OrderDepthLevel.from_api(buy_side))
            if sell_side:
                ask_levels.append(OrderDepthLevel.from_api(sell_side))

        # Trades
        trades_raw = raw.get("trades", [])
        trades = tuple(Trade.from_api(t) for t in trades_raw)

        mm = depth.get("marketMakerExpected", False)

        return cls(
            quote=quote,
            bid_levels=tuple(bid_levels),
            ask_levels=tuple(ask_levels),
            recent_trades=trades,
            market_maker_expected=bool(mm),
        )


# ---------------------------------------------------------------------------
# Order / StopLoss results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OrderResult:
    """Result of placing or deleting an order."""

    success: bool
    order_id: str
    status: str
    message: str

    @classmethod
    def from_api(cls, raw: dict) -> OrderResult:
        status = raw.get("orderRequestStatus", raw.get("status", ""))
        success = str(status).upper() == "SUCCESS"
        order_id = str(raw.get("orderId", raw.get("order_id", "")))
        message = raw.get("message", raw.get("messages", ""))
        if isinstance(message, list):
            message = "; ".join(str(m) for m in message)
        return cls(
            success=success,
            order_id=order_id,
            status=str(status),
            message=str(message),
        )


@dataclass(frozen=True, slots=True)
class StopLossResult:
    """Result of placing or modifying a stop-loss."""

    success: bool
    stop_id: str
    status: str

    @classmethod
    def from_api(cls, raw: dict) -> StopLossResult:
        status = raw.get("status", raw.get("orderRequestStatus", ""))
        success = str(status).upper() in ("SUCCESS", "OK", "ACTIVE")
        stop_id = str(raw.get("stoplossOrderId", raw.get("stopLossId", raw.get("stop_id", raw.get("id", "")))))
        return cls(
            success=success,
            stop_id=stop_id,
            status=str(status),
        )


# ---------------------------------------------------------------------------
# Account & Portfolio
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Position:
    """A single instrument position within an account."""

    name: str
    orderbook_id: str
    instrument_type: str
    volume: float
    value: float
    acquired_value: float
    profit: float
    profit_percent: float
    last_price: float
    change_percent: float
    account_id: str
    currency: str

    @classmethod
    def from_api(cls, raw: dict) -> Position:
        instrument = raw.get("instrument", {})
        orderbook = instrument.get("orderbook", {})
        account = raw.get("account", {})
        perf = raw.get("lastTradingDayPerformance", {})

        # Quote values from the orderbook sub-object
        ob_quote = orderbook.get("quote", {})
        latest = _val(ob_quote.get("latest"), _val(ob_quote.get("last"), 0.0))
        change_pct = _val(ob_quote.get("changePercent"), _val(ob_quote.get("change_percent"), 0.0))

        return cls(
            name=orderbook.get("name", instrument.get("name", "")),
            orderbook_id=str(orderbook.get("id", raw.get("id", ""))),
            instrument_type=instrument.get("type", orderbook.get("type", "")),
            volume=float(_val(raw.get("volume"), 0.0)),
            value=float(_val(raw.get("value"), 0.0)),
            acquired_value=float(_val(raw.get("acquiredValue"), 0.0)),
            profit=float(_val(perf.get("absolute"), 0.0)),
            profit_percent=float(_val(perf.get("relative"), 0.0)),
            last_price=float(latest),
            change_percent=float(change_pct),
            account_id=str(account.get("id", "")),
            currency=instrument.get("currency", ""),
        )


@dataclass(frozen=True, slots=True)
class Order:
    """An open or filled order."""

    order_id: str
    orderbook_id: str
    side: str
    price: float
    volume: int
    status: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> Order:
        return cls(
            order_id=str(raw.get("orderId", raw.get("id", ""))),
            orderbook_id=str(raw.get("orderBookId", raw.get("orderbookId", raw.get("orderbook_id", "")))),
            side=str(raw.get("orderType", raw.get("side", ""))),
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
            status=str(raw.get("status", raw.get("statusDescription", ""))),
            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
        )


@dataclass(frozen=True, slots=True)
class Deal:
    """A completed deal (execution)."""

    deal_id: str
    orderbook_id: str
    side: str
    price: float
    volume: int
    time: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> Deal:
        return cls(
            deal_id=str(raw.get("dealId", raw.get("id", ""))),
            orderbook_id=str(raw.get("orderBookId", raw.get("orderbookId", ""))),
            side=str(raw.get("orderType", raw.get("side", ""))),
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
            time=str(raw.get("dealTime", raw.get("time", ""))),
            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
        )


@dataclass(frozen=True, slots=True)
class StopLoss:
    """An active stop-loss order."""

    stop_id: str
    orderbook_id: str
    trigger_price: float
    trigger_type: str
    sell_price: float
    volume: int
    status: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> StopLoss:
        trigger = raw.get("trigger", {})
        order_event = raw.get("orderEvent", raw.get("order", {}))
        return cls(
            stop_id=str(raw.get("id", raw.get("stopLossId", ""))),
            orderbook_id=str((raw.get("orderbook") or {}).get("id", raw.get("orderBookId", raw.get("orderbookId", "")))),
            trigger_price=float(_val(trigger.get("value"), _val(raw.get("triggerPrice"), 0.0))),
            trigger_type=str(trigger.get("type", raw.get("triggerType", "LAST_PRICE"))),
            sell_price=float(_val(order_event.get("price"), _val(raw.get("sellPrice"), 0.0))),
            volume=int(_val(order_event.get("volume"), _val(raw.get("volume"), 0))),
            status=str(raw.get("status", "")),
            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SearchHit:
    """Instrument search result."""

    orderbook_id: str
    name: str
    instrument_type: str
    tradeable: bool
    last_price: float
    change_percent: float

    @classmethod
    def from_api(cls, raw: dict) -> SearchHit:
        return cls(
            orderbook_id=str(raw.get("id", raw.get("orderbookId", ""))),
            name=str(raw.get("name", "")),
            instrument_type=str(raw.get("instrumentType", raw.get("type", ""))),
            tradeable=bool(raw.get("tradable", raw.get("tradeable", False))),
            last_price=float(_val(raw.get("lastPrice"), _val(raw.get("last_price"), 0.0))),
            change_percent=float(_val(raw.get("changePercent"), _val(raw.get("change_percent"), 0.0))),
        )


# ---------------------------------------------------------------------------
# Tick Table
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TickEntry:
    """One row from the tick-size table."""

    min_price: float
    max_price: float
    tick_size: float

    @classmethod
    def from_api(cls, raw: dict) -> TickEntry:
        return cls(
            min_price=float(raw.get("min", raw.get("minPrice", 0.0))),
            max_price=float(raw.get("max", raw.get("maxPrice", 0.0))),
            tick_size=float(raw.get("tick", raw.get("tickSize", raw.get("tick_size", 0.0)))),
        )


# ---------------------------------------------------------------------------
# OHLC
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OHLC:
    """A single OHLCV candle."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_api(cls, raw: dict) -> OHLC:
        return cls(
            timestamp=_ts(raw.get("timestamp")),
            open=float(raw.get("open", 0.0)),
            high=float(raw.get("high", 0.0)),
            low=float(raw.get("low", 0.0)),
            close=float(raw.get("close", 0.0)),
            volume=int(raw.get("totalVolumeTraded", raw.get("volume", 0))),
        )


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AccountCash:
    """Account-level cash info."""

    buying_power: float
    total_value: float
    own_capital: float

    @classmethod
    def from_api(cls, raw: dict) -> AccountCash:
        return cls(
            buying_power=float(_val(raw.get("buyingPower"), 0.0)),
            total_value=float(_val(raw.get("totalValue"), 0.0)),
            own_capital=float(_val(raw.get("ownCapital"), _val(raw.get("buyingPowerWithoutCredit"), 0.0))),
        )


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Transaction:
    """A historical account transaction."""

    transaction_id: str
    transaction_type: str
    instrument_name: str
    amount: float
    price: float
    volume: float
    date: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> Transaction:
        account = raw.get("account", {})
        return cls(
            transaction_id=str(raw.get("id", "")),
            transaction_type=str(raw.get("type", "")),
            instrument_name=str(raw.get("instrumentName", raw.get("description", ""))),
            amount=float(_val(raw.get("amount"), 0.0)),
            price=float(_val(raw.get("priceInTradedCurrency"), _val(raw.get("price"), 0.0))),
            volume=float(_val(raw.get("volume"), 0.0)),
            date=str(raw.get("date", raw.get("tradeDate", ""))),
            account_id=str(account.get("id", raw.get("accountId", ""))),
        )


# ---------------------------------------------------------------------------
# Instrument Info
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InstrumentInfo:
    """Core instrument metadata (works for certificates, warrants, stocks)."""

    orderbook_id: str
    name: str
    instrument_type: str
    currency: str
    leverage: float
    barrier: float
    underlying_name: str
    underlying_price: float

    @classmethod
    def from_api(cls, raw: dict) -> InstrumentInfo:
        return cls(
            orderbook_id=str(raw.get("id", raw.get("orderbookId", ""))),
            name=str(raw.get("name", "")),
            instrument_type=str(raw.get("instrumentType", raw.get("type", ""))),
            currency=str(raw.get("currency", "")),
            leverage=float(_val(raw.get("leverage"), 0.0)),
            barrier=float(_val(raw.get("barrier"), _val(raw.get("barrierLevel"), 0.0))),
            underlying_name=str(raw.get("underlyingName", raw.get("underlying", {}).get("name", ""))),
            underlying_price=float(_val(
                raw.get("underlyingPrice"),
                _val(raw.get("underlying", {}).get("price"), 0.0),
            )),
        )


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NewsArticle:
    """A single news article linked to an instrument."""

    article_id: str
    headline: str
    date: str
    source: str

    @classmethod
    def from_api(cls, raw: dict) -> NewsArticle:
        return cls(
            article_id=str(raw.get("id", raw.get("articleId", ""))),
            headline=str(raw.get("headline", "")),
            date=_ts(raw.get("timePublishedMillis", raw.get("date", raw.get("timePublished", "")))),
            source=str(raw.get("newsSource", raw.get("source", ""))),
        )

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/search.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/scanner.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/tick_rules.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/account.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/auth.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza/client.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 796ms:
"""Instrument search — find stocks, certificates, warrants, etc.

Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
for instrument discovery.
"""

from __future__ import annotations

import logging
from typing import Any

from avanza.constants import InstrumentType

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import SearchHit

logger = logging.getLogger("portfolio.avanza.search")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(
    query: str,
    limit: int = 10,
    instrument_type: str | None = None,
) -> list[SearchHit]:
    """Search for instruments on Avanza.

    Args:
        query: Search string (ISIN, ticker, name fragment, etc.).
        limit: Maximum number of results.
        instrument_type: Optional filter (e.g. ``"certificate"``,
            ``"stock"``, ``"warrant"``).  When *None*, all types are
            searched.

    Returns:
        List of :class:`~portfolio.avanza.types.SearchHit`.
    """
    client = AvanzaClient.get_instance()

    inst_type = (
        InstrumentType(instrument_type) if instrument_type else InstrumentType.ANY
    )

    raw: Any = client.avanza.search_for_instrument(inst_type, query, limit)
    logger.debug(
        "search query=%r type=%s limit=%d hits=%d",
        query,
        inst_type.name,
        limit,
        len(raw) if isinstance(raw, list) else 0,
    )

    hits: list[dict[str, Any]]
    if isinstance(raw, list):
        hits = raw
    elif isinstance(raw, dict):
        hits = raw.get("hits", raw.get("results", []))
    else:
        hits = []

    return [SearchHit.from_api(h) for h in hits]


def find_warrants(query: str = "", limit: int = 20) -> list[SearchHit]:
    """Search specifically for warrants.

    Convenience wrapper around :func:`search` with
    ``instrument_type="warrant"``.
    """
    return search(query=query, limit=limit, instrument_type="warrant")


def find_certificates(query: str = "", limit: int = 20) -> list[SearchHit]:
    """Search specifically for certificates.

    Convenience wrapper around :func:`search` with
    ``instrument_type="certificate"``.
    """
    return search(query=query, limit=limit, instrument_type="certificate")

 succeeded in 818ms:
"""Instrument scanner — find and rank the best warrants/certificates.

Chains search → detail fetch → ranking to answer questions like:
"Find the best bull mini-future for oil right now"

Works with EITHER auth method:
- TOTP (AvanzaClient) — preferred, faster
- BankID session (avanza_session.api_get/api_post) — fallback

Usage:
    from portfolio.avanza.scanner import scan_instruments

    results = scan_instruments(
        query="OLJA",           # underlying asset keyword
        direction="BULL",       # BULL or BEAR
        instrument_type="certificate",  # certificate, warrant, or None for both
        sort_by="spread",       # spread, leverage, price, barrier_distance
        limit=10,
    )
    for r in results:
        print(f"{r['name']:40s} lev={r['leverage']:5.1f}x  spread={r['spread_pct']:.2f}%  bid={r['bid']}")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass

from portfolio.avanza.types import _val

logger = logging.getLogger("portfolio.avanza.scanner")


# ---------------------------------------------------------------------------
# Dual-auth API helpers — try TOTP first, fall back to BankID session
# ---------------------------------------------------------------------------

def _get_api():
    """Return (search_fn, instrument_fn, marketdata_fn, thread_safe) that work
    with whichever auth is currently available.

    Returns:
        Tuple of four:
        - search(instrument_type_str, query, limit) -> dict or list
        - get_instrument(api_type, ob_id) -> dict
        - get_market_data(ob_id) -> dict
        - thread_safe: bool — True for TOTP (requests.Session), False for BankID (Playwright)
    """
    # Try TOTP client first (thread-safe, supports parallel fetching)
    try:
        from portfolio.avanza.client import AvanzaClient
        client = AvanzaClient.get_instance()
        avanza = client.avanza

        def _search(itype_str, query, limit):
            from avanza.constants import InstrumentType
            return avanza.search_for_instrument(InstrumentType(itype_str), query, limit)

        def _instrument(api_type, ob_id):
            return avanza.get_instrument(api_type, ob_id)

        def _marketdata(ob_id):
            return avanza.get_market_data(ob_id)

        logger.debug("Scanner using TOTP client (thread-safe)")
        return _search, _instrument, _marketdata, True
    except Exception:
        logger.debug("TOTP client unavailable, falling back to BankID session")

    # Fall back to BankID session (Playwright — NOT thread-safe, must be sequential)
    try:
        from portfolio.avanza_session import api_get, api_post

        def _search(itype_str, query, limit):
            return api_post("/_api/search/filtered-search", {"query": query, "limit": limit})

        def _instrument(api_type, ob_id):
            return api_get(f"/_api/market-guide/{api_type}/{ob_id}")

        def _marketdata(ob_id):
            try:
                return api_get(f"/_api/trading-critical/rest/marketdata/{ob_id}")
            except Exception:
                return {}

        logger.debug("Scanner using BankID session (sequential only)")
        return _search, _instrument, _marketdata, False
    except Exception as e:
        raise RuntimeError(
            "No Avanza auth available. Either configure TOTP credentials "
            "or run scripts/avanza_login.py for BankID session."
        ) from e


@dataclass
class ScannedInstrument:
    """Rich instrument data combining search + market-guide + marketdata."""

    orderbook_id: str
    name: str
    instrument_type: str  # CERTIFICATE, WARRANT, etc.
    direction: str  # BULL, BEAR, LONG, SHORT, or ""

    # Price
    bid: float | None
    ask: float | None
    last: float | None
    spread_pct: float | None  # (ask-bid)/bid * 100

    # Instrument details
    leverage: float | None
    barrier: float | None
    barrier_distance_pct: float | None  # distance from last to barrier

    # Underlying
    underlying_name: str
    underlying_price: float | None

    # Market quality
    volume_today: int
    turnover_today: float
    market_maker: bool

    # Order depth (best level)
    bid_volume: int
    ask_volume: int

    # Computed score (lower = better for spread, higher = better for leverage)
    score: float


def scan_instruments(
    query: str,
    direction: str = "",
    instrument_type: str | None = None,
    sort_by: str = "spread",
    limit: int = 10,
    max_search: int = 30,
    min_leverage: float = 0,
    max_spread_pct: float = 100,
    workers: int = 6,
) -> list[ScannedInstrument]:
    """Search Avanza and fetch details for the best instruments.

    Args:
        query: Search keyword (e.g. "OLJA", "SILVER", "GULD", "TSMC").
        direction: "BULL" or "BEAR" (filters results by name). Empty = both.
        instrument_type: "certificate", "warrant", or None for both.
        sort_by: Ranking criterion — "spread", "leverage", "price", "barrier_distance".
        limit: Max results to return (after filtering and ranking).
        max_search: How many search results to fetch before filtering.
        min_leverage: Minimum leverage to include.
        max_spread_pct: Maximum spread % to include (filters illiquid instruments).
        workers: Thread pool size for parallel detail fetching.

    Returns:
        List of ScannedInstrument, sorted by the chosen criterion.
    """
    search_fn, instrument_fn, marketdata_fn, thread_safe = _get_api()

    # --- Step 1: Search ---
    search_query = f"{direction} {query}".strip() if direction else query
    types_to_search = []
    if instrument_type:
        types_to_search.append(instrument_type)
    else:
        types_to_search.extend(["certificate", "warrant"])

    all_hits: list[dict] = []
    for itype in types_to_search:
        try:
            raw = search_fn(itype, search_query, max_search)
            hits = raw.get("hits", raw) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
            all_hits.extend(hits)
        except Exception as e:
            logger.warning("Search failed for type=%s query=%r: %s", itype, search_query, e)

    if not all_hits:
        logger.info("No search results for query=%r direction=%s", query, direction)
        return []

    # Filter by direction if specified
    dir_upper = direction.upper()
    if dir_upper:
        all_hits = [h for h in all_hits if dir_upper in (h.get("title", "") or "").upper()]

    # Filter tradeable only
    all_hits = [h for h in all_hits if h.get("tradeable", h.get("tradable", True))]

    # Deduplicate by orderbook ID
    seen = set()
    unique_hits = []
    for h in all_hits:
        ob_id = str(h.get("orderBookId", h.get("id", "")))
        if ob_id and ob_id not in seen:
            seen.add(ob_id)
            unique_hits.append(h)
    all_hits = unique_hits[:max_search]

    logger.info("Scanner: %d candidates after search+filter for %r %s", len(all_hits), query, direction)

    # --- Step 2: Fetch details in parallel ---
    def fetch_detail(hit: dict) -> ScannedInstrument | None:
        ob_id = str(hit.get("orderBookId", hit.get("id", "")))
        name = hit.get("title", hit.get("name", ""))
        itype = hit.get("type", hit.get("instrumentType", ""))

        # Determine API type for market-guide
        api_type = "certificate"
        type_lower = itype.lower() if itype else ""
        if "warrant" in type_lower or "mini" in name.upper():
            api_type = "warrant"
        elif "stock" in type_lower:
            api_type = "stock"

        try:
            # Fetch instrument details (leverage, barrier, underlying)
            info = instrument_fn(api_type, ob_id)
            if not info or not isinstance(info, dict):
                return None

            # Extract quote
            quote = info.get("quote", {})
            bid = _val(quote.get("buy"))
            ask = _val(quote.get("sell"))
            last = _val(quote.get("last"))

            # Compute spread
            spread_pct = None
            if bid and ask and bid > 0:
                spread_pct = round((ask - bid) / bid * 100, 3)

            # Extract leverage and barrier
            ki = info.get("keyIndicators", {})
            leverage = _val(ki.get("leverage"))
            barrier = _val(ki.get("barrierLevel"))

            # Barrier distance
            barrier_dist_pct = None
            if barrier and last and last > 0:
                barrier_dist_pct = round(abs(last - barrier) / last * 100, 2)

            # Underlying
            underlying = info.get("underlying", {})
            underlying_name = underlying.get("name", "")
            underlying_quote = underlying.get("quote", {})
            underlying_price = _val(underlying_quote.get("last"))

            # Volume/turnover
            volume = _val(quote.get("totalVolumeTraded"), 0) or 0
            turnover = _val(quote.get("totalValueTraded"), 0) or 0

            # Detect direction from name
            name_upper = name.upper()
            detected_dir = ""
            for d in ("BULL", "BEAR", "MINI L", "MINI S"):
                if d in name_upper:
                    detected_dir = "BULL" if d in ("BULL", "MINI L") else "BEAR"
                    break

            # Also try market data for order depth (fast call)
            bid_vol = 0
            ask_vol = 0
            mm = False
            with suppress(Exception):
                md = marketdata_fn(ob_id)
                if isinstance(md, dict):
                    od = md.get("orderDepth", md.get("orderDepthLevels", {}))
                    levels = od.get("levels", od) if isinstance(od, dict) else od
                    if isinstance(levels, list) and levels:
                        first = levels[0]
                        bid_side = first.get("buySide", first.get("buy", {}))
                        ask_side = first.get("sellSide", first.get("sell", {}))
                        bid_vol = int(bid_side.get("volume", 0))
                        ask_vol = int(ask_side.get("volume", 0))
                    mm = md.get("marketMakerExpected", False)

            return ScannedInstrument(
                orderbook_id=ob_id,
                name=name,
                instrument_type=itype,
                direction=detected_dir,
                bid=bid,
                ask=ask,
                last=last,
                spread_pct=spread_pct,
                leverage=leverage,
                barrier=barrier,
                barrier_distance_pct=barrier_dist_pct,
                underlying_name=underlying_name,
                underlying_price=underlying_price,
                volume_today=int(volume),
                turnover_today=float(turnover),
                market_maker=mm,
                bid_volume=bid_vol,
                ask_volume=ask_vol,
                score=0.0,
            )
        except Exception as e:
            logger.debug("Detail fetch failed for %s (%s): %s", ob_id, name, e)
            return None

    results: list[ScannedInstrument] = []
    t0 = time.perf_counter()
    if thread_safe and workers > 1:
        # TOTP: parallel fetch via thread pool
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_detail, h): h for h in all_hits}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        # BankID/Playwright: sequential (not thread-safe)
        for h in all_hits:
            result = fetch_detail(h)
            if result is not None:
                results.append(result)
    dt = (time.perf_counter() - t0) * 1000
    logger.info("Scanner: fetched %d instrument details in %.0fms (%s)",
                len(results), dt, "parallel" if thread_safe else "sequential")

    # --- Step 3: Filter ---
    if min_leverage > 0:
        results = [r for r in results if r.leverage and r.leverage >= min_leverage]
    if max_spread_pct < 100:
        results = [r for r in results if r.spread_pct is not None and r.spread_pct <= max_spread_pct]

    # Filter out instruments with no bid/ask (not tradeable right now)
    results = [r for r in results if r.bid is not None and r.ask is not None]

    # --- Step 4: Score and sort ---
    for r in results:
        if sort_by == "spread":
            r.score = r.spread_pct if r.spread_pct is not None else 999
        elif sort_by == "leverage":
            r.score = -(r.leverage or 0)  # negative so higher leverage sorts first
        elif sort_by == "price":
            r.score = r.last or 999
        elif sort_by == "barrier_distance":
            r.score = -(r.barrier_distance_pct or 0)  # negative = larger distance first
        else:
            r.score = r.spread_pct if r.spread_pct is not None else 999

    results.sort(key=lambda r: r.score)
    return results[:limit]


def format_scan_results(results: list[ScannedInstrument]) -> str:
    """Format scan results as a readable table string."""
    if not results:
        return "No instruments found."

    lines = []
    lines.append(f"{'Name':45s} {'ID':>8s} {'Lev':>5s} {'Bid':>8s} {'Ask':>8s} "
                 f"{'Spread':>7s} {'Barrier':>8s} {'Dist%':>6s} {'Vol':>6s} {'MM':>3s}")
    lines.append("-" * 115)

    for r in results:
        lev = f"{r.leverage:.1f}x" if r.leverage else "  -  "
        bid = f"{r.bid:.2f}" if r.bid else "   -   "
        ask = f"{r.ask:.2f}" if r.ask else "   -   "
        spread = f"{r.spread_pct:.2f}%" if r.spread_pct is not None else "  -  "
        barrier = f"{r.barrier:.1f}" if r.barrier else "   -   "
        dist = f"{r.barrier_distance_pct:.1f}%" if r.barrier_distance_pct is not None else "  -  "
        vol = f"{r.volume_today:,}" if r.volume_today else "  0"
        mm = "Yes" if r.market_maker else " No"

        lines.append(f"{r.name[:45]:45s} {r.orderbook_id:>8s} {lev:>5s} {bid:>8s} {ask:>8s} "
                     f"{spread:>7s} {barrier:>8s} {dist:>6s} {vol:>6s} {mm:>3s}")

    return "\n".join(lines)

 succeeded in 808ms:
"""Account data — positions, buying power, transactions.

Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
raw delegators for account-level queries.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from typing import Any

from avanza.constants import TransactionsDetailsType

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import AccountCash, Position, Transaction

logger = logging.getLogger("portfolio.avanza.account")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_positions(account_id: str | None = None) -> list[Position]:
    """Fetch all positions, optionally filtered to a single account.

    Args:
        account_id: When provided only positions for this account are
            returned.  Otherwise all positions across all accounts.

    Returns:
        List of :class:`~portfolio.avanza.types.Position`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_positions_raw()

    # The API may return a dict with a nested positions list, or a list.
    positions_raw: list[dict[str, Any]]
    if isinstance(raw, dict):
        # Prefer "withOrderbook" (newer API), fall back to "positions"
        positions_raw = raw.get("withOrderbook", raw.get("positions", []))
    elif isinstance(raw, list):
        positions_raw = raw
    else:
        positions_raw = []

    positions = [Position.from_api(p) for p in positions_raw]

    if account_id is not None:
        positions = [p for p in positions if p.account_id == str(account_id)]

    logger.debug(
        "get_positions account_id=%s total=%d filtered=%d",
        account_id,
        len(positions_raw),
        len(positions),
    )
    return positions


def get_buying_power(account_id: str | None = None) -> AccountCash:
    """Fetch buying power / cash info for a specific account.

    Args:
        account_id: Account to query.  Defaults to the client's configured
            account.

    Returns:
        :class:`~portfolio.avanza.types.AccountCash`.
    """
    client = AvanzaClient.get_instance()
    acct = str(account_id) if account_id else client.account_id
    raw: Any = client.get_overview_raw()

    # The overview contains a list of accounts — find the right one.
    accounts: list[dict[str, Any]]
    if isinstance(raw, dict):
        accounts = raw.get("accounts", [])
    elif isinstance(raw, list):
        accounts = raw
    else:
        accounts = []

    for account in accounts:
        if str(account.get("accountId", account.get("id", ""))) == acct:
            logger.debug("get_buying_power account_id=%s found", acct)
            return AccountCash.from_api(account)

    # Account not found — return zeroes
    logger.warning("get_buying_power account_id=%s not found in overview", acct)
    return AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)


def get_transactions(
    from_date: str,
    to_date: str,
    types: Sequence[str] | None = None,
    account_id: str | None = None,
) -> list[Transaction]:
    """Fetch historical transactions.

    Args:
        from_date: Start date (ISO-8601, e.g. ``"2026-01-01"``).
        to_date: End date (ISO-8601).
        types: Transaction type filters (e.g. ``["BUY", "SELL"]``).
            When *None* all types are returned.
        account_id: Unused by the library call but kept for future
            server-side filtering.

    Returns:
        List of :class:`~portfolio.avanza.types.Transaction`.
    """
    client = AvanzaClient.get_instance()

    tx_types: list[TransactionsDetailsType] | None = None
    if types:
        tx_types = [TransactionsDetailsType(t) for t in types]

    raw: Any = client.avanza.get_transactions_details(
        transaction_details_types=tx_types or [],
        transactions_from=date.fromisoformat(from_date),
        transactions_to=date.fromisoformat(to_date),
    )

    # The API may return a dict with a "transactions" key, or a list.
    tx_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        tx_list = raw.get("transactions", [])
    elif isinstance(raw, list):
        tx_list = raw
    else:
        tx_list = []

    transactions = [Transaction.from_api(t) for t in tx_list]

    if account_id is not None:
        transactions = [t for t in transactions if t.account_id == str(account_id)]

    logger.debug(
        "get_transactions from=%s to=%s types=%s count=%d",
        from_date,
        to_date,
        types,
        len(transactions),
    )
    return transactions

 succeeded in 817ms:
"""Thread-safe TOTP authentication singleton for Avanza.

Wraps the ``avanza-api`` library's ``Avanza`` class with a double-checked
locking singleton so that the entire application shares one authenticated
session regardless of how many threads call ``get_instance()``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("portfolio.avanza.auth")


class AuthError(Exception):
    """Raised when Avanza authentication fails."""


def _create_avanza_client(credentials: dict[str, str]) -> Any:
    """Create and return an authenticated ``avanza.Avanza`` instance.

    Separated from :class:`AvanzaAuth` to allow easy mocking in tests
    (patch ``portfolio.avanza.auth._create_avanza_client``).

    Args:
        credentials: Dict with keys ``username``, ``password``, ``totpSecret``.

    Returns:
        An authenticated ``avanza.Avanza`` instance.

    Raises:
        AuthError: If authentication fails.
    """
    try:
        from avanza import Avanza  # noqa: WPS433 — late import

        client = Avanza(credentials, quiet=True)
        return client
    except Exception as exc:
        raise AuthError(f"Avanza authentication failed: {exc}") from exc


class AvanzaAuth:
    """Thread-safe singleton managing Avanza TOTP authentication.

    Usage::

        auth = AvanzaAuth.get_instance(username, password, totp_secret)
        auth.client  # -> avanza.Avanza instance

    Call ``AvanzaAuth.reset()`` to tear down the singleton (e.g. in tests or
    on session expiry).
    """

    _instance: AvanzaAuth | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        client: Any,
        push_subscription_id: str,
        csrf_token: str,
        authentication_session: str,
        customer_id: str,
    ) -> None:
        self.client = client
        self.push_subscription_id = push_subscription_id
        self.csrf_token = csrf_token
        self.authentication_session = authentication_session
        self.customer_id = customer_id

    @classmethod
    def get_instance(
        cls,
        username: str,
        password: str,
        totp_secret: str,
    ) -> AvanzaAuth:
        """Return the singleton, creating it on first call.

        Uses double-checked locking so that only the first caller pays the
        cost of TOTP authentication; subsequent callers return immediately.
        """
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            # Double-check after acquiring the lock
            if cls._instance is not None:
                return cls._instance

            credentials = {
                "username": username,
                "password": password,
                "totpSecret": totp_secret,
            }

            client = _create_avanza_client(credentials)

            instance = cls(
                client=client,
                push_subscription_id=getattr(client, "_push_subscription_id", ""),
                csrf_token=getattr(client, "_security_token", ""),
                authentication_session=getattr(client, "_authentication_session", ""),
                customer_id=getattr(client, "_customer_id", ""),
            )
            cls._instance = instance
            logger.info(
                "AvanzaAuth singleton created (customer_id=%s)",
                instance.customer_id,
            )
            return instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton — useful for tests or re-auth."""
        with cls._lock:
            cls._instance = None
            logger.info("AvanzaAuth singleton reset")

 succeeded in 837ms:
"""Tick-size rules — price rounding for Avanza order books.

Caches tick tables per orderbook ID so repeated rounding calls do not
hit the API.  Uses integer arithmetic internally to avoid floating-point
drift.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import TickEntry

logger = logging.getLogger("portfolio.avanza.tick_rules")

# Module-level cache: ob_id -> list of TickEntry
_cache: dict[str, list[TickEntry]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tick_rules(ob_id: str) -> list[TickEntry]:
    """Fetch (and cache) the tick-size table for an orderbook.

    Args:
        ob_id: Avanza orderbook ID.

    Returns:
        List of :class:`~portfolio.avanza.types.TickEntry` sorted by
        ``min_price``.
    """
    if ob_id in _cache:
        return _cache[ob_id]

    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.get_order_book_raw(ob_id)

    tick_list_raw: list[dict[str, Any]] = raw.get("tickSizeList", raw.get("tickSizes", []))
    entries = [TickEntry.from_api(t) for t in tick_list_raw]
    entries.sort(key=lambda e: e.min_price)

    _cache[ob_id] = entries
    logger.debug("get_tick_rules ob_id=%s entries=%d (cached)", ob_id, len(entries))
    return entries


def round_to_tick(price: float, ob_id: str, direction: str = "down") -> float:
    """Round a price to the nearest valid tick.

    Uses integer arithmetic (multiply -> floor/ceil -> divide) to avoid
    floating-point drift.

    Args:
        price: The price to round.
        ob_id: Avanza orderbook ID (needed to fetch the tick table).
        direction: ``"down"`` (floor) or ``"up"`` (ceil).

    Returns:
        The rounded price.

    Raises:
        ValueError: If *direction* is not ``"down"`` or ``"up"``.
        ValueError: If no tick rule matches *price*.
    """
    if direction not in ("down", "up"):
        raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")

    entries = get_tick_rules(ob_id)
    tick = _find_tick_for_price(price, entries)

    if tick is None:
        raise ValueError(f"No tick rule found for price {price} (ob_id={ob_id})")

    # Integer arithmetic to avoid float drift:
    # steps = price / tick  ->  round to int  ->  result = steps * tick
    # We use a precision multiplier derived from the tick's decimal places.
    precision = _decimal_places(tick)
    multiplier = 10 ** precision

    # Convert to integer domain
    price_int = price * multiplier
    tick_int = round(tick * multiplier)

    if tick_int == 0:
        return price  # degenerate tick; return unchanged

    if direction == "down":
        steps = math.floor(price_int / tick_int)
    else:
        steps = math.ceil(price_int / tick_int)

    result = (steps * tick_int) / multiplier
    return round(result, precision)


def clear_cache() -> None:
    """Clear the module-level tick-rule cache."""
    _cache.clear()
    logger.debug("tick_rules cache cleared")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_tick_for_price(price: float, entries: list[TickEntry]) -> float | None:
    """Find the tick size applicable for *price*.

    Returns ``None`` if no entry matches.
    """
    for entry in entries:
        if entry.min_price <= price <= entry.max_price:
            return entry.tick_size
        # Handle unbounded upper range (max_price == 0 means infinity)
        if entry.min_price <= price and entry.max_price == 0:
            return entry.tick_size
    # Fallback: if price exceeds all ranges, use the last entry
    if entries:
        return entries[-1].tick_size
    return None


def _decimal_places(value: float) -> int:
    """Count the number of significant decimal places in *value*."""
    s = f"{value:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0

 succeeded in 853ms:
"""Singleton HTTP client wrapping the avanza-api library.

Provides raw delegator methods that return whatever the underlying
``avanza.Avanza`` instance returns.  Typed higher-level modules (market
data, trading, account, etc.) will wrap these delegators and return our
own dataclasses from :mod:`portfolio.avanza.types`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from portfolio.avanza.auth import AvanzaAuth

logger = logging.getLogger("portfolio.avanza.client")

DEFAULT_ACCOUNT_ID = "1625505"


class AvanzaClient:
    """Singleton client wrapping the avanza-api library.

    Usage::

        client = AvanzaClient.get_instance(config)
        raw = client.get_market_data_raw("2213050")
    """

    _instance: AvanzaClient | None = None
    _lock = threading.Lock()

    def __init__(self, auth: AvanzaAuth, account_id: str) -> None:
        self._auth = auth
        self._account_id = account_id

    @classmethod
    def get_instance(cls, config: dict[str, Any] | None = None) -> AvanzaClient:
        """Return the singleton, creating it on first call.

        Args:
            config: Application config dict.  Must contain an ``"avanza"`` key
                with ``"username"``, ``"password"``, and ``"totp_secret"`` when
                creating for the first time.  Ignored on subsequent calls.
        """
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            if config is None:
                raise ValueError(
                    "AvanzaClient.get_instance() requires config on first call"
                )

            avanza_cfg = config.get("avanza", {})
            auth = AvanzaAuth.get_instance(
                username=avanza_cfg["username"],
                password=avanza_cfg["password"],
                totp_secret=avanza_cfg["totp_secret"],
            )
            account_id = str(avanza_cfg.get("account_id", DEFAULT_ACCOUNT_ID))

            instance = cls(auth=auth, account_id=account_id)
            cls._instance = instance
            logger.info(
                "AvanzaClient singleton created (account_id=%s)", account_id
            )
            return instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton."""
        with cls._lock:
            cls._instance = None
            logger.info("AvanzaClient singleton reset")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def avanza(self) -> Any:
        """The underlying ``avanza.Avanza`` instance."""
        return self._auth.client

    @property
    def push_subscription_id(self) -> str:
        return self._auth.push_subscription_id

    @property
    def csrf_token(self) -> str:
        return self._auth.csrf_token

    @property
    def session(self) -> Any:
        """The underlying ``requests.Session`` used by the avanza-api library."""
        return self._auth.client._session

    # ------------------------------------------------------------------
    # Raw delegators — return whatever the library returns
    # ------------------------------------------------------------------

    def get_positions_raw(self) -> Any:
        return self.avanza.get_accounts_positions()

    def get_overview_raw(self) -> Any:
        return self.avanza.get_overview()

    def get_market_data_raw(self, ob_id: str) -> Any:
        return self.avanza.get_market_data(ob_id)

    def get_order_book_raw(self, ob_id: str) -> Any:
        return self.avanza.get_order_book(ob_id)

    def get_deals_raw(self) -> Any:
        return self.avanza.get_deals()

    def get_orders_raw(self) -> Any:
        return self.avanza.get_orders()

    def get_all_stop_losses_raw(self) -> Any:
        return self.avanza.get_all_stop_losses()

    def get_news_raw(self, ob_id: str) -> Any:
        return self.avanza.get_news(ob_id)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'python -m compileall portfolio' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 722ms:
Listing 'portfolio'...
Listing 'portfolio\\avanza'...
Compiling 'portfolio\\avanza\\__init__.py'...
Compiling 'portfolio\\avanza\\account.py'...
Compiling 'portfolio\\avanza\\auth.py'...
Compiling 'portfolio\\avanza\\client.py'...
Compiling 'portfolio\\avanza\\market_data.py'...
Compiling 'portfolio\\avanza\\scanner.py'...
Compiling 'portfolio\\avanza\\search.py'...
Compiling 'portfolio\\avanza\\streaming.py'...
Compiling 'portfolio\\avanza\\tick_rules.py'...
Compiling 'portfolio\\avanza\\trading.py'...
Compiling 'portfolio\\avanza\\types.py'...
Compiling 'portfolio\\avanza_client.py'...
Compiling 'portfolio\\avanza_control.py'...
Compiling 'portfolio\\avanza_order_lock.py'...
Compiling 'portfolio\\avanza_orders.py'...
Compiling 'portfolio\\avanza_resilient_page.py'...
Compiling 'portfolio\\avanza_session.py'...
Compiling 'portfolio\\avanza_tracker.py'...

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from portfolio.avanza.market_data import get_quote
try:
    get_quote('123')
except Exception as e:
    print(type(e).__name__, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-avanza-api
2026-05-08T15:29:20.328632Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__init__.py", line 11, in <module>
    from portfolio.avanza.account import get_buying_power, get_positions, get_transactions
  File "Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\account.py", line 14, in <module>
    from avanza.constants import TransactionsDetailsType
ModuleNotFoundError: No module named 'avanza'

 exited 1 in 625ms:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__init__.py", line 11, in <module>
    from portfolio.avanza.account import get_buying_power, get_positions, get_transactions
  File "Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\account.py", line 14, in <module>
    from avanza.constants import TransactionsDetailsType
ModuleNotFoundError: No module named 'avanza'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File -Include requirements*.txt,pyproject.toml,setup.py,setup.cfg | ForEach-Object { Write-Output ('### ' + "'$_.FullName); Select-String -Path $_.FullName -Pattern '"'avanza|filelock|websocket' -CaseSensitive:"'$false }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 582ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/avanza_session.py -Pattern '"'^def |''^class '"'" in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 705ms:

portfolio\avanza_session.py:55:[7mclass [0mAvanzaSessionError(Exception):
portfolio\avanza_session.py:59:[7mdef [0mload_session() -> dict:
portfolio\avanza_session.py:101:[7mdef [0msession_remaining_minutes() -> float | None:
portfolio\avanza_session.py:118:[7mdef [0mis_session_expiring_soon(threshold_minutes: float = 60.0) -> bool:
portfolio\avanza_session.py:129:[7mdef [0m_get_playwright_context():
portfolio\avanza_session.py:151:[7mdef [0mclose_playwright():
portfolio\avanza_session.py:175:[7mdef [0mverify_session() -> bool:
portfolio\avanza_session.py:207:[7mdef [0m_with_browser_recovery(op: Callable[[Any], Any], *, op_name: str) -> Any:
portfolio\avanza_session.py:233:[7mdef [0mapi_get(path: str, **kwargs) -> Any:
portfolio\avanza_session.py:266:[7mdef [0m_get_csrf(ctx=None) -> str:
portfolio\avanza_session.py:289:[7mdef [0mapi_post(path: str, payload: dict) -> Any:
portfolio\avanza_session.py:342:[7mdef [0mapi_delete(path: str) -> Any:
portfolio\avanza_session.py:380:[7mdef [0mget_buying_power(account_id: str | None = None) -> dict | None:
portfolio\avanza_session.py:537:[7mdef [0mplace_buy_order(
portfolio\avanza_session.py:559:[7mdef [0mplace_sell_order(
portfolio\avanza_session.py:570:[7mdef [0m_place_order(
portfolio\avanza_session.py:628:[7mdef [0mcancel_order(order_id: str, account_id: str | None = None) -> dict:
portfolio\avanza_session.py:643:[7mdef [0mget_open_orders(account_id: str | None = None) -> list[dict]:
portfolio\avanza_session.py:662:[7mdef [0mget_quote(orderbook_id: str) -> dict:
portfolio\avanza_session.py:671:[7mdef [0mget_positions() -> list[dict]:
portfolio\avanza_session.py:715:[7mdef [0mplace_stop_loss(
portfolio\avanza_session.py:809:[7mdef [0mplace_trailing_stop(
portfolio\avanza_session.py:844:[7mdef [0mget_stop_losses() -> list[dict]:
portfolio\avanza_session.py:862:[7mdef [0mget_stop_losses_strict() -> list[dict]:
portfolio\avanza_session.py:883:[7mdef [0mcancel_stop_loss(stop_id: str, account_id: str | None = None) -> dict:
portfolio\avanza_session.py:929:[7mdef [0mcancel_all_stop_losses_for(
portfolio\avanza_session.py:1114:[7mdef [0mrearm_stop_losses_from_snapshot(snapshot: list[dict]) -> dict:
portfolio\avanza_session.py:1222:[7mdef [0mget_instrument_price(orderbook_id: str) -> dict[str, Any]:


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_session.py | Select-Object -Index (220..370)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 625ms:
            logger.warning(
                "avanza_session: browser dead on %s (%r) — teardown + relaunch + retry",
                op_name, exc,
            )
            close_playwright()
            ctx = _get_playwright_context()
            return op(ctx)


# --- API convenience functions ---


def api_get(path: str, **kwargs) -> Any:
    """Make an authenticated GET request to Avanza API.

    Args:
        path: API path (e.g., "/_api/position-data/positions")

    Returns:
        Parsed JSON response.

    Raises:
        AvanzaSessionError: if session is invalid.
    """
    # A-AV-1: Hold _pw_lock for the entire request. Playwright's sync_api
    # is NOT thread-safe and the metals fast-tick + main 8-worker pool race.
    # 2026-04-13: Wrapped in _with_browser_recovery so TargetClosedError
    # (browser died mid-flight) triggers a teardown + relaunch + retry.
    url = f"{API_BASE}{path}" if path.startswith("/") else path

    def _op(ctx):
        resp = ctx.request.get(url)
        if resp.status == 401:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 401 Unauthorized. "
                "Run: python scripts/avanza_login.py"
            )
        if not resp.ok:
            raise RuntimeError(f"Avanza API error {resp.status}: {resp.text()[:500]}")
        return resp.json()

    return _with_browser_recovery(_op, op_name=f"GET {path}")


def _get_csrf(ctx=None) -> str:
    """Extract CSRF token from Playwright context cookies.

    If ``ctx`` is provided (e.g. from inside an already-locked _with_recovery
    block) it is used directly — avoids re-entering the RLock and avoids a
    stale context reference after a relaunch. Otherwise acquires the lock
    and fetches a fresh context.
    """
    if ctx is not None:
        for c in ctx.cookies():
            if c["name"] == "AZACSRF":
                return c["value"]
        raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")

    # A-AV-1: ctx.cookies() reads Playwright internal state — needs lock.
    with _pw_lock:
        ctx = _get_playwright_context()
        for c in ctx.cookies():
            if c["name"] == "AZACSRF":
                return c["value"]
        raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")


def api_post(path: str, payload: dict) -> Any:
    """Make an authenticated POST request to Avanza API.

    Automatically includes the X-SecurityToken (CSRF) header.

    Args:
        path: API path (e.g., "/_api/trading-critical/rest/order/new")
        payload: Request body dict.

    Returns:
        Parsed JSON response.
    """
    # A-AV-1: Hold lock across CSRF read + POST so a concurrent request
    # cannot rotate the cookie jar mid-flight.
    # 2026-04-13: Wrapped in _with_browser_recovery. CSRF is read from the
    # same ctx used for the POST, so a relaunch picks up fresh cookies in
    # both places atomically (no stale-CSRF-against-fresh-context mismatch).
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    body_data = json.dumps(payload)

    def _op(ctx):
        csrf = _get_csrf(ctx)
        resp = ctx.request.post(
            url,
            data=body_data,
            headers={
                "Content-Type": "application/json",
                "X-SecurityToken": csrf,
            },
        )
        if resp.status == 401:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 401 Unauthorized. "
                "Run: python scripts/avanza_login.py"
            )
        if resp.status == 403:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 403 Forbidden — CSRF token may be stale. "
                "Run: python scripts/avanza_login.py"
            )
        body = resp.text()
        try:
            return json.loads(body)
        except (json.JSONDecodeError, TypeError):
            if not resp.ok:
                raise RuntimeError(f"Avanza API error {resp.status}: {body[:500]}") from None
            return {"raw": body}

    return _with_browser_recovery(_op, op_name=f"POST {path}")


def api_delete(path: str) -> Any:
    """Make an authenticated DELETE request to Avanza API.

    Automatically includes the X-SecurityToken (CSRF) header.

    Args:
        path: API path (e.g., "/_api/trading/stoploss/{stop_id}")

    Returns:
        Dict with ``http_status`` and ``ok`` keys.
    """
    # A-AV-1: Hold lock across CSRF read + DELETE.
    # 2026-04-13: Wrapped in _with_browser_recovery (see api_get/api_post).
    url = f"{API_BASE}{path}" if path.startswith("/") else path

    def _op(ctx):
        csrf = _get_csrf(ctx)
        resp = ctx.request.delete(
            url,
            headers={
                "Content-Type": "application/json",
                "X-SecurityToken": csrf,
            },
        )
        if resp.status == 401:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 401 Unauthorized. "
                "Run: python scripts/avanza_login.py"
            )

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_session.py | Select-Object -Index (370..430)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 632ms:
            )
        return {"http_status": resp.status, "ok": 200 <= resp.status < 300 or resp.status == 404}

    return _with_browser_recovery(_op, op_name=f"DELETE {path}")


# --- Trading convenience functions ---


def get_buying_power(account_id: str | None = None) -> dict | None:
    """Get buying power and account value for an account.

    2026-04-09 (Bug C7 fix): ported the multi-shape + multi-field-ID fallback
    pattern from ``data/metals_avanza_helpers.fetch_account_cash`` after Avanza
    changed the ``/_api/account-overview/overview/categorizedAccounts`` response
    shape mid-day. The endpoint used to return a single top-level key
    ``categorizedAccounts`` (an array of categories each with an ``accounts``
    child). The new shape exposes three top-level keys simultaneously:
    ``categories`` (new categorized path), ``accounts`` (flat list of all user
    accounts), and ``loans``. At the same time, the per-account ID field
    renamed from ``accountId`` to ``id`` (the other Avanza endpoints such as
    ``position-data/positions`` already use ``id`` — see ``get_positions``).

    Previously this function assumed the legacy shape + legacy ID field, so on
    the new shape the iteration walked an empty list, then hit ``cats[0]`` on
    an empty list (IndexError) or — if the shape still exposed the legacy key
    but with no matches — silently returned fake numbers derived from the
    first category's totalValue. That made callers like ``fish_straddle`` and
    ``fish_monitor_live`` size positions off wrong cash balances.

    We now try all three shapes (legacy categorized → flat → new categorized)
    and all four known ID fields (``accountId``, ``id``, ``accountNumber``,
    ``number``), taking whichever finds the target account first. On any
    failure path we return ``None`` so callers can distinguish "API call failed"
    from "balance is legitimately zero" — callers must now explicitly handle
    the ``None`` case (previously they silently got ``buying_power=0``, which
    was a dangerous silent failure).

    Args:
        account_id: Avanza account ID (default: ``DEFAULT_ACCOUNT_ID``).

    Returns:
        Dict with ``buying_power``, ``total_value``, ``own_capital`` (all SEK)
        on success. ``None`` on any failure (HTTP error, account not found,
        shape drift, etc.). Failures are logged with enough diagnostic context
        (sample keys, counts per shape) to identify the next shape drift
        without guessing.
    """
    aid = str(account_id or DEFAULT_ACCOUNT_ID)

    try:
        data = api_get("/_api/account-overview/overview/categorizedAccounts")
    except Exception as e:
        logger.warning(
            "get_buying_power: api_get raised account_id=%s exception=%r",
            aid, e,
        )
        return None

    if not isinstance(data, dict):
        logger.warning(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern '\\bfrom avanza\\b|\\bimport avanza\\b|avanza-api' -CaseSensitive:"'$false | ForEach-Object { "$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 687ms:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:6: 2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:34: # Singleton client instance (avanza-api library)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:88: Authenticated Avanza client instance (avanza-api library)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:97: from avanza import Avanza
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:100: "avanza-api package not installed. Run: pip install avanza-api"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:243: """Get the trading account ID from Avanza overview (cached).
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:302: from avanza.constants import OrderType
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:323: from avanza.constants import OrderType
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\account.py:14: from avanza.constants import TransactionsDetailsType
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\auth.py:3: Wraps the ``avanza-api`` library's ``Avanza`` class with a double-checked
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\auth.py:37: from avanza import Avanza  # noqa: WPS433 — late import
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\client.py:1: """Singleton HTTP client wrapping the avanza-api library.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\client.py:23: """Singleton client wrapping the avanza-api library.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\client.py:104: """The underlying ``requests.Session`` used by the avanza-api library."""
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\market_data.py:13: from avanza.constants import Resolution, TimePeriod
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\scanner.py:59: from avanza.constants import InstrumentType
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\search.py:12: from avanza.constants import InstrumentType
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:13: from avanza.constants import (
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:19: from avanza.entities import StopLossOrderEvent, StopLossTrigger
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\auth.cpython-312.pyc:7: Wraps the ``avanza-api`` library's ``Avanza`` class with a double-checked
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\client.cpython-312.pyc:5: a?  Singleton HTTP client wrapping the avanza-api library.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\client.cpython-312.pyc:22: �Zdd�Zdd�Zdd�Zdd�Zdd�Zdd�Zdd�Zy)�AvanzaClientz�Singleton client wrapping the avanza-api library.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\client.cpython-312.pyc:47: csrf_tokenr(   s    r   r.   zAvanzaClient.csrf_tokenb   s   � ��z�z�$�$�$r   c                �B   � | j                   j                  j                  S )zCThe underlying ``requests.Session`` used by the avanza-api library.)r   r*   �_sessionr(   s    r   �sessionzAvanzaClient.sessionf   s   � � �z�z� � �)�)�)r   c                �6   � | j                   j                  �       S r   )r   �get_accounts_positionsr(   s    r   �get_positions_rawzAvanzaClient.get_positions_rawo   s   � ��{�{�1�1�3�3r   c                �6   � | j                   j                  �       S r   )r   �get_overviewr(   s    r   �get_overview_rawzAvanzaClient.get_overview_rawr   s   � ��{�{�'�'�)�)r   c                �8   � | j                   j                  |�      S r   )r   �get_market_data�r   �ob_ids     r   �get_market_data_rawz AvanzaClient.get_market_data_rawu   s   � ��{�{�*�*�5�1�1r   c                �8   � | j                   j                  |�      S r   )r   �get_order_bookr:   s     r   �get_order_book_rawzAvanzaClient.get_order_book_rawx   s   � ��{�{�)�)�%�0�0r   c                �6   � | j                   j                  �       S r   )r   �	get_dealsr(   s    r   �
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:14: 2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:41: Authenticated Avanza client instance (avanza-api library)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:45: Nr   )�Avanzaz=avanza-api package not installed. Run: pip install avanza-apir
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:114: a�  Get the trading account ID from Avanza overview (cached).

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'AvanzaClient\\.get_instance\\(' | ForEach-Object { \""'$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 632ms:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\account.py:37: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\account.py:74: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\account.py:116: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\client.py:27: client = AvanzaClient.get_instance(config)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\client.py:56: "AvanzaClient.get_instance() requires config on first call"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\market_data.py:54: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\market_data.py:66: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\market_data.py:88: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\market_data.py:126: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\market_data.py:138: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\scanner.py:55: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\search.py:42: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\tick_rules.py:41: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:80: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:128: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:159: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:174: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:194: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:239: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:328: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:351: client = AvanzaClient.get_instance()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\client.cpython-312.pyc:26: client = AvanzaClient.get_instance(config)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\client.cpython-312.pyc:39: Nz9AvanzaClient.get_instance() requires config on first call�avanza�username�password�totp_secret)r   r   r   r   )r   r   z.AvanzaClient singleton created (account_id=%s))

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_session.py | Select-Object -Index (660..740)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 704ms:

def get_quote(orderbook_id: str) -> dict:
    """Get bid/ask/last quote for an instrument. Fast single-endpoint call.

    Returns:
        Dict with buy, sell, last, changePercent, highest, lowest.
    """
    return api_get(f"/_api/market-guide/stock/{orderbook_id}/quote")


def get_positions() -> list[dict]:
    """Get all positions via session-based auth.

    Returns:
        List of position dicts with name, value, profit, etc.
    """
    data = api_get("/_api/position-data/positions")
    positions = []
    for entry in data.get("withOrderbook", []):
        inst = entry.get("instrument", {})
        orderbook = inst.get("orderbook", {})
        quote = orderbook.get("quote", {})
        volume_obj = entry.get("volume", {})
        value_obj = entry.get("value", {})
        acquired_obj = entry.get("acquiredValue", {})
        account = entry.get("account", {})

        vol = volume_obj.get("value", 0) if isinstance(volume_obj, dict) else volume_obj
        val = value_obj.get("value", 0) if isinstance(value_obj, dict) else value_obj
        acq = acquired_obj.get("value", 0) if isinstance(acquired_obj, dict) else acquired_obj
        latest = quote.get("latest", {})
        last_price = latest.get("value", 0) if isinstance(latest, dict) else latest
        change_pct_obj = quote.get("changePercent", {})
        change_pct = change_pct_obj.get("value", 0) if isinstance(change_pct_obj, dict) else change_pct_obj

        positions.append({
            "name": inst.get("name", orderbook.get("name", "")),
            "orderbook_id": str(orderbook.get("id", "")),
            "instrument_id": str(inst.get("id", "")),
            "type": inst.get("type", orderbook.get("type", "")),
            "volume": vol,
            "value": val,
            "acquired_value": acq,
            "profit": val - acq if val and acq else 0,
            "profit_percent": ((val - acq) / acq * 100) if acq else 0,
            "currency": inst.get("currency", "SEK"),
            "last_price": last_price,
            "change_percent": change_pct,
            "account_id": account.get("id", ""),
            "account_type": account.get("type", ""),
        })
    return positions


def place_stop_loss(
    orderbook_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    account_id: str | None = None,
    valid_days: int = 8,
    trigger_type: str = "LESS_OR_EQUAL",
    value_type: str = "MONETARY",
) -> dict:
    """Place a hardware stop-loss order on Avanza.

    IMPORTANT: Uses /_api/trading/stoploss/new, NOT the regular order API.

    Args:
        orderbook_id: Avanza orderbook ID.
        trigger_price: Price at which to trigger the stop-loss.
            For FOLLOW_DOWNWARDS with PERCENTAGE, this is the trail %.
        sell_price: Price to sell at when triggered.
            For trailing stops (FOLLOW_DOWNWARDS), set to 0 (market).
        volume: Number of units to sell.
        account_id: Defaults to DEFAULT_ACCOUNT_ID.
        valid_days: Days until the stop-loss expires (default 8).
        trigger_type: LESS_OR_EQUAL, MORE_OR_EQUAL, FOLLOW_DOWNWARDS, FOLLOW_UPWARDS.
        value_type: MONETARY (absolute price) or PERCENTAGE.

    Returns:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import math
for price,tick in [(0.35,0.05),(0.3,0.05),(0.1,0.01),(0.29,0.01),(0.14,0.02),(2.1,0.1),(2.3,0.1)]:
    precision = len(f'{tick:.10f}'.rstrip('0').split('.')[1]) if '.' in f'{tick:.10f}'.rstrip('0') else 0
    m = 10**precision
    price_int = price*m
    tick_int = round(tick*m)
    down = round((math.floor(price_int/tick_int)*tick_int)/m,precision)
    up = round((math.ceil(price_int/tick_int)*tick_int)/m,precision)
    print(price,tick,repr(price_int),tick_int,down,up)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 761ms:
0.35 0.05 35.0 5 0.35 0.35
0.3 0.05 30.0 5 0.3 0.3
0.1 0.01 10.0 1 0.1 0.1
0.29 0.01 28.999999999999996 1 0.28 0.29
0.14 0.02 14.000000000000002 2 0.14 0.16
2.1 0.1 21.0 1 2.1 2.1
2.3 0.1 23.0 1 2.3 2.3

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/file_utils.py -Pattern '"'^def load_json|''^def atomic_write_json'"' -Context 0,40" in Q:\finance-analyzer\.worktrees\adv-avanza-api
2026-05-08T15:30:40.049723Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mSelect-String: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mSelect-String -Path portfolio/file_utils.py -Pattern '^def load_json|[0m …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\file_utils.py' because it does not exist.[0m

 exited 1 in 779ms:
[31;1mSelect-String: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mSelect-String -Path portfolio/file_utils.py -Pattern '^def load_json|[0m …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\file_utils.py' because it does not exist.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Force portfolio' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 678ms:

    Directory: Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    17:29                [44;1m__pycache__[0m
d----          2026-05-08    17:29                [44;1mavanza[0m
-a---          2026-05-08    17:25          13438 avanza_client.py
-a---          2026-05-08    17:25          15330 avanza_control.py
-a---          2026-05-08    17:25           3643 avanza_order_lock.py
-a---          2026-05-08    17:25          16708 avanza_orders.py
-a---          2026-05-08    17:25           7995 avanza_resilient_page.py
-a---          2026-05-08    17:25          47835 avanza_session.py
-a---          2026-05-08    17:25           4184 avanza_tracker.py


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -File | Where-Object { $_.Name -in @('"'file_utils.py','http_retry.py','telegram_notifications.py') } | Select-Object FullName" in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 710ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'metals_avanza_helpers|portfolio\\.file_utils|portfolio\\.http_retry|portfolio\\.telegram_notifications' | ForEach-Object { \""'$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 767ms:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:18: from portfolio.file_utils import load_json
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:332: ``data/metals_avanza_helpers.place_order``,
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:343: ``data/metals_avanza_helpers.place_order``.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:17: from data.metals_avanza_helpers import (
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:21: from data.metals_avanza_helpers import (
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:24: from data.metals_avanza_helpers import (
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:27: from data.metals_avanza_helpers import (
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:30: from data.metals_avanza_helpers import (
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:33: from data.metals_avanza_helpers import (
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:173: # 2026-04-13: cross-process order lock (see metals_avanza_helpers.place_order).
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_orders.py:33: from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_orders.py:34: from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_orders.py:35: from portfolio.telegram_notifications import send_telegram
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:21: from portfolio.file_utils import load_json
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:384: pattern from ``data/metals_avanza_helpers.fetch_account_cash`` after Avanza
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_tracker.py:21: from portfolio.file_utils import load_json
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:162: ``data/metals_avanza_helpers.place_order``,
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:173: ``data/metals_avanza_helpers.place_order``.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:203:    N)N)/�__doc__�logging�datetimer   �pathlibr   �typingr   �portfolio.avanza_order_lockr   �portfolio.file_utilsr   �	getLoggerr   �__file__�resolve�parent�BASE_DIRr   r	   �setrN   �__annotations__r&   r   �dictr   �boolr"   r(   r+   r-   �listr3   r9   r;   �floatrV   r\   r]   rX   �intrp   rs   rm   r�   r�   r*   r   r   �<module>r�      s�  ��	� � � � � 9� *�	��	�	�4�	5����>�!�!�#�*�*�1�1����&�� "+�� �S��X� +� �����4� �2�4� �"�:���3� �4��:� ��C� �D��c��N� �2&�t�D�z� &�R�U� �&�� �" ��S�4�Z� �%�� %�^  $�	Q��Q��Q� 
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_control.cpython-312.pyc:170: __future__r   re   �logging�portfolio.avanza_order_lockr   �	getLoggerrj   �data.metals_avanza_helpersr   r   r   r@   r   rE   r	   r-   r
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_orders.cpython-312.pyc:189: �4��g��s�3� 4+�+9�*:�%�	A� � �K�K�#�U�4�[�1�D��c�6�"��� D����?��C�C��D�s   �A �	A4�A/�/A4)0�__doc__rm   �logging�rer   r9   r   r   r   �pathlibr   �portfolio.avanza_controlr   r   �portfolio.file_utilsr	   r
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_orders.cpython-312.pyc:190:    �portfolio.http_retryr   � portfolio.telegram_notificationsr   �	getLoggerr   �__file__�resolve�parentrh   r   r=   r   �compilerz   ru   r8   r   �listrj   r    r#   rk   �floatrB   rE   rW   rg   rK   rP   rQ   r   r   r   �<module>r�      s�  ���. � � 	� � � -� -� � F� =� 1� :�	��	�	�4�	5����>�!�!�#�*�*�1�1�F�:���6�6���� � �
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:163: pattern from ``data/metals_avanza_helpers.fetch_account_cash`` after Avanza
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:614:    �portfolio.avanza_resilient_pager   �portfolio.file_utilsr   �	getLoggerr"   �__file__�resolve�parent�BASE_DIR�DATA_DIRr   r$   rJ   �EXPIRY_BUFFER_MINUTESr�   r�   �RLockr7   r;   r>   r8   r*   r   r�   r&   ri  r,   �boolr0   rA   rF   rN   r@   rS   rg   rn   r~   r�   r�   r=  r�   r�   r�   r�   r�   r�   r�   r  r+  r0  r4  r6  r>  r`  rq  rx  r   r   r   �<module>r�     s�  ��� � � � � $� 3� 3� � � 9� A� *�	��	�	�5�	6����>�!�!�#�*�*�1�1���f����/�/��� ;�;� �"�� � � � � !�k� � �9�?�?���������?�� ?�'�d� '�T�5�4�<� �")�� )�� )��, �0�� �@�x���s�
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_tracker.cpython-312.pyc:71: � �� � ���s   �5 �	A� A)�__doc__�logging�pathlibr   �typingr   �portfolio.file_utilsr   �	getLoggerr   �__file__�resolve�parent�BASE_DIRr   �dict�strr   r*   r.   r0   r:   � r   r   �<module>rH      s�   ���  � � � *�	��	�	�5�	6����>�!�!�#�*�*�1�1����&��	;��c�4�i�� 	;�!�T�#�t�C��H�~�"5�6� !�H

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
try:
    import portfolio.avanza_client
    print('ok')
except Exception as e:
    import traceback
    traceback.print_exc()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 741ms:
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py", line 17, in <module>
    from portfolio.avanza_order_lock import avanza_order_lock
  File "Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_order_lock.py", line 43, in <module>
    import filelock
ModuleNotFoundError: No module named 'filelock'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_resilient_page.py' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 604ms:
"""Auto-recovering Playwright Page wrapper for Avanza-bound loops.

Problem: long-running loops (`data/metals_loop.py`, `portfolio/golddigger/`,
`portfolio/main.py` via `avanza_session.py`) open a headless Chromium at
startup and hold the `page` reference for days. When the browser dies
(OS sleep, memory pressure, WSL ping hiccup, external BankID re-auth) the
Python process keeps running but every `page.evaluate()` throws
`playwright._impl._errors.TargetClosedError` — silently, for days. The
bug discovered 2026-04-13: metals loop was emitting this error 662 times
between 2026-04-09 and 2026-04-13, making zero trades.

Solution: pass a `ResilientPage` instead of a raw Playwright `Page`. On
`TargetClosedError` (or equivalent browser-dead message) the wrapper tears
down the dead browser+context, relaunches Chromium, reloads the saved
`avanza_storage_state.json`, and retries the failing call once. Only then
does it propagate the error.

This is the minimal surface — `evaluate()` and `context.cookies()` — that
the existing helpers use. Other Page methods pass through unchanged via
`__getattr__`; they get no auto-recovery (good enough: they're only used
during startup, where crash-and-bat-restart is acceptable).

Usage:

    with sync_playwright() as pw:
        page = ResilientPage.open(pw, "data/avanza_storage_state.json")
        # pass `page` to existing helpers — zero call-site changes needed
        fetch_price(page, ob_id, api_type)
        fetch_account_cash(page, account_id)
        # on shutdown:
        page.close()
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

logger = logging.getLogger("portfolio.avanza_resilient_page")

_INITIAL_URL = "https://www.avanza.se/min-ekonomi/oversikt.html"
_INITIAL_URL_WAIT_MS = 2000


def is_browser_dead_error(exc: BaseException) -> bool:
    """True if ``exc`` signals a dead Playwright browser/context.

    Checks both the exception class name (Playwright's
    ``TargetClosedError`` name changed across versions) and the message
    (the stable cross-version signal). Exposed for tests and for
    ``avanza_session.py`` which wants the same classifier without
    importing Playwright internals.
    """
    name = type(exc).__name__
    if name == "TargetClosedError":
        return True
    msg = str(exc)
    for marker in (
        "Target page, context or browser has been closed",
        "Target closed",
        "Browser has been closed",
        "has been closed",
    ):
        if marker in msg:
            return True
    return False


class ResilientPage:
    """Playwright ``Page`` wrapper that auto-relaunches the browser on death."""

    def __init__(
        self,
        pw: Any,
        storage_state_path: str,
        *,
        headless: bool = True,
        locale: str = "sv-SE",
        initial_url: str | None = _INITIAL_URL,
        initial_url_wait_ms: int = _INITIAL_URL_WAIT_MS,
    ) -> None:
        self._pw = pw
        self._storage_state_path = storage_state_path
        self._headless = headless
        self._locale = locale
        self._initial_url = initial_url
        self._initial_url_wait_ms = initial_url_wait_ms
        self._browser = None
        self._ctx = None
        self._page = None
        self._relaunch_count = 0
        self._last_relaunch_ts: str | None = None

    @classmethod
    def open(
        cls,
        pw: Any,
        storage_state_path: str,
        *,
        headless: bool = True,
        locale: str = "sv-SE",
        initial_url: str | None = _INITIAL_URL,
        initial_url_wait_ms: int = _INITIAL_URL_WAIT_MS,
    ) -> ResilientPage:
        """Construct and open the browser. Preferred entry point."""
        rp = cls(
            pw,
            storage_state_path,
            headless=headless,
            locale=locale,
            initial_url=initial_url,
            initial_url_wait_ms=initial_url_wait_ms,
        )
        rp._open()
        return rp

    def _open(self) -> None:
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._ctx = self._browser.new_context(
            storage_state=self._storage_state_path,
            locale=self._locale,
        )
        self._page = self._ctx.new_page()
        if self._initial_url:
            self._page.goto(self._initial_url, wait_until="domcontentloaded")
            if self._initial_url_wait_ms:
                self._page.wait_for_timeout(self._initial_url_wait_ms)

    def _close_quietly(self) -> None:
        for closer in (self._ctx, self._browser):
            if closer is None:
                continue
            try:
                closer.close()
            except Exception as exc:
                logger.debug("ResilientPage teardown: %s", exc)
        self._ctx = None
        self._browser = None
        self._page = None

    def _relaunch(self, *, reason: str) -> None:
        self._relaunch_count += 1
        self._last_relaunch_ts = datetime.datetime.now(datetime.UTC).isoformat()
        logger.warning(
            "ResilientPage: browser dead (%s) — relaunching (count=%d)",
            reason, self._relaunch_count,
        )
        self._close_quietly()
        self._open()

    def close(self) -> None:
        """Teardown browser. Safe to call multiple times."""
        self._close_quietly()

    # --- Recovery-aware proxy API ---

    def evaluate(self, script: str, arg: Any = None) -> Any:
        """``page.evaluate(script, arg)`` with one-shot auto-recovery.

        On ``TargetClosedError`` (or equivalent), teardown + relaunch +
        retry. If the retry also fails with a browser-dead error, propagate.
        """
        try:
            if arg is None:
                return self._page.evaluate(script)
            return self._page.evaluate(script, arg)
        except Exception as exc:
            if not is_browser_dead_error(exc):
                raise
            self._relaunch(reason="evaluate")
            if arg is None:
                return self._page.evaluate(script)
            return self._page.evaluate(script, arg)

    def goto(self, *args, **kwargs) -> Any:
        """``page.goto()`` with one-shot auto-recovery."""
        try:
            return self._page.goto(*args, **kwargs)
        except Exception as exc:
            if not is_browser_dead_error(exc):
                raise
            self._relaunch(reason="goto")
            return self._page.goto(*args, **kwargs)

    @property
    def context(self):
        """Return a context proxy whose ``cookies()`` auto-recovers."""
        return _ResilientContextProxy(self)

    # Passthrough for everything else (wait_for_timeout, locator, on, etc.)
    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup fails — so this does
        # NOT shadow evaluate/goto/context/close defined above.
        if name.startswith("_"):
            raise AttributeError(name)
        target = self.__dict__.get("_page")
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)

    # --- Observability ---

    @property
    def relaunch_count(self) -> int:
        return self._relaunch_count

    @property
    def last_relaunch_ts(self) -> str | None:
        return self._last_relaunch_ts


class _ResilientContextProxy:
    """Proxy for ``BrowserContext`` that auto-recovers ``cookies()`` calls."""

    def __init__(self, resilient_page: ResilientPage) -> None:
        self._rp = resilient_page

    def cookies(self) -> list[dict]:
        try:
            return self._rp._ctx.cookies()
        except Exception as exc:
            if not is_browser_dead_error(exc):
                raise
            self._rp._relaunch(reason="context.cookies")
            return self._rp._ctx.cookies()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._rp._ctx, name)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_control.py | Select-Object -Index (340..390)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 593ms:
        ValueError: If *side* is not "BUY" or "SELL" (C2 fail-safe).
    """
    normalized_side = (side or "").strip().upper()
    if normalized_side not in ("BUY", "SELL"):
        raise ValueError(
            f"Invalid order side: {side!r} (must be 'BUY' or 'SELL')"
        )
    if normalized_side == "BUY":
        result = _place_buy_order(ob_id, price, volume, account_id)
    else:
        result = _place_sell_order(ob_id, price, volume, account_id)
    ok = result.get("orderRequestStatus") == "SUCCESS"
    return ok, result


def place_stop_loss_no_page(account_id, ob_id, trigger_price, sell_price, volume, valid_days=8):
    """Hardware stop-loss via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _place_stop_loss_session(ob_id, trigger_price, sell_price, volume, account_id, valid_days)
    ok = result.get("status") == "SUCCESS"
    return ok, result


def place_trailing_stop_no_page(account_id, ob_id, trail_percent, volume, valid_days=8):
    """Hardware trailing stop via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _place_trailing_stop_session(ob_id, trail_percent, volume, account_id, valid_days)
    ok = result.get("status") == "SUCCESS"
    return ok, result


def delete_order_no_page(account_id, order_id):
    """Cancel order via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _cancel_order(order_id, account_id)
    ok = result.get("orderRequestStatus") == "SUCCESS"
    return ok, result


def delete_stop_loss_no_page(account_id, stop_id):
    """Delete stop-loss via BankID session — no Playwright page needed.


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_control.py | Select-Object -Index (390..430)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 641ms:

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    resolved_account_id = str(account_id or get_account_id())
    try:
        result = _api_delete(f"/_api/trading/stoploss/{resolved_account_id}/{stop_id}")
        # H18: Check for error indicators in the response.
        # API returns {} on success (200 with empty body).
        # A non-empty response with error keys indicates failure.
        if isinstance(result, dict) and result.get("errorCode"):
            logger.warning("Delete stop-loss returned error for stop %s: %s", stop_id, result)
            return False, result
        return True, result
    except Exception as e:
        logger.error("Delete stop-loss (no page) failed for stop %s: %s", stop_id, e, exc_info=True)
        return False, {"error": str(e)}


__all__ = [
    "check_session_alive",
    "delete_order",
    "delete_order_live",
    "delete_order_no_page",
    "delete_stop_loss",
    "delete_stop_loss_no_page",
    "fetch_account_cash",
    "fetch_page_positions",
    "fetch_price",
    "fetch_price_no_page",
    "fetch_price_no_page_with_fallback",
    "fetch_price_with_fallback",
    "find_instrument",
    "get_account_id",
    "get_csrf",
    "get_open_orders",
    "get_portfolio_value",
    "get_positions",
    "get_price_info",
    "normalize_api_type",
    "place_buy_order",

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern '\\bget_quote\\(' | ForEach-Object { \""'$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 626ms:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:662: def get_quote(orderbook_id: str) -> dict:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\market_data.py:48: def get_quote(ob_id: str, instrument_type: str = "certificate") -> Quote:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/avanza_session.py,portfolio/avanza_client.py -Pattern 'DEFAULT_ACCOUNT_ID|1625505|ALLOWED_ACCOUNT_IDS' | ForEach-Object { \""'$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 570ms:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:35: DEFAULT_ACCOUNT_ID = "1625505"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:38: ALLOWED_ACCOUNT_IDS = {"1625505"}
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:410: account_id: Avanza account ID (default: ``DEFAULT_ACCOUNT_ID``).
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:419: aid = str(account_id or DEFAULT_ACCOUNT_ID)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:550: account_id: Defaults to DEFAULT_ACCOUNT_ID.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:585: effective_account_id = str(account_id or DEFAULT_ACCOUNT_ID)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:586: if effective_account_id not in ALLOWED_ACCOUNT_IDS:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:634: "accountId": str(account_id or DEFAULT_ACCOUNT_ID),
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:645: aid = str(account_id or DEFAULT_ACCOUNT_ID)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:736: account_id: Defaults to DEFAULT_ACCOUNT_ID.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:744: acct = str(account_id or DEFAULT_ACCOUNT_ID)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:745: if acct not in ALLOWED_ACCOUNT_IDS:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:826: account_id: Defaults to DEFAULT_ACCOUNT_ID.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:896: account_id: Avanza account ID. Defaults to ``DEFAULT_ACCOUNT_ID``.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:906: acct = str(account_id or DEFAULT_ACCOUNT_ID)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:1031: # stop, which may differ from DEFAULT_ACCOUNT_ID for multi-account users.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:30: # without filtering. Mirror the ALLOWED_ACCOUNT_IDS pattern from
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:32: ALLOWED_ACCOUNT_IDS: set[str] = {"1625505"}
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:186: if str(account.get("accountId", "")) not in ALLOWED_ACCOUNT_IDS:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:206: A-AV-2: Filters to ALLOWED_ACCOUNT_IDS so pension/other-account values
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:216: if str(account.get("accountId", "")) not in ALLOWED_ACCOUNT_IDS:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:246: ALLOWED_ACCOUNT_IDS*. Any ISK account not in the whitelist is rejected.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:251: Account ID string (guaranteed to be in ALLOWED_ACCOUNT_IDS)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:270: if candidate in ALLOWED_ACCOUNT_IDS:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:276: f"ISK account IDs seen: {seen_ids}. Whitelist: {sorted(ALLOWED_ACCOUNT_IDS)}. "
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:277: "If this is a legitimate new account, update ALLOWED_ACCOUNT_IDS in "

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File portfolio | Select-String -Pattern 'return True, result|return False, result|orderRequestStatus|errorCode|ok\\s*=|http_status' | ForEach-Object { \""'$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 623ms:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:300: Dict with orderId, orderRequestStatus, message
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:321: Dict with orderId, orderRequestStatus, message
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:355: "Placing %s order: orderbook=%s price=%.2f vol=%d until=%s account=%s",
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:391: Dict with orderId, orderRequestStatus, messages
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:194: http_status = int(result.get("status") or 0)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:202: success = parsed.get("orderRequestStatus") == "SUCCESS"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:204: "http_status": http_status,
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:238: http_status = int(result.get("status") or 0)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:241: success = (200 <= http_status < 300) or http_status == 404
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:250: "http_status": http_status,
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:352: ok = result.get("orderRequestStatus") == "SUCCESS"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:363: ok = result.get("status") == "SUCCESS"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:374: ok = result.get("status") == "SUCCESS"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:385: ok = result.get("orderRequestStatus") == "SUCCESS"
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:401: if isinstance(result, dict) and result.get("errorCode"):
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:403: return False, result
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:404: return True, result
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_order_lock.py:82: target.parent.mkdir(parents=True, exist_ok=True)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_orders.py:365: status = result.get("orderRequestStatus", "UNKNOWN")
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:351: Dict with ``http_status`` and ``ok`` keys.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:372: return {"http_status": resp.status, "ok": 200 <= resp.status < 300 or resp.status == 404}
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:554: Dict with orderRequestStatus, orderId, message.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:617: status = result.get("orderRequestStatus", "UNKNOWN")
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:681: orderbook = inst.get("orderbook", {})
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:899: Dict with keys ``status`` ("SUCCESS"/"FAILED"), ``http_status`` (int),
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:902: ``http_status=0`` and an ``error`` key describing the cause.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:905: return {"status": "FAILED", "http_status": 0, "stop_id": "", "error": "empty stop_id"}
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:914: return {"status": "FAILED", "http_status": 0, "stop_id": stop_id, "error": str(exc)}
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:915: http_status = int(result.get("http_status", 0)) if isinstance(result, dict) else 0
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:917: ok = (200 <= http_status < 300) or http_status == 404
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:919: logger.info("cancel_stop_loss(%s) -> %s", stop_id, http_status)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:921: logger.warning("cancel_stop_loss(%s) failed: http=%s result=%s", stop_id, http_status, result)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:924: "http_status": http_status,
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:100: raw.get("orderRequestStatus"),
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:145: raw.get("orderRequestStatus"),
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:162: status = str(raw.get("orderRequestStatus", "")).upper()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:286: raw.get("status", raw.get("orderRequestStatus")),
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\types.py:186: status = raw.get("orderRequestStatus", raw.get("status", ""))
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\types.py:210: status = raw.get("status", raw.get("orderRequestStatus", ""))
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\types.py:244: orderbook = instrument.get("orderbook", {})
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\trading.cpython-312.pyc:34: �   zvolume must be >= 1, got r   zprice must be > 0, got �   �     @�@zOrder total z.2fz SEK below minimum 1000 SEK)�	conditionz2place_order side=%s ob_id=%s price=%s vol=%d -> %s�orderRequestStatus)�
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\__pycache__\types.cpython-312.pyc:107: � |D �       �      } | ||t        |�      t        |�      ��      S )N�orderRequestStatusrp   r   �SUCCESS�orderIdro   rq   �messagesz; c              3  �2   K  � | ]  }t        |�      �� � y �wr^   )r   )r_   �ms     r   ra   z'OrderResult.from_api.<locals>.<genexpr>�   s   � �� �8��1��A���s   �)rn   ro   rp   rq   )r5   r   �upperr
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:140: Dict with orderId, orderRequestStatus, message
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:150: Dict with orderId, orderRequestStatus, message
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:174: �   zVolume must be >= 1, got r   zPrice must be > 0, got zDPlacing %s order: orderbook=%s price=%.2f vol=%d until=%s account=%szplace_order_totp/�/��op)rI   �
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:199: Dict with orderId, orderRequestStatus, messages
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_control.cpython-312.pyc:72: }N�statusr   �bodyr!   �orderRequestStatus�SUCCESS��http_status�parsedrZ   z$Delete order failed for order %s: %sT��exc_info�r   r?   r   r   �evaluate�intr%   �json�loads�	TypeError�JSONDecodeError�	Exception�loggerrU   )r.   rB   �order_id�csrfrC   �resultr^   �	body_textr_   �success�excs              r)   �delete_order_liverq   �   sH  � � �D�>�D���w��0�0�0��j�<�N�,<�=��&*�
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_control.cpython-312.pyc:162: z/_api/trading/stoploss/r|   �	errorCodez/Delete stop-loss returned error for stop %s: %sFTz1Delete stop-loss (no page) failed for stop %s: %sr`   rU   N)
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_orders.cpython-312.pyc:165:         d#| d   � d|� �|�       n1# t        $ r }t        j                  d$|�       Y d%}~nd%}~ww xY wY d%}~y%Y d%}~y%d%}~ww xY w)&z<Execute a confirmed order on Avanza and notify via Telegram.r$   r+   r%   r)   r(   )r%   r)   r(   �orderRequestStatus�UNKNOWN�orderId�?r^   rH   �SUCCESS�executedr3   �avanza_order_idzAVANZA z
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:143: Dict with ``http_status`` and ``ok`` keys.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:145: Nrr   rs   )rt   rX   rY   ��   �,  �  )�http_statusrK   )rn   rI   �deleter]   rF   r   )rL   rz   rM   ra   s      �r   rb   zapi_delete.<locals>._ope  s�   �� ���~���{�{�!�!�� 2�#'�� "� 
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:243: Dict with orderRequestStatus, orderId, message.
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:254: validUntilr�   zplace_order/rV   �rO   z%/_api/trading-critical/rest/order/newN�orderRequestStatus�UNKNOWN�SUCCESSu   Order %s failed: %s — %s�message� z'Order %s placed: %dx @ %.3f SEK (id=%s)�orderId�?)r!   r@   r�   �ALLOWED_ACCOUNT_IDS�roundr   �today�	isoformatr
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:441: Dict with keys ``status`` ("SUCCESS"/"FAILED"), ``http_status`` (int),
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_session.cpython-312.pyc:444: ``http_status=0`` and an ``error`` key describing the cause.

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-avanza-api ls-tree -r --name-only HEAD' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 562ms:
BASELINE.md
portfolio/avanza/__init__.py
portfolio/avanza/account.py
portfolio/avanza/auth.py
portfolio/avanza/client.py
portfolio/avanza/market_data.py
portfolio/avanza/scanner.py
portfolio/avanza/search.py
portfolio/avanza/streaming.py
portfolio/avanza/tick_rules.py
portfolio/avanza/trading.py
portfolio/avanza/types.py
portfolio/avanza_client.py
portfolio/avanza_control.py
portfolio/avanza_order_lock.py
portfolio/avanza_orders.py
portfolio/avanza_resilient_page.py
portfolio/avanza_session.py
portfolio/avanza_tracker.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "'portfolio/file_utils.py','portfolio/http_retry.py','portfolio/telegram_notifications.py','data/metals_avanza_helpers.py' | ForEach-Object { Write-Output (\""'$_ = " + (Test-Path $_)) }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 562ms:
portfolio/file_utils.py = False
portfolio/http_retry.py = False
portfolio/telegram_notifications.py = False
data/metals_avanza_helpers.py = False

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza/client.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -First 80' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza/tick_rules.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza_control.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -First 420' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza/scanner.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -First 120' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza_control.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -Index (392..404)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza_orders.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -First 60' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 722ms:
   1: """Tick-size rules — price rounding for Avanza order books.
   2: 
   3: Caches tick tables per orderbook ID so repeated rounding calls do not
   4: hit the API.  Uses integer arithmetic internally to avoid floating-point
   5: drift.
   6: """
   7: 
   8: from __future__ import annotations
   9: 
  10: import logging
  11: import math
  12: from typing import Any
  13: 
  14: from portfolio.avanza.client import AvanzaClient
  15: from portfolio.avanza.types import TickEntry
  16: 
  17: logger = logging.getLogger("portfolio.avanza.tick_rules")
  18: 
  19: # Module-level cache: ob_id -> list of TickEntry
  20: _cache: dict[str, list[TickEntry]] = {}
  21: 
  22: 
  23: # ---------------------------------------------------------------------------
  24: # Public API
  25: # ---------------------------------------------------------------------------
  26: 
  27: 
  28: def get_tick_rules(ob_id: str) -> list[TickEntry]:
  29:     """Fetch (and cache) the tick-size table for an orderbook.
  30: 
  31:     Args:
  32:         ob_id: Avanza orderbook ID.
  33: 
  34:     Returns:
  35:         List of :class:`~portfolio.avanza.types.TickEntry` sorted by
  36:         ``min_price``.
  37:     """
  38:     if ob_id in _cache:
  39:         return _cache[ob_id]
  40: 
  41:     client = AvanzaClient.get_instance()
  42:     raw: dict[str, Any] = client.get_order_book_raw(ob_id)
  43: 
  44:     tick_list_raw: list[dict[str, Any]] = raw.get("tickSizeList", raw.get("tickSizes", []))
  45:     entries = [TickEntry.from_api(t) for t in tick_list_raw]
  46:     entries.sort(key=lambda e: e.min_price)
  47: 
  48:     _cache[ob_id] = entries
  49:     logger.debug("get_tick_rules ob_id=%s entries=%d (cached)", ob_id, len(entries))
  50:     return entries
  51: 
  52: 
  53: def round_to_tick(price: float, ob_id: str, direction: str = "down") -> float:
  54:     """Round a price to the nearest valid tick.
  55: 
  56:     Uses integer arithmetic (multiply -> floor/ceil -> divide) to avoid
  57:     floating-point drift.
  58: 
  59:     Args:
  60:         price: The price to round.
  61:         ob_id: Avanza orderbook ID (needed to fetch the tick table).
  62:         direction: ``"down"`` (floor) or ``"up"`` (ceil).
  63: 
  64:     Returns:
  65:         The rounded price.
  66: 
  67:     Raises:
  68:         ValueError: If *direction* is not ``"down"`` or ``"up"``.
  69:         ValueError: If no tick rule matches *price*.
  70:     """
  71:     if direction not in ("down", "up"):
  72:         raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
  73: 
  74:     entries = get_tick_rules(ob_id)
  75:     tick = _find_tick_for_price(price, entries)
  76: 
  77:     if tick is None:
  78:         raise ValueError(f"No tick rule found for price {price} (ob_id={ob_id})")
  79: 
  80:     # Integer arithmetic to avoid float drift:
  81:     # steps = price / tick  ->  round to int  ->  result = steps * tick
  82:     # We use a precision multiplier derived from the tick's decimal places.
  83:     precision = _decimal_places(tick)
  84:     multiplier = 10 ** precision
  85: 
  86:     # Convert to integer domain
  87:     price_int = price * multiplier
  88:     tick_int = round(tick * multiplier)
  89: 
  90:     if tick_int == 0:
  91:         return price  # degenerate tick; return unchanged
  92: 
  93:     if direction == "down":
  94:         steps = math.floor(price_int / tick_int)
  95:     else:
  96:         steps = math.ceil(price_int / tick_int)
  97: 
  98:     result = (steps * tick_int) / multiplier
  99:     return round(result, precision)
 100: 
 101: 
 102: def clear_cache() -> None:
 103:     """Clear the module-level tick-rule cache."""
 104:     _cache.clear()
 105:     logger.debug("tick_rules cache cleared")
 106: 
 107: 
 108: # ---------------------------------------------------------------------------
 109: # Internals
 110: # ---------------------------------------------------------------------------
 111: 
 112: 
 113: def _find_tick_for_price(price: float, entries: list[TickEntry]) -> float | None:
 114:     """Find the tick size applicable for *price*.
 115: 
 116:     Returns ``None`` if no entry matches.
 117:     """
 118:     for entry in entries:
 119:         if entry.min_price <= price <= entry.max_price:
 120:             return entry.tick_size
 121:         # Handle unbounded upper range (max_price == 0 means infinity)
 122:         if entry.min_price <= price and entry.max_price == 0:
 123:             return entry.tick_size
 124:     # Fallback: if price exceeds all ranges, use the last entry
 125:     if entries:
 126:         return entries[-1].tick_size
 127:     return None
 128: 
 129: 
 130: def _decimal_places(value: float) -> int:
 131:     """Count the number of significant decimal places in *value*."""
 132:     s = f"{value:.10f}".rstrip("0")
 133:     if "." in s:
 134:         return len(s.split(".")[1])
 135:     return 0

 succeeded in 774ms:
   1: """Singleton HTTP client wrapping the avanza-api library.
   2: 
   3: Provides raw delegator methods that return whatever the underlying
   4: ``avanza.Avanza`` instance returns.  Typed higher-level modules (market
   5: data, trading, account, etc.) will wrap these delegators and return our
   6: own dataclasses from :mod:`portfolio.avanza.types`.
   7: """
   8: 
   9: from __future__ import annotations
  10: 
  11: import logging
  12: import threading
  13: from typing import Any
  14: 
  15: from portfolio.avanza.auth import AvanzaAuth
  16: 
  17: logger = logging.getLogger("portfolio.avanza.client")
  18: 
  19: DEFAULT_ACCOUNT_ID = "1625505"
  20: 
  21: 
  22: class AvanzaClient:
  23:     """Singleton client wrapping the avanza-api library.
  24: 
  25:     Usage::
  26: 
  27:         client = AvanzaClient.get_instance(config)
  28:         raw = client.get_market_data_raw("2213050")
  29:     """
  30: 
  31:     _instance: AvanzaClient | None = None
  32:     _lock = threading.Lock()
  33: 
  34:     def __init__(self, auth: AvanzaAuth, account_id: str) -> None:
  35:         self._auth = auth
  36:         self._account_id = account_id
  37: 
  38:     @classmethod
  39:     def get_instance(cls, config: dict[str, Any] | None = None) -> AvanzaClient:
  40:         """Return the singleton, creating it on first call.
  41: 
  42:         Args:
  43:             config: Application config dict.  Must contain an ``"avanza"`` key
  44:                 with ``"username"``, ``"password"``, and ``"totp_secret"`` when
  45:                 creating for the first time.  Ignored on subsequent calls.
  46:         """
  47:         if cls._instance is not None:
  48:             return cls._instance
  49: 
  50:         with cls._lock:
  51:             if cls._instance is not None:
  52:                 return cls._instance
  53: 
  54:             if config is None:
  55:                 raise ValueError(
  56:                     "AvanzaClient.get_instance() requires config on first call"
  57:                 )
  58: 
  59:             avanza_cfg = config.get("avanza", {})
  60:             auth = AvanzaAuth.get_instance(
  61:                 username=avanza_cfg["username"],
  62:                 password=avanza_cfg["password"],
  63:                 totp_secret=avanza_cfg["totp_secret"],
  64:             )
  65:             account_id = str(avanza_cfg.get("account_id", DEFAULT_ACCOUNT_ID))
  66: 
  67:             instance = cls(auth=auth, account_id=account_id)
  68:             cls._instance = instance
  69:             logger.info(
  70:                 "AvanzaClient singleton created (account_id=%s)", account_id
  71:             )
  72:             return instance
  73: 
  74:     @classmethod
  75:     def reset(cls) -> None:
  76:         """Tear down the singleton."""
  77:         with cls._lock:
  78:             cls._instance = None
  79:             logger.info("AvanzaClient singleton reset")
  80: 

 succeeded in 779ms:
   1: """Avanza order confirmation flow — human-in-the-loop for real money.
   2: 
   3: Workflow:
   4: 1. Layer 2 calls request_order() → saves intent to pending orders, returns details
   5:    (including a unique 6-hex `confirm_token`).
   6: 2. Layer 2 sends Telegram message with order details + "Reply CONFIRM <token>
   7:    to execute".
   8: 3. Main loop calls check_pending_orders() each cycle.
   9: 4. On CONFIRM <token> reply → execute the order whose token matches, notify
  10:    via Telegram.
  11: 5. On timeout (5 min) → expire the pending order, notify.
  12: 
  13: P1-10 (2026-05-02): per-order `confirm_token` eliminates three races the
  14: old bare-CONFIRM design suffered from (see test class docstrings):
  15: - stale-CONFIRM race (replayed CONFIRM confirms a NEWER order)
  16: - wrong-order race (sort-by-time-DESC matches the wrong order)
  17: - no-pending-yet race (CONFIRM lands before the order it was for)
  18: 
  19: Bare CONFIRM (no token) is still accepted but ONLY matches LEGACY orders
  20: that have no `confirm_token` field — i.e. orders that were already in
  21: flight when this code was deployed. New orders MUST be confirmed by token.
  22: """
  23: 
  24: import contextlib
  25: import logging
  26: import re
  27: import secrets
  28: import uuid
  29: from datetime import UTC, datetime, timedelta
  30: from pathlib import Path
  31: 
  32: from portfolio.avanza_control import place_buy_order, place_sell_order
  33: from portfolio.file_utils import atomic_write_json, load_json
  34: from portfolio.http_retry import fetch_with_retry
  35: from portfolio.telegram_notifications import send_telegram
  36: 
  37: logger = logging.getLogger("portfolio.avanza_orders")
  38: 
  39: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
  40: PENDING_FILE = DATA_DIR / "avanza_pending_orders.json"
  41: EXPIRY_MINUTES = 5
  42: 
  43: # P1-10 (2026-05-02): per-order confirmation nonce. 6 hex chars = 24 bits
  44: # of entropy ≈ ~16M possible tokens. Collision probability across the at-most
  45: # ~5 in-flight pending orders is effectively zero (birthday bound:
  46: # ~5^2/(2*16M) ≈ 7.5e-7). Long enough to survive typos, short enough that
  47: # users will actually type it on a phone keyboard.
  48: _CONFIRM_TOKEN_HEX_CHARS = 6
  49: # Token validation: anything outside [0-9a-f] is silently dropped rather
  50: # than confirmed against an unknown order. This prevents 'CONFIRM xyz' (a
  51: # typo) from accidentally confirming any order via the legacy bare-CONFIRM
  52: # path or matching a token-holding order.
  53: _HEX_TOKEN_RE = re.compile(r"^[0-9a-f]+$")
  54: # CONFIRM prefix matcher. Word boundary required because "confirmed" /
  55: # "confirms" / "confirmation" parse to "confirm" + a hex-valid suffix
  56: # ("ed", "s", "ation") which would silently match against legacy orders
  57: # or non-existent tokens. Anchored at start since the user is asked to
  58: # reply with "CONFIRM <token>" as the entire message.
  59: _CONFIRM_PREFIX_RE = re.compile(r"^confirm(?:\s+|$)")
  60: 

 succeeded in 815ms:
   1: """Canonical Avanza control facade for reads, quotes, and browser-session trades.
   2: 
   3: Use this module as the shared import path for Avanza operations in strategy code.
   4: It keeps the currently working Playwright-page execution path for metals/gold
   5: while exposing the broader account/session helpers from ``portfolio.avanza_*``.
   6: """
   7: 
   8: from __future__ import annotations
   9: 
  10: import json
  11: import logging
  12: 
  13: from portfolio.avanza_order_lock import avanza_order_lock
  14: 
  15: logger = logging.getLogger("portfolio.avanza_control")
  16: 
  17: from data.metals_avanza_helpers import (
  18:     check_session_alive,
  19:     get_csrf,
  20: )
  21: from data.metals_avanza_helpers import (
  22:     fetch_account_cash as _fetch_account_cash,
  23: )
  24: from data.metals_avanza_helpers import (
  25:     fetch_positions as _fetch_page_positions,
  26: )
  27: from data.metals_avanza_helpers import (
  28:     fetch_price as _fetch_page_price,
  29: )
  30: from data.metals_avanza_helpers import (
  31:     place_order as _place_page_order,
  32: )
  33: from data.metals_avanza_helpers import (
  34:     place_stop_loss as _place_page_stop_loss,
  35: )
  36: from portfolio.avanza_client import (
  37:     delete_order,
  38:     find_instrument,
  39:     get_account_id,
  40:     get_open_orders,
  41:     get_portfolio_value,
  42:     get_positions,
  43:     place_buy_order,
  44:     place_sell_order,
  45: )
  46: from portfolio.avanza_client import (
  47:     get_price as get_price_info,
  48: )
  49: 
  50: _TYPE_ALIASES = {
  51:     "cert": "certificate",
  52:     "certifikat": "certificate",
  53:     "certificate": "certificate",
  54:     "warrant": "warrant",
  55:     "mini": "warrant",
  56:     "mini-future": "warrant",
  57:     "mini_future": "warrant",
  58:     "stock": "stock",
  59:     "share": "stock",
  60:     "fund": "fund",
  61:     "etf": "exchange_traded_fund",
  62:     "exchange_traded_fund": "exchange_traded_fund",
  63: }
  64: 
  65: _PRICE_FALLBACK_TYPES = (
  66:     "certificate",
  67:     "warrant",
  68:     "stock",
  69:     "exchange_traded_fund",
  70:     "fund",
  71: )
  72: 
  73: 
  74: def normalize_api_type(api_type: str | None, default: str = "certificate") -> str:
  75:     """Normalize Avanza instrument type names for market-guide lookups."""
  76:     normalized = (api_type or "").strip().lower()
  77:     if not normalized:
  78:         return default
  79:     return _TYPE_ALIASES.get(normalized, normalized)
  80: 
  81: 
  82: def fetch_price(page, orderbook_id: str, api_type: str = "certificate"):
  83:     """Fetch a quote from the market-guide API using an authenticated page."""
  84:     return _fetch_page_price(page, orderbook_id, normalize_api_type(api_type))
  85: 
  86: 
  87: def fetch_price_with_fallback(page, orderbook_id: str, api_type: str | None = None):
  88:     """Try the preferred market-guide type and then the common fallback types."""
  89:     if not orderbook_id:
  90:         return None
  91: 
  92:     candidates: list[str] = []
  93:     preferred = normalize_api_type(api_type) if api_type else ""
  94:     if preferred:
  95:         candidates.append(preferred)
  96:     for fallback in _PRICE_FALLBACK_TYPES:
  97:         if fallback not in candidates:
  98:             candidates.append(fallback)
  99: 
 100:     for candidate in candidates:
 101:         data = fetch_price(page, orderbook_id, candidate)
 102:         if not data:
 103:             continue
 104:         if data.get("bid") is None and data.get("ask") is None and data.get("last") is None:
 105:             continue
 106:         payload = dict(data)
 107:         payload["api_type"] = candidate
 108:         return payload
 109:     return None
 110: 
 111: 
 112: def fetch_account_cash(page, account_id: str | None = None):
 113:     """Fetch buying power for an account via the authenticated browser session."""
 114:     resolved_account_id = str(account_id or get_account_id())
 115:     return _fetch_account_cash(page, resolved_account_id)
 116: 
 117: 
 118: def fetch_page_positions(page, account_id: str | None = None):
 119:     """Fetch current positions keyed by orderbook id via the page session.
 120: 
 121:     Returns dict[ob_id -> {name, units, value, avg_price, api_type}] on
 122:     success, or None on transient failure. An empty dict `{}` is a valid
 123:     response meaning the account is flat — callers should distinguish it
 124:     from None.
 125:     """
 126:     resolved_account_id = str(account_id or get_account_id())
 127:     return _fetch_page_positions(page, resolved_account_id)
 128: 
 129: 
 130: def place_order(page, account_id: str | None, ob_id: str, side: str, price: float, volume: int):
 131:     """Place a BUY/SELL order via the authenticated browser session."""
 132:     resolved_account_id = str(account_id or get_account_id())
 133:     normalized_side = (side or "").strip().upper()
 134:     return _place_page_order(page, resolved_account_id, ob_id, normalized_side, price, volume)
 135: 
 136: 
 137: def place_stop_loss(
 138:     page,
 139:     account_id: str | None,
 140:     ob_id: str,
 141:     trigger_price: float,
 142:     sell_price: float,
 143:     volume: int,
 144:     valid_days: int = 8,
 145: ):
 146:     """Place a hardware stop-loss order via the authenticated browser session."""
 147:     resolved_account_id = str(account_id or get_account_id())
 148:     return _place_page_stop_loss(
 149:         page,
 150:         resolved_account_id,
 151:         ob_id,
 152:         trigger_price,
 153:         sell_price,
 154:         volume,
 155:         valid_days=valid_days,
 156:     )
 157: 
 158: 
 159: def delete_order_live(page, account_id: str | None, order_id: str):
 160:     """Cancel an open order via the authenticated page session.
 161: 
 162:     IMPORTANT: Uses POST to /_api/trading-critical/rest/order/delete with
 163:     JSON body {accountId, orderId}. The DELETE HTTP verb to
 164:     /_api/trading-critical/rest/order/{accountId}/{orderId} returns 404
 165:     (Avanza API change discovered 2026-03-24).
 166:     """
 167:     csrf = get_csrf(page)
 168:     if not csrf:
 169:         return False, {"error": "no CSRF token"}
 170: 
 171:     resolved_account_id = str(account_id or get_account_id())
 172:     try:
 173:         # 2026-04-13: cross-process order lock (see metals_avanza_helpers.place_order).
 174:         with avanza_order_lock(op=f"delete_order_live/{order_id}"):
 175:             result = page.evaluate(
 176:                 """async (args) => {
 177:                     const [accountId, orderId, token] = args;
 178:                     const resp = await fetch(
 179:                         'https://www.avanza.se/_api/trading-critical/rest/order/delete',
 180:                         {
 181:                             method: 'POST',
 182:                             headers: {
 183:                                 'Content-Type': 'application/json',
 184:                                 'X-SecurityToken': token,
 185:                             },
 186:                             credentials: 'include',
 187:                             body: JSON.stringify({accountId: accountId, orderId: orderId}),
 188:                         }
 189:                     );
 190:                     return {status: resp.status, body: await resp.text()};
 191:                 }""",
 192:                 [resolved_account_id, order_id, csrf],
 193:             )
 194:         http_status = int(result.get("status") or 0)
 195:         body_text = result.get("body", "")
 196:         parsed = {}
 197:         try:
 198:             if body_text:
 199:                 parsed = json.loads(body_text)
 200:         except (TypeError, json.JSONDecodeError):
 201:             parsed = {}
 202:         success = parsed.get("orderRequestStatus") == "SUCCESS"
 203:         return success, {
 204:             "http_status": http_status,
 205:             "parsed": parsed,
 206:             "body": body_text,
 207:         }
 208:     except Exception as exc:
 209:         logger.error("Delete order failed for order %s: %s", order_id, exc, exc_info=True)
 210:         return False, {"error": str(exc)}
 211: 
 212: 
 213: def delete_stop_loss(page, account_id: str | None, stop_id: str):
 214:     """Delete an existing Avanza stop-loss order via the authenticated page."""
 215:     csrf = get_csrf(page)
 216:     if not csrf:
 217:         return False, {"error": "no CSRF token"}
 218: 
 219:     resolved_account_id = str(account_id or get_account_id())
 220:     try:
 221:         # 2026-04-13: cross-process order lock. SL delete is mutating.
 222:         with avanza_order_lock(op=f"delete_stop_loss/{stop_id}"):
 223:             result = page.evaluate(
 224:                 """async (args) => {
 225:                     const [accountId, stopId, token] = args;
 226:                     const resp = await fetch(
 227:                         'https://www.avanza.se/_api/trading/stoploss/' + accountId + '/' + stopId,
 228:                         {
 229:                             method: 'DELETE',
 230:                             headers: {'X-SecurityToken': token},
 231:                             credentials: 'include',
 232:                         }
 233:                     );
 234:                     return {status: resp.status, body: await resp.text()};
 235:                 }""",
 236:                 [resolved_account_id, stop_id, csrf],
 237:             )
 238:         http_status = int(result.get("status") or 0)
 239:         # 2xx = deleted successfully.  404 = stop already gone (triggered/expired/cancelled).
 240:         # Both mean the stop no longer exists, which is the goal of a cancel.
 241:         success = (200 <= http_status < 300) or http_status == 404
 242:         body_text = result.get("body", "")
 243:         parsed = {}
 244:         try:
 245:             if body_text:
 246:                 parsed = json.loads(body_text)
 247:         except (TypeError, json.JSONDecodeError):
 248:             parsed = {}
 249:         return success, {
 250:             "http_status": http_status,
 251:             "parsed": parsed,
 252:             "body": body_text,
 253:         }
 254:     except Exception as exc:
 255:         logger.error("Delete stop-loss failed for stop %s: %s", stop_id, exc, exc_info=True)
 256:         return False, {"error": str(exc)}
 257: 
 258: 
 259: 
 260: # --- Page-free API (uses BankID session, no Playwright page needed) ---
 261: 
 262: from portfolio.avanza_session import (
 263:     api_delete as _api_delete,
 264: )
 265: from portfolio.avanza_session import (
 266:     api_get as _api_get,
 267: )
 268: from portfolio.avanza_session import (
 269:     cancel_order as _cancel_order,
 270: )
 271: from portfolio.avanza_session import (
 272:     place_buy_order as _place_buy_order,
 273: )
 274: from portfolio.avanza_session import (
 275:     place_sell_order as _place_sell_order,
 276: )
 277: from portfolio.avanza_session import (
 278:     place_stop_loss as _place_stop_loss_session,
 279: )
 280: from portfolio.avanza_session import (
 281:     place_trailing_stop as _place_trailing_stop_session,
 282: )
 283: from portfolio.avanza_session import (
 284:     verify_session,
 285: )
 286: 
 287: 
 288: def fetch_price_no_page(orderbook_id: str, api_type: str = "certificate"):
 289:     """Fetch a quote without a Playwright page — uses BankID session API."""
 290:     normalized = normalize_api_type(api_type)
 291:     try:
 292:         data = _api_get(f"/_api/market-guide/{normalized}/{orderbook_id}")
 293:         quote = data.get("quote", {})
 294:         ki = data.get("keyIndicators", {})
 295:         underlying = data.get("underlying", {})
 296:         def _v(obj):
 297:             return obj.get("value") if isinstance(obj, dict) else obj
 298:         return {
 299:             "bid": _v(quote.get("buy")),
 300:             "ask": _v(quote.get("sell")),
 301:             "last": _v(quote.get("last")),
 302:             "change_pct": _v(quote.get("changePercent")),
 303:             "high": _v(quote.get("highest")),
 304:             "low": _v(quote.get("lowest")),
 305:             "underlying": _v(underlying.get("quote", {}).get("last")),
 306:             "underlying_name": underlying.get("name"),
 307:             "leverage": _v(ki.get("leverage")),
 308:             "barrier": _v(ki.get("barrierLevel")),
 309:             "api_type": normalized,
 310:         }
 311:     except Exception as e:
 312:         logger.error("Warrant price fetch failed for orderbook %s: %s", orderbook_id, e, exc_info=True)
 313:         return None
 314: 
 315: 
 316: def fetch_price_no_page_with_fallback(orderbook_id: str, api_type: str | None = None):
 317:     """Try preferred type then fallback chain — no Playwright page needed."""
 318:     if not orderbook_id:
 319:         return None
 320:     candidates = []
 321:     preferred = normalize_api_type(api_type) if api_type else ""
 322:     if preferred:
 323:         candidates.append(preferred)
 324:     for fb in _PRICE_FALLBACK_TYPES:
 325:         if fb not in candidates:
 326:             candidates.append(fb)
 327:     for candidate in candidates:
 328:         data = fetch_price_no_page(orderbook_id, candidate)
 329:         if data and (data.get("bid") is not None or data.get("ask") is not None or data.get("last") is not None):
 330:             return data
 331:     return None
 332: 
 333: 
 334: def place_order_no_page(account_id, ob_id, side, price, volume):
 335:     """Place BUY/SELL via BankID session — no Playwright page needed.
 336: 
 337:     Returns:
 338:         Tuple (ok: bool, result: dict) matching the page-based interface.
 339: 
 340:     Raises:
 341:         ValueError: If *side* is not "BUY" or "SELL" (C2 fail-safe).
 342:     """
 343:     normalized_side = (side or "").strip().upper()
 344:     if normalized_side not in ("BUY", "SELL"):
 345:         raise ValueError(
 346:             f"Invalid order side: {side!r} (must be 'BUY' or 'SELL')"
 347:         )
 348:     if normalized_side == "BUY":
 349:         result = _place_buy_order(ob_id, price, volume, account_id)
 350:     else:
 351:         result = _place_sell_order(ob_id, price, volume, account_id)
 352:     ok = result.get("orderRequestStatus") == "SUCCESS"
 353:     return ok, result
 354: 
 355: 
 356: def place_stop_loss_no_page(account_id, ob_id, trigger_price, sell_price, volume, valid_days=8):
 357:     """Hardware stop-loss via BankID session — no Playwright page needed.
 358: 
 359:     Returns:
 360:         Tuple (ok: bool, result: dict) matching the page-based interface.
 361:     """
 362:     result = _place_stop_loss_session(ob_id, trigger_price, sell_price, volume, account_id, valid_days)
 363:     ok = result.get("status") == "SUCCESS"
 364:     return ok, result
 365: 
 366: 
 367: def place_trailing_stop_no_page(account_id, ob_id, trail_percent, volume, valid_days=8):
 368:     """Hardware trailing stop via BankID session — no Playwright page needed.
 369: 
 370:     Returns:
 371:         Tuple (ok: bool, result: dict) matching the page-based interface.
 372:     """
 373:     result = _place_trailing_stop_session(ob_id, trail_percent, volume, account_id, valid_days)
 374:     ok = result.get("status") == "SUCCESS"
 375:     return ok, result
 376: 
 377: 
 378: def delete_order_no_page(account_id, order_id):
 379:     """Cancel order via BankID session — no Playwright page needed.
 380: 
 381:     Returns:
 382:         Tuple (ok: bool, result: dict) matching the page-based interface.
 383:     """
 384:     result = _cancel_order(order_id, account_id)
 385:     ok = result.get("orderRequestStatus") == "SUCCESS"
 386:     return ok, result
 387: 
 388: 
 389: def delete_stop_loss_no_page(account_id, stop_id):
 390:     """Delete stop-loss via BankID session — no Playwright page needed.
 391: 
 392:     Returns:
 393:         Tuple (ok: bool, result: dict) matching the page-based interface.
 394:     """
 395:     resolved_account_id = str(account_id or get_account_id())
 396:     try:
 397:         result = _api_delete(f"/_api/trading/stoploss/{resolved_account_id}/{stop_id}")
 398:         # H18: Check for error indicators in the response.
 399:         # API returns {} on success (200 with empty body).
 400:         # A non-empty response with error keys indicates failure.
 401:         if isinstance(result, dict) and result.get("errorCode"):
 402:             logger.warning("Delete stop-loss returned error for stop %s: %s", stop_id, result)
 403:             return False, result
 404:         return True, result
 405:     except Exception as e:
 406:         logger.error("Delete stop-loss (no page) failed for stop %s: %s", stop_id, e, exc_info=True)
 407:         return False, {"error": str(e)}
 408: 
 409: 
 410: __all__ = [
 411:     "check_session_alive",
 412:     "delete_order",
 413:     "delete_order_live",
 414:     "delete_order_no_page",
 415:     "delete_stop_loss",
 416:     "delete_stop_loss_no_page",
 417:     "fetch_account_cash",
 418:     "fetch_page_positions",
 419:     "fetch_price",
 420:     "fetch_price_no_page",

 succeeded in 821ms:
 393:         Tuple (ok: bool, result: dict) matching the page-based interface.
 394:     """
 395:     resolved_account_id = str(account_id or get_account_id())
 396:     try:
 397:         result = _api_delete(f"/_api/trading/stoploss/{resolved_account_id}/{stop_id}")
 398:         # H18: Check for error indicators in the response.
 399:         # API returns {} on success (200 with empty body).
 400:         # A non-empty response with error keys indicates failure.
 401:         if isinstance(result, dict) and result.get("errorCode"):
 402:             logger.warning("Delete stop-loss returned error for stop %s: %s", stop_id, result)
 403:             return False, result
 404:         return True, result
 405:     except Exception as e:

 succeeded in 825ms:
   1: """Instrument scanner — find and rank the best warrants/certificates.
   2: 
   3: Chains search → detail fetch → ranking to answer questions like:
   4: "Find the best bull mini-future for oil right now"
   5: 
   6: Works with EITHER auth method:
   7: - TOTP (AvanzaClient) — preferred, faster
   8: - BankID session (avanza_session.api_get/api_post) — fallback
   9: 
  10: Usage:
  11:     from portfolio.avanza.scanner import scan_instruments
  12: 
  13:     results = scan_instruments(
  14:         query="OLJA",           # underlying asset keyword
  15:         direction="BULL",       # BULL or BEAR
  16:         instrument_type="certificate",  # certificate, warrant, or None for both
  17:         sort_by="spread",       # spread, leverage, price, barrier_distance
  18:         limit=10,
  19:     )
  20:     for r in results:
  21:         print(f"{r['name']:40s} lev={r['leverage']:5.1f}x  spread={r['spread_pct']:.2f}%  bid={r['bid']}")
  22: """
  23: 
  24: from __future__ import annotations
  25: 
  26: import logging
  27: import time
  28: from concurrent.futures import ThreadPoolExecutor, as_completed
  29: from contextlib import suppress
  30: from dataclasses import dataclass
  31: 
  32: from portfolio.avanza.types import _val
  33: 
  34: logger = logging.getLogger("portfolio.avanza.scanner")
  35: 
  36: 
  37: # ---------------------------------------------------------------------------
  38: # Dual-auth API helpers — try TOTP first, fall back to BankID session
  39: # ---------------------------------------------------------------------------
  40: 
  41: def _get_api():
  42:     """Return (search_fn, instrument_fn, marketdata_fn, thread_safe) that work
  43:     with whichever auth is currently available.
  44: 
  45:     Returns:
  46:         Tuple of four:
  47:         - search(instrument_type_str, query, limit) -> dict or list
  48:         - get_instrument(api_type, ob_id) -> dict
  49:         - get_market_data(ob_id) -> dict
  50:         - thread_safe: bool — True for TOTP (requests.Session), False for BankID (Playwright)
  51:     """
  52:     # Try TOTP client first (thread-safe, supports parallel fetching)
  53:     try:
  54:         from portfolio.avanza.client import AvanzaClient
  55:         client = AvanzaClient.get_instance()
  56:         avanza = client.avanza
  57: 
  58:         def _search(itype_str, query, limit):
  59:             from avanza.constants import InstrumentType
  60:             return avanza.search_for_instrument(InstrumentType(itype_str), query, limit)
  61: 
  62:         def _instrument(api_type, ob_id):
  63:             return avanza.get_instrument(api_type, ob_id)
  64: 
  65:         def _marketdata(ob_id):
  66:             return avanza.get_market_data(ob_id)
  67: 
  68:         logger.debug("Scanner using TOTP client (thread-safe)")
  69:         return _search, _instrument, _marketdata, True
  70:     except Exception:
  71:         logger.debug("TOTP client unavailable, falling back to BankID session")
  72: 
  73:     # Fall back to BankID session (Playwright — NOT thread-safe, must be sequential)
  74:     try:
  75:         from portfolio.avanza_session import api_get, api_post
  76: 
  77:         def _search(itype_str, query, limit):
  78:             return api_post("/_api/search/filtered-search", {"query": query, "limit": limit})
  79: 
  80:         def _instrument(api_type, ob_id):
  81:             return api_get(f"/_api/market-guide/{api_type}/{ob_id}")
  82: 
  83:         def _marketdata(ob_id):
  84:             try:
  85:                 return api_get(f"/_api/trading-critical/rest/marketdata/{ob_id}")
  86:             except Exception:
  87:                 return {}
  88: 
  89:         logger.debug("Scanner using BankID session (sequential only)")
  90:         return _search, _instrument, _marketdata, False
  91:     except Exception as e:
  92:         raise RuntimeError(
  93:             "No Avanza auth available. Either configure TOTP credentials "
  94:             "or run scripts/avanza_login.py for BankID session."
  95:         ) from e
  96: 
  97: 
  98: @dataclass
  99: class ScannedInstrument:
 100:     """Rich instrument data combining search + market-guide + marketdata."""
 101: 
 102:     orderbook_id: str
 103:     name: str
 104:     instrument_type: str  # CERTIFICATE, WARRANT, etc.
 105:     direction: str  # BULL, BEAR, LONG, SHORT, or ""
 106: 
 107:     # Price
 108:     bid: float | None
 109:     ask: float | None
 110:     last: float | None
 111:     spread_pct: float | None  # (ask-bid)/bid * 100
 112: 
 113:     # Instrument details
 114:     leverage: float | None
 115:     barrier: float | None
 116:     barrier_distance_pct: float | None  # distance from last to barrier
 117: 
 118:     # Underlying
 119:     underlying_name: str
 120:     underlying_price: float | None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza_client.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -First 25' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 609ms:
   1: """Avanza API client for portfolio monitoring and trading.
   2: 
   3: Supports two authentication methods:
   4: 1. BankID session (preferred) — captured by scripts/avanza_login.py, stored in
   5:    data/avanza_session.json. No credentials needed, valid ~24h.
   6: 2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
   7:    from config.json.
   8: 
   9: The client transparently tries BankID session first, then falls back to TOTP.
  10: """
  11: 
  12: import logging
  13: from datetime import date
  14: from pathlib import Path
  15: from typing import Any
  16: 
  17: from portfolio.avanza_order_lock import avanza_order_lock
  18: from portfolio.file_utils import load_json
  19: 
  20: logger = logging.getLogger("portfolio.avanza_client")
  21: 
  22: BASE_DIR = Path(__file__).resolve().parent.parent
  23: CONFIG_FILE = BASE_DIR / "config.json"
  24: 
  25: # A-AV-2 (2026-04-11): Hardcoded account whitelist. The TOTP path scans for

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'avanza_session import get_positions|avanza_session\\.get_positions\\(|session_get_positions|get_positions\\(\\)' | ForEach-Object { \""'$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 585ms:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:162: def get_positions() -> list[dict]:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:174: from portfolio.avanza_session import get_positions as session_get_positions
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:175: return session_get_positions()
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_session.py:671: def get_positions() -> list[dict]:
Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\__pycache__\avanza_client.cpython-312.pyc:88: r"   r   r;   r   r   r7   r-   r(   �get_overview�get�strr	   �append)�session_get_positionsr!   r1   �overviewr?   rH   �poss          r   r;   r;   �   sA  � � ��	�W�(�*�*�

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza_client.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -Index (160..180)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 621ms:
 161: 
 162: def get_positions() -> list[dict]:
 163:     """Get all current positions from the Avanza account.
 164: 
 165:     Tries BankID session first, then falls back to TOTP client.
 166: 
 167:     Returns:
 168:         List of position dicts, each with name, value, profit, etc.
 169:         Returns empty list if no positions or on error.
 170:     """
 171:     # Try session-based auth first
 172:     if _try_session_auth():
 173:         try:
 174:             from portfolio.avanza_session import get_positions as session_get_positions
 175:             return session_get_positions()
 176:         except Exception as e:
 177:             logger.warning("Session-based positions fetch failed, trying TOTP: %s", e)
 178:             reset_session()
 179: 
 180:     client = get_client()
 181:     overview = client.get_overview()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza/__init__.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ }' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 581ms:
   1: """Unified Avanza API package.
   2: 
   3:     from portfolio.avanza import get_quote, place_order, get_positions
   4: """
   5: 
   6: # Auth & client
   7: from portfolio.avanza.auth import AvanzaAuth, AuthError
   8: from portfolio.avanza.client import AvanzaClient
   9: 
  10: # Account
  11: from portfolio.avanza.account import get_buying_power, get_positions, get_transactions
  12: 
  13: # Market data
  14: from portfolio.avanza.market_data import get_instrument_info, get_market_data, get_news, get_ohlc, get_quote
  15: 
  16: # Search
  17: from portfolio.avanza.search import find_certificates, find_warrants, search
  18: 
  19: # Tick rules
  20: from portfolio.avanza.tick_rules import clear_cache as clear_tick_cache, get_tick_rules, round_to_tick
  21: 
  22: # Scanner
  23: from portfolio.avanza.scanner import ScannedInstrument, format_scan_results, scan_instruments
  24: 
  25: # Streaming
  26: from portfolio.avanza.streaming import AvanzaStream
  27: 
  28: # Trading
  29: from portfolio.avanza.trading import (
  30:     cancel_order, delete_stop_loss, get_deals, get_orders, get_stop_losses,
  31:     modify_order, place_order, place_stop_loss, place_trailing_stop,
  32: )
  33: 
  34: __all__ = [
  35:     "AvanzaAuth", "AuthError", "AvanzaClient", "AvanzaStream", "ScannedInstrument",
  36:     "scan_instruments", "format_scan_results",
  37:     "get_positions", "get_buying_power", "get_transactions",
  38:     "get_quote", "get_market_data", "get_ohlc", "get_instrument_info", "get_news",
  39:     "search", "find_warrants", "find_certificates",
  40:     "get_tick_rules", "round_to_tick", "clear_tick_cache",
  41:     "place_order", "modify_order", "cancel_order",
  42:     "get_orders", "get_deals",
  43:     "place_stop_loss", "place_trailing_stop", "get_stop_losses", "delete_stop_loss",
  44: ]

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/avanza_session.py | Select-Object -Index (620..640)' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 575ms:
        logger.info(
            "Order %s placed: %dx @ %.3f SEK (id=%s)",
            side, volume, price, result.get("orderId", "?"),
        )
    return result


def cancel_order(order_id: str, account_id: str | None = None) -> dict:
    """Cancel an open order.

    IMPORTANT: Uses POST (not DELETE verb) — Avanza API change 2026-03-24.
    """
    payload = {
        "accountId": str(account_id or DEFAULT_ACCOUNT_ID),
        "orderId": str(order_id),
    }
    # 2026-04-13: cross-process order lock — cancel is a mutation, same
    # concurrency concern as place_order (don't want two cancels racing).
    with avanza_order_lock(op=f"cancel_order/{order_id}"):
        return api_post("/_api/trading-critical/rest/order/delete", payload)


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content BASELINE.md' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 645ms:
# Empty baseline for adversarial review diffs

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content portfolio/avanza/trading.py | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ } | Select-Object -First 120' in Q:\finance-analyzer\.worktrees\adv-avanza-api
 succeeded in 549ms:
   1: """Trading operations — orders, stop-losses, deals.
   2: 
   3: Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
   4: for placing, modifying, and cancelling orders and stop-losses.
   5: """
   6: 
   7: from __future__ import annotations
   8: 
   9: import logging
  10: from datetime import date, timedelta
  11: from typing import Any
  12: 
  13: from avanza.constants import (
  14:     Condition,
  15:     OrderType,
  16:     StopLossPriceType,
  17:     StopLossTriggerType,
  18: )
  19: from avanza.entities import StopLossOrderEvent, StopLossTrigger
  20: 
  21: from portfolio.avanza.client import AvanzaClient
  22: from portfolio.avanza.types import (
  23:     Deal,
  24:     Order,
  25:     OrderResult,
  26:     StopLoss,
  27:     StopLossResult,
  28: )
  29: 
  30: logger = logging.getLogger("portfolio.avanza.trading")
  31: 
  32: 
  33: # ---------------------------------------------------------------------------
  34: # Orders
  35: # ---------------------------------------------------------------------------
  36: 
  37: 
  38: def place_order(
  39:     side: str,
  40:     ob_id: str,
  41:     price: float,
  42:     volume: int,
  43:     condition: str = "NORMAL",
  44:     valid_until: str | None = None,
  45:     account_id: str | None = None,
  46: ) -> OrderResult:
  47:     """Place a BUY or SELL order.
  48: 
  49:     Args:
  50:         side: ``"BUY"`` or ``"SELL"``.
  51:         ob_id: Avanza orderbook ID.
  52:         price: Limit price.
  53:         volume: Number of units.
  54:         condition: Order condition (``"NORMAL"``, ``"FILL_OR_KILL"``,
  55:             ``"FILL_AND_KILL"``).
  56:         valid_until: ISO date string (default: today).
  57:         account_id: Override default account.
  58: 
  59:     Returns:
  60:         :class:`~portfolio.avanza.types.OrderResult`.
  61: 
  62:     Raises:
  63:         ValueError: If volume < 1, price <= 0, or order total < 1000 SEK.
  64:     """
  65:     if volume < 1:
  66:         raise ValueError(f"volume must be >= 1, got {volume}")
  67:     if price <= 0:
  68:         raise ValueError(f"price must be > 0, got {price}")
  69: 
  70:     # 2026-04-17: match portfolio/avanza_session.py:590 convention — orders
  71:     # below 1000 SEK pay the Avanza courtage minimum and are almost always
  72:     # a caller bug. Unified-package callers should hit the same guard as
  73:     # the legacy path.
  74:     order_total = round(volume * price, 2)
  75:     if order_total < 1000.0:
  76:         raise ValueError(
  77:             f"Order total {order_total:.2f} SEK below minimum 1000 SEK"
  78:         )
  79: 
  80:     client = AvanzaClient.get_instance()
  81:     acct = account_id or client.account_id
  82:     valid = date.fromisoformat(valid_until) if valid_until else date.today()
  83: 
  84:     raw: dict[str, Any] = client.avanza.place_order(
  85:         acct,
  86:         ob_id,
  87:         OrderType(side),
  88:         price,
  89:         valid,
  90:         volume,
  91:         condition=Condition(condition),
  92:     )
  93: 
  94:     logger.info(
  95:         "place_order side=%s ob_id=%s price=%s vol=%d -> %s",
  96:         side,
  97:         ob_id,
  98:         price,
  99:         volume,
 100:         raw.get("orderRequestStatus"),
 101:     )
 102:     return OrderResult.from_api(raw)
 103: 
 104: 
 105: def modify_order(
 106:     order_id: str,
 107:     ob_id: str,
 108:     price: float,
 109:     volume: int,
 110:     condition: str = "NORMAL",
 111:     valid_until: str | None = None,
 112:     account_id: str | None = None,
 113: ) -> OrderResult:
 114:     """Modify an existing order.
 115: 
 116:     Args:
 117:         order_id: Existing order ID to modify.
 118:         ob_id: Avanza orderbook ID (unused by API but kept for consistency).
 119:         price: New limit price.
 120:         volume: New volume.

Identified several correctness issues in the new Avanza package and legacy wrappers; preparing the review output.
  ✓ Inspect diff against merge base
  ✓ Analyze changed files for actionable bugs
  → Produce JSON review findings
codex
The patch introduces multiple runtime and trading correctness problems in core entry points, including a first-call initialization failure, mis-rounded tick prices, and order-management paths that bypass or misreport safety checks. Those issues are enough to make the change unsafe to consider correct.

Full review comments:

- [P1] Add a bootstrap path for first-time AvanzaClient initialization — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\client.py:54-57
  Every public helper in the new `portfolio.avanza` package eventually calls `AvanzaClient.get_instance()` with no arguments, so the first `get_quote`, `place_order`, `scan_instruments`, etc. in a fresh process raises `ValueError` here unless the caller already knows to initialize the singleton out-of-band. As written, the exported convenience API is unusable on first use.

- [P1] Avoid float drift when converting prices to tick steps — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\tick_rules.py:87-95
  `price_int` stays a float here, so exact on-tick prices can move by a full tick when binary rounding nudges the division just below or above an integer boundary. For example, `0.29` with a `0.01` tick rounds down to `0.28`, and `0.14` with a `0.02` tick rounds up to `0.16`, which means callers can submit worse order prices even when the input was already valid.

- [P1] Reapply the account whitelist on session-backed positions — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:172-175
  When BankID session auth is available, this branch returns `avanza_session.get_positions()` directly. That helper does not filter by `ALLOWED_ACCOUNT_IDS`, so the pension/other-account leakage that the TOTP branch explicitly guards against is back on the preferred auth path; multi-account users will see non-trading holdings in the portfolio view again.

- [P1] Check `_api_delete`'s `ok` flag before declaring stop-loss success — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:397-404
  In the no-page path, `_api_delete()` now returns `{"http_status": ..., "ok": ...}` from `avanza_session.api_delete`, but this wrapper only looks for `errorCode`. A 403/500 therefore comes back as `(True, result)` even though the stop-loss is still active, so downstream sell/re-arm logic can proceed under the false assumption that the encumbered volume has been cleared.

- [P1] Guard unified trading mutations with `avanza_order_lock` — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:80-84
  These new trading entry points call Avanza directly without the cross-process `avanza_order_lock`, unlike the legacy `portfolio.avanza_client`, `portfolio.avanza_session`, and page-session helpers added in this same patch. If one process adopts `portfolio.avanza.trading` while another still uses the older paths, the shared buying-power/position races the lock was introduced to prevent are reintroduced immediately.

- [P2] Preserve `instrument_type` in the BankID scanner fallback — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\scanner.py:77-78
  This fallback drops `itype_str` entirely and posts only `{query, limit}`. When the TOTP client is unavailable, `scan_instruments(..., instrument_type='warrant')` can therefore return stocks/certificates from the BankID search endpoint, and the later filtering/ranking code does not remove them, so the `instrument_type` argument becomes unreliable on that path.
The patch introduces multiple runtime and trading correctness problems in core entry points, including a first-call initialization failure, mis-rounded tick prices, and order-management paths that bypass or misreport safety checks. Those issues are enough to make the change unsafe to consider correct.

Full review comments:

- [P1] Add a bootstrap path for first-time AvanzaClient initialization — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\client.py:54-57
  Every public helper in the new `portfolio.avanza` package eventually calls `AvanzaClient.get_instance()` with no arguments, so the first `get_quote`, `place_order`, `scan_instruments`, etc. in a fresh process raises `ValueError` here unless the caller already knows to initialize the singleton out-of-band. As written, the exported convenience API is unusable on first use.

- [P1] Avoid float drift when converting prices to tick steps — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\tick_rules.py:87-95
  `price_int` stays a float here, so exact on-tick prices can move by a full tick when binary rounding nudges the division just below or above an integer boundary. For example, `0.29` with a `0.01` tick rounds down to `0.28`, and `0.14` with a `0.02` tick rounds up to `0.16`, which means callers can submit worse order prices even when the input was already valid.

- [P1] Reapply the account whitelist on session-backed positions — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_client.py:172-175
  When BankID session auth is available, this branch returns `avanza_session.get_positions()` directly. That helper does not filter by `ALLOWED_ACCOUNT_IDS`, so the pension/other-account leakage that the TOTP branch explicitly guards against is back on the preferred auth path; multi-account users will see non-trading holdings in the portfolio view again.

- [P1] Check `_api_delete`'s `ok` flag before declaring stop-loss success — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza_control.py:397-404
  In the no-page path, `_api_delete()` now returns `{"http_status": ..., "ok": ...}` from `avanza_session.api_delete`, but this wrapper only looks for `errorCode`. A 403/500 therefore comes back as `(True, result)` even though the stop-loss is still active, so downstream sell/re-arm logic can proceed under the false assumption that the encumbered volume has been cleared.

- [P1] Guard unified trading mutations with `avanza_order_lock` — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\trading.py:80-84
  These new trading entry points call Avanza directly without the cross-process `avanza_order_lock`, unlike the legacy `portfolio.avanza_client`, `portfolio.avanza_session`, and page-session helpers added in this same patch. If one process adopts `portfolio.avanza.trading` while another still uses the older paths, the shared buying-power/position races the lock was introduced to prevent are reintroduced immediately.

- [P2] Preserve `instrument_type` in the BankID scanner fallback — Q:\finance-analyzer\.worktrees\adv-avanza-api\portfolio\avanza\scanner.py:77-78
  This fallback drops `itype_str` entirely and posts only `{query, limit}`. When the TOTP client is unavailable, `scan_instruments(..., instrument_type='warrant')` can therefore return stocks/certificates from the BankID search endpoint, and the later filtering/ranking code does not remove them, so the `instrument_type` argument becomes unreliable on that path.
