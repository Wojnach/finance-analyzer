OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-orchestration
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e0833-cfe5-7801-b813-dfa3b121d491
--------
user
changes against 'empty-baseline'
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9 && git diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-08T15:28:00.874348Z ERROR codex_core::tools::router: error=Exit code: 1
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


 exited 1 in 541ms:
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
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Location; Get-ChildItem -Force' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 594ms:

[32;1mPath[0m
[32;1m----[0m
Q:\finance-analyzer\.worktrees\adv-orchestration

[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-orchestration
[32;1mPSChildName         : [0mportfolio
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mportfolio
[32;1mParent              : [0mQ:\finance-analyzer\.worktrees\adv-orchestration
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\finance-analyzer\.worktrees\adv-orchestration\portfolio
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-08 17:25:55
[32;1mCreationTimeUtc     : [0m2026-05-08 15:25:55
[32;1mLastAccessTime      : [0m2026-05-08 17:27:54
[32;1mLastAccessTimeUtc   : [0m2026-05-08 15:27:54
[32;1mLastWriteTime       : [0m2026-05-08 17:25:55
[32;1mLastWriteTimeUtc    : [0m2026-05-08 15:25:55
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mportfolio
[32;1mResolvedTarget      : [0mQ:\finance-analyzer\.worktrees\adv-orchestration\portfolio
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m61
[32;1mName          : [0m.git


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m46
[32;1mName          : [0mBASELINE.md



exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content .git' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 539ms:
gitdir: Q:/finance-analyzer/.git/worktrees/adv-orchestration

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel; git status --short' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-08T15:28:09.237012Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-orchestration'
'Q:/finance-analyzer/.worktrees/adv-orchestration/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-orchestration
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-orchestration'
'Q:/finance-analyzer/.worktrees/adv-orchestration/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-orchestration

 exited 1 in 571ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-orchestration'
'Q:/finance-analyzer/.worktrees/adv-orchestration/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-orchestration
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-orchestration'
'Q:/finance-analyzer/.worktrees/adv-orchestration/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-orchestration

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-orchestration' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-08T15:28:12.372005Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
error: could not lock config file C:/Users/Herc2/.gitconfig: Permission denied

 exited 1 in 547ms:
error: could not lock config file C:/Users/Herc2/.gitconfig: Permission denied

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-orchestration rev-parse --show-toplevel; git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-orchestration diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9; git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-orchestration diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 616ms:
Q:/finance-analyzer/.worktrees/adv-orchestration
 portfolio/agent_invocation.py   | 1407 +++++++++++++++++++++++++++
 portfolio/autonomous.py         |  846 +++++++++++++++++
 portfolio/claude_gate.py        |  684 +++++++++++++
 portfolio/config_validator.py   |   79 ++
 portfolio/health.py             |  448 +++++++++
 portfolio/loop_contract.py      | 2003 +++++++++++++++++++++++++++++++++++++++
 portfolio/main.py               | 1393 +++++++++++++++++++++++++++
 portfolio/market_timing.py      |  341 +++++++
 portfolio/multi_agent_layer2.py |  242 +++++
 portfolio/perception_gate.py    |   95 ++
 portfolio/reflection.py         |  241 +++++
 portfolio/reporting.py          | 1307 +++++++++++++++++++++++++
 portfolio/session_calendar.py   |  211 +++++
 portfolio/tickers.py            |  307 ++++++
 portfolio/trigger.py            |  475 ++++++++++
 15 files changed, 10079 insertions(+)
portfolio/agent_invocation.py
portfolio/autonomous.py
portfolio/claude_gate.py
portfolio/config_validator.py
portfolio/health.py
portfolio/loop_contract.py
portfolio/main.py
portfolio/market_timing.py
portfolio/multi_agent_layer2.py
portfolio/perception_gate.py
portfolio/reflection.py
portfolio/reporting.py
portfolio/session_calendar.py
portfolio/tickers.py
portfolio/trigger.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -File | Select-Object FullName' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 605ms:

[32;1mFullName[0m
[32;1m--------[0m
Q:\finance-analyzer\.worktrees\adv-orchestration\BASELINE.md
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\agent_invocation.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\autonomous.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\claude_gate.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\config_validator.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\health.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\loop_contract.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\main.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\market_timing.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\multi_agent_layer2.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\perception_gate.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\reflection.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\reporting.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\session_calendar.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\tickers.py
Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\trigger.py


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'python -m py_compile portfolio/*.py' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-08T15:28:27.593788Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[Errno 22] Invalid argument: 'portfolio/*.py'
 exited 1 in 599ms:
[Errno 22] Invalid argument: 'portfolio/*.py'
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | ForEach-Object { python -m py_compile "'$_.FullName }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 1427ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | ForEach-Object { Write-Output ('### ' + "'$_.Name); Select-String -Path $_.FullName -Pattern '"'"'^(def|class) '"' | ForEach-Object { "'$_.LineNumber.ToString().PadLeft(4) + '"': ' + "'$_.Line.Trim() }; Write-Output '"'' }" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 739ms:
### agent_invocation.py
  82: def _completion_watchdog() -> None:
 106: def _ensure_completion_watchdog() -> None:
 136: def _stop_completion_watchdog(timeout_s: float = 1.0) -> None:
 153: def _load_stack_overflow_counter() -> int:
 162: def _save_stack_overflow_counter(count: int) -> None:
 181: def _build_tier_prompt(tier, reasons):
 224: def _extract_ticker(reasons):
 243: def _log_trigger(reasons, status, tier=None):
 254: def _load_guard_warnings():
 282: def _build_decision_feedback(ticker, max_entries=5):
 333: def _last_jsonl_ts(path):
 341: def _safe_last_jsonl_ts(path, label):
 350: def _safe_elapsed_s():
 393: def _scan_agent_log_for_auth_failure(label: str, extra_context: dict | None = None) -> bool:
 446: def _kill_overrun_agent(fallback_reasons=None, fallback_tier=None):
 539: def invoke_agent(reasons, tier=3):
 939: def _write_fishing_context(journal_entry):
1057: def _record_new_trades():
1093: def check_agent_completion():
1122: def _check_agent_completion_locked():
1335: def get_completion_stats(hours=24):

### autonomous.py
  48: def _consensus_accuracy():
  80: def autonomous_decision(config, signals, prices_usd, fx_rate, state,
  92: def _autonomous_decision_inner(config, signals, prices_usd, fx_rate, state,
 189: def _classify_tickers(signals, patient_state, bold_state, tier, triggered_tickers):
 261: def _ticker_prediction(ticker, sig, tf_entries):
 362: def _build_reflection(prev_entry, current_prices):
 396: def _detect_regime(signals):
 413: def _strategy_reasoning(predictions, strategy):
 437: def _build_watchlist(predictions, reasons):
 452: def _format_price(price):
 464: def _tf_heatmap(tf_entries):
 484: def _build_telegram(actionable, hold_count, sell_count, patient_state, bold_state,
 505: def _build_telegram_mode_a(actionable, hold_count, sell_count, patient_state, bold_state,
 658: def _build_telegram_mode_b(actionable, hold_count, sell_count, patient_state, bold_state,
 786: def _should_send(predictions, reasons, tier):
 821: def _update_throttle():
 835: def _load_bold_state_safe():

### claude_gate.py
  80: def _load_config_layer2_enabled() -> bool:
  96: def _clean_env() -> dict:
 142: def _is_real_auth_marker_line(line: str, marker: str) -> bool:
 162: def record_critical_error(
 202: def detect_auth_failure(output: str, caller: str, context: dict | None = None) -> bool:
 258: def _find_claude_cmd() -> str | None:
 263: def _log_invocation(entry: dict) -> None:
 271: def _count_today_invocations() -> int:
 294: def _popen_kwargs_for_tree_kill() -> dict:
 301: def _kill_process_tree(proc: subprocess.Popen, *, label: str = "claude") -> None:
 347: def _run_with_tree_kill(
 396: def invoke_claude(
 554: def invoke_claude_text(
 651: def get_invocation_stats() -> dict:

### config_validator.py
  33: def validate_config(config: dict) -> list[str]:
  50: def validate_config_file() -> dict:

### health.py
  20: def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
  44: def load_health() -> dict:
  52: def reset_session_start():
  64: def heartbeat() -> None:
  96: class heartbeat_keepalive:  # noqa: N801 — context-manager naming convention
 152: def check_staleness(max_age_seconds: int = 300) -> tuple:
 165: def check_agent_silence(max_market_seconds: int = 7200,
 219: def update_module_failures(failures: list):
 250: def update_signal_health(signal_name: str, success: bool):
 259: def update_signal_health_batch(results: dict):
 297: def get_signal_health(signal_name: str = None) -> dict:
 310: def get_signal_health_summary() -> dict:
 334: def get_health_summary() -> dict:
 367: def check_outcome_staleness(max_age_hours: int = 36) -> dict:
 420: def check_dead_signals(recent_entries: int = 20) -> list[str]:

### loop_contract.py
 214: def _get_layer2_grace_s(health: dict | None) -> int:
 233: class CycleReport:
 258: class Violation:
 267: def _parse_iso(ts: str | None) -> datetime | None:
 276: def check_layer2_journal_activity(now: datetime | None = None) -> list[Violation]:
 498: def verify_contract(report: CycleReport, previous_signal_counts: dict | None = None) -> list[Violation]:
 725: def check_signal_accuracy_degradation_safe() -> list[Violation]:
 747: def check_snapshot_freshness_safe() -> list[Violation]:
 779: def _check_snapshot_freshness() -> list[Violation]:
 834: def check_signal_log_reconciliation_safe() -> list[Violation]:
 843: def _check_signal_log_reconciliation() -> list[Violation]:
 883: def _has_unresolved_critical_entry(
 980: def _dispatch_critical_errors_for_degradation(
1111: class MetalsCycleReport:
1135: def verify_metals_contract(report: MetalsCycleReport) -> list[Violation]:
1223: class BotCycleReport:
1246: def verify_bot_contract(report: BotCycleReport) -> list[Violation]:
1323: class ViolationTracker:
1421: def _build_heal_prompt(violations: list[Violation], loop_name: str = "main") -> str:
1456: def _log_violations(violations: list[Violation], cycle_id: int):
1479: def _hash_violation_message(message: str) -> str:
1492: def violation_identity_payload(
1551: def _hash_violation_identity(violation: "Violation") -> str:
1568: def _normalize_recent_hashes(prior: dict) -> list[dict]:
1600: def _telegram_will_actually_deliver(config: dict | None,
1629: def _filter_critical_by_cooldown(critical: list[Violation], now: float,
1694: def _clear_alert_state_for_passed_invariants(
1733: def _persist_alert_cooldown(
1762: def _alert_violations(violations: list[Violation], config: dict,
1852: def _trigger_self_heal(violations: list[Violation], tracker: ViolationTracker,
1890: def verify_and_act(report, config: dict,

### main.py
  57: def _acquire_singleton_lock():
  95: def _release_singleton_lock():
 247: def _extract_triggered_tickers(reasons):
 263: def _run_post_cycle(config, report=None):
 426: def run(force_report=False, active_symbols=None):
 975: def _load_crash_counter() -> int:
 983: def _save_crash_counter(count: int) -> None:
 994: def _crash_alert(error_msg):
1038: def _crash_sleep():
1067: def _safe_crash_recovery(traceback_text: str) -> None:
1106: def _reset_crash_counter():
1115: def _sleep_for_next_cycle(previous_cycle_started, interval_s):
1129: def loop(interval=None):

### market_timing.py
  29: def _is_eu_dst(dt):
  53: def _eu_market_open_hour_utc(dt):
  67: def _is_us_dst(dt):
  92: def _market_close_hour_utc(dt):
 109: def _easter_sunday(year):
 124: def _observed(d):
 137: def _nth_weekday(year, month, weekday, n):
 147: def _last_weekday(year, month, weekday):
 164: def us_market_holidays(year):
 190: def is_us_market_holiday(dt=None):
 198: def swedish_market_holidays(year):
 235: def is_swedish_market_holiday(dt=None):
 243: def _is_agent_window(now=None):
 262: def _market_open_hour_utc(dt):
 274: def is_us_stock_market_open(now=None, pre_market_buffer_min=0, post_market_buffer_min=0):
 303: def should_skip_gpu(ticker, config=None, now=None):
 326: def get_market_state():

### multi_agent_layer2.py
  70: def build_specialist_prompts(
  97: def build_synthesis_prompt(
 122: def get_report_paths() -> list[str]:
 127: def launch_specialists(
 185: def wait_for_specialists(
 232: def cleanup_reports() -> None:

### perception_gate.py
  29: def should_invoke(reasons, tier, config=None):
  88: def _load_compact_summary():

### reflection.py
  30: def _count_trades(portfolio):
  35: def _compute_strategy_metrics(portfolio):
  91: def _regime_distribution():
 101: def _generate_insights(patient_metrics, bold_metrics):
 129: def should_reflect(config=None):
 177: def compute_reflection():
 202: def save_reflection(reflection):
 208: def maybe_reflect(config=None):
 223: def load_latest_reflection(max_age_days=7):

### reporting.py
  37: def _track_module_outcome(name: str, ok: bool, exc: BaseException | None = None) -> None:
  90: def _cross_asset_signals(all_signals):
 108: def write_agent_summary(
 806: def _update_signal_state_since(summary: dict) -> None:
 841: def _get_held_tickers():
 861: def _write_compact_summary(summary):
1019: def write_tiered_summary(summary, tier, triggered_tickers=None):
1034: def _portfolio_snapshot(state_file, prices_usd=None, fx_rate=None):
1075: def _macro_headline(summary):
1128: def _write_tier1_summary(summary):
1197: def _write_tier2_summary(summary, triggered_tickers=None):

### session_calendar.py
  21: class SessionInfo:
  50: def _eu_dst(dt: datetime) -> bool:
  70: def _cet_offset(dt: datetime) -> int:
  75: def _cet_to_utc(hour: int, minute: int, dt: datetime) -> time:
  82: def _make_session_end(now: datetime, cet_hour: int, cet_minute: int) -> datetime:
 115: def get_session_info(instrument_type: str,
 207: def remaining_session_minutes(instrument_type: str = "warrant",

### tickers.py
 171: def _clean_disabled_reason(lines: list[str]) -> str:
 184: def _parse_disabled_reasons() -> dict[str, str]:
 228: def get_disabled_reason(signal_name: str) -> str | None:

### trigger.py
  64: def _update_sustained(
  98: def _today_str():
 102: def _load_state():
 106: def _save_state(state):
 121: def _check_recent_trade(state):
 152: def check_triggers(signals, prices_usd, fear_greeds, sentiments):
 374: def _reason_is_downshiftable(reason: str, threshold: float) -> bool:
 392: def _should_downshift_to_t1(reasons, threshold: float | None = None) -> bool:
 413: def classify_tier(reasons, state=None):
 465: def update_tier_state(tier, state=None):


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content BASELINE.md' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 555ms:
# Empty baseline for adversarial review diffs

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | ForEach-Object { Write-Output ('### ' + "'$_.Name); Get-Content $_.FullName -TotalCount 60; Write-Output '"'---' }" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 745ms:
### agent_invocation.py
"""Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""

import logging
import os
import platform
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from portfolio.api_utils import load_config as _load_config
from portfolio.claude_gate import detect_auth_failure
from portfolio.file_utils import atomic_append_jsonl, last_jsonl_entry, load_jsonl
from portfolio.message_store import send_or_store
from portfolio.telegram_notifications import escape_markdown_v1

logger = logging.getLogger("portfolio.agent")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
TELEGRAM_FILE = DATA_DIR / "telegram_messages.jsonl"
PATIENT_PORTFOLIO = DATA_DIR / "portfolio_state.json"
BOLD_PORTFOLIO = DATA_DIR / "portfolio_state_bold.json"

# BUG-214: Drawdown circuit breaker thresholds.
# Advisory at WARN level, hard-block at BLOCK level.
# User accepts 10-20% knockout risk; only de-risk at 50%+.
_DRAWDOWN_WARN_PCT = 20.0
_DRAWDOWN_BLOCK_PCT = 50.0

_agent_proc = None
_agent_log = None
_agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
_agent_start = 0
# P2B follow-up (Codex P2 #2, 2026-04-17): fallback wall-clock timestamp
# for timeout enforcement when `_agent_start` (monotonic) gets poisoned.
# The clamp alone could silently disable the P1B T1 timeout check; this
# fallback lets _safe_elapsed_s() recover a plausible elapsed from wall
# clock so the hung agent still gets killed. Always set alongside
# _agent_start so the pair are in sync.
_agent_start_wall = 0.0
_agent_timeout = 900  # per-invocation timeout (set from tier config)
_agent_tier = None  # tier of the currently running agent
_agent_reasons = None  # trigger reasons for the current invocation
_journal_ts_before = None  # last journal timestamp before agent started
_telegram_ts_before = None  # last telegram timestamp before agent started

# BUG-219: Transaction counts at invoke time — used by check_agent_completion()
# to detect new trades and call record_trade() for overtrading prevention.
# PR-R4-4: record_trade() was never called from production code; this wires it.
_patient_txn_count_before = 0
_bold_txn_count_before = 0

# Stack overflow detection — exit code 3221225794 = Windows STATUS_STACK_OVERFLOW (0xC00000FD)
_STACK_OVERFLOW_EXIT_CODE = 3221225794
---
### autonomous.py
"""Autonomous decision engine for the main portfolio loop.

Replaces _maybe_send_alert() when layer2.enabled=false. Provides:
- Signal-based ticker classification and prediction
- Journal entries (same format as Claude Layer 2)
- Decision log with full signal data
- Rich Telegram messages (Mode A / Mode B)
- Throttling for routine HOLD messages

No trade execution — decisions are logged as recommendations only.
"""

import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
from portfolio.message_store import send_or_store
from portfolio.notification_text import (
    format_confidence,
    format_fear_greed,
    format_portfolio_context,
    format_vote_summary,
    humanize_ticker,
)
from portfolio.portfolio_mgr import load_bold_state, portfolio_value
from portfolio.telegram_notifications import escape_markdown_v1

logger = logging.getLogger("portfolio.autonomous")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
DECISIONS_FILE = DATA_DIR / "layer2_decisions.jsonl"
THROTTLE_FILE = DATA_DIR / "autonomous_throttle.json"

_HOLD_COOLDOWN_SECONDS = 1800  # 30 minutes between routine HOLD messages
_TF_ORDER = ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]
_MIN_BUY_VOTES = 3             # raw BUY votes required to classify as BUY
_BUY_MUST_DOMINATE = True      # BUY votes must exceed SELL votes

_consensus_acc_cache = None
_consensus_acc_cache_ts = 0
_CONSENSUS_ACC_TTL = 300  # re-read every 5 minutes


def _consensus_accuracy():
    """Load cached consensus accuracy from agent_summary (compact preferred)."""
    global _consensus_acc_cache, _consensus_acc_cache_ts
    import time
    now = time.monotonic()
    if now - _consensus_acc_cache_ts < _CONSENSUS_ACC_TTL:
        return _consensus_acc_cache

    for fname in ("agent_summary_compact.json", "agent_summary.json"):
        path = DATA_DIR / fname
        summary = load_json(path, default=None)
        if not summary:
            continue
---
### claude_gate.py
"""Centralized Claude Code invocation gatekeeper.

This module is the ONLY approved way to invoke Claude Code (``claude -p``)
from anywhere in the codebase.  All callers — agent_invocation, metals_loop,
silver_monitor, claude_fundamental, analyze, bigbet, iskbets, etc. — MUST
route through ``invoke_claude()`` defined here.

Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
Doing so bypasses the kill switch, rate limiter, and invocation tracking.

Usage::

    from portfolio.claude_gate import invoke_claude

    success, exit_code = invoke_claude(
        prompt="Analyze BTC-USD",
        caller="silver_monitor",
        model="sonnet",
        max_turns=20,
        timeout=180,
    )
"""

import contextlib
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_jsonl

logger = logging.getLogger("portfolio.claude_gate")

import threading

# ---------------------------------------------------------------------------
# Master kill switch.  Set to False to block ALL Claude Code invocations
# across the entire codebase — no exceptions.
# ---------------------------------------------------------------------------
CLAUDE_ENABLED = True

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
INVOCATIONS_LOG = DATA_DIR / "claude_invocations.jsonl"
# 2026-04-13: Append-only journal of failures that EVERY future Claude Code
# session must see. Intentionally separate from claude_invocations.jsonl so
# hooks and startup scripts can cheaply poll it without parsing routine
# invocation noise. Consumed by scripts/check_critical_errors.py, which is
# referenced from CLAUDE.md to guarantee surfacing at session start.
CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"

# A-IN-3 (2026-04-11): In-process concurrency lock. Without this, the main
# loop's 8-worker ticker pool + the metals loop's fast-tick + signal
---
### config_validator.py
"""Config validation for portfolio system startup.

Validates config.json has all required keys before the main loop starts.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("portfolio.config_validator")

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

# Required: missing any of these is a fatal error
REQUIRED_KEYS = [
    ("telegram", "token"),
    ("telegram", "chat_id"),
    ("alpaca", "key"),
    ("alpaca", "secret"),
]

# Optional: missing these produces a warning but isn't fatal
OPTIONAL_KEYS = [
    ("mistral_api_key",),
    ("iskbets",),
    ("newsapi_key",),
    ("alpha_vantage", "api_key"),
    ("golddigger", "fred_api_key"),
    ("bgeometrics", "api_token"),
]


def validate_config(config: dict) -> list[str]:
    """Validate config dict. Returns list of error strings (empty = valid)."""
    errors = []
    for key_path in REQUIRED_KEYS:
        obj = config
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                errors.append(f"missing required key: {'.'.join(key_path)}")
                break
            obj = obj[key]
        else:
            # Key exists — check it's not empty/placeholder
            if isinstance(obj, str) and not obj.strip():
                errors.append(f"empty value for required key: {'.'.join(key_path)}")
    return errors


def validate_config_file() -> dict:
    """Load config.json, validate, and return it.

    Logs warnings for missing optional keys.
    Raises ValueError if required keys are missing.
    """
    if not CONFIG_FILE.exists():
        raise ValueError(f"config.json not found at {CONFIG_FILE}")

    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
---
### health.py
"""Health monitoring for the finance-analyzer Layer 1 loop."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, last_jsonl_entry, load_json, load_jsonl_tail

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEALTH_FILE = DATA_DIR / "health_state.json"

# C10/H17: Protect all read-modify-write sequences in health.py.
_health_lock = threading.Lock()


def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
                  last_trigger_reason: str = None, error: str = None):
    """Called at end of each Layer 1 cycle to update health state."""
    with _health_lock:
        state = load_health()
        state["last_heartbeat"] = datetime.now(UTC).isoformat()
        state["cycle_count"] = cycle_count
        state["signals_ok"] = signals_ok
        state["signals_failed"] = signals_failed
        state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
        if last_trigger_reason:
            state["last_trigger_reason"] = last_trigger_reason
            state["last_trigger_time"] = datetime.now(UTC).isoformat()
            # Cache the invocation timestamp so check_agent_silence() can avoid
            # re-parsing invocations.jsonl on every call.
            state["last_invocation_ts"] = state["last_trigger_time"]
        if error:
            state["errors"] = state.get("errors", [])[-19:] + [
                {"ts": datetime.now(UTC).isoformat(), "error": error}
            ]
            state["error_count"] = state.get("error_count", 0) + 1
        atomic_write_json(HEALTH_FILE, state)


def load_health() -> dict:
    """Load current health state. Returns defaults if missing or corrupt."""
    state = load_json(HEALTH_FILE)
    if state is not None:
        return state
    return {"start_time": time.time(), "cycle_count": 0, "error_count": 0, "errors": []}


def reset_session_start():
    """Reset start_time to current time — call at loop startup.

    Prevents uptime_seconds from inheriting a stale start_time
    from a previous session's health_state.json.
    """
    with _health_lock:
        state = load_health()
        state["start_time"] = time.time()
---
### loop_contract.py
"""Loop Contract — runtime invariant verification for all system loops.

After every cycle, verify functions check that critical operations
actually happened. Violations are logged, alerted, and optionally
trigger a self-healing Claude Code session.

Supports: main loop, metals loop, GoldDigger, Elongir.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    last_jsonl_entry,
    load_json,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONTRACT_STATE_FILE = DATA_DIR / "contract_state.json"
CONTRACT_LOG_FILE = DATA_DIR / "contract_violations.jsonl"
CONFIG_FILE = BASE_DIR / "config.json"
HEALTH_STATE_FILE = DATA_DIR / "health_state.json"
LAYER2_JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"

# 2026-04-28: Per-invariant Telegram alert cooldown. Background: the
# accuracy_degradation invariant uses a throttled-replay design (replays
# cached violations every cycle to keep ViolationTracker.consecutive
# alive). Without per-alert dedup _alert_violations shipped one Telegram
# per cycle for 192 cycles in a row before we noticed. The cooldown only
# suppresses *exact* replays — same invariant + same message text within
# the window. Any text change (a new degraded signal joining the alert
# list, a different trigger reason on layer2_journal_activity) bypasses
# the cooldown and re-fires immediately. Configurable via
# notification.contract_alert_cooldown_s; defaults to 4 h, which is short
# enough that a stuck regression still pages the user a few times per day
# but long enough that the same-text replay is rate-limited 24x.
DEFAULT_CONTRACT_ALERT_COOLDOWN_S = 4 * 3600

# 2026-04-28 (Codex P2): TTL for the critical_errors.jsonl dedup. After
# this many seconds, a same-text degradation replay re-emits a fresh
# critical_errors row so the auto-fix-agent dispatcher
# (PF-FixAgentDispatcher, 24 h lookback in scripts/fix_agent_dispatcher.py)
# keeps seeing the incident as long as it persists. 6 h gives the
# dispatcher 4 fresh entries per dispatcher-day, well inside its
# lookback window, while still rate-limiting the same-issue noise 4x
# vs the per-cycle pattern that prompted this fix.
DEFAULT_CRITICAL_ERRORS_DEDUP_TTL_S = 6 * 3600

# 2026-04-28 (Codex P2): invariants whose CRITICAL violations get routed
# to critical_errors.jsonl after ViolationTracker has had a chance to
---
### main.py
#!/usr/bin/env python3
"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data

This is the orchestrator module. All logic has been extracted to:
- shared_state.py — mutable globals, caching, rate limiters
- market_timing.py — DST-aware market hours, agent window
- fx_rates.py — USD/SEK exchange rate fetching
- indicators.py — compute_indicators, detect_regime, technical_signal
- data_collector.py — Binance/Alpaca/yfinance kline fetchers
- signal_engine.py — 30-signal voting system, generate_signal
- portfolio_mgr.py — portfolio state load/save/value
- reporting.py — agent_summary.json builder
- telegram_notifications.py — Telegram send/escape/alert
- digest.py — 4-hour digest builder
- daily_digest.py — morning daily digest (focus instruments + movers)
- message_throttle.py — analysis message rate limiting
- agent_invocation.py — Layer 2 Claude Code subprocess
- logging_config.py — structured logging setup
"""

import atexit
import logging
import os
import random
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, atomic_write_text, load_json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"

logger = logging.getLogger("portfolio.loop")

# --- Singleton guard (same pattern as metals_loop.py) ---
# C5: Support both Windows (msvcrt) and Linux/WSL (fcntl) locking.
# Previously, non-Windows silently returned True — no protection.
try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import fcntl
except ImportError:
    fcntl = None

_SINGLETON_LOCK_FILE = str(DATA_DIR / "main_loop.singleton.lock")
_DUPLICATE_EXIT_CODE = 11
_singleton_lock_fh = None


def _acquire_singleton_lock():
    """Acquire single-instance lock for main loop (non-blocking).

    C5: Now supports both Windows (msvcrt) and Linux/WSL (fcntl).
---
### market_timing.py
"""Market timing utilities — DST-aware NYSE and EU hours, market state detection.

Includes US (NYSE) and Swedish (Nasdaq Stockholm / Avanza) holiday calendars
so the system skips stock/warrant processing on public holidays, not just weekends.
"""

from datetime import UTC, date, datetime, timedelta

from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS, SYMBOLS

# Backward compat: MARKET_OPEN_HOUR kept at 7 (summer value).
# Callers that need DST-aware EU open should use _eu_market_open_hour_utc().
MARKET_OPEN_HOUR = 7

# Loop intervals by market state.
# 2026-04-09: all bumped to 600s (10 min). The reduced 5-ticker universe +
# warm fingpt daemon means we no longer need a 60s fast tick — giving the
# loop breathing room eliminates cadence overruns without losing anything
# meaningful. Weekend was already 600s. See docs/PLAN_FINGPT_DAEMON.md.
INTERVAL_MARKET_OPEN = 600    # 10 min — previously 60s (pre-daemon era)
INTERVAL_MARKET_CLOSED = 600  # 10 min — previously 120s
INTERVAL_WEEKEND = 600        # 10 min — unchanged

# States where US stock markets are NOT trading — use this tuple instead of
# hardcoding ("closed", "weekend") to avoid missing the "holiday" state.
MARKET_CLOSED_STATES = ("closed", "weekend", "holiday")


def _is_eu_dst(dt):
    """Check if a UTC datetime falls within EU Summer Time (CEST).

    EU DST rule:
      Starts: last Sunday of March at 01:00 UTC
      Ends:   last Sunday of October at 01:00 UTC

    Returns True during CEST (summer), False during CET (winter).
    """
    year = dt.year

    # Last Sunday of March
    mar31 = date(year, 3, 31)
    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
    eu_dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)

    # Last Sunday of October
    oct31 = date(year, 10, 31)
    last_sun_oct = 31 - (oct31.weekday() + 1) % 7
    eu_dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)

    return eu_dst_start <= dt < eu_dst_end


def _eu_market_open_hour_utc(dt):
    """Return the EU market open hour in UTC, adjusted for EU DST.

    H47: London/Frankfurt open at 08:00 local time.
    CEST (summer, BST=UTC+1): 08:00 local = 07:00 UTC
    CET (winter, GMT=UTC+0): 08:00 local = 08:00 UTC

    Previously hardcoded to 7 UTC year-round, which missed the winter hour.
---
### multi_agent_layer2.py
"""Multi-agent Layer 2 orchestration — parallel specialists + synthesis.

Inspired by Claude Code's Coordinator Mode. Instead of one monolithic
agent reading everything, splits analysis into parallel specialists:

    1. Technical Agent: signals, regime, momentum, trend
    2. Risk Agent: portfolio state, exposure, drawdown, stops
    3. Microstructure Agent: order flow, depth, cross-asset context

Each specialist writes a brief report to a temp file. A synthesis agent
reads all three and makes the final BUY/SELL/HOLD decision.

Key design principles (from Claude Code's Agent architecture):
    - Fresh context per agent (no context pollution)
    - 5-word task description forces clarity
    - Standardized report format for mechanical parsing
    - Parent owns the gate — synthesis agent makes final call
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from portfolio.claude_gate import detect_auth_failure

logger = logging.getLogger("portfolio.multi_agent_layer2")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SPECIALISTS = {
    "technical": {
        "focus": "Technical analysis: signals, regime, momentum, trend direction",
        "data_files": [
            "data/agent_context_t2.json",
            "data/accuracy_cache.json",
        ],
        "output_file": "data/_specialist_technical.md",
        "timeout": 120,
        "max_turns": 10,
    },
    "risk": {
        "focus": "Risk assessment: portfolio exposure, drawdown, stop levels, position sizing",
        "data_files": [
            "data/portfolio_state.json",
            "data/portfolio_state_bold.json",
            "data/portfolio_state_warrants.json",
        ],
        "output_file": "data/_specialist_risk.md",
        "timeout": 90,
        "max_turns": 8,
    },
    "microstructure": {
        "focus": "Order flow and cross-asset: depth imbalance, trade flow, VPIN, copper, GVZ, gold/silver ratio",
        "data_files": [
            "data/microstructure_state.json",
---
### perception_gate.py
"""Perception gate — filters low-value Layer 2 invocations.

A rule-based pre-invocation filter (NOT an LLM call). Checks signal
consensus strength and trigger importance. If the gate decides to skip,
the agent is not invoked, saving tokens and latency.

Config:
    "perception_gate": {
        "enabled": true,
        "min_signal_strength": 0.3,
        "skip_tiers": [1]
    }
"""

import logging
from pathlib import Path

from portfolio.api_utils import load_config
from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.perception_gate")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Trigger keywords that always bypass the gate
_BYPASS_KEYWORDS = ("consensus", "F&G crossed", "post-trade")


def should_invoke(reasons, tier, config=None):
    """Decide whether to invoke Layer 2.

    Args:
        reasons: list[str] of trigger reasons.
        tier: int (1, 2, or 3).
        config: optional config dict. Loaded from disk if None.

    Returns:
        (should_invoke: bool, reason: str) explaining the decision.
    """
    if config is None:
        config = load_config()

    gate_cfg = config.get("perception_gate", {})
    if not gate_cfg.get("enabled", False):
        return True, "gate disabled"

    skip_tiers = gate_cfg.get("skip_tiers", [1])
    if tier not in skip_tiers:
        return True, f"T{tier} not in skip_tiers"

    # Force-bypass for important triggers
    for reason in reasons:
        for keyword in _BYPASS_KEYWORDS:
            if keyword in reason:
                return True, f"bypass: {keyword!r} in trigger"

    # Check signals from compact summary
    min_strength = gate_cfg.get("min_signal_strength", 0.3)
    summary = _load_compact_summary()
    if summary is None:
---
### reflection.py
"""Periodic trade reflection — computes structured performance metrics.

Pure Python (no LLM call). After every N trades, generates a reflection
summary stored in data/reflections.jsonl. Layer 2 reads the latest
reflection as context for self-assessment.

Config:
    "reflection": {
        "enabled": true,
        "trade_interval": 10,
        "max_age_days": 7
    }
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl

logger = logging.getLogger("portfolio.reflection")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFLECTIONS_FILE = DATA_DIR / "reflections.jsonl"
PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"


def _count_trades(portfolio):
    """Count total trades in a portfolio state."""
    return len(portfolio.get("transactions", []))


def _compute_strategy_metrics(portfolio):
    """Compute win rate, avg PnL, and total PnL for a strategy.

    Returns dict with trades, win_rate, avg_pnl_pct, total_pnl_pct, holdings.
    """
    txns = portfolio.get("transactions", [])
    initial = portfolio.get("initial_value_sek", 500000)
    cash = portfolio.get("cash_sek", initial)
    holdings = portfolio.get("holdings", {})
    holding_tickers = [t for t, h in holdings.items() if h.get("shares", 0) > 0]

    if not txns:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_pnl_pct": None,
            "total_pnl_pct": round((cash - initial) / initial * 100, 2),
            "holdings": holding_tickers,
        }

    # Match BUY/SELL pairs per ticker for PnL
    buys = {}  # ticker -> list of (price_sek, shares)
    sells = []  # list of (ticker, pnl_pct)

    for tx in txns:
        ticker = tx.get("ticker", "")
---
### reporting.py
"""Agent summary reporting — builds JSON summaries for Layer 2 consumption."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import portfolio.shared_state as _ss
from portfolio.file_utils import load_json
from portfolio.indicators import detect_regime
from portfolio.portfolio_mgr import _atomic_write_json, portfolio_value
from portfolio.shared_state import _cached
from portfolio.signal_registry import get_enhanced_signals
from portfolio.tickers import CRYPTO_SYMBOLS, STOCK_SYMBOLS

logger = logging.getLogger("portfolio.reporting")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
COMPACT_SUMMARY_FILE = DATA_DIR / "agent_summary_compact.json"
TIER1_FILE = DATA_DIR / "agent_context_t1.json"
TIER2_FILE = DATA_DIR / "agent_context_t2.json"
SIGNAL_STATE_SINCE_FILE = DATA_DIR / "signal_state_since.json"

# 2026-04-22: escalate persistent silent failures to critical_errors.jsonl.
# The MC seed=None bug survived weeks undetected because reporting.py's bare
# `except Exception → logger.warning` suppressed it below the radar. Track
# consecutive failures per module and escalate once when the streak crosses
# _FAILURE_STREAK_THRESHOLD, so the CLAUDE.md startup check surfaces the
# pattern instead of operators needing to grep log tails.
_module_failure_streaks: dict[str, int] = {}
_module_escalated: set[str] = set()
_FAILURE_STREAK_THRESHOLD = 10  # ~10 minutes at 60s cycles


def _track_module_outcome(name: str, ok: bool, exc: BaseException | None = None) -> None:
    """Track consecutive failures for a reporting submodule. Escalate once
    per streak when the threshold is crossed. Resets on success."""
    if ok:
        _module_failure_streaks.pop(name, None)
        _module_escalated.discard(name)
        return
    streak = _module_failure_streaks.get(name, 0) + 1
    _module_failure_streaks[name] = streak
    if streak >= _FAILURE_STREAK_THRESHOLD and name not in _module_escalated:
        _module_escalated.add(name)
        try:
            from portfolio.claude_gate import record_critical_error
            record_critical_error(
                category="reporting_module_failure_streak",
                caller=f"reporting.{name}",
                message=(
                    f"reporting.{name} has failed {streak} consecutive cycles. "
                    "This module's bare-except was silently suppressing errors "
                    "below the radar — investigate and append a resolution."
                ),
                context={
                    "module": name,
                    "streak": streak,
---
### session_calendar.py
"""Session calendar — instrument-specific trading hours and session state.

Provides remaining-session time, session boundaries, and session mismatch
detection for the exit optimizer.

Usage:
    from portfolio.session_calendar import get_session_info
    info = get_session_info("warrant", underlying="XAG-USD")
    # info.remaining_minutes, info.session_end, info.is_extended, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from portfolio.market_timing import _is_us_dst


@dataclass(frozen=True)
class SessionInfo:
    """Trading session state for an instrument.

    Attributes:
        session_end: Absolute datetime (UTC) of normal session close.
        extended_end: Absolute datetime (UTC) of extended session close, if applicable.
        remaining_minutes: Minutes until effective close (extended if available).
        is_open: Whether the instrument is currently tradeable.
        is_extended: Whether we're in the extended (evening) session.
        underlying_open: Whether the underlying's primary market is open (for warrants).
        phase: Human-readable phase: "open", "extended", "pre_open", "closed".
    """
    session_end: datetime
    extended_end: datetime | None
    remaining_minutes: float
    is_open: bool
    is_extended: bool
    underlying_open: bool
    phase: str


# ---------------------------------------------------------------------------
# Session definitions (times in UTC)
# ---------------------------------------------------------------------------

# Avanza commodity warrants: 08:15-21:55 CET = 07:15-20:55 UTC (winter)
# CET = UTC+1 (winter), CEST = UTC+2 (summer)
# We handle DST for EU sessions too.

def _eu_dst(dt: datetime) -> bool:
    """Check if datetime falls in EU Central European Summer Time (CEST).

    EU DST: last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC.
    """
    year = dt.year

    # Last Sunday of March
    mar31 = datetime(year, 3, 31, tzinfo=UTC)
    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
    dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)
---
### tickers.py
"""Single source of truth for all ticker lists, source mappings, and symbol constants.

Every module that needs ticker definitions should import from here instead
of maintaining its own copy.
"""

import re
from functools import lru_cache
from pathlib import Path

# ── Tier 1: Full signals (30 signals, 7 timeframes) ──────────────────────

SYMBOLS = {
    # Crypto (Binance spot)
    "BTC-USD": {"binance": "BTCUSDT"},
    "ETH-USD": {"binance": "ETHUSDT"},
    # Metals (Binance futures)
    "XAU-USD": {"binance_fapi": "XAUUSDT"},
    "XAG-USD": {"binance_fapi": "XAGUSDT"},
    # US Equities (Alpaca IEX) — MSTR kept as BTC NAV-premium reference for metals_loop
    "MSTR": {"alpaca": "MSTR"},
    # Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT
    # Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT
    #   Reduces main loop load to stay under 60s cadence. Cycle p50 was 143s with
    #   12 tickers — dropping to 5 is expected to bring p50 under target. MSTR retained
    #   because data/metals_loop.py uses it for BTC NAV-premium tracking.
}

# ── Asset-class subsets ───────────────────────────────────────────────────

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
STOCK_SYMBOLS = {"MSTR"}

# All known tickers (union of all subsets)
ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS

# ── Derived mappings (all from SYMBOLS — single source of truth) ─────────

BINANCE_SPOT_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance"
}
BINANCE_FAPI_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance_fapi"
}
BINANCE_MAP = {**BINANCE_SPOT_MAP, **BINANCE_FAPI_MAP}

# Ticker -> (source_type, symbol) mapping (used by macro_context)
TICKER_SOURCE_MAP = {
    t: next(iter(src.items())) for t, src in SYMBOLS.items()
}

# Yahoo Finance symbol mapping — stock tickers map to themselves
YF_MAP = {t: t for t in STOCK_SYMBOLS}

# ── Signal names (used by outcome_tracker, accuracy_stats) ───────────────
# Canonical source is portfolio.signal_registry.get_signal_names().
# This static list is kept for backward compatibility with modules that
---
### trigger.py
"""Smart trigger system — detects meaningful market changes to reduce noise.

Layer 1 runs on a 10-minute cadence during every market state (see
``portfolio/market_timing.py:INTERVAL_MARKET_OPEN``). Layer 2 is invoked when:
- Signal consensus: any ticker NEWLY reaches BUY or SELL from HOLD
- Signal flip sustained for SUSTAINED_CHECKS consecutive cycles (see below)
- Price moved >2% since last trigger
- Fear & Greed crossed extreme threshold (20 or 80)
- Sentiment reversal: sustained for SUSTAINED_CHECKS cycles (filters oscillation)
- Post-trade reassessment: after a BUY/SELL trade

No periodic cooldown — Layer 2 is only invoked when Layer 1 detects a
meaningful change. The Tier 3 periodic full review (every 2h market / 4h
off-hours) provides the "heartbeat" via classify_tier(), but only when
another trigger has already fired.
"""

import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.trigger")

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "data" / "trigger_state.json"
PORTFOLIO_FILE = BASE_DIR / "data" / "portfolio_state.json"
PORTFOLIO_BOLD_FILE = BASE_DIR / "data" / "portfolio_state_bold.json"

PRICE_THRESHOLD = 0.02  # 2% move
FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries
# A signal flip triggers Layer 2 when EITHER of these holds:
#   - SUSTAINED_CHECKS consecutive cycles show the new action, OR
#   - SUSTAINED_DURATION_S seconds of wall-clock time have elapsed since
#     the flip first appeared.
# The count path is the original behavior (unchanged at the 60s cadence).
# The duration path is new (added 2026-04-09 with the cadence bump to 600s);
# at 600s cadence the count path would require ≥30 min of sustained flip
# before triggering, which effectively disables the trigger for fast-moving
# events. The duration gate bounds the worst case to ~1 cycle after flip
# (≈10 min at 600s cadence, ≈2 min at 60s cadence — both unchanged or better
# than the old count-only behavior).
SUSTAINED_CHECKS = 3
SUSTAINED_DURATION_S = 120

# Ranging regime dampening (2026-04-22): when a ticker's regime is "ranging",
# require a minimum consensus confidence before triggering Layer 2. In ranging
# markets, consensus oscillates between HOLD and weak BUY/SELL, producing 20+
# Layer 2 invocations per day that all return HOLD — wasting compute and token
# budget. Setting this to 0.0 disables dampening without code change.
RANGING_CONSENSUS_MIN_CONFIDENCE = 0.35

# Startup grace period — after a restart, the first loop iteration updates the
# baseline without triggering Layer 2. This prevents spurious T3 full reviews
# every time the loop is restarted for a code update.
_GRACE_PERIOD_KEY = "last_loop_pid"  # stored in trigger_state.json
---

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python - <<'PY'
import pkgutil
mods=['ruff','pyflakes','flake8','mypy']
for m in mods:
    print(m, pkgutil.find_loader(m))
PY" in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-08T15:29:08.009301Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m python - <[36;1m<[0m'PY'[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m           ~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mMissing file specification after redirection operator.[0m

 exited 1 in 629ms:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m python - <[36;1m<[0m'PY'[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m           ~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mMissing file specification after redirection operator.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import pkgutil
mods=['ruff','pyflakes','flake8','mypy']
for m in mods:
    print(m, bool(pkgutil.find_loader(m)))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 627ms:
ruff False
pyflakes False
flake8 False
mypy False
<stdin>:4: DeprecationWarning: 'pkgutil.find_loader' is deprecated and slated for removal in Python 3.14; use importlib.util.find_spec() instead

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, pathlib
root = pathlib.Path('portfolio')
files = sorted(root.glob('*.py'))
internal = {p.stem for p in files}
for path in files:
    tree=ast.parse(path.read_text(encoding='utf-8'))
    missing=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('portfolio.'):
            mod=node.module.split('.',1)[1].split('.')[0]
            if mod not in internal:
                missing.append((node.lineno,node.module))
    if missing:
        print(path)
        for lineno,mod in missing:
            print(' ',lineno,mod)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 978ms:
portfolio\agent_invocation.py
  14 portfolio.api_utils
  16 portfolio.file_utils
  17 portfolio.message_store
  18 portfolio.telegram_notifications
  155 portfolio.file_utils
  164 portfolio.file_utils
  267 portfolio.file_utils
  605 portfolio.journal
  644 portfolio.risk_management
  719 portfolio.trade_guards
  1030 portfolio.file_utils
  1065 portfolio.file_utils
  1066 portfolio.trade_guards
  876 portfolio.file_utils
  896 portfolio.file_utils
  1041 portfolio.file_utils
portfolio\autonomous.py
  18 portfolio.file_utils
  19 portfolio.message_store
  20 portfolio.notification_text
  27 portfolio.portfolio_mgr
  28 portfolio.telegram_notifications
  825 portfolio.file_utils
portfolio\claude_gate.py
  36 portfolio.file_utils
portfolio\health.py
  9 portfolio.file_utils
  356 portfolio.data_collector
portfolio\loop_contract.py
  18 portfolio.file_utils
  783 portfolio.accuracy_stats
  740 portfolio.accuracy_degradation
  1822 portfolio.message_store
portfolio\main.py
  31 portfolio.file_utils
  126 portfolio.api_utils
  129 portfolio.data_collector
  145 portfolio.digest
  148 portfolio.fx_rates
  149 portfolio.http_retry
  152 portfolio.indicators
  171 portfolio.portfolio_mgr
  190 portfolio.shared_state
  213 portfolio.signal_engine
  229 portfolio.telegram_notifications
  358 portfolio.file_utils
  1130 portfolio.logging_config
  285 portfolio.market_health
  290 portfolio.daily_digest
  300 portfolio.accuracy_degradation
  314 portfolio.accuracy_stats
  319 portfolio.message_throttle
  324 portfolio.alpha_vantage
  330 portfolio.local_llm_report
  346 portfolio.metals_precompute
  352 portfolio.oil_precompute
  370 portfolio.fin_evolve
  376 portfolio.crypto_scheduler
  382 portfolio.file_utils
  383 portfolio.signal_postmortem
  672 portfolio.llm_batch
  673 portfolio.llm_batch
  674 portfolio.shared_state
  689 portfolio.sentiment
  746 portfolio.signal_engine
  787 portfolio.cumulative_tracker
  887 portfolio.iskbets
  895 portfolio.avanza_orders
  956 portfolio.risk_management
  1032 portfolio.message_store
  1170 portfolio.alpha_vantage
  1179 portfolio.iskbets
  1180 portfolio.telegram_poller
  1266 portfolio.outcome_tracker
  408 portfolio.log_rotation
  720 portfolio.signal_engine
  721 portfolio.signal_engine
  771 portfolio.message_store
  825 portfolio.outcome_tracker
  874 portfolio.bigbet
  1272 portfolio.forecast_accuracy
  1280 portfolio.decision_outcome_tracker
  1288 portfolio.signal_decay_alert
  1297 portfolio.accuracy_stats
  390 portfolio.file_utils
  633 portfolio.signal_engine
  634 portfolio.signal_engine
  1020 portfolio.message_store
  1300 portfolio.forecast_accuracy
  935 portfolio.telegram_notifications
  947 portfolio.telegram_notifications
  1155 portfolio.message_store
  1303 portfolio.local_llm_report
  1308 portfolio.local_llm_report
  1316 portfolio.prophecy
  1320 portfolio.forecast_accuracy
  1326 portfolio.data_refresh
  1329 portfolio.ml_trainer
  1341 portfolio.analyze
  1347 portfolio.focus_analysis
  1357 portfolio.analyze
  1365 portfolio.api_utils
  1366 portfolio.price_targets
  1381 portfolio.avanza_client
portfolio\perception_gate.py
  18 portfolio.api_utils
  19 portfolio.file_utils
portfolio\reflection.py
  19 portfolio.file_utils
  138 portfolio.api_utils
portfolio\reporting.py
  9 portfolio.file_utils
  10 portfolio.indicators
  11 portfolio.portfolio_mgr
  12 portfolio.shared_state
  13 portfolio.signal_registry
  112 portfolio.api_utils
  846 portfolio.shared_state
  123 portfolio.signal_engine
  223 portfolio.macro_context
  253 portfolio.market_health
  263 portfolio.earnings_calendar
  273 portfolio.exposure_coach
  274 portfolio.market_health
  290 portfolio.signal_postmortem
  299 portfolio.accuracy_stats
  339 portfolio.accuracy_stats
  361 portfolio.accuracy_stats
  379 portfolio.alpha_vantage
  389 portfolio.onchain_data
  405 portfolio.futures_data
  453 portfolio.trade_guards
  463 portfolio.risk_management
  514 portfolio.price_targets
  537 portfolio.equity_curve
  553 portfolio.avanza_tracker
  576 portfolio.forecast_accuracy
  656 portfolio.cumulative_tracker
  666 portfolio.warrant_portfolio
  676 portfolio.cost_model
  677 portfolio.exit_optimizer
  722 portfolio.prophecy
  818 portfolio.signal_state_since
  237 portfolio.data_collector
  475 portfolio.monte_carlo
  476 portfolio.monte_carlo_risk
  503 portfolio.price_targets
  566 portfolio.ticker_accuracy
  591 portfolio.signals.forecast
portfolio\trigger.py
  25 portfolio.file_utils

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\session_calendar.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 565ms:
"""Session calendar — instrument-specific trading hours and session state.

Provides remaining-session time, session boundaries, and session mismatch
detection for the exit optimizer.

Usage:
    from portfolio.session_calendar import get_session_info
    info = get_session_info("warrant", underlying="XAG-USD")
    # info.remaining_minutes, info.session_end, info.is_extended, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from portfolio.market_timing import _is_us_dst


@dataclass(frozen=True)
class SessionInfo:
    """Trading session state for an instrument.

    Attributes:
        session_end: Absolute datetime (UTC) of normal session close.
        extended_end: Absolute datetime (UTC) of extended session close, if applicable.
        remaining_minutes: Minutes until effective close (extended if available).
        is_open: Whether the instrument is currently tradeable.
        is_extended: Whether we're in the extended (evening) session.
        underlying_open: Whether the underlying's primary market is open (for warrants).
        phase: Human-readable phase: "open", "extended", "pre_open", "closed".
    """
    session_end: datetime
    extended_end: datetime | None
    remaining_minutes: float
    is_open: bool
    is_extended: bool
    underlying_open: bool
    phase: str


# ---------------------------------------------------------------------------
# Session definitions (times in UTC)
# ---------------------------------------------------------------------------

# Avanza commodity warrants: 08:15-21:55 CET = 07:15-20:55 UTC (winter)
# CET = UTC+1 (winter), CEST = UTC+2 (summer)
# We handle DST for EU sessions too.

def _eu_dst(dt: datetime) -> bool:
    """Check if datetime falls in EU Central European Summer Time (CEST).

    EU DST: last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC.
    """
    year = dt.year

    # Last Sunday of March
    mar31 = datetime(year, 3, 31, tzinfo=UTC)
    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
    dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)

    # Last Sunday of October
    oct31 = datetime(year, 10, 31, tzinfo=UTC)
    last_sun_oct = 31 - (oct31.weekday() + 1) % 7
    dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)

    return dst_start <= dt < dst_end


def _cet_offset(dt: datetime) -> int:
    """Return CET/CEST offset from UTC in hours (1 or 2)."""
    return 2 if _eu_dst(dt) else 1


def _cet_to_utc(hour: int, minute: int, dt: datetime) -> time:
    """Convert CET time to UTC time object, adjusted for DST on given date."""
    offset = _cet_offset(dt)
    utc_hour = (hour - offset) % 24
    return time(utc_hour, minute)


def _make_session_end(now: datetime, cet_hour: int, cet_minute: int) -> datetime:
    """Create a UTC datetime for today's session end from CET time."""
    offset = _cet_offset(now)
    utc_hour = cet_hour - offset
    end = now.replace(hour=utc_hour, minute=cet_minute, second=0, microsecond=0)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return end


# Session specs: (open_cet, close_cet) as (hour, minute) tuples
SESSIONS = {
    "warrant": {
        "open_cet": (8, 15),
        "close_cet": (21, 55),
        "has_extended": False,  # Already includes evening trading
        "description": "Avanza commodity warrants",
    },
    "stock_se": {
        "open_cet": (9, 0),
        "close_cet": (17, 25),
        "has_extended": False,
        "description": "Nasdaq Stockholm equities",
    },
    "crypto": {
        "open_cet": (0, 0),
        "close_cet": (23, 59),
        "has_extended": False,
        "description": "Crypto 24/7",
    },
}


def get_session_info(instrument_type: str,
                     underlying: str | None = None,
                     now: datetime | None = None) -> SessionInfo:
    """Get current session state for an instrument.

    Args:
        instrument_type: "warrant", "stock_se", "stock_us", "crypto".
        underlying: Underlying ticker for warrants (e.g., "XAG-USD").
        now: Current UTC time. Defaults to now.

    Returns:
        SessionInfo with remaining time, phase, and session boundaries.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    # Crypto: always open (24/7)
    if instrument_type == "crypto":
        # Use midnight as "session end" — effectively infinite session
        end = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return SessionInfo(
            session_end=end,
            extended_end=None,
            remaining_minutes=(end - now).total_seconds() / 60,
            is_open=True,
            is_extended=False,
            underlying_open=True,
            phase="open",
        )

    # US stocks: NYSE hours with DST
    if instrument_type == "stock_us":
        us_dst = _is_us_dst(now)
        open_utc = 13 if us_dst else 14   # 09:30 ET
        close_utc = 20 if us_dst else 21  # 16:00 ET

        session_end = now.replace(hour=close_utc, minute=0, second=0, microsecond=0)
        is_open = (now.weekday() < 5 and
                   now.replace(hour=open_utc, minute=30, second=0) <= now < session_end)

        remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
        phase = "open" if is_open else "closed"

        # Check if underlying is open (for warrants referencing US stocks)
        underlying_open = is_open

        return SessionInfo(
            session_end=session_end,
            extended_end=None,
            remaining_minutes=remaining,
            is_open=is_open,
            is_extended=False,
            underlying_open=underlying_open,
            phase=phase,
        )

    # EU-based instruments (warrants, Stockholm stocks)
    spec = SESSIONS.get(instrument_type, SESSIONS["warrant"])
    oh, om = spec["open_cet"]
    ch, cm = spec["close_cet"]

    session_end = _make_session_end(now, ch, cm)
    session_open = _make_session_end(now, oh, om)

    is_weekday = now.weekday() < 5
    is_open = is_weekday and session_open <= now < session_end

    remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
    phase = "open" if is_open else "closed"

    # For warrants, check if underlying's primary market is open
    underlying_open = True  # Metals trade ~24h
    if underlying and not underlying.endswith("-USD"):
        # US stock underlying — check US market hours
        us_info = get_session_info("stock_us", now=now)
        underlying_open = us_info.is_open

    return SessionInfo(
        session_end=session_end,
        extended_end=None,
        remaining_minutes=remaining,
        is_open=is_open,
        is_extended=False,
        underlying_open=underlying_open,
        phase=phase,
    )


def remaining_session_minutes(instrument_type: str = "warrant",
                              now: datetime | None = None) -> float:
    """Shortcut: get remaining minutes for an instrument's session."""
    info = get_session_info(instrument_type, now=now)
    return info.remaining_minutes

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | Select-String -Pattern 'session_calendar|get_session_info|remaining_session_minutes'" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 620ms:

portfolio\reporting.py:678:        from portfolio.[7msession_calendar[0m import get_session_info
portfolio\reporting.py:690:                sess = [7mget_session_info[0m("warrant", underlying=underlying)
portfolio\session_calendar.py:7:    from portfolio.[7msession_calendar[0m import get_session_info
portfolio\session_calendar.py:8:    info = [7mget_session_info[0m("warrant", underlying="XAG-USD")
portfolio\session_calendar.py:115:def [7mget_session_info[0m(instrument_type: str,
portfolio\session_calendar.py:193:        us_info = [7mget_session_info[0m("stock_us", now=now)
portfolio\session_calendar.py:207:def [7mremaining_session_minutes[0m(instrument_type: str = "warrant",
portfolio\session_calendar.py:210:    info = [7mget_session_info[0m(instrument_type, now=now)


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\reporting.py | Select-Object -Index (670..700)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 619ms:
        logger.warning("[reporting] warrant_portfolio failed", exc_info=True)
        _module_warnings.append("warrant_portfolio")

    # Exit optimizer plans for warrant positions
    try:
        from portfolio.cost_model import get_cost_model
        from portfolio.exit_optimizer import MarketSnapshot, Position, compute_exit_plan
        from portfolio.session_calendar import get_session_info
        warrant_state_path = DATA_DIR / "portfolio_state_warrants.json"
        warrant_state = load_json(warrant_state_path)
        if warrant_state is not None:
            exit_plans = {}
            for wk, wpos in warrant_state.get("positions", {}).items():
                if not wpos.get("active") or wpos.get("units", 0) <= 0:
                    continue
                underlying = wpos.get("underlying", "")
                und_price = prices_usd.get(underlying, 0)
                if und_price <= 0:
                    continue
                sess = get_session_info("warrant", underlying=underlying)
                if not sess.is_open or sess.remaining_minutes < 2:
                    continue
                position = Position(
                    symbol=underlying,
                    qty=wpos["units"],
                    entry_price_sek=wpos.get("entry_price", 0),
                    entry_underlying_usd=wpos.get("entry_underlying", und_price),
                    entry_ts=datetime.fromisoformat(wpos["entry_ts"]) if wpos.get("entry_ts") else datetime.now(UTC),
                    instrument_type="warrant",
                    leverage=wpos.get("leverage", 5.0),
                    financing_level=wpos.get("financing_level"),

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\trigger.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 578ms:
"""Smart trigger system — detects meaningful market changes to reduce noise.

Layer 1 runs on a 10-minute cadence during every market state (see
``portfolio/market_timing.py:INTERVAL_MARKET_OPEN``). Layer 2 is invoked when:
- Signal consensus: any ticker NEWLY reaches BUY or SELL from HOLD
- Signal flip sustained for SUSTAINED_CHECKS consecutive cycles (see below)
- Price moved >2% since last trigger
- Fear & Greed crossed extreme threshold (20 or 80)
- Sentiment reversal: sustained for SUSTAINED_CHECKS cycles (filters oscillation)
- Post-trade reassessment: after a BUY/SELL trade

No periodic cooldown — Layer 2 is only invoked when Layer 1 detects a
meaningful change. The Tier 3 periodic full review (every 2h market / 4h
off-hours) provides the "heartbeat" via classify_tier(), but only when
another trigger has already fired.
"""

import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.trigger")

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "data" / "trigger_state.json"
PORTFOLIO_FILE = BASE_DIR / "data" / "portfolio_state.json"
PORTFOLIO_BOLD_FILE = BASE_DIR / "data" / "portfolio_state_bold.json"

PRICE_THRESHOLD = 0.02  # 2% move
FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries
# A signal flip triggers Layer 2 when EITHER of these holds:
#   - SUSTAINED_CHECKS consecutive cycles show the new action, OR
#   - SUSTAINED_DURATION_S seconds of wall-clock time have elapsed since
#     the flip first appeared.
# The count path is the original behavior (unchanged at the 60s cadence).
# The duration path is new (added 2026-04-09 with the cadence bump to 600s);
# at 600s cadence the count path would require ≥30 min of sustained flip
# before triggering, which effectively disables the trigger for fast-moving
# events. The duration gate bounds the worst case to ~1 cycle after flip
# (≈10 min at 600s cadence, ≈2 min at 60s cadence — both unchanged or better
# than the old count-only behavior).
SUSTAINED_CHECKS = 3
SUSTAINED_DURATION_S = 120

# Ranging regime dampening (2026-04-22): when a ticker's regime is "ranging",
# require a minimum consensus confidence before triggering Layer 2. In ranging
# markets, consensus oscillates between HOLD and weak BUY/SELL, producing 20+
# Layer 2 invocations per day that all return HOLD — wasting compute and token
# budget. Setting this to 0.0 disables dampening without code change.
RANGING_CONSENSUS_MIN_CONFIDENCE = 0.35

# Startup grace period — after a restart, the first loop iteration updates the
# baseline without triggering Layer 2. This prevents spurious T3 full reviews
# every time the loop is restarted for a code update.
_GRACE_PERIOD_KEY = "last_loop_pid"  # stored in trigger_state.json
_startup_grace_active = True  # True until first check_triggers call completes


def _update_sustained(
    state_dict: dict, key: str, value, now_ts: float
) -> tuple[bool, bool]:
    """Update sustained-debounce state for a key and return gate results.

    Shared by signal flip (section 2) and sentiment reversal (section 5).
    Increments count if value unchanged, resets if changed. Returns
    (count_ok, duration_ok) indicating whether either debounce gate passed.

    Duration tracking uses time.monotonic() internally to avoid NTP-jump
    false negatives. On process restart, monotonic origin resets and the
    duration gate conservatively starts fresh (correct behavior — a
    restart already resets the sustained counter).
    """
    mono_now = time.monotonic()
    prev = state_dict.get(key, {})
    if prev.get("value") == value:
        state_dict[key] = {
            "value": value,
            "count": prev["count"] + 1,
            "_mono_start": prev.get("_mono_start", mono_now),
        }
    else:
        state_dict[key] = {
            "value": value,
            "count": 1,
            "_mono_start": mono_now,
        }
    entry = state_dict[key]
    count_ok = entry["count"] >= SUSTAINED_CHECKS
    duration_ok = (mono_now - entry["_mono_start"]) >= SUSTAINED_DURATION_S
    return count_ok, duration_ok


def _today_str():
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _load_state():
    return load_json(STATE_FILE, default={})


def _save_state(state):
    # Prune triggered_consensus entries for tickers not in current signals
    # to prevent unbounded growth when tickers are removed from tracking
    tc = state.get("triggered_consensus", {})
    current_tickers = state.get("_current_tickers")
    if current_tickers is not None:
        removed = {k for k in tc if k not in current_tickers}
        if removed:
            logger.info("trigger: pruning %d stale ticker(s) from baseline: %s", len(removed), ", ".join(sorted(removed)))
        pruned = {k: v for k, v in tc.items() if k in current_tickers}
        state["triggered_consensus"] = pruned
    state.pop("_current_tickers", None)  # don't persist internal field
    atomic_write_json(STATE_FILE, state)


def _check_recent_trade(state):
    """Check if Layer 2 executed a trade since our last trigger.

    Returns True if a recent trade was detected.
    """
    last_checked_tx = state.get("last_checked_tx_count", {})

    trade_detected = False
    new_tx_counts = {}

    for label, pf_file in [("patient", PORTFOLIO_FILE), ("bold", PORTFOLIO_BOLD_FILE)]:
        try:
            pf = load_json(pf_file, default=None)
            if pf is None:
                continue
            txs = pf.get("transactions", [])
            current_count = len(txs)
            prev_count = last_checked_tx.get(label, current_count)
            new_tx_counts[label] = current_count

            if current_count > prev_count:
                trade_detected = True
        except (KeyError, AttributeError) as exc:
            logger.warning("Failed to parse portfolio file %s: %s", pf_file, exc)

    if new_tx_counts:
        state["last_checked_tx_count"] = new_tx_counts

    return trade_detected


def check_triggers(signals, prices_usd, fear_greeds, sentiments):
    global _startup_grace_active
    state = _load_state()
    state["_current_tickers"] = set(signals.keys())  # for pruning in _save_state

    # Startup grace period: on the first iteration after a restart, update the
    # baseline (prices, signals, consensus) WITHOUT triggering Layer 2.
    # This lets the loop restart for code updates without spurious T3 reviews.
    current_pid = os.getpid()
    saved_pid = state.get(_GRACE_PERIOD_KEY)
    if _startup_grace_active and saved_pid != current_pid:
        import logging
        _logger = logging.getLogger("portfolio.trigger")
        _logger.info(
            "Startup grace period: updating baseline without triggering "
            "(pid %s -> %s)", saved_pid, current_pid,
        )
        state[_GRACE_PERIOD_KEY] = current_pid
        # Update baselines so next iteration compares from NOW
        state["last"] = {
            "signals": {
                t: {"action": s["action"], "confidence": s["confidence"]}
                for t, s in signals.items()
            },
            "prices": dict(prices_usd),
            "fear_greeds": {
                t: fg if isinstance(fg, dict) else {} for t, fg in fear_greeds.items()
            },
            "sentiments": dict(sentiments),
            "time": time.time(),
        }
        # Update triggered_consensus baseline to current state
        tc = state.get("triggered_consensus", {})
        for ticker, sig in signals.items():
            tc[ticker] = sig["action"]
        state["triggered_consensus"] = tc
        state["today_date"] = _today_str()
        _startup_grace_active = False
        _save_state(state)
        return False, []

    _startup_grace_active = False
    prev = state.get("last", {})
    sustained = state.get("sustained_counts", {})
    reasons = []

    # 0. Trade reset — if Layer 2 made a trade, trigger reassessment
    if _check_recent_trade(state):
        state["last_trigger_time"] = 0
        reasons.append("post-trade reassessment")

    # 1. Signal consensus — trigger ONLY when a ticker first reaches BUY/SELL
    #    from HOLD. BUY↔SELL direction flips are handled by the sustained flip
    #    trigger (#2). Uses persistent triggered_consensus that is NOT wiped
    #    when unrelated triggers (sentiment, etc.) fire.
    #
    #    Ranging regime dampening (2026-04-22): in ranging regime, low-confidence
    #    consensus crossings are noise — require RANGING_CONSENSUS_MIN_CONFIDENCE
    #    to actually fire the trigger. Prevents 20+ HOLD invocations per day.
    triggered_consensus = state.get("triggered_consensus", {})
    for ticker, sig in signals.items():
        action = sig["action"]
        last_tc = triggered_consensus.get(ticker, "HOLD")
        if action in ("BUY", "SELL") and last_tc == "HOLD":
            conf = sig.get("confidence", 0)
            # Ranging regime dampening: skip low-confidence consensus triggers
            ticker_regime = (sig.get("extra") or {}).get("_regime", "unknown")
            if (
                ticker_regime == "ranging"
                and RANGING_CONSENSUS_MIN_CONFIDENCE > 0
                and conf < RANGING_CONSENSUS_MIN_CONFIDENCE
            ):
                logger.info(
                    "Ranging dampening: %s consensus %s (%.0f%%) suppressed "
                    "(min %.0f%%)",
                    ticker, action, conf * 100,
                    RANGING_CONSENSUS_MIN_CONFIDENCE * 100,
                )
                # Still update baseline so we don't re-trigger next cycle
                triggered_consensus[ticker] = action
                continue
            # New consensus from HOLD — trigger
            reasons.append(f"{ticker} consensus {action} ({conf:.0%})")
            triggered_consensus[ticker] = action
        elif action == "HOLD" and last_tc != "HOLD":
            # Consensus cleared — reset so next BUY/SELL is "new"
            triggered_consensus[ticker] = "HOLD"
        elif action in ("BUY", "SELL") and action != last_tc:
            # Direction flip (BUY↔SELL) — update baseline silently,
            # let sustained flip trigger (#2) handle it
            triggered_consensus[ticker] = action
    state["triggered_consensus"] = triggered_consensus

    # 2. Signal flip — triggers when the new action has been seen for
    #    SUSTAINED_CHECKS consecutive cycles OR for SUSTAINED_DURATION_S
    #    wall-clock seconds, whichever comes first. The duration gate was
    #    added 2026-04-09 so the trigger still fires within ~1 cycle at
    #    long cadences (e.g. 600s); at the historical 60s cadence the count
    #    gate still dominates and behavior is unchanged.
    prev_triggered = prev.get("signals", {})
    _flip_now_ts = time.time()
    for ticker, sig in signals.items():
        current_action = sig["action"]
        count_ok, duration_ok = _update_sustained(
            sustained, ticker, current_action, _flip_now_ts,
        )

        triggered_action = prev_triggered.get(ticker, {}).get("action")
        if triggered_action and current_action != triggered_action and (count_ok or duration_ok):
            reasons.append(
                f"{ticker} flipped {triggered_action}->{current_action} (sustained)"
            )

    # 3. Price move >2% since last trigger
    prev_prices = prev.get("prices", {})
    for ticker, price in prices_usd.items():
        old_price = prev_prices.get(ticker)
        if old_price and old_price > 0:
            pct = abs(price - old_price) / old_price
            if pct >= PRICE_THRESHOLD:
                direction = "up" if price > old_price else "down"
                reasons.append(f"{ticker} moved {pct:.1%} {direction}")

    # 4. Fear & Greed crossed threshold
    prev_fg = prev.get("fear_greeds", {})
    for ticker, fg in fear_greeds.items():
        val = fg.get("value", 50) if isinstance(fg, dict) else 50
        old_val = (
            prev_fg.get(ticker, {}).get("value", 50)
            if isinstance(prev_fg.get(ticker), dict)
            else 50
        )
        for threshold in FG_THRESHOLDS:
            if (old_val > threshold) != (val > threshold):
                reasons.append(f"F&G crossed {threshold} ({old_val}->{val})")
                break

    # 5. Sentiment reversal — same OR-debounce as section 2.
    sustained_sent = state.get("sustained_sentiment", {})
    stable_sent = state.get("stable_sentiment", {})
    _sent_now_ts = time.time()
    for ticker, sent in sentiments.items():
        count_ok, duration_ok = _update_sustained(
            sustained_sent, ticker, sent, _sent_now_ts,
        )
        if count_ok or duration_ok:
            last_stable = stable_sent.get(ticker)
            if (
                last_stable
                and last_stable != sent
                and sent != "neutral"
                and last_stable != "neutral"
            ):
                reasons.append(
                    f"{ticker} sentiment {last_stable}->{sent} (sustained)"
                )
            stable_sent[ticker] = sent
    state["sustained_sentiment"] = sustained_sent
    state["stable_sentiment"] = stable_sent

    triggered = len(reasons) > 0

    if triggered:
        state["last_trigger_time"] = time.time()
        state["last"] = {
            "signals": {
                t: {"action": s["action"], "confidence": s["confidence"]}
                for t, s in signals.items()
            },
            "prices": dict(prices_usd),
            "fear_greeds": {
                t: fg if isinstance(fg, dict) else {} for t, fg in fear_greeds.items()
            },
            "sentiments": dict(sentiments),
            "time": time.time(),
        }
        # C4/NEW-2: only update last_trigger_date when a real trigger fires, so that
        # classify_tier() can correctly detect the first real trigger of the day.
        state["last_trigger_date"] = _today_str()

    # Track today_date for other purposes
    state["today_date"] = _today_str()

    state["sustained_counts"] = sustained
    _save_state(state)

    return triggered, reasons


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

# Full review interval: 4h during market hours, 4h off-hours (T1 only)
_FULL_REVIEW_MARKET_HOURS = 4
_FULL_REVIEW_OFF_HOURS = 4  # Off-hours caps at T1, not T3

# Option P (2026-04-17): confidence-aware tier downshift.
# When every reason in a T2 trigger is either a low-conviction consensus
# crossing (<TIER_DOWNSHIFT_CONFIDENCE) or a fade flip (*->HOLD sustained),
# downshift T2 -> T1 to save Claude token budget. T3 triggers (first-of-day,
# F&G extreme, periodic full review) are NEVER downshifted. Sustained
# direction flips (BUY<->SELL) and non-consensus triggers (post-trade, price
# move, sentiment) block downshift. Setting this to 0.0 disables downshift
# without code change.
TIER_DOWNSHIFT_CONFIDENCE = 0.40

# Precompiled patterns for downshift eligibility analysis on reason strings
# produced by check_triggers(). Reason shape stays stable across releases;
# if the format ever changes, these miss -> downshift fails open (tier
# stays T2, safe over-invocation rather than under-invocation).
#
# Word boundaries (\b) on "consensus" and "flipped" prevent substring
# collisions — e.g. a hypothetical future reason containing "nonconsensus"
# or "preflipped" would NOT accidentally match and trigger a downshift.
# Current check_triggers has no such reasons, but anchoring is cheap
# insurance against future regressions. Added 2026-04-17 after an
# adversarial self-review surfaced the issue.
_CONSENSUS_CONF_RE = re.compile(r'\bconsensus (?:BUY|SELL) \((\d+)%\)')
_FADE_FLIP_RE = re.compile(r'\bflipped (?:BUY|SELL)->HOLD \(sustained\)')


def _reason_is_downshiftable(reason: str, threshold: float) -> bool:
    """Return True if this reason is low-conviction enough to allow T2->T1.

    A reason qualifies if it is either:
      - A consensus crossing with confidence < threshold, or
      - A fade flip (*->HOLD sustained).

    Any other reason type (direction flip, post-trade, price move, F&G,
    sentiment, startup) returns False and blocks downshift for the whole
    reason list.
    """
    m = _CONSENSUS_CONF_RE.search(reason)
    if m:
        conf_pct = int(m.group(1))
        return conf_pct < threshold * 100
    return bool(_FADE_FLIP_RE.search(reason))


def _should_downshift_to_t1(reasons, threshold: float | None = None) -> bool:
    """Decide whether a T2 tier should be downshifted to T1.

    Returns True only when every reason is either a low-conviction consensus
    crossing or a fade flip — i.e. all reasons are individually downshiftable.
    A single high-conviction or non-consensus reason blocks downshift.

    Empty reason list returns False (no downshift). Called only after
    classify_tier() has already chosen T2 — T1 and T3 are never affected.

    threshold=None (default) looks up TIER_DOWNSHIFT_CONFIDENCE at call time,
    allowing runtime overrides via mock.patch or module-attribute reassignment
    (the module-level constant is the single config knob). Passing an explicit
    float overrides for testing.
    """
    if not reasons:
        return False
    effective = TIER_DOWNSHIFT_CONFIDENCE if threshold is None else threshold
    return all(_reason_is_downshiftable(r, effective) for r in reasons)


def classify_tier(reasons, state=None):
    """Classify trigger reasons into invocation tier (1=quick, 2=signal, 3=full).

    Tier 3 (Full Review): periodic review, F&G extreme, first of day.
    Tier 2 (Signal Analysis): new consensus, price moves, post-trade, signal flips.
    Tier 1 (Quick Check): sentiment noise, repeated triggers.

    M10/NEW-4: pass state=<dict> to avoid a redundant disk read when the caller
    already has the trigger state loaded. Falls back to loading from file if None.
    """
    if state is None:
        state = _load_state()

    # Tier 3: periodic full review
    last_full = state.get("last_full_review_time", 0)
    hours_since = (time.time() - last_full) / 3600

    now_utc = datetime.now(UTC)
    from portfolio.market_timing import _eu_market_open_hour_utc, _market_close_hour_utc
    close_hour = _market_close_hour_utc(now_utc)
    eu_open = _eu_market_open_hour_utc(now_utc)
    market_open = now_utc.weekday() < 5 and eu_open <= now_utc.hour < close_hour

    # C4/NEW-2: first-of-day T3 check must precede the off-hours periodic cap.
    # An off-hours trigger 4+ hours after the last full review would otherwise
    # return T1 early (line below), skipping the first-of-day T3 entirely.
    if state.get("last_trigger_date") != _today_str():
        return 3  # first real trigger of the day

    if any("F&G crossed" in r for r in reasons):
        return 3

    if market_open and hours_since >= _FULL_REVIEW_MARKET_HOURS:
        return 3
    if not market_open and hours_since >= _FULL_REVIEW_OFF_HOURS:
        return 1  # T1 quick check only — save T3 budget for market hours

    # Tier 2: new actionable signals
    tier2_patterns = ["consensus", "moved", "post-trade", "flipped"]
    if any(p in r for r in reasons for p in tier2_patterns):
        # Option P (2026-04-17): downshift T2 -> T1 when every reason is
        # low-conviction (consensus <40% confidence or *->HOLD fade flip).
        # Preserves trigger firing + signal/accuracy data; only cuts Claude
        # analysis depth on signals that reliably return HOLD anyway.
        if _should_downshift_to_t1(reasons):
            return 1
        return 2

    # Tier 1: cooldowns, sentiment noise, repeated triggers
    return 1


def update_tier_state(tier, state=None):
    """Update trigger state after a tier classification.

    Called by the main loop after classify_tier() to persist tier-specific state.
    M10/NEW-4: accepts an optional state dict to avoid re-reading trigger_state.json.
    """
    if state is None:
        state = _load_state()
    if tier == 3:
        state["last_full_review_time"] = time.time()
    _save_state(state)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\health.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 617ms:
"""Health monitoring for the finance-analyzer Layer 1 loop."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, last_jsonl_entry, load_json, load_jsonl_tail

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEALTH_FILE = DATA_DIR / "health_state.json"

# C10/H17: Protect all read-modify-write sequences in health.py.
_health_lock = threading.Lock()


def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
                  last_trigger_reason: str = None, error: str = None):
    """Called at end of each Layer 1 cycle to update health state."""
    with _health_lock:
        state = load_health()
        state["last_heartbeat"] = datetime.now(UTC).isoformat()
        state["cycle_count"] = cycle_count
        state["signals_ok"] = signals_ok
        state["signals_failed"] = signals_failed
        state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
        if last_trigger_reason:
            state["last_trigger_reason"] = last_trigger_reason
            state["last_trigger_time"] = datetime.now(UTC).isoformat()
            # Cache the invocation timestamp so check_agent_silence() can avoid
            # re-parsing invocations.jsonl on every call.
            state["last_invocation_ts"] = state["last_trigger_time"]
        if error:
            state["errors"] = state.get("errors", [])[-19:] + [
                {"ts": datetime.now(UTC).isoformat(), "error": error}
            ]
            state["error_count"] = state.get("error_count", 0) + 1
        atomic_write_json(HEALTH_FILE, state)


def load_health() -> dict:
    """Load current health state. Returns defaults if missing or corrupt."""
    state = load_json(HEALTH_FILE)
    if state is not None:
        return state
    return {"start_time": time.time(), "cycle_count": 0, "error_count": 0, "errors": []}


def reset_session_start():
    """Reset start_time to current time — call at loop startup.

    Prevents uptime_seconds from inheriting a stale start_time
    from a previous session's health_state.json.
    """
    with _health_lock:
        state = load_health()
        state["start_time"] = time.time()
        atomic_write_json(HEALTH_FILE, state)


def heartbeat() -> None:
    """Touch only the last_heartbeat timestamp.

    Called as a one-shot or periodically from a keepalive thread while
    long-blocking work is in flight. Layer 2 invocation can block up to
    600s (T2) or 900s (T3), but update_health() only runs at end-of-cycle
    (AFTER Layer 2 returns). Without periodic touches the dashboard
    /api/health endpoint flips fresh→stale every triggering cycle, which
    is misleading: the loop is alive, just waiting on the subprocess.

    Other state (cycle_count, signals_ok/failed, errors) is left untouched
    — those reflect the previously-completed cycle, still the most recent
    ground truth. update_health() at end-of-cycle overwrites them with
    this cycle's results.

    Failure-tolerant: callers wrap in try/except since this is a "nice to
    have" hint and must never crash the loop. The atomic write means a
    partial run leaves the prior file intact.
    """
    with _health_lock:
        state = load_health()
        state["last_heartbeat"] = datetime.now(UTC).isoformat()
        atomic_write_json(HEALTH_FILE, state)


# Keepalive default interval. The dashboard's stale gate fires at 300s
# (check_staleness max_age_seconds=300), so 60s gives 5x headroom while
# leaving plenty of margin for missed ticks (e.g. GIL contention, slow
# atomic_write_json on a heavily-loaded disk).
_HEARTBEAT_KEEPALIVE_INTERVAL_S = 60.0


class heartbeat_keepalive:  # noqa: N801 — context-manager naming convention
    """Context manager that ticks heartbeat() every interval seconds.

    Wraps long-blocking work (Layer 2 T2/T3 subprocess, autonomous decision
    paths, anything that can block longer than the 300s stale threshold)
    so /api/health stays fresh for the duration. Background daemon thread
    is auto-stopped on context exit; a 2s join timeout prevents shutdown
    deadlocks (the thread only sleeps + writes, both bounded).

    Usage:
        with heartbeat_keepalive():
            result = invoke_agent(reasons_list, tier=tier)

    The first beat is synchronous (so a fast-returning subprocess gets at
    least one heartbeat even if it finishes before the first interval).
    Subsequent beats run on the daemon thread until __exit__.

    Failure-tolerant by design: tick exceptions are swallowed at WARNING
    level — a Disk-full or permission-denied during keepalive must never
    abort an in-flight Layer 2 trade decision.
    """

    def __init__(self, interval: float = _HEARTBEAT_KEEPALIVE_INTERVAL_S) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "heartbeat_keepalive":
        # Synchronous first beat — covers the case where the wrapped call
        # returns before the keepalive thread's first tick.
        try:
            heartbeat()
        except Exception:
            logger.warning("heartbeat_keepalive initial beat failed", exc_info=True)

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="heartbeat-keepalive",
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Event.wait returns True when set (stop signaled), False on timeout.
        # So we tick on each timeout and exit on the first True.
        while not self._stop.wait(self._interval):
            try:
                heartbeat()
            except Exception:
                logger.warning("heartbeat_keepalive tick failed", exc_info=True)


def check_staleness(max_age_seconds: int = 300) -> tuple:
    """Check if the loop heartbeat is stale.
    Returns (is_stale: bool, age_seconds: float, state: dict)
    """
    state = load_health()
    hb = state.get("last_heartbeat")
    if not hb:
        return True, float("inf"), state
    last = datetime.fromisoformat(hb)
    age = (datetime.now(UTC) - last).total_seconds()
    return age > max_age_seconds, age, state


def check_agent_silence(max_market_seconds: int = 7200,
                        max_offhours_seconds: int = 14400) -> dict:
    """Detect silent Layer 2 agent (no invocation for too long).

    Args:
        max_market_seconds: Max allowed silence during market hours (default 2h).
        max_offhours_seconds: Max allowed silence outside market hours (default 4h).

    Returns:
        dict with keys: silent (bool), age_seconds (float), threshold (int), market_open (bool)
    """
    # Try cached timestamp from health_state first (avoids re-parsing invocations.jsonl)
    last_ts = None
    state = load_health()
    last_ts = state.get("last_invocation_ts")

    # Fall back to parsing invocations.jsonl if health_state doesn't have the timestamp.
    if not last_ts:
        invocations_file = DATA_DIR / "invocations.jsonl"
        last_ts = last_jsonl_entry(invocations_file, field="ts")
        if last_ts is None:
            return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
        # Write back to health state so subsequent calls hit the cache
        # instead of re-parsing the JSONL file every time.
        with _health_lock:
            wb_state = load_health()
            wb_state["last_invocation_ts"] = last_ts
            atomic_write_json(HEALTH_FILE, wb_state)

    if not last_ts:
        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}

    try:
        last = datetime.fromisoformat(last_ts)
    except (ValueError, TypeError):
        logger.warning("Corrupt last_invocation_ts in health state: %r", last_ts)
        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
    now = datetime.now(UTC)
    age = (now - last).total_seconds()

    # DST-aware market hours check
    from portfolio.market_timing import get_market_state
    market_state, _, _ = get_market_state()
    market_open = (market_state == "open")
    threshold = max_market_seconds if market_open else max_offhours_seconds

    return {
        "silent": age > threshold,
        "age_seconds": round(age, 1),
        "threshold": threshold,
        "market_open": market_open,
    }


def update_module_failures(failures: list):
    """Record which reporting modules failed in the current cycle.

    Called by reporting.py after generating the agent summary.
    Persists module names + timestamp in health_state.json so the dashboard
    and monitoring scripts can see per-module status without parsing logs.

    Recovery semantics (2026-05-03): when called with an empty list AND a
    prior failure record exists, this clears the record so the dashboard
    reflects the *current* state, not a stale "last known failure" — the
    bug surfaced after a 2026-05-03 cycle-0 transient that left dashboard
    /api/health falsely flagging monte_carlo / price_targets / equity_curve
    as failed for hours after the modules had recovered.

    The clean-no-prior-failure case still skips the write to avoid
    spamming the disk every 60s for the common case.
    """
    with _health_lock:
        state = load_health()
        if failures:
            state["last_module_failures"] = {
                "ts": datetime.now(UTC).isoformat(),
                "modules": list(failures),
            }
            atomic_write_json(HEALTH_FILE, state)
        elif state.get("last_module_failures") is not None:
            # Recovery: prior failure cleared on a clean cycle. One-shot write.
            state.pop("last_module_failures", None)
            atomic_write_json(HEALTH_FILE, state)


def update_signal_health(signal_name: str, success: bool):
    """Record a single signal execution result.

    For batch updates (multiple signals per cycle), prefer
    update_signal_health_batch() to avoid repeated disk writes.
    """
    update_signal_health_batch({signal_name: success})


def update_signal_health_batch(results: dict):
    """Record multiple signal execution results in a single disk write.

    Args:
        results: dict of {signal_name: bool} where True=success, False=failure.
    """
    if not results:
        return
    with _health_lock:
        state = load_health()
        sh = state.setdefault("signal_health", {})
        now = datetime.now(UTC).isoformat()

        for signal_name, success in results.items():
            entry = sh.setdefault(signal_name, {
                "total_calls": 0,
                "total_failures": 0,
                "last_success": None,
                "last_failure": None,
                "recent_results": [],
            })
            entry["total_calls"] = entry.get("total_calls", 0) + 1
            if success:
                entry["last_success"] = now
            else:
                entry["total_failures"] = entry.get("total_failures", 0) + 1
                entry["last_failure"] = now

            # Rolling window: keep last 50 results for recent success rate
            recent = entry.get("recent_results", [])
            recent.append(success)
            if len(recent) > 50:
                recent = recent[-50:]
            entry["recent_results"] = recent

        atomic_write_json(HEALTH_FILE, state)


def get_signal_health(signal_name: str = None) -> dict:
    """Get signal health data.

    If signal_name is given, returns that signal's health dict.
    Otherwise returns the full signal_health dict for all signals.
    """
    state = load_health()
    sh = state.get("signal_health", {})
    if signal_name:
        return sh.get(signal_name, {})
    return sh


def get_signal_health_summary() -> dict:
    """Compact signal health summary for reporting.

    Returns dict of signal_name -> {success_rate, total_calls, total_failures,
    last_failure} for signals with at least 1 call.
    """
    sh = get_signal_health()
    summary = {}
    for sig_name, data in sh.items():
        total = data.get("total_calls", 0)
        if total == 0:
            continue
        failures = data.get("total_failures", 0)
        recent = data.get("recent_results", [])
        recent_rate = (sum(1 for r in recent if r) / len(recent) * 100) if recent else 0
        summary[sig_name] = {
            "success_rate_pct": round(recent_rate, 1),
            "total_calls": total,
            "total_failures": failures,
            "last_failure": data.get("last_failure"),
        }
    return summary


def get_health_summary() -> dict:
    """Return a summary dict suitable for API/dashboard consumption."""
    state = load_health()
    is_stale, age, _ = check_staleness()
    agent_silence = check_agent_silence()
    summary = {
        "status": "stale" if is_stale else "healthy",
        "heartbeat_age_seconds": round(age, 1),
        "cycle_count": state.get("cycle_count", 0),
        "error_count": state.get("error_count", 0),
        "last_trigger": state.get("last_trigger_reason"),
        "last_trigger_time": state.get("last_trigger_time"),
        "recent_errors": state.get("errors", [])[-5:],
        "signals_ok": state.get("signals_ok", 0),
        "signals_failed": state.get("signals_failed", 0),
        "agent_silent": agent_silence["silent"],
        "agent_silence_seconds": agent_silence["age_seconds"],
        "module_failures": state.get("last_module_failures"),
        "signal_health": get_signal_health_summary(),
    }
    # Include circuit breaker status if data_collector has been imported
    try:
        from portfolio.data_collector import alpaca_cb, binance_fapi_cb, binance_spot_cb
        summary["circuit_breakers"] = {
            "binance_spot": binance_spot_cb.get_status(),
            "binance_fapi": binance_fapi_cb.get_status(),
            "alpaca": alpaca_cb.get_status(),
        }
    except Exception as e:
        logger.warning("Circuit breaker status unavailable: %s", e)
    return summary


def check_outcome_staleness(max_age_hours: int = 36) -> dict:
    """Check if outcome backfill is stale (no recent outcomes in signal_log).

    Returns dict with: stale (bool), newest_outcome_age_hours (float),
    entries_without_outcomes (int).
    """
    signal_log = DATA_DIR / "signal_log.jsonl"

    now = time.time()
    newest_outcome_ts = 0
    missing_count = 0
    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
    entries = load_jsonl_tail(signal_log, max_entries=50)
    if not entries:
        return {"stale": True, "newest_outcome_age_hours": float("inf"),
                "entries_without_outcomes": 0}

    try:
        for entry in entries:
            outcomes = entry.get("outcomes", {})
            has_any = any(
                outcomes.get(t, {}).get("1d") is not None
                for t in outcomes
            )
            if has_any:
                # Parse outcome timestamps to find newest
                for t_outcomes in outcomes.values():
                    for h_data in t_outcomes.values():
                        if isinstance(h_data, dict) and h_data.get("ts"):
                            try:
                                ots = datetime.fromisoformat(h_data["ts"]).timestamp()
                                newest_outcome_ts = max(newest_outcome_ts, ots)
                            except (ValueError, TypeError):
                                pass
            else:
                missing_count += 1
    except Exception as exc:
        logger.warning("check_outcome_staleness error: %s", exc)
        return {"stale": True, "newest_outcome_age_hours": float("inf"),
                "entries_without_outcomes": 0}

    if newest_outcome_ts == 0:
        age_hours = float("inf")
    else:
        age_hours = (now - newest_outcome_ts) / 3600

    return {
        "stale": age_hours > max_age_hours,
        "newest_outcome_age_hours": round(age_hours, 1),
        "entries_without_outcomes": missing_count,
    }


def check_dead_signals(recent_entries: int = 20) -> list[str]:
    """Detect signals that voted HOLD on every ticker in the last N entries.

    Returns list of signal names that are effectively dead (100% HOLD).
    """
    signal_log = DATA_DIR / "signal_log.jsonl"

    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
    entries = load_jsonl_tail(signal_log, max_entries=recent_entries)
    if not entries:
        return []

    # Collect vote counts per signal
    from collections import defaultdict
    vote_counts = defaultdict(lambda: {"total": 0, "non_hold": 0})

    for entry in entries:
        for _ticker, tdata in entry.get("tickers", {}).items():
            for sig_name, vote in tdata.get("signals", {}).items():
                vote_counts[sig_name]["total"] += 1
                if vote in ("BUY", "SELL"):
                    vote_counts[sig_name]["non_hold"] += 1

    # Signals with >0 total votes but 0 non-HOLD votes are dead
    dead = []
    for sig_name, counts in vote_counts.items():
        if counts["total"] >= recent_entries and counts["non_hold"] == 0:
            dead.append(sig_name)
    return sorted(dead)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\market_timing.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 753ms:
"""Market timing utilities — DST-aware NYSE and EU hours, market state detection.

Includes US (NYSE) and Swedish (Nasdaq Stockholm / Avanza) holiday calendars
so the system skips stock/warrant processing on public holidays, not just weekends.
"""

from datetime import UTC, date, datetime, timedelta

from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS, SYMBOLS

# Backward compat: MARKET_OPEN_HOUR kept at 7 (summer value).
# Callers that need DST-aware EU open should use _eu_market_open_hour_utc().
MARKET_OPEN_HOUR = 7

# Loop intervals by market state.
# 2026-04-09: all bumped to 600s (10 min). The reduced 5-ticker universe +
# warm fingpt daemon means we no longer need a 60s fast tick — giving the
# loop breathing room eliminates cadence overruns without losing anything
# meaningful. Weekend was already 600s. See docs/PLAN_FINGPT_DAEMON.md.
INTERVAL_MARKET_OPEN = 600    # 10 min — previously 60s (pre-daemon era)
INTERVAL_MARKET_CLOSED = 600  # 10 min — previously 120s
INTERVAL_WEEKEND = 600        # 10 min — unchanged

# States where US stock markets are NOT trading — use this tuple instead of
# hardcoding ("closed", "weekend") to avoid missing the "holiday" state.
MARKET_CLOSED_STATES = ("closed", "weekend", "holiday")


def _is_eu_dst(dt):
    """Check if a UTC datetime falls within EU Summer Time (CEST).

    EU DST rule:
      Starts: last Sunday of March at 01:00 UTC
      Ends:   last Sunday of October at 01:00 UTC

    Returns True during CEST (summer), False during CET (winter).
    """
    year = dt.year

    # Last Sunday of March
    mar31 = date(year, 3, 31)
    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
    eu_dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)

    # Last Sunday of October
    oct31 = date(year, 10, 31)
    last_sun_oct = 31 - (oct31.weekday() + 1) % 7
    eu_dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)

    return eu_dst_start <= dt < eu_dst_end


def _eu_market_open_hour_utc(dt):
    """Return the EU market open hour in UTC, adjusted for EU DST.

    H47: London/Frankfurt open at 08:00 local time.
    CEST (summer, BST=UTC+1): 08:00 local = 07:00 UTC
    CET (winter, GMT=UTC+0): 08:00 local = 08:00 UTC

    Previously hardcoded to 7 UTC year-round, which missed the winter hour.
    """
    if _is_eu_dst(dt):
        return 7
    return 8


def _is_us_dst(dt):
    """Check if a UTC datetime falls within US Eastern Daylight Time (EDT).

    US DST rule (since 2007):
      Starts: second Sunday of March at 02:00 local (07:00 UTC)
      Ends:   first Sunday of November at 02:00 local (06:00 UTC)

    Returns True during EDT (Mar-Nov), False during EST (Nov-Mar).
    """
    year = dt.year

    # Second Sunday of March
    mar1_wd = date(year, 3, 1).weekday()  # 0=Mon..6=Sun
    first_sun_mar = 1 + (6 - mar1_wd) % 7
    second_sun_mar = first_sun_mar + 7
    dst_start = datetime(year, 3, second_sun_mar, 7, 0, tzinfo=UTC)

    # First Sunday of November
    nov1_wd = date(year, 11, 1).weekday()
    first_sun_nov = 1 + (6 - nov1_wd) % 7
    dst_end = datetime(year, 11, first_sun_nov, 6, 0, tzinfo=UTC)

    return dst_start <= dt < dst_end


def _market_close_hour_utc(dt):
    """Return the NYSE close hour in UTC, adjusted for DST.

    NYSE closes at 16:00 ET.
    EDT (Mar-Nov): 16:00 ET = 20:00 UTC
    EST (Nov-Mar): 16:00 ET = 21:00 UTC
    """
    if _is_us_dst(dt):
        return 20
    return 21


# ---------------------------------------------------------------------------
# Holiday calendars
# ---------------------------------------------------------------------------


def _easter_sunday(year):
    """Compute Easter Sunday for a given year using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d):
    """Return the NYSE-observed date for a fixed holiday.

    If the holiday falls on Saturday, NYSE observes it Friday.
    If Sunday, NYSE observes it Monday.
    """
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year, month, weekday, n):
    """Return the nth occurrence of a weekday in a given month.

    weekday: 0=Mon, 6=Sun.  n: 1-based (1=first, 2=second, etc.)
    """
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    """Return the last occurrence of a weekday in a given month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


# Holiday set cache — avoids recalculating Easter + date arithmetic every cycle.
# Keyed by (country, year). Invalidated implicitly on year boundary since the
# key includes the year.
_holiday_cache: dict[tuple[str, int], set] = {}


def us_market_holidays(year):
    """Return the set of NYSE holiday dates for a given year.

    Covers all 10 NYSE holidays including observed-date shifts.
    Results are cached per year.
    """
    key = ("us", year)
    if key in _holiday_cache:
        return _holiday_cache[key]
    easter = _easter_sunday(year)
    holidays = {
        _observed(date(year, 1, 1)),                 # New Year's Day
        _nth_weekday(year, 1, 0, 3),                 # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                 # Presidents' Day (3rd Mon Feb)
        easter - timedelta(days=2),                   # Good Friday
        _last_weekday(year, 5, 0),                    # Memorial Day (last Mon May)
        _observed(date(year, 6, 19)),                 # Juneteenth
        _observed(date(year, 7, 4)),                  # Independence Day
        _nth_weekday(year, 9, 0, 1),                  # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                 # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),                # Christmas
    }
    _holiday_cache[key] = holidays
    return holidays


def is_us_market_holiday(dt=None):
    """Return True if the given UTC datetime falls on a NYSE holiday."""
    if dt is None:
        dt = datetime.now(UTC)
    d = dt.date() if hasattr(dt, "date") else dt
    return d in us_market_holidays(d.year)


def swedish_market_holidays(year):
    """Return the set of Nasdaq Stockholm / Avanza holiday dates for a given year.

    Covers full days when Avanza warrant trading is closed.
    Results are cached per year.
    """
    key = ("se", year)
    if key in _holiday_cache:
        return _holiday_cache[key]
    easter = _easter_sunday(year)

    # Midsummer Eve: Friday before Midsummer Day (Saturday between Jun 20-26)
    # Midsummer Day = first Saturday on or after Jun 20
    jun20 = date(year, 6, 20)
    days_to_sat = (5 - jun20.weekday()) % 7
    midsummer_day = jun20 + timedelta(days=days_to_sat)
    midsummer_eve = midsummer_day - timedelta(days=1)

    holidays = {
        date(year, 1, 1),                            # New Year's Day
        date(year, 1, 6),                             # Epiphany
        easter - timedelta(days=2),                   # Good Friday
        easter + timedelta(days=1),                   # Easter Monday
        date(year, 5, 1),                             # May Day
        easter + timedelta(days=39),                  # Ascension Day
        easter + timedelta(days=48),                  # Whitsun Eve (Pingstafton)
        date(year, 6, 6),                             # National Day
        midsummer_eve,                                # Midsummer Eve
        date(year, 12, 24),                           # Christmas Eve
        date(year, 12, 25),                           # Christmas Day
        date(year, 12, 26),                           # Boxing Day
        date(year, 12, 31),                           # New Year's Eve
    }
    _holiday_cache[key] = holidays
    return holidays


def is_swedish_market_holiday(dt=None):
    """Return True if the given UTC datetime falls on a Swedish market holiday."""
    if dt is None:
        dt = datetime.now(UTC)
    d = dt.date() if hasattr(dt, "date") else dt
    return d in swedish_market_holidays(d.year)


def _is_agent_window(now=None):
    """Check if current time is within the Layer 2 invocation window.

    Window: EU market open through US market close.
    Summer: 07:00–20:00 UTC
    Winter: 08:00–21:00 UTC
    Weekends and US market holidays: no agent invocation.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.weekday() >= 5:
        return False
    if is_us_market_holiday(now):
        return False
    eu_open = _eu_market_open_hour_utc(now)
    close_hour = _market_close_hour_utc(now)
    return eu_open <= now.hour < close_hour


def _market_open_hour_utc(dt):
    """Return the NYSE open hour in UTC, adjusted for DST.

    NYSE opens at 09:30 ET.
    EDT (Mar-Nov): 09:30 ET = 13:30 UTC -> hour 13
    EST (Nov-Mar): 09:30 ET = 14:30 UTC -> hour 14
    """
    if _is_us_dst(dt):
        return 13
    return 14


def is_us_stock_market_open(now=None, pre_market_buffer_min=0, post_market_buffer_min=0):
    """Check if US stock market (NYSE) is currently open.

    Args:
        now: UTC datetime (default: current time)
        pre_market_buffer_min: minutes before open to consider "open"
        post_market_buffer_min: minutes after close to consider "open"

    Returns:
        True if within [open - pre_buffer, close + post_buffer] on weekdays.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.weekday() >= 5:
        return False
    if is_us_market_holiday(now):
        return False

    open_hour = _market_open_hour_utc(now)
    close_hour = _market_close_hour_utc(now)

    # Convert to minutes-since-midnight for easy buffer math
    now_min = now.hour * 60 + now.minute
    open_min = open_hour * 60 + 30 - pre_market_buffer_min   # NYSE opens at :30
    close_min = close_hour * 60 + post_market_buffer_min      # NYSE closes at :00

    return open_min <= now_min < close_min


def should_skip_gpu(ticker, config=None, now=None):
    """Determine if GPU-intensive signals should be skipped for this ticker.

    Returns True for US stocks when the US market is closed.
    Returns False for crypto and metals (always run GPU signals).
    """
    if ticker not in STOCK_SYMBOLS:
        return False

    gpu_cfg = (config or {}).get("gpu_signals", {})
    if not gpu_cfg.get("skip_stocks_offhours", True):
        return False

    pre_buffer = gpu_cfg.get("pre_market_buffer_min", 30)
    post_buffer = gpu_cfg.get("post_market_buffer_min", 15)

    return not is_us_stock_market_open(
        now=now,
        pre_market_buffer_min=pre_buffer,
        post_market_buffer_min=post_buffer,
    )


def get_market_state():
    now = datetime.now(UTC)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    all_symbols = set(SYMBOLS.keys())
    always_on = CRYPTO_SYMBOLS | METALS_SYMBOLS
    if weekday >= 5:
        return "weekend", always_on, INTERVAL_WEEKEND
    # US holiday: treat like off-hours (crypto + metals only, 2-min interval)
    if is_us_market_holiday(now):
        return "holiday", always_on, INTERVAL_MARKET_CLOSED
    eu_open = _eu_market_open_hour_utc(now)
    close_hour = _market_close_hour_utc(now)
    if eu_open <= hour < close_hour:
        return "open", all_symbols, INTERVAL_MARKET_OPEN
    return "closed", always_on, INTERVAL_MARKET_CLOSED

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\perception_gate.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 742ms:
"""Perception gate — filters low-value Layer 2 invocations.

A rule-based pre-invocation filter (NOT an LLM call). Checks signal
consensus strength and trigger importance. If the gate decides to skip,
the agent is not invoked, saving tokens and latency.

Config:
    "perception_gate": {
        "enabled": true,
        "min_signal_strength": 0.3,
        "skip_tiers": [1]
    }
"""

import logging
from pathlib import Path

from portfolio.api_utils import load_config
from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.perception_gate")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Trigger keywords that always bypass the gate
_BYPASS_KEYWORDS = ("consensus", "F&G crossed", "post-trade")


def should_invoke(reasons, tier, config=None):
    """Decide whether to invoke Layer 2.

    Args:
        reasons: list[str] of trigger reasons.
        tier: int (1, 2, or 3).
        config: optional config dict. Loaded from disk if None.

    Returns:
        (should_invoke: bool, reason: str) explaining the decision.
    """
    if config is None:
        config = load_config()

    gate_cfg = config.get("perception_gate", {})
    if not gate_cfg.get("enabled", False):
        return True, "gate disabled"

    skip_tiers = gate_cfg.get("skip_tiers", [1])
    if tier not in skip_tiers:
        return True, f"T{tier} not in skip_tiers"

    # Force-bypass for important triggers
    for reason in reasons:
        for keyword in _BYPASS_KEYWORDS:
            if keyword in reason:
                return True, f"bypass: {keyword!r} in trigger"

    # Check signals from compact summary
    min_strength = gate_cfg.get("min_signal_strength", 0.3)
    summary = _load_compact_summary()
    if summary is None:
        return True, "no summary available, pass through"

    signals = summary.get("signals", {})
    if not signals:
        return False, "no signals in summary"

    max_confidence = 0.0
    non_hold_count = 0
    for _ticker, sig in signals.items():
        if not isinstance(sig, dict):
            continue
        action = sig.get("action", "HOLD")
        conf = sig.get("confidence", 0.0)
        if action != "HOLD":
            non_hold_count += 1
            if conf > max_confidence:
                max_confidence = conf

    if non_hold_count == 0:
        return False, "no non-HOLD signals"

    if max_confidence < min_strength:
        return False, f"max confidence {max_confidence:.2f} < {min_strength}"

    return True, f"{non_hold_count} active signals, max conf {max_confidence:.2f}"


def _load_compact_summary():
    """Load the compact summary JSON."""
    path = DATA_DIR / "agent_summary_compact.json"
    result = load_json(path)
    if result is not None:
        return result
    path = DATA_DIR / "agent_summary.json"
    return load_json(path)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | Select-String -Pattern 'should_invoke\\('" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 676ms:

portfolio\agent_invocation.py:615:        should, gate_reason = _[7mshould_invoke([0mreasons, tier)
portfolio\perception_gate.py:29:def [7mshould_invoke([0mreasons, tier, config=None):


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\reflection.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 626ms:
"""Periodic trade reflection — computes structured performance metrics.

Pure Python (no LLM call). After every N trades, generates a reflection
summary stored in data/reflections.jsonl. Layer 2 reads the latest
reflection as context for self-assessment.

Config:
    "reflection": {
        "enabled": true,
        "trade_interval": 10,
        "max_age_days": 7
    }
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl

logger = logging.getLogger("portfolio.reflection")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFLECTIONS_FILE = DATA_DIR / "reflections.jsonl"
PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"


def _count_trades(portfolio):
    """Count total trades in a portfolio state."""
    return len(portfolio.get("transactions", []))


def _compute_strategy_metrics(portfolio):
    """Compute win rate, avg PnL, and total PnL for a strategy.

    Returns dict with trades, win_rate, avg_pnl_pct, total_pnl_pct, holdings.
    """
    txns = portfolio.get("transactions", [])
    initial = portfolio.get("initial_value_sek", 500000)
    cash = portfolio.get("cash_sek", initial)
    holdings = portfolio.get("holdings", {})
    holding_tickers = [t for t, h in holdings.items() if h.get("shares", 0) > 0]

    if not txns:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_pnl_pct": None,
            "total_pnl_pct": round((cash - initial) / initial * 100, 2),
            "holdings": holding_tickers,
        }

    # Match BUY/SELL pairs per ticker for PnL
    buys = {}  # ticker -> list of (price_sek, shares)
    sells = []  # list of (ticker, pnl_pct)

    for tx in txns:
        ticker = tx.get("ticker", "")
        action = tx.get("action", "")
        shares = tx.get("shares", 0)
        price = tx.get("price_sek", 0)

        if action == "BUY" and price > 0:
            buys.setdefault(ticker, []).append((price, shares))
        elif action == "SELL" and price > 0:
            # Compute PnL against avg cost of prior buys
            buy_list = buys.get(ticker, [])
            if buy_list:
                total_cost = sum(p * s for p, s in buy_list)
                total_shares = sum(s for _, s in buy_list)
                avg_cost = total_cost / total_shares if total_shares > 0 else price
                pnl_pct = (price - avg_cost) / avg_cost * 100
                sells.append((ticker, pnl_pct))

    wins = sum(1 for _, pnl in sells if pnl > 0)
    win_rate = round(wins / len(sells), 2) if sells else None
    avg_pnl = round(sum(pnl for _, pnl in sells) / len(sells), 2) if sells else None
    total_pnl_pct = round((cash - initial) / initial * 100, 2)

    return {
        "trades": len(txns),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl,
        "total_pnl_pct": total_pnl_pct,
        "holdings": holding_tickers,
    }


def _regime_distribution():
    """Count regime occurrences in recent journal entries."""
    entries = load_jsonl(JOURNAL_FILE, limit=100)
    dist = {}
    for e in entries:
        regime = e.get("regime", "unknown")
        dist[regime] = dist.get(regime, 0) + 1
    return dist


def _generate_insights(patient_metrics, bold_metrics):
    """Generate human-readable insights from metrics."""
    insights = []

    for label, m in [("Patient", patient_metrics), ("Bold", bold_metrics)]:
        trades = m.get("trades", 0)
        if trades == 0:
            insights.append(f"{label}: no trades yet")
            continue

        win_rate = m.get("win_rate")
        if win_rate is not None:
            if win_rate == 0:
                insights.append(f"{label}: all {trades} closed trades were losses")
            elif win_rate >= 0.7:
                insights.append(f"{label}: strong {win_rate:.0%} win rate over {trades} trades")
            elif win_rate < 0.4:
                insights.append(f"{label}: weak {win_rate:.0%} win rate — review entry criteria")

        total_pnl = m.get("total_pnl_pct", 0)
        if total_pnl < -5:
            insights.append(f"{label}: down {abs(total_pnl):.1f}% — consider reducing size")
        elif total_pnl > 5:
            insights.append(f"{label}: up {total_pnl:.1f}% — strategy working")

    return insights


def should_reflect(config=None):
    """Check whether a new reflection is due.

    Returns True if:
    - Feature is enabled
    - Total trade count crossed the interval threshold since last reflection
    - OR last reflection is older than max_age_days
    """
    if config is None:
        from portfolio.api_utils import load_config
        config = load_config()

    ref_cfg = config.get("reflection", {})
    if not ref_cfg.get("enabled", False):
        return False

    interval = ref_cfg.get("trade_interval", 10)
    max_age_days = ref_cfg.get("max_age_days", 7)

    # Count total trades across both portfolios
    patient = load_json(PORTFOLIO_FILE, {})
    bold = load_json(BOLD_FILE, {})
    total_trades = _count_trades(patient) + _count_trades(bold)

    # Load last reflection
    reflections = load_jsonl(REFLECTIONS_FILE)
    if not reflections:
        return total_trades >= interval

    last = reflections[-1]
    last_trade_count = last.get("trade_count_total", 0)

    # Check trade interval
    if total_trades - last_trade_count >= interval:
        return True

    # Check age
    try:
        last_ts = datetime.fromisoformat(last["ts"])
        age = datetime.now(UTC) - last_ts
        if age > timedelta(days=max_age_days):
            return True
    except (KeyError, ValueError):
        return True

    return False


def compute_reflection():
    """Compute a reflection entry from current portfolio states.

    Returns a reflection dict ready to be saved.
    """
    patient = load_json(PORTFOLIO_FILE, {})
    bold = load_json(BOLD_FILE, {})

    patient_metrics = _compute_strategy_metrics(patient)
    bold_metrics = _compute_strategy_metrics(bold)
    regime_dist = _regime_distribution()
    insights = _generate_insights(patient_metrics, bold_metrics)

    total_trades = _count_trades(patient) + _count_trades(bold)

    return {
        "ts": datetime.now(UTC).isoformat(),
        "patient": patient_metrics,
        "bold": bold_metrics,
        "regime_distribution": regime_dist,
        "trade_count_total": total_trades,
        "insights": insights,
    }


def save_reflection(reflection):
    """Save a reflection entry to the JSONL file."""
    atomic_append_jsonl(REFLECTIONS_FILE, reflection)
    logger.info("Reflection saved: %d total trades", reflection.get("trade_count_total", 0))


def maybe_reflect(config=None):
    """Check if a reflection is due and compute/save one if so.

    Called from main loop. Non-blocking, non-critical.
    """
    try:
        if should_reflect(config):
            reflection = compute_reflection()
            save_reflection(reflection)
            return True
    except Exception as e:
        logger.warning("reflection failed: %s", e)
    return False


def load_latest_reflection(max_age_days=7):
    """Load the most recent reflection if it exists and isn't too old.

    Returns the reflection dict or None.
    """
    reflections = load_jsonl(REFLECTIONS_FILE)
    if not reflections:
        return None

    last = reflections[-1]
    try:
        ts = datetime.fromisoformat(last["ts"])
        age = datetime.now(UTC) - ts
        if age > timedelta(days=max_age_days):
            return None
    except (KeyError, ValueError):
        pass

    return last

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\tickers.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 611ms:
"""Single source of truth for all ticker lists, source mappings, and symbol constants.

Every module that needs ticker definitions should import from here instead
of maintaining its own copy.
"""

import re
from functools import lru_cache
from pathlib import Path

# ── Tier 1: Full signals (30 signals, 7 timeframes) ──────────────────────

SYMBOLS = {
    # Crypto (Binance spot)
    "BTC-USD": {"binance": "BTCUSDT"},
    "ETH-USD": {"binance": "ETHUSDT"},
    # Metals (Binance futures)
    "XAU-USD": {"binance_fapi": "XAUUSDT"},
    "XAG-USD": {"binance_fapi": "XAGUSDT"},
    # US Equities (Alpaca IEX) — MSTR kept as BTC NAV-premium reference for metals_loop
    "MSTR": {"alpaca": "MSTR"},
    # Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT
    # Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT
    #   Reduces main loop load to stay under 60s cadence. Cycle p50 was 143s with
    #   12 tickers — dropping to 5 is expected to bring p50 under target. MSTR retained
    #   because data/metals_loop.py uses it for BTC NAV-premium tracking.
}

# ── Asset-class subsets ───────────────────────────────────────────────────

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
STOCK_SYMBOLS = {"MSTR"}

# All known tickers (union of all subsets)
ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS

# ── Derived mappings (all from SYMBOLS — single source of truth) ─────────

BINANCE_SPOT_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance"
}
BINANCE_FAPI_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance_fapi"
}
BINANCE_MAP = {**BINANCE_SPOT_MAP, **BINANCE_FAPI_MAP}

# Ticker -> (source_type, symbol) mapping (used by macro_context)
TICKER_SOURCE_MAP = {
    t: next(iter(src.items())) for t, src in SYMBOLS.items()
}

# Yahoo Finance symbol mapping — stock tickers map to themselves
YF_MAP = {t: t for t in STOCK_SYMBOLS}

# ── Signal names (used by outcome_tracker, accuracy_stats) ───────────────
# Canonical source is portfolio.signal_registry.get_signal_names().
# This static list is kept for backward compatibility with modules that
# import SIGNAL_NAMES directly (outcome_tracker, accuracy_stats).

# Signals that are force-HOLD (disabled due to poor accuracy).
# Kept in SIGNAL_NAMES for historical tracking but excluded from active reports.
DISABLED_SIGNALS = {
    "ml",               # 41.7% accuracy (1714 sam) — worse than coin flip
    "fibonacci",        # 2026-04-29: 43.6% at 1d (17024 sam), 43.3% at 3h (8811 sam).
                        # Consistently below coin flip across ALL horizons and tickers
                        # with massive sample size. Was accuracy-gated but still computed
                        # every cycle (~50ms wasted). Formal disable saves CPU.
    # "cot_positioning" re-enabled 2026-04-13 for shadow validation (was
    # force-HOLD pending live validation, 0 samples). COT is a weekly signal
    # (CFTC Friday release) — expected to contribute mostly at 3d/5d horizons
    # where the system already has edge (XAG 5d consensus 61.2%). The
    # existing accuracy gate in signal_engine.py auto-disables any signal
    # below 45% accuracy once 30+ samples accumulate, so re-enabling is
    # self-correcting.
    "futures_basis",    # 0 accuracy samples — pending live validation
    "hurst_regime",     # pending live validation (added 2026-04-11)
    "shannon_entropy",  # pending live validation (added 2026-04-12)
    "vix_term_structure",  # pending live validation (added 2026-04-13)
    "gold_real_yield_paradox",  # pending live validation (added 2026-04-14)
    "cross_asset_tsmom",  # pending live validation (added 2026-04-15)
    "copper_gold_ratio",  # pending live validation (added 2026-04-17)
    # "statistical_jump_regime" RE-ENABLED 2026-04-29: 52.7% accuracy (110 sam)
    # at 1d — above 47% gate, marginal but worth live validation. Shadow-safe
    # since 2026-04-18. If it degrades below 47% the accuracy gate auto-disables.
    "network_momentum",  # pending live validation (added 2026-04-19)
    "ovx_metals_spillover",  # pending live validation (added 2026-04-20)
    "xtrend_equity_spillover",  # pending live validation (added 2026-04-21)
    "complexity_gap_regime",  # pending live validation (added 2026-04-22)
    "realized_skewness",  # KILLED 2026-04-29: 33.3% at 1d (90 sam). Below coin flip.
    "mahalanobis_turbulence",  # pending live validation (added 2026-04-24)
    "crypto_evrp",  # pending live validation (added 2026-04-25)
    "hash_ribbons",  # pending live validation (added 2026-04-26)
    "drift_regime_gate",  # pending live validation (added 2026-04-28)
    "vol_ratio_regime",  # pending live validation (added 2026-04-29)
    "residual_pair_reversion",  # pending live validation (added 2026-04-30)
    "williams_vix_fix",  # pending live validation (added 2026-05-01)
    "treasury_risk_rotation",  # pending live validation (added 2026-05-07)
    "futures_flow",     # 2026-05-07: 38.3% at 1d (2168 sam). Actively harmful —
                        # 12pp worse than coin flip. In cross_asset_flow cluster
                        # but still wastes compute. Was accuracy-gated at runtime
                        # but formal disable saves ~50ms/cycle.
    "trend",            # 2026-05-07: 46.1% at 1d (17880 sam), 40.3% at 3h.
                        # Massive sample, consistently below threshold across ALL
                        # horizons. 92-100% correlated with ema/macro_regime in
                        # pure_trend cluster. In ranging regime (current) this is
                        # pure noise. ema (50.0%) is the cluster leader.
    "macd",             # 2026-05-07: 44.2% at 1d (6136 sam), 43.7% at 3h.
                        # Below threshold across all horizons. Only 5.3% activation
                        # on XAG. In oscillator_trend cluster where momentum_factors
                        # (53.2%) is the better signal.
    # "econ_calendar" RE-ENABLED 2026-04-23. BUG-218 fixed: added post_event_relief
    # sub-signal that emits BUY after high-impact events pass (4-24h relief window)
    # and during event-free calm windows (>72h to next event). The composite is now
    # 5 sub-signals (3 SELL + 1 BUY + 1 neutral) instead of 4 SELL-only.
    # 62.6% accuracy before disabling. Accuracy gate will auto-gate if BUY
    # signals degrade the composite.
    "orderbook_flow",   # 2026-04-11: 51.1% accuracy (360 sam), 93.3% activation rate,
                        # no recent data. Pure noise in every consensus decision.
                        # Re-evaluate after 2 weeks of accuracy data collection.
    # "forecast" RE-ENABLED 2026-04-21. The 36-39% accuracy measured on 2026-04-12
    # was polluted by Kronos voting 100% HOLD in shadow mode — Kronos occupied 3 of 6
    # slots in _health_weighted_vote whenever its subprocess succeeded, dragging every
    # composite vote toward HOLD regardless of Chronos's verdict. With Kronos retired
    # in portfolio/signals/forecast.py (same PR), the composite is now Chronos-only.
    # Chronos effective accuracy: 1h=45.4%, 24h=52.4% (4d ago). The 47% tiered
    # accuracy gate will force-HOLD 1h while letting 24h contribute. Forecast stayed
    # in this set for 10 days, which ALSO silenced forecast_predictions.jsonl and
    # forecast_health.jsonl because signal_engine.py skips disabled signals before
    # invocation — so we lost all shadow/health visibility while the signal was off.
    # Re-enabling restores both the signal and the logging. If accuracy degrades
    # again post-Kronos-retire, move into REGIME_GATED_SIGNALS (24h-only) rather
    # than re-disabling blindly.
    "oscillators",      # 2026-04-14: below 45% on ALL tickers at 1d (BTC 35.8%, ETH 36.3%,
                        # XAG 34.9%, XAU 40.2%, MSTR 42.6%; 5065 total sam). Also weak at
                        # 3h (34-45% per ticker). Regime-gated in ranging but noise everywhere.
    "smart_money",      # 2026-04-24: below 40% on ALL Tier 1 tickers at 1d — BTC 39.8% (123),
                        # ETH 34.9% (146), MSTR 33.3% (264), XAU N/A. Not salvageable.
                        # Cross-ticker consistent failure. 51.6% aggregate masks per-ticker disaster.
    "claude_fundamental",  # 2026-05-03: CRASHED to 19.8% recent 1d (222 sam) from 57.9%
                        # all-time. Root cause: Opus tier has 95% BUY bias (76/80 votes BUY),
                        # Sonnet 73% BUY bias. Haiku 83% abstention (useless). In ranging
                        # market these BUY calls are mostly wrong. Bias detectors (added
                        # 2026-04-25) couldn't prevent structural LLM bullish lean.
                        # Re-enable after fixing bias detector thresholds.
    "sentiment",        # 2026-05-03: 33.8% at 3h recent (3629 sam), 45.9% all-time (39579 sam).
                        # CryptoBERT predictions are noise. High-volume signal actively hurting
                        # consensus. Always in macro_external cluster but dragging down peers.
}
# 2026-04-11 research session changes:
# - orderbook_flow DISABLED: 93.3% active, 51.1% accuracy, 0 recent data. Noise.
# - credit_spread_risk ENABLED: 66.9% accuracy (257 sam), BUY 80.3%. Directional
#   gate at 40% will auto-gate SELL (49.1%) while allowing strong BUY votes.
# - crypto_macro ENABLED: 56.5% accuracy (1273 sam). BUY-biased (93%) so bias
#   penalty (0.5x) applies. Provides crypto-specific on-chain edge.
# funding: removed from DISABLED — 74.2% at 3h (535 samples) but 29.9% at 1d.
# Horizon-gated via REGIME_GATED_SIGNALS to only vote at 3h/4h.

# 2026-05-05: Surface the disable reason to the dashboard tooltip by parsing the
# inline comments next to each DISABLED_SIGNALS entry. Done via source-file
# parsing (rather than a parallel dict) so the comments stay the single source
# of truth. Falls back to None if the file shape changes.
_DISABLED_REASON_ENTRY_RE = re.compile(
    r'^(\s*)"([a-z_][a-z0-9_]*)"\s*,\s*(?:#\s*(.*))?$'
)
_DISABLED_REASON_CONT_RE = re.compile(r'^(\s+)#\s*(.*)$')


def _clean_disabled_reason(lines: list[str]) -> str:
    """Join continuation comments and trim to a single short summary."""
    if not lines:
        return ""
    text = " ".join(lines).strip()
    for sep in (". ", " — "):
        if sep in text:
            text = text.split(sep, 1)[0].rstrip(".")
            break
    return text[:160].rstrip()


@lru_cache(maxsize=1)
def _parse_disabled_reasons() -> dict[str, str]:
    """Parse the DISABLED_SIGNALS literal in this file into {name: reason}.

    A continuation comment is recognised when its `#` is indented strictly
    further than the entry name's column, which excludes flush-left
    separator comments (e.g. the commented-out re-enable notes) from
    bleeding into the previous entry's reason.
    """
    try:
        src = Path(__file__).resolve().read_text(encoding="utf-8")
    except OSError:
        return {}
    block_match = re.search(
        r'^DISABLED_SIGNALS\s*=\s*\{(.*?)^\}',
        src, re.MULTILINE | re.DOTALL,
    )
    if not block_match:
        return {}
    out: dict[str, str] = {}
    current: str | None = None
    current_lines: list[str] = []
    entry_indent = 0
    for raw in block_match.group(1).splitlines():
        m_entry = _DISABLED_REASON_ENTRY_RE.match(raw)
        if m_entry:
            if current is not None:
                out[current] = _clean_disabled_reason(current_lines)
            current = m_entry.group(2)
            entry_indent = len(m_entry.group(1))
            first = (m_entry.group(3) or "").strip()
            current_lines = [first] if first else []
            continue
        m_cont = _DISABLED_REASON_CONT_RE.match(raw)
        if m_cont and current is not None:
            indent = len(m_cont.group(1))
            if indent > entry_indent:
                txt = m_cont.group(2).strip()
                if txt:
                    current_lines.append(txt)
    if current is not None:
        out[current] = _clean_disabled_reason(current_lines)
    return out


def get_disabled_reason(signal_name: str) -> str | None:
    """Return a short reason for why `signal_name` is disabled, or None.

    Returns None for signals not in DISABLED_SIGNALS, and for disabled
    signals whose comment was empty or unparseable.
    """
    if signal_name not in DISABLED_SIGNALS:
        return None
    reasons = _parse_disabled_reasons()
    reason = reasons.get(signal_name)
    return reason if reason else None


# Signals that require local GPU inference.
# Skipped for US stocks outside market hours to save GPU resources.
# claude_fundamental excluded — uses remote API, has its own market-hours gate.
GPU_SIGNALS = frozenset({"ministral", "qwen3", "forecast"})

SIGNAL_NAMES = [
    "rsi",
    "macd",
    "ema",
    "bb",
    "fear_greed",
    "sentiment",
    "ministral",
    "ml",
    "funding",
    "volume",
    "qwen3",
    # custom_lora removed — disabled signal, was polluting accuracy stats
    # Enhanced composite signals
    "trend",
    "momentum",
    "volume_flow",
    "volatility_sig",
    "candlestick",
    "structure",
    "fibonacci",
    "smart_money",
    "oscillators",
    "heikin_ashi",
    "mean_reversion",
    "calendar",
    "macro_regime",
    "momentum_factors",
    "news_event",
    "econ_calendar",
    "forecast",
    "claude_fundamental",
    "futures_flow",
    "crypto_macro",
    "orderbook_flow",
    "metals_cross_asset",
    "dxy_cross_asset",
    "cot_positioning",
    "credit_spread_risk",
    "onchain",
    "futures_basis",
    "hurst_regime",
    "shannon_entropy",
    "vix_term_structure",
    "gold_real_yield_paradox",
    "cross_asset_tsmom",
    "copper_gold_ratio",
    "statistical_jump_regime",
    "network_momentum",
    "ovx_metals_spillover",
    "xtrend_equity_spillover",
    "complexity_gap_regime",
    "realized_skewness",
    "mahalanobis_turbulence",
    "crypto_evrp",
    "hash_ribbons",
    "drift_regime_gate",
    "vol_ratio_regime",
    "residual_pair_reversion",
    "williams_vix_fix",
    "treasury_risk_rotation",
]

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\config_validator.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 628ms:
"""Config validation for portfolio system startup.

Validates config.json has all required keys before the main loop starts.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("portfolio.config_validator")

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

# Required: missing any of these is a fatal error
REQUIRED_KEYS = [
    ("telegram", "token"),
    ("telegram", "chat_id"),
    ("alpaca", "key"),
    ("alpaca", "secret"),
]

# Optional: missing these produces a warning but isn't fatal
OPTIONAL_KEYS = [
    ("mistral_api_key",),
    ("iskbets",),
    ("newsapi_key",),
    ("alpha_vantage", "api_key"),
    ("golddigger", "fred_api_key"),
    ("bgeometrics", "api_token"),
]


def validate_config(config: dict) -> list[str]:
    """Validate config dict. Returns list of error strings (empty = valid)."""
    errors = []
    for key_path in REQUIRED_KEYS:
        obj = config
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                errors.append(f"missing required key: {'.'.join(key_path)}")
                break
            obj = obj[key]
        else:
            # Key exists — check it's not empty/placeholder
            if isinstance(obj, str) and not obj.strip():
                errors.append(f"empty value for required key: {'.'.join(key_path)}")
    return errors


def validate_config_file() -> dict:
    """Load config.json, validate, and return it.

    Logs warnings for missing optional keys.
    Raises ValueError if required keys are missing.
    """
    if not CONFIG_FILE.exists():
        raise ValueError(f"config.json not found at {CONFIG_FILE}")

    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)

    # Check optional keys and warn
    for key_path in OPTIONAL_KEYS:
        obj = config
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                logger.warning("optional config key missing: %s", '.'.join(key_path))
                break
            obj = obj[key]

    # Check required keys
    errors = validate_config(config)
    if errors:
        for err in errors:
            logger.error("config validation: %s", err)
        raise ValueError(f"config.json validation failed: {'; '.join(errors)}")

    logger.info("config.json validated successfully")
    return config

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\multi_agent_layer2.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 605ms:
"""Multi-agent Layer 2 orchestration — parallel specialists + synthesis.

Inspired by Claude Code's Coordinator Mode. Instead of one monolithic
agent reading everything, splits analysis into parallel specialists:

    1. Technical Agent: signals, regime, momentum, trend
    2. Risk Agent: portfolio state, exposure, drawdown, stops
    3. Microstructure Agent: order flow, depth, cross-asset context

Each specialist writes a brief report to a temp file. A synthesis agent
reads all three and makes the final BUY/SELL/HOLD decision.

Key design principles (from Claude Code's Agent architecture):
    - Fresh context per agent (no context pollution)
    - 5-word task description forces clarity
    - Standardized report format for mechanical parsing
    - Parent owns the gate — synthesis agent makes final call
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from portfolio.claude_gate import detect_auth_failure

logger = logging.getLogger("portfolio.multi_agent_layer2")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SPECIALISTS = {
    "technical": {
        "focus": "Technical analysis: signals, regime, momentum, trend direction",
        "data_files": [
            "data/agent_context_t2.json",
            "data/accuracy_cache.json",
        ],
        "output_file": "data/_specialist_technical.md",
        "timeout": 120,
        "max_turns": 10,
    },
    "risk": {
        "focus": "Risk assessment: portfolio exposure, drawdown, stop levels, position sizing",
        "data_files": [
            "data/portfolio_state.json",
            "data/portfolio_state_bold.json",
            "data/portfolio_state_warrants.json",
        ],
        "output_file": "data/_specialist_risk.md",
        "timeout": 90,
        "max_turns": 8,
    },
    "microstructure": {
        "focus": "Order flow and cross-asset: depth imbalance, trade flow, VPIN, copper, GVZ, gold/silver ratio",
        "data_files": [
            "data/microstructure_state.json",
            "data/seasonality_profiles.json",
        ],
        "output_file": "data/_specialist_microstructure.md",
        "timeout": 90,
        "max_turns": 8,
    },
}


def build_specialist_prompts(
    ticker: str,
    trigger_reasons: list[str],
) -> dict[str, str]:
    """Build prompts for each specialist agent.

    Returns dict keyed by specialist name with prompt strings.
    """
    reason_str = ", ".join(trigger_reasons[:5])
    prompts = {}

    for name, spec in SPECIALISTS.items():
        data_reads = " ".join(f"Read {f}." for f in spec["data_files"])
        prompts[name] = (
            f"You are a {name} specialist for the trading system. "
            f"Ticker: {ticker}. Trigger: {reason_str}. "
            f"Focus: {spec['focus']}. "
            f"{data_reads} "
            f"Write a brief analysis (max 500 words) to {spec['output_file']}. "
            "Include: current state, key signals, recommendation "
            "(bullish/bearish/neutral), and confidence (low/medium/high). "
            "Be concise and data-driven. Do NOT make trade decisions."
        )

    return prompts


def build_synthesis_prompt(
    ticker: str,
    trigger_reasons: list[str],
    report_paths: list[str] | None = None,
) -> str:
    """Build the synthesis agent prompt that reads all specialist reports."""
    reason_str = ", ".join(trigger_reasons[:5])
    if report_paths is None:
        report_paths = get_report_paths()
    reads = " ".join(f"Read {p}." for p in report_paths)

    return (
        "You are the Layer 2 synthesis agent. "
        f"Ticker: {ticker}. Trigger: {reason_str}. "
        "Read docs/TRADING_PLAYBOOK.md for trading rules. "
        "If data/trading_insights.md exists, read it for recent performance context. "
        f"{reads} "
        "These are reports from 3 specialist agents (technical, risk, microstructure). "
        "Synthesize their findings into a trading decision for BOTH Patient and Bold strategies. "
        "If specialists disagree, explain why you sided with one over the other. "
        "Read data/portfolio_state.json and data/portfolio_state_bold.json for current positions. "
        "Write journal entry and send Telegram per the playbook."
    )


def get_report_paths() -> list[str]:
    """Get output file paths for all specialists."""
    return [spec["output_file"] for spec in SPECIALISTS.values()]


def launch_specialists(
    ticker: str,
    trigger_reasons: list[str],
) -> list[subprocess.Popen]:
    """Launch all specialist agents in parallel.

    Returns list of Popen processes. Caller must wait for them.
    """
    prompts = build_specialist_prompts(ticker, trigger_reasons)
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        logger.warning("claude not on PATH, cannot launch specialists")
        return []

    procs = []
    agent_env = os.environ.copy()
    agent_env.pop("CLAUDECODE", None)
    agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    agent_env["NODE_OPTIONS"] = "--stack-size=16384"
    # P2 follow-up (Codex P1 #1, 2026-04-17): specialists spawn as headless
    # subprocesses with no interactive stdin, same as invoke_agent. Without
    # this, when multi_agent=true fires, three specialist Claude sessions
    # hit CLAUDE.md's STARTUP CHECK, see unresolved critical errors, and
    # hang asking "How would you like to proceed?" until specialist_timeout_s.
    agent_env["PF_HEADLESS_AGENT"] = "1"

    for name, prompt in prompts.items():
        spec = SPECIALISTS[name]
        # 2026-04-13: DO NOT add `--bare` here either. Same reason as
        # agent_invocation.py: `--bare` disables OAuth/keychain auth and
        # requires ANTHROPIC_API_KEY. User runs Max-subscription OAuth only.
        # Commit 857fd45 (2026-04-01) added `--bare` to specialist launches;
        # removed 2026-04-13 after confirming it broke all specialist runs.
        cmd = [
            claude_cmd, "-p", prompt,
            "--allowedTools", "Read,Write",
            "--max-turns", str(spec["max_turns"]),
        ]
        try:
            log_path = DATA_DIR / f"_specialist_{name}.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=agent_env,
            )
            proc._log_fh = log_fh  # attach for cleanup
            proc._name = name
            procs.append(proc)
            logger.info("Specialist %s launched pid=%s", name, proc.pid)
        except Exception as e:
            logger.error("Failed to launch specialist %s: %s", name, e)

    return procs


def wait_for_specialists(
    procs: list[subprocess.Popen],
    timeout: int = 150,
) -> dict[str, bool]:
    """Wait for all specialist agents to complete.

    Returns dict of specialist_name -> success (True/False).
    """
    results = {}
    deadline = time.time() + timeout

    for proc in procs:
        remaining = max(1, deadline - time.time())
        name = getattr(proc, "_name", "unknown")
        try:
            proc.wait(timeout=remaining)
            success = proc.returncode == 0
            results[name] = success
            if not success:
                logger.warning("Specialist %s exited with code %d", name, proc.returncode)
        except subprocess.TimeoutExpired:
            logger.warning("Specialist %s timed out, killing", name)
            proc.kill()
            proc.wait(timeout=5)
            results[name] = False
        finally:
            log_fh = getattr(proc, "_log_fh", None)
            if log_fh:
                log_fh.close()

        # 2026-04-13: Auth-error scan — specialist log is truncated per run
        # ("w" mode in launch_specialists), so reading the whole file is safe.
        # Override success to False if auth failure detected so synthesis
        # doesn't proceed with an empty specialist report masquerading as OK.
        try:
            log_path = DATA_DIR / f"_specialist_{name}.log"
            if log_path.exists():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if detect_auth_failure(text, caller=f"layer2_specialist_{name}",
                                       context={"specialist": name}):
                    results[name] = False
        except Exception as e:
            logger.warning("Auth-error scan of specialist %s log failed: %s", name, e)

    return results


def cleanup_reports() -> None:
    """Remove specialist report files after synthesis."""
    for spec in SPECIALISTS.values():
        path = BASE_DIR / spec["output_file"]
        if path.exists():
            path.unlink()
    # Also clean up log files
    for name in SPECIALISTS:
        log_path = DATA_DIR / f"_specialist_{name}.log"
        if log_path.exists():
            log_path.unlink()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (520..780)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 822ms:
    # BEFORE _agent_proc / _agent_tier / _agent_reasons are cleared (so
    # the auth-failure record carries the right tier + trigger context).
    # Best-effort: failures are swallowed inside the helper so a busted
    # log read can never break the kill path.
    auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
    _scan_agent_log_for_auth_failure(auth_label)

    # BUG-91: Log the timed-out invocation before returning
    _log_trigger(
        _agent_reasons or fallback_reasons or [],
        "timeout",
        tier=_agent_tier or fallback_tier,
    )

    _agent_proc = None
    return kill_ok


def invoke_agent(reasons, tier=3):
    global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before

    # Check if Layer 2 is auto-disabled due to consecutive stack overflows
    if _consecutive_stack_overflows >= _MAX_STACK_OVERFLOWS:
        logger.info(
            "Layer 2 skipped: auto-disabled after %d consecutive stack overflows",
            _consecutive_stack_overflows,
        )
        _log_trigger(reasons, "skipped_stack_overflow", tier=tier)
        return False

    # Check if Layer 2 is enabled — allows running data loop without Claude quota
    try:
        config = _load_config()
        l2_cfg = config.get("layer2", {})
        if not l2_cfg.get("enabled", True):
            logger.info("Layer 2 disabled (config.layer2.enabled=false), skipping")
            return False
    except Exception as e:
        logger.warning("Failed to load config for layer2 check: %s", e)

    tier_cfg = TIER_CONFIG.get(tier, TIER_CONFIG[3])
    timeout = tier_cfg["timeout"]

    # 2026-05-05: this reentrancy block reads/mutates the same _agent_proc /
    # _agent_log / _agent_start state that the watchdog tick observes via
    # _check_agent_completion_locked. Without the lock, the watchdog could
    # observe a freshly-killed _agent_proc.poll() exit code and write a
    # "failed"/"incomplete" row at the same time _kill_overrun_agent is
    # writing its "timeout" row — exactly the double-log the lock was added
    # to prevent. Hold _completion_lock for the entire read-decide-kill
    # path; _kill_overrun_agent itself does NOT take the lock so this is
    # safe (no reentrant acquire).
    with _completion_lock:
        if _agent_proc and _agent_proc.poll() is None:
            # BUG-203: use monotonic clock for elapsed — wall clock is NTP-jump-prone.
            # P2B (2026-04-17): via _safe_elapsed_s() so a poisoned _agent_start
            # can't cause a negative elapsed that silently skips the timeout.
            elapsed = _safe_elapsed_s()
            if elapsed > _agent_timeout:
                # P1B (2026-04-17): helper so check_agent_completion can share
                # the kill path — see _kill_overrun_agent docstring.
                kill_ok = _kill_overrun_agent(
                    fallback_reasons=reasons, fallback_tier=tier,
                )
                # BUG-92: If kill failed, don't spawn new agent (old one may
                # still be running)
                if not kill_ok:
                    logger.error(
                        "Not spawning new agent — old process may still be running"
                    )
                    return False
            else:
                logger.info(
                    "Agent still running (pid %s, %.0fs), skipping",
                    _agent_proc.pid, elapsed,
                )
                return False

    if _agent_log:
        _agent_log.close()
        _agent_log = None

    try:
        from portfolio.journal import write_context

        n = write_context()
        logger.info("Layer 2 context: %d journal entries", n)
    except Exception as e:
        logger.warning("journal context failed: %s", e)

    # Perception gate: skip low-value invocations
    try:
        from portfolio.perception_gate import should_invoke as _should_invoke
        should, gate_reason = _should_invoke(reasons, tier)
        if not should:
            logger.info("Perception gate skipped: %s", gate_reason)
            _log_trigger(reasons, "skipped_gate", tier=tier)
            return False
    except Exception as e:
        logger.warning("perception gate error (passing through): %s", e)

    # BUG-214: Drawdown circuit breaker — first-ever automated risk gate on
    # the primary trading path. Advisory below _DRAWDOWN_BLOCK_PCT, hard-block
    # above it. Respects user's high risk tolerance (memory/feedback_risk_tolerance.md).
    #
    # 2026-05-02 (adversarial review 05-01 P0-5): the bare `except Exception`
    # used to swallow all errors and proceed. That meant a portfolio in
    # 50%+ drawdown could continue trading if anything in the check threw
    # (ImportError, IO error on portfolio_state.json mid-rename, KeyError
    # on a missing dd dict key). The fail-safe direction for a circuit
    # breaker is BLOCK on failure, not pass.
    #
    # New behavior:
    # - Per-portfolio errors (file read, dict access) are tolerated for THAT
    #   portfolio only — we still check the other portfolio.
    # - A complete failure to even load the check (ImportError) is logged
    #   ERROR + treated as block (fail-safe).
    # - The narrow per-portfolio try/except still tolerates transient I/O,
    #   so a missing portfolio_state.json mid-rename doesn't take the loop
    #   down.
    _drawdown_context = ""
    try:
        from portfolio.risk_management import check_drawdown
    except Exception as e:
        logger.error("DRAWDOWN BLOCK: check_drawdown unavailable (%s) — fail-safe block", e)
        _log_trigger(reasons, "blocked_drawdown_unavailable", tier=tier)
        return False

    for label, pf_path in [("Patient", PATIENT_PORTFOLIO), ("Bold", BOLD_PORTFOLIO)]:
        if not pf_path.exists():
            continue
        try:
            dd = check_drawdown(str(pf_path), max_drawdown_pct=_DRAWDOWN_WARN_PCT)
            if dd["current_drawdown_pct"] > _DRAWDOWN_BLOCK_PCT:
                logger.error(
                    "DRAWDOWN BLOCK: %s portfolio at %.1f%% drawdown (>%.0f%%) — skipping invocation",
                    label, dd["current_drawdown_pct"], _DRAWDOWN_BLOCK_PCT,
                )
                _log_trigger(reasons, f"blocked_drawdown_{label.lower()}", tier=tier)
                return False
            if dd["current_drawdown_pct"] > _DRAWDOWN_WARN_PCT:
                logger.warning(
                    "DRAWDOWN WARNING: %s portfolio at %.1f%% drawdown (peak %.0f, current %.0f SEK)",
                    label, dd["current_drawdown_pct"], dd["peak_value"], dd["current_value"],
                )
            _drawdown_context += (
                f"\n[DRAWDOWN {label}] {dd['current_drawdown_pct']:.1f}% from peak "
                f"(peak={dd['peak_value']:.0f}, current={dd['current_value']:.0f} SEK)"
            )
        except Exception as e:
            # Per-portfolio failure: log ERROR (not WARNING), but tolerate so
            # the OTHER portfolio still gets checked. This keeps a transient IO
            # error on one file from disabling the gate entirely. If BOTH
            # portfolios fail, neither will set the block flag, and the
            # invocation proceeds — by design, since blocking trading on a pure
            # IO race that the next cycle will re-check is too aggressive.
            logger.error(
                "DRAWDOWN check failed for %s portfolio (proceeding for this portfolio only): %s",
                label, e,
            )

    # Adversarial review 05-01 P1-12 (2026-05-02): trade-guards pre-execution gate.
    # `should_block_trade` was implemented in trade_guards.py for ARCH-29 but
    # never imported by production code — only by tests. Wire it here so an
    # invocation triggered by a ticker that is in cooldown for BOTH Patient
    # and Bold gets short-circuited before the multi-agent / subprocess spawn
    # (saves ~600s of T2 subprocess + Claude tokens for a decision that
    # cannot be acted on).
    #
    # Semantics:
    #   1. Pull the trade_guard_warnings already computed by reporting.py and
    #      stored in agent_summary.json.
    #   2. Build _guard_context for the prompt (advisory) — Layer 2 should
    #      see active cooldowns/loss-streaks regardless of the gate decision.
    #   3. Block ONLY when should_block_trade(...) is True AND the trigger
    #      ticker is blocked for BOTH strategies. Single-strategy block
    #      proceeds (the unblocked strategy can still trade).
    #   4. Failure to load warnings (missing agent_summary, IO race) is
    #      fail-OPEN — unlike drawdown, cooldowns are soft constraints and a
    #      single missed gate cycle is not a safety risk.
    _guard_context = ""
    try:
        guard_result = _load_guard_warnings()
    except Exception as e:
        logger.warning("trade-guards load failed (proceeding): %s", e)
        guard_result = {"warnings": [], "summary": "load_failed"}

    if guard_result.get("warnings"):
        _guard_context += f"\n[TRADE GUARDS] {guard_result.get('summary', '')}"
        for w in guard_result["warnings"][:10]:  # cap context size
            sev = w.get("severity", "?")
            tkr = w.get("ticker") or w.get("details", {}).get("ticker", "?")
            strat = w.get("strategy") or w.get("details", {}).get("strategy", "?")
            msg = w.get("message", w.get("guard", "?"))
            _guard_context += f"\n  [{sev.upper()}] {tkr}/{strat}: {msg}"

    try:
        from portfolio.trade_guards import should_block_trade
        if should_block_trade(guard_result):
            # Determine the trigger ticker and check whether BOTH strategies
            # are blocked on it. Anything else (single-strategy block, or
            # block on a different ticker than the trigger) is advisory.
            trigger_ticker = _extract_ticker(reasons)
            blocked_strategies = {
                w.get("strategy") or w.get("details", {}).get("strategy")
                for w in guard_result["warnings"]
                if w.get("severity") == "block"
                and (
                    w.get("ticker") == trigger_ticker
                    or w.get("details", {}).get("ticker") == trigger_ticker
                )
            }
            blocked_strategies.discard(None)
            if {"patient", "bold"}.issubset(blocked_strategies):
                logger.error(
                    "TRADE GUARDS BLOCK: %s in cooldown for BOTH strategies — "
                    "skipping invocation",
                    trigger_ticker,
                )
                _log_trigger(reasons, "blocked_trade_guards", tier=tier)
                return False
    except Exception as e:
        # Import failures or shape mismatches must not derail the invocation.
        logger.warning("trade-guards gate failed (proceeding): %s", e)

    # Multi-agent mode: parallel specialists + synthesis (Coordinator Mode pattern)
    # Enabled via config.layer2.multi_agent = true, only for T2/T3
    try:
        config = _load_config()
        multi_agent = config.get("layer2", {}).get("multi_agent", False)
    except Exception:
        multi_agent = False

    if multi_agent and tier >= 2:
        try:
            from portfolio.multi_agent_layer2 import (
                build_synthesis_prompt,
                launch_specialists,
                wait_for_specialists,
            )
            # Extract primary ticker from reasons
            ticker = _extract_ticker(reasons)
            logger.info("Multi-agent T%d: launching 3 specialists for %s", tier, ticker)
            procs = launch_specialists(ticker, reasons)
            if procs:
                # C3/NEW-1: timeout reduced from 150s to 30s (configurable via
                # layer2.specialist_timeout_s) to avoid blocking the main loop.
                # TODO: run specialists in background thread, collect results async.
                specialist_timeout = config.get("layer2", {}).get("specialist_timeout_s", 30)
                results = wait_for_specialists(procs, timeout=specialist_timeout)
                success_count = sum(1 for v in results.values() if v)
                logger.info("Specialists complete: %d/%d succeeded", success_count, len(results))
                # Even if some fail, proceed with synthesis using available reports
                prompt = build_synthesis_prompt(ticker, reasons)
                # Fall through to normal agent launch with synthesis prompt
            else:
                logger.warning("No specialists launched, falling back to single-agent")
                prompt = _build_tier_prompt(tier, reasons)
        except Exception as e:
            logger.warning("Multi-agent failed (%s), falling back to single-agent", e)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (780..980)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 728ms:
            logger.warning("Multi-agent failed (%s), falling back to single-agent", e)
            prompt = _build_tier_prompt(tier, reasons)
    else:
        prompt = _build_tier_prompt(tier, reasons)

    # BUG-214: Append drawdown context so Layer 2 sees current risk levels.
    if _drawdown_context:
        prompt += "\n\n[RISK DATA]" + _drawdown_context
    # P1-12 (2026-05-02): also surface trade-guard warnings to Layer 2 so
    # it can avoid suggesting actions that the guards would just block in
    # check_overtrading_guards anyway.
    if _guard_context:
        prompt += "\n\n[TRADE GUARDS]" + _guard_context

    # Decision feedback loop (2026-05-02 research): inject recent decisions
    # for the trigger ticker so Layer 2 can see its own track record and
    # calibrate (e.g., "I said SELL at $73 — price is now $75, was I wrong?").
    try:
        feedback_ticker = _extract_ticker(reasons)
        _feedback = _build_decision_feedback(feedback_ticker)
        if _feedback:
            prompt += "\n\n" + _feedback
    except Exception as e:
        logger.debug("decision feedback failed (non-fatal): %s", e)

    max_turns = tier_cfg["max_turns"]

    # Try direct claude invocation first; fall back to bat file for T3
    claude_cmd = shutil.which("claude")
    if claude_cmd:
        # 2026-04-13: DO NOT add `--bare`. It disables OAuth/keychain auth
        # and only accepts ANTHROPIC_API_KEY. This user runs on a Max
        # subscription with no API key, so `--bare` silently breaks every
        # invocation ("Not logged in" to stdout, exit 0). Commit b4bb57d
        # added it on 2026-03-27; removed on 2026-04-13 after 3 weeks of
        # silent Layer 2 failures. See portfolio/claude_gate.py
        # (detect_auth_failure) for the runtime guard.
        cmd = [
            claude_cmd, "-p", prompt,
            "--allowedTools", "Edit,Read,Bash,Write",
            "--max-turns", str(max_turns),
        ]
    else:
        # Fallback: use pf-agent.bat (always Tier 3)
        agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
        if not agent_bat.exists():
            logger.warning("Agent script not found at %s", agent_bat)
            return False
        cmd = ["cmd", "/c", str(agent_bat)]
        logger.info("claude not on PATH, falling back to pf-agent.bat (T3)")

    log_fh = None
    try:
        agent_log_path = DATA_DIR / "agent.log"
        # Capture the current file size BEFORE opening in append mode, so
        # check_agent_completion() can read only this invocation's output
        # (for auth-error detection) and not the entire log history.
        global _agent_log_start_offset
        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
        log_fh = open(agent_log_path, "a", encoding="utf-8")
        # Strip Claude Code session markers to avoid "nested session" error
        # when the parent process tree has Claude Code running
        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)
        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        # Increase Node.js stack size to prevent stack overflow in Claude CLI
        agent_env["NODE_OPTIONS"] = "--stack-size=16384"
        # P2 (2026-04-17): mark this subprocess as headless so CLAUDE.md's
        # STARTUP CHECK protocol doesn't ask "How would you like to proceed?"
        # when it finds unresolved critical_errors.jsonl entries. The agent
        # has no stdin (pipe only), so any prompt that blocks on user input
        # makes it hit the tier timeout with zero work done. The CLAUDE.md
        # conditional turns that into "log the unresolved entries in your
        # journal entry and proceed with the trigger task".
        agent_env["PF_HEADLESS_AGENT"] = "1"
        _agent_proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=agent_env,
        )
        _agent_log = log_fh  # transfer ownership on success
        log_fh = None  # prevent cleanup below from closing it
        _agent_start = time.monotonic()
        _agent_start_wall = time.time()  # wall-clock fallback for P2B
        _agent_timeout = timeout
        _agent_tier = tier
        _agent_reasons = list(reasons)
        _journal_ts_before = _safe_last_jsonl_ts(JOURNAL_FILE, "journal")
        _telegram_ts_before = _safe_last_jsonl_ts(TELEGRAM_FILE, "telegram")
        # BUG-219: Snapshot transaction counts so check_agent_completion()
        # can detect new trades and call record_trade().
        global _patient_txn_count_before, _bold_txn_count_before
        try:
            from portfolio.file_utils import load_json
            _patient_txn_count_before = len(
                (load_json(PATIENT_PORTFOLIO, default={}) or {}).get("transactions", [])
            )
            _bold_txn_count_before = len(
                (load_json(BOLD_PORTFOLIO, default={}) or {}).get("transactions", [])
            )
        except Exception:
            _patient_txn_count_before = 0
            _bold_txn_count_before = 0
        # 2026-04-17: Publish the tier into health_state so loop_contract
        # can pick the right per-tier grace window for the journal-activity
        # check. Without this, the contract defaults to T3 grace (20m),
        # which is conservative but can delay detection when an all-T1
        # cadence runs silent. See loop_contract._get_layer2_grace_s() for
        # the consumer and LAYER2_JOURNAL_GRACE_S_BY_TIER for the table.
        # Best-effort: never fail the invocation because health_state is
        # unwriteable (atomic_write_json handles the happy path; any
        # exception is logged and swallowed).
        try:
            from portfolio.file_utils import atomic_write_json, load_json
            # 2026-04-17 Codex P2: when claude is missing from PATH we fall
            # back to pf-agent.bat which is unconditionally T3 regardless of
            # the requested tier. Record the *effective* tier so the
            # per-tier grace window in loop_contract reflects what's
            # actually running.
            effective_tier = 3 if not claude_cmd else tier
            health_path = DATA_DIR / "health_state.json"
            health = load_json(health_path, default={}) or {}
            health["last_invocation_tier"] = effective_tier
            health["last_invocation_tier_ts"] = datetime.now(UTC).isoformat()
            atomic_write_json(health_path, health)
        except Exception as e:
            logger.warning("health_state tier publish failed: %s", e)
        logger.info(
            "Agent T%d invoked pid=%s max_turns=%d timeout=%ds (%s)",
            tier, _agent_proc.pid, max_turns, timeout,
            ", ".join(reasons[:3]),
        )
        # 2026-05-05: arm the completion watchdog so the wall-clock
        # timeout fires within ~30 s of the real budget even when the
        # main loop's run() cycle bloats. See module-level note at
        # _COMPLETION_WATCHDOG_INTERVAL_S.
        _ensure_completion_watchdog()
        # Save Layer 2 invocation notification (save-only, not sent to Telegram)
        try:
            config = _load_config()
            reason_str = escape_markdown_v1(", ".join(reasons[:3]))
            if len(reasons) > 3:
                reason_str += f" (+{len(reasons) - 3} more)"
            tier_label = tier_cfg["label"]
            notify_msg = f"_Layer 2 T{tier} ({tier_label}): {reason_str}_"
            send_or_store(notify_msg, config, category="invocation")
        except Exception as e:
            logger.warning("invocation notification failed: %s", e)
        return True
    except Exception as e:
        logger.error("invoking agent: %s", e)
        if log_fh is not None:
            log_fh.close()
        return False


def _write_fishing_context(journal_entry):
    """Extract fishing context from Layer 2 journal entry.

    Called after Layer 2 completes. Creates a structured context file
    that the fish engine reads as its strongest tactic vote.
    """
    try:
        tickers = journal_entry.get('tickers', {})
        xag = tickers.get('XAG-USD')
        if not xag:
            return

        outlook = xag.get('outlook', '')
        conviction = float(xag.get('conviction', 0))
        levels = xag.get('levels', [])
        thesis = xag.get('thesis', '')

        # Determine direction bias
        if outlook == 'bullish' and conviction >= 0.4:
            direction_bias = 'bullish'
            tactic_vote = 'LONG'
            allow_long = True
            allow_short = conviction < 0.6  # block short only if very bullish
        elif outlook == 'bearish' and conviction >= 0.4:
            direction_bias = 'bearish'
            tactic_vote = 'SHORT'
            allow_long = conviction < 0.6
            allow_short = True
        else:
            direction_bias = 'neutral'
            tactic_vote = None
            allow_long = True
            allow_short = True

        # Check for event context from watchlist
        watchlist = journal_entry.get('watchlist', [])
        event_context = ''
        for item in watchlist:
            if isinstance(item, str) and any(
                w in item.lower() for w in ['event', 'fomc', 'cpi', 'tariff', 'opec']
            ):
                event_context = item[:100]
                break

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (980..1160)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 711ms:
                break

        # Determine position size multiplier from regime
        regime = journal_entry.get('regime', 'ranging')
        if regime == 'high-vol':
            position_size_multiplier = 0.5
        elif regime in ('trending-up', 'trending-down'):
            position_size_multiplier = 1.0
        else:
            position_size_multiplier = 0.75  # ranging = slightly reduced

        context = {
            'timestamp': journal_entry.get('ts', ''),
            'valid_until': '',  # fish engine uses 4h staleness check
            'ticker': 'XAG-USD',
            'direction_bias': direction_bias,
            'bias_confidence': conviction,
            'bias_reasoning': thesis[:200] if thesis else '',
            'allow_long': allow_long,
            'allow_short': allow_short,
            'max_hold_minutes': 120,
            'position_size_multiplier': position_size_multiplier,
            'allow_overnight': conviction >= 0.6 and outlook == 'bullish',
            'event_context': event_context,
            'bull_case': '',
            'bear_case': '',
            'journal_action': '',
            'journal_confidence': conviction,
            'tactic_vote': tactic_vote,
            'tactic_weight': 2.0,
            'levels': levels,
        }

        # Extract bull/bear cases from decisions
        decisions = journal_entry.get('decisions', {})
        for strategy in ('patient', 'bold'):
            dec = decisions.get(strategy, {})
            reasoning = dec.get('reasoning', '')
            action = dec.get('action', 'HOLD')
            if action != 'HOLD':
                context['journal_action'] = action
            if reasoning:
                if not context['bull_case'] and 'bullish' in reasoning.lower():
                    context['bull_case'] = reasoning[:150]
                elif not context['bear_case'] and (
                    'bearish' in reasoning.lower() or 'sell' in reasoning.lower()
                ):
                    context['bear_case'] = reasoning[:150]

        from portfolio.file_utils import atomic_write_json

        # H22/NEW-3: use DATA_DIR absolute path instead of relative 'data/...'
        atomic_write_json(DATA_DIR / 'fishing_context.json', context)

    except Exception as e:
        logger.warning('Fishing context error: %s', e)
        # BUG-181: Write neutral context on failure to prevent stale bias
        try:
            from datetime import UTC, datetime

            from portfolio.file_utils import atomic_write_json
            atomic_write_json(DATA_DIR / 'fishing_context.json', {
                'timestamp': datetime.now(UTC).isoformat(),
                'ticker': 'XAG-USD',
                'direction_bias': 'neutral',
                'bias_confidence': 0.0,
                'bias_reasoning': f'Context extraction failed: {e}',
                'allow_long': True,
                'allow_short': True,
                'tactic_vote': None,
                'tactic_weight': 0.0,
            })
        except Exception:
            logger.warning("Failed to write neutral journal entry", exc_info=True)


def _record_new_trades():
    """BUG-219 / PR-R4-4: Check for new transactions since invoke_agent()
    and call record_trade() for each, activating overtrading prevention.

    Never raises — all errors are logged and swallowed so the completion
    path is never broken by guard bookkeeping failures.
    """
    try:
        from portfolio.file_utils import load_json
        from portfolio.trade_guards import record_trade

        for strategy, pf_path, count_before in [
            ("patient", PATIENT_PORTFOLIO, _patient_txn_count_before),
            ("bold", BOLD_PORTFOLIO, _bold_txn_count_before),
        ]:
            state = load_json(pf_path, default={}) or {}
            txns = state.get("transactions", [])
            if len(txns) <= count_before:
                continue
            # New transactions appeared — record each for guard tracking
            new_txns = txns[count_before:]
            for txn in new_txns:
                ticker = txn.get("ticker")
                direction = txn.get("action")
                if not ticker or direction not in ("BUY", "SELL"):
                    continue
                pnl_pct = txn.get("pnl_pct")
                record_trade(ticker, direction, strategy, pnl_pct=pnl_pct)
                logger.info(
                    "BUG-219: recorded %s %s %s pnl=%.2f%% for overtrading guards",
                    strategy, direction, ticker, pnl_pct or 0.0,
                )
    except Exception as e:
        logger.warning("BUG-219: record_trade wiring failed: %s", e)


def check_agent_completion():
    """Check if a running agent has completed and log completion info.

    Thread-safe: serialised by ``_completion_lock`` so the main-loop call
    site (``portfolio.main.run``) and the 30 s daemon watchdog
    (``_completion_watchdog``) cannot race on ``_agent_proc`` /
    ``_agent_start`` state. Both call paths share the same lock; whichever
    reaches the lock first observes the completion and writes the
    invocations.jsonl row, the other returns ``None`` because
    ``_agent_proc`` is cleared at the end of the handler.

    Returns:
        dict with the following keys (None if no agent is running or the
        agent is still in progress and under its timeout):

        * ``status`` — "success", "incomplete", "failed", "auth_error",
          "timeout" (P1B, 2026-04-17), or "stack_overflow"
        * ``exit_code`` — int or None (None on timeout-kill path)
        * ``duration_s`` — float, always >= 0 (P2B clamp)
        * ``tier`` — int, the tier of the completed agent
        * ``reasons`` — list[str], the triggers for this invocation
        * ``journal_written`` — bool
        * ``telegram_sent`` — bool
        * ``completed_at`` — ISO-8601 UTC timestamp
    """
    with _completion_lock:
        return _check_agent_completion_locked()


def _check_agent_completion_locked():
    """Body of ``check_agent_completion``. The caller MUST hold
    ``_completion_lock``. Split out so the watchdog tick can call into
    the same code path without re-acquiring the lock recursively.
    """
    global _agent_proc, _agent_log, _agent_start, _agent_start_wall
    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before

    if _agent_proc is None:
        return None

    exit_code = _agent_proc.poll()
    if exit_code is None:
        # Still running. P1B (2026-04-17): enforce the wall-clock timeout
        # here too — the lazy check in try_invoke_agent only fires when a
        # new trigger arrives, so a hung agent could run indefinitely if
        # no new triggers came through (yesterday: T1 timeout=120s ran
        # 603s). Share the same kill helper used by try_invoke_agent to
        # keep kill semantics identical.
        elapsed = _safe_elapsed_s()
        if _agent_timeout and elapsed > _agent_timeout:
            killed_tier = _agent_tier
            killed_reasons = list(_agent_reasons or [])
            _kill_overrun_agent()
            return {
                "status": "timeout",
                "exit_code": None,
                "duration_s": round(elapsed, 1),
                "tier": killed_tier,
                "reasons": killed_reasons,
                "journal_written": False,
                "telegram_sent": False,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        return None

    # Process has finished — collect completion info.
    # P2B (2026-04-17): via _safe_elapsed_s() so a poisoned _agent_start
    # can't produce the negative duration_s seen in yesterday's 13:45:45
    # auth_failure entry (-1776254571.5, matching time.monotonic() - time.time()).

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (1160..1340)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 615ms:
    # auth_failure entry (-1776254571.5, matching time.monotonic() - time.time()).
    duration_s = round(_safe_elapsed_s(), 1)
    completed_at = datetime.now(UTC).isoformat()

    # BUG-97: _last_jsonl_ts can raise OSError if file is locked on Windows
    try:
        journal_ts_after = _last_jsonl_ts(JOURNAL_FILE)
    except Exception:
        logger.warning("Failed to read journal timestamp after agent completion")
        journal_ts_after = None
    journal_written = (
        _journal_ts_before is not None
        and journal_ts_after is not None
        and journal_ts_after != _journal_ts_before
    )

    # BUG-97: Same protection for telegram file
    try:
        telegram_ts_after = _last_jsonl_ts(TELEGRAM_FILE)
    except Exception:
        logger.warning("Failed to read telegram timestamp after agent completion")
        telegram_ts_after = None
    telegram_sent = (
        _telegram_ts_before is not None
        and telegram_ts_after is not None
        and telegram_ts_after != _telegram_ts_before
    )

    # Without a baseline from invoke_agent(), stay conservative and do not infer
    # success from pre-existing files in the workspace.
    if _journal_ts_before is None:
        journal_written = False
    if _telegram_ts_before is None:
        telegram_sent = False

    # 2026-04-13: Scan agent.log for auth-error markers (see claude_gate.py
    # detect_auth_failure). Claude CLI can exit 0 while printing "Not logged
    # in" to stdout — that's exactly the 3-week silent Layer 2 outage that
    # motivated this detection. We captured _agent_log_start_offset before
    # spawning the subprocess, so we only scan output from this invocation.
    #
    # P1-3 (2026-05-02 last-followups): scan logic extracted to
    # ``_scan_agent_log_for_auth_failure`` so the timeout-kill path
    # (``_kill_overrun_agent``) can share the exact same semantics. Without
    # the helper, fixing one path and forgetting the other would re-open
    # the same asymmetry the timeout path used to have.
    auth_error_detected = _scan_agent_log_for_auth_failure(
        f"layer2_t{_agent_tier}",
        extra_context={"exit_code": exit_code, "duration_s": duration_s},
    )

    # Determine status
    if auth_error_detected:
        status = "auth_error"
    elif exit_code != 0:
        status = "failed"
    elif journal_written and telegram_sent:
        status = "success"
    else:
        status = "incomplete"

    result = {
        "status": status,
        "exit_code": exit_code,
        "duration_s": duration_s,
        "tier": _agent_tier,
        # Codex P2 #3 follow-up (2026-04-17): include `reasons` so the
        # completion-path and timeout-path dicts have symmetric shape.
        # Callers that dispatch on reasons shouldn't need to know which
        # path produced the dict.
        "reasons": list(_agent_reasons or []),
        "completed_at": completed_at,
        "journal_written": journal_written,
        "telegram_sent": telegram_sent,
    }

    # Log to invocations file
    log_entry = {
        "ts": completed_at,
        "reasons": _agent_reasons or [],
        "status": status,
        "tier": _agent_tier,
        "exit_code": exit_code,
        "duration_s": duration_s,
        "journal_written": journal_written,
        "telegram_sent": telegram_sent,
    }
    try:
        atomic_append_jsonl(INVOCATIONS_FILE, log_entry)
    except Exception as e:
        logger.warning("Failed to log agent completion: %s", e)

    # Post-process: extract fishing context from journal for metals fish engine
    if journal_written:
        with suppress(Exception):
            new_journal_entry = last_jsonl_entry(JOURNAL_FILE)
            if new_journal_entry:
                _write_fishing_context(new_journal_entry)

    # BUG-219 / PR-R4-4: Wire record_trade() into production.
    # After a successful agent run, check if new transactions appeared in
    # either portfolio and record them for overtrading prevention guards
    # (cooldowns, loss escalation, position rate limits).
    _record_new_trades()

    logger.info(
        "Agent completed: status=%s exit=%d duration=%.1fs tier=%s journal=%s telegram=%s",
        status, exit_code, duration_s, _agent_tier, journal_written, telegram_sent,
    )

    # Telegram alert on any agent failure (not just stack overflow)
    if status == "failed":
        try:
            config = _load_config()
            send_or_store(
                f"*L2 FAILED* T{_agent_tier} exit={exit_code} "
                f"({duration_s:.0f}s) journal={journal_written} tg={telegram_sent}",
                config, category="error",
            )
        except Exception as e:
            logger.warning("Agent failure alert failed: %s", e)

    # Track consecutive stack overflow crashes
    global _consecutive_stack_overflows
    if exit_code == _STACK_OVERFLOW_EXIT_CODE:
        _consecutive_stack_overflows += 1
        _save_stack_overflow_counter(_consecutive_stack_overflows)
        logger.error(
            "Claude CLI stack overflow (exit %d), %d consecutive. "
            "Check project root for problematic files or update Claude Code.",
            exit_code, _consecutive_stack_overflows,
        )
        if _consecutive_stack_overflows == _MAX_STACK_OVERFLOWS:
            logger.error(
                "Layer 2 auto-disabled after %d consecutive stack overflows",
                _MAX_STACK_OVERFLOWS,
            )
            try:
                config = _load_config()
                send_or_store(
                    f"*ALERT* Layer 2 auto-disabled after {_MAX_STACK_OVERFLOWS} "
                    f"consecutive stack overflows (exit {exit_code}). "
                    "Claude CLI is crashing — investigate project root.",
                    config, category="alert",
                )
            except Exception as e:
                logger.warning("Stack overflow alert failed: %s", e)
    else:
        # BUG-95: Reset counter on any non-stack-overflow completion (success or otherwise).
        # This prevents false positive auto-disable when the consecutive chain is broken.
        if _consecutive_stack_overflows > 0:
            _consecutive_stack_overflows = 0
            _save_stack_overflow_counter(0)

    # Clean up
    if _agent_log:
        try:
            _agent_log.close()
        except Exception as e:
            logger.warning("Agent log close failed: %s", e)
    _agent_proc = None
    _agent_log = None
    _agent_start = 0
    _agent_start_wall = 0.0
    _agent_tier = None
    _agent_reasons = None
    _journal_ts_before = None
    _telegram_ts_before = None
    _patient_txn_count_before = 0
    _bold_txn_count_before = 0

    return result


def get_completion_stats(hours=24):
    """Compute rolling completion stats from the invocations log.

    Args:
        hours: Number of hours to look back (default 24).

    Returns:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (1340..1410)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 602ms:
    Returns:
        dict with keys: total, success, incomplete, failed, timeout,
        auth_error, completion_rate.  Returns zeroed stats if no data is
        available.

    Codex P2 #4 follow-up (2026-04-17): "timeout" and "auth_error" were
    being dropped entirely by the status filter. Before P1B, timeouts
    only fired when a new trigger arrived, so they were rare. After
    P1B check_agent_completion enforces timeout every cycle — these
    are now meaningful failure categories that belong in the health
    rollup. Added as distinct buckets to preserve history and keep
    completion_rate honest (timeouts count as failures for rate).
    """
    entries = load_jsonl(INVOCATIONS_FILE)
    cutoff = datetime.now(UTC).timestamp() - (hours * 3600)

    total = 0
    success = 0
    incomplete = 0
    failed = 0
    timeout = 0
    auth_error = 0

    tracked_statuses = ("success", "incomplete", "failed", "timeout", "auth_error")
    for entry in entries:
        entry_status = entry.get("status", "")
        if entry_status not in tracked_statuses:
            continue

        ts_str = entry.get("ts", "")
        if not ts_str:
            continue

        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            entry_ts = dt.timestamp()
        except (ValueError, TypeError):
            continue

        if entry_ts < cutoff:
            continue

        total += 1
        if entry_status == "success":
            success += 1
        elif entry_status == "incomplete":
            incomplete += 1
        elif entry_status == "failed":
            failed += 1
        elif entry_status == "timeout":
            timeout += 1
        elif entry_status == "auth_error":
            auth_error += 1

    completion_rate = (success / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "success": success,
        "incomplete": incomplete,
        "failed": failed,
        "timeout": timeout,
        "auth_error": auth_error,
        "completion_rate": round(completion_rate, 1),
    }

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (430..520)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 587ms:
            "tier": _agent_tier,
            "reasons": (_agent_reasons or [])[:5],
        }
        if extra_context:
            ctx.update(extra_context)
        return detect_auth_failure(
            new_output,
            caller=label,
            context=ctx,
        )
    except Exception as e:
        logger.warning("Auth-error scan of agent.log failed (%s): %s", label, e)
        return False


def _kill_overrun_agent(fallback_reasons=None, fallback_tier=None):
    """Kill the running _agent_proc and clear module state.

    P1B (2026-04-17): extracted from ``try_invoke_agent`` so it can also
    be called from ``check_agent_completion``. Previously the timeout
    check lived only inside try_invoke_agent, meaning a hung agent could
    run indefinitely if no new triggers fired (yesterday evidence: T1
    invoked 16:04:58 with timeout=120s completed at 16:15:01 = 603s).

    Logs the trigger with status="timeout" and clears ``_agent_proc`` /
    ``_agent_log`` on the way out.

    P1-3 (2026-05-02 last-followups): also scans the captured agent.log
    slice for claude-CLI auth-error markers BEFORE clearing module state,
    so the silent-auth-failure detector covers the timeout path too — not
    just the happy completion path. See ``_scan_agent_log_for_auth_failure``
    for full rationale.

    Args:
        fallback_reasons: Reason list to use for the trigger log entry if
            ``_agent_reasons`` is empty (caller context for the missing
            _reasons.).
        fallback_tier: Tier to log if ``_agent_tier`` is None.

    Returns:
        bool: True if the kill succeeded (or the process had already
        exited). False if the kill command itself failed — caller must
        NOT spawn a replacement in that case because the old process
        may still be holding resources.
    """
    global _agent_proc, _agent_log

    if _agent_proc is None:
        return True

    pid = _agent_proc.pid
    elapsed = _safe_elapsed_s()
    logger.info("Agent pid=%s timed out (%.0fs), killing", pid, elapsed)

    kill_ok = True
    if platform.system() == "Windows":
        # BUG-92: Check taskkill return code to detect kill failure
        # BUG-189: rc=128 means process already exited — treat as success
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
        )
        if result.returncode not in (0, 128):
            logger.error(
                "taskkill failed (rc=%d): %s",
                result.returncode,
                result.stderr.decode(errors="replace").strip(),
            )
            kill_ok = False
        elif result.returncode == 128:
            logger.info("Agent pid=%s already exited (rc=128)", pid)
    else:
        _agent_proc.kill()
    try:
        _agent_proc.wait(timeout=15)  # BUG-189: 15s for Claude CLI Node.js teardown
    except subprocess.TimeoutExpired:
        if kill_ok:
            logger.error("Agent pid=%s did not exit after kill+15s wait", pid)
        kill_ok = False

    if _agent_log:
        try:
            _agent_log.close()
        except Exception as e:
            logger.warning("Agent log close failed: %s", e)
        _agent_log = None

    # P1-3 (2026-05-02 last-followups): scan the captured agent.log slice
    # for auth-error markers before forgetting the dead subprocess. Done
    # AFTER closing _agent_log (so any buffered output is flushed) but
    # BEFORE _agent_proc / _agent_tier / _agent_reasons are cleared (so

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (520..560)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 647ms:
    # BEFORE _agent_proc / _agent_tier / _agent_reasons are cleared (so
    # the auth-failure record carries the right tier + trigger context).
    # Best-effort: failures are swallowed inside the helper so a busted
    # log read can never break the kill path.
    auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
    _scan_agent_log_for_auth_failure(auth_label)

    # BUG-91: Log the timed-out invocation before returning
    _log_trigger(
        _agent_reasons or fallback_reasons or [],
        "timeout",
        tier=_agent_tier or fallback_tier,
    )

    _agent_proc = None
    return kill_ok


def invoke_agent(reasons, tier=3):
    global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before

    # Check if Layer 2 is auto-disabled due to consecutive stack overflows
    if _consecutive_stack_overflows >= _MAX_STACK_OVERFLOWS:
        logger.info(
            "Layer 2 skipped: auto-disabled after %d consecutive stack overflows",
            _consecutive_stack_overflows,
        )
        _log_trigger(reasons, "skipped_stack_overflow", tier=tier)
        return False

    # Check if Layer 2 is enabled — allows running data loop without Claude quota
    try:
        config = _load_config()
        l2_cfg = config.get("layer2", {})
        if not l2_cfg.get("enabled", True):
            logger.info("Layer 2 disabled (config.layer2.enabled=false), skipping")
            return False
    except Exception as e:
        logger.warning("Failed to load config for layer2 check: %s", e)


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\agent_invocation.py | Select-Object -Index (230..270)" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 680ms:
    for r in reasons:
        # Match patterns like XAG-USD, BTC-USD, ETH-USD
        m = re.search(r'\b([A-Z]{2,5}-USD)\b', r)
        if m:
            return m.group(1)
        # Match stock tickers like NVDA, PLTR
        m = re.search(r'\b([A-Z]{2,5})\b(?:\s+flipped|\s+crossed|\s+broke)', r)
        if m:
            return m.group(1)
    return "XAG-USD"  # default to silver


def _log_trigger(reasons, status, tier=None):
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "reasons": reasons,
        "status": status,
    }
    if tier is not None:
        entry["tier"] = tier
    atomic_append_jsonl(INVOCATIONS_FILE, entry)


def _load_guard_warnings():
    """Read trade_guard_warnings from agent_summary.json.

    P1-12 (2026-05-02): the trade-guards pre-execution gate consumes the
    warnings already computed by reporting.py and stored in agent_summary.
    Reading them here (rather than recomputing) keeps the gate consistent
    with what Layer 2 sees in its prompt context, and is much cheaper.

    Returns a dict shaped like trade_guards.get_all_guard_warnings():
        {"warnings": [...], "summary": "..."}
    Defaults to empty/no_summary when agent_summary is missing or has
    no trade_guard_warnings field — caller treats that as "no blocks".
    """
    from portfolio.file_utils import load_json
    summary_path = DATA_DIR / "agent_summary.json"
    summary = load_json(summary_path, default=None)
    if not isinstance(summary, dict):
        return {"warnings": [], "summary": "no_summary"}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
files=['portfolio/main.py','portfolio/agent_invocation.py','portfolio/trigger.py','portfolio/reflection.py','portfolio/session_calendar.py']
for f in files:
    print('FILE', f)
    for i,line in enumerate(Path(f).read_text(encoding='utf-8').splitlines(), start=1):
        if f.endswith('main.py') and i in range(29,33):
            print(i, line)
        if f.endswith('agent_invocation.py') and i in range(224,241):
            print(i, line)
        if f.endswith('trigger.py') and i in range(210,232):
            print(i, line)
        if f.endswith('reflection.py') and i in range(57,71):
            print(i, line)
        if f.endswith('session_calendar.py') and i in range(167,177):
            print(i, line)
    print()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 659ms:
FILE portfolio/main.py
29 from pathlib import Path
30 
31 from portfolio.file_utils import atomic_write_json, atomic_write_text, load_json
32 

FILE portfolio/agent_invocation.py
224 def _extract_ticker(reasons):
225     """Extract the primary ticker from trigger reasons.
226 
227     Looks for common ticker patterns like 'XAG-USD', 'BTC-USD', 'NVDA'.
228     Falls back to 'XAG-USD' if no ticker found.
229     """
230     import re
231     for r in reasons:
232         # Match patterns like XAG-USD, BTC-USD, ETH-USD
233         m = re.search(r'\b([A-Z]{2,5}-USD)\b', r)
234         if m:
235             return m.group(1)
236         # Match stock tickers like NVDA, PLTR
237         m = re.search(r'\b([A-Z]{2,5})\b(?:\s+flipped|\s+crossed|\s+broke)', r)
238         if m:
239             return m.group(1)
240     return "XAG-USD"  # default to silver

FILE portfolio/trigger.py
210     #    to actually fire the trigger. Prevents 20+ HOLD invocations per day.
211     triggered_consensus = state.get("triggered_consensus", {})
212     for ticker, sig in signals.items():
213         action = sig["action"]
214         last_tc = triggered_consensus.get(ticker, "HOLD")
215         if action in ("BUY", "SELL") and last_tc == "HOLD":
216             conf = sig.get("confidence", 0)
217             # Ranging regime dampening: skip low-confidence consensus triggers
218             ticker_regime = (sig.get("extra") or {}).get("_regime", "unknown")
219             if (
220                 ticker_regime == "ranging"
221                 and RANGING_CONSENSUS_MIN_CONFIDENCE > 0
222                 and conf < RANGING_CONSENSUS_MIN_CONFIDENCE
223             ):
224                 logger.info(
225                     "Ranging dampening: %s consensus %s (%.0f%%) suppressed "
226                     "(min %.0f%%)",
227                     ticker, action, conf * 100,
228                     RANGING_CONSENSUS_MIN_CONFIDENCE * 100,
229                 )
230                 # Still update baseline so we don't re-trigger next cycle
231                 triggered_consensus[ticker] = action

FILE portfolio/reflection.py
57     sells = []  # list of (ticker, pnl_pct)
58 
59     for tx in txns:
60         ticker = tx.get("ticker", "")
61         action = tx.get("action", "")
62         shares = tx.get("shares", 0)
63         price = tx.get("price_sek", 0)
64 
65         if action == "BUY" and price > 0:
66             buys.setdefault(ticker, []).append((price, shares))
67         elif action == "SELL" and price > 0:
68             # Compute PnL against avg cost of prior buys
69             buy_list = buys.get(ticker, [])
70             if buy_list:

FILE portfolio/session_calendar.py
167             extended_end=None,
168             remaining_minutes=remaining,
169             is_open=is_open,
170             is_extended=False,
171             underlying_open=underlying_open,
172             phase=phase,
173         )
174 
175     # EU-based instruments (warrants, Stockholm stocks)
176     spec = SESSIONS.get(instrument_type, SESSIONS["warrant"])


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
f='portfolio/session_calendar.py'
for i,line in enumerate(Path(f).read_text(encoding='utf-8').splitlines(), start=1):
    if i in range(175,186):
        print(i, line)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 625ms:
175     # EU-based instruments (warrants, Stockholm stocks)
176     spec = SESSIONS.get(instrument_type, SESSIONS["warrant"])
177     oh, om = spec["open_cet"]
178     ch, cm = spec["close_cet"]
179 
180     session_end = _make_session_end(now, ch, cm)
181     session_open = _make_session_end(now, oh, om)
182 
183     is_weekday = now.weekday() < 5
184     is_open = is_weekday and session_open <= now < session_end
185 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'python -m portfolio.main' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-08T15:31:51.861099Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\main.py", line 31, in <module>
    from portfolio.file_utils import atomic_write_json, atomic_write_text, load_json
ModuleNotFoundError: No module named 'portfolio.file_utils'

 exited 1 in 690ms:
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\main.py", line 31, in <module>
    from portfolio.file_utils import atomic_write_json, atomic_write_text, load_json
ModuleNotFoundError: No module named 'portfolio.file_utils'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from datetime import UTC, datetime
from portfolio.session_calendar import get_session_info
for dt in [datetime(2026,12,24,12,0,tzinfo=UTC), datetime(2026,6,19,12,0,tzinfo=UTC)]:
    info=get_session_info('warrant', now=dt)
    print(dt.isoformat(), info.is_open, info.phase, info.remaining_minutes)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 649ms:
2026-12-24T12:00:00+00:00 True open 535.0
2026-06-19T12:00:00+00:00 True open 475.0

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types, importlib
# stub file_utils
m=types.ModuleType('portfolio.file_utils')
_state={}
def load_json(path, default=None):
    return _state.copy()
def atomic_write_json(path, data):
    _state.clear(); _state.update(data)
m.load_json=load_json
m.atomic_write_json=atomic_write_json
sys.modules['portfolio.file_utils']=m
trigger=importlib.import_module('portfolio.trigger')
# reset globals/state
_state.clear(); trigger._startup_grace_active=False
# initial baseline as HOLD and prior state exists
_state.update({'triggered_consensus':{'MSTR':'HOLD'}, 'last':{'signals':{}, 'prices':{}, 'fear_greeds':{}, 'sentiments':{}}})
signals={'MSTR':{'action':'BUY','confidence':0.2,'extra':{'_regime':'ranging'}}}
print(trigger.check_triggers(signals, {}, {}, {}))
print('state after low conf', _state['triggered_consensus'])
signals={'MSTR':{'action':'BUY','confidence':0.8,'extra':{'_regime':'ranging'}}}
print(trigger.check_triggers(signals, {}, {}, {}))
print('state after high conf', _state['triggered_consensus'])
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 670ms:
(False, [])
state after low conf {'MSTR': 'BUY'}
(False, [])
state after high conf {'MSTR': 'BUY'}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types, importlib
# minimal stubs for module imports
for name in ['portfolio.api_utils','portfolio.file_utils','portfolio.message_store','portfolio.telegram_notifications','portfolio.claude_gate']:
    if name not in sys.modules:
        m=types.ModuleType(name)
        sys.modules[name]=m
sys.modules['portfolio.api_utils'].load_config=lambda: {}
sys.modules['portfolio.claude_gate'].detect_auth_failure=lambda *a, **k: False
sys.modules['portfolio.file_utils'].atomic_append_jsonl=lambda *a, **k: None
sys.modules['portfolio.file_utils'].last_jsonl_entry=lambda *a, **k: None
sys.modules['portfolio.file_utils'].load_jsonl=lambda *a, **k: []
sys.modules['portfolio.message_store'].send_or_store=lambda *a, **k: None
sys.modules['portfolio.telegram_notifications'].escape_markdown_v1=lambda s: s
mod=importlib.import_module('portfolio.agent_invocation')
for reasons in [
    ['MSTR consensus BUY (55%)'],
    ['MSTR moved 3.0% up'],
    ['MSTR flipped HOLD->BUY (sustained)'],
]:
    print(reasons[0], '=>', mod._extract_ticker(reasons))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-08T15:32:16.573435Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\agent_invocation.py", line 171, in <module>
    _consecutive_stack_overflows = _load_stack_overflow_counter()
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\agent_invocation.py", line 155, in _load_stack_overflow_counter
    from portfolio.file_utils import load_json
ImportError: cannot import name 'load_json' from 'portfolio.file_utils' (unknown location). Did you mean: 'load_jsonl'?

 exited 1 in 636ms:
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\agent_invocation.py", line 171, in <module>
    _consecutive_stack_overflows = _load_stack_overflow_counter()
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\agent_invocation.py", line 155, in _load_stack_overflow_counter
    from portfolio.file_utils import load_json
ImportError: cannot import name 'load_json' from 'portfolio.file_utils' (unknown location). Did you mean: 'load_jsonl'?

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types, importlib
stubs = {}
for name in ['portfolio.api_utils','portfolio.file_utils','portfolio.message_store','portfolio.telegram_notifications','portfolio.claude_gate']:
    m=types.ModuleType(name)
    sys.modules[name]=m
sys.modules['portfolio.api_utils'].load_config=lambda: {}
fu=sys.modules['portfolio.file_utils']
fu.atomic_append_jsonl=lambda *a, **k: None
fu.last_jsonl_entry=lambda *a, **k: None
fu.load_jsonl=lambda *a, **k: []
fu.load_json=lambda *a, **k: {}
fu.atomic_write_json=lambda *a, **k: None
sys.modules['portfolio.message_store'].send_or_store=lambda *a, **k: None
sys.modules['portfolio.telegram_notifications'].escape_markdown_v1=lambda s: s
sys.modules['portfolio.claude_gate'].detect_auth_failure=lambda *a, **k: False
mod=importlib.import_module('portfolio.agent_invocation')
for reasons in [
    ['MSTR consensus BUY (55%)'],
    ['MSTR moved 3.0% up'],
    ['MSTR flipped HOLD->BUY (sustained)'],
    ['BTC-USD consensus BUY (55%)'],
]:
    print(reasons[0], '=>', mod._extract_ticker(reasons))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 608ms:
MSTR consensus BUY (55%) => XAG-USD
MSTR moved 3.0% up => XAG-USD
MSTR flipped HOLD->BUY (sustained) => MSTR
BTC-USD consensus BUY (55%) => BTC-USD

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types, importlib
m=types.ModuleType('portfolio.file_utils')
m.atomic_append_jsonl=lambda *a, **k: None
m.load_json=lambda *a, **k: {}
m.load_jsonl=lambda *a, **k: []
sys.modules['portfolio.file_utils']=m
ref=importlib.import_module('portfolio.reflection')
portfolio={
 'initial_value_sek':1000,
 'cash_sek':1100,
 'holdings':{},
 'transactions':[
   {'ticker':'BTC-USD','action':'BUY','shares':1,'price_sek':100},
   {'ticker':'BTC-USD','action':'SELL','shares':1,'price_sek':120},
   {'ticker':'BTC-USD','action':'BUY','shares':1,'price_sek':200},
   {'ticker':'BTC-USD','action':'SELL','shares':1,'price_sek':220},
 ]
}
print(ref._compute_strategy_metrics(portfolio))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 579ms:
{'trades': 4, 'win_rate': 1.0, 'avg_pnl_pct': 33.33, 'total_pnl_pct': 10.0, 'holdings': []}

codex
The patch is not runnable as committed because the new main entry point imports internal modules that are missing from the tree. It also contains several logic errors in core trigger/session/reporting paths that would cause wrong or missed behavior even after startup issues were fixed.

Full review comments:

- [P0] Ship the internal modules that `portfolio.main` imports — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/main.py:31-31
  `portfolio.main` imports `portfolio.file_utils` here, but that module (and many other `portfolio.*` dependencies referenced throughout this patch) is not present in the committed tree. As a result, `python -m portfolio.main` currently aborts immediately with `ModuleNotFoundError`, so none of the new orchestration code can start at all.

- [P1] Parse stock-trigger reason strings in `_extract_ticker` — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/agent_invocation.py:236-240
  For stock triggers such as `MSTR consensus BUY (...)` or `MSTR moved ...`, this regex never matches and `_extract_ticker()` falls back to `XAG-USD`. That gives the wrong ticker to decision-feedback, multi-agent prompts, and the trade-guard block check, so stock invocations can analyze silver context or miss a real block on the triggered symbol.

- [P1] Leave suppressed ranging consensuses eligible for a later trigger — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/trigger.py:230-231
  When a ranging-market consensus is suppressed for low confidence, this still records `triggered_consensus[ticker] = action`. If that BUY/SELL persists and confidence rises above the threshold on a later cycle, it no longer looks like a new HOLD→BUY/SELL transition, so Layer 2 never fires at all. In practice, a ticker that starts with a weak ranging signal can miss its first actionable consensus unless it drops back to HOLD first.

- [P2] Consume matched buy lots when computing reflection PnL — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/reflection.py:67-70
  This sell path never removes or reduces the buy lots it has already matched, so once the same ticker is round-tripped more than once every later sell is still compared against all historical buys. That inflates `win_rate` and `avg_pnl_pct` in the saved reflections; e.g. two separate profitable trades on the same asset are reported against a blended cost basis from both entries instead of the remaining open lot(s).

- [P2] Mark Swedish holiday sessions as closed for warrants/stocks — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/session_calendar.py:183-184
  EU sessions are treated as open on any weekday, but Swedish holidays are never checked here even though `market_timing.py` already defines them. On weekday holidays such as Christmas Eve or Midsummer Eve, `get_session_info("warrant")` returns `is_open=True`; `reporting.write_agent_summary()` uses that flag before building warrant exit plans, so the system will act as if Avanza is tradeable when the market is actually closed.
The patch is not runnable as committed because the new main entry point imports internal modules that are missing from the tree. It also contains several logic errors in core trigger/session/reporting paths that would cause wrong or missed behavior even after startup issues were fixed.

Full review comments:

- [P0] Ship the internal modules that `portfolio.main` imports — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/main.py:31-31
  `portfolio.main` imports `portfolio.file_utils` here, but that module (and many other `portfolio.*` dependencies referenced throughout this patch) is not present in the committed tree. As a result, `python -m portfolio.main` currently aborts immediately with `ModuleNotFoundError`, so none of the new orchestration code can start at all.

- [P1] Parse stock-trigger reason strings in `_extract_ticker` — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/agent_invocation.py:236-240
  For stock triggers such as `MSTR consensus BUY (...)` or `MSTR moved ...`, this regex never matches and `_extract_ticker()` falls back to `XAG-USD`. That gives the wrong ticker to decision-feedback, multi-agent prompts, and the trade-guard block check, so stock invocations can analyze silver context or miss a real block on the triggered symbol.

- [P1] Leave suppressed ranging consensuses eligible for a later trigger — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/trigger.py:230-231
  When a ranging-market consensus is suppressed for low confidence, this still records `triggered_consensus[ticker] = action`. If that BUY/SELL persists and confidence rises above the threshold on a later cycle, it no longer looks like a new HOLD→BUY/SELL transition, so Layer 2 never fires at all. In practice, a ticker that starts with a weak ranging signal can miss its first actionable consensus unless it drops back to HOLD first.

- [P2] Consume matched buy lots when computing reflection PnL — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/reflection.py:67-70
  This sell path never removes or reduces the buy lots it has already matched, so once the same ticker is round-tripped more than once every later sell is still compared against all historical buys. That inflates `win_rate` and `avg_pnl_pct` in the saved reflections; e.g. two separate profitable trades on the same asset are reported against a blended cost basis from both entries instead of the remaining open lot(s).

- [P2] Mark Swedish holiday sessions as closed for warrants/stocks — Q:/finance-analyzer/.worktrees/adv-orchestration/portfolio/session_calendar.py:183-184
  EU sessions are treated as open on any weekday, but Swedish holidays are never checked here even though `market_timing.py` already defines them. On weekday holidays such as Christmas Eve or Midsummer Eve, `get_session_info("warrant")` returns `is_open=True`; `reporting.write_agent_summary()` uses that flag before building warrant exit plans, so the system will act as if Avanza is tradeable when the market is actually closed.
