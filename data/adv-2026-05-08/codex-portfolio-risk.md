OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-portfolio-risk
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e0833-df89-7043-9910-b339a35b6e67
--------
user
changes against 'empty-baseline'
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9 && git diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
2026-05-08T15:28:04.455226Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
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


 exited 1 in 568ms:
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
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Location; Get-ChildItem -Force' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 641ms:

[32;1mPath[0m
[32;1m----[0m
Q:\finance-analyzer\.worktrees\adv-portfolio-risk

[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-portfolio-risk
[32;1mPSChildName         : [0mportfolio
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mportfolio
[32;1mParent              : [0mQ:\finance-analyzer\.worktrees\adv-portfolio-risk
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-08 17:25:55
[32;1mCreationTimeUtc     : [0m2026-05-08 15:25:55
[32;1mLastAccessTime      : [0m2026-05-08 17:27:58
[32;1mLastAccessTimeUtc   : [0m2026-05-08 15:27:58
[32;1mLastWriteTime       : [0m2026-05-08 17:25:55
[32;1mLastWriteTimeUtc    : [0m2026-05-08 15:25:55
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mportfolio
[32;1mResolvedTarget      : [0mQ:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m62
[32;1mName          : [0m.git


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m46
[32;1mName          : [0mBASELINE.md



exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content .git; Get-Content BASELINE.md' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 610ms:
gitdir: Q:/finance-analyzer/.git/worktrees/adv-portfolio-risk
# Empty baseline for adversarial review diffs

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel; Set-Location portfolio; git rev-parse --show-toplevel' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
2026-05-08T15:28:15.124291Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-portfolio-risk'
'Q:/finance-analyzer/.worktrees/adv-portfolio-risk/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-portfolio-risk
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-portfolio-risk'
'Q:/finance-analyzer/.worktrees/adv-portfolio-risk/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-portfolio-risk

 exited 1 in 583ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-portfolio-risk'
'Q:/finance-analyzer/.worktrees/adv-portfolio-risk/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-portfolio-risk
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-portfolio-risk'
'Q:/finance-analyzer/.worktrees/adv-portfolio-risk/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-portfolio-risk

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-portfolio-risk diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9 && git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-portfolio-risk diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 573ms:
 portfolio/circuit_breaker.py       | 134 +++++
 portfolio/cost_model.py            | 116 +++++
 portfolio/equity_curve.py          | 600 ++++++++++++++++++++++
 portfolio/kelly_sizing.py          | 389 +++++++++++++++
 portfolio/monte_carlo.py           | 422 ++++++++++++++++
 portfolio/monte_carlo_risk.py      | 515 +++++++++++++++++++
 portfolio/portfolio_mgr.py         | 180 +++++++
 portfolio/portfolio_validator.py   | 299 +++++++++++
 portfolio/risk_management.py       | 999 +++++++++++++++++++++++++++++++++++++
 portfolio/trade_guards.py          | 388 ++++++++++++++
 portfolio/trade_risk_classifier.py | 121 +++++
 portfolio/trade_validation.py      | 114 +++++
 portfolio/warrant_portfolio.py     | 266 ++++++++++
 13 files changed, 4543 insertions(+)
portfolio/circuit_breaker.py
portfolio/cost_model.py
portfolio/equity_curve.py
portfolio/kelly_sizing.py
portfolio/monte_carlo.py
portfolio/monte_carlo_risk.py
portfolio/portfolio_mgr.py
portfolio/portfolio_validator.py
portfolio/risk_management.py
portfolio/trade_guards.py
portfolio/trade_risk_classifier.py
portfolio/trade_validation.py
portfolio/warrant_portfolio.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -File | Select-Object FullName' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 588ms:

[32;1mFullName[0m
[32;1m--------[0m
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\BASELINE.md
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\circuit_breaker.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\cost_model.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\equity_curve.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\kelly_sizing.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo_risk.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_mgr.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_validator.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\risk_management.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\trade_guards.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\trade_risk_classifier.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\trade_validation.py
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\warrant_portfolio.py


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import compileall, sys
ok = compileall.compile_dir('portfolio', force=True, quiet=1)
print('ok=', ok)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 633ms:
ok= True

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p=Path('portfolio/risk_management.py')
for i,line in enumerate(p.read_text().splitlines(),1):
    print(f\"{i:4}: {line}\")
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p=Path('portfolio/portfolio_mgr.py')
for i,line in enumerate(p.read_text().splitlines(),1):
    print(f\"{i:4}: {line}\")
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p=Path('portfolio/warrant_portfolio.py')
for i,line in enumerate(p.read_text().splitlines(),1):
    print(f\"{i:4}: {line}\")
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 664ms:
   1: """Risk management utilities for portfolio intelligence system.
   2: 
   3: Provides:
   4: - Maximum drawdown circuit breaker
   5: - ATR-based trailing stop-loss tracking
   6: - Position age tracking
   7: - Portfolio value history logging
   8: - Transaction cost analysis
   9: """
  10: 
  11: import datetime
  12: import json
  13: import logging
  14: import pathlib
  15: import threading
  16: 
  17: from portfolio.file_utils import atomic_append_jsonl, load_json
  18: 
  19: logger = logging.getLogger(__name__)
  20: 
  21: 
  22: # FEAT-3: Cache peak values per (path, key) to avoid re-scanning the full
  23: # JSONL on every 60s cycle. The file is append-only, so the peak can only
  24: # increase. Cache stores: peak value + byte offset of last read position.
  25: # Invalidated when the file shrinks (log rotation).
  26: _peak_cache: dict[tuple, dict] = {}
  27: 
  28: # Adversarial review 04-29 PR-P1-2 (2026-05-02): the main loop's 8-worker
  29: # ThreadPoolExecutor invokes check_drawdown() concurrently across the
  30: # patient + bold portfolios, and update_health periodically calls
  31: # _streaming_max. Without a lock, two threads could last-writer-wins on
  32: # _peak_cache[cache_key] = {...}, occasionally losing a cached offset and
  33: # forcing the next call to do a full O(file_size) scan instead of the
  34: # O(delta) streaming read. Same class of bug as today's commit cdcbbd0f
  35: # for signal_history.update_history. Lock scope: the entire read-decide-
  36: # update sequence in _streaming_max — both reading the cached offset and
  37: # writing back the new offset have to be atomic relative to other workers.
  38: _peak_cache_lock = threading.Lock()
  39: 
  40: 
  41: def _streaming_max(history_path: pathlib.Path, value_key: str, floor: float) -> float:
  42:     """A-PR-2 (2026-04-11): Find the maximum value at `value_key` in a JSONL file.
  43: 
  44:     FEAT-3 (2026-04-29): Uses a byte-offset cache so subsequent calls only
  45:     scan new entries appended since the last call. Falls back to a full scan
  46:     if the file shrinks (rotation) or on any seek error.
  47: 
  48:     PR-P1-2 (2026-05-02): _peak_cache reads + writes are serialized under
  49:     _peak_cache_lock to avoid the 8-worker ThreadPoolExecutor losing cached
  50:     offsets to last-writer-wins races (see lock comment above).
  51: 
  52:     Streams line-by-line so memory stays O(1) regardless of file size.
  53:     Returns `floor` (typically initial_value) if file missing/empty.
  54:     """
  55:     if not history_path.exists():
  56:         return floor
  57: 
  58:     cache_key = (str(history_path), value_key)
  59: 
  60:     try:
  61:         file_size = history_path.stat().st_size
  62:     except OSError:
  63:         file_size = 0
  64: 
  65:     # Snapshot cache under the lock — keeps the (read offset, decide
  66:     # restart vs resume, hold last good peak) sequence consistent with
  67:     # the matching write at the bottom of the function.
  68:     with _peak_cache_lock:
  69:         cached = _peak_cache.get(cache_key)
  70:         if cached is not None:
  71:             if file_size >= cached["offset"]:
  72:                 start_offset = cached["offset"]
  73:                 peak = cached["peak"]
  74:             else:
  75:                 # File shrank (rotation) — full re-scan
  76:                 start_offset = 0
  77:                 peak = floor
  78:         else:
  79:             start_offset = 0
  80:             peak = floor
  81: 
  82:     try:
  83:         with open(history_path, encoding="utf-8") as f:
  84:             if start_offset > 0:
  85:                 f.seek(start_offset)
  86:             for line in f:
  87:                 line = line.strip()
  88:                 if not line:
  89:                     continue
  90:                 try:
  91:                     entry = json.loads(line)
  92:                 except json.JSONDecodeError:
  93:                     continue
  94:                 val = entry.get(value_key, 0)
  95:                 if val > peak:
  96:                     peak = val
  97:             end_offset = f.tell()
  98:     except OSError as e:
  99:         logger.warning("Could not stream history file %s: %s", history_path.name, e)
 100:         with _peak_cache_lock:
 101:             cached_after = _peak_cache.get(cache_key)
 102:         if cached_after is not None:
 103:             return cached_after["peak"]
 104:         return peak
 105: 
 106:     with _peak_cache_lock:
 107:         _peak_cache[cache_key] = {"peak": peak, "offset": end_offset}
 108:     return peak
 109: 
 110: DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
 111: 
 112: INITIAL_VALUE_DEFAULT = 500_000  # SEK
 113: 
 114: # Adversarial review 05-01 P1-15 (2026-05-02): persistent fallback for fx_rate.
 115: # Sane historical USD/SEK band (matches portfolio/fx_rates.py:42 sanity gate).
 116: # The hardcoded last-resort value matches portfolio/fx_rates.py:66 — both
 117: # modules need the same number so a fallback path through risk_management
 118: # doesn't disagree with the live fx fetcher.
 119: _FX_RATE_MIN = 7.0
 120: _FX_RATE_MAX = 15.0
 121: _FX_RATE_HARDCODED_FALLBACK = 10.50
 122: _FX_CACHE_FILENAME = "fx_rate_cache.json"
 123: 
 124: 
 125: def _resolve_fx_rate(agent_summary: dict) -> float:
 126:     """Return USD→SEK rate, preferring (1) summary, (2) cached, (3) hardcoded.
 127: 
 128:     Adversarial review 05-01 P1-15: the original code did
 129:     ``fx_rate = agent_summary.get("fx_rate", 1.0)``. When agent_summary was
 130:     missing/empty/lacking the field — early loop cycle, agent_summary
 131:     rotation, fx_rates.py crashed mid-fetch — the 1.0 default understated
 132:     SEK valuations by ~10x and could trigger a false drawdown breach
 133:     (a 5_435_000 SEK position valued at fx_rate=1.0 looks like 95%
 134:     drawdown from the 500_000 SEK initial).
 135: 
 136:     Resolution order:
 137:       1. ``agent_summary["fx_rate"]`` if it's a finite number in [7, 15].
 138:          (Same sanity band as portfolio/fx_rates.py:42.) Successful values
 139:          update the disk cache for future fallbacks.
 140:       2. Cached rate from ``DATA_DIR/fx_rate_cache.json`` if present and
 141:          in-band. The cache is best-effort: corrupt JSON or missing/invalid
 142:          rate field is treated as no-cache.
 143:       3. ``_FX_RATE_HARDCODED_FALLBACK`` (10.50) — matches
 144:          portfolio/fx_rates.py:66 so both modules disagree-by-zero on the
 145:          absolute worst-case path.
 146: 
 147:     Note: 1.0 is explicitly rejected by the sanity band, so the legacy
 148:     pattern ``agent_summary.get("fx_rate", 1.0)`` continues to defer
 149:     correctly even if a stale agent_summary still embeds 1.0 itself.
 150:     """
 151:     raw = agent_summary.get("fx_rate") if isinstance(agent_summary, dict) else None
 152:     try:
 153:         rate = float(raw) if raw is not None else None
 154:     except (TypeError, ValueError):
 155:         rate = None
 156:     if rate is not None and _FX_RATE_MIN <= rate <= _FX_RATE_MAX:
 157:         # Cache the good rate for future fallback paths.
 158:         try:
 159:             from portfolio.file_utils import atomic_write_json
 160:             atomic_write_json(DATA_DIR / _FX_CACHE_FILENAME, {
 161:                 "rate": rate,
 162:                 "ts": datetime.datetime.now(datetime.UTC).isoformat(),
 163:             })
 164:         except Exception as e:
 165:             logger.debug("fx cache persist failed: %s", e)
 166:         return rate
 167: 
 168:     # Try disk cache.
 169:     cached = load_json(DATA_DIR / _FX_CACHE_FILENAME, default=None)
 170:     if isinstance(cached, dict):
 171:         try:
 172:             cached_rate = float(cached.get("rate"))
 173:             if _FX_RATE_MIN <= cached_rate <= _FX_RATE_MAX:
 174:                 return cached_rate
 175:         except (TypeError, ValueError):
 176:             pass
 177: 
 178:     logger.warning(
 179:         "fx_rate fallback to hardcoded %.2f — agent_summary missing/invalid "
 180:         "and no usable cache at %s. Portfolio valuations may be ~10%% off if "
 181:         "SEK has moved.",
 182:         _FX_RATE_HARDCODED_FALLBACK, DATA_DIR / _FX_CACHE_FILENAME,
 183:     )
 184:     return _FX_RATE_HARDCODED_FALLBACK
 185: 
 186: 
 187: def _compute_portfolio_value(portfolio: dict, agent_summary: dict) -> float:
 188:     """Compute current total portfolio value in SEK.
 189: 
 190:     Value = cash_sek + sum(shares * current_price_sek) for each holding.
 191:     current_price_sek is derived from agent_summary prices * fx_rate.
 192:     """
 193:     cash = portfolio.get("cash_sek", 0)
 194:     holdings = portfolio.get("holdings", {})
 195:     # P1-15 (2026-05-02): use cached fallback chain instead of raw .get(..., 1.0).
 196:     fx_rate = _resolve_fx_rate(agent_summary)
 197:     signals = agent_summary.get("signals", {})
 198: 
 199:     holdings_value = 0.0
 200:     for ticker, pos in holdings.items():
 201:         shares = pos.get("shares", 0)
 202:         if shares <= 0:
 203:             continue
 204:         # Try to get current price from agent_summary signals
 205:         if ticker in signals:
 206:             price_usd = signals[ticker].get("price_usd", 0)
 207:             holdings_value += shares * price_usd * fx_rate
 208:         else:
 209:             # Fallback: use avg_cost_usd from holdings if no live price
 210:             avg_cost = pos.get("avg_cost_usd", 0)
 211:             holdings_value += shares * avg_cost * fx_rate
 212: 
 213:     return cash + holdings_value
 214: 
 215: 
 216: def check_drawdown(portfolio_path: str, max_drawdown_pct: float = 20.0,
 217:                    agent_summary_path: str | None = None) -> dict:
 218:     """Check if portfolio has exceeded maximum drawdown threshold.
 219: 
 220:     Computes current portfolio value against the initial value and the peak
 221:     value recorded in portfolio_value_history.jsonl (if available).
 222: 
 223:     Args:
 224:         portfolio_path: Path to portfolio_state JSON file.
 225:         max_drawdown_pct: Maximum allowed drawdown percentage (default 20%).
 226:         agent_summary_path: Path to agent_summary.json for live prices.
 227:             If None, uses DATA_DIR / "agent_summary.json".
 228: 
 229:     Returns:
 230:         dict with:
 231:             - breached: bool -- True if drawdown exceeds threshold
 232:             - current_drawdown_pct: float -- current drawdown from peak (positive number)
 233:             - peak_value: float -- highest portfolio value seen
 234:             - current_value: float -- current portfolio value in SEK
 235:             - initial_value: float -- starting portfolio value
 236:     """
 237:     portfolio = load_json(portfolio_path, default={})
 238:     initial_value = portfolio.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
 239: 
 240:     if agent_summary_path is None:
 241:         agent_summary_path = str(DATA_DIR / "agent_summary.json")
 242: 
 243:     # If portfolio has no holdings, value is just cash
 244:     if not portfolio.get("holdings"):
 245:         current_value = portfolio.get("cash_sek", initial_value)
 246:     else:
 247:         summary = load_json(agent_summary_path, default={})
 248:         if summary:
 249:             current_value = _compute_portfolio_value(portfolio, summary)
 250:         else:
 251:             # Fallback: cash only. This is NOT truly conservative —
 252:             # it ignores unrealized P&L on holdings. If the price
 253:             # feed is stale while holdings are underwater, the
 254:             # drawdown reading will look tiny and the circuit
 255:             # breaker will never trip. Surface the blind spot via
 256:             # WARNING so dashboards/oncall can see "feed stale"
 257:             # rather than silently trusting the number. (2026-04-17
 258:             # adversarial review.)
 259:             holding_count = sum(
 260:                 1 for h in portfolio.get("holdings", {}).values()
 261:                 if h.get("shares", 0) > 0
 262:             )
 263:             logger.warning(
 264:                 "check_drawdown: agent_summary empty at %s — falling "
 265:                 "back to cash-only value for %d live position(s). "
 266:                 "Drawdown circuit breaker reading may be optimistic.",
 267:                 agent_summary_path, holding_count,
 268:             )
 269:             current_value = portfolio.get("cash_sek", initial_value)
 270: 
 271:     # Determine peak value from history file or initial value
 272:     history_path = DATA_DIR / "portfolio_value_history.jsonl"
 273:     pf_name = pathlib.Path(portfolio_path).stem  # e.g. "portfolio_state" or "portfolio_state_bold"
 274:     is_bold = "bold" in pf_name
 275:     value_key = "bold_value_sek" if is_bold else "patient_value_sek"
 276: 
 277:     # A-PR-2: Stream the FULL history (not just last 2000 entries) to find
 278:     # the true historical peak. The 2000-entry tail only covered ~33h, so
 279:     # any rally older than that fell off the back and the drawdown circuit
 280:     # breaker became blind to multi-day peaks.
 281:     peak_value = _streaming_max(history_path, value_key, floor=initial_value)
 282: 
 283:     # Also compare against current value in case it's a new peak
 284:     if current_value > peak_value:
 285:         peak_value = current_value
 286: 
 287:     # Guard against NaN/Inf in peak_value or current_value — corrupted
 288:     # history or failed computation. NaN silently passes all comparison
 289:     # checks (NaN > 50.0 is False), bypassing the circuit breaker.
 290:     import math
 291:     if not math.isfinite(peak_value) or not math.isfinite(current_value):
 292:         logger.critical(
 293:             "check_drawdown: non-finite value detected (peak=%.2f, current=%.2f) "
 294:             "— treating as 100%% drawdown (fail-safe).",
 295:             peak_value, current_value,
 296:         )
 297:         return {
 298:             "breached": True,
 299:             "current_drawdown_pct": 100.0,
 300:             "peak_value": round(peak_value, 2) if math.isfinite(peak_value) else 0.0,
 301:             "current_value": round(current_value, 2) if math.isfinite(current_value) else 0.0,
 302:             "initial_value": initial_value,
 303:         }
 304: 
 305:     # Calculate drawdown
 306:     if peak_value > 0:
 307:         current_drawdown_pct = ((peak_value - current_value) / peak_value) * 100
 308:     else:
 309:         current_drawdown_pct = 0.0
 310: 
 311:     return {
 312:         "breached": current_drawdown_pct > max_drawdown_pct,
 313:         "current_drawdown_pct": round(current_drawdown_pct, 4),
 314:         "peak_value": round(peak_value, 2),
 315:         "current_value": round(current_value, 2),
 316:         "initial_value": initial_value,
 317:     }
 318: 
 319: 
 320: def compute_stop_levels(holdings: dict, agent_summary: dict) -> dict:
 321:     """Compute ATR-based stop-loss levels for all positions.
 322: 
 323:     For each holding with shares > 0, calculates stop-loss levels based on
 324:     2x ATR (Average True Range) from the entry price.
 325: 
 326:     Args:
 327:         holdings: The "holdings" dict from portfolio state.
 328:             Each entry: {ticker: {"shares": N, "avg_cost_usd": X, ...}}
 329:         agent_summary: Parsed agent_summary.json dict.
 330: 
 331:     Returns:
 332:         dict keyed by ticker, each with:
 333:             - entry_price_usd: float (avg_cost_usd)
 334:             - current_price_usd: float (from agent_summary)
 335:             - atr_pct: float (from agent_summary)
 336:             - stop_price_usd: float (entry_price * (1 - 2 * atr_pct/100))
 337:             - triggered: bool (current_price < stop_price)
 338:             - distance_to_stop_pct: float (positive = above stop, negative = below)
 339:             - pnl_pct: float (current vs entry)
 340:     """
 341:     signals = agent_summary.get("signals", {})
 342:     result = {}
 343: 
 344:     for ticker, pos in holdings.items():
 345:         shares = pos.get("shares", 0)
 346:         if shares <= 0:
 347:             continue
 348: 
 349:         entry_price = pos.get("avg_cost_usd", 0)
 350:         if entry_price <= 0:
 351:             continue
 352: 
 353:         # Get current price and ATR from agent_summary
 354:         if ticker not in signals:
 355:             # Ticker not in current summary (e.g., stock after hours)
 356:             result[ticker] = {
 357:                 "entry_price_usd": entry_price,
 358:                 "current_price_usd": None,
 359:                 "atr_pct": None,
 360:                 "stop_price_usd": None,
 361:                 "triggered": False,
 362:                 "distance_to_stop_pct": None,
 363:                 "pnl_pct": None,
 364:                 "note": "No live data available (market closed or ticker not in summary)",
 365:             }
 366:             continue
 367: 
 368:         sig = signals[ticker]
 369:         current_price = sig.get("price_usd", 0)
 370:         atr_pct = sig.get("atr_pct", 0)
 371: 
 372:         # 2x ATR stop-loss — cap ATR at 15% to prevent meaninglessly wide stops for warrants
 373:         atr_pct = min(atr_pct, 15.0)
 374:         stop_price = entry_price * (1 - 2 * atr_pct / 100)
 375: 
 376:         # Distance from current price to stop
 377:         if stop_price > 0:
 378:             distance_to_stop_pct = ((current_price - stop_price) / stop_price) * 100
 379:         else:
 380:             distance_to_stop_pct = float("inf")
 381: 
 382:         triggered = current_price < stop_price if current_price > 0 else False
 383:         pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
 384: 
 385:         result[ticker] = {
 386:             "entry_price_usd": round(entry_price, 6),
 387:             "current_price_usd": round(current_price, 6),
 388:             "atr_pct": round(atr_pct, 4),
 389:             "stop_price_usd": round(stop_price, 6),
 390:             "triggered": triggered,
 391:             "distance_to_stop_pct": round(distance_to_stop_pct, 4),
 392:             "pnl_pct": round(pnl_pct, 4),
 393:         }
 394: 
 395:     return result
 396: 
 397: 
 398: def compute_probabilistic_stops(holdings: dict, agent_summary: dict) -> dict:
 399:     """Compute Monte Carlo probabilistic stop-loss quality for held positions.
 400: 
 401:     For each position, simulates remaining-session price paths and estimates
 402:     the probability of hitting the ATR stop level, plus the expected time to
 403:     hit if triggered.
 404: 
 405:     Args:
 406:         holdings: The "holdings" dict from portfolio state.
 407:         agent_summary: Parsed agent_summary.json dict.
 408: 
 409:     Returns:
 410:         dict keyed by ticker, each with:
 411:             - stop_price_usd: ATR-based stop level
 412:             - stop_hit_prob: P(hitting stop this session), 0.0-1.0
 413:             - expected_hit_time_min: E[time to stop | hit], or None
 414:             - knockout_prob: For warrants, P(reaching financing level)
 415:             - current_price_usd: current price
 416:     """
 417:     try:
 418:         from portfolio.exit_optimizer import _first_hit_times, simulate_intraday_paths
 419:         from portfolio.session_calendar import remaining_session_minutes
 420:     except ImportError:
 421:         logger.warning("exit_optimizer or session_calendar not available")
 422:         return {}
 423: 
 424:     import numpy as np
 425: 
 426:     signals = agent_summary.get("signals", {})
 427:     result = {}
 428: 
 429:     for ticker, pos in holdings.items():
 430:         shares = pos.get("shares", 0)
 431:         if shares <= 0:
 432:             continue
 433: 
 434:         entry_price = pos.get("avg_cost_usd", 0)
 435:         if entry_price <= 0:
 436:             continue
 437: 
 438:         sig = signals.get(ticker)
 439:         if not sig:
 440:             continue
 441: 
 442:         current_price = sig.get("price_usd", 0)
 443:         atr_pct = sig.get("atr_pct", 0)
 444:         if current_price <= 0 or atr_pct <= 0:
 445:             continue
 446: 
 447:         # Determine instrument type for session lookup
 448:         if ticker.endswith("-USD") and ticker.startswith(("BTC", "ETH")):
 449:             inst_type = "crypto"
 450:         elif ticker.startswith(("XAG", "XAU")):
 451:             inst_type = "warrant"
 452:         else:
 453:             inst_type = "stock"
 454: 
 455:         # Get remaining session minutes
 456:         remaining = remaining_session_minutes(inst_type)
 457:         if remaining < 2:
 458:             continue
 459: 
 460:         # Estimate volatility from ATR
 461:         import math
 462:         vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)
 463: 
 464:         # ATR stop level
 465:         stop_price = entry_price * (1 - 2 * atr_pct / 100)
 466:         if stop_price <= 0:
 467:             stop_price = entry_price * 0.01
 468: 
 469:         # Simulate paths
 470:         paths = simulate_intraday_paths(
 471:             price=current_price,
 472:             volatility=vol,
 473:             drift=0.0,
 474:             remaining_minutes=int(remaining),
 475:             instrument_type=inst_type,
 476:             n_paths=2000,
 477:         )
 478: 
 479:         # Stop hit probability
 480:         session_min = np.min(paths[:, 1:], axis=1)
 481:         stop_hit_prob = float(np.mean(session_min <= stop_price))
 482: 
 483:         # Expected time to hit stop
 484:         hit_times = _first_hit_times(paths, stop_price, direction="below")
 485:         hitting = hit_times[hit_times > 0]
 486:         expected_hit_time = float(np.mean(hitting)) if len(hitting) > 0 else None
 487: 
 488:         result[ticker] = {
 489:             "stop_price_usd": round(stop_price, 4),
 490:             "stop_hit_prob": round(stop_hit_prob, 4),
 491:             "expected_hit_time_min": round(expected_hit_time, 1) if expected_hit_time else None,
 492:             "knockout_prob": None,
 493:             "current_price_usd": round(current_price, 4),
 494:         }
 495: 
 496:     return result
 497: 
 498: 
 499: def get_position_ages(portfolio: dict) -> dict:
 500:     """Calculate age of each position from first BUY transaction.
 501: 
 502:     Args:
 503:         portfolio: Full portfolio state dict (with "holdings" and "transactions").
 504: 
 505:     Returns:
 506:         dict keyed by ticker (only tickers currently held with shares > 0):
 507:             - age_hours: float
 508:             - age_days: float
 509:             - first_buy: str (ISO-8601 timestamp of first BUY)
 510:             - num_buys: int (total BUY transactions for this ticker)
 511:             - num_sells: int (total SELL transactions for this ticker)
 512:     """
 513:     holdings = portfolio.get("holdings", {})
 514:     transactions = portfolio.get("transactions", [])
 515:     now = datetime.datetime.now(datetime.UTC)
 516:     result = {}
 517: 
 518:     for ticker, pos in holdings.items():
 519:         shares = pos.get("shares", 0)
 520:         if shares <= 0:
 521:             continue
 522: 
 523:         # Find all BUY and SELL transactions for this ticker
 524:         first_buy_ts = None
 525:         num_buys = 0
 526:         num_sells = 0
 527: 
 528:         for tx in transactions:
 529:             if tx.get("ticker") != ticker:
 530:                 continue
 531:             action = tx.get("action", "")
 532:             ts_str = tx.get("timestamp", "")
 533: 
 534:             if action == "BUY":
 535:                 num_buys += 1
 536:                 try:
 537:                     ts = datetime.datetime.fromisoformat(ts_str)
 538:                     if first_buy_ts is None or ts < first_buy_ts:
 539:                         first_buy_ts = ts
 540:                 except (ValueError, TypeError):
 541:                     pass
 542:             elif action == "SELL":
 543:                 num_sells += 1
 544: 
 545:         if first_buy_ts is not None:
 546:             # Ensure timezone-aware comparison
 547:             if first_buy_ts.tzinfo is None:
 548:                 first_buy_ts = first_buy_ts.replace(tzinfo=datetime.UTC)
 549:             age_delta = now - first_buy_ts
 550:             age_hours = age_delta.total_seconds() / 3600
 551:             age_days = age_hours / 24
 552: 
 553:             result[ticker] = {
 554:                 "age_hours": round(age_hours, 2),
 555:                 "age_days": round(age_days, 2),
 556:                 "first_buy": first_buy_ts.isoformat(),
 557:                 "num_buys": num_buys,
 558:                 "num_sells": num_sells,
 559:             }
 560: 
 561:     return result
 562: 
 563: 
 564: def log_portfolio_value(patient_path: str | None = None,
 565:                         bold_path: str | None = None,
 566:                         agent_summary_path: str | None = None):
 567:     """Append current portfolio values to data/portfolio_value_history.jsonl.
 568: 
 569:     Each entry contains:
 570:         - ts: ISO-8601 UTC timestamp
 571:         - patient_value_sek: total patient portfolio value
 572:         - bold_value_sek: total bold portfolio value
 573:         - patient_pnl_pct: patient P&L percentage
 574:         - bold_pnl_pct: bold P&L percentage
 575:         - prices: dict of current USD prices from agent_summary
 576: 
 577:     Args:
 578:         patient_path: Path to patient portfolio state JSON.
 579:         bold_path: Path to bold portfolio state JSON.
 580:         agent_summary_path: Path to agent_summary.json.
 581:     """
 582:     if patient_path is None:
 583:         patient_path = str(DATA_DIR / "portfolio_state.json")
 584:     if bold_path is None:
 585:         bold_path = str(DATA_DIR / "portfolio_state_bold.json")
 586:     if agent_summary_path is None:
 587:         agent_summary_path = str(DATA_DIR / "agent_summary.json")
 588: 
 589:     patient = load_json(patient_path, default={})
 590:     bold = load_json(bold_path, default={})
 591:     # P1-15 (2026-05-02): the {"fx_rate": 1.0} default here was load-bearing
 592:     # for the false-circuit-breaker bug. Using a missing-fx_rate default lets
 593:     # _resolve_fx_rate inside _compute_portfolio_value walk the cache chain
 594:     # instead of taking the 1.0 at face value.
 595:     summary = load_json(agent_summary_path, default={"signals": {}})
 596: 
 597:     patient_value = _compute_portfolio_value(patient, summary)
 598:     bold_value = _compute_portfolio_value(bold, summary)
 599: 
 600:     patient_initial = patient.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
 601:     bold_initial = bold.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
 602: 
 603:     patient_pnl_pct = ((patient_value - patient_initial) / patient_initial) * 100 if patient_initial > 0 else 0
 604:     bold_pnl_pct = ((bold_value - bold_initial) / bold_initial) * 100 if bold_initial > 0 else 0
 605: 
 606:     # Collect current prices
 607:     prices = {}
 608:     for ticker, sig in summary.get("signals", {}).items():
 609:         price = sig.get("price_usd")
 610:         if price is not None:
 611:             prices[ticker] = price
 612: 
 613:     entry = {
 614:         "ts": datetime.datetime.now(datetime.UTC).isoformat(),
 615:         "patient_value_sek": round(patient_value, 2),
 616:         "bold_value_sek": round(bold_value, 2),
 617:         "patient_pnl_pct": round(patient_pnl_pct, 4),
 618:         "bold_pnl_pct": round(bold_pnl_pct, 4),
 619:         "fx_rate": _resolve_fx_rate(summary),  # P1-15 (2026-05-02)
 620:         "prices": prices,
 621:     }
 622: 
 623:     history_path = DATA_DIR / "portfolio_value_history.jsonl"
 624:     atomic_append_jsonl(history_path, entry)
 625: 
 626: 
 627: def transaction_cost_analysis(portfolio: dict) -> dict:
 628:     """Analyze transaction costs for a portfolio.
 629: 
 630:     Examines all transactions and accumulated fees to produce a cost report.
 631: 
 632:     Args:
 633:         portfolio: Full portfolio state dict.
 634: 
 635:     Returns:
 636:         dict with:
 637:             - total_fees_sek: float -- accumulated fees
 638:             - fees_as_pct_of_initial: float -- total_fees / initial_value * 100
 639:             - avg_fee_per_trade: float -- average fee per transaction
 640:             - total_trades: int -- number of transactions
 641:             - fees_as_pct_of_pnl: float | None -- total_fees / abs(pnl) * 100
 642:               (None if no P&L to compare against)
 643:             - total_buy_volume_sek: float -- sum of BUY allocs
 644:             - total_sell_volume_sek: float -- sum of SELL proceeds
 645:             - buy_count: int
 646:             - sell_count: int
 647:     """
 648:     transactions = portfolio.get("transactions", [])
 649:     initial_value = portfolio.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
 650:     cash = portfolio.get("cash_sek", initial_value)
 651: 
 652:     # total_fees_sek from portfolio state
 653:     total_fees_from_state = portfolio.get("total_fees_sek", 0) or 0
 654: 
 655:     # Also compute fees from transaction records (fee_sek field)
 656:     computed_fees = 0.0
 657:     total_buy_volume = 0.0
 658:     total_sell_volume = 0.0
 659:     buy_count = 0
 660:     sell_count = 0
 661: 
 662:     for tx in transactions:
 663:         fee = tx.get("fee_sek", 0) or 0
 664:         computed_fees += fee
 665:         action = tx.get("action", "")
 666:         total_sek = tx.get("total_sek", 0) or 0
 667: 
 668:         if action == "BUY":
 669:             buy_count += 1
 670:             total_buy_volume += total_sek
 671:         elif action == "SELL":
 672:             sell_count += 1
 673:             total_sell_volume += total_sek
 674: 
 675:     # Use the larger of state fees vs computed fees (handles missing fee_sek fields)
 676:     total_fees = max(total_fees_from_state, computed_fees)
 677: 
 678:     total_trades = len(transactions)
 679:     avg_fee = total_fees / total_trades if total_trades > 0 else 0
 680: 
 681:     fees_as_pct_initial = (total_fees / initial_value) * 100 if initial_value > 0 else 0
 682: 
 683:     # PnL: cash + holdings_value - initial_value
 684:     # For simplicity here, we approximate with cash - initial (since holdings
 685:     # value requires live prices). A full PnL needs _compute_portfolio_value.
 686:     # However, if portfolio is all-cash (no holdings), this is exact.
 687:     holdings = portfolio.get("holdings", {})
 688:     has_open_positions = any(
 689:         pos.get("shares", 0) > 0 for pos in holdings.values()
 690:     )
 691: 
 692:     if has_open_positions:
 693:         # PnL is approximate (doesn't include unrealized gains)
 694:         pnl_note = "approximate (excludes unrealized gains/losses)"
 695:         pnl = cash - initial_value  # unrealized not included
 696:     else:
 697:         pnl_note = "exact (all positions closed)"
 698:         pnl = cash - initial_value
 699: 
 700:     if abs(pnl) > 0.01:
 701:         fees_as_pct_pnl = (total_fees / abs(pnl)) * 100
 702:     else:
 703:         fees_as_pct_pnl = None
 704: 
 705:     return {
 706:         "total_fees_sek": round(total_fees, 2),
 707:         "fees_as_pct_of_initial": round(fees_as_pct_initial, 4),
 708:         "avg_fee_per_trade": round(avg_fee, 2),
 709:         "total_trades": total_trades,
 710:         "fees_as_pct_of_pnl": round(fees_as_pct_pnl, 4) if fees_as_pct_pnl is not None else None,
 711:         "total_buy_volume_sek": round(total_buy_volume, 2),
 712:         "total_sell_volume_sek": round(total_sell_volume, 2),
 713:         "buy_count": buy_count,
 714:         "sell_count": sell_count,
 715:         "pnl_sek": round(pnl, 2),
 716:         "pnl_note": pnl_note,
 717:     }
 718: 
 719: 
 720: # ---------------------------------------------------------------------------
 721: # Risk Audit Flags — pre-trade risk checks for Layer 2
 722: # ---------------------------------------------------------------------------
 723: 
 724: # Hard-coded correlation pairs based on known relationships
 725: CORRELATED_PAIRS = {
 726:     "ETH-USD": ["BTC-USD"],
 727:     "BTC-USD": ["ETH-USD"],
 728:     "XAG-USD": ["XAU-USD"],
 729:     "XAU-USD": ["XAG-USD"],
 730:     "NVDA": ["AMD", "AVGO", "TSM"],
 731:     "AMD": ["NVDA", "AVGO", "TSM"],
 732:     "AVGO": ["NVDA", "AMD", "TSM"],
 733:     "TSM": ["NVDA", "AMD", "AVGO"],
 734:     "GOOGL": ["META", "AMZN"],
 735:     "META": ["GOOGL", "AMZN"],
 736:     "AMZN": ["GOOGL", "META"],
 737:     "AAPL": ["GOOGL", "META", "AMZN"],
 738: }
 739: 
 740: 
 741: def check_concentration_risk(ticker, action, portfolio, agent_summary, strategy="patient"):
 742:     """Check if a new BUY would create excessive concentration.
 743: 
 744:     Args:
 745:         ticker: Instrument to potentially buy.
 746:         action: "BUY" or "SELL".
 747:         portfolio: Portfolio state dict.
 748:         agent_summary: Parsed agent_summary dict.
 749:         strategy: "patient" or "bold".
 750: 
 751:     Returns:
 752:         dict with flag info, or None if no risk.
 753:     """
 754:     if action != "BUY":
 755:         return None
 756: 
 757:     cash = portfolio.get("cash_sek", 0)
 758:     holdings = portfolio.get("holdings", {})
 759:     # P1-15 (2026-05-02): use cached fallback chain instead of raw .get(..., 1.0).
 760:     fx_rate = _resolve_fx_rate(agent_summary)
 761:     signals = agent_summary.get("signals", {})
 762: 
 763:     # Compute current portfolio value
 764:     total_value = cash
 765:     for t, pos in holdings.items():
 766:         shares = pos.get("shares", 0)
 767:         if shares <= 0:
 768:             continue
 769:         price = signals.get(t, {}).get("price_usd", pos.get("avg_cost_usd", 0))
 770:         total_value += shares * price * fx_rate
 771: 
 772:     if total_value <= 0:
 773:         return None
 774: 
 775:     # BUG-176: Compute proposed allocation as % of total portfolio, capped at cash
 776:     alloc_pct = 0.30 if strategy == "bold" else 0.15
 777:     proposed_alloc = min(total_value * alloc_pct, cash)
 778: 
 779:     # Existing position value for this ticker
 780:     existing = holdings.get(ticker, {})
 781:     existing_shares = existing.get("shares", 0)
 782:     existing_price = signals.get(ticker, {}).get("price_usd", existing.get("avg_cost_usd", 0))
 783:     existing_value = existing_shares * existing_price * fx_rate
 784: 
 785:     new_position_value = existing_value + proposed_alloc
 786:     concentration_pct = (new_position_value / total_value) * 100
 787: 
 788:     if concentration_pct > 40:
 789:         return {
 790:             "flag": "concentration",
 791:             "severity": "warning",
 792:             "ticker": ticker,
 793:             "strategy": strategy,
 794:             "concentration_pct": round(concentration_pct, 1),
 795:             "message": (
 796:                 f"{ticker} would be {concentration_pct:.1f}% of {strategy} portfolio "
 797:                 f"(>{40}% threshold)"
 798:             ),
 799:         }
 800:     return None
 801: 
 802: 
 803: def check_regime_mismatch(ticker, action, agent_summary):
 804:     """Check if trade direction contradicts the market regime.
 805: 
 806:     BUY in trending-down (without volume confirmation) or
 807:     SELL in trending-up is a regime mismatch.
 808: 
 809:     Returns:
 810:         dict with flag info, or None if no mismatch.
 811:     """
 812:     if action == "HOLD":
 813:         return None
 814: 
 815:     sig = agent_summary.get("signals", {}).get(ticker, {})
 816:     regime = sig.get("regime", "ranging")
 817:     extra = sig.get("extra", {})
 818:     volume_ratio = extra.get("volume_ratio")
 819: 
 820:     mismatch = False
 821:     reason = ""
 822: 
 823:     if action == "BUY" and regime == "trending-down":
 824:         # BUY against downtrend — only flag with confirmed low volume.
 825:         # Missing volume data (None) = unknown, not a mismatch.
 826:         if volume_ratio is not None and volume_ratio < 1.5:
 827:             mismatch = True
 828:             reason = f"BUY in trending-down regime (RVOL={volume_ratio:.1f}, need >1.5x for reversal)"
 829:     elif action == "SELL" and regime == "trending-up" and volume_ratio is not None and volume_ratio < 1.5:
 830:         mismatch = True
 831:         reason = f"SELL in trending-up regime (RVOL={volume_ratio:.1f}, need >1.5x for reversal)"
 832: 
 833:     if mismatch:
 834:         return {
 835:             "flag": "regime_mismatch",
 836:             "severity": "warning",
 837:             "ticker": ticker,
 838:             "regime": regime,
 839:             "action": action,
 840:             "message": f"{ticker}: {reason}",
 841:         }
 842:     return None
 843: 
 844: 
 845: def check_correlation_risk(ticker, action, portfolio, strategy="patient"):
 846:     """Check if BUY would add correlated exposure to an existing position.
 847: 
 848:     Returns:
 849:         dict with flag info, or None if no correlation risk.
 850:     """
 851:     if action != "BUY":
 852:         return None
 853: 
 854:     correlated = CORRELATED_PAIRS.get(ticker, [])
 855:     if not correlated:
 856:         return None
 857: 
 858:     holdings = portfolio.get("holdings", {})
 859:     held_correlated = []
 860:     for t in correlated:
 861:         pos = holdings.get(t, {})
 862:         if pos.get("shares", 0) > 0:
 863:             held_correlated.append(t)
 864: 
 865:     if held_correlated:
 866:         return {
 867:             "flag": "correlation",
 868:             "severity": "warning",
 869:             "ticker": ticker,
 870:             "strategy": strategy,
 871:             "correlated_held": held_correlated,
 872:             "message": (
 873:                 f"{ticker}: correlated with held position(s) {', '.join(held_correlated)} "
 874:                 f"in {strategy} portfolio"
 875:             ),
 876:         }
 877:     return None
 878: 
 879: 
 880: def check_atr_stop_proximity(ticker, action, portfolio, agent_summary):
 881:     """Check if current price is within 1x ATR of the computed stop level.
 882: 
 883:     This flags positions that are dangerously close to their stop-loss.
 884: 
 885:     Returns:
 886:         dict with flag info, or None if no proximity risk.
 887:     """
 888:     if action == "HOLD":
 889:         return None
 890: 
 891:     holdings = portfolio.get("holdings", {})
 892:     pos = holdings.get(ticker, {})
 893:     shares = pos.get("shares", 0)
 894:     if shares <= 0:
 895:         return None
 896: 
 897:     entry_price = pos.get("avg_cost_usd", 0)
 898:     if entry_price <= 0:
 899:         return None
 900: 
 901:     sig = agent_summary.get("signals", {}).get(ticker, {})
 902:     current_price = sig.get("price_usd", 0)
 903:     atr_pct = sig.get("atr_pct", 0)
 904: 
 905:     if current_price <= 0 or atr_pct <= 0:
 906:         return None
 907: 
 908:     # 2x ATR stop level
 909:     stop_price = entry_price * (1 - 2 * atr_pct / 100)
 910:     # Distance from current price to stop (in ATR units)
 911:     atr_value = current_price * atr_pct / 100
 912:     if atr_value <= 0:
 913:         return None
 914: 
 915:     distance_to_stop = current_price - stop_price
 916:     distance_in_atr = distance_to_stop / atr_value
 917: 
 918:     if distance_in_atr < 1.0:
 919:         return {
 920:             "flag": "atr_stop_proximity",
 921:             "severity": "warning",
 922:             "ticker": ticker,
 923:             "current_price": round(current_price, 4),
 924:             "stop_price": round(stop_price, 4),
 925:             "distance_atr": round(distance_in_atr, 2),
 926:             "message": (
 927:                 f"{ticker}: price ${current_price:.2f} is {distance_in_atr:.1f}x ATR "
 928:                 f"from stop ${stop_price:.2f} (danger zone < 1.0x ATR)"
 929:             ),
 930:         }
 931:     return None
 932: 
 933: 
 934: def compute_all_risk_flags(signals, patient_pf, bold_pf, agent_summary, config=None):
 935:     """Compute all risk audit flags for all tickers.
 936: 
 937:     Args:
 938:         signals: Dict of ticker -> signal data (from agent_summary).
 939:         patient_pf: Patient portfolio state dict.
 940:         bold_pf: Bold portfolio state dict.
 941:         agent_summary: Full agent_summary dict.
 942:         config: Optional config dict.
 943: 
 944:     Returns:
 945:         dict with:
 946:             - flags: list of flag dicts
 947:             - summary: str (human-readable summary)
 948:     """
 949:     cfg = (config or {}).get("risk_audit", {})
 950:     if cfg.get("enabled") is False:
 951:         return {"flags": [], "summary": "Risk audit disabled"}
 952: 
 953:     all_flags = []
 954: 
 955:     for ticker, sig in signals.items():
 956:         action = sig.get("action", "HOLD")
 957:         if action == "HOLD":
 958:             # Still check ATR proximity for held positions
 959:             for strategy, pf in [("patient", patient_pf), ("bold", bold_pf)]:
 960:                 flag = check_atr_stop_proximity(ticker, "CHECK", pf, agent_summary)
 961:                 if flag:
 962:                     flag["strategy"] = strategy
 963:                     all_flags.append(flag)
 964:             continue
 965: 
 966:         for strategy, pf in [("patient", patient_pf), ("bold", bold_pf)]:
 967:             # Concentration
 968:             flag = check_concentration_risk(ticker, action, pf, agent_summary, strategy)
 969:             if flag:
 970:                 all_flags.append(flag)
 971: 
 972:             # Correlation
 973:             flag = check_correlation_risk(ticker, action, pf, strategy)
 974:             if flag:
 975:                 all_flags.append(flag)
 976: 
 977:             # ATR stop proximity
 978:             flag = check_atr_stop_proximity(ticker, action, pf, agent_summary)
 979:             if flag:
 980:                 flag["strategy"] = strategy
 981:                 all_flags.append(flag)
 982: 
 983:         # Regime mismatch (independent of strategy)
 984:         flag = check_regime_mismatch(ticker, action, agent_summary)
 985:         if flag:
 986:             all_flags.append(flag)
 987: 
 988:     summary_parts = []
 989:     if all_flags:
 990:         by_flag = {}
 991:         for f in all_flags:
 992:             by_flag.setdefault(f["flag"], []).append(f)
 993:         for flag_name, flags in by_flag.items():
 994:             summary_parts.append(f"{flag_name}: {len(flags)}")
 995: 
 996:     return {
 997:         "flags": all_flags,
 998:         "summary": "; ".join(summary_parts) if summary_parts else "All clear",
 999:     }

 succeeded in 680ms:
   1: """Portfolio state management — load, save, atomic writes, value calculation."""
   2: 
   3: import logging
   4: import math
   5: import shutil
   6: import threading
   7: from datetime import UTC, datetime
   8: from pathlib import Path
   9: 
  10: logger = logging.getLogger("portfolio.portfolio_mgr")
  11: 
  12: from portfolio.file_utils import atomic_write_json as _atomic_write_json
  13: from portfolio.file_utils import load_json
  14: 
  15: BASE_DIR = Path(__file__).resolve().parent.parent
  16: DATA_DIR = BASE_DIR / "data"
  17: STATE_FILE = DATA_DIR / "portfolio_state.json"
  18: BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
  19: INITIAL_CASH_SEK = 500_000
  20: 
  21: _DEFAULT_STATE = {
  22:     "cash_sek": INITIAL_CASH_SEK,
  23:     "holdings": {},
  24:     "transactions": [],
  25:     "initial_value_sek": INITIAL_CASH_SEK,
  26: }
  27: 
  28: # C8: Per-file locks for concurrency safety
  29: _state_locks: dict[str, threading.Lock] = {}
  30: _locks_lock = threading.Lock()
  31: 
  32: _MAX_BACKUPS = 3  # Rolling backup count
  33: 
  34: 
  35: def _get_lock(path: Path) -> threading.Lock:
  36:     """Get or create a per-file lock for concurrency safety (C8)."""
  37:     key = str(path)
  38:     with _locks_lock:
  39:         if key not in _state_locks:
  40:             _state_locks[key] = threading.Lock()
  41:         return _state_locks[key]
  42: 
  43: 
  44: def _rotate_backups(path: Path):
  45:     """C7: Create rolling .bak backups before overwriting state.
  46: 
  47:     Maintains up to _MAX_BACKUPS copies: path.bak, path.bak2, path.bak3.
  48:     Only backs up if the file exists and has content.
  49:     """
  50:     if not path.exists() or path.stat().st_size == 0:
  51:         return
  52:     try:
  53:         # Rotate existing backups: .bak2 → .bak3, .bak → .bak2
  54:         for i in range(_MAX_BACKUPS, 1, -1):
  55:             src = path.with_suffix(f".json.bak{i - 1}" if i > 2 else ".json.bak")
  56:             dst = path.with_suffix(f".json.bak{i}")
  57:             if src.exists():
  58:                 shutil.copy2(str(src), str(dst))
  59:         # Current file → .bak
  60:         shutil.copy2(str(path), str(path.with_suffix(".json.bak")))
  61:     except OSError as e:
  62:         logger.warning("Failed to rotate backups for %s: %s", path.name, e)
  63: 
  64: 
  65: def _validated_state(loaded):
  66:     """Merge loaded state with defaults to ensure all required keys exist."""
  67:     if not loaded or not isinstance(loaded, dict):
  68:         return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}
  69:     result = {**_DEFAULT_STATE, **loaded}
  70:     # Ensure types are correct for critical fields
  71:     if not isinstance(result.get("holdings"), dict):
  72:         result["holdings"] = {}
  73:     if not isinstance(result.get("transactions"), list):
  74:         result["transactions"] = []
  75:     return result
  76: 
  77: 
  78: def _load_state_from(path: Path):
  79:     """Load portfolio state from a specific file.
  80: 
  81:     C7: On corruption, logs CRITICAL and attempts recovery from backups.
  82:     Returns validated defaults only if file AND all backups are missing/corrupt.
  83:     """
  84:     loaded = load_json(str(path), default=None)
  85:     if loaded is not None:
  86:         return _validated_state(loaded)
  87: 
  88:     # File is missing or corrupt — check if the file exists (corruption vs missing)
  89:     if path.exists():
  90:         logger.critical(
  91:             "CORRUPT portfolio state file: %s — attempting backup recovery", path.name
  92:         )
  93:         # Try backups in order
  94:         for i in range(1, _MAX_BACKUPS + 1):
  95:             bak = path.with_suffix(f".json.bak{i}" if i > 1 else ".json.bak")
  96:             if bak.exists():
  97:                 loaded = load_json(str(bak), default=None)
  98:                 if loaded is not None:
  99:                     logger.warning("Recovered %s from backup %s", path.name, bak.name)
 100:                     return _validated_state(loaded)
 101:         logger.critical(
 102:             "ALL backups corrupt/missing for %s — returning fresh defaults", path.name
 103:         )
 104: 
 105:     return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}
 106: 
 107: 
 108: def _save_state_to(path: Path, state):
 109:     """Save state with rolling backup (C7) and lock (C8)."""
 110:     lock = _get_lock(path)
 111:     with lock:
 112:         _rotate_backups(path)
 113:         _atomic_write_json(path, state)
 114: 
 115: 
 116: def load_state():
 117:     """Load Patient portfolio state. Returns validated defaults if missing or corrupt."""
 118:     return _load_state_from(STATE_FILE)
 119: 
 120: 
 121: def save_state(state):
 122:     """Save Patient portfolio state with backup rotation (C7)."""
 123:     _save_state_to(STATE_FILE, state)
 124: 
 125: 
 126: def load_bold_state():
 127:     """Load Bold portfolio state. Returns validated defaults if missing or corrupt."""
 128:     return _load_state_from(BOLD_STATE_FILE)
 129: 
 130: 
 131: def save_bold_state(state):
 132:     """Save Bold portfolio state with backup rotation (C7)."""
 133:     _save_state_to(BOLD_STATE_FILE, state)
 134: 
 135: 
 136: def update_state(mutate_fn, bold=False):
 137:     """Atomic read-modify-write for portfolio state (C8).
 138: 
 139:     Holds a lock for the entire read-modify-write cycle to prevent
 140:     concurrent callers from overwriting each other's mutations.
 141: 
 142:     Args:
 143:         mutate_fn: Callable that receives the current state dict and mutates it.
 144:             The function should modify the dict in-place and optionally return it.
 145:         bold: If True, operates on the Bold portfolio instead of Patient.
 146: 
 147:     Returns:
 148:         The updated state dict.
 149:     """
 150:     path = BOLD_STATE_FILE if bold else STATE_FILE
 151:     lock = _get_lock(path)
 152:     with lock:
 153:         state = _load_state_from(path)
 154:         result = mutate_fn(state)
 155:         if result is not None:
 156:             state = result
 157:         _rotate_backups(path)
 158:         _atomic_write_json(path, state)
 159:     return state
 160: 
 161: 
 162: def portfolio_value(state, prices_usd, fx_rate):
 163:     if not isinstance(fx_rate, (int, float)) or not math.isfinite(fx_rate) or fx_rate <= 0:
 164:         logger.warning("portfolio_value: invalid fx_rate=%r, returning cash only", fx_rate)
 165:         return state.get("cash_sek", 0)
 166:     total = state.get("cash_sek", 0)
 167:     for ticker, h in state.get("holdings", {}).items():
 168:         try:
 169:             shares = h.get("shares", 0)
 170:             price = prices_usd.get(ticker)
 171:             if shares > 0 and price is not None and price > 0:
 172:                 total += shares * price * fx_rate
 173:             elif shares > 0 and (price is None or price <= 0):
 174:                 logger.warning(
 175:                     "portfolio_value: invalid price for %s: %r (shares=%s)",
 176:                     ticker, price, shares,
 177:                 )
 178:         except (TypeError, ValueError, AttributeError) as e:
 179:             logger.warning("portfolio_value: error calculating %s: %s", ticker, e)
 180:     return total

 succeeded in 666ms:
   1: """Warrant portfolio tracking — leverage-aware P&L for Avanza warrants.
   2: 
   3: Tracks actual warrant positions with leverage-multiplied P&L based on
   4: the underlying instrument's price movement.
   5: """
   6: 
   7: import logging
   8: from datetime import UTC
   9: from pathlib import Path
  10: 
  11: from portfolio.file_utils import atomic_write_json, load_json
  12: 
  13: logger = logging.getLogger("portfolio.warrant_portfolio")
  14: 
  15: BASE_DIR = Path(__file__).resolve().parent.parent
  16: DATA_DIR = BASE_DIR / "data"
  17: WARRANT_STATE_FILE = DATA_DIR / "portfolio_state_warrants.json"
  18: 
  19: _DEFAULT_STATE = {
  20:     "holdings": {},
  21:     "transactions": [],
  22: }
  23: 
  24: 
  25: def load_warrant_state():
  26:     """Load warrant portfolio state from disk.
  27: 
  28:     Returns:
  29:         dict with "holdings" and "transactions" keys.
  30:     """
  31:     state = load_json(WARRANT_STATE_FILE)
  32:     if state is None:
  33:         return _DEFAULT_STATE.copy()
  34:     # Ensure required keys exist
  35:     if "holdings" not in state:
  36:         state["holdings"] = {}
  37:     if "transactions" not in state:
  38:         state["transactions"] = []
  39:     return state
  40: 
  41: 
  42: def save_warrant_state(state):
  43:     """Atomically write warrant portfolio state.
  44: 
  45:     Args:
  46:         state: dict with "holdings" and "transactions".
  47:     """
  48:     atomic_write_json(WARRANT_STATE_FILE, state)
  49:     logger.info("Warrant state saved (%d holdings)", len(state.get("holdings", {})))
  50: 
  51: 
  52: def warrant_pnl(holding, current_underlying_usd, fx_rate):
  53:     """Compute P&L for a single warrant position.
  54: 
  55:     Uses the underlying price change multiplied by leverage factor.
  56: 
  57:     Args:
  58:         holding: dict with keys:
  59:             - units: number of warrant units held
  60:             - entry_price_sek: price per unit at entry (SEK)
  61:             - underlying: underlying ticker (e.g., "XAG-USD")
  62:             - leverage: leverage factor (e.g., 5 for 5x)
  63:             - underlying_entry_price_usd: underlying price at entry (USD)
  64:             - name: human-readable name (optional)
  65:         current_underlying_usd: current price of the underlying in USD.
  66:         fx_rate: current USD/SEK exchange rate.
  67: 
  68:     Returns:
  69:         dict: {
  70:             "pnl_pct": float (percentage P&L),
  71:             "pnl_sek": float (absolute P&L in SEK),
  72:             "current_implied_sek": float (current implied value per unit),
  73:             "total_value_sek": float (total current value),
  74:             "entry_value_sek": float (total entry value),
  75:             "underlying_change_pct": float (underlying price change %),
  76:             "source": "implied"
  77:         }
  78:         Returns None if required data is missing.
  79:     """
  80:     if not holding or not current_underlying_usd or not fx_rate:
  81:         return None
  82: 
  83:     units = holding.get("units", 0)
  84:     entry_price_sek = holding.get("entry_price_sek", 0)
  85:     leverage = holding.get("leverage", 1)
  86:     underlying_entry = holding.get("underlying_entry_price_usd", 0)
  87: 
  88:     if not units or not entry_price_sek or not underlying_entry:
  89:         return None
  90: 
  91:     # Underlying change
  92:     underlying_change = (current_underlying_usd - underlying_entry) / underlying_entry
  93:     underlying_change_pct = round(underlying_change * 100, 2)
  94: 
  95:     # Implied warrant P&L = underlying change * leverage
  96:     implied_pnl_pct = underlying_change * leverage
  97:     implied_pnl_pct_rounded = round(implied_pnl_pct * 100, 2)
  98: 
  99:     # Current implied value
 100:     current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)
 101:     total_value_sek = current_implied_sek * units
 102:     entry_value_sek = entry_price_sek * units
 103:     pnl_sek = round(total_value_sek - entry_value_sek, 2)
 104: 
 105:     return {
 106:         "pnl_pct": implied_pnl_pct_rounded,
 107:         "pnl_sek": pnl_sek,
 108:         "current_implied_sek": round(current_implied_sek, 2),
 109:         "total_value_sek": round(total_value_sek, 2),
 110:         "entry_value_sek": round(entry_value_sek, 2),
 111:         "underlying_change_pct": underlying_change_pct,
 112:         "source": "implied",
 113:     }
 114: 
 115: 
 116: def get_warrant_summary(prices_usd, fx_rate):
 117:     """Build a summary of all warrant positions with current P&L.
 118: 
 119:     Args:
 120:         prices_usd: dict {ticker: price_usd} for all instruments.
 121:         fx_rate: current USD/SEK exchange rate.
 122: 
 123:     Returns:
 124:         dict: {
 125:             "positions": {
 126:                 config_key: {
 127:                     "name": "MINI L SILVER AVA 140",
 128:                     "underlying": "XAG-USD",
 129:                     "leverage": 5,
 130:                     "units": 100,
 131:                     "pnl": { ... warrant_pnl output ... },
 132:                 }
 133:             },
 134:             "total_value_sek": float,
 135:             "total_pnl_sek": float,
 136:         }
 137:     """
 138:     state = load_warrant_state()
 139:     holdings = state.get("holdings", {})
 140: 
 141:     if not holdings:
 142:         return {"positions": {}, "total_value_sek": 0, "total_pnl_sek": 0}
 143: 
 144:     positions = {}
 145:     total_value = 0.0
 146:     total_pnl = 0.0
 147: 
 148:     for key, holding in holdings.items():
 149:         underlying = holding.get("underlying")
 150:         if not underlying:
 151:             continue
 152: 
 153:         current_price = prices_usd.get(underlying)
 154:         if not current_price:
 155:             continue
 156: 
 157:         pnl = warrant_pnl(holding, current_price, fx_rate)
 158: 
 159:         position = {
 160:             "name": holding.get("name", key),
 161:             "underlying": underlying,
 162:             "leverage": holding.get("leverage", 1),
 163:             "units": holding.get("units", 0),
 164:         }
 165: 
 166:         if pnl:
 167:             position["pnl"] = pnl
 168:             total_value += pnl["total_value_sek"]
 169:             total_pnl += pnl["pnl_sek"]
 170:         else:
 171:             position["pnl"] = None
 172: 
 173:         positions[key] = position
 174: 
 175:     return {
 176:         "positions": positions,
 177:         "total_value_sek": round(total_value, 2),
 178:         "total_pnl_sek": round(total_pnl, 2),
 179:     }
 180: 
 181: 
 182: def record_warrant_transaction(config_key, action, units, price_sek, underlying_price_usd,
 183:                                 leverage, name=None, underlying=None):
 184:     """Record a warrant buy/sell transaction.
 185: 
 186:     Args:
 187:         config_key: Warrant config key (e.g., "MINI-SILVER").
 188:         action: "BUY" or "SELL".
 189:         units: Number of units.
 190:         price_sek: Price per unit in SEK.
 191:         underlying_price_usd: Underlying price at transaction time.
 192:         leverage: Leverage factor.
 193:         name: Human-readable name (optional).
 194:         underlying: Underlying ticker (optional).
 195:     """
 196:     from datetime import datetime
 197: 
 198:     state = load_warrant_state()
 199: 
 200:     txn = {
 201:         "timestamp": datetime.now(UTC).isoformat(),
 202:         "config_key": config_key,
 203:         "action": action,
 204:         "units": units,
 205:         "price_sek": price_sek,
 206:         "underlying_price_usd": underlying_price_usd,
 207:         "leverage": leverage,
 208:     }
 209:     if name:
 210:         txn["name"] = name
 211:     if underlying:
 212:         txn["underlying"] = underlying
 213: 
 214:     state["transactions"].append(txn)
 215: 
 216:     holdings = state["holdings"]
 217:     if action == "BUY":
 218:         if config_key in holdings:
 219:             # Average in
 220:             existing = holdings[config_key]
 221:             old_units = existing.get("units", 0)
 222:             old_price = existing.get("entry_price_sek", 0)
 223:             new_units = old_units + units
 224:             if new_units > 0:
 225:                 avg_price = (old_units * old_price + units * price_sek) / new_units
 226:                 existing["units"] = new_units
 227:                 existing["entry_price_sek"] = round(avg_price, 2)
 228: 
 229:                 # PR-P1-1 (2026-05-02): also volume-weight the underlying
 230:                 # entry. Previously this stayed pinned to the FIRST entry's
 231:                 # spot price, causing the metals stop-loss reference (read
 232:                 # from `underlying_entry_price_usd` via warrant_pnl()) to
 233:                 # trip earlier than the VWAP entry implied — partial-add
 234:                 # positions would hard-stop on small underlying drawdowns.
 235:                 # Defensive: if either price is zero, fall back gracefully.
 236:                 # See tests/test_warrant_portfolio.TestWarrantAvgInUnderlyingEntry.
 237:                 old_underlying = existing.get("underlying_entry_price_usd", 0) or 0
 238:                 new_underlying = underlying_price_usd or 0
 239:                 if old_underlying > 0 and new_underlying > 0:
 240:                     avg_underlying = (
 241:                         old_units * old_underlying + units * new_underlying
 242:                     ) / new_units
 243:                     existing["underlying_entry_price_usd"] = round(avg_underlying, 4)
 244:                 elif new_underlying > 0:
 245:                     # Existing was 0/missing — adopt the new valid spot.
 246:                     existing["underlying_entry_price_usd"] = round(new_underlying, 4)
 247:                 # else: both zero — leave existing alone (degenerate input).
 248:         else:
 249:             holdings[config_key] = {
 250:                 "units": units,
 251:                 "entry_price_sek": price_sek,
 252:                 "underlying": underlying or "",
 253:                 "leverage": leverage,
 254:                 "underlying_entry_price_usd": underlying_price_usd,
 255:                 "name": name or config_key,
 256:             }
 257:     elif action == "SELL" and config_key in holdings:
 258:         existing = holdings[config_key]
 259:         remaining = existing.get("units", 0) - units
 260:         if remaining <= 0:
 261:             del holdings[config_key]
 262:         else:
 263:             existing["units"] = remaining
 264: 
 265:     save_warrant_state(state)
 266:     logger.info("Warrant %s %s: %d units @ %.2f SEK", action, config_key, units, price_sek)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for name in ['circuit_breaker.py','cost_model.py','equity_curve.py','kelly_sizing.py','monte_carlo.py','monte_carlo_risk.py','portfolio_validator.py','trade_guards.py','trade_risk_classifier.py','trade_validation.py']:
    p=Path('portfolio')/name
    print('\\n### FILE', name)
    for i,line in enumerate(p.read_text().splitlines(),1):
        print(f\"{i:4}: {line}\")
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 716ms:

### FILE circuit_breaker.py
   1: """Circuit breaker for data source API calls.
   2: 
   3: Prevents repeated calls to failing APIs. States:
   4:   CLOSED  — normal operation, requests pass through
   5:   OPEN    — API is failing, requests blocked until recovery timeout
   6:   HALF_OPEN — testing recovery, one request allowed through
   7: """
   8: 
   9: import enum
  10: import logging
  11: import threading
  12: import time
  13: 
  14: logger = logging.getLogger("portfolio.circuit_breaker")
  15: 
  16: 
  17: class State(enum.Enum):
  18:     CLOSED = "CLOSED"
  19:     OPEN = "OPEN"
  20:     HALF_OPEN = "HALF_OPEN"
  21: 
  22: 
  23: class CircuitBreaker:
  24:     """Thread-safe circuit breaker for a single data source."""
  25: 
  26:     def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60,
  27:                  max_recovery_timeout: int = 300):
  28:         self.name = name
  29:         self.failure_threshold = failure_threshold
  30:         self.recovery_timeout = recovery_timeout
  31:         self._base_recovery_timeout = recovery_timeout
  32:         self._max_recovery_timeout = max_recovery_timeout
  33:         self._state = State.CLOSED
  34:         self._failure_count = 0
  35:         self._last_failure_time: float | None = None
  36:         self._lock = threading.Lock()
  37:         self._half_open_probe_sent = False  # BUG-93: Only one request in HALF_OPEN
  38: 
  39:     @property
  40:     def state(self) -> State:
  41:         return self._state
  42: 
  43:     def record_success(self) -> None:
  44:         """Record a successful request. Resets failure count; HALF_OPEN -> CLOSED."""
  45:         with self._lock:
  46:             if self._state == State.HALF_OPEN:
  47:                 logger.info("Circuit breaker '%s': HALF_OPEN -> CLOSED (recovery confirmed)", self.name)
  48:                 self._state = State.CLOSED
  49:                 self._half_open_probe_sent = False  # BUG-93: Reset probe flag
  50:                 # BUG-245: Reset backoff on successful recovery
  51:                 self.recovery_timeout = self._base_recovery_timeout
  52:             self._failure_count = 0
  53: 
  54:     def record_failure(self) -> None:
  55:         """Record a failed request. Increments count; CLOSED -> OPEN at threshold, HALF_OPEN -> OPEN."""
  56:         with self._lock:
  57:             self._failure_count += 1
  58:             self._last_failure_time = time.monotonic()
  59: 
  60:             if self._state == State.HALF_OPEN:
  61:                 # BUG-245: Exponential backoff — double timeout on each failed
  62:                 # recovery probe, capped at max. Reduces retry pressure during
  63:                 # extended outages (e.g., Binance maintenance windows).
  64:                 prev_timeout = self.recovery_timeout
  65:                 self.recovery_timeout = min(
  66:                     self.recovery_timeout * 2, self._max_recovery_timeout
  67:                 )
  68:                 logger.warning(
  69:                     "Circuit breaker '%s': HALF_OPEN -> OPEN (recovery failed, %d failures, "
  70:                     "next probe in %ds, was %ds)",
  71:                     self.name, self._failure_count, self.recovery_timeout, prev_timeout,
  72:                 )
  73:                 self._state = State.OPEN
  74:                 self._half_open_probe_sent = False  # BUG-93: Reset probe flag
  75:             elif self._state == State.CLOSED and self._failure_count >= self.failure_threshold:
  76:                 logger.warning(
  77:                     "Circuit breaker '%s': CLOSED -> OPEN (threshold %d reached)",
  78:                     self.name, self.failure_threshold,
  79:                 )
  80:                 self._state = State.OPEN
  81: 
  82:     def allow_request(self) -> bool:
  83:         """Return True if a request should proceed."""
  84:         with self._lock:
  85:             if self._state == State.CLOSED:
  86:                 return True
  87: 
  88:             if self._state == State.OPEN:
  89:                 if self._last_failure_time is None:
  90:                     return False
  91:                 elapsed = time.monotonic() - self._last_failure_time
  92:                 if elapsed >= self.recovery_timeout:
  93:                     logger.info(
  94:                         "Circuit breaker '%s': OPEN -> HALF_OPEN (%.1fs elapsed, testing recovery)",
  95:                         self.name, elapsed,
  96:                     )
  97:                     self._state = State.HALF_OPEN
  98:                     self._half_open_probe_sent = True  # BUG-93: This IS the probe
  99:                     return True
 100:                 return False
 101: 
 102:             # BUG-93/BUG-187: HALF_OPEN — the probe request is always sent via
 103:             # the OPEN→HALF_OPEN transition above (which sets probe_sent=True and
 104:             # returns True). This branch handles the case where a second request
 105:             # arrives while still in HALF_OPEN (waiting for success/failure).
 106:             return False
 107: 
 108:     def get_status(self) -> dict:
 109:         """Return current circuit breaker status."""
 110:         with self._lock:
 111:             return {
 112:                 "name": self.name,
 113:                 "state": self._state.value,
 114:                 "failure_count": self._failure_count,
 115:                 "last_failure_time": self._last_failure_time,
 116:             }
 117: 
 118:     def reset(self) -> None:
 119:         """Force the breaker back to CLOSED with zero failures.
 120: 
 121:         Intended use: operational override (manual recovery) and test
 122:         isolation. Production code should NOT call this in normal flow
 123:         — let record_success/record_failure drive the state machine.
 124: 
 125:         2026-05-02: added when test_consensus xdist flakes traced back
 126:         to module-level breakers tripping during one test and leaking
 127:         into the next on the same xdist worker.
 128:         """
 129:         with self._lock:
 130:             self._state = State.CLOSED
 131:             self._failure_count = 0
 132:             self._last_failure_time = None
 133:             self._half_open_probe_sent = False
 134:             self.recovery_timeout = self._base_recovery_timeout

### FILE cost_model.py
   1: """Cost model for trade execution — fees, spread, and slippage.
   2: 
   3: Provides instrument-specific cost estimation for the exit optimizer.
   4: Supports Avanza warrants, stocks, and crypto exchanges.
   5: 
   6: Usage:
   7:     from portfolio.cost_model import get_cost_model
   8:     costs = get_cost_model("warrant")
   9:     exit_cost = costs.total_cost_sek(trade_value_sek=50000)
  10: """
  11: 
  12: from __future__ import annotations
  13: 
  14: from dataclasses import dataclass
  15: 
  16: 
  17: @dataclass(frozen=True)
  18: class CostModel:
  19:     """All-in cost model for a single trade (one leg).
  20: 
  21:     Attributes:
  22:         courtage_bps: Brokerage fee in basis points (e.g., 6.9 = 0.069%).
  23:         min_fee_sek: Minimum fee in SEK (courtage floor).
  24:         spread_bps: Half-spread cost in basis points. For a sell, you cross
  25:             from mid to bid, losing half the spread.
  26:         slippage_bps: Expected adverse price movement in basis points
  27:             between decision and fill (market impact + latency).
  28:         label: Human-readable name for this cost model.
  29:     """
  30:     courtage_bps: float = 0.0
  31:     min_fee_sek: float = 0.0
  32:     spread_bps: float = 0.0
  33:     slippage_bps: float = 0.0
  34:     label: str = "default"
  35: 
  36:     def total_cost_sek(self, trade_value_sek: float) -> float:
  37:         """Compute total one-way cost for a trade of given value.
  38: 
  39:         Returns:
  40:             Total cost in SEK (always non-negative).
  41:         """
  42:         if trade_value_sek <= 0:
  43:             return 0.0
  44:         courtage = max(trade_value_sek * self.courtage_bps / 10_000, self.min_fee_sek)
  45:         spread = trade_value_sek * self.spread_bps / 10_000
  46:         slippage = trade_value_sek * self.slippage_bps / 10_000
  47:         return courtage + spread + slippage
  48: 
  49:     def total_cost_pct(self) -> float:
  50:         """Total cost as a percentage of trade value (excluding min fee)."""
  51:         return (self.courtage_bps + self.spread_bps + self.slippage_bps) / 100.0
  52: 
  53:     def round_trip_pct(self) -> float:
  54:         """Round-trip cost (buy + sell) as a percentage."""
  55:         return self.total_cost_pct() * 2
  56: 
  57: 
  58: # ---------------------------------------------------------------------------
  59: # Preset cost models for known instrument types
  60: # ---------------------------------------------------------------------------
  61: 
  62: # Avanza warrants/certificates: 0 courtage on many, spread is the real cost.
  63: # Typical MINI silver spread: 0.6-1.0% (30-50 bps half-spread).
  64: WARRANT_COSTS = CostModel(
  65:     courtage_bps=0.0,
  66:     min_fee_sek=0.0,
  67:     spread_bps=40.0,    # 0.40% half-spread (conservative)
  68:     slippage_bps=10.0,   # 0.10% slippage on market orders
  69:     label="avanza_warrant",
  70: )
  71: 
  72: # Avanza stocks (Mini courtage class): 0.069% with 1 SEK minimum
  73: STOCK_COSTS = CostModel(
  74:     courtage_bps=6.9,
  75:     min_fee_sek=1.0,
  76:     spread_bps=5.0,      # 0.05% half-spread (liquid US stocks)
  77:     slippage_bps=2.0,     # 0.02% slippage
  78:     label="avanza_stock",
  79: )
  80: 
  81: # Crypto (Binance-equivalent fees, used for simulated portfolio)
  82: CRYPTO_COSTS = CostModel(
  83:     courtage_bps=5.0,     # 0.05% taker fee
  84:     min_fee_sek=0.0,
  85:     spread_bps=5.0,       # 0.05% half-spread
  86:     slippage_bps=5.0,     # 0.05% slippage
  87:     label="crypto",
  88: )
  89: 
  90: # Elongir silver bot (specific spread from config)
  91: ELONGIR_COSTS = CostModel(
  92:     courtage_bps=25.0,    # 0.25% commission
  93:     min_fee_sek=0.0,
  94:     spread_bps=40.0,      # 0.40% half-spread
  95:     slippage_bps=10.0,    # 0.10% slippage
  96:     label="elongir_silver",
  97: )
  98: 
  99: _COST_MODELS = {
 100:     "warrant": WARRANT_COSTS,
 101:     "stock": STOCK_COSTS,
 102:     "crypto": CRYPTO_COSTS,
 103:     "elongir": ELONGIR_COSTS,
 104: }
 105: 
 106: 
 107: def get_cost_model(instrument_type: str) -> CostModel:
 108:     """Look up cost model by instrument type.
 109: 
 110:     Args:
 111:         instrument_type: One of "warrant", "stock", "crypto", "elongir".
 112: 
 113:     Returns:
 114:         CostModel for the instrument type. Falls back to STOCK_COSTS if unknown.
 115:     """
 116:     return _COST_MODELS.get(instrument_type, STOCK_COSTS)

### FILE equity_curve.py
   1: """Equity curve analysis and portfolio metrics.
   2: 
   3: Loads portfolio value history from the JSONL log and computes performance
   4: metrics useful for charting and strategy comparison.
   5: """
   6: 
   7: import datetime
   8: import math
   9: import pathlib
  10: from collections import defaultdict
  11: 
  12: from portfolio.file_utils import load_jsonl
  13: 
  14: DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
  15: 
  16: DEFAULT_HISTORY_PATH = DATA_DIR / "portfolio_value_history.jsonl"
  17: INITIAL_VALUE = 500_000  # SEK
  18: RISK_FREE_RATE_ANNUAL = 0.035  # 3.5% Swedish risk-free rate (approximate)
  19: 
  20: # The portfolio loop runs 24/7 (crypto always active), so
  21: # portfolio_value_history.jsonl has entries every calendar day.
  22: # Annualization must use 365, not 252 (stock-only convention).
  23: ANNUALIZATION_DAYS = 365
  24: 
  25: 
  26: def load_equity_curve(path: str | None = None) -> list[dict]:
  27:     """Load portfolio value history for charting.
  28: 
  29:     Reads the JSONL file and returns a list of dicts sorted by timestamp.
  30:     Each dict contains:
  31:         - ts: ISO-8601 timestamp
  32:         - patient_value_sek: float
  33:         - bold_value_sek: float
  34:         - patient_pnl_pct: float
  35:         - bold_pnl_pct: float
  36:         - fx_rate: float
  37:         - prices: dict of ticker -> USD price
  38: 
  39:     Args:
  40:         path: Path to the portfolio_value_history.jsonl file.
  41:             Defaults to data/portfolio_value_history.jsonl.
  42: 
  43:     Returns:
  44:         list of dicts sorted by timestamp (oldest first).
  45:         Empty list if file doesn't exist or is empty.
  46:     """
  47:     if path is None:
  48:         path = str(DEFAULT_HISTORY_PATH)
  49: 
  50:     result = load_jsonl(path)
  51: 
  52:     # Sort by timestamp
  53:     result.sort(key=lambda x: x.get("ts", ""))
  54:     return result
  55: 
  56: 
  57: def _parse_ts(ts_str: str) -> datetime.datetime:
  58:     """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
  59:     dt = datetime.datetime.fromisoformat(ts_str)
  60:     if dt.tzinfo is None:
  61:         dt = dt.replace(tzinfo=datetime.UTC)
  62:     return dt
  63: 
  64: 
  65: def _daily_returns(curve: list[dict], value_key: str) -> list[float]:
  66:     """Compute daily returns from the equity curve.
  67: 
  68:     Groups entries by date, takes the last entry per day, and computes
  69:     day-over-day percentage returns.
  70: 
  71:     Args:
  72:         curve: Sorted equity curve list.
  73:         value_key: Either "patient_value_sek" or "bold_value_sek".
  74: 
  75:     Returns:
  76:         list of daily return percentages.
  77:     """
  78:     if not curve:
  79:         return []
  80: 
  81:     # Group by date, take last value per day
  82:     daily_values = {}
  83:     for entry in curve:
  84:         ts_str = entry.get("ts", "")
  85:         if not ts_str:
  86:             continue
  87:         try:
  88:             dt = _parse_ts(ts_str)
  89:             date_key = dt.date()
  90:             value = entry.get(value_key, 0)
  91:             if value > 0:
  92:                 daily_values[date_key] = value
  93:         except (ValueError, TypeError):
  94:             continue
  95: 
  96:     if len(daily_values) < 2:
  97:         return []
  98: 
  99:     sorted_dates = sorted(daily_values.keys())
 100:     returns = []
 101:     for i in range(1, len(sorted_dates)):
 102:         prev_val = daily_values[sorted_dates[i - 1]]
 103:         curr_val = daily_values[sorted_dates[i]]
 104:         if prev_val > 0:
 105:             daily_ret = ((curr_val - prev_val) / prev_val) * 100
 106:             returns.append(daily_ret)
 107:         else:
 108:             returns.append(0.0)
 109: 
 110:     return returns
 111: 
 112: 
 113: def compute_metrics(curve: list[dict], strategy: str) -> dict:
 114:     """Compute portfolio metrics from equity curve.
 115: 
 116:     Args:
 117:         curve: List of equity curve entries (from load_equity_curve).
 118:         strategy: "patient" or "bold".
 119: 
 120:     Returns:
 121:         dict with:
 122:             - max_drawdown_pct: float -- maximum peak-to-trough drawdown
 123:             - sharpe_ratio: float -- annualized Sharpe ratio (or None if insufficient data)
 124:             - sortino_ratio: float -- annualized Sortino ratio (or None)
 125:             - win_rate: float -- percentage of positive-return days
 126:             - avg_daily_return_pct: float -- mean daily return
 127:             - best_day_pct: float -- best single-day return
 128:             - worst_day_pct: float -- worst single-day return
 129:             - days_in_drawdown: int -- number of days below previous peak
 130:             - total_return_pct: float -- total return from start to end
 131:             - annualized_return_pct: float -- annualized return (or None if < 1 day)
 132:             - volatility_annual_pct: float -- annualized daily volatility
 133:             - num_data_points: int -- number of entries in curve
 134:             - date_range: tuple of (first_ts, last_ts)
 135:     """
 136:     value_key = f"{strategy}_value_sek"
 137: 
 138:     result = {
 139:         "max_drawdown_pct": 0.0,
 140:         "sharpe_ratio": None,
 141:         "sortino_ratio": None,
 142:         "win_rate": 0.0,
 143:         "avg_daily_return_pct": 0.0,
 144:         "best_day_pct": 0.0,
 145:         "worst_day_pct": 0.0,
 146:         "days_in_drawdown": 0,
 147:         "total_return_pct": 0.0,
 148:         "annualized_return_pct": None,
 149:         "volatility_annual_pct": 0.0,
 150:         "num_data_points": len(curve),
 151:         "date_range": None,
 152:     }
 153: 
 154:     if not curve:
 155:         return result
 156: 
 157:     # Extract values
 158:     values = []
 159:     timestamps = []
 160:     for entry in curve:
 161:         val = entry.get(value_key)
 162:         ts = entry.get("ts", "")
 163:         if val is not None and val > 0 and ts:
 164:             values.append(val)
 165:             timestamps.append(ts)
 166: 
 167:     if not values:
 168:         return result
 169: 
 170:     result["num_data_points"] = len(values)
 171:     result["date_range"] = (timestamps[0], timestamps[-1])
 172: 
 173:     # --- Total return ---
 174:     first_val = values[0]
 175:     last_val = values[-1]
 176:     total_return_pct = ((last_val - first_val) / first_val) * 100
 177:     result["total_return_pct"] = round(total_return_pct, 4)
 178: 
 179:     # --- Annualized return ---
 180:     try:
 181:         first_dt = _parse_ts(timestamps[0])
 182:         last_dt = _parse_ts(timestamps[-1])
 183:         days_elapsed = (last_dt - first_dt).total_seconds() / 86400
 184:         if days_elapsed >= 1:
 185:             years = days_elapsed / 365.25
 186:             # Annualized return = (final/initial)^(1/years) - 1
 187:             if first_val > 0 and last_val > 0:
 188:                 annualized = (pow(last_val / first_val, 1 / years) - 1) * 100
 189:                 result["annualized_return_pct"] = round(annualized, 4)
 190:     except (ValueError, TypeError, ZeroDivisionError):
 191:         pass
 192: 
 193:     # --- Maximum drawdown ---
 194:     peak = values[0]
 195:     max_dd = 0.0
 196:     # Group by date for drawdown day counting
 197:     date_was_below = set()
 198: 
 199:     for i, val in enumerate(values):
 200:         if val > peak:
 201:             peak = val
 202:         dd = ((peak - val) / peak) * 100
 203:         if dd > max_dd:
 204:             max_dd = dd
 205:         if dd > 0.01:  # Meaningfully below peak
 206:             try:
 207:                 dt = _parse_ts(timestamps[i])
 208:                 date_was_below.add(dt.date())
 209:             except (ValueError, TypeError):
 210:                 pass
 211: 
 212:     result["max_drawdown_pct"] = round(max_dd, 4)
 213:     result["days_in_drawdown"] = len(date_was_below)
 214: 
 215:     # --- Daily returns ---
 216:     daily_rets = _daily_returns(curve, value_key)
 217: 
 218:     if daily_rets:
 219:         result["avg_daily_return_pct"] = round(sum(daily_rets) / len(daily_rets), 6)
 220:         result["best_day_pct"] = round(max(daily_rets), 4)
 221:         result["worst_day_pct"] = round(min(daily_rets), 4)
 222: 
 223:         # Win rate
 224:         positive_days = sum(1 for r in daily_rets if r > 0)
 225:         result["win_rate"] = round((positive_days / len(daily_rets)) * 100, 2)
 226: 
 227:         # Volatility (annualized)
 228:         if len(daily_rets) >= 2:
 229:             mean_ret = sum(daily_rets) / len(daily_rets)
 230:             variance = sum((r - mean_ret) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
 231:             daily_vol = math.sqrt(variance)
 232:             annual_vol = daily_vol * math.sqrt(ANNUALIZATION_DAYS)
 233:             result["volatility_annual_pct"] = round(annual_vol, 4)
 234: 
 235:             # Sharpe ratio (annualized)
 236:             daily_rf = RISK_FREE_RATE_ANNUAL / ANNUALIZATION_DAYS
 237:             # Convert daily returns to decimal for Sharpe
 238:             daily_rets_dec = [r / 100 for r in daily_rets]
 239:             mean_excess = sum(r - daily_rf for r in daily_rets_dec) / len(daily_rets_dec)
 240:             if daily_vol > 0:
 241:                 # Annualize Sharpe: mean_excess / daily_std * sqrt(ANNUALIZATION_DAYS)
 242:                 # BUG-225: extract mean to avoid O(n^2) recomputation inside generator
 243:                 mean_dec = sum(daily_rets_dec) / len(daily_rets_dec)
 244:                 daily_std_dec = math.sqrt(
 245:                     sum((r - mean_dec) ** 2
 246:                         for r in daily_rets_dec) / (len(daily_rets_dec) - 1)
 247:                 )
 248:                 if daily_std_dec > 0:
 249:                     sharpe = (mean_excess / daily_std_dec) * math.sqrt(ANNUALIZATION_DAYS)
 250:                     result["sharpe_ratio"] = round(sharpe, 4)
 251: 
 252:             # H19: Sortino — divide by TOTAL observations, not just downside count (standard formula)
 253:             squared_devs = [min(r - daily_rf, 0) ** 2 for r in daily_rets_dec]
 254:             downside_dev = math.sqrt(sum(squared_devs) / len(daily_rets_dec))
 255:             if downside_dev > 0:
 256:                 sortino = (mean_excess / downside_dev) * math.sqrt(ANNUALIZATION_DAYS)
 257:                 result["sortino_ratio"] = round(sortino, 4)
 258: 
 259:     return result
 260: 
 261: 
 262: def compare_strategies(curve: list[dict]) -> dict:
 263:     """Compare patient vs bold strategy performance.
 264: 
 265:     Args:
 266:         curve: Equity curve from load_equity_curve().
 267: 
 268:     Returns:
 269:         dict with:
 270:             - patient: metrics dict
 271:             - bold: metrics dict
 272:             - comparison: dict with relative performance
 273:     """
 274:     patient = compute_metrics(curve, "patient")
 275:     bold = compute_metrics(curve, "bold")
 276: 
 277:     comparison = {
 278:         "return_diff_pct": round(patient["total_return_pct"] - bold["total_return_pct"], 4),
 279:         "leader": "patient" if patient["total_return_pct"] > bold["total_return_pct"] else "bold",
 280:         "drawdown_diff_pct": round(patient["max_drawdown_pct"] - bold["max_drawdown_pct"], 4),
 281:         "lower_drawdown": "patient" if patient["max_drawdown_pct"] < bold["max_drawdown_pct"] else "bold",
 282:     }
 283: 
 284:     # Risk-adjusted comparison
 285:     if patient.get("sharpe_ratio") is not None and bold.get("sharpe_ratio") is not None:
 286:         comparison["sharpe_leader"] = "patient" if patient["sharpe_ratio"] > bold["sharpe_ratio"] else "bold"
 287:         comparison["sharpe_diff"] = round(patient["sharpe_ratio"] - bold["sharpe_ratio"], 4)
 288: 
 289:     return {
 290:         "patient": patient,
 291:         "bold": bold,
 292:         "comparison": comparison,
 293:     }
 294: 
 295: 
 296: def get_latest_values(curve: list[dict]) -> dict | None:
 297:     """Get the most recent portfolio values from the curve.
 298: 
 299:     Returns:
 300:         dict with patient_value_sek, bold_value_sek, ts, or None if curve is empty.
 301:     """
 302:     if not curve:
 303:         return None
 304:     latest = curve[-1]
 305:     return {
 306:         "ts": latest.get("ts"),
 307:         "patient_value_sek": latest.get("patient_value_sek"),
 308:         "bold_value_sek": latest.get("bold_value_sek"),
 309:         "patient_pnl_pct": latest.get("patient_pnl_pct"),
 310:         "bold_pnl_pct": latest.get("bold_pnl_pct"),
 311:     }
 312: 
 313: 
 314: def _pair_round_trips(transactions):
 315:     """Match BUY and SELL transactions into round-trip pairs.
 316: 
 317:     Uses FIFO matching: each SELL is paired with the earliest available BUY
 318:     shares for that ticker.
 319: 
 320:     Args:
 321:         transactions: List of transaction dicts from portfolio state.
 322: 
 323:     Returns:
 324:         list of round-trip dicts, each with:
 325:             - ticker: str
 326:             - buy_ts: str (ISO-8601)
 327:             - sell_ts: str (ISO-8601)
 328:             - buy_price_sek: float (per share)
 329:             - sell_price_sek: float (per share)
 330:             - shares: float
 331:             - pnl_pct: float
 332:             - pnl_sek: float
 333:             - hold_hours: float
 334:             - fee_sek: float (total fees for this round trip)
 335:     """
 336: 
 337:     # Group BUYs by ticker — maintain FIFO order
 338:     buy_queues = defaultdict(list)
 339:     for tx in transactions:
 340:         if tx.get("action") == "BUY":
 341:             ticker = tx.get("ticker", "")
 342:             shares = tx.get("shares", 0)
 343:             total_sek = tx.get("total_sek", 0)
 344:             fee = tx.get("fee_sek", 0) or 0
 345:             if shares > 0:
 346:                 price_per_share = total_sek / shares
 347:                 buy_queues[ticker].append({
 348:                     "ts": tx.get("timestamp", ""),
 349:                     "remaining_shares": shares,
 350:                     "original_shares": shares,
 351:                     "price_per_share": price_per_share,
 352:                     "fee_sek": fee,
 353:                 })
 354: 
 355:     round_trips = []
 356: 
 357:     for tx in transactions:
 358:         if tx.get("action") != "SELL":
 359:             continue
 360:         ticker = tx.get("ticker", "")
 361:         sell_shares = tx.get("shares", 0)
 362:         sell_total = tx.get("total_sek", 0)
 363:         sell_fee = tx.get("fee_sek", 0) or 0
 364:         sell_ts = tx.get("timestamp", "")
 365: 
 366:         if sell_shares <= 0 or ticker not in buy_queues:
 367:             continue
 368: 
 369:         sell_price_per_share = sell_total / sell_shares if sell_shares > 0 else 0
 370:         shares_to_match = sell_shares
 371: 
 372:         while shares_to_match > 0 and buy_queues[ticker]:
 373:             buy = buy_queues[ticker][0]
 374:             matched = min(shares_to_match, buy["remaining_shares"])
 375: 
 376:             # Compute hold time
 377:             hold_hours = 0
 378:             try:
 379:                 buy_dt = datetime.datetime.fromisoformat(buy["ts"])
 380:                 sell_dt = datetime.datetime.fromisoformat(sell_ts)
 381:                 if buy_dt.tzinfo is None:
 382:                     buy_dt = buy_dt.replace(tzinfo=datetime.UTC)
 383:                 if sell_dt.tzinfo is None:
 384:                     sell_dt = sell_dt.replace(tzinfo=datetime.UTC)
 385:                 hold_hours = (sell_dt - buy_dt).total_seconds() / 3600
 386:             except (ValueError, TypeError):
 387:                 pass
 388: 
 389:             buy_price = buy["price_per_share"]
 390:             pnl_pct = ((sell_price_per_share - buy_price) / buy_price * 100) if buy_price > 0 else 0
 391: 
 392:             # Proportional fees — use original buy quantity as denominator
 393:             # to avoid inflating fee allocation on subsequent partial matches
 394:             # (BUG-37 regression covered by test_equity_curve_fifo.TestBug37FeeDoubleCount).
 395:             #
 396:             # P0-6 (2026-05-02): pnl_sek is now NET of buy+sell fees so
 397:             # downstream metrics (profit_factor, total_pnl_sek, Calmar) report
 398:             # realised SEK. `pnl_pct` remains the gross price-% (price-move
 399:             # only) — used for streaks and expectancy. `fee_sek` field still
 400:             # reports total fees so consumers can reconstruct gross if needed.
 401:             # See tests/test_equity_curve_fifo.TestPnlSekNetOfFees and
 402:             # tests/test_portfolio_metrics.TestProfitFactorNetOfFees.
 403:             buy_fee_share = (buy["fee_sek"] * matched / buy["original_shares"]) if buy["original_shares"] > 0 else 0
 404:             sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0
 405:             pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share
 406: 
 407:             round_trips.append({
 408:                 "ticker": ticker,
 409:                 "buy_ts": buy["ts"],
 410:                 "sell_ts": sell_ts,
 411:                 "buy_price_sek": round(buy_price, 4),
 412:                 "sell_price_sek": round(sell_price_per_share, 4),
 413:                 "shares": round(matched, 8),
 414:                 "pnl_pct": round(pnl_pct, 4),
 415:                 "pnl_sek": round(pnl_sek, 2),
 416:                 "hold_hours": round(hold_hours, 2),
 417:                 "fee_sek": round(buy_fee_share + sell_fee_share, 2),
 418:             })
 419: 
 420:             buy["remaining_shares"] -= matched
 421:             shares_to_match -= matched
 422: 
 423:             if buy["remaining_shares"] <= 1e-10:
 424:                 buy_queues[ticker].pop(0)
 425: 
 426:     return round_trips
 427: 
 428: 
 429: def compute_trade_metrics(transactions, initial_value=INITIAL_VALUE):
 430:     """Compute per-trade performance metrics from transaction history.
 431: 
 432:     Args:
 433:         transactions: List of transaction dicts from portfolio state.
 434:         initial_value: Starting portfolio value in SEK.
 435: 
 436:     Returns:
 437:         dict with:
 438:             - profit_factor: gross_profit / gross_loss (None if no losses)
 439:             - avg_hold_hours: average hold time per round trip
 440:             - trade_frequency_per_week: trades per 7 calendar days
 441:             - win_loss_ratio: avg_win_pct / avg_loss_pct (None if no losses)
 442:             - max_consecutive_wins: longest win streak
 443:             - max_consecutive_losses: longest loss streak
 444:             - expectancy_pct: (win_rate * avg_win) - (loss_rate * avg_loss)
 445:             - calmar_ratio: annualized_return / max_drawdown (None if insufficient data)
 446:             - round_trips: int (number of paired trades)
 447:             - total_pnl_sek: float (sum of round-trip P&L)
 448:     """
 449:     trips = _pair_round_trips(transactions)
 450: 
 451:     result = {
 452:         "profit_factor": None,
 453:         "avg_hold_hours": 0,
 454:         "trade_frequency_per_week": 0,
 455:         "win_loss_ratio": None,
 456:         "max_consecutive_wins": 0,
 457:         "max_consecutive_losses": 0,
 458:         "expectancy_pct": 0,
 459:         "calmar_ratio": None,
 460:         "round_trips": len(trips),
 461:         "total_pnl_sek": 0,
 462:     }
 463: 
 464:     if not trips:
 465:         return result
 466: 
 467:     # Gross profit/loss
 468:     gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)
 469:     gross_loss = abs(sum(t["pnl_sek"] for t in trips if t["pnl_sek"] < 0))
 470: 
 471:     if gross_loss > 0:
 472:         result["profit_factor"] = round(gross_profit / gross_loss, 4)
 473: 
 474:     # Average hold time
 475:     hold_hours = [t["hold_hours"] for t in trips if t["hold_hours"] > 0]
 476:     if hold_hours:
 477:         result["avg_hold_hours"] = round(sum(hold_hours) / len(hold_hours), 2)
 478: 
 479:     # Trade frequency
 480:     try:
 481:         timestamps = []
 482:         for t in trips:
 483:             ts_str = t.get("sell_ts") or t.get("buy_ts")
 484:             if ts_str:
 485:                 timestamps.append(datetime.datetime.fromisoformat(ts_str))
 486:         if len(timestamps) >= 2:
 487:             span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
 488:             if span_days > 0:
 489:                 result["trade_frequency_per_week"] = round(len(trips) / span_days * 7, 2)
 490:     except (ValueError, TypeError):
 491:         pass
 492: 
 493:     # Win/loss stats
 494:     wins = [t for t in trips if t["pnl_pct"] > 0]
 495:     losses = [t for t in trips if t["pnl_pct"] <= 0]
 496:     win_count = len(wins)
 497:     loss_count = len(losses)
 498:     total_count = len(trips)
 499: 
 500:     avg_win_pct = sum(t["pnl_pct"] for t in wins) / win_count if wins else 0
 501:     avg_loss_pct = abs(sum(t["pnl_pct"] for t in losses) / loss_count) if losses else 0
 502: 
 503:     if avg_loss_pct > 0:
 504:         result["win_loss_ratio"] = round(avg_win_pct / avg_loss_pct, 4)
 505: 
 506:     # Streaks
 507:     max_wins = 0
 508:     max_losses = 0
 509:     current_wins = 0
 510:     current_losses = 0
 511:     for t in trips:
 512:         if t["pnl_pct"] > 0:
 513:             current_wins += 1
 514:             current_losses = 0
 515:             max_wins = max(max_wins, current_wins)
 516:         else:
 517:             current_losses += 1
 518:             current_wins = 0
 519:             max_losses = max(max_losses, current_losses)
 520: 
 521:     result["max_consecutive_wins"] = max_wins
 522:     result["max_consecutive_losses"] = max_losses
 523: 
 524:     # Expectancy
 525:     win_rate = win_count / total_count if total_count > 0 else 0
 526:     loss_rate = loss_count / total_count if total_count > 0 else 0
 527:     result["expectancy_pct"] = round(
 528:         (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct), 4
 529:     )
 530: 
 531:     # Total P&L
 532:     result["total_pnl_sek"] = round(sum(t["pnl_sek"] for t in trips), 2)
 533: 
 534:     # Calmar ratio (annualized return / max drawdown)
 535:     # Compute a mini equity curve from round-trip PnLs
 536:     if initial_value > 0 and len(trips) >= 2:
 537:         equity = [initial_value]
 538:         for t in trips:
 539:             equity.append(equity[-1] + t["pnl_sek"])
 540: 
 541:         peak = equity[0]
 542:         max_dd = 0
 543:         for val in equity:
 544:             if val > peak:
 545:                 peak = val
 546:             dd = (peak - val) / peak if peak > 0 else 0
 547:             if dd > max_dd:
 548:                 max_dd = dd
 549: 
 550:         total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0
 551:         try:
 552:             first_ts = datetime.datetime.fromisoformat(trips[0]["buy_ts"])
 553:             last_ts = datetime.datetime.fromisoformat(trips[-1]["sell_ts"])
 554:             days = (last_ts - first_ts).total_seconds() / 86400
 555:             if days >= 1 and max_dd > 0:
 556:                 years = days / 365.25
 557:                 annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
 558:                 result["calmar_ratio"] = round(annualized / max_dd, 4)
 559:         except (ValueError, TypeError):
 560:             pass
 561: 
 562:     return result
 563: 
 564: 
 565: if __name__ == "__main__":
 566:     curve = load_equity_curve()
 567:     if not curve:
 568:         print("No equity curve data found in", DEFAULT_HISTORY_PATH)
 569:         print("Run risk_management.log_portfolio_value() to start logging.")
 570:     else:
 571:         print(f"Loaded {len(curve)} data points")
 572:         results = compare_strategies(curve)
 573: 
 574:         for strategy in ("patient", "bold"):
 575:             m = results[strategy]
 576:             print(f"\n{'='*50}")
 577:             print(f"  {strategy.upper()} STRATEGY METRICS")
 578:             print(f"{'='*50}")
 579:             print(f"  Total return:     {m['total_return_pct']:+.2f}%")
 580:             print(f"  Max drawdown:     {m['max_drawdown_pct']:.2f}%")
 581:             if m["sharpe_ratio"] is not None:
 582:                 print(f"  Sharpe ratio:     {m['sharpe_ratio']:.2f}")
 583:             if m["sortino_ratio"] is not None:
 584:                 print(f"  Sortino ratio:    {m['sortino_ratio']:.2f}")
 585:             print(f"  Win rate:         {m['win_rate']:.1f}%")
 586:             print(f"  Best day:         {m['best_day_pct']:+.2f}%")
 587:             print(f"  Worst day:        {m['worst_day_pct']:+.2f}%")
 588:             print(f"  Days in drawdown: {m['days_in_drawdown']}")
 589:             print(f"  Volatility (ann): {m['volatility_annual_pct']:.2f}%")
 590:             if m["date_range"]:
 591:                 print(f"  Date range:       {m['date_range'][0][:10]} to {m['date_range'][1][:10]}")
 592: 
 593:         c = results["comparison"]
 594:         print(f"\n{'='*50}")
 595:         print("  COMPARISON")
 596:         print(f"{'='*50}")
 597:         print(f"  Return leader:    {c['leader']} (by {abs(c['return_diff_pct']):.2f}%)")
 598:         print(f"  Lower drawdown:   {c['lower_drawdown']} (by {abs(c['drawdown_diff_pct']):.2f}%)")
 599:         if "sharpe_leader" in c:
 600:             print(f"  Better Sharpe:    {c['sharpe_leader']} (by {abs(c['sharpe_diff']):.2f})")

### FILE kelly_sizing.py
   1: """Position sizing using Kelly criterion.
   2: 
   3: Kelly fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
   4: 
   5: Uses signal accuracy as win probability and historical trade outcomes for avg_win/avg_loss.
   6: Returns recommended position size as fraction of portfolio.
   7: """
   8: 
   9: from pathlib import Path
  10: 
  11: from portfolio.file_utils import load_json
  12: 
  13: BASE_DIR = Path(__file__).resolve().parent.parent
  14: DATA_DIR = BASE_DIR / "data"
  15: PATIENT_FILE = DATA_DIR / "portfolio_state.json"
  16: BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
  17: AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
  18: 
  19: 
  20: 
  21: def kelly_fraction(win_prob, avg_win_pct, avg_loss_pct):
  22:     """Compute Kelly criterion fraction.
  23: 
  24:     The Kelly fraction gives the optimal bet size as a fraction of bankroll
  25:     that maximizes long-run geometric growth rate.
  26: 
  27:     Formula: f* = (p * b - q) / b
  28:     where p = win probability, q = 1 - p, b = avg_win / avg_loss
  29: 
  30:     Args:
  31:         win_prob: Probability of winning (0.0 to 1.0).
  32:         avg_win_pct: Average win as a positive percentage (e.g. 2.5 for +2.5%).
  33:         avg_loss_pct: Average loss as a positive percentage (e.g. 1.8 for -1.8%).
  34: 
  35:     Returns:
  36:         float: Kelly fraction (0.0 to 1.0). Clamped to [0, 1].
  37:             Returns 0.0 if inputs are invalid or edge is negative.
  38:     """
  39:     if win_prob <= 0 or win_prob >= 1:
  40:         return 0.0
  41:     if avg_win_pct <= 0 or avg_loss_pct <= 0:
  42:         return 0.0
  43: 
  44:     # b = ratio of avg win to avg loss
  45:     b = avg_win_pct / avg_loss_pct
  46:     q = 1.0 - win_prob
  47: 
  48:     # Kelly formula: f* = (p * b - q) / b
  49:     kelly = (win_prob * b - q) / b
  50: 
  51:     # Clamp to [0, 1] — negative Kelly means negative edge, don't bet
  52:     return max(0.0, min(1.0, kelly))
  53: 
  54: 
  55: def _compute_trade_stats(transactions, ticker=None):
  56:     """Compute win rate and average win/loss from historical transactions.
  57: 
  58:     Pairs BUY and SELL transactions to compute realized P&L per round-trip.
  59: 
  60:     Args:
  61:         transactions: List of transaction dicts from portfolio state.
  62:         ticker: If specified, filter to this ticker only. None = all tickers.
  63: 
  64:     Returns:
  65:         dict: {win_rate, avg_win_pct, avg_loss_pct, total_trades, wins, losses}
  66:               Returns None if insufficient data (fewer than 2 round-trips).
  67:     """
  68:     # Group transactions by ticker
  69:     from collections import defaultdict
  70:     buys_by_ticker = defaultdict(list)
  71:     sells_by_ticker = defaultdict(list)
  72: 
  73:     for t in transactions:
  74:         t_ticker = t.get("ticker", "")
  75:         if ticker and t_ticker != ticker:
  76:             continue
  77:         action = t.get("action", "")
  78:         if action == "BUY":
  79:             buys_by_ticker[t_ticker].append(t)
  80:         elif action == "SELL":
  81:             sells_by_ticker[t_ticker].append(t)
  82: 
  83:     # Compute P&L for each sell vs weighted average buy price
  84:     pnl_list = []
  85:     for t_ticker, sells in sells_by_ticker.items():
  86:         buys = buys_by_ticker.get(t_ticker, [])
  87:         if not buys:
  88:             continue
  89: 
  90:         # Compute weighted average buy price (in SEK per share)
  91:         total_shares_bought = sum(b.get("shares", 0) for b in buys)
  92:         total_cost = sum(b.get("total_sek", 0) for b in buys)
  93:         if total_shares_bought <= 0:
  94:             continue
  95:         avg_buy_price = total_cost / total_shares_bought
  96: 
  97:         for sell in sells:
  98:             sell_shares = sell.get("shares", 0)
  99:             sell_total = sell.get("total_sek", 0)
 100:             if sell_shares <= 0:
 101:                 continue
 102:             sell_price_per_share = sell_total / sell_shares
 103:             pnl_pct = (sell_price_per_share - avg_buy_price) / avg_buy_price * 100
 104:             pnl_list.append(pnl_pct)
 105: 
 106:     if len(pnl_list) < 2:
 107:         return None
 108: 
 109:     wins = [p for p in pnl_list if p > 0]
 110:     losses = [abs(p) for p in pnl_list if p <= 0]
 111: 
 112:     win_rate = len(wins) / len(pnl_list) if pnl_list else 0.0
 113:     avg_win = sum(wins) / len(wins) if wins else 0.0
 114:     avg_loss = sum(losses) / len(losses) if losses else 0.0
 115: 
 116:     return {
 117:         "win_rate": win_rate,
 118:         "avg_win_pct": avg_win,
 119:         "avg_loss_pct": avg_loss,
 120:         "total_trades": len(pnl_list),
 121:         "wins": len(wins),
 122:         "losses": len(losses),
 123:     }
 124: 
 125: 
 126: def _get_signal_accuracy(agent_summary, ticker=None):
 127:     """Extract consensus accuracy from agent_summary.
 128: 
 129:     Uses signal_accuracy_1d.consensus.accuracy as the win probability proxy.
 130:     Falls back to weighted_confidence from the ticker's signal data.
 131: 
 132:     Args:
 133:         agent_summary: Parsed agent_summary.json dict.
 134:         ticker: Ticker to get accuracy for (used for weighted_confidence fallback).
 135: 
 136:     Returns:
 137:         float: Estimated win probability (0.0 to 1.0).
 138:     """
 139:     # Primary: use overall consensus accuracy
 140:     acc_data = agent_summary.get("signal_accuracy_1d", {})
 141:     consensus_acc = acc_data.get("consensus", {}).get("accuracy")
 142:     if consensus_acc is not None and consensus_acc > 0:
 143:         return consensus_acc
 144: 
 145:     # Fallback: use weighted_confidence from the ticker's signals
 146:     if ticker:
 147:         signals = agent_summary.get("signals", {})
 148:         ticker_data = signals.get(ticker, {})
 149:         weighted_conf = ticker_data.get("weighted_confidence")
 150:         if weighted_conf is not None:
 151:             return weighted_conf
 152: 
 153:     # Last resort: 50/50
 154:     return 0.5
 155: 
 156: 
 157: def _get_ticker_signal_accuracy(agent_summary, ticker):
 158:     """Get the accuracy of the most relevant signals for a specific ticker.
 159: 
 160:     Computes a weighted average of signal accuracies, weighted by each signal's
 161:     normalized weight. Only considers signals that are actively voting (non-HOLD).
 162: 
 163:     P1-11 (2026-05-02 adversarial follow-up): per-signal accuracy is now
 164:     looked up per-ticker first via `agent_summary["per_ticker_signal_accuracy"]`
 165:     (produced by `accuracy_stats.accuracy_by_ticker_signal_cached()` upstream).
 166:     If a ticker/signal isn't present per-ticker, OR has fewer than 5 samples
 167:     on this ticker, it falls back to the system-wide
 168:     `signal_accuracy_1d.signals` block. Without this, a signal that's 70%
 169:     accurate on XAG-USD but 30% on BTC-USD shows up as ~50% in the
 170:     system-wide aggregate, distorting Kelly sizing on both. The fallback
 171:     keeps backwards compatibility with older agent_summary writers that
 172:     don't yet emit the per-ticker block.
 173: 
 174:     Args:
 175:         agent_summary: Parsed agent_summary.json dict.
 176:         ticker: Ticker symbol.
 177: 
 178:     Returns:
 179:         float: Weighted signal accuracy (0.0 to 1.0), or None if insufficient data.
 180:     """
 181:     acc_data = agent_summary.get("signal_accuracy_1d", {})
 182:     sig_accuracies = acc_data.get("signals", {})
 183:     sig_weights = agent_summary.get("signal_weights", {})
 184: 
 185:     # P1-11: per-ticker per-signal accuracy block (preferred when present).
 186:     per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
 187:     per_ticker_for_ticker = per_ticker_block.get(ticker, {}) or {}
 188: 
 189:     signals = agent_summary.get("signals", {})
 190:     ticker_data = signals.get(ticker, {})
 191:     extra = ticker_data.get("extra", {}) if isinstance(ticker_data, dict) else {}
 192:     votes = extra.get("_votes", {})
 193: 
 194:     if not votes:
 195:         return None
 196:     # Either source can satisfy us — only bail if BOTH are empty.
 197:     if not sig_accuracies and not per_ticker_for_ticker:
 198:         return None
 199: 
 200:     weighted_sum = 0.0
 201:     weight_total = 0.0
 202: 
 203:     for sig_name, vote in votes.items():
 204:         if vote == "HOLD":
 205:             continue
 206: 
 207:         # Prefer per-ticker accuracy when it has enough samples for THIS
 208:         # ticker; otherwise fall back to the system-wide row.
 209:         per_ticker_sig = per_ticker_for_ticker.get(sig_name) or {}
 210:         per_ticker_samples = per_ticker_sig.get("samples", 0)
 211:         if per_ticker_samples >= 5:
 212:             accuracy = per_ticker_sig.get("accuracy", 0.5)
 213:             samples = per_ticker_samples
 214:         else:
 215:             sig_acc = sig_accuracies.get(sig_name, {})
 216:             accuracy = sig_acc.get("accuracy", 0.5)
 217:             samples = sig_acc.get("samples", 0)
 218:             if samples < 5:
 219:                 continue  # unreliable on both axes
 220: 
 221:         weight = sig_weights.get(sig_name, {}).get("normalized_weight", 1.0)
 222:         weighted_sum += accuracy * weight
 223:         weight_total += weight
 224: 
 225:     if weight_total <= 0:
 226:         return None
 227: 
 228:     return weighted_sum / weight_total
 229: 
 230: 
 231: def recommended_size(ticker, portfolio_path=None, agent_summary=None, strategy="patient"):
 232:     """Compute recommended position size using Kelly criterion.
 233: 
 234:     Combines signal accuracy (as win probability) with historical trade
 235:     performance (avg win/loss) to compute optimal position sizing.
 236: 
 237:     Args:
 238:         ticker: Ticker symbol to compute sizing for.
 239:         portfolio_path: Path to portfolio state JSON. If None, uses default
 240:             based on strategy.
 241:         agent_summary: Parsed agent_summary dict. If None, loads from file.
 242:         strategy: "patient" or "bold" - determines default portfolio and max alloc.
 243: 
 244:     Returns:
 245:         dict: {
 246:             kelly_pct: Full Kelly fraction (0-1),
 247:             half_kelly_pct: Half Kelly (more conservative),
 248:             quarter_kelly_pct: Quarter Kelly (most conservative),
 249:             recommended_sek: Recommended trade size in SEK (using half Kelly),
 250:             max_alloc_sek: Maximum allocation per strategy rules,
 251:             win_prob: Estimated win probability used,
 252:             avg_win_pct: Average win percentage used,
 253:             avg_loss_pct: Average loss percentage used,
 254:             source: Description of data source used for estimates,
 255:         }
 256:     """
 257:     # Load portfolio
 258:     if portfolio_path is None:
 259:         portfolio_path = BOLD_FILE if strategy == "bold" else PATIENT_FILE
 260:     portfolio = load_json(portfolio_path, default={})
 261:     cash_sek = portfolio.get("cash_sek", 0)
 262:     transactions = portfolio.get("transactions", [])
 263: 
 264:     # Load agent summary
 265:     if agent_summary is None:
 266:         agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
 267: 
 268:     # Max allocation per strategy rules
 269:     alloc_frac = 0.30 if strategy == "bold" else 0.15
 270:     max_alloc = cash_sek * alloc_frac
 271: 
 272:     # Estimate win probability
 273:     # Priority: ticker-specific weighted signal accuracy > consensus accuracy > 50%
 274:     # P1-11 (2026-05-02): _get_ticker_signal_accuracy now prefers
 275:     # per_ticker_signal_accuracy over the system-wide aggregate when a
 276:     # per-ticker block is supplied in agent_summary. Source string
 277:     # advertises which path was used so operators can audit the decision.
 278:     per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
 279:     has_per_ticker = bool(per_ticker_block.get(ticker))
 280:     win_prob = _get_ticker_signal_accuracy(agent_summary, ticker)
 281:     if has_per_ticker:
 282:         source = f"per-ticker weighted signal accuracy for {ticker}"
 283:     else:
 284:         source = f"weighted signal accuracy for {ticker} (system-wide fallback)"
 285: 
 286:     if win_prob is None:
 287:         win_prob = _get_signal_accuracy(agent_summary, ticker)
 288:         source = "consensus accuracy"
 289: 
 290:     # Estimate avg win/loss from historical trades
 291:     trade_stats = _compute_trade_stats(transactions, ticker=ticker)
 292:     if trade_stats is None:
 293:         # Try all tickers if not enough ticker-specific data
 294:         trade_stats = _compute_trade_stats(transactions, ticker=None)
 295: 
 296:     if trade_stats and trade_stats["avg_win_pct"] > 0 and trade_stats["avg_loss_pct"] > 0:
 297:         avg_win = trade_stats["avg_win_pct"]
 298:         avg_loss = trade_stats["avg_loss_pct"]
 299:         source += f" + trade history ({trade_stats['total_trades']} trades)"
 300:     else:
 301:         # Default estimates based on typical crypto/stock moves
 302:         # Use ATR from agent summary if available
 303:         signals = agent_summary.get("signals", {})
 304:         ticker_data = signals.get(ticker, {})
 305:         atr_pct = ticker_data.get("atr_pct", 1.5) if isinstance(ticker_data, dict) else 1.5
 306: 
 307:         # Assume avg win = 1.5x ATR, avg loss = 1x ATR (realistic risk/reward)
 308:         avg_win = atr_pct * 1.5
 309:         avg_loss = atr_pct * 1.0
 310:         source += f" + ATR-based estimates (win={avg_win:.1f}%, loss={avg_loss:.1f}%)"
 311: 
 312:     # Compute Kelly
 313:     full_kelly = kelly_fraction(win_prob, avg_win, avg_loss)
 314:     half_kelly = full_kelly / 2.0
 315:     quarter_kelly = full_kelly / 4.0
 316: 
 317:     # Apply exposure ceiling from market health / exposure coach.
 318:     # Scales position sizes down when the broad market is unhealthy.
 319:     exposure_rec = agent_summary.get("exposure_recommendation", {})
 320:     exposure_ceiling = exposure_rec.get("exposure_ceiling", 1.0)
 321: 
 322:     # Recommended size = half Kelly * cash * exposure_ceiling, capped at max allocation
 323:     rec_sek = min(half_kelly * cash_sek * exposure_ceiling, max_alloc)
 324: 
 325:     # Minimum trade size check
 326:     if rec_sek < 500:
 327:         rec_sek = 0  # Below minimum trade size
 328: 
 329:     return {
 330:         "kelly_pct": round(full_kelly, 4),
 331:         "half_kelly_pct": round(half_kelly, 4),
 332:         "quarter_kelly_pct": round(quarter_kelly, 4),
 333:         "recommended_sek": round(rec_sek, 0),
 334:         "max_alloc_sek": round(max_alloc, 0),
 335:         "exposure_ceiling": round(exposure_ceiling, 2),
 336:         "win_prob": round(win_prob, 4),
 337:         "avg_win_pct": round(avg_win, 2),
 338:         "avg_loss_pct": round(avg_loss, 2),
 339:         "source": source,
 340:     }
 341: 
 342: 
 343: def print_sizing_report(tickers=None, strategy="patient"):
 344:     """Print Kelly sizing recommendations for given tickers.
 345: 
 346:     Args:
 347:         tickers: List of ticker symbols. If None, uses all from agent_summary.
 348:         strategy: "patient" or "bold".
 349:     """
 350:     agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
 351:     if tickers is None:
 352:         tickers = list(agent_summary.get("signals", {}).keys())
 353: 
 354:     if not tickers:
 355:         print("No tickers found in agent_summary.json")
 356:         return
 357: 
 358:     print(f"=== Kelly Sizing Report ({strategy.title()}) ===")
 359:     print()
 360:     print(
 361:         f"{'Ticker':<10} {'Kelly%':>7} {'Half-K%':>8} {'Rec SEK':>10} "
 362:         f"{'Max SEK':>10} {'Win Prob':>9}"
 363:     )
 364:     print(
 365:         f"{'------':<10} {'------':>7} {'-------':>8} {'-------':>10} "
 366:         f"{'-------':>10} {'--------':>9}"
 367:     )
 368: 
 369:     for ticker in sorted(tickers):
 370:         rec = recommended_size(ticker, agent_summary=agent_summary, strategy=strategy)
 371:         print(
 372:             f"{ticker:<10} {rec['kelly_pct']*100:>6.1f}% {rec['half_kelly_pct']*100:>7.1f}% "
 373:             f"{rec['recommended_sek']:>10,.0f} {rec['max_alloc_sek']:>10,.0f} "
 374:             f"{rec['win_prob']*100:>8.1f}%"
 375:         )
 376: 
 377:     print()
 378:     print("Note: Recommended size uses Half Kelly (more conservative).")
 379:     print("Full Kelly is optimal but volatile. Quarter Kelly is safest.")
 380: 
 381: 
 382: if __name__ == "__main__":
 383:     import sys
 384: 
 385:     strategy = "bold" if "--bold" in sys.argv else "patient"
 386:     tickers = [a for a in sys.argv[1:] if not a.startswith("--")]
 387:     if not tickers:
 388:         tickers = None
 389:     print_sizing_report(tickers=tickers, strategy=strategy)

### FILE monte_carlo.py
   1: """Monte Carlo price simulation engine.
   2: 
   3: Generates stochastic price paths using Geometric Brownian Motion (GBM)
   4: with antithetic variates for variance reduction. Converts directional
   5: signal probabilities into price quantile bands, stop-loss probabilities,
   6: and expected return distributions.
   7: 
   8: Short-term focus: 3h, 1d, 3d horizons for active trading decisions.
   9: 
  10: Usage:
  11:     from portfolio.monte_carlo import simulate_ticker
  12:     result = simulate_ticker("BTC-USD", agent_summary)
  13:     # result = {
  14:     #   "price_bands_1d": {"p5": 82100, "p25": 84300, "p50": 86200, ...},
  15:     #   "p_stop_hit_1d": 0.12,
  16:     #   "expected_return_1d": {"mean_pct": 0.3, "std_pct": 2.1, "skew": 0.1},
  17:     #   ...
  18:     # }
  19: """
  20: 
  21: import logging
  22: import math
  23: 
  24: import numpy as np
  25: from scipy.stats import norm
  26: 
  27: logger = logging.getLogger("portfolio.monte_carlo")
  28: 
  29: # Default parameters
  30: DEFAULT_N_PATHS = 10_000   # 5K pairs with antithetic variates
  31: DEFAULT_HORIZONS = [1, 3]  # days
  32: MIN_VOLATILITY = 0.05      # 5% annualized floor (prevents degenerate sims)
  33: 
  34: # Per-asset-class ATR fallbacks when actual ATR is missing from signals.
  35: # The generic 2.0% underestimates tail risk for crypto/metals.
  36: _ATR_DEFAULT_BY_CLASS = {
  37:     "crypto": 3.5,  # BTC 3-4% typical daily ATR
  38:     "metals": 4.0,  # XAG 4-6%, XAU 2-3% — biased toward silver
  39:     "stocks": 2.0,  # MSTR ~3% but broader equities lower
  40: }
  41: 
  42: 
  43: # ---------------------------------------------------------------------------
  44: # Volatility & drift estimation from existing system data
  45: # ---------------------------------------------------------------------------
  46: 
  47: def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
  48:     """Convert ATR% (14-period) to annualized volatility.
  49: 
  50:     ATR measures average true range over `period` candles. For hourly candles
  51:     (our primary timeframe), we annualize: vol = atr_frac * sqrt(trading_periods/period).
  52: 
  53:     Uses 252 trading days (standard for stocks/crypto).
  54: 
  55:     Args:
  56:         atr_pct: ATR as percentage of price (e.g., 3.5 means 3.5%).
  57:         period: ATR lookback period (default 14).
  58: 
  59:     Returns:
  60:         Annualized volatility as a decimal (e.g., 0.20 = 20%).
  61:     """
  62:     atr_frac = atr_pct / 100.0
  63:     annual_factor = math.sqrt(252.0 / period)
  64:     vol = atr_frac * annual_factor
  65:     return max(vol, MIN_VOLATILITY)
  66: 
  67: 
  68: def drift_from_probability(p_up: float, volatility: float) -> float:
  69:     """Convert directional probability P(up) into annualized drift.
  70: 
  71:     Uses the inverse of the GBM CDF relationship:
  72:         P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / (sigma*sqrt(T)))
  73: 
  74:     For 1-day horizon (T = 1/252):
  75:         mu = sigma * N_inv(p_up) * sqrt(252) + 0.5 * sigma^2
  76: 
  77:     This ensures the GBM simulation produces paths where the fraction
  78:     ending above spot matches the input probability.
  79: 
  80:     Args:
  81:         p_up: Probability of price being higher at horizon (0.0-1.0).
  82:         volatility: Annualized volatility (decimal).
  83: 
  84:     Returns:
  85:         Annualized drift (decimal). Positive = upward bias.
  86:     """
  87:     # Clamp p_up to avoid infinite drift at extremes
  88:     p_up = max(0.01, min(0.99, p_up))
  89: 
  90:     # N_inv(p_up) gives the z-score for the desired probability
  91:     z = norm.ppf(p_up)
  92: 
  93:     # mu = sigma * z * sqrt(252) + 0.5 * sigma^2
  94:     # This is derived from P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / sigma*sqrt(T))
  95:     mu = volatility * z * math.sqrt(252.0) + 0.5 * volatility**2
  96: 
  97:     return mu
  98: 
  99: 
 100: def _atr_default_for_ticker(ticker: str) -> float:
 101:     """Return the per-asset-class ATR fallback for a ticker."""
 102:     from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS
 103:     if ticker in CRYPTO_SYMBOLS:
 104:         return _ATR_DEFAULT_BY_CLASS["crypto"]
 105:     if ticker in METALS_SYMBOLS:
 106:         return _ATR_DEFAULT_BY_CLASS["metals"]
 107:     return _ATR_DEFAULT_BY_CLASS["stocks"]
 108: 
 109: 
 110: # ---------------------------------------------------------------------------
 111: # Core simulation engine
 112: # ---------------------------------------------------------------------------
 113: 
 114: class MonteCarloEngine:
 115:     """Geometric Brownian Motion price path simulator with antithetic variates.
 116: 
 117:     Generates terminal price distributions for short-term horizons (hours to days).
 118:     Uses antithetic variates for 50-75% variance reduction at zero extra cost.
 119: 
 120:     Args:
 121:         price: Current asset price (USD).
 122:         volatility: Annualized volatility (decimal, e.g., 0.20 = 20%).
 123:         drift: Annualized drift (decimal). Use drift_from_probability() to derive.
 124:         horizon_days: Simulation horizon in days (supports fractional, e.g., 0.125 = 3h).
 125:         n_paths: Number of simulated terminal prices to generate.
 126:         seed: Random seed for reproducibility.
 127:     """
 128: 
 129:     def __init__(self, price: float, volatility: float, drift: float = 0.0,
 130:                  horizon_days: float = 1.0, n_paths: int = DEFAULT_N_PATHS,
 131:                  seed: int | None = None):
 132:         self.price = price
 133:         self.volatility = max(volatility, MIN_VOLATILITY)
 134:         self.drift = drift
 135:         self.horizon_days = horizon_days
 136:         self.n_paths = n_paths
 137:         self.seed = seed
 138:         self._terminal_prices = None
 139: 
 140:     def simulate_paths(self) -> np.ndarray:
 141:         """Generate terminal prices via GBM with antithetic variates.
 142: 
 143:         Formula: S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
 144: 
 145:         Antithetic variates: for each random draw Z, we also compute the
 146:         path with -Z. This creates negative correlation between paired
 147:         estimates, reducing variance of the mean estimator.
 148: 
 149:         Returns:
 150:             1D array of terminal prices, shape (n_paths,).
 151:         """
 152:         rng = np.random.default_rng(self.seed)
 153: 
 154:         T = self.horizon_days / 252.0  # Convert to years
 155:         sigma = self.volatility
 156:         mu = self.drift
 157: 
 158:         # Draw half the paths, use antithetic for the other half
 159:         n_half = self.n_paths // 2
 160:         Z = rng.standard_normal(n_half)
 161: 
 162:         # Deterministic component
 163:         drift_term = (mu - 0.5 * sigma**2) * T
 164:         vol_term = sigma * math.sqrt(T)
 165: 
 166:         # Original paths
 167:         log_returns_pos = drift_term + vol_term * Z
 168:         # Antithetic paths (negate Z)
 169:         log_returns_neg = drift_term + vol_term * (-Z)
 170: 
 171:         # Combine
 172:         log_returns = np.concatenate([log_returns_pos, log_returns_neg])
 173:         terminal_prices = self.price * np.exp(log_returns)
 174: 
 175:         # If n_paths is odd, add one more path
 176:         if self.n_paths % 2 == 1:
 177:             extra_Z = rng.standard_normal(1)
 178:             extra_price = self.price * np.exp(drift_term + vol_term * extra_Z)
 179:             terminal_prices = np.concatenate([terminal_prices, extra_price])
 180: 
 181:         self._terminal_prices = terminal_prices
 182:         return terminal_prices
 183: 
 184:     def _ensure_simulated(self):
 185:         """Run simulation if not already done."""
 186:         if self._terminal_prices is None:
 187:             self.simulate_paths()
 188: 
 189:     def price_quantiles(self, percentiles: list[int] | None = None) -> dict:
 190:         """Extract price quantile bands from simulated distribution.
 191: 
 192:         Args:
 193:             percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95]).
 194: 
 195:         Returns:
 196:             Dict mapping percentile → price (e.g., {5: 87.2, 50: 90.1, 95: 93.5}).
 197:         """
 198:         self._ensure_simulated()
 199:         if percentiles is None:
 200:             percentiles = [5, 25, 50, 75, 95]
 201: 
 202:         values = np.percentile(self._terminal_prices, percentiles)
 203:         return {p: round(float(v), 2) for p, v in zip(percentiles, values)}
 204: 
 205:     def probability_below(self, threshold: float) -> float:
 206:         """Compute P(terminal price < threshold).
 207: 
 208:         Useful for stop-loss probability: "what's the chance price hits my stop?"
 209: 
 210:         Args:
 211:             threshold: Price level to test.
 212: 
 213:         Returns:
 214:             Probability (0.0-1.0).
 215:         """
 216:         self._ensure_simulated()
 217:         if threshold <= 0:
 218:             return 0.0
 219:         return float(np.mean(self._terminal_prices < threshold))
 220: 
 221:     def probability_above(self, threshold: float) -> float:
 222:         """Compute P(terminal price > threshold).
 223: 
 224:         Useful for profit target probability.
 225: 
 226:         Args:
 227:             threshold: Price level to test.
 228: 
 229:         Returns:
 230:             Probability (0.0-1.0).
 231:         """
 232:         self._ensure_simulated()
 233:         if threshold <= 0:
 234:             return 1.0
 235:         return float(np.mean(self._terminal_prices > threshold))
 236: 
 237:     def expected_return(self) -> dict:
 238:         """Compute return distribution statistics.
 239: 
 240:         Returns:
 241:             Dict with mean_pct, std_pct, skew of the return distribution.
 242:         """
 243:         self._ensure_simulated()
 244:         returns = (self._terminal_prices - self.price) / self.price * 100.0
 245: 
 246:         mean_pct = float(np.mean(returns))
 247:         std_pct = float(np.std(returns))
 248: 
 249:         # Skewness
 250:         if std_pct > 0:
 251:             skew = float(np.mean(((returns - mean_pct) / std_pct) ** 3))
 252:         else:
 253:             skew = 0.0
 254: 
 255:         return {
 256:             "mean_pct": round(mean_pct, 2),
 257:             "std_pct": round(std_pct, 2),
 258:             "skew": round(skew, 2),
 259:         }
 260: 
 261: 
 262: # ---------------------------------------------------------------------------
 263: # Convenience: simulate a single ticker from agent_summary data
 264: # ---------------------------------------------------------------------------
 265: 
 266: def simulate_ticker(ticker: str, agent_summary: dict,
 267:                     n_paths: int = DEFAULT_N_PATHS,
 268:                     horizons: list[int] | None = None,
 269:                     seed: int | None = None) -> dict | None:
 270:     """Simulate price distribution for a ticker using agent_summary data.
 271: 
 272:     Extracts price, ATR volatility, and directional probability from the
 273:     signal data, then runs GBM simulation at each horizon.
 274: 
 275:     Args:
 276:         ticker: Instrument ticker (e.g., "BTC-USD").
 277:         agent_summary: Full agent_summary dict (or compact version).
 278:         n_paths: Number of MC paths per horizon.
 279:         horizons: List of horizon days (default: [1, 3]).
 280:         seed: Random seed.
 281: 
 282:     Returns:
 283:         Dict with price bands, stop probability, expected return per horizon.
 284:         None if ticker not found in summary.
 285:     """
 286:     if horizons is None:
 287:         horizons = DEFAULT_HORIZONS
 288: 
 289:     signals = agent_summary.get("signals", {})
 290:     ticker_data = signals.get(ticker)
 291:     if not ticker_data:
 292:         return None
 293: 
 294:     price = ticker_data.get("price_usd", 0)
 295:     if price <= 0:
 296:         return None
 297: 
 298:     extra = ticker_data.get("extra", {})
 299:     atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct") or _atr_default_for_ticker(ticker)
 300: 
 301:     # Get volatility from ATR
 302:     vol = volatility_from_atr(atr_pct)
 303: 
 304:     # Get directional probability for drift
 305:     # Try to use existing probability computation if available
 306:     p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
 307:     drift = drift_from_probability(p_up, vol)
 308: 
 309:     # Compute ATR-based stop level (2x ATR below entry/current)
 310:     stop_price = price * (1 - 2 * atr_pct / 100)
 311: 
 312:     result = {}
 313: 
 314:     for h in horizons:
 315:         h_key = f"{h}d" if h >= 1 else f"{int(h * 24)}h"
 316: 
 317:         mc = MonteCarloEngine(
 318:             price=price,
 319:             volatility=vol,
 320:             drift=drift,
 321:             horizon_days=h,
 322:             n_paths=n_paths,
 323:             seed=seed,
 324:         )
 325:         mc.simulate_paths()
 326: 
 327:         result[f"price_bands_{h_key}"] = mc.price_quantiles()
 328:         result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)
 329:         result[f"expected_return_{h_key}"] = mc.expected_return()
 330: 
 331:     # Add metadata
 332:     result["price_usd"] = price
 333:     result["atr_pct"] = round(atr_pct, 2)
 334:     result["volatility_annual"] = round(vol, 3)
 335:     result["p_up"] = round(p_up, 3)
 336:     result["drift_annual"] = round(drift, 4)
 337:     result["stop_price"] = round(stop_price, 2)
 338: 
 339:     return result
 340: 
 341: 
 342: def _get_directional_probability(ticker: str, ticker_data: dict,
 343:                                   agent_summary: dict) -> float:
 344:     """Extract directional probability from available data.
 345: 
 346:     Priority:
 347:     1. Pre-computed focus_probabilities (from ticker_accuracy.py)
 348:     2. Weighted confidence + action from signal engine
 349:     3. Fallback: 0.5 (no edge)
 350:     """
 351:     # 1. Check focus_probabilities in summary
 352:     focus_probs = agent_summary.get("focus_probabilities", {})
 353:     ticker_probs = focus_probs.get(ticker, {})
 354:     prob_1d = ticker_probs.get("1d", {})
 355:     if prob_1d and "probability" in prob_1d:
 356:         return prob_1d["probability"]
 357: 
 358:     # 2. Derive from weighted confidence + action
 359:     extra = ticker_data.get("extra", {})
 360:     action = extra.get("_weighted_action") or ticker_data.get("action", "HOLD")
 361:     conf = extra.get("_weighted_confidence") or ticker_data.get("weighted_confidence", 0.5)
 362: 
 363:     if action == "BUY":
 364:         return 0.5 + conf * 0.3  # Scale to 0.5-0.8 range
 365:     elif action == "SELL":
 366:         return 0.5 - conf * 0.3  # Scale to 0.2-0.5 range
 367:     else:
 368:         return 0.5  # HOLD = no directional edge
 369: 
 370: 
 371: # ---------------------------------------------------------------------------
 372: # Batch simulation for all interesting tickers
 373: # ---------------------------------------------------------------------------
 374: 
 375: def simulate_all(agent_summary: dict, tickers: list[str] | None = None,
 376:                  n_paths: int = DEFAULT_N_PATHS, seed: int | None = None) -> dict:
 377:     """Run MC simulation for multiple tickers.
 378: 
 379:     Args:
 380:         agent_summary: Full agent summary dict.
 381:         tickers: List of tickers to simulate. If None, uses held positions + focus tickers.
 382:         n_paths: Paths per ticker per horizon.
 383:         seed: Base seed (incremented per ticker for independence).
 384: 
 385:     Returns:
 386:         Dict mapping ticker → simulation results.
 387:     """
 388:     if tickers is None:
 389:         tickers = _interesting_tickers(agent_summary)
 390: 
 391:     results = {}
 392:     for i, ticker in enumerate(tickers):
 393:         try:
 394:             # 2026-04-22: guard seed=None. reporting.py calls simulate_all
 395:             # without a seed; `seed + i` was crashing every cycle with
 396:             # TypeError and silently disabling MC risk sim for all tickers.
 397:             ticker_seed = None if seed is None else seed + i
 398:             result = simulate_ticker(ticker, agent_summary,
 399:                                      n_paths=n_paths, seed=ticker_seed)
 400:             if result:
 401:                 results[ticker] = result
 402:         except Exception:
 403:             logger.warning("MC simulation failed for %s", ticker, exc_info=True)
 404: 
 405:     return results
 406: 
 407: 
 408: def _interesting_tickers(agent_summary: dict) -> list[str]:
 409:     """Determine which tickers to simulate (held + focus + signaling)."""
 410:     tickers = set()
 411: 
 412:     # Focus tickers (from config, surfaced in summary)
 413:     for ft in agent_summary.get("focus_tickers", ["XAG-USD", "BTC-USD"]):
 414:         tickers.add(ft)
 415: 
 416:     # Tickers with active consensus (BUY or SELL)
 417:     for ticker, data in agent_summary.get("signals", {}).items():
 418:         action = data.get("action", "HOLD")
 419:         if action in ("BUY", "SELL"):
 420:             tickers.add(ticker)
 421: 
 422:     return sorted(tickers)

### FILE monte_carlo_risk.py
   1: """Portfolio VaR with t-copula correlated simulation.
   2: 
   3: Computes Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and
   4: joint drawdown probabilities for multi-position portfolios using a Student-t
   5: copula for tail dependence.
   6: 
   7: The t-copula captures the empirical fact that assets crash together more
   8: often than a Gaussian copula predicts (tail dependence lambda ~0.18 at
   9: df=4 vs. 0 for Gaussian).
  10: 
  11: Usage:
  12:     from portfolio.monte_carlo_risk import compute_portfolio_var
  13:     result = compute_portfolio_var(portfolio_state, agent_summary)
  14:     # result = {
  15:     #   "var_95_usd": -1234.56,
  16:     #   "cvar_95_usd": -1567.89,
  17:     #   "var_99_usd": -2345.67,
  18:     #   "total_exposure_usd": 50000.0,
  19:     #   ...
  20:     # }
  21: """
  22: 
  23: import logging
  24: import math
  25: 
  26: import numpy as np
  27: from scipy.stats import norm
  28: from scipy.stats import t as t_dist
  29: 
  30: from portfolio.monte_carlo import (
  31:     MIN_VOLATILITY,
  32:     drift_from_probability,
  33:     volatility_from_atr,
  34: )
  35: 
  36: logger = logging.getLogger("portfolio.monte_carlo_risk")
  37: 
  38: DEFAULT_DF = 4        # Degrees of freedom for t-copula (4 = moderate fat tails)
  39: DEFAULT_N_PATHS = 10_000
  40: 
  41: 
  42: # ---------------------------------------------------------------------------
  43: # Correlation matrix estimation
  44: # ---------------------------------------------------------------------------
  45: 
  46: def estimate_correlation_matrix(
  47:     returns: dict[str, np.ndarray],
  48:     tickers: list[str],
  49: ) -> np.ndarray:
  50:     """Estimate correlation matrix from historical return series.
  51: 
  52:     For tickers with insufficient data, defaults to zero correlation
  53:     (independent assumption) which is conservative for VaR.
  54: 
  55:     Args:
  56:         returns: Dict mapping ticker → array of log-returns.
  57:         tickers: Ordered list of tickers (defines matrix row/column order).
  58: 
  59:     Returns:
  60:         Correlation matrix, shape (n, n), guaranteed symmetric and PSD.
  61:     """
  62:     n = len(tickers)
  63:     if n == 0:
  64:         return np.array([]).reshape(0, 0)
  65: 
  66:     corr = np.eye(n)
  67: 
  68:     # Build return matrix for tickers with data
  69:     for i in range(n):
  70:         for j in range(i + 1, n):
  71:             ri = returns.get(tickers[i])
  72:             rj = returns.get(tickers[j])
  73:             if ri is not None and rj is not None:
  74:                 # Align lengths (use shorter)
  75:                 min_len = min(len(ri), len(rj))
  76:                 if min_len >= 20:  # Need at least 20 observations
  77:                     c = np.corrcoef(ri[:min_len], rj[:min_len])[0, 1]
  78:                     if np.isfinite(c):
  79:                         corr[i, j] = c
  80:                         corr[j, i] = c
  81: 
  82:     # Ensure PSD via eigenvalue clipping (Higham nearest PSD)
  83:     corr = _nearest_psd(corr)
  84:     return corr
  85: 
  86: 
  87: def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
  88:     """Project matrix to nearest positive semi-definite correlation matrix.
  89: 
  90:     Uses eigenvalue clipping: set negative eigenvalues to a small positive
  91:     value, then rescale diagonal to 1.
  92:     """
  93:     eigenvalues, eigenvectors = np.linalg.eigh(matrix)
  94:     # Clip negative eigenvalues
  95:     eigenvalues = np.maximum(eigenvalues, 1e-8)
  96:     # Reconstruct
  97:     result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
  98:     # Rescale to correlation matrix (diagonal = 1)
  99:     d = np.sqrt(np.diag(result))
 100:     d[d == 0] = 1.0
 101:     result = result / np.outer(d, d)
 102:     # Enforce exact symmetry
 103:     result = (result + result.T) / 2
 104:     np.fill_diagonal(result, 1.0)
 105:     return result
 106: 
 107: 
 108: # ---------------------------------------------------------------------------
 109: # Known correlation pairs (fallback when no historical data)
 110: # ---------------------------------------------------------------------------
 111: 
 112: # From risk_management.py — approximate correlation strengths
 113: CORRELATION_PRIORS = {
 114:     ("BTC-USD", "ETH-USD"): 0.75,
 115:     ("XAG-USD", "XAU-USD"): 0.85,
 116:     ("NVDA", "AMD"): 0.70,
 117:     ("NVDA", "AVGO"): 0.65,
 118:     ("NVDA", "TSM"): 0.60,
 119:     ("AMD", "AVGO"): 0.60,
 120:     ("AMD", "TSM"): 0.55,
 121:     ("AVGO", "TSM"): 0.55,
 122:     ("GOOGL", "META"): 0.65,
 123:     ("GOOGL", "AMZN"): 0.60,
 124:     ("META", "AMZN"): 0.60,
 125:     ("AAPL", "GOOGL"): 0.55,
 126:     ("AAPL", "META"): 0.50,
 127:     ("AAPL", "AMZN"): 0.55,
 128: }
 129: 
 130: 
 131: def _get_prior_correlation(ticker_a: str, ticker_b: str) -> float:
 132:     """Look up prior correlation for a pair (order-independent)."""
 133:     return CORRELATION_PRIORS.get(
 134:         (ticker_a, ticker_b),
 135:         CORRELATION_PRIORS.get((ticker_b, ticker_a), 0.0),
 136:     )
 137: 
 138: 
 139: def build_correlation_matrix(
 140:     tickers: list[str],
 141:     historical_returns: dict[str, np.ndarray] | None = None,
 142: ) -> np.ndarray:
 143:     """Build correlation matrix using historical data with prior fallback.
 144: 
 145:     If historical returns are available and sufficient, uses empirical
 146:     estimates. Otherwise falls back to hardcoded priors from CORRELATED_PAIRS.
 147: 
 148:     Args:
 149:         tickers: Ordered list of tickers.
 150:         historical_returns: Optional dict of ticker → log-return arrays.
 151: 
 152:     Returns:
 153:         Correlation matrix (n x n).
 154:     """
 155:     n = len(tickers)
 156:     if n == 0:
 157:         return np.array([]).reshape(0, 0)
 158: 
 159:     if historical_returns:
 160:         # Check if we have sufficient data (>= 30 observations per ticker)
 161:         has_data = sum(
 162:             1 for t in tickers
 163:             if t in historical_returns and len(historical_returns[t]) >= 30
 164:         )
 165:         if has_data >= 2:
 166:             return estimate_correlation_matrix(historical_returns, tickers)
 167: 
 168:     # Fallback: use priors
 169:     corr = np.eye(n)
 170:     for i in range(n):
 171:         for j in range(i + 1, n):
 172:             c = _get_prior_correlation(tickers[i], tickers[j])
 173:             corr[i, j] = c
 174:             corr[j, i] = c
 175: 
 176:     return _nearest_psd(corr)
 177: 
 178: 
 179: # ---------------------------------------------------------------------------
 180: # Portfolio risk simulator
 181: # ---------------------------------------------------------------------------
 182: 
 183: class PortfolioRiskSimulator:
 184:     """Multi-position portfolio risk simulator using Student-t copula.
 185: 
 186:     Generates correlated terminal returns for all positions jointly,
 187:     then computes portfolio-level P&L, VaR, and CVaR.
 188: 
 189:     The t-copula with df=4 captures tail dependence (assets crash together)
 190:     that a Gaussian copula misses entirely.
 191: 
 192:     Args:
 193:         positions: Dict mapping ticker → {shares, price_usd, volatility, drift}.
 194:         correlation_matrix: Correlation matrix for position tickers.
 195:         horizon_days: Simulation horizon.
 196:         n_paths: Number of simulation paths.
 197:         df: Degrees of freedom for t-copula (lower = fatter tails).
 198:         seed: Random seed.
 199:     """
 200: 
 201:     def __init__(
 202:         self,
 203:         positions: dict,
 204:         correlation_matrix: np.ndarray,
 205:         horizon_days: float = 1.0,
 206:         n_paths: int = DEFAULT_N_PATHS,
 207:         df: int = DEFAULT_DF,
 208:         seed: int | None = None,
 209:     ):
 210:         # Filter out zero-share positions
 211:         self._tickers = [t for t, p in positions.items() if p.get("shares", 0) != 0]
 212:         self._positions = {t: positions[t] for t in self._tickers}
 213:         self._n_assets = len(self._tickers)
 214: 
 215:         # If we filtered tickers, extract the sub-matrix
 216:         if self._n_assets > 0 and correlation_matrix.size > 0:
 217:             all_tickers = list(positions.keys())
 218:             indices = [all_tickers.index(t) for t in self._tickers]
 219:             self._corr = correlation_matrix[np.ix_(indices, indices)]
 220:         else:
 221:             self._corr = np.array([]).reshape(0, 0)
 222: 
 223:         self.horizon_days = horizon_days
 224:         self.n_paths = n_paths
 225:         self.df = df
 226:         self.seed = seed
 227: 
 228:         self._returns = None  # (n_paths, n_assets) log-returns
 229:         self._pnl = None      # (n_paths,) portfolio P&L in USD
 230: 
 231:     def simulate_correlated_returns(self) -> np.ndarray:
 232:         """Generate correlated returns using t-copula.
 233: 
 234:         Algorithm:
 235:         1. Cholesky decompose correlation matrix: L = chol(R)
 236:         2. Generate independent standard normals: Z ~ N(0, I)
 237:         3. Correlate: W = Z @ L^T
 238:         4. Generate chi-squared: S ~ chi2(df)
 239:         5. Scale to t: T = W * sqrt(df / S) → multivariate t
 240:         6. Transform marginals: apply per-asset drift and volatility
 241: 
 242:         Returns:
 243:             Array of shape (n_paths, n_assets) with log-returns.
 244:         """
 245:         if self._n_assets == 0:
 246:             self._returns = np.empty((self.n_paths, 0))
 247:             return self._returns
 248: 
 249:         rng = np.random.default_rng(self.seed)
 250:         T = self.horizon_days / 252.0
 251: 
 252:         # Step 1: Cholesky decomposition
 253:         try:
 254:             L = np.linalg.cholesky(self._corr)
 255:         except np.linalg.LinAlgError:
 256:             logger.warning(
 257:                 "Cholesky decomposition failed for %d-asset correlation matrix — "
 258:                 "falling back to nearest PSD projection",
 259:                 self._n_assets,
 260:             )
 261:             L = np.linalg.cholesky(_nearest_psd(self._corr))
 262: 
 263:         # Step 2: Independent standard normals
 264:         Z = rng.standard_normal((self.n_paths, self._n_assets))
 265: 
 266:         # Step 3: Correlate
 267:         W = Z @ L.T
 268: 
 269:         # Step 4: Chi-squared scaling for t-distribution
 270:         S = rng.chisquare(self.df, size=self.n_paths)
 271: 
 272:         # Step 5: Scale to multivariate t
 273:         # T_i = W_i * sqrt(df / S) for each path
 274:         scale = np.sqrt(self.df / S)[:, np.newaxis]
 275:         T_samples = W * scale
 276: 
 277:         # Step 6: Transform to uniform via t CDF, then to Gaussian marginals
 278:         # U = F_t(T; df) → uniform on [0,1]
 279:         U = t_dist.cdf(T_samples, df=self.df)
 280: 
 281:         # Transform each marginal to GBM log-return
 282:         # C9 FIX: Use norm.ppf (Gaussian inverse CDF), NOT t_dist.ppf.
 283:         # Using t_dist.ppf(t_dist.cdf(x, df), df) is an identity transform —
 284:         # it round-trips back to the same t-distributed samples, inflating
 285:         # variance by ~sqrt(df/(df-2)) ≈ sqrt(2) at df=4. The correct
 286:         # t-copula + GBM approach is: t-copula for dependence structure,
 287:         # Gaussian marginals for the GBM diffusion term.
 288:         returns = np.empty_like(U)
 289:         for i, ticker in enumerate(self._tickers):
 290:             pos = self._positions[ticker]
 291:             sigma = max(pos["volatility"], MIN_VOLATILITY)
 292:             mu = pos.get("drift", 0.0)
 293: 
 294:             # Inverse normal CDF to get standard normal quantiles
 295:             Z_marginal = norm.ppf(U[:, i])
 296: 
 297:             # GBM log-return: (mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z
 298:             drift_term = (mu - 0.5 * sigma**2) * T
 299:             vol_term = sigma * math.sqrt(T)
 300:             returns[:, i] = drift_term + vol_term * Z_marginal
 301: 
 302:         self._returns = returns
 303:         return returns
 304: 
 305:     def _ensure_simulated(self):
 306:         """Run simulation if not already done."""
 307:         if self._returns is None:
 308:             self.simulate_correlated_returns()
 309: 
 310:     def portfolio_pnl(self) -> np.ndarray:
 311:         """Compute portfolio P&L distribution in USD.
 312: 
 313:         P&L = sum over assets of (shares * price * (exp(log_return) - 1))
 314: 
 315:         Returns:
 316:             1D array of portfolio P&L values, shape (n_paths,).
 317:         """
 318:         self._ensure_simulated()
 319: 
 320:         if self._n_assets == 0:
 321:             self._pnl = np.zeros(self.n_paths)
 322:             return self._pnl
 323: 
 324:         pnl = np.zeros(self.n_paths)
 325:         for i, ticker in enumerate(self._tickers):
 326:             pos = self._positions[ticker]
 327:             shares = pos["shares"]
 328:             price = pos["price_usd"]
 329:             # P&L per path = shares * price * (exp(return) - 1)
 330:             pnl += shares * price * (np.exp(self._returns[:, i]) - 1)
 331: 
 332:         self._pnl = pnl
 333:         return pnl
 334: 
 335:     def _ensure_pnl(self):
 336:         """Compute P&L if not already done."""
 337:         if self._pnl is None:
 338:             self.portfolio_pnl()
 339: 
 340:     def var(self, confidence: float = 0.95) -> float:
 341:         """Compute Value-at-Risk at given confidence level.
 342: 
 343:         VaR is the loss at the (1-confidence) percentile of the P&L distribution.
 344:         Returns a negative number (loss).
 345: 
 346:         Args:
 347:             confidence: Confidence level (0.95 or 0.99 typically).
 348: 
 349:         Returns:
 350:             VaR in USD (negative = loss).
 351:         """
 352:         self._ensure_pnl()
 353:         if self._n_assets == 0:
 354:             return 0.0
 355:         percentile = (1 - confidence) * 100
 356:         return float(np.percentile(self._pnl, percentile))
 357: 
 358:     def cvar(self, confidence: float = 0.95) -> float:
 359:         """Compute Conditional VaR (Expected Shortfall) at given confidence.
 360: 
 361:         CVaR is the average loss in the worst (1-confidence) scenarios.
 362:         Always worse than VaR (further from zero).
 363: 
 364:         Args:
 365:             confidence: Confidence level.
 366: 
 367:         Returns:
 368:             CVaR in USD (negative = loss).
 369:         """
 370:         self._ensure_pnl()
 371:         if self._n_assets == 0:
 372:             return 0.0
 373:         var_threshold = self.var(confidence)
 374:         tail = self._pnl[self._pnl <= var_threshold]
 375:         if len(tail) == 0:
 376:             return var_threshold
 377:         return float(np.mean(tail))
 378: 
 379:     def drawdown_probability(self, threshold_pct: float) -> float:
 380:         """Probability of portfolio losing more than threshold_pct.
 381: 
 382:         Args:
 383:             threshold_pct: Loss threshold as percentage (e.g., 5.0 = 5%).
 384: 
 385:         Returns:
 386:             Probability (0.0-1.0) of exceeding the loss threshold.
 387:         """
 388:         self._ensure_pnl()
 389:         if self._n_assets == 0:
 390:             return 0.0
 391: 
 392:         total_value = sum(
 393:             p["shares"] * p["price_usd"] for p in self._positions.values()
 394:         )
 395:         if total_value <= 0:
 396:             return 0.0
 397: 
 398:         loss_threshold = -total_value * threshold_pct / 100.0
 399:         return float(np.mean(self._pnl < loss_threshold))
 400: 
 401: 
 402: # ---------------------------------------------------------------------------
 403: # Convenience: compute portfolio VaR from system data structures
 404: # ---------------------------------------------------------------------------
 405: 
 406: def compute_portfolio_var(
 407:     portfolio_state: dict,
 408:     agent_summary: dict,
 409:     n_paths: int = DEFAULT_N_PATHS,
 410:     historical_returns: dict[str, np.ndarray] | None = None,
 411:     seed: int | None = None,
 412: ) -> dict:
 413:     """Compute portfolio VaR from portfolio state and agent summary.
 414: 
 415:     Extracts held positions, current prices, volatilities, and directional
 416:     probabilities from the system data, then runs t-copula simulation.
 417: 
 418:     Args:
 419:         portfolio_state: Portfolio state dict (holdings, cash_sek, etc.).
 420:         agent_summary: Agent summary with signals, fx_rate.
 421:         n_paths: Number of MC paths.
 422:         historical_returns: Optional empirical return series for correlation.
 423:         seed: Random seed.
 424: 
 425:     Returns:
 426:         Dict with VaR metrics in both USD and SEK.
 427:     """
 428:     holdings = portfolio_state.get("holdings", {})
 429:     signals = agent_summary.get("signals", {})
 430:     fx_rate = agent_summary.get("fx_rate", 10.0)
 431: 
 432:     # Build positions dict
 433:     positions = {}
 434:     tickers = []
 435: 
 436:     for ticker, holding in holdings.items():
 437:         shares = holding.get("shares", 0)
 438:         if shares <= 0:
 439:             continue
 440: 
 441:         ticker_data = signals.get(ticker, {})
 442:         price = ticker_data.get("price_usd", 0)
 443:         if price <= 0:
 444:             continue
 445: 
 446:         extra = ticker_data.get("extra", {})
 447:         atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
 448:         vol = volatility_from_atr(atr_pct)
 449: 
 450:         # Get directional probability for drift
 451:         from portfolio.monte_carlo import _get_directional_probability
 452:         p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
 453:         drift = drift_from_probability(p_up, vol)
 454: 
 455:         tickers.append(ticker)
 456:         positions[ticker] = {
 457:             "shares": shares,
 458:             "price_usd": price,
 459:             "volatility": vol,
 460:             "drift": drift,
 461:         }
 462: 
 463:     if not tickers:
 464:         return {
 465:             "var_95_usd": 0.0,
 466:             "var_99_usd": 0.0,
 467:             "cvar_95_usd": 0.0,
 468:             "cvar_99_usd": 0.0,
 469:             "var_95_sek": 0.0,
 470:             "var_99_sek": 0.0,
 471:             "cvar_95_sek": 0.0,
 472:             "cvar_99_sek": 0.0,
 473:             "total_exposure_usd": 0.0,
 474:             "total_exposure_sek": 0.0,
 475:             "n_positions": 0,
 476:             "drawdown_1pct_prob": 0.0,
 477:             "drawdown_5pct_prob": 0.0,
 478:         }
 479: 
 480:     # Build correlation matrix
 481:     corr = build_correlation_matrix(tickers, historical_returns)
 482: 
 483:     # Run simulation
 484:     sim = PortfolioRiskSimulator(
 485:         positions=positions,
 486:         correlation_matrix=corr,
 487:         horizon_days=1,
 488:         n_paths=n_paths,
 489:         df=DEFAULT_DF,
 490:         seed=seed,
 491:     )
 492:     sim.simulate_correlated_returns()
 493: 
 494:     total_exposure = sum(p["shares"] * p["price_usd"] for p in positions.values())
 495: 
 496:     var95 = sim.var(0.95)
 497:     var99 = sim.var(0.99)
 498:     cvar95 = sim.cvar(0.95)
 499:     cvar99 = sim.cvar(0.99)
 500: 
 501:     return {
 502:         "var_95_usd": round(var95, 2),
 503:         "var_99_usd": round(var99, 2),
 504:         "cvar_95_usd": round(cvar95, 2),
 505:         "cvar_99_usd": round(cvar99, 2),
 506:         "var_95_sek": round(var95 * fx_rate, 2),
 507:         "var_99_sek": round(var99 * fx_rate, 2),
 508:         "cvar_95_sek": round(cvar95 * fx_rate, 2),
 509:         "cvar_99_sek": round(cvar99 * fx_rate, 2),
 510:         "total_exposure_usd": round(total_exposure, 2),
 511:         "total_exposure_sek": round(total_exposure * fx_rate, 2),
 512:         "n_positions": len(tickers),
 513:         "drawdown_1pct_prob": round(sim.drawdown_probability(1.0), 3),
 514:         "drawdown_5pct_prob": round(sim.drawdown_probability(5.0), 3),
 515:     }

### FILE portfolio_validator.py
   1: """Portfolio state validation utilities.
   2: 
   3: Validates portfolio state files for data integrity, ensuring that cash,
   4: holdings, fees, and transaction records are all internally consistent.
   5: """
   6: 
   7: import pathlib
   8: from collections import defaultdict
   9: 
  10: from portfolio.file_utils import load_json
  11: 
  12: 
  13: def validate_portfolio(portfolio: dict) -> list[str]:
  14:     """Validate portfolio state integrity.
  15: 
  16:     Performs comprehensive checks on a portfolio state dict to ensure
  17:     all invariants hold. Returns a list of error messages -- an empty
  18:     list means the portfolio is valid.
  19: 
  20:     Checks:
  21:     1. Cash is non-negative
  22:     2. All shares are non-negative
  23:     3. Cash reconciliation: initial - sum(BUY allocs) + sum(SELL net_proceeds) = cash_sek
  24:     4. Holdings shares reconciliation: sum(BUY shares) - sum(SELL shares) per ticker
  25:     5. Fee reconciliation: total_fees_sek = sum(all fee_sek in transactions)
  26:     6. No duplicate timestamps in transactions (same ticker + same timestamp)
  27:     7. Transaction field completeness
  28:     8. Holdings avg_cost_usd consistency
  29: 
  30:     Args:
  31:         portfolio: Full portfolio state dict with keys:
  32:             - cash_sek, holdings, transactions, initial_value_sek, total_fees_sek
  33: 
  34:     Returns:
  35:         list of error message strings (empty = valid).
  36:     """
  37:     errors = []
  38: 
  39:     # --- Extract fields with defaults ---
  40:     cash_sek = portfolio.get("cash_sek")
  41:     holdings = portfolio.get("holdings", {})
  42:     transactions = portfolio.get("transactions", [])
  43:     initial_value = portfolio.get("initial_value_sek", 500_000)
  44:     total_fees_sek = portfolio.get("total_fees_sek")
  45: 
  46:     # --- Check 0: Required fields exist ---
  47:     if cash_sek is None:
  48:         errors.append("Missing required field: cash_sek")
  49:         cash_sek = 0
  50:     if total_fees_sek is None:
  51:         errors.append("Missing or null field: total_fees_sek (should be 0 if no fees)")
  52:         total_fees_sek = 0
  53:     if "initial_value_sek" not in portfolio:
  54:         errors.append("Missing required field: initial_value_sek")
  55: 
  56:     # --- Check 1: Cash is non-negative ---
  57:     if cash_sek < 0:
  58:         errors.append(f"Cash is negative: {cash_sek:.2f} SEK")
  59: 
  60:     # --- Check 2: All shares are non-negative ---
  61:     for ticker, pos in holdings.items():
  62:         shares = pos.get("shares", 0)
  63:         if shares < 0:
  64:             errors.append(f"Negative shares for {ticker}: {shares}")
  65:         avg_cost = pos.get("avg_cost_usd")
  66:         if avg_cost is not None and avg_cost < 0:
  67:             errors.append(f"Negative avg_cost_usd for {ticker}: {avg_cost}")
  68: 
  69:     # --- Check 3: Cash reconciliation ---
  70:     # cash = initial - sum(BUY total_sek) + sum(SELL total_sek)
  71:     # Note: BUY total_sek = full allocation (including fee)
  72:     # SELL total_sek = net proceeds (after fee deducted)
  73:     total_buy_alloc = 0.0
  74:     total_sell_proceeds = 0.0
  75:     for tx in transactions:
  76:         action = tx.get("action", "")
  77:         total_sek = tx.get("total_sek", 0) or 0
  78:         if action == "BUY":
  79:             total_buy_alloc += total_sek
  80:         elif action == "SELL":
  81:             total_sell_proceeds += total_sek
  82: 
  83:     expected_cash = initial_value - total_buy_alloc + total_sell_proceeds
  84:     cash_diff = abs(expected_cash - cash_sek)
  85:     if cash_diff > 1.0:  # Allow 1 SEK tolerance for floating point
  86:         errors.append(
  87:             f"Cash reconciliation failed: expected {expected_cash:.2f} SEK "
  88:             f"(initial {initial_value} - buys {total_buy_alloc:.2f} + sells {total_sell_proceeds:.2f}), "
  89:             f"got {cash_sek:.2f} SEK (diff: {cash_diff:.2f})"
  90:         )
  91: 
  92:     # --- Check 4: Holdings shares reconciliation ---
  93:     # For each ticker: net_shares = sum(BUY shares) - sum(SELL shares)
  94:     ticker_bought = defaultdict(float)
  95:     ticker_sold = defaultdict(float)
  96:     for tx in transactions:
  97:         ticker = tx.get("ticker", "")
  98:         action = tx.get("action", "")
  99:         shares = tx.get("shares", 0) or 0
 100:         if action == "BUY":
 101:             ticker_bought[ticker] += shares
 102:         elif action == "SELL":
 103:             ticker_sold[ticker] += shares
 104: 
 105:     # Check tickers that appear in transactions
 106:     all_tx_tickers = set(ticker_bought.keys()) | set(ticker_sold.keys())
 107:     for ticker in all_tx_tickers:
 108:         expected_shares = ticker_bought[ticker] - ticker_sold[ticker]
 109: 
 110:         # Get actual shares from holdings
 111:         if ticker in holdings:
 112:             actual_shares = holdings[ticker].get("shares", 0)
 113:         else:
 114:             actual_shares = 0
 115: 
 116:         # Compare (with tolerance for floating point and rounding from repeated partial sells)
 117:         share_diff = abs(expected_shares - actual_shares)
 118:         if share_diff > 1e-6:
 119:             if expected_shares <= 1e-9 and actual_shares == 0:
 120:                 # Both effectively zero -- OK (sold all, removed from holdings)
 121:                 continue
 122:             if expected_shares <= 1e-9 and ticker not in holdings:
 123:                 # Fully sold, ticker removed from holdings -- OK
 124:                 continue
 125:             # Tolerance for small remainders from repeated partial sells (e.g.,
 126:             # multiple 50% sells that don't sum exactly to total bought due to
 127:             # floating-point rounding). Allow up to 1% of total bought shares.
 128:             total_bought = ticker_bought[ticker]
 129:             relative_diff = share_diff / total_bought if total_bought > 0 else float("inf")
 130:             if actual_shares == 0 and ticker not in holdings and relative_diff < 0.01:
 131:                 # Small remainder from rounding, ticker removed -- acceptable
 132:                 continue
 133:             errors.append(
 134:                 f"Holdings mismatch for {ticker}: expected {expected_shares:.8f} shares "
 135:                 f"(bought {ticker_bought[ticker]:.8f} - sold {ticker_sold[ticker]:.8f}), "
 136:                 f"got {actual_shares:.8f} in holdings (diff: {share_diff:.8f})"
 137:             )
 138: 
 139:     # Check for holdings tickers not in transactions
 140:     for ticker in holdings:
 141:         shares = holdings[ticker].get("shares", 0)
 142:         if shares > 0 and ticker not in all_tx_tickers:
 143:             errors.append(
 144:                 f"Holdings contains {ticker} with {shares} shares but no matching transactions"
 145:             )
 146: 
 147:     # --- Check 5: Fee reconciliation ---
 148:     computed_fees = 0.0
 149:     tx_with_fees = 0
 150:     tx_without_fees = 0
 151:     for tx in transactions:
 152:         fee = tx.get("fee_sek")
 153:         if fee is not None:
 154:             computed_fees += fee
 155:             tx_with_fees += 1
 156:         else:
 157:             tx_without_fees += 1
 158: 
 159:     if tx_without_fees > 0 and tx_with_fees > 0:
 160:         errors.append(
 161:             f"Inconsistent fee tracking: {tx_with_fees} transactions have fee_sek, "
 162:             f"{tx_without_fees} do not"
 163:         )
 164: 
 165:     # Only compare fees if transactions have fee_sek fields
 166:     if tx_with_fees > 0:
 167:         fee_diff = abs(computed_fees - total_fees_sek)
 168:         if fee_diff > 0.01:  # 0.01 SEK tolerance
 169:             errors.append(
 170:                 f"Fee reconciliation failed: sum of transaction fees = {computed_fees:.2f} SEK, "
 171:                 f"total_fees_sek = {total_fees_sek:.2f} SEK (diff: {fee_diff:.2f})"
 172:             )
 173:     elif len(transactions) > 0 and total_fees_sek == 0:
 174:         # Transactions exist but no fee tracking at all -- warn
 175:         errors.append(
 176:             "No fee_sek fields in any transaction and total_fees_sek is 0, "
 177:             "but transactions exist. Fees may not be tracked."
 178:         )
 179: 
 180:     # --- Check 6: No duplicate timestamps per ticker ---
 181:     seen_tx = set()
 182:     for i, tx in enumerate(transactions):
 183:         key = (tx.get("ticker", ""), tx.get("timestamp", ""), tx.get("action", ""))
 184:         if key in seen_tx:
 185:             errors.append(
 186:                 f"Duplicate transaction at index {i}: {key[2]} {key[0]} at {key[1]}"
 187:             )
 188:         seen_tx.add(key)
 189: 
 190:     # --- Check 7: Transaction field completeness ---
 191:     required_tx_fields = ["timestamp", "ticker", "action", "shares", "price_usd",
 192:                           "total_sek", "reason"]
 193: 
 194:     for i, tx in enumerate(transactions):
 195:         for field in required_tx_fields:
 196:             if field not in tx or tx[field] is None:
 197:                 errors.append(f"Transaction {i} missing required field: {field}")
 198: 
 199:         # Validate action value
 200:         action = tx.get("action", "")
 201:         if action not in ("BUY", "SELL"):
 202:             errors.append(f"Transaction {i} has invalid action: '{action}' (expected BUY or SELL)")
 203: 
 204:         # Validate shares > 0
 205:         shares = tx.get("shares", 0)
 206:         if shares is not None and shares <= 0:
 207:             errors.append(f"Transaction {i} has non-positive shares: {shares}")
 208: 
 209:         # Validate total_sek > 0
 210:         total = tx.get("total_sek", 0)
 211:         if total is not None and total <= 0:
 212:             errors.append(f"Transaction {i} has non-positive total_sek: {total}")
 213: 
 214:     # --- Check 8: Holdings avg_cost_usd consistency ---
 215:     # For tickers with multiple BUY transactions, verify avg_cost is plausible
 216:     for ticker, pos in holdings.items():
 217:         shares = pos.get("shares", 0)
 218:         if shares <= 0:
 219:             continue
 220:         avg_cost = pos.get("avg_cost_usd")
 221:         if avg_cost is None:
 222:             errors.append(f"Holdings {ticker} missing avg_cost_usd")
 223:             continue
 224: 
 225:         # Compute weighted average from BUY transactions
 226:         total_cost = 0.0
 227:         total_bought = 0.0
 228:         for tx in transactions:
 229:             if tx.get("ticker") != ticker or tx.get("action") != "BUY":
 230:                 continue
 231:             tx_shares = tx.get("shares", 0) or 0
 232:             tx_price = tx.get("price_usd", 0) or 0
 233:             total_cost += tx_shares * tx_price
 234:             total_bought += tx_shares
 235: 
 236:         if total_bought > 0:
 237:             expected_avg = total_cost / total_bought
 238:             avg_diff_pct = abs(expected_avg - avg_cost) / expected_avg * 100 if expected_avg > 0 else 0
 239:             if avg_diff_pct > 1.0:  # More than 1% off
 240:                 errors.append(
 241:                     f"Holdings {ticker} avg_cost_usd ({avg_cost:.4f}) differs from "
 242:                     f"computed weighted average ({expected_avg:.4f}) by {avg_diff_pct:.2f}%"
 243:                 )
 244: 
 245:     return errors
 246: 
 247: 
 248: def validate_portfolio_file(path: str) -> list[str]:
 249:     """Validate a portfolio state JSON file.
 250: 
 251:     Convenience wrapper that loads the file and runs validate_portfolio().
 252: 
 253:     Args:
 254:         path: Path to the portfolio state JSON file.
 255: 
 256:     Returns:
 257:         list of error message strings (empty = valid).
 258:     """
 259:     # A-PR-3 (2026-04-11): Use file_utils.load_json instead of raw open()+
 260:     # json.load(). The raw path had a TOCTOU race with concurrent saves
 261:     # from portfolio_mgr.atomic_write_json: a partial write could be read
 262:     # mid-flight, producing a JSONDecodeError that propagated as a "valid
 263:     # but malformed" portfolio. load_json() retries on transient JSON
 264:     # decode failures and uses the same atomic-rename window as the writer.
 265:     p = pathlib.Path(path)
 266:     if not p.exists():
 267:         return [f"Portfolio file not found: {path}"]
 268:     portfolio = load_json(p)
 269:     if portfolio is None:
 270:         return [f"Invalid or unreadable JSON in {path}"]
 271: 
 272:     return validate_portfolio(portfolio)
 273: 
 274: 
 275: def validate_all() -> dict[str, list[str]]:
 276:     """Validate both patient and bold portfolio files.
 277: 
 278:     Returns:
 279:         dict with keys "patient" and "bold", each containing a list of errors.
 280:     """
 281:     data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
 282:     return {
 283:         "patient": validate_portfolio_file(str(data_dir / "portfolio_state.json")),
 284:         "bold": validate_portfolio_file(str(data_dir / "portfolio_state_bold.json")),
 285:     }
 286: 
 287: 
 288: if __name__ == "__main__":
 289:     results = validate_all()
 290:     for strategy, errs in results.items():
 291:         print(f"\n{'='*60}")
 292:         print(f"  {strategy.upper()} PORTFOLIO VALIDATION")
 293:         print(f"{'='*60}")
 294:         if errs:
 295:             for e in errs:
 296:                 print(f"  ERROR: {e}")
 297:         else:
 298:             print("  VALID - all checks passed")
 299:     print()

### FILE trade_guards.py
   1: """Trade guards — overtrading prevention for the trading agent.
   2: 
   3: Three guards:
   4: 1. Per-ticker cooldown: No re-trade on same ticker within N minutes.
   5: 2. Consecutive-loss escalation: After losses, increase cooldown multiplier.
   6: 3. Position rate limit: Max N new positions per time window.
   7: 
   8: State is persisted to data/trade_guard_state.json.
   9: """
  10: 
  11: import logging
  12: import threading
  13: from datetime import UTC, datetime, timedelta
  14: from pathlib import Path
  15: 
  16: from portfolio.file_utils import atomic_write_json, load_json
  17: 
  18: logger = logging.getLogger("portfolio.trade_guards")
  19: 
  20: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
  21: STATE_FILE = DATA_DIR / "trade_guard_state.json"
  22: 
  23: # Defaults
  24: DEFAULT_TICKER_COOLDOWN_MINUTES = 30
  25: DEFAULT_BOLD_POSITION_LIMIT = 1       # max new positions per window
  26: DEFAULT_BOLD_POSITION_WINDOW_H = 4    # hours
  27: DEFAULT_PATIENT_POSITION_LIMIT = 1
  28: DEFAULT_PATIENT_POSITION_WINDOW_H = 8
  29: LOSS_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive_losses -> cooldown multiplier
  30: LOSS_DECAY_HOURS = 24  # halve escalation multiplier every N hours without a trade
  31: 
  32: _state_lock = threading.Lock()
  33: 
  34: 
  35: def _load_state():
  36:     """Load trade guard state from disk."""
  37:     return load_json(str(STATE_FILE), default={
  38:         "ticker_trades": {},
  39:         "consecutive_losses": {"patient": 0, "bold": 0},
  40:         "last_loss_ts": {"patient": None, "bold": None},
  41:         "new_position_timestamps": {"patient": [], "bold": []},
  42:     })
  43: 
  44: 
  45: def _save_state(state):
  46:     """Persist trade guard state to disk."""
  47:     atomic_write_json(STATE_FILE, state)
  48: 
  49: 
  50: def _portfolios_have_transactions():
  51:     """Return True if any portfolio file has at least one recorded transaction.
  52: 
  53:     Used by the C4 sanity check to distinguish "no trades happened yet"
  54:     (quiet startup state) from "trades happened but weren't recorded"
  55:     (broken wiring — real bug).
  56: 
  57:     2026-04-22 follow-up: include warrants portfolio — CLAUDE.md lists it as
  58:     an independent strategy state file, and warrants-only activity would
  59:     have left C4 silent forever.
  60:     """
  61:     for pf_name in (
  62:         "portfolio_state.json",
  63:         "portfolio_state_bold.json",
  64:         "portfolio_state_warrants.json",
  65:     ):
  66:         pf = load_json(str(DATA_DIR / pf_name), default={})
  67:         if pf and pf.get("transactions"):
  68:             return True
  69:     return False
  70: 
  71: 
  72: def _get_cooldown_multiplier(consecutive_losses, last_loss_ts_str=None):
  73:     """Get cooldown multiplier based on consecutive loss count with time decay.
  74: 
  75:     After LOSS_DECAY_HOURS without a new trade, the multiplier halves
  76:     repeatedly (geometric decay). E.g. 8x → 4x after 24h → 2x after 48h → 1x.
  77:     """
  78:     if consecutive_losses >= 4:
  79:         base = LOSS_ESCALATION[4]
  80:     else:
  81:         base = LOSS_ESCALATION.get(consecutive_losses, 1)
  82: 
  83:     if base <= 1 or not last_loss_ts_str:
  84:         return base
  85: 
  86:     # Apply time-based decay
  87:     try:
  88:         last_loss = datetime.fromisoformat(
  89:             last_loss_ts_str.replace("Z", "+00:00")
  90:         )
  91:         if last_loss.tzinfo is None:
  92:             last_loss = last_loss.replace(tzinfo=UTC)
  93:         elapsed_hours = (datetime.now(UTC) - last_loss).total_seconds() / 3600
  94:         if elapsed_hours > LOSS_DECAY_HOURS:
  95:             halvings = int(elapsed_hours // LOSS_DECAY_HOURS)
  96:             base = max(1, base >> halvings)  # bit-shift right = halve
  97:     except (ValueError, TypeError, OverflowError):
  98:         pass
  99: 
 100:     return base
 101: 
 102: 
 103: def check_overtrading_guards(ticker, action, strategy, portfolio, config=None):
 104:     """Check all trade guards for a proposed trade.
 105: 
 106:     Args:
 107:         ticker: Instrument ticker (e.g., "BTC-USD").
 108:         action: "BUY" or "SELL".
 109:         strategy: "patient" or "bold".
 110:         portfolio: Portfolio state dict.
 111:         config: Optional config dict with trade_guards settings.
 112: 
 113:     Returns:
 114:         list of warning dicts, each with:
 115:             - guard: str (guard name)
 116:             - severity: "warning" or "block"
 117:             - message: str
 118:             - details: dict (guard-specific data)
 119:         Empty list means all guards pass.
 120:     """
 121:     cfg = (config or {}).get("trade_guards", {})
 122:     if cfg.get("enabled") is False:
 123:         return []
 124: 
 125:     warnings = []
 126:     with _state_lock:
 127:         state = _load_state()
 128:     now = datetime.now(UTC)
 129: 
 130:     # --- Guard 1: Per-ticker cooldown ---
 131:     base_cooldown = cfg.get("ticker_cooldown_minutes", DEFAULT_TICKER_COOLDOWN_MINUTES)
 132:     consecutive = state.get("consecutive_losses", {}).get(strategy, 0)
 133:     last_loss_ts = state.get("last_loss_ts", {}).get(strategy)
 134:     multiplier = _get_cooldown_multiplier(consecutive, last_loss_ts)
 135:     effective_cooldown = base_cooldown * multiplier
 136: 
 137:     key = f"{strategy}:{ticker}"
 138:     ticker_trades = state.get("ticker_trades", {})
 139:     last_trade_str = ticker_trades.get(key)
 140:     if last_trade_str:
 141:         try:
 142:             last_trade = datetime.fromisoformat(last_trade_str)
 143:             # M8: ensure aware datetime before comparison with aware now
 144:             if last_trade.tzinfo is None:
 145:                 last_trade = last_trade.replace(tzinfo=UTC)
 146:             elapsed = (now - last_trade).total_seconds() / 60
 147:             if elapsed < effective_cooldown:
 148:                 remaining = effective_cooldown - elapsed
 149:                 warnings.append({
 150:                     "guard": "ticker_cooldown",
 151:                     "severity": "block",
 152:                     "message": (
 153:                         f"{ticker} traded {elapsed:.0f}m ago by {strategy}. "
 154:                         f"Cooldown: {effective_cooldown:.0f}m (base {base_cooldown}m × {multiplier}x). "
 155:                         f"{remaining:.0f}m remaining."
 156:                     ),
 157:                     "details": {
 158:                         "ticker": ticker,
 159:                         "strategy": strategy,
 160:                         "elapsed_min": round(elapsed, 1),
 161:                         "cooldown_min": effective_cooldown,
 162:                         "multiplier": multiplier,
 163:                         "remaining_min": round(remaining, 1),
 164:                     },
 165:                 })
 166:         except (ValueError, TypeError):
 167:             pass
 168: 
 169:     # --- Guard 2: Consecutive-loss escalation (informational) ---
 170:     if consecutive >= 2:
 171:         base_mult = _get_cooldown_multiplier(consecutive, None)
 172:         warnings.append({
 173:             "guard": "consecutive_losses",
 174:             "severity": "warning",
 175:             "message": (
 176:                 f"{strategy}: {consecutive} consecutive losses. "
 177:                 f"Cooldown multiplier: {multiplier}x"
 178:                 f"{f' (decayed from {base_mult}x)' if multiplier < base_mult else ''}."
 179:             ),
 180:             "details": {
 181:                 "strategy": strategy,
 182:                 "consecutive_losses": consecutive,
 183:                 "multiplier": multiplier,
 184:                 "base_multiplier": base_mult,
 185:                 "decayed": multiplier < base_mult,
 186:             },
 187:         })
 188: 
 189:     # --- Guard 3: Position rate limit (BUY only) ---
 190:     if action == "BUY":
 191:         is_bold = strategy == "bold"
 192:         limit = cfg.get(
 193:             f"{'bold' if is_bold else 'patient'}_position_limit",
 194:             DEFAULT_BOLD_POSITION_LIMIT if is_bold else DEFAULT_PATIENT_POSITION_LIMIT,
 195:         )
 196:         window_h = cfg.get(
 197:             f"{'bold' if is_bold else 'patient'}_position_window_h",
 198:             DEFAULT_BOLD_POSITION_WINDOW_H if is_bold else DEFAULT_PATIENT_POSITION_WINDOW_H,
 199:         )
 200:         cutoff = now - timedelta(hours=window_h)
 201: 
 202:         timestamps = state.get("new_position_timestamps", {}).get(strategy, [])
 203:         recent = []
 204:         for ts_str in timestamps:
 205:             try:
 206:                 ts = datetime.fromisoformat(ts_str)
 207:                 # M8: ensure aware datetime before comparison with aware cutoff
 208:                 if ts.tzinfo is None:
 209:                     ts = ts.replace(tzinfo=UTC)
 210:                 if ts >= cutoff:
 211:                     recent.append(ts)
 212:             except (ValueError, TypeError):
 213:                 continue
 214: 
 215:         if len(recent) >= limit:
 216:             warnings.append({
 217:                 "guard": "position_rate_limit",
 218:                 "severity": "block",
 219:                 "message": (
 220:                     f"{strategy}: {len(recent)} new position(s) in last {window_h}h "
 221:                     f"(limit: {limit})."
 222:                 ),
 223:                 "details": {
 224:                     "strategy": strategy,
 225:                     "recent_count": len(recent),
 226:                     "limit": limit,
 227:                     "window_hours": window_h,
 228:                 },
 229:             })
 230: 
 231:     return warnings
 232: 
 233: 
 234: _wiring_confirmed = False  # process-scoped flag — positive proof for C4
 235: 
 236: 
 237: def record_trade(ticker, direction, strategy, pnl_pct=None, config=None):
 238:     """Record a completed trade for guard tracking.
 239: 
 240:     Call this after executing a trade to update cooldowns and loss streaks.
 241: 
 242:     Args:
 243:         ticker: Instrument ticker.
 244:         direction: "BUY" or "SELL".
 245:         strategy: "patient" or "bold".
 246:         pnl_pct: Realized P&L percentage (for SELL trades). None for BUY.
 247:         config: Optional config dict.
 248:     """
 249:     # 2026-04-22 follow-up: positive-proof wiring check. The previous C4
 250:     # warning was *reactive* — it could only tell you after a trade had
 251:     # already slipped through unguarded. Log INFO once per process the first
 252:     # time this function fires, so operators get explicit confirmation the
 253:     # BUG-219/PR-R4-4 wiring is alive rather than having to infer it from
 254:     # absence-of-warnings.
 255:     global _wiring_confirmed
 256:     if not _wiring_confirmed:
 257:         logger.info(
 258:             "C4: record_trade() wiring confirmed — first call this process "
 259:             "(ticker=%s direction=%s strategy=%s)",
 260:             ticker, direction, strategy,
 261:         )
 262:         _wiring_confirmed = True
 263: 
 264:     with _state_lock:
 265:         state = _load_state()
 266:         now = datetime.now(UTC)
 267:         now_str = now.isoformat()
 268: 
 269:         # Update ticker trade timestamp
 270:         key = f"{strategy}:{ticker}"
 271:         if "ticker_trades" not in state:
 272:             state["ticker_trades"] = {}
 273:         state["ticker_trades"][key] = now_str
 274: 
 275:         # Update consecutive losses on SELL
 276:         if direction == "SELL" and pnl_pct is not None:
 277:             if "consecutive_losses" not in state:
 278:                 state["consecutive_losses"] = {"patient": 0, "bold": 0}
 279:             if "last_loss_ts" not in state:
 280:                 state["last_loss_ts"] = {"patient": None, "bold": None}
 281:             if pnl_pct < 0:
 282:                 state["consecutive_losses"][strategy] = (
 283:                     state["consecutive_losses"].get(strategy, 0) + 1
 284:                 )
 285:                 state["last_loss_ts"][strategy] = now_str
 286:             else:
 287:                 state["consecutive_losses"][strategy] = 0
 288:                 state["last_loss_ts"][strategy] = None
 289: 
 290:         # Track new position timestamps (BUY only)
 291:         if direction == "BUY":
 292:             if "new_position_timestamps" not in state:
 293:                 state["new_position_timestamps"] = {"patient": [], "bold": []}
 294:             if strategy not in state["new_position_timestamps"]:
 295:                 state["new_position_timestamps"][strategy] = []
 296:             state["new_position_timestamps"][strategy].append(now_str)
 297: 
 298:             # Prune old timestamps (keep last 24h).
 299:             cutoff = now - timedelta(hours=24)
 300:             pruned = []
 301:             for ts in state["new_position_timestamps"][strategy]:
 302:                 try:
 303:                     dt = datetime.fromisoformat(ts)
 304:                     if dt.tzinfo is None:
 305:                         dt = dt.replace(tzinfo=UTC)
 306:                     if dt >= cutoff:
 307:                         pruned.append(ts)
 308:                 except (ValueError, TypeError):
 309:                     continue
 310:             state["new_position_timestamps"][strategy] = pruned
 311: 
 312:         _save_state(state)
 313: 
 314: 
 315: def get_all_guard_warnings(signals, patient_pf, bold_pf, config=None):
 316:     """Get trade guard warnings for all tickers with BUY/SELL signals.
 317: 
 318:     Args:
 319:         signals: Dict of ticker -> signal data (from agent_summary).
 320:         patient_pf: Patient portfolio state dict.
 321:         bold_pf: Bold portfolio state dict.
 322:         config: Optional config dict.
 323: 
 324:     Returns:
 325:         dict with:
 326:             - warnings: list of warning dicts
 327:             - summary: str (human-readable summary)
 328:     """
 329:     cfg = (config or {}).get("trade_guards", {})
 330:     if cfg.get("enabled") is False:
 331:         return {"warnings": [], "summary": "Trade guards disabled"}
 332: 
 333:     all_warnings = []
 334: 
 335:     for ticker, sig in signals.items():
 336:         action = sig.get("action", "HOLD")
 337:         if action == "HOLD":
 338:             continue
 339: 
 340:         for strategy, portfolio in [("patient", patient_pf), ("bold", bold_pf)]:
 341:             warns = check_overtrading_guards(
 342:                 ticker, action, strategy, portfolio, config
 343:             )
 344:             all_warnings.extend(warns)
 345: 
 346:     summary_parts = []
 347:     if all_warnings:
 348:         by_guard = {}
 349:         for w in all_warnings:
 350:             guard = w["guard"]
 351:             by_guard.setdefault(guard, []).append(w)
 352:         for guard, warns in by_guard.items():
 353:             summary_parts.append(f"{guard}: {len(warns)} warning(s)")
 354: 
 355:     # C4: Detect broken record_trade() wiring.
 356:     # 2026-04-22: original check fired every cycle whenever state was empty,
 357:     # even when no trades had happened yet (portfolios untouched) — noisy and
 358:     # misleading post-BUG-219/PR-R4-4 which wired _record_new_trades().
 359:     # Now only warn when portfolios DO have transactions but guard state is
 360:     # still empty — that's the real signal the wiring is broken.
 361:     with _state_lock:
 362:         state = _load_state()
 363:     if not state.get("ticker_trades") and all_warnings == [] and _portfolios_have_transactions():
 364:         logger.warning(
 365:             "C4: portfolios have transactions but trade_guard_state.json "
 366:             "has no recorded trades — record_trade() wiring appears broken. "
 367:             "Overtrading guards are NON-FUNCTIONAL."
 368:         )
 369: 
 370:     return {
 371:         "warnings": all_warnings,
 372:         "summary": "; ".join(summary_parts) if summary_parts else "All clear",
 373:     }
 374: 
 375: 
 376: def should_block_trade(guard_result):
 377:     """Check if any guard warning has 'block' severity.
 378: 
 379:     ARCH-29: Convenience function for Layer 2 go/no-go decisions.
 380: 
 381:     Args:
 382:         guard_result: Return value from get_all_guard_warnings().
 383: 
 384:     Returns:
 385:         True if any warning has severity="block", False otherwise.
 386:     """
 387:     warnings = guard_result.get("warnings", [])
 388:     return any(w.get("severity") == "block" for w in warnings)

### FILE trade_risk_classifier.py
   1: """Trade Risk Classifier — scores proposed trades as LOW / MEDIUM / HIGH risk.
   2: 
   3: Scoring system (0-11 total points):
   4:   Position size risk   (0-3): >20% = 3, >10% = 2, >5% = 1
   5:   Regime risk          (0-3): trending-up=0, trending-down=1, ranging=2, high-vol=3, capitulation=3
   6:   Counter-trend risk   (0-2): BUY in trending-down or SELL in trending-up = 2
   7:   Weak consensus       (0-2): <60% = 2, <70% = 1
   8:   Low confidence       (0-1): <50% = 1
   9:   Concentration risk   (0-2): total exposure >40% = 2, >25% = 1
  10: 
  11: Thresholds: 0-3 = LOW, 4-6 = MEDIUM, 7+ = HIGH.
  12: HOLD is always LOW (score 0, no factors).
  13: """
  14: 
  15: import logging
  16: 
  17: logger = logging.getLogger("portfolio.trade_risk_classifier")
  18: 
  19: # Regime risk mapping
  20: _REGIME_SCORES = {
  21:     "trending-up": 0,
  22:     "trending-down": 1,
  23:     "ranging": 2,
  24:     "high-vol": 3,
  25:     "capitulation": 3,
  26: }
  27: 
  28: 
  29: def classify_trade_risk(
  30:     action: str,
  31:     confidence: float,
  32:     position_pct: float,
  33:     regime: str,
  34:     consensus_ratio: float,
  35:     existing_exposure_pct: float = 0.0,
  36: ) -> dict:
  37:     """Classify a proposed trade into LOW / MEDIUM / HIGH risk.
  38: 
  39:     Parameters
  40:     ----------
  41:     action : str
  42:         Trade action: "BUY", "SELL", or "HOLD".
  43:     confidence : float
  44:         Signal confidence (0.0-1.0).
  45:     position_pct : float
  46:         Proposed position size as percentage of portfolio (0-100).
  47:     regime : str
  48:         Current market regime (trending-up, trending-down, ranging, high-vol, capitulation).
  49:     consensus_ratio : float
  50:         Signal consensus ratio (0.0-1.0). Fraction of voters that agree.
  51:     existing_exposure_pct : float
  52:         Current total portfolio exposure percentage (0-100).
  53: 
  54:     Returns
  55:     -------
  56:     dict
  57:         {"level": "LOW"/"MEDIUM"/"HIGH", "score": int, "factors": list[str]}
  58:     """
  59:     action_upper = action.upper()
  60: 
  61:     # HOLD is always LOW risk
  62:     if action_upper == "HOLD":
  63:         return {"level": "LOW", "score": 0, "factors": []}
  64: 
  65:     score = 0
  66:     factors = []
  67: 
  68:     # 1. Position size risk (0-3)
  69:     if position_pct > 20:
  70:         score += 3
  71:         factors.append(f"large position ({position_pct:.1f}% > 20%)")
  72:     elif position_pct > 10:
  73:         score += 2
  74:         factors.append(f"medium position ({position_pct:.1f}% > 10%)")
  75:     elif position_pct > 5:
  76:         score += 1
  77:         factors.append(f"notable position ({position_pct:.1f}% > 5%)")
  78: 
  79:     # 2. Regime risk (0-3)
  80:     regime_lower = regime.lower()
  81:     regime_score = _REGIME_SCORES.get(regime_lower, 0)
  82:     if regime_score > 0:
  83:         score += regime_score
  84:         factors.append(f"regime={regime_lower} (+{regime_score})")
  85: 
  86:     # 3. Counter-trend risk (0-2)
  87:     if (action_upper == "BUY" and regime_lower == "trending-down") or \
  88:        (action_upper == "SELL" and regime_lower == "trending-up"):
  89:         score += 2
  90:         factors.append(f"counter-trend {action_upper} in {regime_lower}")
  91: 
  92:     # 4. Weak consensus (0-2)
  93:     if consensus_ratio < 0.60:
  94:         score += 2
  95:         factors.append(f"weak consensus ({consensus_ratio:.0%} < 60%)")
  96:     elif consensus_ratio < 0.70:
  97:         score += 1
  98:         factors.append(f"moderate consensus ({consensus_ratio:.0%} < 70%)")
  99: 
 100:     # 5. Low confidence (0-1)
 101:     if confidence < 0.50:
 102:         score += 1
 103:         factors.append(f"low confidence ({confidence:.0%} < 50%)")
 104: 
 105:     # 6. Concentration risk (0-2)
 106:     if existing_exposure_pct > 40:
 107:         score += 2
 108:         factors.append(f"high concentration ({existing_exposure_pct:.1f}% > 40%)")
 109:     elif existing_exposure_pct > 25:
 110:         score += 1
 111:         factors.append(f"moderate concentration ({existing_exposure_pct:.1f}% > 25%)")
 112: 
 113:     # Determine level
 114:     if score <= 3:
 115:         level = "LOW"
 116:     elif score <= 6:
 117:         level = "MEDIUM"
 118:     else:
 119:         level = "HIGH"
 120: 
 121:     return {"level": level, "score": score, "factors": factors}

### FILE trade_validation.py
   1: """Pre-trade validation -- sanity checks before order placement.
   2: 
   3: Validates trade parameters (price, volume, spread, cash, position size) before
   4: any order is placed. Returns a ValidationResult with pass/fail, reason, and
   5: optional warnings for near-limit conditions.
   6: """
   7: 
   8: import logging
   9: from dataclasses import dataclass, field
  10: 
  11: logger = logging.getLogger("portfolio.trade_validation")
  12: 
  13: 
  14: @dataclass
  15: class ValidationResult:
  16:     """Result of pre-trade validation."""
  17:     valid: bool
  18:     reason: str = ""
  19:     warnings: list = field(default_factory=list)
  20: 
  21: 
  22: def validate_trade(
  23:     action: str,
  24:     price: float,
  25:     volume: float,
  26:     cash_available: float,
  27:     bid: float | None = None,
  28:     ask: float | None = None,
  29:     last_known_price: float | None = None,
  30:     max_spread_pct: float = 2.0,
  31:     max_cash_pct: float = 50.0,
  32:     min_order_sek: float = 500.0,
  33:     max_price_deviation_pct: float = 5.0,
  34: ) -> ValidationResult:
  35:     """Validate a trade before placing it.
  36: 
  37:     Checks (in order):
  38:     1. Basic parameter validation (positive price, volume, valid action)
  39:     2. Minimum order size
  40:     3. Cash sufficiency (BUY only)
  41:     4. Position size as % of cash (BUY only)
  42:     5. Bid/ask spread width
  43:     6. Price sanity vs last known price
  44: 
  45:     Returns ValidationResult with valid=True/False, reason, and warnings.
  46:     """
  47:     warnings: list[str] = []
  48: 
  49:     # --- 1. Basic parameter validation ---
  50:     if action not in ("BUY", "SELL"):
  51:         return ValidationResult(False, f"Invalid action: {action}")
  52:     if price <= 0:
  53:         return ValidationResult(False, f"Invalid price: {price}")
  54:     if volume <= 0:
  55:         return ValidationResult(False, f"Invalid volume: {volume}")
  56: 
  57:     order_value = price * volume
  58: 
  59:     # --- 2. Minimum order size ---
  60:     if order_value < min_order_sek:
  61:         return ValidationResult(
  62:             False,
  63:             f"Order value {order_value:.0f} SEK below minimum {min_order_sek:.0f} SEK",
  64:         )
  65: 
  66:     # --- 3 & 4. BUY-specific checks ---
  67:     if action == "BUY":
  68:         # Cash sufficiency
  69:         if order_value > cash_available:
  70:             return ValidationResult(
  71:                 False,
  72:                 f"Insufficient cash: need {order_value:.0f} SEK, have {cash_available:.0f} SEK",
  73:             )
  74:         # Position size limit
  75:         if cash_available > 0:
  76:             cash_pct = (order_value / cash_available) * 100
  77:             if cash_pct > max_cash_pct:
  78:                 return ValidationResult(
  79:                     False,
  80:                     f"Position too large: {cash_pct:.1f}% of cash (max {max_cash_pct:.1f}%)",
  81:                 )
  82: 
  83:     # --- 5. Bid/ask spread check ---
  84:     if bid is not None and ask is not None and bid > 0:
  85:         spread_pct = ((ask - bid) / bid) * 100
  86:         if spread_pct > max_spread_pct:
  87:             return ValidationResult(
  88:                 False,
  89:                 f"Spread too wide: {spread_pct:.2f}% (max {max_spread_pct:.1f}%)",
  90:             )
  91:         if spread_pct > max_spread_pct * 0.7:
  92:             warnings.append(f"Spread warning: {spread_pct:.2f}% approaching limit")
  93: 
  94:     # --- 6. Price sanity vs last known ---
  95:     if last_known_price is not None and last_known_price > 0:
  96:         deviation_pct = abs(price - last_known_price) / last_known_price * 100
  97:         if deviation_pct > max_price_deviation_pct:
  98:             return ValidationResult(
  99:                 False,
 100:                 f"Price deviation {deviation_pct:.2f}% from last known "
 101:                 f"{last_known_price:.2f} (max {max_price_deviation_pct:.1f}%)",
 102:             )
 103:         if deviation_pct > max_price_deviation_pct * 0.7:
 104:             warnings.append(f"Price moved {deviation_pct:.2f}% from last known")
 105: 
 106:     logger.debug(
 107:         "Trade validated: %s %.4f @ %.2f SEK (value %.0f SEK)%s",
 108:         action,
 109:         volume,
 110:         price,
 111:         order_value,
 112:         f" -- warnings: {warnings}" if warnings else "",
 113:     )
 114:     return ValidationResult(True, "All checks passed", warnings)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
mods = [
'portfolio.circuit_breaker','portfolio.cost_model','portfolio.equity_curve','portfolio.kelly_sizing',
'portfolio.monte_carlo','portfolio.monte_carlo_risk','portfolio.portfolio_mgr','portfolio.portfolio_validator',
'portfolio.risk_management','portfolio.trade_guards','portfolio.trade_risk_classifier','portfolio.trade_validation','portfolio.warrant_portfolio']
for m in mods:
    try:
        __import__(m)
        print(m, 'OK')
    except Exception as e:
        print(m, type(e).__name__, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 729ms:
portfolio.circuit_breaker OK
portfolio.cost_model OK
portfolio.equity_curve ModuleNotFoundError No module named 'portfolio.file_utils'
portfolio.kelly_sizing ModuleNotFoundError No module named 'portfolio.file_utils'
portfolio.monte_carlo ModuleNotFoundError No module named 'scipy'
portfolio.monte_carlo_risk ModuleNotFoundError No module named 'scipy'
portfolio.portfolio_mgr ModuleNotFoundError No module named 'portfolio.file_utils'
portfolio.portfolio_validator ModuleNotFoundError No module named 'portfolio.file_utils'
portfolio.risk_management ModuleNotFoundError No module named 'portfolio.file_utils'
portfolio.trade_guards ModuleNotFoundError No module named 'portfolio.file_utils'
portfolio.trade_risk_classifier OK
portfolio.trade_validation OK
portfolio.warrant_portfolio ModuleNotFoundError No module named 'portfolio.file_utils'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'ruff check portfolio' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
2026-05-08T15:29:09.743626Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mruff: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mruff[0m check portfolio[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mThe term 'ruff' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

 exited 1 in 758ms:
[31;1mruff: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mruff[0m check portfolio[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mThe term 'ruff' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
files=['kelly_sizing.py','monte_carlo.py','monte_carlo_risk.py','portfolio_validator.py','trade_guards.py','equity_curve.py']
for name in files:
    text=Path('portfolio',name).read_text().splitlines()
    print(f'\\n===== {name} ({len(text)} lines) =====')
    # print markers every 100 lines
    for start in range(1,len(text)+1,120):
        end=min(start+119,len(text))
        print(f'-- lines {start}-{end} --')
        for i in range(start,end+1):
            print(f\"{i:4}: {text[i-1]}\")
        print()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 861ms:

===== kelly_sizing.py (389 lines) =====
-- lines 1-120 --
   1: """Position sizing using Kelly criterion.
   2: 
   3: Kelly fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
   4: 
   5: Uses signal accuracy as win probability and historical trade outcomes for avg_win/avg_loss.
   6: Returns recommended position size as fraction of portfolio.
   7: """
   8: 
   9: from pathlib import Path
  10: 
  11: from portfolio.file_utils import load_json
  12: 
  13: BASE_DIR = Path(__file__).resolve().parent.parent
  14: DATA_DIR = BASE_DIR / "data"
  15: PATIENT_FILE = DATA_DIR / "portfolio_state.json"
  16: BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
  17: AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
  18: 
  19: 
  20: 
  21: def kelly_fraction(win_prob, avg_win_pct, avg_loss_pct):
  22:     """Compute Kelly criterion fraction.
  23: 
  24:     The Kelly fraction gives the optimal bet size as a fraction of bankroll
  25:     that maximizes long-run geometric growth rate.
  26: 
  27:     Formula: f* = (p * b - q) / b
  28:     where p = win probability, q = 1 - p, b = avg_win / avg_loss
  29: 
  30:     Args:
  31:         win_prob: Probability of winning (0.0 to 1.0).
  32:         avg_win_pct: Average win as a positive percentage (e.g. 2.5 for +2.5%).
  33:         avg_loss_pct: Average loss as a positive percentage (e.g. 1.8 for -1.8%).
  34: 
  35:     Returns:
  36:         float: Kelly fraction (0.0 to 1.0). Clamped to [0, 1].
  37:             Returns 0.0 if inputs are invalid or edge is negative.
  38:     """
  39:     if win_prob <= 0 or win_prob >= 1:
  40:         return 0.0
  41:     if avg_win_pct <= 0 or avg_loss_pct <= 0:
  42:         return 0.0
  43: 
  44:     # b = ratio of avg win to avg loss
  45:     b = avg_win_pct / avg_loss_pct
  46:     q = 1.0 - win_prob
  47: 
  48:     # Kelly formula: f* = (p * b - q) / b
  49:     kelly = (win_prob * b - q) / b
  50: 
  51:     # Clamp to [0, 1] — negative Kelly means negative edge, don't bet
  52:     return max(0.0, min(1.0, kelly))
  53: 
  54: 
  55: def _compute_trade_stats(transactions, ticker=None):
  56:     """Compute win rate and average win/loss from historical transactions.
  57: 
  58:     Pairs BUY and SELL transactions to compute realized P&L per round-trip.
  59: 
  60:     Args:
  61:         transactions: List of transaction dicts from portfolio state.
  62:         ticker: If specified, filter to this ticker only. None = all tickers.
  63: 
  64:     Returns:
  65:         dict: {win_rate, avg_win_pct, avg_loss_pct, total_trades, wins, losses}
  66:               Returns None if insufficient data (fewer than 2 round-trips).
  67:     """
  68:     # Group transactions by ticker
  69:     from collections import defaultdict
  70:     buys_by_ticker = defaultdict(list)
  71:     sells_by_ticker = defaultdict(list)
  72: 
  73:     for t in transactions:
  74:         t_ticker = t.get("ticker", "")
  75:         if ticker and t_ticker != ticker:
  76:             continue
  77:         action = t.get("action", "")
  78:         if action == "BUY":
  79:             buys_by_ticker[t_ticker].append(t)
  80:         elif action == "SELL":
  81:             sells_by_ticker[t_ticker].append(t)
  82: 
  83:     # Compute P&L for each sell vs weighted average buy price
  84:     pnl_list = []
  85:     for t_ticker, sells in sells_by_ticker.items():
  86:         buys = buys_by_ticker.get(t_ticker, [])
  87:         if not buys:
  88:             continue
  89: 
  90:         # Compute weighted average buy price (in SEK per share)
  91:         total_shares_bought = sum(b.get("shares", 0) for b in buys)
  92:         total_cost = sum(b.get("total_sek", 0) for b in buys)
  93:         if total_shares_bought <= 0:
  94:             continue
  95:         avg_buy_price = total_cost / total_shares_bought
  96: 
  97:         for sell in sells:
  98:             sell_shares = sell.get("shares", 0)
  99:             sell_total = sell.get("total_sek", 0)
 100:             if sell_shares <= 0:
 101:                 continue
 102:             sell_price_per_share = sell_total / sell_shares
 103:             pnl_pct = (sell_price_per_share - avg_buy_price) / avg_buy_price * 100
 104:             pnl_list.append(pnl_pct)
 105: 
 106:     if len(pnl_list) < 2:
 107:         return None
 108: 
 109:     wins = [p for p in pnl_list if p > 0]
 110:     losses = [abs(p) for p in pnl_list if p <= 0]
 111: 
 112:     win_rate = len(wins) / len(pnl_list) if pnl_list else 0.0
 113:     avg_win = sum(wins) / len(wins) if wins else 0.0
 114:     avg_loss = sum(losses) / len(losses) if losses else 0.0
 115: 
 116:     return {
 117:         "win_rate": win_rate,
 118:         "avg_win_pct": avg_win,
 119:         "avg_loss_pct": avg_loss,
 120:         "total_trades": len(pnl_list),

-- lines 121-240 --
 121:         "wins": len(wins),
 122:         "losses": len(losses),
 123:     }
 124: 
 125: 
 126: def _get_signal_accuracy(agent_summary, ticker=None):
 127:     """Extract consensus accuracy from agent_summary.
 128: 
 129:     Uses signal_accuracy_1d.consensus.accuracy as the win probability proxy.
 130:     Falls back to weighted_confidence from the ticker's signal data.
 131: 
 132:     Args:
 133:         agent_summary: Parsed agent_summary.json dict.
 134:         ticker: Ticker to get accuracy for (used for weighted_confidence fallback).
 135: 
 136:     Returns:
 137:         float: Estimated win probability (0.0 to 1.0).
 138:     """
 139:     # Primary: use overall consensus accuracy
 140:     acc_data = agent_summary.get("signal_accuracy_1d", {})
 141:     consensus_acc = acc_data.get("consensus", {}).get("accuracy")
 142:     if consensus_acc is not None and consensus_acc > 0:
 143:         return consensus_acc
 144: 
 145:     # Fallback: use weighted_confidence from the ticker's signals
 146:     if ticker:
 147:         signals = agent_summary.get("signals", {})
 148:         ticker_data = signals.get(ticker, {})
 149:         weighted_conf = ticker_data.get("weighted_confidence")
 150:         if weighted_conf is not None:
 151:             return weighted_conf
 152: 
 153:     # Last resort: 50/50
 154:     return 0.5
 155: 
 156: 
 157: def _get_ticker_signal_accuracy(agent_summary, ticker):
 158:     """Get the accuracy of the most relevant signals for a specific ticker.
 159: 
 160:     Computes a weighted average of signal accuracies, weighted by each signal's
 161:     normalized weight. Only considers signals that are actively voting (non-HOLD).
 162: 
 163:     P1-11 (2026-05-02 adversarial follow-up): per-signal accuracy is now
 164:     looked up per-ticker first via `agent_summary["per_ticker_signal_accuracy"]`
 165:     (produced by `accuracy_stats.accuracy_by_ticker_signal_cached()` upstream).
 166:     If a ticker/signal isn't present per-ticker, OR has fewer than 5 samples
 167:     on this ticker, it falls back to the system-wide
 168:     `signal_accuracy_1d.signals` block. Without this, a signal that's 70%
 169:     accurate on XAG-USD but 30% on BTC-USD shows up as ~50% in the
 170:     system-wide aggregate, distorting Kelly sizing on both. The fallback
 171:     keeps backwards compatibility with older agent_summary writers that
 172:     don't yet emit the per-ticker block.
 173: 
 174:     Args:
 175:         agent_summary: Parsed agent_summary.json dict.
 176:         ticker: Ticker symbol.
 177: 
 178:     Returns:
 179:         float: Weighted signal accuracy (0.0 to 1.0), or None if insufficient data.
 180:     """
 181:     acc_data = agent_summary.get("signal_accuracy_1d", {})
 182:     sig_accuracies = acc_data.get("signals", {})
 183:     sig_weights = agent_summary.get("signal_weights", {})
 184: 
 185:     # P1-11: per-ticker per-signal accuracy block (preferred when present).
 186:     per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
 187:     per_ticker_for_ticker = per_ticker_block.get(ticker, {}) or {}
 188: 
 189:     signals = agent_summary.get("signals", {})
 190:     ticker_data = signals.get(ticker, {})
 191:     extra = ticker_data.get("extra", {}) if isinstance(ticker_data, dict) else {}
 192:     votes = extra.get("_votes", {})
 193: 
 194:     if not votes:
 195:         return None
 196:     # Either source can satisfy us — only bail if BOTH are empty.
 197:     if not sig_accuracies and not per_ticker_for_ticker:
 198:         return None
 199: 
 200:     weighted_sum = 0.0
 201:     weight_total = 0.0
 202: 
 203:     for sig_name, vote in votes.items():
 204:         if vote == "HOLD":
 205:             continue
 206: 
 207:         # Prefer per-ticker accuracy when it has enough samples for THIS
 208:         # ticker; otherwise fall back to the system-wide row.
 209:         per_ticker_sig = per_ticker_for_ticker.get(sig_name) or {}
 210:         per_ticker_samples = per_ticker_sig.get("samples", 0)
 211:         if per_ticker_samples >= 5:
 212:             accuracy = per_ticker_sig.get("accuracy", 0.5)
 213:             samples = per_ticker_samples
 214:         else:
 215:             sig_acc = sig_accuracies.get(sig_name, {})
 216:             accuracy = sig_acc.get("accuracy", 0.5)
 217:             samples = sig_acc.get("samples", 0)
 218:             if samples < 5:
 219:                 continue  # unreliable on both axes
 220: 
 221:         weight = sig_weights.get(sig_name, {}).get("normalized_weight", 1.0)
 222:         weighted_sum += accuracy * weight
 223:         weight_total += weight
 224: 
 225:     if weight_total <= 0:
 226:         return None
 227: 
 228:     return weighted_sum / weight_total
 229: 
 230: 
 231: def recommended_size(ticker, portfolio_path=None, agent_summary=None, strategy="patient"):
 232:     """Compute recommended position size using Kelly criterion.
 233: 
 234:     Combines signal accuracy (as win probability) with historical trade
 235:     performance (avg win/loss) to compute optimal position sizing.
 236: 
 237:     Args:
 238:         ticker: Ticker symbol to compute sizing for.
 239:         portfolio_path: Path to portfolio state JSON. If None, uses default
 240:             based on strategy.

-- lines 241-360 --
 241:         agent_summary: Parsed agent_summary dict. If None, loads from file.
 242:         strategy: "patient" or "bold" - determines default portfolio and max alloc.
 243: 
 244:     Returns:
 245:         dict: {
 246:             kelly_pct: Full Kelly fraction (0-1),
 247:             half_kelly_pct: Half Kelly (more conservative),
 248:             quarter_kelly_pct: Quarter Kelly (most conservative),
 249:             recommended_sek: Recommended trade size in SEK (using half Kelly),
 250:             max_alloc_sek: Maximum allocation per strategy rules,
 251:             win_prob: Estimated win probability used,
 252:             avg_win_pct: Average win percentage used,
 253:             avg_loss_pct: Average loss percentage used,
 254:             source: Description of data source used for estimates,
 255:         }
 256:     """
 257:     # Load portfolio
 258:     if portfolio_path is None:
 259:         portfolio_path = BOLD_FILE if strategy == "bold" else PATIENT_FILE
 260:     portfolio = load_json(portfolio_path, default={})
 261:     cash_sek = portfolio.get("cash_sek", 0)
 262:     transactions = portfolio.get("transactions", [])
 263: 
 264:     # Load agent summary
 265:     if agent_summary is None:
 266:         agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
 267: 
 268:     # Max allocation per strategy rules
 269:     alloc_frac = 0.30 if strategy == "bold" else 0.15
 270:     max_alloc = cash_sek * alloc_frac
 271: 
 272:     # Estimate win probability
 273:     # Priority: ticker-specific weighted signal accuracy > consensus accuracy > 50%
 274:     # P1-11 (2026-05-02): _get_ticker_signal_accuracy now prefers
 275:     # per_ticker_signal_accuracy over the system-wide aggregate when a
 276:     # per-ticker block is supplied in agent_summary. Source string
 277:     # advertises which path was used so operators can audit the decision.
 278:     per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
 279:     has_per_ticker = bool(per_ticker_block.get(ticker))
 280:     win_prob = _get_ticker_signal_accuracy(agent_summary, ticker)
 281:     if has_per_ticker:
 282:         source = f"per-ticker weighted signal accuracy for {ticker}"
 283:     else:
 284:         source = f"weighted signal accuracy for {ticker} (system-wide fallback)"
 285: 
 286:     if win_prob is None:
 287:         win_prob = _get_signal_accuracy(agent_summary, ticker)
 288:         source = "consensus accuracy"
 289: 
 290:     # Estimate avg win/loss from historical trades
 291:     trade_stats = _compute_trade_stats(transactions, ticker=ticker)
 292:     if trade_stats is None:
 293:         # Try all tickers if not enough ticker-specific data
 294:         trade_stats = _compute_trade_stats(transactions, ticker=None)
 295: 
 296:     if trade_stats and trade_stats["avg_win_pct"] > 0 and trade_stats["avg_loss_pct"] > 0:
 297:         avg_win = trade_stats["avg_win_pct"]
 298:         avg_loss = trade_stats["avg_loss_pct"]
 299:         source += f" + trade history ({trade_stats['total_trades']} trades)"
 300:     else:
 301:         # Default estimates based on typical crypto/stock moves
 302:         # Use ATR from agent summary if available
 303:         signals = agent_summary.get("signals", {})
 304:         ticker_data = signals.get(ticker, {})
 305:         atr_pct = ticker_data.get("atr_pct", 1.5) if isinstance(ticker_data, dict) else 1.5
 306: 
 307:         # Assume avg win = 1.5x ATR, avg loss = 1x ATR (realistic risk/reward)
 308:         avg_win = atr_pct * 1.5
 309:         avg_loss = atr_pct * 1.0
 310:         source += f" + ATR-based estimates (win={avg_win:.1f}%, loss={avg_loss:.1f}%)"
 311: 
 312:     # Compute Kelly
 313:     full_kelly = kelly_fraction(win_prob, avg_win, avg_loss)
 314:     half_kelly = full_kelly / 2.0
 315:     quarter_kelly = full_kelly / 4.0
 316: 
 317:     # Apply exposure ceiling from market health / exposure coach.
 318:     # Scales position sizes down when the broad market is unhealthy.
 319:     exposure_rec = agent_summary.get("exposure_recommendation", {})
 320:     exposure_ceiling = exposure_rec.get("exposure_ceiling", 1.0)
 321: 
 322:     # Recommended size = half Kelly * cash * exposure_ceiling, capped at max allocation
 323:     rec_sek = min(half_kelly * cash_sek * exposure_ceiling, max_alloc)
 324: 
 325:     # Minimum trade size check
 326:     if rec_sek < 500:
 327:         rec_sek = 0  # Below minimum trade size
 328: 
 329:     return {
 330:         "kelly_pct": round(full_kelly, 4),
 331:         "half_kelly_pct": round(half_kelly, 4),
 332:         "quarter_kelly_pct": round(quarter_kelly, 4),
 333:         "recommended_sek": round(rec_sek, 0),
 334:         "max_alloc_sek": round(max_alloc, 0),
 335:         "exposure_ceiling": round(exposure_ceiling, 2),
 336:         "win_prob": round(win_prob, 4),
 337:         "avg_win_pct": round(avg_win, 2),
 338:         "avg_loss_pct": round(avg_loss, 2),
 339:         "source": source,
 340:     }
 341: 
 342: 
 343: def print_sizing_report(tickers=None, strategy="patient"):
 344:     """Print Kelly sizing recommendations for given tickers.
 345: 
 346:     Args:
 347:         tickers: List of ticker symbols. If None, uses all from agent_summary.
 348:         strategy: "patient" or "bold".
 349:     """
 350:     agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
 351:     if tickers is None:
 352:         tickers = list(agent_summary.get("signals", {}).keys())
 353: 
 354:     if not tickers:
 355:         print("No tickers found in agent_summary.json")
 356:         return
 357: 
 358:     print(f"=== Kelly Sizing Report ({strategy.title()}) ===")
 359:     print()
 360:     print(

-- lines 361-389 --
 361:         f"{'Ticker':<10} {'Kelly%':>7} {'Half-K%':>8} {'Rec SEK':>10} "
 362:         f"{'Max SEK':>10} {'Win Prob':>9}"
 363:     )
 364:     print(
 365:         f"{'------':<10} {'------':>7} {'-------':>8} {'-------':>10} "
 366:         f"{'-------':>10} {'--------':>9}"
 367:     )
 368: 
 369:     for ticker in sorted(tickers):
 370:         rec = recommended_size(ticker, agent_summary=agent_summary, strategy=strategy)
 371:         print(
 372:             f"{ticker:<10} {rec['kelly_pct']*100:>6.1f}% {rec['half_kelly_pct']*100:>7.1f}% "
 373:             f"{rec['recommended_sek']:>10,.0f} {rec['max_alloc_sek']:>10,.0f} "
 374:             f"{rec['win_prob']*100:>8.1f}%"
 375:         )
 376: 
 377:     print()
 378:     print("Note: Recommended size uses Half Kelly (more conservative).")
 379:     print("Full Kelly is optimal but volatile. Quarter Kelly is safest.")
 380: 
 381: 
 382: if __name__ == "__main__":
 383:     import sys
 384: 
 385:     strategy = "bold" if "--bold" in sys.argv else "patient"
 386:     tickers = [a for a in sys.argv[1:] if not a.startswith("--")]
 387:     if not tickers:
 388:         tickers = None
 389:     print_sizing_report(tickers=tickers, strategy=strategy)


===== monte_carlo.py (422 lines) =====
-- lines 1-120 --
   1: """Monte Carlo price simulation engine.
   2: 
   3: Generates stochastic price paths using Geometric Brownian Motion (GBM)
   4: with antithetic variates for variance reduction. Converts directional
   5: signal probabilities into price quantile bands, stop-loss probabilities,
   6: and expected return distributions.
   7: 
   8: Short-term focus: 3h, 1d, 3d horizons for active trading decisions.
   9: 
  10: Usage:
  11:     from portfolio.monte_carlo import simulate_ticker
  12:     result = simulate_ticker("BTC-USD", agent_summary)
  13:     # result = {
  14:     #   "price_bands_1d": {"p5": 82100, "p25": 84300, "p50": 86200, ...},
  15:     #   "p_stop_hit_1d": 0.12,
  16:     #   "expected_return_1d": {"mean_pct": 0.3, "std_pct": 2.1, "skew": 0.1},
  17:     #   ...
  18:     # }
  19: """
  20: 
  21: import logging
  22: import math
  23: 
  24: import numpy as np
  25: from scipy.stats import norm
  26: 
  27: logger = logging.getLogger("portfolio.monte_carlo")
  28: 
  29: # Default parameters
  30: DEFAULT_N_PATHS = 10_000   # 5K pairs with antithetic variates
  31: DEFAULT_HORIZONS = [1, 3]  # days
  32: MIN_VOLATILITY = 0.05      # 5% annualized floor (prevents degenerate sims)
  33: 
  34: # Per-asset-class ATR fallbacks when actual ATR is missing from signals.
  35: # The generic 2.0% underestimates tail risk for crypto/metals.
  36: _ATR_DEFAULT_BY_CLASS = {
  37:     "crypto": 3.5,  # BTC 3-4% typical daily ATR
  38:     "metals": 4.0,  # XAG 4-6%, XAU 2-3% — biased toward silver
  39:     "stocks": 2.0,  # MSTR ~3% but broader equities lower
  40: }
  41: 
  42: 
  43: # ---------------------------------------------------------------------------
  44: # Volatility & drift estimation from existing system data
  45: # ---------------------------------------------------------------------------
  46: 
  47: def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
  48:     """Convert ATR% (14-period) to annualized volatility.
  49: 
  50:     ATR measures average true range over `period` candles. For hourly candles
  51:     (our primary timeframe), we annualize: vol = atr_frac * sqrt(trading_periods/period).
  52: 
  53:     Uses 252 trading days (standard for stocks/crypto).
  54: 
  55:     Args:
  56:         atr_pct: ATR as percentage of price (e.g., 3.5 means 3.5%).
  57:         period: ATR lookback period (default 14).
  58: 
  59:     Returns:
  60:         Annualized volatility as a decimal (e.g., 0.20 = 20%).
  61:     """
  62:     atr_frac = atr_pct / 100.0
  63:     annual_factor = math.sqrt(252.0 / period)
  64:     vol = atr_frac * annual_factor
  65:     return max(vol, MIN_VOLATILITY)
  66: 
  67: 
  68: def drift_from_probability(p_up: float, volatility: float) -> float:
  69:     """Convert directional probability P(up) into annualized drift.
  70: 
  71:     Uses the inverse of the GBM CDF relationship:
  72:         P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / (sigma*sqrt(T)))
  73: 
  74:     For 1-day horizon (T = 1/252):
  75:         mu = sigma * N_inv(p_up) * sqrt(252) + 0.5 * sigma^2
  76: 
  77:     This ensures the GBM simulation produces paths where the fraction
  78:     ending above spot matches the input probability.
  79: 
  80:     Args:
  81:         p_up: Probability of price being higher at horizon (0.0-1.0).
  82:         volatility: Annualized volatility (decimal).
  83: 
  84:     Returns:
  85:         Annualized drift (decimal). Positive = upward bias.
  86:     """
  87:     # Clamp p_up to avoid infinite drift at extremes
  88:     p_up = max(0.01, min(0.99, p_up))
  89: 
  90:     # N_inv(p_up) gives the z-score for the desired probability
  91:     z = norm.ppf(p_up)
  92: 
  93:     # mu = sigma * z * sqrt(252) + 0.5 * sigma^2
  94:     # This is derived from P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / sigma*sqrt(T))
  95:     mu = volatility * z * math.sqrt(252.0) + 0.5 * volatility**2
  96: 
  97:     return mu
  98: 
  99: 
 100: def _atr_default_for_ticker(ticker: str) -> float:
 101:     """Return the per-asset-class ATR fallback for a ticker."""
 102:     from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS
 103:     if ticker in CRYPTO_SYMBOLS:
 104:         return _ATR_DEFAULT_BY_CLASS["crypto"]
 105:     if ticker in METALS_SYMBOLS:
 106:         return _ATR_DEFAULT_BY_CLASS["metals"]
 107:     return _ATR_DEFAULT_BY_CLASS["stocks"]
 108: 
 109: 
 110: # ---------------------------------------------------------------------------
 111: # Core simulation engine
 112: # ---------------------------------------------------------------------------
 113: 
 114: class MonteCarloEngine:
 115:     """Geometric Brownian Motion price path simulator with antithetic variates.
 116: 
 117:     Generates terminal price distributions for short-term horizons (hours to days).
 118:     Uses antithetic variates for 50-75% variance reduction at zero extra cost.
 119: 
 120:     Args:

-- lines 121-240 --
 121:         price: Current asset price (USD).
 122:         volatility: Annualized volatility (decimal, e.g., 0.20 = 20%).
 123:         drift: Annualized drift (decimal). Use drift_from_probability() to derive.
 124:         horizon_days: Simulation horizon in days (supports fractional, e.g., 0.125 = 3h).
 125:         n_paths: Number of simulated terminal prices to generate.
 126:         seed: Random seed for reproducibility.
 127:     """
 128: 
 129:     def __init__(self, price: float, volatility: float, drift: float = 0.0,
 130:                  horizon_days: float = 1.0, n_paths: int = DEFAULT_N_PATHS,
 131:                  seed: int | None = None):
 132:         self.price = price
 133:         self.volatility = max(volatility, MIN_VOLATILITY)
 134:         self.drift = drift
 135:         self.horizon_days = horizon_days
 136:         self.n_paths = n_paths
 137:         self.seed = seed
 138:         self._terminal_prices = None
 139: 
 140:     def simulate_paths(self) -> np.ndarray:
 141:         """Generate terminal prices via GBM with antithetic variates.
 142: 
 143:         Formula: S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
 144: 
 145:         Antithetic variates: for each random draw Z, we also compute the
 146:         path with -Z. This creates negative correlation between paired
 147:         estimates, reducing variance of the mean estimator.
 148: 
 149:         Returns:
 150:             1D array of terminal prices, shape (n_paths,).
 151:         """
 152:         rng = np.random.default_rng(self.seed)
 153: 
 154:         T = self.horizon_days / 252.0  # Convert to years
 155:         sigma = self.volatility
 156:         mu = self.drift
 157: 
 158:         # Draw half the paths, use antithetic for the other half
 159:         n_half = self.n_paths // 2
 160:         Z = rng.standard_normal(n_half)
 161: 
 162:         # Deterministic component
 163:         drift_term = (mu - 0.5 * sigma**2) * T
 164:         vol_term = sigma * math.sqrt(T)
 165: 
 166:         # Original paths
 167:         log_returns_pos = drift_term + vol_term * Z
 168:         # Antithetic paths (negate Z)
 169:         log_returns_neg = drift_term + vol_term * (-Z)
 170: 
 171:         # Combine
 172:         log_returns = np.concatenate([log_returns_pos, log_returns_neg])
 173:         terminal_prices = self.price * np.exp(log_returns)
 174: 
 175:         # If n_paths is odd, add one more path
 176:         if self.n_paths % 2 == 1:
 177:             extra_Z = rng.standard_normal(1)
 178:             extra_price = self.price * np.exp(drift_term + vol_term * extra_Z)
 179:             terminal_prices = np.concatenate([terminal_prices, extra_price])
 180: 
 181:         self._terminal_prices = terminal_prices
 182:         return terminal_prices
 183: 
 184:     def _ensure_simulated(self):
 185:         """Run simulation if not already done."""
 186:         if self._terminal_prices is None:
 187:             self.simulate_paths()
 188: 
 189:     def price_quantiles(self, percentiles: list[int] | None = None) -> dict:
 190:         """Extract price quantile bands from simulated distribution.
 191: 
 192:         Args:
 193:             percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95]).
 194: 
 195:         Returns:
 196:             Dict mapping percentile → price (e.g., {5: 87.2, 50: 90.1, 95: 93.5}).
 197:         """
 198:         self._ensure_simulated()
 199:         if percentiles is None:
 200:             percentiles = [5, 25, 50, 75, 95]
 201: 
 202:         values = np.percentile(self._terminal_prices, percentiles)
 203:         return {p: round(float(v), 2) for p, v in zip(percentiles, values)}
 204: 
 205:     def probability_below(self, threshold: float) -> float:
 206:         """Compute P(terminal price < threshold).
 207: 
 208:         Useful for stop-loss probability: "what's the chance price hits my stop?"
 209: 
 210:         Args:
 211:             threshold: Price level to test.
 212: 
 213:         Returns:
 214:             Probability (0.0-1.0).
 215:         """
 216:         self._ensure_simulated()
 217:         if threshold <= 0:
 218:             return 0.0
 219:         return float(np.mean(self._terminal_prices < threshold))
 220: 
 221:     def probability_above(self, threshold: float) -> float:
 222:         """Compute P(terminal price > threshold).
 223: 
 224:         Useful for profit target probability.
 225: 
 226:         Args:
 227:             threshold: Price level to test.
 228: 
 229:         Returns:
 230:             Probability (0.0-1.0).
 231:         """
 232:         self._ensure_simulated()
 233:         if threshold <= 0:
 234:             return 1.0
 235:         return float(np.mean(self._terminal_prices > threshold))
 236: 
 237:     def expected_return(self) -> dict:
 238:         """Compute return distribution statistics.
 239: 
 240:         Returns:

-- lines 241-360 --
 241:             Dict with mean_pct, std_pct, skew of the return distribution.
 242:         """
 243:         self._ensure_simulated()
 244:         returns = (self._terminal_prices - self.price) / self.price * 100.0
 245: 
 246:         mean_pct = float(np.mean(returns))
 247:         std_pct = float(np.std(returns))
 248: 
 249:         # Skewness
 250:         if std_pct > 0:
 251:             skew = float(np.mean(((returns - mean_pct) / std_pct) ** 3))
 252:         else:
 253:             skew = 0.0
 254: 
 255:         return {
 256:             "mean_pct": round(mean_pct, 2),
 257:             "std_pct": round(std_pct, 2),
 258:             "skew": round(skew, 2),
 259:         }
 260: 
 261: 
 262: # ---------------------------------------------------------------------------
 263: # Convenience: simulate a single ticker from agent_summary data
 264: # ---------------------------------------------------------------------------
 265: 
 266: def simulate_ticker(ticker: str, agent_summary: dict,
 267:                     n_paths: int = DEFAULT_N_PATHS,
 268:                     horizons: list[int] | None = None,
 269:                     seed: int | None = None) -> dict | None:
 270:     """Simulate price distribution for a ticker using agent_summary data.
 271: 
 272:     Extracts price, ATR volatility, and directional probability from the
 273:     signal data, then runs GBM simulation at each horizon.
 274: 
 275:     Args:
 276:         ticker: Instrument ticker (e.g., "BTC-USD").
 277:         agent_summary: Full agent_summary dict (or compact version).
 278:         n_paths: Number of MC paths per horizon.
 279:         horizons: List of horizon days (default: [1, 3]).
 280:         seed: Random seed.
 281: 
 282:     Returns:
 283:         Dict with price bands, stop probability, expected return per horizon.
 284:         None if ticker not found in summary.
 285:     """
 286:     if horizons is None:
 287:         horizons = DEFAULT_HORIZONS
 288: 
 289:     signals = agent_summary.get("signals", {})
 290:     ticker_data = signals.get(ticker)
 291:     if not ticker_data:
 292:         return None
 293: 
 294:     price = ticker_data.get("price_usd", 0)
 295:     if price <= 0:
 296:         return None
 297: 
 298:     extra = ticker_data.get("extra", {})
 299:     atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct") or _atr_default_for_ticker(ticker)
 300: 
 301:     # Get volatility from ATR
 302:     vol = volatility_from_atr(atr_pct)
 303: 
 304:     # Get directional probability for drift
 305:     # Try to use existing probability computation if available
 306:     p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
 307:     drift = drift_from_probability(p_up, vol)
 308: 
 309:     # Compute ATR-based stop level (2x ATR below entry/current)
 310:     stop_price = price * (1 - 2 * atr_pct / 100)
 311: 
 312:     result = {}
 313: 
 314:     for h in horizons:
 315:         h_key = f"{h}d" if h >= 1 else f"{int(h * 24)}h"
 316: 
 317:         mc = MonteCarloEngine(
 318:             price=price,
 319:             volatility=vol,
 320:             drift=drift,
 321:             horizon_days=h,
 322:             n_paths=n_paths,
 323:             seed=seed,
 324:         )
 325:         mc.simulate_paths()
 326: 
 327:         result[f"price_bands_{h_key}"] = mc.price_quantiles()
 328:         result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)
 329:         result[f"expected_return_{h_key}"] = mc.expected_return()
 330: 
 331:     # Add metadata
 332:     result["price_usd"] = price
 333:     result["atr_pct"] = round(atr_pct, 2)
 334:     result["volatility_annual"] = round(vol, 3)
 335:     result["p_up"] = round(p_up, 3)
 336:     result["drift_annual"] = round(drift, 4)
 337:     result["stop_price"] = round(stop_price, 2)
 338: 
 339:     return result
 340: 
 341: 
 342: def _get_directional_probability(ticker: str, ticker_data: dict,
 343:                                   agent_summary: dict) -> float:
 344:     """Extract directional probability from available data.
 345: 
 346:     Priority:
 347:     1. Pre-computed focus_probabilities (from ticker_accuracy.py)
 348:     2. Weighted confidence + action from signal engine
 349:     3. Fallback: 0.5 (no edge)
 350:     """
 351:     # 1. Check focus_probabilities in summary
 352:     focus_probs = agent_summary.get("focus_probabilities", {})
 353:     ticker_probs = focus_probs.get(ticker, {})
 354:     prob_1d = ticker_probs.get("1d", {})
 355:     if prob_1d and "probability" in prob_1d:
 356:         return prob_1d["probability"]
 357: 
 358:     # 2. Derive from weighted confidence + action
 359:     extra = ticker_data.get("extra", {})
 360:     action = extra.get("_weighted_action") or ticker_data.get("action", "HOLD")

-- lines 361-422 --
 361:     conf = extra.get("_weighted_confidence") or ticker_data.get("weighted_confidence", 0.5)
 362: 
 363:     if action == "BUY":
 364:         return 0.5 + conf * 0.3  # Scale to 0.5-0.8 range
 365:     elif action == "SELL":
 366:         return 0.5 - conf * 0.3  # Scale to 0.2-0.5 range
 367:     else:
 368:         return 0.5  # HOLD = no directional edge
 369: 
 370: 
 371: # ---------------------------------------------------------------------------
 372: # Batch simulation for all interesting tickers
 373: # ---------------------------------------------------------------------------
 374: 
 375: def simulate_all(agent_summary: dict, tickers: list[str] | None = None,
 376:                  n_paths: int = DEFAULT_N_PATHS, seed: int | None = None) -> dict:
 377:     """Run MC simulation for multiple tickers.
 378: 
 379:     Args:
 380:         agent_summary: Full agent summary dict.
 381:         tickers: List of tickers to simulate. If None, uses held positions + focus tickers.
 382:         n_paths: Paths per ticker per horizon.
 383:         seed: Base seed (incremented per ticker for independence).
 384: 
 385:     Returns:
 386:         Dict mapping ticker → simulation results.
 387:     """
 388:     if tickers is None:
 389:         tickers = _interesting_tickers(agent_summary)
 390: 
 391:     results = {}
 392:     for i, ticker in enumerate(tickers):
 393:         try:
 394:             # 2026-04-22: guard seed=None. reporting.py calls simulate_all
 395:             # without a seed; `seed + i` was crashing every cycle with
 396:             # TypeError and silently disabling MC risk sim for all tickers.
 397:             ticker_seed = None if seed is None else seed + i
 398:             result = simulate_ticker(ticker, agent_summary,
 399:                                      n_paths=n_paths, seed=ticker_seed)
 400:             if result:
 401:                 results[ticker] = result
 402:         except Exception:
 403:             logger.warning("MC simulation failed for %s", ticker, exc_info=True)
 404: 
 405:     return results
 406: 
 407: 
 408: def _interesting_tickers(agent_summary: dict) -> list[str]:
 409:     """Determine which tickers to simulate (held + focus + signaling)."""
 410:     tickers = set()
 411: 
 412:     # Focus tickers (from config, surfaced in summary)
 413:     for ft in agent_summary.get("focus_tickers", ["XAG-USD", "BTC-USD"]):
 414:         tickers.add(ft)
 415: 
 416:     # Tickers with active consensus (BUY or SELL)
 417:     for ticker, data in agent_summary.get("signals", {}).items():
 418:         action = data.get("action", "HOLD")
 419:         if action in ("BUY", "SELL"):
 420:             tickers.add(ticker)
 421: 
 422:     return sorted(tickers)


===== monte_carlo_risk.py (515 lines) =====
-- lines 1-120 --
   1: """Portfolio VaR with t-copula correlated simulation.
   2: 
   3: Computes Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and
   4: joint drawdown probabilities for multi-position portfolios using a Student-t
   5: copula for tail dependence.
   6: 
   7: The t-copula captures the empirical fact that assets crash together more
   8: often than a Gaussian copula predicts (tail dependence lambda ~0.18 at
   9: df=4 vs. 0 for Gaussian).
  10: 
  11: Usage:
  12:     from portfolio.monte_carlo_risk import compute_portfolio_var
  13:     result = compute_portfolio_var(portfolio_state, agent_summary)
  14:     # result = {
  15:     #   "var_95_usd": -1234.56,
  16:     #   "cvar_95_usd": -1567.89,
  17:     #   "var_99_usd": -2345.67,
  18:     #   "total_exposure_usd": 50000.0,
  19:     #   ...
  20:     # }
  21: """
  22: 
  23: import logging
  24: import math
  25: 
  26: import numpy as np
  27: from scipy.stats import norm
  28: from scipy.stats import t as t_dist
  29: 
  30: from portfolio.monte_carlo import (
  31:     MIN_VOLATILITY,
  32:     drift_from_probability,
  33:     volatility_from_atr,
  34: )
  35: 
  36: logger = logging.getLogger("portfolio.monte_carlo_risk")
  37: 
  38: DEFAULT_DF = 4        # Degrees of freedom for t-copula (4 = moderate fat tails)
  39: DEFAULT_N_PATHS = 10_000
  40: 
  41: 
  42: # ---------------------------------------------------------------------------
  43: # Correlation matrix estimation
  44: # ---------------------------------------------------------------------------
  45: 
  46: def estimate_correlation_matrix(
  47:     returns: dict[str, np.ndarray],
  48:     tickers: list[str],
  49: ) -> np.ndarray:
  50:     """Estimate correlation matrix from historical return series.
  51: 
  52:     For tickers with insufficient data, defaults to zero correlation
  53:     (independent assumption) which is conservative for VaR.
  54: 
  55:     Args:
  56:         returns: Dict mapping ticker → array of log-returns.
  57:         tickers: Ordered list of tickers (defines matrix row/column order).
  58: 
  59:     Returns:
  60:         Correlation matrix, shape (n, n), guaranteed symmetric and PSD.
  61:     """
  62:     n = len(tickers)
  63:     if n == 0:
  64:         return np.array([]).reshape(0, 0)
  65: 
  66:     corr = np.eye(n)
  67: 
  68:     # Build return matrix for tickers with data
  69:     for i in range(n):
  70:         for j in range(i + 1, n):
  71:             ri = returns.get(tickers[i])
  72:             rj = returns.get(tickers[j])
  73:             if ri is not None and rj is not None:
  74:                 # Align lengths (use shorter)
  75:                 min_len = min(len(ri), len(rj))
  76:                 if min_len >= 20:  # Need at least 20 observations
  77:                     c = np.corrcoef(ri[:min_len], rj[:min_len])[0, 1]
  78:                     if np.isfinite(c):
  79:                         corr[i, j] = c
  80:                         corr[j, i] = c
  81: 
  82:     # Ensure PSD via eigenvalue clipping (Higham nearest PSD)
  83:     corr = _nearest_psd(corr)
  84:     return corr
  85: 
  86: 
  87: def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
  88:     """Project matrix to nearest positive semi-definite correlation matrix.
  89: 
  90:     Uses eigenvalue clipping: set negative eigenvalues to a small positive
  91:     value, then rescale diagonal to 1.
  92:     """
  93:     eigenvalues, eigenvectors = np.linalg.eigh(matrix)
  94:     # Clip negative eigenvalues
  95:     eigenvalues = np.maximum(eigenvalues, 1e-8)
  96:     # Reconstruct
  97:     result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
  98:     # Rescale to correlation matrix (diagonal = 1)
  99:     d = np.sqrt(np.diag(result))
 100:     d[d == 0] = 1.0
 101:     result = result / np.outer(d, d)
 102:     # Enforce exact symmetry
 103:     result = (result + result.T) / 2
 104:     np.fill_diagonal(result, 1.0)
 105:     return result
 106: 
 107: 
 108: # ---------------------------------------------------------------------------
 109: # Known correlation pairs (fallback when no historical data)
 110: # ---------------------------------------------------------------------------
 111: 
 112: # From risk_management.py — approximate correlation strengths
 113: CORRELATION_PRIORS = {
 114:     ("BTC-USD", "ETH-USD"): 0.75,
 115:     ("XAG-USD", "XAU-USD"): 0.85,
 116:     ("NVDA", "AMD"): 0.70,
 117:     ("NVDA", "AVGO"): 0.65,
 118:     ("NVDA", "TSM"): 0.60,
 119:     ("AMD", "AVGO"): 0.60,
 120:     ("AMD", "TSM"): 0.55,

-- lines 121-240 --
 121:     ("AVGO", "TSM"): 0.55,
 122:     ("GOOGL", "META"): 0.65,
 123:     ("GOOGL", "AMZN"): 0.60,
 124:     ("META", "AMZN"): 0.60,
 125:     ("AAPL", "GOOGL"): 0.55,
 126:     ("AAPL", "META"): 0.50,
 127:     ("AAPL", "AMZN"): 0.55,
 128: }
 129: 
 130: 
 131: def _get_prior_correlation(ticker_a: str, ticker_b: str) -> float:
 132:     """Look up prior correlation for a pair (order-independent)."""
 133:     return CORRELATION_PRIORS.get(
 134:         (ticker_a, ticker_b),
 135:         CORRELATION_PRIORS.get((ticker_b, ticker_a), 0.0),
 136:     )
 137: 
 138: 
 139: def build_correlation_matrix(
 140:     tickers: list[str],
 141:     historical_returns: dict[str, np.ndarray] | None = None,
 142: ) -> np.ndarray:
 143:     """Build correlation matrix using historical data with prior fallback.
 144: 
 145:     If historical returns are available and sufficient, uses empirical
 146:     estimates. Otherwise falls back to hardcoded priors from CORRELATED_PAIRS.
 147: 
 148:     Args:
 149:         tickers: Ordered list of tickers.
 150:         historical_returns: Optional dict of ticker → log-return arrays.
 151: 
 152:     Returns:
 153:         Correlation matrix (n x n).
 154:     """
 155:     n = len(tickers)
 156:     if n == 0:
 157:         return np.array([]).reshape(0, 0)
 158: 
 159:     if historical_returns:
 160:         # Check if we have sufficient data (>= 30 observations per ticker)
 161:         has_data = sum(
 162:             1 for t in tickers
 163:             if t in historical_returns and len(historical_returns[t]) >= 30
 164:         )
 165:         if has_data >= 2:
 166:             return estimate_correlation_matrix(historical_returns, tickers)
 167: 
 168:     # Fallback: use priors
 169:     corr = np.eye(n)
 170:     for i in range(n):
 171:         for j in range(i + 1, n):
 172:             c = _get_prior_correlation(tickers[i], tickers[j])
 173:             corr[i, j] = c
 174:             corr[j, i] = c
 175: 
 176:     return _nearest_psd(corr)
 177: 
 178: 
 179: # ---------------------------------------------------------------------------
 180: # Portfolio risk simulator
 181: # ---------------------------------------------------------------------------
 182: 
 183: class PortfolioRiskSimulator:
 184:     """Multi-position portfolio risk simulator using Student-t copula.
 185: 
 186:     Generates correlated terminal returns for all positions jointly,
 187:     then computes portfolio-level P&L, VaR, and CVaR.
 188: 
 189:     The t-copula with df=4 captures tail dependence (assets crash together)
 190:     that a Gaussian copula misses entirely.
 191: 
 192:     Args:
 193:         positions: Dict mapping ticker → {shares, price_usd, volatility, drift}.
 194:         correlation_matrix: Correlation matrix for position tickers.
 195:         horizon_days: Simulation horizon.
 196:         n_paths: Number of simulation paths.
 197:         df: Degrees of freedom for t-copula (lower = fatter tails).
 198:         seed: Random seed.
 199:     """
 200: 
 201:     def __init__(
 202:         self,
 203:         positions: dict,
 204:         correlation_matrix: np.ndarray,
 205:         horizon_days: float = 1.0,
 206:         n_paths: int = DEFAULT_N_PATHS,
 207:         df: int = DEFAULT_DF,
 208:         seed: int | None = None,
 209:     ):
 210:         # Filter out zero-share positions
 211:         self._tickers = [t for t, p in positions.items() if p.get("shares", 0) != 0]
 212:         self._positions = {t: positions[t] for t in self._tickers}
 213:         self._n_assets = len(self._tickers)
 214: 
 215:         # If we filtered tickers, extract the sub-matrix
 216:         if self._n_assets > 0 and correlation_matrix.size > 0:
 217:             all_tickers = list(positions.keys())
 218:             indices = [all_tickers.index(t) for t in self._tickers]
 219:             self._corr = correlation_matrix[np.ix_(indices, indices)]
 220:         else:
 221:             self._corr = np.array([]).reshape(0, 0)
 222: 
 223:         self.horizon_days = horizon_days
 224:         self.n_paths = n_paths
 225:         self.df = df
 226:         self.seed = seed
 227: 
 228:         self._returns = None  # (n_paths, n_assets) log-returns
 229:         self._pnl = None      # (n_paths,) portfolio P&L in USD
 230: 
 231:     def simulate_correlated_returns(self) -> np.ndarray:
 232:         """Generate correlated returns using t-copula.
 233: 
 234:         Algorithm:
 235:         1. Cholesky decompose correlation matrix: L = chol(R)
 236:         2. Generate independent standard normals: Z ~ N(0, I)
 237:         3. Correlate: W = Z @ L^T
 238:         4. Generate chi-squared: S ~ chi2(df)
 239:         5. Scale to t: T = W * sqrt(df / S) → multivariate t
 240:         6. Transform marginals: apply per-asset drift and volatility

-- lines 241-360 --
 241: 
 242:         Returns:
 243:             Array of shape (n_paths, n_assets) with log-returns.
 244:         """
 245:         if self._n_assets == 0:
 246:             self._returns = np.empty((self.n_paths, 0))
 247:             return self._returns
 248: 
 249:         rng = np.random.default_rng(self.seed)
 250:         T = self.horizon_days / 252.0
 251: 
 252:         # Step 1: Cholesky decomposition
 253:         try:
 254:             L = np.linalg.cholesky(self._corr)
 255:         except np.linalg.LinAlgError:
 256:             logger.warning(
 257:                 "Cholesky decomposition failed for %d-asset correlation matrix — "
 258:                 "falling back to nearest PSD projection",
 259:                 self._n_assets,
 260:             )
 261:             L = np.linalg.cholesky(_nearest_psd(self._corr))
 262: 
 263:         # Step 2: Independent standard normals
 264:         Z = rng.standard_normal((self.n_paths, self._n_assets))
 265: 
 266:         # Step 3: Correlate
 267:         W = Z @ L.T
 268: 
 269:         # Step 4: Chi-squared scaling for t-distribution
 270:         S = rng.chisquare(self.df, size=self.n_paths)
 271: 
 272:         # Step 5: Scale to multivariate t
 273:         # T_i = W_i * sqrt(df / S) for each path
 274:         scale = np.sqrt(self.df / S)[:, np.newaxis]
 275:         T_samples = W * scale
 276: 
 277:         # Step 6: Transform to uniform via t CDF, then to Gaussian marginals
 278:         # U = F_t(T; df) → uniform on [0,1]
 279:         U = t_dist.cdf(T_samples, df=self.df)
 280: 
 281:         # Transform each marginal to GBM log-return
 282:         # C9 FIX: Use norm.ppf (Gaussian inverse CDF), NOT t_dist.ppf.
 283:         # Using t_dist.ppf(t_dist.cdf(x, df), df) is an identity transform —
 284:         # it round-trips back to the same t-distributed samples, inflating
 285:         # variance by ~sqrt(df/(df-2)) ≈ sqrt(2) at df=4. The correct
 286:         # t-copula + GBM approach is: t-copula for dependence structure,
 287:         # Gaussian marginals for the GBM diffusion term.
 288:         returns = np.empty_like(U)
 289:         for i, ticker in enumerate(self._tickers):
 290:             pos = self._positions[ticker]
 291:             sigma = max(pos["volatility"], MIN_VOLATILITY)
 292:             mu = pos.get("drift", 0.0)
 293: 
 294:             # Inverse normal CDF to get standard normal quantiles
 295:             Z_marginal = norm.ppf(U[:, i])
 296: 
 297:             # GBM log-return: (mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z
 298:             drift_term = (mu - 0.5 * sigma**2) * T
 299:             vol_term = sigma * math.sqrt(T)
 300:             returns[:, i] = drift_term + vol_term * Z_marginal
 301: 
 302:         self._returns = returns
 303:         return returns
 304: 
 305:     def _ensure_simulated(self):
 306:         """Run simulation if not already done."""
 307:         if self._returns is None:
 308:             self.simulate_correlated_returns()
 309: 
 310:     def portfolio_pnl(self) -> np.ndarray:
 311:         """Compute portfolio P&L distribution in USD.
 312: 
 313:         P&L = sum over assets of (shares * price * (exp(log_return) - 1))
 314: 
 315:         Returns:
 316:             1D array of portfolio P&L values, shape (n_paths,).
 317:         """
 318:         self._ensure_simulated()
 319: 
 320:         if self._n_assets == 0:
 321:             self._pnl = np.zeros(self.n_paths)
 322:             return self._pnl
 323: 
 324:         pnl = np.zeros(self.n_paths)
 325:         for i, ticker in enumerate(self._tickers):
 326:             pos = self._positions[ticker]
 327:             shares = pos["shares"]
 328:             price = pos["price_usd"]
 329:             # P&L per path = shares * price * (exp(return) - 1)
 330:             pnl += shares * price * (np.exp(self._returns[:, i]) - 1)
 331: 
 332:         self._pnl = pnl
 333:         return pnl
 334: 
 335:     def _ensure_pnl(self):
 336:         """Compute P&L if not already done."""
 337:         if self._pnl is None:
 338:             self.portfolio_pnl()
 339: 
 340:     def var(self, confidence: float = 0.95) -> float:
 341:         """Compute Value-at-Risk at given confidence level.
 342: 
 343:         VaR is the loss at the (1-confidence) percentile of the P&L distribution.
 344:         Returns a negative number (loss).
 345: 
 346:         Args:
 347:             confidence: Confidence level (0.95 or 0.99 typically).
 348: 
 349:         Returns:
 350:             VaR in USD (negative = loss).
 351:         """
 352:         self._ensure_pnl()
 353:         if self._n_assets == 0:
 354:             return 0.0
 355:         percentile = (1 - confidence) * 100
 356:         return float(np.percentile(self._pnl, percentile))
 357: 
 358:     def cvar(self, confidence: float = 0.95) -> float:
 359:         """Compute Conditional VaR (Expected Shortfall) at given confidence.
 360: 

-- lines 361-480 --
 361:         CVaR is the average loss in the worst (1-confidence) scenarios.
 362:         Always worse than VaR (further from zero).
 363: 
 364:         Args:
 365:             confidence: Confidence level.
 366: 
 367:         Returns:
 368:             CVaR in USD (negative = loss).
 369:         """
 370:         self._ensure_pnl()
 371:         if self._n_assets == 0:
 372:             return 0.0
 373:         var_threshold = self.var(confidence)
 374:         tail = self._pnl[self._pnl <= var_threshold]
 375:         if len(tail) == 0:
 376:             return var_threshold
 377:         return float(np.mean(tail))
 378: 
 379:     def drawdown_probability(self, threshold_pct: float) -> float:
 380:         """Probability of portfolio losing more than threshold_pct.
 381: 
 382:         Args:
 383:             threshold_pct: Loss threshold as percentage (e.g., 5.0 = 5%).
 384: 
 385:         Returns:
 386:             Probability (0.0-1.0) of exceeding the loss threshold.
 387:         """
 388:         self._ensure_pnl()
 389:         if self._n_assets == 0:
 390:             return 0.0
 391: 
 392:         total_value = sum(
 393:             p["shares"] * p["price_usd"] for p in self._positions.values()
 394:         )
 395:         if total_value <= 0:
 396:             return 0.0
 397: 
 398:         loss_threshold = -total_value * threshold_pct / 100.0
 399:         return float(np.mean(self._pnl < loss_threshold))
 400: 
 401: 
 402: # ---------------------------------------------------------------------------
 403: # Convenience: compute portfolio VaR from system data structures
 404: # ---------------------------------------------------------------------------
 405: 
 406: def compute_portfolio_var(
 407:     portfolio_state: dict,
 408:     agent_summary: dict,
 409:     n_paths: int = DEFAULT_N_PATHS,
 410:     historical_returns: dict[str, np.ndarray] | None = None,
 411:     seed: int | None = None,
 412: ) -> dict:
 413:     """Compute portfolio VaR from portfolio state and agent summary.
 414: 
 415:     Extracts held positions, current prices, volatilities, and directional
 416:     probabilities from the system data, then runs t-copula simulation.
 417: 
 418:     Args:
 419:         portfolio_state: Portfolio state dict (holdings, cash_sek, etc.).
 420:         agent_summary: Agent summary with signals, fx_rate.
 421:         n_paths: Number of MC paths.
 422:         historical_returns: Optional empirical return series for correlation.
 423:         seed: Random seed.
 424: 
 425:     Returns:
 426:         Dict with VaR metrics in both USD and SEK.
 427:     """
 428:     holdings = portfolio_state.get("holdings", {})
 429:     signals = agent_summary.get("signals", {})
 430:     fx_rate = agent_summary.get("fx_rate", 10.0)
 431: 
 432:     # Build positions dict
 433:     positions = {}
 434:     tickers = []
 435: 
 436:     for ticker, holding in holdings.items():
 437:         shares = holding.get("shares", 0)
 438:         if shares <= 0:
 439:             continue
 440: 
 441:         ticker_data = signals.get(ticker, {})
 442:         price = ticker_data.get("price_usd", 0)
 443:         if price <= 0:
 444:             continue
 445: 
 446:         extra = ticker_data.get("extra", {})
 447:         atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
 448:         vol = volatility_from_atr(atr_pct)
 449: 
 450:         # Get directional probability for drift
 451:         from portfolio.monte_carlo import _get_directional_probability
 452:         p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
 453:         drift = drift_from_probability(p_up, vol)
 454: 
 455:         tickers.append(ticker)
 456:         positions[ticker] = {
 457:             "shares": shares,
 458:             "price_usd": price,
 459:             "volatility": vol,
 460:             "drift": drift,
 461:         }
 462: 
 463:     if not tickers:
 464:         return {
 465:             "var_95_usd": 0.0,
 466:             "var_99_usd": 0.0,
 467:             "cvar_95_usd": 0.0,
 468:             "cvar_99_usd": 0.0,
 469:             "var_95_sek": 0.0,
 470:             "var_99_sek": 0.0,
 471:             "cvar_95_sek": 0.0,
 472:             "cvar_99_sek": 0.0,
 473:             "total_exposure_usd": 0.0,
 474:             "total_exposure_sek": 0.0,
 475:             "n_positions": 0,
 476:             "drawdown_1pct_prob": 0.0,
 477:             "drawdown_5pct_prob": 0.0,
 478:         }
 479: 
 480:     # Build correlation matrix

-- lines 481-515 --
 481:     corr = build_correlation_matrix(tickers, historical_returns)
 482: 
 483:     # Run simulation
 484:     sim = PortfolioRiskSimulator(
 485:         positions=positions,
 486:         correlation_matrix=corr,
 487:         horizon_days=1,
 488:         n_paths=n_paths,
 489:         df=DEFAULT_DF,
 490:         seed=seed,
 491:     )
 492:     sim.simulate_correlated_returns()
 493: 
 494:     total_exposure = sum(p["shares"] * p["price_usd"] for p in positions.values())
 495: 
 496:     var95 = sim.var(0.95)
 497:     var99 = sim.var(0.99)
 498:     cvar95 = sim.cvar(0.95)
 499:     cvar99 = sim.cvar(0.99)
 500: 
 501:     return {
 502:         "var_95_usd": round(var95, 2),
 503:         "var_99_usd": round(var99, 2),
 504:         "cvar_95_usd": round(cvar95, 2),
 505:         "cvar_99_usd": round(cvar99, 2),
 506:         "var_95_sek": round(var95 * fx_rate, 2),
 507:         "var_99_sek": round(var99 * fx_rate, 2),
 508:         "cvar_95_sek": round(cvar95 * fx_rate, 2),
 509:         "cvar_99_sek": round(cvar99 * fx_rate, 2),
 510:         "total_exposure_usd": round(total_exposure, 2),
 511:         "total_exposure_sek": round(total_exposure * fx_rate, 2),
 512:         "n_positions": len(tickers),
 513:         "drawdown_1pct_prob": round(sim.drawdown_probability(1.0), 3),
 514:         "drawdown_5pct_prob": round(sim.drawdown_probability(5.0), 3),
 515:     }


===== portfolio_validator.py (299 lines) =====
-- lines 1-120 --
   1: """Portfolio state validation utilities.
   2: 
   3: Validates portfolio state files for data integrity, ensuring that cash,
   4: holdings, fees, and transaction records are all internally consistent.
   5: """
   6: 
   7: import pathlib
   8: from collections import defaultdict
   9: 
  10: from portfolio.file_utils import load_json
  11: 
  12: 
  13: def validate_portfolio(portfolio: dict) -> list[str]:
  14:     """Validate portfolio state integrity.
  15: 
  16:     Performs comprehensive checks on a portfolio state dict to ensure
  17:     all invariants hold. Returns a list of error messages -- an empty
  18:     list means the portfolio is valid.
  19: 
  20:     Checks:
  21:     1. Cash is non-negative
  22:     2. All shares are non-negative
  23:     3. Cash reconciliation: initial - sum(BUY allocs) + sum(SELL net_proceeds) = cash_sek
  24:     4. Holdings shares reconciliation: sum(BUY shares) - sum(SELL shares) per ticker
  25:     5. Fee reconciliation: total_fees_sek = sum(all fee_sek in transactions)
  26:     6. No duplicate timestamps in transactions (same ticker + same timestamp)
  27:     7. Transaction field completeness
  28:     8. Holdings avg_cost_usd consistency
  29: 
  30:     Args:
  31:         portfolio: Full portfolio state dict with keys:
  32:             - cash_sek, holdings, transactions, initial_value_sek, total_fees_sek
  33: 
  34:     Returns:
  35:         list of error message strings (empty = valid).
  36:     """
  37:     errors = []
  38: 
  39:     # --- Extract fields with defaults ---
  40:     cash_sek = portfolio.get("cash_sek")
  41:     holdings = portfolio.get("holdings", {})
  42:     transactions = portfolio.get("transactions", [])
  43:     initial_value = portfolio.get("initial_value_sek", 500_000)
  44:     total_fees_sek = portfolio.get("total_fees_sek")
  45: 
  46:     # --- Check 0: Required fields exist ---
  47:     if cash_sek is None:
  48:         errors.append("Missing required field: cash_sek")
  49:         cash_sek = 0
  50:     if total_fees_sek is None:
  51:         errors.append("Missing or null field: total_fees_sek (should be 0 if no fees)")
  52:         total_fees_sek = 0
  53:     if "initial_value_sek" not in portfolio:
  54:         errors.append("Missing required field: initial_value_sek")
  55: 
  56:     # --- Check 1: Cash is non-negative ---
  57:     if cash_sek < 0:
  58:         errors.append(f"Cash is negative: {cash_sek:.2f} SEK")
  59: 
  60:     # --- Check 2: All shares are non-negative ---
  61:     for ticker, pos in holdings.items():
  62:         shares = pos.get("shares", 0)
  63:         if shares < 0:
  64:             errors.append(f"Negative shares for {ticker}: {shares}")
  65:         avg_cost = pos.get("avg_cost_usd")
  66:         if avg_cost is not None and avg_cost < 0:
  67:             errors.append(f"Negative avg_cost_usd for {ticker}: {avg_cost}")
  68: 
  69:     # --- Check 3: Cash reconciliation ---
  70:     # cash = initial - sum(BUY total_sek) + sum(SELL total_sek)
  71:     # Note: BUY total_sek = full allocation (including fee)
  72:     # SELL total_sek = net proceeds (after fee deducted)
  73:     total_buy_alloc = 0.0
  74:     total_sell_proceeds = 0.0
  75:     for tx in transactions:
  76:         action = tx.get("action", "")
  77:         total_sek = tx.get("total_sek", 0) or 0
  78:         if action == "BUY":
  79:             total_buy_alloc += total_sek
  80:         elif action == "SELL":
  81:             total_sell_proceeds += total_sek
  82: 
  83:     expected_cash = initial_value - total_buy_alloc + total_sell_proceeds
  84:     cash_diff = abs(expected_cash - cash_sek)
  85:     if cash_diff > 1.0:  # Allow 1 SEK tolerance for floating point
  86:         errors.append(
  87:             f"Cash reconciliation failed: expected {expected_cash:.2f} SEK "
  88:             f"(initial {initial_value} - buys {total_buy_alloc:.2f} + sells {total_sell_proceeds:.2f}), "
  89:             f"got {cash_sek:.2f} SEK (diff: {cash_diff:.2f})"
  90:         )
  91: 
  92:     # --- Check 4: Holdings shares reconciliation ---
  93:     # For each ticker: net_shares = sum(BUY shares) - sum(SELL shares)
  94:     ticker_bought = defaultdict(float)
  95:     ticker_sold = defaultdict(float)
  96:     for tx in transactions:
  97:         ticker = tx.get("ticker", "")
  98:         action = tx.get("action", "")
  99:         shares = tx.get("shares", 0) or 0
 100:         if action == "BUY":
 101:             ticker_bought[ticker] += shares
 102:         elif action == "SELL":
 103:             ticker_sold[ticker] += shares
 104: 
 105:     # Check tickers that appear in transactions
 106:     all_tx_tickers = set(ticker_bought.keys()) | set(ticker_sold.keys())
 107:     for ticker in all_tx_tickers:
 108:         expected_shares = ticker_bought[ticker] - ticker_sold[ticker]
 109: 
 110:         # Get actual shares from holdings
 111:         if ticker in holdings:
 112:             actual_shares = holdings[ticker].get("shares", 0)
 113:         else:
 114:             actual_shares = 0
 115: 
 116:         # Compare (with tolerance for floating point and rounding from repeated partial sells)
 117:         share_diff = abs(expected_shares - actual_shares)
 118:         if share_diff > 1e-6:
 119:             if expected_shares <= 1e-9 and actual_shares == 0:
 120:                 # Both effectively zero -- OK (sold all, removed from holdings)

-- lines 121-240 --
 121:                 continue
 122:             if expected_shares <= 1e-9 and ticker not in holdings:
 123:                 # Fully sold, ticker removed from holdings -- OK
 124:                 continue
 125:             # Tolerance for small remainders from repeated partial sells (e.g.,
 126:             # multiple 50% sells that don't sum exactly to total bought due to
 127:             # floating-point rounding). Allow up to 1% of total bought shares.
 128:             total_bought = ticker_bought[ticker]
 129:             relative_diff = share_diff / total_bought if total_bought > 0 else float("inf")
 130:             if actual_shares == 0 and ticker not in holdings and relative_diff < 0.01:
 131:                 # Small remainder from rounding, ticker removed -- acceptable
 132:                 continue
 133:             errors.append(
 134:                 f"Holdings mismatch for {ticker}: expected {expected_shares:.8f} shares "
 135:                 f"(bought {ticker_bought[ticker]:.8f} - sold {ticker_sold[ticker]:.8f}), "
 136:                 f"got {actual_shares:.8f} in holdings (diff: {share_diff:.8f})"
 137:             )
 138: 
 139:     # Check for holdings tickers not in transactions
 140:     for ticker in holdings:
 141:         shares = holdings[ticker].get("shares", 0)
 142:         if shares > 0 and ticker not in all_tx_tickers:
 143:             errors.append(
 144:                 f"Holdings contains {ticker} with {shares} shares but no matching transactions"
 145:             )
 146: 
 147:     # --- Check 5: Fee reconciliation ---
 148:     computed_fees = 0.0
 149:     tx_with_fees = 0
 150:     tx_without_fees = 0
 151:     for tx in transactions:
 152:         fee = tx.get("fee_sek")
 153:         if fee is not None:
 154:             computed_fees += fee
 155:             tx_with_fees += 1
 156:         else:
 157:             tx_without_fees += 1
 158: 
 159:     if tx_without_fees > 0 and tx_with_fees > 0:
 160:         errors.append(
 161:             f"Inconsistent fee tracking: {tx_with_fees} transactions have fee_sek, "
 162:             f"{tx_without_fees} do not"
 163:         )
 164: 
 165:     # Only compare fees if transactions have fee_sek fields
 166:     if tx_with_fees > 0:
 167:         fee_diff = abs(computed_fees - total_fees_sek)
 168:         if fee_diff > 0.01:  # 0.01 SEK tolerance
 169:             errors.append(
 170:                 f"Fee reconciliation failed: sum of transaction fees = {computed_fees:.2f} SEK, "
 171:                 f"total_fees_sek = {total_fees_sek:.2f} SEK (diff: {fee_diff:.2f})"
 172:             )
 173:     elif len(transactions) > 0 and total_fees_sek == 0:
 174:         # Transactions exist but no fee tracking at all -- warn
 175:         errors.append(
 176:             "No fee_sek fields in any transaction and total_fees_sek is 0, "
 177:             "but transactions exist. Fees may not be tracked."
 178:         )
 179: 
 180:     # --- Check 6: No duplicate timestamps per ticker ---
 181:     seen_tx = set()
 182:     for i, tx in enumerate(transactions):
 183:         key = (tx.get("ticker", ""), tx.get("timestamp", ""), tx.get("action", ""))
 184:         if key in seen_tx:
 185:             errors.append(
 186:                 f"Duplicate transaction at index {i}: {key[2]} {key[0]} at {key[1]}"
 187:             )
 188:         seen_tx.add(key)
 189: 
 190:     # --- Check 7: Transaction field completeness ---
 191:     required_tx_fields = ["timestamp", "ticker", "action", "shares", "price_usd",
 192:                           "total_sek", "reason"]
 193: 
 194:     for i, tx in enumerate(transactions):
 195:         for field in required_tx_fields:
 196:             if field not in tx or tx[field] is None:
 197:                 errors.append(f"Transaction {i} missing required field: {field}")
 198: 
 199:         # Validate action value
 200:         action = tx.get("action", "")
 201:         if action not in ("BUY", "SELL"):
 202:             errors.append(f"Transaction {i} has invalid action: '{action}' (expected BUY or SELL)")
 203: 
 204:         # Validate shares > 0
 205:         shares = tx.get("shares", 0)
 206:         if shares is not None and shares <= 0:
 207:             errors.append(f"Transaction {i} has non-positive shares: {shares}")
 208: 
 209:         # Validate total_sek > 0
 210:         total = tx.get("total_sek", 0)
 211:         if total is not None and total <= 0:
 212:             errors.append(f"Transaction {i} has non-positive total_sek: {total}")
 213: 
 214:     # --- Check 8: Holdings avg_cost_usd consistency ---
 215:     # For tickers with multiple BUY transactions, verify avg_cost is plausible
 216:     for ticker, pos in holdings.items():
 217:         shares = pos.get("shares", 0)
 218:         if shares <= 0:
 219:             continue
 220:         avg_cost = pos.get("avg_cost_usd")
 221:         if avg_cost is None:
 222:             errors.append(f"Holdings {ticker} missing avg_cost_usd")
 223:             continue
 224: 
 225:         # Compute weighted average from BUY transactions
 226:         total_cost = 0.0
 227:         total_bought = 0.0
 228:         for tx in transactions:
 229:             if tx.get("ticker") != ticker or tx.get("action") != "BUY":
 230:                 continue
 231:             tx_shares = tx.get("shares", 0) or 0
 232:             tx_price = tx.get("price_usd", 0) or 0
 233:             total_cost += tx_shares * tx_price
 234:             total_bought += tx_shares
 235: 
 236:         if total_bought > 0:
 237:             expected_avg = total_cost / total_bought
 238:             avg_diff_pct = abs(expected_avg - avg_cost) / expected_avg * 100 if expected_avg > 0 else 0
 239:             if avg_diff_pct > 1.0:  # More than 1% off
 240:                 errors.append(

-- lines 241-299 --
 241:                     f"Holdings {ticker} avg_cost_usd ({avg_cost:.4f}) differs from "
 242:                     f"computed weighted average ({expected_avg:.4f}) by {avg_diff_pct:.2f}%"
 243:                 )
 244: 
 245:     return errors
 246: 
 247: 
 248: def validate_portfolio_file(path: str) -> list[str]:
 249:     """Validate a portfolio state JSON file.
 250: 
 251:     Convenience wrapper that loads the file and runs validate_portfolio().
 252: 
 253:     Args:
 254:         path: Path to the portfolio state JSON file.
 255: 
 256:     Returns:
 257:         list of error message strings (empty = valid).
 258:     """
 259:     # A-PR-3 (2026-04-11): Use file_utils.load_json instead of raw open()+
 260:     # json.load(). The raw path had a TOCTOU race with concurrent saves
 261:     # from portfolio_mgr.atomic_write_json: a partial write could be read
 262:     # mid-flight, producing a JSONDecodeError that propagated as a "valid
 263:     # but malformed" portfolio. load_json() retries on transient JSON
 264:     # decode failures and uses the same atomic-rename window as the writer.
 265:     p = pathlib.Path(path)
 266:     if not p.exists():
 267:         return [f"Portfolio file not found: {path}"]
 268:     portfolio = load_json(p)
 269:     if portfolio is None:
 270:         return [f"Invalid or unreadable JSON in {path}"]
 271: 
 272:     return validate_portfolio(portfolio)
 273: 
 274: 
 275: def validate_all() -> dict[str, list[str]]:
 276:     """Validate both patient and bold portfolio files.
 277: 
 278:     Returns:
 279:         dict with keys "patient" and "bold", each containing a list of errors.
 280:     """
 281:     data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
 282:     return {
 283:         "patient": validate_portfolio_file(str(data_dir / "portfolio_state.json")),
 284:         "bold": validate_portfolio_file(str(data_dir / "portfolio_state_bold.json")),
 285:     }
 286: 
 287: 
 288: if __name__ == "__main__":
 289:     results = validate_all()
 290:     for strategy, errs in results.items():
 291:         print(f"\n{'='*60}")
 292:         print(f"  {strategy.upper()} PORTFOLIO VALIDATION")
 293:         print(f"{'='*60}")
 294:         if errs:
 295:             for e in errs:
 296:                 print(f"  ERROR: {e}")
 297:         else:
 298:             print("  VALID - all checks passed")
 299:     print()


===== trade_guards.py (388 lines) =====
-- lines 1-120 --
   1: """Trade guards — overtrading prevention for the trading agent.
   2: 
   3: Three guards:
   4: 1. Per-ticker cooldown: No re-trade on same ticker within N minutes.
   5: 2. Consecutive-loss escalation: After losses, increase cooldown multiplier.
   6: 3. Position rate limit: Max N new positions per time window.
   7: 
   8: State is persisted to data/trade_guard_state.json.
   9: """
  10: 
  11: import logging
  12: import threading
  13: from datetime import UTC, datetime, timedelta
  14: from pathlib import Path
  15: 
  16: from portfolio.file_utils import atomic_write_json, load_json
  17: 
  18: logger = logging.getLogger("portfolio.trade_guards")
  19: 
  20: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
  21: STATE_FILE = DATA_DIR / "trade_guard_state.json"
  22: 
  23: # Defaults
  24: DEFAULT_TICKER_COOLDOWN_MINUTES = 30
  25: DEFAULT_BOLD_POSITION_LIMIT = 1       # max new positions per window
  26: DEFAULT_BOLD_POSITION_WINDOW_H = 4    # hours
  27: DEFAULT_PATIENT_POSITION_LIMIT = 1
  28: DEFAULT_PATIENT_POSITION_WINDOW_H = 8
  29: LOSS_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive_losses -> cooldown multiplier
  30: LOSS_DECAY_HOURS = 24  # halve escalation multiplier every N hours without a trade
  31: 
  32: _state_lock = threading.Lock()
  33: 
  34: 
  35: def _load_state():
  36:     """Load trade guard state from disk."""
  37:     return load_json(str(STATE_FILE), default={
  38:         "ticker_trades": {},
  39:         "consecutive_losses": {"patient": 0, "bold": 0},
  40:         "last_loss_ts": {"patient": None, "bold": None},
  41:         "new_position_timestamps": {"patient": [], "bold": []},
  42:     })
  43: 
  44: 
  45: def _save_state(state):
  46:     """Persist trade guard state to disk."""
  47:     atomic_write_json(STATE_FILE, state)
  48: 
  49: 
  50: def _portfolios_have_transactions():
  51:     """Return True if any portfolio file has at least one recorded transaction.
  52: 
  53:     Used by the C4 sanity check to distinguish "no trades happened yet"
  54:     (quiet startup state) from "trades happened but weren't recorded"
  55:     (broken wiring — real bug).
  56: 
  57:     2026-04-22 follow-up: include warrants portfolio — CLAUDE.md lists it as
  58:     an independent strategy state file, and warrants-only activity would
  59:     have left C4 silent forever.
  60:     """
  61:     for pf_name in (
  62:         "portfolio_state.json",
  63:         "portfolio_state_bold.json",
  64:         "portfolio_state_warrants.json",
  65:     ):
  66:         pf = load_json(str(DATA_DIR / pf_name), default={})
  67:         if pf and pf.get("transactions"):
  68:             return True
  69:     return False
  70: 
  71: 
  72: def _get_cooldown_multiplier(consecutive_losses, last_loss_ts_str=None):
  73:     """Get cooldown multiplier based on consecutive loss count with time decay.
  74: 
  75:     After LOSS_DECAY_HOURS without a new trade, the multiplier halves
  76:     repeatedly (geometric decay). E.g. 8x → 4x after 24h → 2x after 48h → 1x.
  77:     """
  78:     if consecutive_losses >= 4:
  79:         base = LOSS_ESCALATION[4]
  80:     else:
  81:         base = LOSS_ESCALATION.get(consecutive_losses, 1)
  82: 
  83:     if base <= 1 or not last_loss_ts_str:
  84:         return base
  85: 
  86:     # Apply time-based decay
  87:     try:
  88:         last_loss = datetime.fromisoformat(
  89:             last_loss_ts_str.replace("Z", "+00:00")
  90:         )
  91:         if last_loss.tzinfo is None:
  92:             last_loss = last_loss.replace(tzinfo=UTC)
  93:         elapsed_hours = (datetime.now(UTC) - last_loss).total_seconds() / 3600
  94:         if elapsed_hours > LOSS_DECAY_HOURS:
  95:             halvings = int(elapsed_hours // LOSS_DECAY_HOURS)
  96:             base = max(1, base >> halvings)  # bit-shift right = halve
  97:     except (ValueError, TypeError, OverflowError):
  98:         pass
  99: 
 100:     return base
 101: 
 102: 
 103: def check_overtrading_guards(ticker, action, strategy, portfolio, config=None):
 104:     """Check all trade guards for a proposed trade.
 105: 
 106:     Args:
 107:         ticker: Instrument ticker (e.g., "BTC-USD").
 108:         action: "BUY" or "SELL".
 109:         strategy: "patient" or "bold".
 110:         portfolio: Portfolio state dict.
 111:         config: Optional config dict with trade_guards settings.
 112: 
 113:     Returns:
 114:         list of warning dicts, each with:
 115:             - guard: str (guard name)
 116:             - severity: "warning" or "block"
 117:             - message: str
 118:             - details: dict (guard-specific data)
 119:         Empty list means all guards pass.
 120:     """

-- lines 121-240 --
 121:     cfg = (config or {}).get("trade_guards", {})
 122:     if cfg.get("enabled") is False:
 123:         return []
 124: 
 125:     warnings = []
 126:     with _state_lock:
 127:         state = _load_state()
 128:     now = datetime.now(UTC)
 129: 
 130:     # --- Guard 1: Per-ticker cooldown ---
 131:     base_cooldown = cfg.get("ticker_cooldown_minutes", DEFAULT_TICKER_COOLDOWN_MINUTES)
 132:     consecutive = state.get("consecutive_losses", {}).get(strategy, 0)
 133:     last_loss_ts = state.get("last_loss_ts", {}).get(strategy)
 134:     multiplier = _get_cooldown_multiplier(consecutive, last_loss_ts)
 135:     effective_cooldown = base_cooldown * multiplier
 136: 
 137:     key = f"{strategy}:{ticker}"
 138:     ticker_trades = state.get("ticker_trades", {})
 139:     last_trade_str = ticker_trades.get(key)
 140:     if last_trade_str:
 141:         try:
 142:             last_trade = datetime.fromisoformat(last_trade_str)
 143:             # M8: ensure aware datetime before comparison with aware now
 144:             if last_trade.tzinfo is None:
 145:                 last_trade = last_trade.replace(tzinfo=UTC)
 146:             elapsed = (now - last_trade).total_seconds() / 60
 147:             if elapsed < effective_cooldown:
 148:                 remaining = effective_cooldown - elapsed
 149:                 warnings.append({
 150:                     "guard": "ticker_cooldown",
 151:                     "severity": "block",
 152:                     "message": (
 153:                         f"{ticker} traded {elapsed:.0f}m ago by {strategy}. "
 154:                         f"Cooldown: {effective_cooldown:.0f}m (base {base_cooldown}m × {multiplier}x). "
 155:                         f"{remaining:.0f}m remaining."
 156:                     ),
 157:                     "details": {
 158:                         "ticker": ticker,
 159:                         "strategy": strategy,
 160:                         "elapsed_min": round(elapsed, 1),
 161:                         "cooldown_min": effective_cooldown,
 162:                         "multiplier": multiplier,
 163:                         "remaining_min": round(remaining, 1),
 164:                     },
 165:                 })
 166:         except (ValueError, TypeError):
 167:             pass
 168: 
 169:     # --- Guard 2: Consecutive-loss escalation (informational) ---
 170:     if consecutive >= 2:
 171:         base_mult = _get_cooldown_multiplier(consecutive, None)
 172:         warnings.append({
 173:             "guard": "consecutive_losses",
 174:             "severity": "warning",
 175:             "message": (
 176:                 f"{strategy}: {consecutive} consecutive losses. "
 177:                 f"Cooldown multiplier: {multiplier}x"
 178:                 f"{f' (decayed from {base_mult}x)' if multiplier < base_mult else ''}."
 179:             ),
 180:             "details": {
 181:                 "strategy": strategy,
 182:                 "consecutive_losses": consecutive,
 183:                 "multiplier": multiplier,
 184:                 "base_multiplier": base_mult,
 185:                 "decayed": multiplier < base_mult,
 186:             },
 187:         })
 188: 
 189:     # --- Guard 3: Position rate limit (BUY only) ---
 190:     if action == "BUY":
 191:         is_bold = strategy == "bold"
 192:         limit = cfg.get(
 193:             f"{'bold' if is_bold else 'patient'}_position_limit",
 194:             DEFAULT_BOLD_POSITION_LIMIT if is_bold else DEFAULT_PATIENT_POSITION_LIMIT,
 195:         )
 196:         window_h = cfg.get(
 197:             f"{'bold' if is_bold else 'patient'}_position_window_h",
 198:             DEFAULT_BOLD_POSITION_WINDOW_H if is_bold else DEFAULT_PATIENT_POSITION_WINDOW_H,
 199:         )
 200:         cutoff = now - timedelta(hours=window_h)
 201: 
 202:         timestamps = state.get("new_position_timestamps", {}).get(strategy, [])
 203:         recent = []
 204:         for ts_str in timestamps:
 205:             try:
 206:                 ts = datetime.fromisoformat(ts_str)
 207:                 # M8: ensure aware datetime before comparison with aware cutoff
 208:                 if ts.tzinfo is None:
 209:                     ts = ts.replace(tzinfo=UTC)
 210:                 if ts >= cutoff:
 211:                     recent.append(ts)
 212:             except (ValueError, TypeError):
 213:                 continue
 214: 
 215:         if len(recent) >= limit:
 216:             warnings.append({
 217:                 "guard": "position_rate_limit",
 218:                 "severity": "block",
 219:                 "message": (
 220:                     f"{strategy}: {len(recent)} new position(s) in last {window_h}h "
 221:                     f"(limit: {limit})."
 222:                 ),
 223:                 "details": {
 224:                     "strategy": strategy,
 225:                     "recent_count": len(recent),
 226:                     "limit": limit,
 227:                     "window_hours": window_h,
 228:                 },
 229:             })
 230: 
 231:     return warnings
 232: 
 233: 
 234: _wiring_confirmed = False  # process-scoped flag — positive proof for C4
 235: 
 236: 
 237: def record_trade(ticker, direction, strategy, pnl_pct=None, config=None):
 238:     """Record a completed trade for guard tracking.
 239: 
 240:     Call this after executing a trade to update cooldowns and loss streaks.

-- lines 241-360 --
 241: 
 242:     Args:
 243:         ticker: Instrument ticker.
 244:         direction: "BUY" or "SELL".
 245:         strategy: "patient" or "bold".
 246:         pnl_pct: Realized P&L percentage (for SELL trades). None for BUY.
 247:         config: Optional config dict.
 248:     """
 249:     # 2026-04-22 follow-up: positive-proof wiring check. The previous C4
 250:     # warning was *reactive* — it could only tell you after a trade had
 251:     # already slipped through unguarded. Log INFO once per process the first
 252:     # time this function fires, so operators get explicit confirmation the
 253:     # BUG-219/PR-R4-4 wiring is alive rather than having to infer it from
 254:     # absence-of-warnings.
 255:     global _wiring_confirmed
 256:     if not _wiring_confirmed:
 257:         logger.info(
 258:             "C4: record_trade() wiring confirmed — first call this process "
 259:             "(ticker=%s direction=%s strategy=%s)",
 260:             ticker, direction, strategy,
 261:         )
 262:         _wiring_confirmed = True
 263: 
 264:     with _state_lock:
 265:         state = _load_state()
 266:         now = datetime.now(UTC)
 267:         now_str = now.isoformat()
 268: 
 269:         # Update ticker trade timestamp
 270:         key = f"{strategy}:{ticker}"
 271:         if "ticker_trades" not in state:
 272:             state["ticker_trades"] = {}
 273:         state["ticker_trades"][key] = now_str
 274: 
 275:         # Update consecutive losses on SELL
 276:         if direction == "SELL" and pnl_pct is not None:
 277:             if "consecutive_losses" not in state:
 278:                 state["consecutive_losses"] = {"patient": 0, "bold": 0}
 279:             if "last_loss_ts" not in state:
 280:                 state["last_loss_ts"] = {"patient": None, "bold": None}
 281:             if pnl_pct < 0:
 282:                 state["consecutive_losses"][strategy] = (
 283:                     state["consecutive_losses"].get(strategy, 0) + 1
 284:                 )
 285:                 state["last_loss_ts"][strategy] = now_str
 286:             else:
 287:                 state["consecutive_losses"][strategy] = 0
 288:                 state["last_loss_ts"][strategy] = None
 289: 
 290:         # Track new position timestamps (BUY only)
 291:         if direction == "BUY":
 292:             if "new_position_timestamps" not in state:
 293:                 state["new_position_timestamps"] = {"patient": [], "bold": []}
 294:             if strategy not in state["new_position_timestamps"]:
 295:                 state["new_position_timestamps"][strategy] = []
 296:             state["new_position_timestamps"][strategy].append(now_str)
 297: 
 298:             # Prune old timestamps (keep last 24h).
 299:             cutoff = now - timedelta(hours=24)
 300:             pruned = []
 301:             for ts in state["new_position_timestamps"][strategy]:
 302:                 try:
 303:                     dt = datetime.fromisoformat(ts)
 304:                     if dt.tzinfo is None:
 305:                         dt = dt.replace(tzinfo=UTC)
 306:                     if dt >= cutoff:
 307:                         pruned.append(ts)
 308:                 except (ValueError, TypeError):
 309:                     continue
 310:             state["new_position_timestamps"][strategy] = pruned
 311: 
 312:         _save_state(state)
 313: 
 314: 
 315: def get_all_guard_warnings(signals, patient_pf, bold_pf, config=None):
 316:     """Get trade guard warnings for all tickers with BUY/SELL signals.
 317: 
 318:     Args:
 319:         signals: Dict of ticker -> signal data (from agent_summary).
 320:         patient_pf: Patient portfolio state dict.
 321:         bold_pf: Bold portfolio state dict.
 322:         config: Optional config dict.
 323: 
 324:     Returns:
 325:         dict with:
 326:             - warnings: list of warning dicts
 327:             - summary: str (human-readable summary)
 328:     """
 329:     cfg = (config or {}).get("trade_guards", {})
 330:     if cfg.get("enabled") is False:
 331:         return {"warnings": [], "summary": "Trade guards disabled"}
 332: 
 333:     all_warnings = []
 334: 
 335:     for ticker, sig in signals.items():
 336:         action = sig.get("action", "HOLD")
 337:         if action == "HOLD":
 338:             continue
 339: 
 340:         for strategy, portfolio in [("patient", patient_pf), ("bold", bold_pf)]:
 341:             warns = check_overtrading_guards(
 342:                 ticker, action, strategy, portfolio, config
 343:             )
 344:             all_warnings.extend(warns)
 345: 
 346:     summary_parts = []
 347:     if all_warnings:
 348:         by_guard = {}
 349:         for w in all_warnings:
 350:             guard = w["guard"]
 351:             by_guard.setdefault(guard, []).append(w)
 352:         for guard, warns in by_guard.items():
 353:             summary_parts.append(f"{guard}: {len(warns)} warning(s)")
 354: 
 355:     # C4: Detect broken record_trade() wiring.
 356:     # 2026-04-22: original check fired every cycle whenever state was empty,
 357:     # even when no trades had happened yet (portfolios untouched) — noisy and
 358:     # misleading post-BUG-219/PR-R4-4 which wired _record_new_trades().
 359:     # Now only warn when portfolios DO have transactions but guard state is
 360:     # still empty — that's the real signal the wiring is broken.

-- lines 361-388 --
 361:     with _state_lock:
 362:         state = _load_state()
 363:     if not state.get("ticker_trades") and all_warnings == [] and _portfolios_have_transactions():
 364:         logger.warning(
 365:             "C4: portfolios have transactions but trade_guard_state.json "
 366:             "has no recorded trades — record_trade() wiring appears broken. "
 367:             "Overtrading guards are NON-FUNCTIONAL."
 368:         )
 369: 
 370:     return {
 371:         "warnings": all_warnings,
 372:         "summary": "; ".join(summary_parts) if summary_parts else "All clear",
 373:     }
 374: 
 375: 
 376: def should_block_trade(guard_result):
 377:     """Check if any guard warning has 'block' severity.
 378: 
 379:     ARCH-29: Convenience function for Layer 2 go/no-go decisions.
 380: 
 381:     Args:
 382:         guard_result: Return value from get_all_guard_warnings().
 383: 
 384:     Returns:
 385:         True if any warning has severity="block", False otherwise.
 386:     """
 387:     warnings = guard_result.get("warnings", [])
 388:     return any(w.get("severity") == "block" for w in warnings)


===== equity_curve.py (600 lines) =====
-- lines 1-120 --
   1: """Equity curve analysis and portfolio metrics.
   2: 
   3: Loads portfolio value history from the JSONL log and computes performance
   4: metrics useful for charting and strategy comparison.
   5: """
   6: 
   7: import datetime
   8: import math
   9: import pathlib
  10: from collections import defaultdict
  11: 
  12: from portfolio.file_utils import load_jsonl
  13: 
  14: DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
  15: 
  16: DEFAULT_HISTORY_PATH = DATA_DIR / "portfolio_value_history.jsonl"
  17: INITIAL_VALUE = 500_000  # SEK
  18: RISK_FREE_RATE_ANNUAL = 0.035  # 3.5% Swedish risk-free rate (approximate)
  19: 
  20: # The portfolio loop runs 24/7 (crypto always active), so
  21: # portfolio_value_history.jsonl has entries every calendar day.
  22: # Annualization must use 365, not 252 (stock-only convention).
  23: ANNUALIZATION_DAYS = 365
  24: 
  25: 
  26: def load_equity_curve(path: str | None = None) -> list[dict]:
  27:     """Load portfolio value history for charting.
  28: 
  29:     Reads the JSONL file and returns a list of dicts sorted by timestamp.
  30:     Each dict contains:
  31:         - ts: ISO-8601 timestamp
  32:         - patient_value_sek: float
  33:         - bold_value_sek: float
  34:         - patient_pnl_pct: float
  35:         - bold_pnl_pct: float
  36:         - fx_rate: float
  37:         - prices: dict of ticker -> USD price
  38: 
  39:     Args:
  40:         path: Path to the portfolio_value_history.jsonl file.
  41:             Defaults to data/portfolio_value_history.jsonl.
  42: 
  43:     Returns:
  44:         list of dicts sorted by timestamp (oldest first).
  45:         Empty list if file doesn't exist or is empty.
  46:     """
  47:     if path is None:
  48:         path = str(DEFAULT_HISTORY_PATH)
  49: 
  50:     result = load_jsonl(path)
  51: 
  52:     # Sort by timestamp
  53:     result.sort(key=lambda x: x.get("ts", ""))
  54:     return result
  55: 
  56: 
  57: def _parse_ts(ts_str: str) -> datetime.datetime:
  58:     """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
  59:     dt = datetime.datetime.fromisoformat(ts_str)
  60:     if dt.tzinfo is None:
  61:         dt = dt.replace(tzinfo=datetime.UTC)
  62:     return dt
  63: 
  64: 
  65: def _daily_returns(curve: list[dict], value_key: str) -> list[float]:
  66:     """Compute daily returns from the equity curve.
  67: 
  68:     Groups entries by date, takes the last entry per day, and computes
  69:     day-over-day percentage returns.
  70: 
  71:     Args:
  72:         curve: Sorted equity curve list.
  73:         value_key: Either "patient_value_sek" or "bold_value_sek".
  74: 
  75:     Returns:
  76:         list of daily return percentages.
  77:     """
  78:     if not curve:
  79:         return []
  80: 
  81:     # Group by date, take last value per day
  82:     daily_values = {}
  83:     for entry in curve:
  84:         ts_str = entry.get("ts", "")
  85:         if not ts_str:
  86:             continue
  87:         try:
  88:             dt = _parse_ts(ts_str)
  89:             date_key = dt.date()
  90:             value = entry.get(value_key, 0)
  91:             if value > 0:
  92:                 daily_values[date_key] = value
  93:         except (ValueError, TypeError):
  94:             continue
  95: 
  96:     if len(daily_values) < 2:
  97:         return []
  98: 
  99:     sorted_dates = sorted(daily_values.keys())
 100:     returns = []
 101:     for i in range(1, len(sorted_dates)):
 102:         prev_val = daily_values[sorted_dates[i - 1]]
 103:         curr_val = daily_values[sorted_dates[i]]
 104:         if prev_val > 0:
 105:             daily_ret = ((curr_val - prev_val) / prev_val) * 100
 106:             returns.append(daily_ret)
 107:         else:
 108:             returns.append(0.0)
 109: 
 110:     return returns
 111: 
 112: 
 113: def compute_metrics(curve: list[dict], strategy: str) -> dict:
 114:     """Compute portfolio metrics from equity curve.
 115: 
 116:     Args:
 117:         curve: List of equity curve entries (from load_equity_curve).
 118:         strategy: "patient" or "bold".
 119: 
 120:     Returns:

-- lines 121-240 --
 121:         dict with:
 122:             - max_drawdown_pct: float -- maximum peak-to-trough drawdown
 123:             - sharpe_ratio: float -- annualized Sharpe ratio (or None if insufficient data)
 124:             - sortino_ratio: float -- annualized Sortino ratio (or None)
 125:             - win_rate: float -- percentage of positive-return days
 126:             - avg_daily_return_pct: float -- mean daily return
 127:             - best_day_pct: float -- best single-day return
 128:             - worst_day_pct: float -- worst single-day return
 129:             - days_in_drawdown: int -- number of days below previous peak
 130:             - total_return_pct: float -- total return from start to end
 131:             - annualized_return_pct: float -- annualized return (or None if < 1 day)
 132:             - volatility_annual_pct: float -- annualized daily volatility
 133:             - num_data_points: int -- number of entries in curve
 134:             - date_range: tuple of (first_ts, last_ts)
 135:     """
 136:     value_key = f"{strategy}_value_sek"
 137: 
 138:     result = {
 139:         "max_drawdown_pct": 0.0,
 140:         "sharpe_ratio": None,
 141:         "sortino_ratio": None,
 142:         "win_rate": 0.0,
 143:         "avg_daily_return_pct": 0.0,
 144:         "best_day_pct": 0.0,
 145:         "worst_day_pct": 0.0,
 146:         "days_in_drawdown": 0,
 147:         "total_return_pct": 0.0,
 148:         "annualized_return_pct": None,
 149:         "volatility_annual_pct": 0.0,
 150:         "num_data_points": len(curve),
 151:         "date_range": None,
 152:     }
 153: 
 154:     if not curve:
 155:         return result
 156: 
 157:     # Extract values
 158:     values = []
 159:     timestamps = []
 160:     for entry in curve:
 161:         val = entry.get(value_key)
 162:         ts = entry.get("ts", "")
 163:         if val is not None and val > 0 and ts:
 164:             values.append(val)
 165:             timestamps.append(ts)
 166: 
 167:     if not values:
 168:         return result
 169: 
 170:     result["num_data_points"] = len(values)
 171:     result["date_range"] = (timestamps[0], timestamps[-1])
 172: 
 173:     # --- Total return ---
 174:     first_val = values[0]
 175:     last_val = values[-1]
 176:     total_return_pct = ((last_val - first_val) / first_val) * 100
 177:     result["total_return_pct"] = round(total_return_pct, 4)
 178: 
 179:     # --- Annualized return ---
 180:     try:
 181:         first_dt = _parse_ts(timestamps[0])
 182:         last_dt = _parse_ts(timestamps[-1])
 183:         days_elapsed = (last_dt - first_dt).total_seconds() / 86400
 184:         if days_elapsed >= 1:
 185:             years = days_elapsed / 365.25
 186:             # Annualized return = (final/initial)^(1/years) - 1
 187:             if first_val > 0 and last_val > 0:
 188:                 annualized = (pow(last_val / first_val, 1 / years) - 1) * 100
 189:                 result["annualized_return_pct"] = round(annualized, 4)
 190:     except (ValueError, TypeError, ZeroDivisionError):
 191:         pass
 192: 
 193:     # --- Maximum drawdown ---
 194:     peak = values[0]
 195:     max_dd = 0.0
 196:     # Group by date for drawdown day counting
 197:     date_was_below = set()
 198: 
 199:     for i, val in enumerate(values):
 200:         if val > peak:
 201:             peak = val
 202:         dd = ((peak - val) / peak) * 100
 203:         if dd > max_dd:
 204:             max_dd = dd
 205:         if dd > 0.01:  # Meaningfully below peak
 206:             try:
 207:                 dt = _parse_ts(timestamps[i])
 208:                 date_was_below.add(dt.date())
 209:             except (ValueError, TypeError):
 210:                 pass
 211: 
 212:     result["max_drawdown_pct"] = round(max_dd, 4)
 213:     result["days_in_drawdown"] = len(date_was_below)
 214: 
 215:     # --- Daily returns ---
 216:     daily_rets = _daily_returns(curve, value_key)
 217: 
 218:     if daily_rets:
 219:         result["avg_daily_return_pct"] = round(sum(daily_rets) / len(daily_rets), 6)
 220:         result["best_day_pct"] = round(max(daily_rets), 4)
 221:         result["worst_day_pct"] = round(min(daily_rets), 4)
 222: 
 223:         # Win rate
 224:         positive_days = sum(1 for r in daily_rets if r > 0)
 225:         result["win_rate"] = round((positive_days / len(daily_rets)) * 100, 2)
 226: 
 227:         # Volatility (annualized)
 228:         if len(daily_rets) >= 2:
 229:             mean_ret = sum(daily_rets) / len(daily_rets)
 230:             variance = sum((r - mean_ret) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
 231:             daily_vol = math.sqrt(variance)
 232:             annual_vol = daily_vol * math.sqrt(ANNUALIZATION_DAYS)
 233:             result["volatility_annual_pct"] = round(annual_vol, 4)
 234: 
 235:             # Sharpe ratio (annualized)
 236:             daily_rf = RISK_FREE_RATE_ANNUAL / ANNUALIZATION_DAYS
 237:             # Convert daily returns to decimal for Sharpe
 238:             daily_rets_dec = [r / 100 for r in daily_rets]
 239:             mean_excess = sum(r - daily_rf for r in daily_rets_dec) / len(daily_rets_dec)
 240:             if daily_vol > 0:

-- lines 241-360 --
 241:                 # Annualize Sharpe: mean_excess / daily_std * sqrt(ANNUALIZATION_DAYS)
 242:                 # BUG-225: extract mean to avoid O(n^2) recomputation inside generator
 243:                 mean_dec = sum(daily_rets_dec) / len(daily_rets_dec)
 244:                 daily_std_dec = math.sqrt(
 245:                     sum((r - mean_dec) ** 2
 246:                         for r in daily_rets_dec) / (len(daily_rets_dec) - 1)
 247:                 )
 248:                 if daily_std_dec > 0:
 249:                     sharpe = (mean_excess / daily_std_dec) * math.sqrt(ANNUALIZATION_DAYS)
 250:                     result["sharpe_ratio"] = round(sharpe, 4)
 251: 
 252:             # H19: Sortino — divide by TOTAL observations, not just downside count (standard formula)
 253:             squared_devs = [min(r - daily_rf, 0) ** 2 for r in daily_rets_dec]
 254:             downside_dev = math.sqrt(sum(squared_devs) / len(daily_rets_dec))
 255:             if downside_dev > 0:
 256:                 sortino = (mean_excess / downside_dev) * math.sqrt(ANNUALIZATION_DAYS)
 257:                 result["sortino_ratio"] = round(sortino, 4)
 258: 
 259:     return result
 260: 
 261: 
 262: def compare_strategies(curve: list[dict]) -> dict:
 263:     """Compare patient vs bold strategy performance.
 264: 
 265:     Args:
 266:         curve: Equity curve from load_equity_curve().
 267: 
 268:     Returns:
 269:         dict with:
 270:             - patient: metrics dict
 271:             - bold: metrics dict
 272:             - comparison: dict with relative performance
 273:     """
 274:     patient = compute_metrics(curve, "patient")
 275:     bold = compute_metrics(curve, "bold")
 276: 
 277:     comparison = {
 278:         "return_diff_pct": round(patient["total_return_pct"] - bold["total_return_pct"], 4),
 279:         "leader": "patient" if patient["total_return_pct"] > bold["total_return_pct"] else "bold",
 280:         "drawdown_diff_pct": round(patient["max_drawdown_pct"] - bold["max_drawdown_pct"], 4),
 281:         "lower_drawdown": "patient" if patient["max_drawdown_pct"] < bold["max_drawdown_pct"] else "bold",
 282:     }
 283: 
 284:     # Risk-adjusted comparison
 285:     if patient.get("sharpe_ratio") is not None and bold.get("sharpe_ratio") is not None:
 286:         comparison["sharpe_leader"] = "patient" if patient["sharpe_ratio"] > bold["sharpe_ratio"] else "bold"
 287:         comparison["sharpe_diff"] = round(patient["sharpe_ratio"] - bold["sharpe_ratio"], 4)
 288: 
 289:     return {
 290:         "patient": patient,
 291:         "bold": bold,
 292:         "comparison": comparison,
 293:     }
 294: 
 295: 
 296: def get_latest_values(curve: list[dict]) -> dict | None:
 297:     """Get the most recent portfolio values from the curve.
 298: 
 299:     Returns:
 300:         dict with patient_value_sek, bold_value_sek, ts, or None if curve is empty.
 301:     """
 302:     if not curve:
 303:         return None
 304:     latest = curve[-1]
 305:     return {
 306:         "ts": latest.get("ts"),
 307:         "patient_value_sek": latest.get("patient_value_sek"),
 308:         "bold_value_sek": latest.get("bold_value_sek"),
 309:         "patient_pnl_pct": latest.get("patient_pnl_pct"),
 310:         "bold_pnl_pct": latest.get("bold_pnl_pct"),
 311:     }
 312: 
 313: 
 314: def _pair_round_trips(transactions):
 315:     """Match BUY and SELL transactions into round-trip pairs.
 316: 
 317:     Uses FIFO matching: each SELL is paired with the earliest available BUY
 318:     shares for that ticker.
 319: 
 320:     Args:
 321:         transactions: List of transaction dicts from portfolio state.
 322: 
 323:     Returns:
 324:         list of round-trip dicts, each with:
 325:             - ticker: str
 326:             - buy_ts: str (ISO-8601)
 327:             - sell_ts: str (ISO-8601)
 328:             - buy_price_sek: float (per share)
 329:             - sell_price_sek: float (per share)
 330:             - shares: float
 331:             - pnl_pct: float
 332:             - pnl_sek: float
 333:             - hold_hours: float
 334:             - fee_sek: float (total fees for this round trip)
 335:     """
 336: 
 337:     # Group BUYs by ticker — maintain FIFO order
 338:     buy_queues = defaultdict(list)
 339:     for tx in transactions:
 340:         if tx.get("action") == "BUY":
 341:             ticker = tx.get("ticker", "")
 342:             shares = tx.get("shares", 0)
 343:             total_sek = tx.get("total_sek", 0)
 344:             fee = tx.get("fee_sek", 0) or 0
 345:             if shares > 0:
 346:                 price_per_share = total_sek / shares
 347:                 buy_queues[ticker].append({
 348:                     "ts": tx.get("timestamp", ""),
 349:                     "remaining_shares": shares,
 350:                     "original_shares": shares,
 351:                     "price_per_share": price_per_share,
 352:                     "fee_sek": fee,
 353:                 })
 354: 
 355:     round_trips = []
 356: 
 357:     for tx in transactions:
 358:         if tx.get("action") != "SELL":
 359:             continue
 360:         ticker = tx.get("ticker", "")

-- lines 361-480 --
 361:         sell_shares = tx.get("shares", 0)
 362:         sell_total = tx.get("total_sek", 0)
 363:         sell_fee = tx.get("fee_sek", 0) or 0
 364:         sell_ts = tx.get("timestamp", "")
 365: 
 366:         if sell_shares <= 0 or ticker not in buy_queues:
 367:             continue
 368: 
 369:         sell_price_per_share = sell_total / sell_shares if sell_shares > 0 else 0
 370:         shares_to_match = sell_shares
 371: 
 372:         while shares_to_match > 0 and buy_queues[ticker]:
 373:             buy = buy_queues[ticker][0]
 374:             matched = min(shares_to_match, buy["remaining_shares"])
 375: 
 376:             # Compute hold time
 377:             hold_hours = 0
 378:             try:
 379:                 buy_dt = datetime.datetime.fromisoformat(buy["ts"])
 380:                 sell_dt = datetime.datetime.fromisoformat(sell_ts)
 381:                 if buy_dt.tzinfo is None:
 382:                     buy_dt = buy_dt.replace(tzinfo=datetime.UTC)
 383:                 if sell_dt.tzinfo is None:
 384:                     sell_dt = sell_dt.replace(tzinfo=datetime.UTC)
 385:                 hold_hours = (sell_dt - buy_dt).total_seconds() / 3600
 386:             except (ValueError, TypeError):
 387:                 pass
 388: 
 389:             buy_price = buy["price_per_share"]
 390:             pnl_pct = ((sell_price_per_share - buy_price) / buy_price * 100) if buy_price > 0 else 0
 391: 
 392:             # Proportional fees — use original buy quantity as denominator
 393:             # to avoid inflating fee allocation on subsequent partial matches
 394:             # (BUG-37 regression covered by test_equity_curve_fifo.TestBug37FeeDoubleCount).
 395:             #
 396:             # P0-6 (2026-05-02): pnl_sek is now NET of buy+sell fees so
 397:             # downstream metrics (profit_factor, total_pnl_sek, Calmar) report
 398:             # realised SEK. `pnl_pct` remains the gross price-% (price-move
 399:             # only) — used for streaks and expectancy. `fee_sek` field still
 400:             # reports total fees so consumers can reconstruct gross if needed.
 401:             # See tests/test_equity_curve_fifo.TestPnlSekNetOfFees and
 402:             # tests/test_portfolio_metrics.TestProfitFactorNetOfFees.
 403:             buy_fee_share = (buy["fee_sek"] * matched / buy["original_shares"]) if buy["original_shares"] > 0 else 0
 404:             sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0
 405:             pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share
 406: 
 407:             round_trips.append({
 408:                 "ticker": ticker,
 409:                 "buy_ts": buy["ts"],
 410:                 "sell_ts": sell_ts,
 411:                 "buy_price_sek": round(buy_price, 4),
 412:                 "sell_price_sek": round(sell_price_per_share, 4),
 413:                 "shares": round(matched, 8),
 414:                 "pnl_pct": round(pnl_pct, 4),
 415:                 "pnl_sek": round(pnl_sek, 2),
 416:                 "hold_hours": round(hold_hours, 2),
 417:                 "fee_sek": round(buy_fee_share + sell_fee_share, 2),
 418:             })
 419: 
 420:             buy["remaining_shares"] -= matched
 421:             shares_to_match -= matched
 422: 
 423:             if buy["remaining_shares"] <= 1e-10:
 424:                 buy_queues[ticker].pop(0)
 425: 
 426:     return round_trips
 427: 
 428: 
 429: def compute_trade_metrics(transactions, initial_value=INITIAL_VALUE):
 430:     """Compute per-trade performance metrics from transaction history.
 431: 
 432:     Args:
 433:         transactions: List of transaction dicts from portfolio state.
 434:         initial_value: Starting portfolio value in SEK.
 435: 
 436:     Returns:
 437:         dict with:
 438:             - profit_factor: gross_profit / gross_loss (None if no losses)
 439:             - avg_hold_hours: average hold time per round trip
 440:             - trade_frequency_per_week: trades per 7 calendar days
 441:             - win_loss_ratio: avg_win_pct / avg_loss_pct (None if no losses)
 442:             - max_consecutive_wins: longest win streak
 443:             - max_consecutive_losses: longest loss streak
 444:             - expectancy_pct: (win_rate * avg_win) - (loss_rate * avg_loss)
 445:             - calmar_ratio: annualized_return / max_drawdown (None if insufficient data)
 446:             - round_trips: int (number of paired trades)
 447:             - total_pnl_sek: float (sum of round-trip P&L)
 448:     """
 449:     trips = _pair_round_trips(transactions)
 450: 
 451:     result = {
 452:         "profit_factor": None,
 453:         "avg_hold_hours": 0,
 454:         "trade_frequency_per_week": 0,
 455:         "win_loss_ratio": None,
 456:         "max_consecutive_wins": 0,
 457:         "max_consecutive_losses": 0,
 458:         "expectancy_pct": 0,
 459:         "calmar_ratio": None,
 460:         "round_trips": len(trips),
 461:         "total_pnl_sek": 0,
 462:     }
 463: 
 464:     if not trips:
 465:         return result
 466: 
 467:     # Gross profit/loss
 468:     gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)
 469:     gross_loss = abs(sum(t["pnl_sek"] for t in trips if t["pnl_sek"] < 0))
 470: 
 471:     if gross_loss > 0:
 472:         result["profit_factor"] = round(gross_profit / gross_loss, 4)
 473: 
 474:     # Average hold time
 475:     hold_hours = [t["hold_hours"] for t in trips if t["hold_hours"] > 0]
 476:     if hold_hours:
 477:         result["avg_hold_hours"] = round(sum(hold_hours) / len(hold_hours), 2)
 478: 
 479:     # Trade frequency
 480:     try:

-- lines 481-600 --
 481:         timestamps = []
 482:         for t in trips:
 483:             ts_str = t.get("sell_ts") or t.get("buy_ts")
 484:             if ts_str:
 485:                 timestamps.append(datetime.datetime.fromisoformat(ts_str))
 486:         if len(timestamps) >= 2:
 487:             span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
 488:             if span_days > 0:
 489:                 result["trade_frequency_per_week"] = round(len(trips) / span_days * 7, 2)
 490:     except (ValueError, TypeError):
 491:         pass
 492: 
 493:     # Win/loss stats
 494:     wins = [t for t in trips if t["pnl_pct"] > 0]
 495:     losses = [t for t in trips if t["pnl_pct"] <= 0]
 496:     win_count = len(wins)
 497:     loss_count = len(losses)
 498:     total_count = len(trips)
 499: 
 500:     avg_win_pct = sum(t["pnl_pct"] for t in wins) / win_count if wins else 0
 501:     avg_loss_pct = abs(sum(t["pnl_pct"] for t in losses) / loss_count) if losses else 0
 502: 
 503:     if avg_loss_pct > 0:
 504:         result["win_loss_ratio"] = round(avg_win_pct / avg_loss_pct, 4)
 505: 
 506:     # Streaks
 507:     max_wins = 0
 508:     max_losses = 0
 509:     current_wins = 0
 510:     current_losses = 0
 511:     for t in trips:
 512:         if t["pnl_pct"] > 0:
 513:             current_wins += 1
 514:             current_losses = 0
 515:             max_wins = max(max_wins, current_wins)
 516:         else:
 517:             current_losses += 1
 518:             current_wins = 0
 519:             max_losses = max(max_losses, current_losses)
 520: 
 521:     result["max_consecutive_wins"] = max_wins
 522:     result["max_consecutive_losses"] = max_losses
 523: 
 524:     # Expectancy
 525:     win_rate = win_count / total_count if total_count > 0 else 0
 526:     loss_rate = loss_count / total_count if total_count > 0 else 0
 527:     result["expectancy_pct"] = round(
 528:         (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct), 4
 529:     )
 530: 
 531:     # Total P&L
 532:     result["total_pnl_sek"] = round(sum(t["pnl_sek"] for t in trips), 2)
 533: 
 534:     # Calmar ratio (annualized return / max drawdown)
 535:     # Compute a mini equity curve from round-trip PnLs
 536:     if initial_value > 0 and len(trips) >= 2:
 537:         equity = [initial_value]
 538:         for t in trips:
 539:             equity.append(equity[-1] + t["pnl_sek"])
 540: 
 541:         peak = equity[0]
 542:         max_dd = 0
 543:         for val in equity:
 544:             if val > peak:
 545:                 peak = val
 546:             dd = (peak - val) / peak if peak > 0 else 0
 547:             if dd > max_dd:
 548:                 max_dd = dd
 549: 
 550:         total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0
 551:         try:
 552:             first_ts = datetime.datetime.fromisoformat(trips[0]["buy_ts"])
 553:             last_ts = datetime.datetime.fromisoformat(trips[-1]["sell_ts"])
 554:             days = (last_ts - first_ts).total_seconds() / 86400
 555:             if days >= 1 and max_dd > 0:
 556:                 years = days / 365.25
 557:                 annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
 558:                 result["calmar_ratio"] = round(annualized / max_dd, 4)
 559:         except (ValueError, TypeError):
 560:             pass
 561: 
 562:     return result
 563: 
 564: 
 565: if __name__ == "__main__":
 566:     curve = load_equity_curve()
 567:     if not curve:
 568:         print("No equity curve data found in", DEFAULT_HISTORY_PATH)
 569:         print("Run risk_management.log_portfolio_value() to start logging.")
 570:     else:
 571:         print(f"Loaded {len(curve)} data points")
 572:         results = compare_strategies(curve)
 573: 
 574:         for strategy in ("patient", "bold"):
 575:             m = results[strategy]
 576:             print(f"\n{'='*50}")
 577:             print(f"  {strategy.upper()} STRATEGY METRICS")
 578:             print(f"{'='*50}")
 579:             print(f"  Total return:     {m['total_return_pct']:+.2f}%")
 580:             print(f"  Max drawdown:     {m['max_drawdown_pct']:.2f}%")
 581:             if m["sharpe_ratio"] is not None:
 582:                 print(f"  Sharpe ratio:     {m['sharpe_ratio']:.2f}")
 583:             if m["sortino_ratio"] is not None:
 584:                 print(f"  Sortino ratio:    {m['sortino_ratio']:.2f}")
 585:             print(f"  Win rate:         {m['win_rate']:.1f}%")
 586:             print(f"  Best day:         {m['best_day_pct']:+.2f}%")
 587:             print(f"  Worst day:        {m['worst_day_pct']:+.2f}%")
 588:             print(f"  Days in drawdown: {m['days_in_drawdown']}")
 589:             print(f"  Volatility (ann): {m['volatility_annual_pct']:.2f}%")
 590:             if m["date_range"]:
 591:                 print(f"  Date range:       {m['date_range'][0][:10]} to {m['date_range'][1][:10]}")
 592: 
 593:         c = results["comparison"]
 594:         print(f"\n{'='*50}")
 595:         print("  COMPARISON")
 596:         print(f"{'='*50}")
 597:         print(f"  Return leader:    {c['leader']} (by {abs(c['return_diff_pct']):.2f}%)")
 598:         print(f"  Lower drawdown:   {c['lower_drawdown']} (by {abs(c['drawdown_diff_pct']):.2f}%)")
 599:         if "sharpe_leader" in c:
 600:             print(f"  Better Sharpe:    {c['sharpe_leader']} (by {abs(c['sharpe_diff']):.2f})")


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p=Path('portfolio/monte_carlo.py')
lines=p.read_text().splitlines()
ranges=[(1,140),(141,280),(281,422)]
for start,end in ranges:
    print(f'-- lines {start}-{end} --')
    for i in range(start,end+1):
        print(f\"{i:4}: {lines[i-1]}\")
    print()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 647ms:
-- lines 1-140 --
   1: """Monte Carlo price simulation engine.
   2: 
   3: Generates stochastic price paths using Geometric Brownian Motion (GBM)
   4: with antithetic variates for variance reduction. Converts directional
   5: signal probabilities into price quantile bands, stop-loss probabilities,
   6: and expected return distributions.
   7: 
   8: Short-term focus: 3h, 1d, 3d horizons for active trading decisions.
   9: 
  10: Usage:
  11:     from portfolio.monte_carlo import simulate_ticker
  12:     result = simulate_ticker("BTC-USD", agent_summary)
  13:     # result = {
  14:     #   "price_bands_1d": {"p5": 82100, "p25": 84300, "p50": 86200, ...},
  15:     #   "p_stop_hit_1d": 0.12,
  16:     #   "expected_return_1d": {"mean_pct": 0.3, "std_pct": 2.1, "skew": 0.1},
  17:     #   ...
  18:     # }
  19: """
  20: 
  21: import logging
  22: import math
  23: 
  24: import numpy as np
  25: from scipy.stats import norm
  26: 
  27: logger = logging.getLogger("portfolio.monte_carlo")
  28: 
  29: # Default parameters
  30: DEFAULT_N_PATHS = 10_000   # 5K pairs with antithetic variates
  31: DEFAULT_HORIZONS = [1, 3]  # days
  32: MIN_VOLATILITY = 0.05      # 5% annualized floor (prevents degenerate sims)
  33: 
  34: # Per-asset-class ATR fallbacks when actual ATR is missing from signals.
  35: # The generic 2.0% underestimates tail risk for crypto/metals.
  36: _ATR_DEFAULT_BY_CLASS = {
  37:     "crypto": 3.5,  # BTC 3-4% typical daily ATR
  38:     "metals": 4.0,  # XAG 4-6%, XAU 2-3% — biased toward silver
  39:     "stocks": 2.0,  # MSTR ~3% but broader equities lower
  40: }
  41: 
  42: 
  43: # ---------------------------------------------------------------------------
  44: # Volatility & drift estimation from existing system data
  45: # ---------------------------------------------------------------------------
  46: 
  47: def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
  48:     """Convert ATR% (14-period) to annualized volatility.
  49: 
  50:     ATR measures average true range over `period` candles. For hourly candles
  51:     (our primary timeframe), we annualize: vol = atr_frac * sqrt(trading_periods/period).
  52: 
  53:     Uses 252 trading days (standard for stocks/crypto).
  54: 
  55:     Args:
  56:         atr_pct: ATR as percentage of price (e.g., 3.5 means 3.5%).
  57:         period: ATR lookback period (default 14).
  58: 
  59:     Returns:
  60:         Annualized volatility as a decimal (e.g., 0.20 = 20%).
  61:     """
  62:     atr_frac = atr_pct / 100.0
  63:     annual_factor = math.sqrt(252.0 / period)
  64:     vol = atr_frac * annual_factor
  65:     return max(vol, MIN_VOLATILITY)
  66: 
  67: 
  68: def drift_from_probability(p_up: float, volatility: float) -> float:
  69:     """Convert directional probability P(up) into annualized drift.
  70: 
  71:     Uses the inverse of the GBM CDF relationship:
  72:         P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / (sigma*sqrt(T)))
  73: 
  74:     For 1-day horizon (T = 1/252):
  75:         mu = sigma * N_inv(p_up) * sqrt(252) + 0.5 * sigma^2
  76: 
  77:     This ensures the GBM simulation produces paths where the fraction
  78:     ending above spot matches the input probability.
  79: 
  80:     Args:
  81:         p_up: Probability of price being higher at horizon (0.0-1.0).
  82:         volatility: Annualized volatility (decimal).
  83: 
  84:     Returns:
  85:         Annualized drift (decimal). Positive = upward bias.
  86:     """
  87:     # Clamp p_up to avoid infinite drift at extremes
  88:     p_up = max(0.01, min(0.99, p_up))
  89: 
  90:     # N_inv(p_up) gives the z-score for the desired probability
  91:     z = norm.ppf(p_up)
  92: 
  93:     # mu = sigma * z * sqrt(252) + 0.5 * sigma^2
  94:     # This is derived from P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / sigma*sqrt(T))
  95:     mu = volatility * z * math.sqrt(252.0) + 0.5 * volatility**2
  96: 
  97:     return mu
  98: 
  99: 
 100: def _atr_default_for_ticker(ticker: str) -> float:
 101:     """Return the per-asset-class ATR fallback for a ticker."""
 102:     from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS
 103:     if ticker in CRYPTO_SYMBOLS:
 104:         return _ATR_DEFAULT_BY_CLASS["crypto"]
 105:     if ticker in METALS_SYMBOLS:
 106:         return _ATR_DEFAULT_BY_CLASS["metals"]
 107:     return _ATR_DEFAULT_BY_CLASS["stocks"]
 108: 
 109: 
 110: # ---------------------------------------------------------------------------
 111: # Core simulation engine
 112: # ---------------------------------------------------------------------------
 113: 
 114: class MonteCarloEngine:
 115:     """Geometric Brownian Motion price path simulator with antithetic variates.
 116: 
 117:     Generates terminal price distributions for short-term horizons (hours to days).
 118:     Uses antithetic variates for 50-75% variance reduction at zero extra cost.
 119: 
 120:     Args:
 121:         price: Current asset price (USD).
 122:         volatility: Annualized volatility (decimal, e.g., 0.20 = 20%).
 123:         drift: Annualized drift (decimal). Use drift_from_probability() to derive.
 124:         horizon_days: Simulation horizon in days (supports fractional, e.g., 0.125 = 3h).
 125:         n_paths: Number of simulated terminal prices to generate.
 126:         seed: Random seed for reproducibility.
 127:     """
 128: 
 129:     def __init__(self, price: float, volatility: float, drift: float = 0.0,
 130:                  horizon_days: float = 1.0, n_paths: int = DEFAULT_N_PATHS,
 131:                  seed: int | None = None):
 132:         self.price = price
 133:         self.volatility = max(volatility, MIN_VOLATILITY)
 134:         self.drift = drift
 135:         self.horizon_days = horizon_days
 136:         self.n_paths = n_paths
 137:         self.seed = seed
 138:         self._terminal_prices = None
 139: 
 140:     def simulate_paths(self) -> np.ndarray:

-- lines 141-280 --
 141:         """Generate terminal prices via GBM with antithetic variates.
 142: 
 143:         Formula: S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
 144: 
 145:         Antithetic variates: for each random draw Z, we also compute the
 146:         path with -Z. This creates negative correlation between paired
 147:         estimates, reducing variance of the mean estimator.
 148: 
 149:         Returns:
 150:             1D array of terminal prices, shape (n_paths,).
 151:         """
 152:         rng = np.random.default_rng(self.seed)
 153: 
 154:         T = self.horizon_days / 252.0  # Convert to years
 155:         sigma = self.volatility
 156:         mu = self.drift
 157: 
 158:         # Draw half the paths, use antithetic for the other half
 159:         n_half = self.n_paths // 2
 160:         Z = rng.standard_normal(n_half)
 161: 
 162:         # Deterministic component
 163:         drift_term = (mu - 0.5 * sigma**2) * T
 164:         vol_term = sigma * math.sqrt(T)
 165: 
 166:         # Original paths
 167:         log_returns_pos = drift_term + vol_term * Z
 168:         # Antithetic paths (negate Z)
 169:         log_returns_neg = drift_term + vol_term * (-Z)
 170: 
 171:         # Combine
 172:         log_returns = np.concatenate([log_returns_pos, log_returns_neg])
 173:         terminal_prices = self.price * np.exp(log_returns)
 174: 
 175:         # If n_paths is odd, add one more path
 176:         if self.n_paths % 2 == 1:
 177:             extra_Z = rng.standard_normal(1)
 178:             extra_price = self.price * np.exp(drift_term + vol_term * extra_Z)
 179:             terminal_prices = np.concatenate([terminal_prices, extra_price])
 180: 
 181:         self._terminal_prices = terminal_prices
 182:         return terminal_prices
 183: 
 184:     def _ensure_simulated(self):
 185:         """Run simulation if not already done."""
 186:         if self._terminal_prices is None:
 187:             self.simulate_paths()
 188: 
 189:     def price_quantiles(self, percentiles: list[int] | None = None) -> dict:
 190:         """Extract price quantile bands from simulated distribution.
 191: 
 192:         Args:
 193:             percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95]).
 194: 
 195:         Returns:
 196:             Dict mapping percentile → price (e.g., {5: 87.2, 50: 90.1, 95: 93.5}).
 197:         """
 198:         self._ensure_simulated()
 199:         if percentiles is None:
 200:             percentiles = [5, 25, 50, 75, 95]
 201: 
 202:         values = np.percentile(self._terminal_prices, percentiles)
 203:         return {p: round(float(v), 2) for p, v in zip(percentiles, values)}
 204: 
 205:     def probability_below(self, threshold: float) -> float:
 206:         """Compute P(terminal price < threshold).
 207: 
 208:         Useful for stop-loss probability: "what's the chance price hits my stop?"
 209: 
 210:         Args:
 211:             threshold: Price level to test.
 212: 
 213:         Returns:
 214:             Probability (0.0-1.0).
 215:         """
 216:         self._ensure_simulated()
 217:         if threshold <= 0:
 218:             return 0.0
 219:         return float(np.mean(self._terminal_prices < threshold))
 220: 
 221:     def probability_above(self, threshold: float) -> float:
 222:         """Compute P(terminal price > threshold).
 223: 
 224:         Useful for profit target probability.
 225: 
 226:         Args:
 227:             threshold: Price level to test.
 228: 
 229:         Returns:
 230:             Probability (0.0-1.0).
 231:         """
 232:         self._ensure_simulated()
 233:         if threshold <= 0:
 234:             return 1.0
 235:         return float(np.mean(self._terminal_prices > threshold))
 236: 
 237:     def expected_return(self) -> dict:
 238:         """Compute return distribution statistics.
 239: 
 240:         Returns:
 241:             Dict with mean_pct, std_pct, skew of the return distribution.
 242:         """
 243:         self._ensure_simulated()
 244:         returns = (self._terminal_prices - self.price) / self.price * 100.0
 245: 
 246:         mean_pct = float(np.mean(returns))
 247:         std_pct = float(np.std(returns))
 248: 
 249:         # Skewness
 250:         if std_pct > 0:
 251:             skew = float(np.mean(((returns - mean_pct) / std_pct) ** 3))
 252:         else:
 253:             skew = 0.0
 254: 
 255:         return {
 256:             "mean_pct": round(mean_pct, 2),
 257:             "std_pct": round(std_pct, 2),
 258:             "skew": round(skew, 2),
 259:         }
 260: 
 261: 
 262: # ---------------------------------------------------------------------------
 263: # Convenience: simulate a single ticker from agent_summary data
 264: # ---------------------------------------------------------------------------
 265: 
 266: def simulate_ticker(ticker: str, agent_summary: dict,
 267:                     n_paths: int = DEFAULT_N_PATHS,
 268:                     horizons: list[int] | None = None,
 269:                     seed: int | None = None) -> dict | None:
 270:     """Simulate price distribution for a ticker using agent_summary data.
 271: 
 272:     Extracts price, ATR volatility, and directional probability from the
 273:     signal data, then runs GBM simulation at each horizon.
 274: 
 275:     Args:
 276:         ticker: Instrument ticker (e.g., "BTC-USD").
 277:         agent_summary: Full agent_summary dict (or compact version).
 278:         n_paths: Number of MC paths per horizon.
 279:         horizons: List of horizon days (default: [1, 3]).
 280:         seed: Random seed.

-- lines 281-422 --
 281: 
 282:     Returns:
 283:         Dict with price bands, stop probability, expected return per horizon.
 284:         None if ticker not found in summary.
 285:     """
 286:     if horizons is None:
 287:         horizons = DEFAULT_HORIZONS
 288: 
 289:     signals = agent_summary.get("signals", {})
 290:     ticker_data = signals.get(ticker)
 291:     if not ticker_data:
 292:         return None
 293: 
 294:     price = ticker_data.get("price_usd", 0)
 295:     if price <= 0:
 296:         return None
 297: 
 298:     extra = ticker_data.get("extra", {})
 299:     atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct") or _atr_default_for_ticker(ticker)
 300: 
 301:     # Get volatility from ATR
 302:     vol = volatility_from_atr(atr_pct)
 303: 
 304:     # Get directional probability for drift
 305:     # Try to use existing probability computation if available
 306:     p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
 307:     drift = drift_from_probability(p_up, vol)
 308: 
 309:     # Compute ATR-based stop level (2x ATR below entry/current)
 310:     stop_price = price * (1 - 2 * atr_pct / 100)
 311: 
 312:     result = {}
 313: 
 314:     for h in horizons:
 315:         h_key = f"{h}d" if h >= 1 else f"{int(h * 24)}h"
 316: 
 317:         mc = MonteCarloEngine(
 318:             price=price,
 319:             volatility=vol,
 320:             drift=drift,
 321:             horizon_days=h,
 322:             n_paths=n_paths,
 323:             seed=seed,
 324:         )
 325:         mc.simulate_paths()
 326: 
 327:         result[f"price_bands_{h_key}"] = mc.price_quantiles()
 328:         result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)
 329:         result[f"expected_return_{h_key}"] = mc.expected_return()
 330: 
 331:     # Add metadata
 332:     result["price_usd"] = price
 333:     result["atr_pct"] = round(atr_pct, 2)
 334:     result["volatility_annual"] = round(vol, 3)
 335:     result["p_up"] = round(p_up, 3)
 336:     result["drift_annual"] = round(drift, 4)
 337:     result["stop_price"] = round(stop_price, 2)
 338: 
 339:     return result
 340: 
 341: 
 342: def _get_directional_probability(ticker: str, ticker_data: dict,
 343:                                   agent_summary: dict) -> float:
 344:     """Extract directional probability from available data.
 345: 
 346:     Priority:
 347:     1. Pre-computed focus_probabilities (from ticker_accuracy.py)
 348:     2. Weighted confidence + action from signal engine
 349:     3. Fallback: 0.5 (no edge)
 350:     """
 351:     # 1. Check focus_probabilities in summary
 352:     focus_probs = agent_summary.get("focus_probabilities", {})
 353:     ticker_probs = focus_probs.get(ticker, {})
 354:     prob_1d = ticker_probs.get("1d", {})
 355:     if prob_1d and "probability" in prob_1d:
 356:         return prob_1d["probability"]
 357: 
 358:     # 2. Derive from weighted confidence + action
 359:     extra = ticker_data.get("extra", {})
 360:     action = extra.get("_weighted_action") or ticker_data.get("action", "HOLD")
 361:     conf = extra.get("_weighted_confidence") or ticker_data.get("weighted_confidence", 0.5)
 362: 
 363:     if action == "BUY":
 364:         return 0.5 + conf * 0.3  # Scale to 0.5-0.8 range
 365:     elif action == "SELL":
 366:         return 0.5 - conf * 0.3  # Scale to 0.2-0.5 range
 367:     else:
 368:         return 0.5  # HOLD = no directional edge
 369: 
 370: 
 371: # ---------------------------------------------------------------------------
 372: # Batch simulation for all interesting tickers
 373: # ---------------------------------------------------------------------------
 374: 
 375: def simulate_all(agent_summary: dict, tickers: list[str] | None = None,
 376:                  n_paths: int = DEFAULT_N_PATHS, seed: int | None = None) -> dict:
 377:     """Run MC simulation for multiple tickers.
 378: 
 379:     Args:
 380:         agent_summary: Full agent summary dict.
 381:         tickers: List of tickers to simulate. If None, uses held positions + focus tickers.
 382:         n_paths: Paths per ticker per horizon.
 383:         seed: Base seed (incremented per ticker for independence).
 384: 
 385:     Returns:
 386:         Dict mapping ticker → simulation results.
 387:     """
 388:     if tickers is None:
 389:         tickers = _interesting_tickers(agent_summary)
 390: 
 391:     results = {}
 392:     for i, ticker in enumerate(tickers):
 393:         try:
 394:             # 2026-04-22: guard seed=None. reporting.py calls simulate_all
 395:             # without a seed; `seed + i` was crashing every cycle with
 396:             # TypeError and silently disabling MC risk sim for all tickers.
 397:             ticker_seed = None if seed is None else seed + i
 398:             result = simulate_ticker(ticker, agent_summary,
 399:                                      n_paths=n_paths, seed=ticker_seed)
 400:             if result:
 401:                 results[ticker] = result
 402:         except Exception:
 403:             logger.warning("MC simulation failed for %s", ticker, exc_info=True)
 404: 
 405:     return results
 406: 
 407: 
 408: def _interesting_tickers(agent_summary: dict) -> list[str]:
 409:     """Determine which tickers to simulate (held + focus + signaling)."""
 410:     tickers = set()
 411: 
 412:     # Focus tickers (from config, surfaced in summary)
 413:     for ft in agent_summary.get("focus_tickers", ["XAG-USD", "BTC-USD"]):
 414:         tickers.add(ft)
 415: 
 416:     # Tickers with active consensus (BUY or SELL)
 417:     for ticker, data in agent_summary.get("signals", {}).items():
 418:         action = data.get("action", "HOLD")
 419:         if action in ("BUY", "SELL"):
 420:             tickers.add(ticker)
 421: 
 422:     return sorted(tickers)


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p=Path('portfolio/monte_carlo_risk.py')
lines=p.read_text().splitlines()
for start,end in [(1,180),(181,360),(361,515)]:
    print(f'-- lines {start}-{end} --')
    for i in range(start,end+1):
        print(f\"{i:4}: {lines[i-1]}\")
    print()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 641ms:
-- lines 1-180 --
   1: """Portfolio VaR with t-copula correlated simulation.
   2: 
   3: Computes Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and
   4: joint drawdown probabilities for multi-position portfolios using a Student-t
   5: copula for tail dependence.
   6: 
   7: The t-copula captures the empirical fact that assets crash together more
   8: often than a Gaussian copula predicts (tail dependence lambda ~0.18 at
   9: df=4 vs. 0 for Gaussian).
  10: 
  11: Usage:
  12:     from portfolio.monte_carlo_risk import compute_portfolio_var
  13:     result = compute_portfolio_var(portfolio_state, agent_summary)
  14:     # result = {
  15:     #   "var_95_usd": -1234.56,
  16:     #   "cvar_95_usd": -1567.89,
  17:     #   "var_99_usd": -2345.67,
  18:     #   "total_exposure_usd": 50000.0,
  19:     #   ...
  20:     # }
  21: """
  22: 
  23: import logging
  24: import math
  25: 
  26: import numpy as np
  27: from scipy.stats import norm
  28: from scipy.stats import t as t_dist
  29: 
  30: from portfolio.monte_carlo import (
  31:     MIN_VOLATILITY,
  32:     drift_from_probability,
  33:     volatility_from_atr,
  34: )
  35: 
  36: logger = logging.getLogger("portfolio.monte_carlo_risk")
  37: 
  38: DEFAULT_DF = 4        # Degrees of freedom for t-copula (4 = moderate fat tails)
  39: DEFAULT_N_PATHS = 10_000
  40: 
  41: 
  42: # ---------------------------------------------------------------------------
  43: # Correlation matrix estimation
  44: # ---------------------------------------------------------------------------
  45: 
  46: def estimate_correlation_matrix(
  47:     returns: dict[str, np.ndarray],
  48:     tickers: list[str],
  49: ) -> np.ndarray:
  50:     """Estimate correlation matrix from historical return series.
  51: 
  52:     For tickers with insufficient data, defaults to zero correlation
  53:     (independent assumption) which is conservative for VaR.
  54: 
  55:     Args:
  56:         returns: Dict mapping ticker → array of log-returns.
  57:         tickers: Ordered list of tickers (defines matrix row/column order).
  58: 
  59:     Returns:
  60:         Correlation matrix, shape (n, n), guaranteed symmetric and PSD.
  61:     """
  62:     n = len(tickers)
  63:     if n == 0:
  64:         return np.array([]).reshape(0, 0)
  65: 
  66:     corr = np.eye(n)
  67: 
  68:     # Build return matrix for tickers with data
  69:     for i in range(n):
  70:         for j in range(i + 1, n):
  71:             ri = returns.get(tickers[i])
  72:             rj = returns.get(tickers[j])
  73:             if ri is not None and rj is not None:
  74:                 # Align lengths (use shorter)
  75:                 min_len = min(len(ri), len(rj))
  76:                 if min_len >= 20:  # Need at least 20 observations
  77:                     c = np.corrcoef(ri[:min_len], rj[:min_len])[0, 1]
  78:                     if np.isfinite(c):
  79:                         corr[i, j] = c
  80:                         corr[j, i] = c
  81: 
  82:     # Ensure PSD via eigenvalue clipping (Higham nearest PSD)
  83:     corr = _nearest_psd(corr)
  84:     return corr
  85: 
  86: 
  87: def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
  88:     """Project matrix to nearest positive semi-definite correlation matrix.
  89: 
  90:     Uses eigenvalue clipping: set negative eigenvalues to a small positive
  91:     value, then rescale diagonal to 1.
  92:     """
  93:     eigenvalues, eigenvectors = np.linalg.eigh(matrix)
  94:     # Clip negative eigenvalues
  95:     eigenvalues = np.maximum(eigenvalues, 1e-8)
  96:     # Reconstruct
  97:     result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
  98:     # Rescale to correlation matrix (diagonal = 1)
  99:     d = np.sqrt(np.diag(result))
 100:     d[d == 0] = 1.0
 101:     result = result / np.outer(d, d)
 102:     # Enforce exact symmetry
 103:     result = (result + result.T) / 2
 104:     np.fill_diagonal(result, 1.0)
 105:     return result
 106: 
 107: 
 108: # ---------------------------------------------------------------------------
 109: # Known correlation pairs (fallback when no historical data)
 110: # ---------------------------------------------------------------------------
 111: 
 112: # From risk_management.py — approximate correlation strengths
 113: CORRELATION_PRIORS = {
 114:     ("BTC-USD", "ETH-USD"): 0.75,
 115:     ("XAG-USD", "XAU-USD"): 0.85,
 116:     ("NVDA", "AMD"): 0.70,
 117:     ("NVDA", "AVGO"): 0.65,
 118:     ("NVDA", "TSM"): 0.60,
 119:     ("AMD", "AVGO"): 0.60,
 120:     ("AMD", "TSM"): 0.55,
 121:     ("AVGO", "TSM"): 0.55,
 122:     ("GOOGL", "META"): 0.65,
 123:     ("GOOGL", "AMZN"): 0.60,
 124:     ("META", "AMZN"): 0.60,
 125:     ("AAPL", "GOOGL"): 0.55,
 126:     ("AAPL", "META"): 0.50,
 127:     ("AAPL", "AMZN"): 0.55,
 128: }
 129: 
 130: 
 131: def _get_prior_correlation(ticker_a: str, ticker_b: str) -> float:
 132:     """Look up prior correlation for a pair (order-independent)."""
 133:     return CORRELATION_PRIORS.get(
 134:         (ticker_a, ticker_b),
 135:         CORRELATION_PRIORS.get((ticker_b, ticker_a), 0.0),
 136:     )
 137: 
 138: 
 139: def build_correlation_matrix(
 140:     tickers: list[str],
 141:     historical_returns: dict[str, np.ndarray] | None = None,
 142: ) -> np.ndarray:
 143:     """Build correlation matrix using historical data with prior fallback.
 144: 
 145:     If historical returns are available and sufficient, uses empirical
 146:     estimates. Otherwise falls back to hardcoded priors from CORRELATED_PAIRS.
 147: 
 148:     Args:
 149:         tickers: Ordered list of tickers.
 150:         historical_returns: Optional dict of ticker → log-return arrays.
 151: 
 152:     Returns:
 153:         Correlation matrix (n x n).
 154:     """
 155:     n = len(tickers)
 156:     if n == 0:
 157:         return np.array([]).reshape(0, 0)
 158: 
 159:     if historical_returns:
 160:         # Check if we have sufficient data (>= 30 observations per ticker)
 161:         has_data = sum(
 162:             1 for t in tickers
 163:             if t in historical_returns and len(historical_returns[t]) >= 30
 164:         )
 165:         if has_data >= 2:
 166:             return estimate_correlation_matrix(historical_returns, tickers)
 167: 
 168:     # Fallback: use priors
 169:     corr = np.eye(n)
 170:     for i in range(n):
 171:         for j in range(i + 1, n):
 172:             c = _get_prior_correlation(tickers[i], tickers[j])
 173:             corr[i, j] = c
 174:             corr[j, i] = c
 175: 
 176:     return _nearest_psd(corr)
 177: 
 178: 
 179: # ---------------------------------------------------------------------------
 180: # Portfolio risk simulator

-- lines 181-360 --
 181: # ---------------------------------------------------------------------------
 182: 
 183: class PortfolioRiskSimulator:
 184:     """Multi-position portfolio risk simulator using Student-t copula.
 185: 
 186:     Generates correlated terminal returns for all positions jointly,
 187:     then computes portfolio-level P&L, VaR, and CVaR.
 188: 
 189:     The t-copula with df=4 captures tail dependence (assets crash together)
 190:     that a Gaussian copula misses entirely.
 191: 
 192:     Args:
 193:         positions: Dict mapping ticker → {shares, price_usd, volatility, drift}.
 194:         correlation_matrix: Correlation matrix for position tickers.
 195:         horizon_days: Simulation horizon.
 196:         n_paths: Number of simulation paths.
 197:         df: Degrees of freedom for t-copula (lower = fatter tails).
 198:         seed: Random seed.
 199:     """
 200: 
 201:     def __init__(
 202:         self,
 203:         positions: dict,
 204:         correlation_matrix: np.ndarray,
 205:         horizon_days: float = 1.0,
 206:         n_paths: int = DEFAULT_N_PATHS,
 207:         df: int = DEFAULT_DF,
 208:         seed: int | None = None,
 209:     ):
 210:         # Filter out zero-share positions
 211:         self._tickers = [t for t, p in positions.items() if p.get("shares", 0) != 0]
 212:         self._positions = {t: positions[t] for t in self._tickers}
 213:         self._n_assets = len(self._tickers)
 214: 
 215:         # If we filtered tickers, extract the sub-matrix
 216:         if self._n_assets > 0 and correlation_matrix.size > 0:
 217:             all_tickers = list(positions.keys())
 218:             indices = [all_tickers.index(t) for t in self._tickers]
 219:             self._corr = correlation_matrix[np.ix_(indices, indices)]
 220:         else:
 221:             self._corr = np.array([]).reshape(0, 0)
 222: 
 223:         self.horizon_days = horizon_days
 224:         self.n_paths = n_paths
 225:         self.df = df
 226:         self.seed = seed
 227: 
 228:         self._returns = None  # (n_paths, n_assets) log-returns
 229:         self._pnl = None      # (n_paths,) portfolio P&L in USD
 230: 
 231:     def simulate_correlated_returns(self) -> np.ndarray:
 232:         """Generate correlated returns using t-copula.
 233: 
 234:         Algorithm:
 235:         1. Cholesky decompose correlation matrix: L = chol(R)
 236:         2. Generate independent standard normals: Z ~ N(0, I)
 237:         3. Correlate: W = Z @ L^T
 238:         4. Generate chi-squared: S ~ chi2(df)
 239:         5. Scale to t: T = W * sqrt(df / S) → multivariate t
 240:         6. Transform marginals: apply per-asset drift and volatility
 241: 
 242:         Returns:
 243:             Array of shape (n_paths, n_assets) with log-returns.
 244:         """
 245:         if self._n_assets == 0:
 246:             self._returns = np.empty((self.n_paths, 0))
 247:             return self._returns
 248: 
 249:         rng = np.random.default_rng(self.seed)
 250:         T = self.horizon_days / 252.0
 251: 
 252:         # Step 1: Cholesky decomposition
 253:         try:
 254:             L = np.linalg.cholesky(self._corr)
 255:         except np.linalg.LinAlgError:
 256:             logger.warning(
 257:                 "Cholesky decomposition failed for %d-asset correlation matrix — "
 258:                 "falling back to nearest PSD projection",
 259:                 self._n_assets,
 260:             )
 261:             L = np.linalg.cholesky(_nearest_psd(self._corr))
 262: 
 263:         # Step 2: Independent standard normals
 264:         Z = rng.standard_normal((self.n_paths, self._n_assets))
 265: 
 266:         # Step 3: Correlate
 267:         W = Z @ L.T
 268: 
 269:         # Step 4: Chi-squared scaling for t-distribution
 270:         S = rng.chisquare(self.df, size=self.n_paths)
 271: 
 272:         # Step 5: Scale to multivariate t
 273:         # T_i = W_i * sqrt(df / S) for each path
 274:         scale = np.sqrt(self.df / S)[:, np.newaxis]
 275:         T_samples = W * scale
 276: 
 277:         # Step 6: Transform to uniform via t CDF, then to Gaussian marginals
 278:         # U = F_t(T; df) → uniform on [0,1]
 279:         U = t_dist.cdf(T_samples, df=self.df)
 280: 
 281:         # Transform each marginal to GBM log-return
 282:         # C9 FIX: Use norm.ppf (Gaussian inverse CDF), NOT t_dist.ppf.
 283:         # Using t_dist.ppf(t_dist.cdf(x, df), df) is an identity transform —
 284:         # it round-trips back to the same t-distributed samples, inflating
 285:         # variance by ~sqrt(df/(df-2)) ≈ sqrt(2) at df=4. The correct
 286:         # t-copula + GBM approach is: t-copula for dependence structure,
 287:         # Gaussian marginals for the GBM diffusion term.
 288:         returns = np.empty_like(U)
 289:         for i, ticker in enumerate(self._tickers):
 290:             pos = self._positions[ticker]
 291:             sigma = max(pos["volatility"], MIN_VOLATILITY)
 292:             mu = pos.get("drift", 0.0)
 293: 
 294:             # Inverse normal CDF to get standard normal quantiles
 295:             Z_marginal = norm.ppf(U[:, i])
 296: 
 297:             # GBM log-return: (mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z
 298:             drift_term = (mu - 0.5 * sigma**2) * T
 299:             vol_term = sigma * math.sqrt(T)
 300:             returns[:, i] = drift_term + vol_term * Z_marginal
 301: 
 302:         self._returns = returns
 303:         return returns
 304: 
 305:     def _ensure_simulated(self):
 306:         """Run simulation if not already done."""
 307:         if self._returns is None:
 308:             self.simulate_correlated_returns()
 309: 
 310:     def portfolio_pnl(self) -> np.ndarray:
 311:         """Compute portfolio P&L distribution in USD.
 312: 
 313:         P&L = sum over assets of (shares * price * (exp(log_return) - 1))
 314: 
 315:         Returns:
 316:             1D array of portfolio P&L values, shape (n_paths,).
 317:         """
 318:         self._ensure_simulated()
 319: 
 320:         if self._n_assets == 0:
 321:             self._pnl = np.zeros(self.n_paths)
 322:             return self._pnl
 323: 
 324:         pnl = np.zeros(self.n_paths)
 325:         for i, ticker in enumerate(self._tickers):
 326:             pos = self._positions[ticker]
 327:             shares = pos["shares"]
 328:             price = pos["price_usd"]
 329:             # P&L per path = shares * price * (exp(return) - 1)
 330:             pnl += shares * price * (np.exp(self._returns[:, i]) - 1)
 331: 
 332:         self._pnl = pnl
 333:         return pnl
 334: 
 335:     def _ensure_pnl(self):
 336:         """Compute P&L if not already done."""
 337:         if self._pnl is None:
 338:             self.portfolio_pnl()
 339: 
 340:     def var(self, confidence: float = 0.95) -> float:
 341:         """Compute Value-at-Risk at given confidence level.
 342: 
 343:         VaR is the loss at the (1-confidence) percentile of the P&L distribution.
 344:         Returns a negative number (loss).
 345: 
 346:         Args:
 347:             confidence: Confidence level (0.95 or 0.99 typically).
 348: 
 349:         Returns:
 350:             VaR in USD (negative = loss).
 351:         """
 352:         self._ensure_pnl()
 353:         if self._n_assets == 0:
 354:             return 0.0
 355:         percentile = (1 - confidence) * 100
 356:         return float(np.percentile(self._pnl, percentile))
 357: 
 358:     def cvar(self, confidence: float = 0.95) -> float:
 359:         """Compute Conditional VaR (Expected Shortfall) at given confidence.
 360: 

-- lines 361-515 --
 361:         CVaR is the average loss in the worst (1-confidence) scenarios.
 362:         Always worse than VaR (further from zero).
 363: 
 364:         Args:
 365:             confidence: Confidence level.
 366: 
 367:         Returns:
 368:             CVaR in USD (negative = loss).
 369:         """
 370:         self._ensure_pnl()
 371:         if self._n_assets == 0:
 372:             return 0.0
 373:         var_threshold = self.var(confidence)
 374:         tail = self._pnl[self._pnl <= var_threshold]
 375:         if len(tail) == 0:
 376:             return var_threshold
 377:         return float(np.mean(tail))
 378: 
 379:     def drawdown_probability(self, threshold_pct: float) -> float:
 380:         """Probability of portfolio losing more than threshold_pct.
 381: 
 382:         Args:
 383:             threshold_pct: Loss threshold as percentage (e.g., 5.0 = 5%).
 384: 
 385:         Returns:
 386:             Probability (0.0-1.0) of exceeding the loss threshold.
 387:         """
 388:         self._ensure_pnl()
 389:         if self._n_assets == 0:
 390:             return 0.0
 391: 
 392:         total_value = sum(
 393:             p["shares"] * p["price_usd"] for p in self._positions.values()
 394:         )
 395:         if total_value <= 0:
 396:             return 0.0
 397: 
 398:         loss_threshold = -total_value * threshold_pct / 100.0
 399:         return float(np.mean(self._pnl < loss_threshold))
 400: 
 401: 
 402: # ---------------------------------------------------------------------------
 403: # Convenience: compute portfolio VaR from system data structures
 404: # ---------------------------------------------------------------------------
 405: 
 406: def compute_portfolio_var(
 407:     portfolio_state: dict,
 408:     agent_summary: dict,
 409:     n_paths: int = DEFAULT_N_PATHS,
 410:     historical_returns: dict[str, np.ndarray] | None = None,
 411:     seed: int | None = None,
 412: ) -> dict:
 413:     """Compute portfolio VaR from portfolio state and agent summary.
 414: 
 415:     Extracts held positions, current prices, volatilities, and directional
 416:     probabilities from the system data, then runs t-copula simulation.
 417: 
 418:     Args:
 419:         portfolio_state: Portfolio state dict (holdings, cash_sek, etc.).
 420:         agent_summary: Agent summary with signals, fx_rate.
 421:         n_paths: Number of MC paths.
 422:         historical_returns: Optional empirical return series for correlation.
 423:         seed: Random seed.
 424: 
 425:     Returns:
 426:         Dict with VaR metrics in both USD and SEK.
 427:     """
 428:     holdings = portfolio_state.get("holdings", {})
 429:     signals = agent_summary.get("signals", {})
 430:     fx_rate = agent_summary.get("fx_rate", 10.0)
 431: 
 432:     # Build positions dict
 433:     positions = {}
 434:     tickers = []
 435: 
 436:     for ticker, holding in holdings.items():
 437:         shares = holding.get("shares", 0)
 438:         if shares <= 0:
 439:             continue
 440: 
 441:         ticker_data = signals.get(ticker, {})
 442:         price = ticker_data.get("price_usd", 0)
 443:         if price <= 0:
 444:             continue
 445: 
 446:         extra = ticker_data.get("extra", {})
 447:         atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
 448:         vol = volatility_from_atr(atr_pct)
 449: 
 450:         # Get directional probability for drift
 451:         from portfolio.monte_carlo import _get_directional_probability
 452:         p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
 453:         drift = drift_from_probability(p_up, vol)
 454: 
 455:         tickers.append(ticker)
 456:         positions[ticker] = {
 457:             "shares": shares,
 458:             "price_usd": price,
 459:             "volatility": vol,
 460:             "drift": drift,
 461:         }
 462: 
 463:     if not tickers:
 464:         return {
 465:             "var_95_usd": 0.0,
 466:             "var_99_usd": 0.0,
 467:             "cvar_95_usd": 0.0,
 468:             "cvar_99_usd": 0.0,
 469:             "var_95_sek": 0.0,
 470:             "var_99_sek": 0.0,
 471:             "cvar_95_sek": 0.0,
 472:             "cvar_99_sek": 0.0,
 473:             "total_exposure_usd": 0.0,
 474:             "total_exposure_sek": 0.0,
 475:             "n_positions": 0,
 476:             "drawdown_1pct_prob": 0.0,
 477:             "drawdown_5pct_prob": 0.0,
 478:         }
 479: 
 480:     # Build correlation matrix
 481:     corr = build_correlation_matrix(tickers, historical_returns)
 482: 
 483:     # Run simulation
 484:     sim = PortfolioRiskSimulator(
 485:         positions=positions,
 486:         correlation_matrix=corr,
 487:         horizon_days=1,
 488:         n_paths=n_paths,
 489:         df=DEFAULT_DF,
 490:         seed=seed,
 491:     )
 492:     sim.simulate_correlated_returns()
 493: 
 494:     total_exposure = sum(p["shares"] * p["price_usd"] for p in positions.values())
 495: 
 496:     var95 = sim.var(0.95)
 497:     var99 = sim.var(0.99)
 498:     cvar95 = sim.cvar(0.95)
 499:     cvar99 = sim.cvar(0.99)
 500: 
 501:     return {
 502:         "var_95_usd": round(var95, 2),
 503:         "var_99_usd": round(var99, 2),
 504:         "cvar_95_usd": round(cvar95, 2),
 505:         "cvar_99_usd": round(cvar99, 2),
 506:         "var_95_sek": round(var95 * fx_rate, 2),
 507:         "var_99_sek": round(var99 * fx_rate, 2),
 508:         "cvar_95_sek": round(cvar95 * fx_rate, 2),
 509:         "cvar_99_sek": round(cvar99 * fx_rate, 2),
 510:         "total_exposure_usd": round(total_exposure, 2),
 511:         "total_exposure_sek": round(total_exposure * fx_rate, 2),
 512:         "n_positions": len(tickers),
 513:         "drawdown_1pct_prob": round(sim.drawdown_probability(1.0), 3),
 514:         "drawdown_5pct_prob": round(sim.drawdown_probability(5.0), 3),
 515:     }


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for name in ['portfolio_validator.py','trade_guards.py']:
    p=Path('portfolio',name)
    lines=p.read_text().splitlines()
    print(f'===== {name} =====')
    for start in range(1,len(lines)+1,140):
        end=min(start+139,len(lines))
        print(f'-- lines {start}-{end} --')
        for i in range(start,end+1):
            print(f\"{i:4}: {lines[i-1]}\")
        print()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 779ms:
===== portfolio_validator.py =====
-- lines 1-140 --
   1: """Portfolio state validation utilities.
   2: 
   3: Validates portfolio state files for data integrity, ensuring that cash,
   4: holdings, fees, and transaction records are all internally consistent.
   5: """
   6: 
   7: import pathlib
   8: from collections import defaultdict
   9: 
  10: from portfolio.file_utils import load_json
  11: 
  12: 
  13: def validate_portfolio(portfolio: dict) -> list[str]:
  14:     """Validate portfolio state integrity.
  15: 
  16:     Performs comprehensive checks on a portfolio state dict to ensure
  17:     all invariants hold. Returns a list of error messages -- an empty
  18:     list means the portfolio is valid.
  19: 
  20:     Checks:
  21:     1. Cash is non-negative
  22:     2. All shares are non-negative
  23:     3. Cash reconciliation: initial - sum(BUY allocs) + sum(SELL net_proceeds) = cash_sek
  24:     4. Holdings shares reconciliation: sum(BUY shares) - sum(SELL shares) per ticker
  25:     5. Fee reconciliation: total_fees_sek = sum(all fee_sek in transactions)
  26:     6. No duplicate timestamps in transactions (same ticker + same timestamp)
  27:     7. Transaction field completeness
  28:     8. Holdings avg_cost_usd consistency
  29: 
  30:     Args:
  31:         portfolio: Full portfolio state dict with keys:
  32:             - cash_sek, holdings, transactions, initial_value_sek, total_fees_sek
  33: 
  34:     Returns:
  35:         list of error message strings (empty = valid).
  36:     """
  37:     errors = []
  38: 
  39:     # --- Extract fields with defaults ---
  40:     cash_sek = portfolio.get("cash_sek")
  41:     holdings = portfolio.get("holdings", {})
  42:     transactions = portfolio.get("transactions", [])
  43:     initial_value = portfolio.get("initial_value_sek", 500_000)
  44:     total_fees_sek = portfolio.get("total_fees_sek")
  45: 
  46:     # --- Check 0: Required fields exist ---
  47:     if cash_sek is None:
  48:         errors.append("Missing required field: cash_sek")
  49:         cash_sek = 0
  50:     if total_fees_sek is None:
  51:         errors.append("Missing or null field: total_fees_sek (should be 0 if no fees)")
  52:         total_fees_sek = 0
  53:     if "initial_value_sek" not in portfolio:
  54:         errors.append("Missing required field: initial_value_sek")
  55: 
  56:     # --- Check 1: Cash is non-negative ---
  57:     if cash_sek < 0:
  58:         errors.append(f"Cash is negative: {cash_sek:.2f} SEK")
  59: 
  60:     # --- Check 2: All shares are non-negative ---
  61:     for ticker, pos in holdings.items():
  62:         shares = pos.get("shares", 0)
  63:         if shares < 0:
  64:             errors.append(f"Negative shares for {ticker}: {shares}")
  65:         avg_cost = pos.get("avg_cost_usd")
  66:         if avg_cost is not None and avg_cost < 0:
  67:             errors.append(f"Negative avg_cost_usd for {ticker}: {avg_cost}")
  68: 
  69:     # --- Check 3: Cash reconciliation ---
  70:     # cash = initial - sum(BUY total_sek) + sum(SELL total_sek)
  71:     # Note: BUY total_sek = full allocation (including fee)
  72:     # SELL total_sek = net proceeds (after fee deducted)
  73:     total_buy_alloc = 0.0
  74:     total_sell_proceeds = 0.0
  75:     for tx in transactions:
  76:         action = tx.get("action", "")
  77:         total_sek = tx.get("total_sek", 0) or 0
  78:         if action == "BUY":
  79:             total_buy_alloc += total_sek
  80:         elif action == "SELL":
  81:             total_sell_proceeds += total_sek
  82: 
  83:     expected_cash = initial_value - total_buy_alloc + total_sell_proceeds
  84:     cash_diff = abs(expected_cash - cash_sek)
  85:     if cash_diff > 1.0:  # Allow 1 SEK tolerance for floating point
  86:         errors.append(
  87:             f"Cash reconciliation failed: expected {expected_cash:.2f} SEK "
  88:             f"(initial {initial_value} - buys {total_buy_alloc:.2f} + sells {total_sell_proceeds:.2f}), "
  89:             f"got {cash_sek:.2f} SEK (diff: {cash_diff:.2f})"
  90:         )
  91: 
  92:     # --- Check 4: Holdings shares reconciliation ---
  93:     # For each ticker: net_shares = sum(BUY shares) - sum(SELL shares)
  94:     ticker_bought = defaultdict(float)
  95:     ticker_sold = defaultdict(float)
  96:     for tx in transactions:
  97:         ticker = tx.get("ticker", "")
  98:         action = tx.get("action", "")
  99:         shares = tx.get("shares", 0) or 0
 100:         if action == "BUY":
 101:             ticker_bought[ticker] += shares
 102:         elif action == "SELL":
 103:             ticker_sold[ticker] += shares
 104: 
 105:     # Check tickers that appear in transactions
 106:     all_tx_tickers = set(ticker_bought.keys()) | set(ticker_sold.keys())
 107:     for ticker in all_tx_tickers:
 108:         expected_shares = ticker_bought[ticker] - ticker_sold[ticker]
 109: 
 110:         # Get actual shares from holdings
 111:         if ticker in holdings:
 112:             actual_shares = holdings[ticker].get("shares", 0)
 113:         else:
 114:             actual_shares = 0
 115: 
 116:         # Compare (with tolerance for floating point and rounding from repeated partial sells)
 117:         share_diff = abs(expected_shares - actual_shares)
 118:         if share_diff > 1e-6:
 119:             if expected_shares <= 1e-9 and actual_shares == 0:
 120:                 # Both effectively zero -- OK (sold all, removed from holdings)
 121:                 continue
 122:             if expected_shares <= 1e-9 and ticker not in holdings:
 123:                 # Fully sold, ticker removed from holdings -- OK
 124:                 continue
 125:             # Tolerance for small remainders from repeated partial sells (e.g.,
 126:             # multiple 50% sells that don't sum exactly to total bought due to
 127:             # floating-point rounding). Allow up to 1% of total bought shares.
 128:             total_bought = ticker_bought[ticker]
 129:             relative_diff = share_diff / total_bought if total_bought > 0 else float("inf")
 130:             if actual_shares == 0 and ticker not in holdings and relative_diff < 0.01:
 131:                 # Small remainder from rounding, ticker removed -- acceptable
 132:                 continue
 133:             errors.append(
 134:                 f"Holdings mismatch for {ticker}: expected {expected_shares:.8f} shares "
 135:                 f"(bought {ticker_bought[ticker]:.8f} - sold {ticker_sold[ticker]:.8f}), "
 136:                 f"got {actual_shares:.8f} in holdings (diff: {share_diff:.8f})"
 137:             )
 138: 
 139:     # Check for holdings tickers not in transactions
 140:     for ticker in holdings:

-- lines 141-280 --
 141:         shares = holdings[ticker].get("shares", 0)
 142:         if shares > 0 and ticker not in all_tx_tickers:
 143:             errors.append(
 144:                 f"Holdings contains {ticker} with {shares} shares but no matching transactions"
 145:             )
 146: 
 147:     # --- Check 5: Fee reconciliation ---
 148:     computed_fees = 0.0
 149:     tx_with_fees = 0
 150:     tx_without_fees = 0
 151:     for tx in transactions:
 152:         fee = tx.get("fee_sek")
 153:         if fee is not None:
 154:             computed_fees += fee
 155:             tx_with_fees += 1
 156:         else:
 157:             tx_without_fees += 1
 158: 
 159:     if tx_without_fees > 0 and tx_with_fees > 0:
 160:         errors.append(
 161:             f"Inconsistent fee tracking: {tx_with_fees} transactions have fee_sek, "
 162:             f"{tx_without_fees} do not"
 163:         )
 164: 
 165:     # Only compare fees if transactions have fee_sek fields
 166:     if tx_with_fees > 0:
 167:         fee_diff = abs(computed_fees - total_fees_sek)
 168:         if fee_diff > 0.01:  # 0.01 SEK tolerance
 169:             errors.append(
 170:                 f"Fee reconciliation failed: sum of transaction fees = {computed_fees:.2f} SEK, "
 171:                 f"total_fees_sek = {total_fees_sek:.2f} SEK (diff: {fee_diff:.2f})"
 172:             )
 173:     elif len(transactions) > 0 and total_fees_sek == 0:
 174:         # Transactions exist but no fee tracking at all -- warn
 175:         errors.append(
 176:             "No fee_sek fields in any transaction and total_fees_sek is 0, "
 177:             "but transactions exist. Fees may not be tracked."
 178:         )
 179: 
 180:     # --- Check 6: No duplicate timestamps per ticker ---
 181:     seen_tx = set()
 182:     for i, tx in enumerate(transactions):
 183:         key = (tx.get("ticker", ""), tx.get("timestamp", ""), tx.get("action", ""))
 184:         if key in seen_tx:
 185:             errors.append(
 186:                 f"Duplicate transaction at index {i}: {key[2]} {key[0]} at {key[1]}"
 187:             )
 188:         seen_tx.add(key)
 189: 
 190:     # --- Check 7: Transaction field completeness ---
 191:     required_tx_fields = ["timestamp", "ticker", "action", "shares", "price_usd",
 192:                           "total_sek", "reason"]
 193: 
 194:     for i, tx in enumerate(transactions):
 195:         for field in required_tx_fields:
 196:             if field not in tx or tx[field] is None:
 197:                 errors.append(f"Transaction {i} missing required field: {field}")
 198: 
 199:         # Validate action value
 200:         action = tx.get("action", "")
 201:         if action not in ("BUY", "SELL"):
 202:             errors.append(f"Transaction {i} has invalid action: '{action}' (expected BUY or SELL)")
 203: 
 204:         # Validate shares > 0
 205:         shares = tx.get("shares", 0)
 206:         if shares is not None and shares <= 0:
 207:             errors.append(f"Transaction {i} has non-positive shares: {shares}")
 208: 
 209:         # Validate total_sek > 0
 210:         total = tx.get("total_sek", 0)
 211:         if total is not None and total <= 0:
 212:             errors.append(f"Transaction {i} has non-positive total_sek: {total}")
 213: 
 214:     # --- Check 8: Holdings avg_cost_usd consistency ---
 215:     # For tickers with multiple BUY transactions, verify avg_cost is plausible
 216:     for ticker, pos in holdings.items():
 217:         shares = pos.get("shares", 0)
 218:         if shares <= 0:
 219:             continue
 220:         avg_cost = pos.get("avg_cost_usd")
 221:         if avg_cost is None:
 222:             errors.append(f"Holdings {ticker} missing avg_cost_usd")
 223:             continue
 224: 
 225:         # Compute weighted average from BUY transactions
 226:         total_cost = 0.0
 227:         total_bought = 0.0
 228:         for tx in transactions:
 229:             if tx.get("ticker") != ticker or tx.get("action") != "BUY":
 230:                 continue
 231:             tx_shares = tx.get("shares", 0) or 0
 232:             tx_price = tx.get("price_usd", 0) or 0
 233:             total_cost += tx_shares * tx_price
 234:             total_bought += tx_shares
 235: 
 236:         if total_bought > 0:
 237:             expected_avg = total_cost / total_bought
 238:             avg_diff_pct = abs(expected_avg - avg_cost) / expected_avg * 100 if expected_avg > 0 else 0
 239:             if avg_diff_pct > 1.0:  # More than 1% off
 240:                 errors.append(
 241:                     f"Holdings {ticker} avg_cost_usd ({avg_cost:.4f}) differs from "
 242:                     f"computed weighted average ({expected_avg:.4f}) by {avg_diff_pct:.2f}%"
 243:                 )
 244: 
 245:     return errors
 246: 
 247: 
 248: def validate_portfolio_file(path: str) -> list[str]:
 249:     """Validate a portfolio state JSON file.
 250: 
 251:     Convenience wrapper that loads the file and runs validate_portfolio().
 252: 
 253:     Args:
 254:         path: Path to the portfolio state JSON file.
 255: 
 256:     Returns:
 257:         list of error message strings (empty = valid).
 258:     """
 259:     # A-PR-3 (2026-04-11): Use file_utils.load_json instead of raw open()+
 260:     # json.load(). The raw path had a TOCTOU race with concurrent saves
 261:     # from portfolio_mgr.atomic_write_json: a partial write could be read
 262:     # mid-flight, producing a JSONDecodeError that propagated as a "valid
 263:     # but malformed" portfolio. load_json() retries on transient JSON
 264:     # decode failures and uses the same atomic-rename window as the writer.
 265:     p = pathlib.Path(path)
 266:     if not p.exists():
 267:         return [f"Portfolio file not found: {path}"]
 268:     portfolio = load_json(p)
 269:     if portfolio is None:
 270:         return [f"Invalid or unreadable JSON in {path}"]
 271: 
 272:     return validate_portfolio(portfolio)
 273: 
 274: 
 275: def validate_all() -> dict[str, list[str]]:
 276:     """Validate both patient and bold portfolio files.
 277: 
 278:     Returns:
 279:         dict with keys "patient" and "bold", each containing a list of errors.
 280:     """

-- lines 281-299 --
 281:     data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
 282:     return {
 283:         "patient": validate_portfolio_file(str(data_dir / "portfolio_state.json")),
 284:         "bold": validate_portfolio_file(str(data_dir / "portfolio_state_bold.json")),
 285:     }
 286: 
 287: 
 288: if __name__ == "__main__":
 289:     results = validate_all()
 290:     for strategy, errs in results.items():
 291:         print(f"\n{'='*60}")
 292:         print(f"  {strategy.upper()} PORTFOLIO VALIDATION")
 293:         print(f"{'='*60}")
 294:         if errs:
 295:             for e in errs:
 296:                 print(f"  ERROR: {e}")
 297:         else:
 298:             print("  VALID - all checks passed")
 299:     print()

===== trade_guards.py =====
-- lines 1-140 --
   1: """Trade guards — overtrading prevention for the trading agent.
   2: 
   3: Three guards:
   4: 1. Per-ticker cooldown: No re-trade on same ticker within N minutes.
   5: 2. Consecutive-loss escalation: After losses, increase cooldown multiplier.
   6: 3. Position rate limit: Max N new positions per time window.
   7: 
   8: State is persisted to data/trade_guard_state.json.
   9: """
  10: 
  11: import logging
  12: import threading
  13: from datetime import UTC, datetime, timedelta
  14: from pathlib import Path
  15: 
  16: from portfolio.file_utils import atomic_write_json, load_json
  17: 
  18: logger = logging.getLogger("portfolio.trade_guards")
  19: 
  20: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
  21: STATE_FILE = DATA_DIR / "trade_guard_state.json"
  22: 
  23: # Defaults
  24: DEFAULT_TICKER_COOLDOWN_MINUTES = 30
  25: DEFAULT_BOLD_POSITION_LIMIT = 1       # max new positions per window
  26: DEFAULT_BOLD_POSITION_WINDOW_H = 4    # hours
  27: DEFAULT_PATIENT_POSITION_LIMIT = 1
  28: DEFAULT_PATIENT_POSITION_WINDOW_H = 8
  29: LOSS_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive_losses -> cooldown multiplier
  30: LOSS_DECAY_HOURS = 24  # halve escalation multiplier every N hours without a trade
  31: 
  32: _state_lock = threading.Lock()
  33: 
  34: 
  35: def _load_state():
  36:     """Load trade guard state from disk."""
  37:     return load_json(str(STATE_FILE), default={
  38:         "ticker_trades": {},
  39:         "consecutive_losses": {"patient": 0, "bold": 0},
  40:         "last_loss_ts": {"patient": None, "bold": None},
  41:         "new_position_timestamps": {"patient": [], "bold": []},
  42:     })
  43: 
  44: 
  45: def _save_state(state):
  46:     """Persist trade guard state to disk."""
  47:     atomic_write_json(STATE_FILE, state)
  48: 
  49: 
  50: def _portfolios_have_transactions():
  51:     """Return True if any portfolio file has at least one recorded transaction.
  52: 
  53:     Used by the C4 sanity check to distinguish "no trades happened yet"
  54:     (quiet startup state) from "trades happened but weren't recorded"
  55:     (broken wiring — real bug).
  56: 
  57:     2026-04-22 follow-up: include warrants portfolio — CLAUDE.md lists it as
  58:     an independent strategy state file, and warrants-only activity would
  59:     have left C4 silent forever.
  60:     """
  61:     for pf_name in (
  62:         "portfolio_state.json",
  63:         "portfolio_state_bold.json",
  64:         "portfolio_state_warrants.json",
  65:     ):
  66:         pf = load_json(str(DATA_DIR / pf_name), default={})
  67:         if pf and pf.get("transactions"):
  68:             return True
  69:     return False
  70: 
  71: 
  72: def _get_cooldown_multiplier(consecutive_losses, last_loss_ts_str=None):
  73:     """Get cooldown multiplier based on consecutive loss count with time decay.
  74: 
  75:     After LOSS_DECAY_HOURS without a new trade, the multiplier halves
  76:     repeatedly (geometric decay). E.g. 8x → 4x after 24h → 2x after 48h → 1x.
  77:     """
  78:     if consecutive_losses >= 4:
  79:         base = LOSS_ESCALATION[4]
  80:     else:
  81:         base = LOSS_ESCALATION.get(consecutive_losses, 1)
  82: 
  83:     if base <= 1 or not last_loss_ts_str:
  84:         return base
  85: 
  86:     # Apply time-based decay
  87:     try:
  88:         last_loss = datetime.fromisoformat(
  89:             last_loss_ts_str.replace("Z", "+00:00")
  90:         )
  91:         if last_loss.tzinfo is None:
  92:             last_loss = last_loss.replace(tzinfo=UTC)
  93:         elapsed_hours = (datetime.now(UTC) - last_loss).total_seconds() / 3600
  94:         if elapsed_hours > LOSS_DECAY_HOURS:
  95:             halvings = int(elapsed_hours // LOSS_DECAY_HOURS)
  96:             base = max(1, base >> halvings)  # bit-shift right = halve
  97:     except (ValueError, TypeError, OverflowError):
  98:         pass
  99: 
 100:     return base
 101: 
 102: 
 103: def check_overtrading_guards(ticker, action, strategy, portfolio, config=None):
 104:     """Check all trade guards for a proposed trade.
 105: 
 106:     Args:
 107:         ticker: Instrument ticker (e.g., "BTC-USD").
 108:         action: "BUY" or "SELL".
 109:         strategy: "patient" or "bold".
 110:         portfolio: Portfolio state dict.
 111:         config: Optional config dict with trade_guards settings.
 112: 
 113:     Returns:
 114:         list of warning dicts, each with:
 115:             - guard: str (guard name)
 116:             - severity: "warning" or "block"
 117:             - message: str
 118:             - details: dict (guard-specific data)
 119:         Empty list means all guards pass.
 120:     """
 121:     cfg = (config or {}).get("trade_guards", {})
 122:     if cfg.get("enabled") is False:
 123:         return []
 124: 
 125:     warnings = []
 126:     with _state_lock:
 127:         state = _load_state()
 128:     now = datetime.now(UTC)
 129: 
 130:     # --- Guard 1: Per-ticker cooldown ---
 131:     base_cooldown = cfg.get("ticker_cooldown_minutes", DEFAULT_TICKER_COOLDOWN_MINUTES)
 132:     consecutive = state.get("consecutive_losses", {}).get(strategy, 0)
 133:     last_loss_ts = state.get("last_loss_ts", {}).get(strategy)
 134:     multiplier = _get_cooldown_multiplier(consecutive, last_loss_ts)
 135:     effective_cooldown = base_cooldown * multiplier
 136: 
 137:     key = f"{strategy}:{ticker}"
 138:     ticker_trades = state.get("ticker_trades", {})
 139:     last_trade_str = ticker_trades.get(key)
 140:     if last_trade_str:

-- lines 141-280 --
 141:         try:
 142:             last_trade = datetime.fromisoformat(last_trade_str)
 143:             # M8: ensure aware datetime before comparison with aware now
 144:             if last_trade.tzinfo is None:
 145:                 last_trade = last_trade.replace(tzinfo=UTC)
 146:             elapsed = (now - last_trade).total_seconds() / 60
 147:             if elapsed < effective_cooldown:
 148:                 remaining = effective_cooldown - elapsed
 149:                 warnings.append({
 150:                     "guard": "ticker_cooldown",
 151:                     "severity": "block",
 152:                     "message": (
 153:                         f"{ticker} traded {elapsed:.0f}m ago by {strategy}. "
 154:                         f"Cooldown: {effective_cooldown:.0f}m (base {base_cooldown}m × {multiplier}x). "
 155:                         f"{remaining:.0f}m remaining."
 156:                     ),
 157:                     "details": {
 158:                         "ticker": ticker,
 159:                         "strategy": strategy,
 160:                         "elapsed_min": round(elapsed, 1),
 161:                         "cooldown_min": effective_cooldown,
 162:                         "multiplier": multiplier,
 163:                         "remaining_min": round(remaining, 1),
 164:                     },
 165:                 })
 166:         except (ValueError, TypeError):
 167:             pass
 168: 
 169:     # --- Guard 2: Consecutive-loss escalation (informational) ---
 170:     if consecutive >= 2:
 171:         base_mult = _get_cooldown_multiplier(consecutive, None)
 172:         warnings.append({
 173:             "guard": "consecutive_losses",
 174:             "severity": "warning",
 175:             "message": (
 176:                 f"{strategy}: {consecutive} consecutive losses. "
 177:                 f"Cooldown multiplier: {multiplier}x"
 178:                 f"{f' (decayed from {base_mult}x)' if multiplier < base_mult else ''}."
 179:             ),
 180:             "details": {
 181:                 "strategy": strategy,
 182:                 "consecutive_losses": consecutive,
 183:                 "multiplier": multiplier,
 184:                 "base_multiplier": base_mult,
 185:                 "decayed": multiplier < base_mult,
 186:             },
 187:         })
 188: 
 189:     # --- Guard 3: Position rate limit (BUY only) ---
 190:     if action == "BUY":
 191:         is_bold = strategy == "bold"
 192:         limit = cfg.get(
 193:             f"{'bold' if is_bold else 'patient'}_position_limit",
 194:             DEFAULT_BOLD_POSITION_LIMIT if is_bold else DEFAULT_PATIENT_POSITION_LIMIT,
 195:         )
 196:         window_h = cfg.get(
 197:             f"{'bold' if is_bold else 'patient'}_position_window_h",
 198:             DEFAULT_BOLD_POSITION_WINDOW_H if is_bold else DEFAULT_PATIENT_POSITION_WINDOW_H,
 199:         )
 200:         cutoff = now - timedelta(hours=window_h)
 201: 
 202:         timestamps = state.get("new_position_timestamps", {}).get(strategy, [])
 203:         recent = []
 204:         for ts_str in timestamps:
 205:             try:
 206:                 ts = datetime.fromisoformat(ts_str)
 207:                 # M8: ensure aware datetime before comparison with aware cutoff
 208:                 if ts.tzinfo is None:
 209:                     ts = ts.replace(tzinfo=UTC)
 210:                 if ts >= cutoff:
 211:                     recent.append(ts)
 212:             except (ValueError, TypeError):
 213:                 continue
 214: 
 215:         if len(recent) >= limit:
 216:             warnings.append({
 217:                 "guard": "position_rate_limit",
 218:                 "severity": "block",
 219:                 "message": (
 220:                     f"{strategy}: {len(recent)} new position(s) in last {window_h}h "
 221:                     f"(limit: {limit})."
 222:                 ),
 223:                 "details": {
 224:                     "strategy": strategy,
 225:                     "recent_count": len(recent),
 226:                     "limit": limit,
 227:                     "window_hours": window_h,
 228:                 },
 229:             })
 230: 
 231:     return warnings
 232: 
 233: 
 234: _wiring_confirmed = False  # process-scoped flag — positive proof for C4
 235: 
 236: 
 237: def record_trade(ticker, direction, strategy, pnl_pct=None, config=None):
 238:     """Record a completed trade for guard tracking.
 239: 
 240:     Call this after executing a trade to update cooldowns and loss streaks.
 241: 
 242:     Args:
 243:         ticker: Instrument ticker.
 244:         direction: "BUY" or "SELL".
 245:         strategy: "patient" or "bold".
 246:         pnl_pct: Realized P&L percentage (for SELL trades). None for BUY.
 247:         config: Optional config dict.
 248:     """
 249:     # 2026-04-22 follow-up: positive-proof wiring check. The previous C4
 250:     # warning was *reactive* — it could only tell you after a trade had
 251:     # already slipped through unguarded. Log INFO once per process the first
 252:     # time this function fires, so operators get explicit confirmation the
 253:     # BUG-219/PR-R4-4 wiring is alive rather than having to infer it from
 254:     # absence-of-warnings.
 255:     global _wiring_confirmed
 256:     if not _wiring_confirmed:
 257:         logger.info(
 258:             "C4: record_trade() wiring confirmed — first call this process "
 259:             "(ticker=%s direction=%s strategy=%s)",
 260:             ticker, direction, strategy,
 261:         )
 262:         _wiring_confirmed = True
 263: 
 264:     with _state_lock:
 265:         state = _load_state()
 266:         now = datetime.now(UTC)
 267:         now_str = now.isoformat()
 268: 
 269:         # Update ticker trade timestamp
 270:         key = f"{strategy}:{ticker}"
 271:         if "ticker_trades" not in state:
 272:             state["ticker_trades"] = {}
 273:         state["ticker_trades"][key] = now_str
 274: 
 275:         # Update consecutive losses on SELL
 276:         if direction == "SELL" and pnl_pct is not None:
 277:             if "consecutive_losses" not in state:
 278:                 state["consecutive_losses"] = {"patient": 0, "bold": 0}
 279:             if "last_loss_ts" not in state:
 280:                 state["last_loss_ts"] = {"patient": None, "bold": None}

-- lines 281-388 --
 281:             if pnl_pct < 0:
 282:                 state["consecutive_losses"][strategy] = (
 283:                     state["consecutive_losses"].get(strategy, 0) + 1
 284:                 )
 285:                 state["last_loss_ts"][strategy] = now_str
 286:             else:
 287:                 state["consecutive_losses"][strategy] = 0
 288:                 state["last_loss_ts"][strategy] = None
 289: 
 290:         # Track new position timestamps (BUY only)
 291:         if direction == "BUY":
 292:             if "new_position_timestamps" not in state:
 293:                 state["new_position_timestamps"] = {"patient": [], "bold": []}
 294:             if strategy not in state["new_position_timestamps"]:
 295:                 state["new_position_timestamps"][strategy] = []
 296:             state["new_position_timestamps"][strategy].append(now_str)
 297: 
 298:             # Prune old timestamps (keep last 24h).
 299:             cutoff = now - timedelta(hours=24)
 300:             pruned = []
 301:             for ts in state["new_position_timestamps"][strategy]:
 302:                 try:
 303:                     dt = datetime.fromisoformat(ts)
 304:                     if dt.tzinfo is None:
 305:                         dt = dt.replace(tzinfo=UTC)
 306:                     if dt >= cutoff:
 307:                         pruned.append(ts)
 308:                 except (ValueError, TypeError):
 309:                     continue
 310:             state["new_position_timestamps"][strategy] = pruned
 311: 
 312:         _save_state(state)
 313: 
 314: 
 315: def get_all_guard_warnings(signals, patient_pf, bold_pf, config=None):
 316:     """Get trade guard warnings for all tickers with BUY/SELL signals.
 317: 
 318:     Args:
 319:         signals: Dict of ticker -> signal data (from agent_summary).
 320:         patient_pf: Patient portfolio state dict.
 321:         bold_pf: Bold portfolio state dict.
 322:         config: Optional config dict.
 323: 
 324:     Returns:
 325:         dict with:
 326:             - warnings: list of warning dicts
 327:             - summary: str (human-readable summary)
 328:     """
 329:     cfg = (config or {}).get("trade_guards", {})
 330:     if cfg.get("enabled") is False:
 331:         return {"warnings": [], "summary": "Trade guards disabled"}
 332: 
 333:     all_warnings = []
 334: 
 335:     for ticker, sig in signals.items():
 336:         action = sig.get("action", "HOLD")
 337:         if action == "HOLD":
 338:             continue
 339: 
 340:         for strategy, portfolio in [("patient", patient_pf), ("bold", bold_pf)]:
 341:             warns = check_overtrading_guards(
 342:                 ticker, action, strategy, portfolio, config
 343:             )
 344:             all_warnings.extend(warns)
 345: 
 346:     summary_parts = []
 347:     if all_warnings:
 348:         by_guard = {}
 349:         for w in all_warnings:
 350:             guard = w["guard"]
 351:             by_guard.setdefault(guard, []).append(w)
 352:         for guard, warns in by_guard.items():
 353:             summary_parts.append(f"{guard}: {len(warns)} warning(s)")
 354: 
 355:     # C4: Detect broken record_trade() wiring.
 356:     # 2026-04-22: original check fired every cycle whenever state was empty,
 357:     # even when no trades had happened yet (portfolios untouched) — noisy and
 358:     # misleading post-BUG-219/PR-R4-4 which wired _record_new_trades().
 359:     # Now only warn when portfolios DO have transactions but guard state is
 360:     # still empty — that's the real signal the wiring is broken.
 361:     with _state_lock:
 362:         state = _load_state()
 363:     if not state.get("ticker_trades") and all_warnings == [] and _portfolios_have_transactions():
 364:         logger.warning(
 365:             "C4: portfolios have transactions but trade_guard_state.json "
 366:             "has no recorded trades — record_trade() wiring appears broken. "
 367:             "Overtrading guards are NON-FUNCTIONAL."
 368:         )
 369: 
 370:     return {
 371:         "warnings": all_warnings,
 372:         "summary": "; ".join(summary_parts) if summary_parts else "All clear",
 373:     }
 374: 
 375: 
 376: def should_block_trade(guard_result):
 377:     """Check if any guard warning has 'block' severity.
 378: 
 379:     ARCH-29: Convenience function for Layer 2 go/no-go decisions.
 380: 
 381:     Args:
 382:         guard_result: Return value from get_all_guard_warnings().
 383: 
 384:     Returns:
 385:         True if any warning has severity="block", False otherwise.
 386:     """
 387:     warnings = guard_result.get("warnings", [])
 388:     return any(w.get("severity") == "block" for w in warnings)


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p=Path('portfolio/risk_management.py')
lines=p.read_text().splitlines()
for start,end in [(398,528)]:
    print(f'-- lines {start}-{end} --')
    for i in range(start,end+1):
        print(f\"{i:4}: {lines[i-1]}\")
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 647ms:
-- lines 398-528 --
 398: def compute_probabilistic_stops(holdings: dict, agent_summary: dict) -> dict:
 399:     """Compute Monte Carlo probabilistic stop-loss quality for held positions.
 400: 
 401:     For each position, simulates remaining-session price paths and estimates
 402:     the probability of hitting the ATR stop level, plus the expected time to
 403:     hit if triggered.
 404: 
 405:     Args:
 406:         holdings: The "holdings" dict from portfolio state.
 407:         agent_summary: Parsed agent_summary.json dict.
 408: 
 409:     Returns:
 410:         dict keyed by ticker, each with:
 411:             - stop_price_usd: ATR-based stop level
 412:             - stop_hit_prob: P(hitting stop this session), 0.0-1.0
 413:             - expected_hit_time_min: E[time to stop | hit], or None
 414:             - knockout_prob: For warrants, P(reaching financing level)
 415:             - current_price_usd: current price
 416:     """
 417:     try:
 418:         from portfolio.exit_optimizer import _first_hit_times, simulate_intraday_paths
 419:         from portfolio.session_calendar import remaining_session_minutes
 420:     except ImportError:
 421:         logger.warning("exit_optimizer or session_calendar not available")
 422:         return {}
 423: 
 424:     import numpy as np
 425: 
 426:     signals = agent_summary.get("signals", {})
 427:     result = {}
 428: 
 429:     for ticker, pos in holdings.items():
 430:         shares = pos.get("shares", 0)
 431:         if shares <= 0:
 432:             continue
 433: 
 434:         entry_price = pos.get("avg_cost_usd", 0)
 435:         if entry_price <= 0:
 436:             continue
 437: 
 438:         sig = signals.get(ticker)
 439:         if not sig:
 440:             continue
 441: 
 442:         current_price = sig.get("price_usd", 0)
 443:         atr_pct = sig.get("atr_pct", 0)
 444:         if current_price <= 0 or atr_pct <= 0:
 445:             continue
 446: 
 447:         # Determine instrument type for session lookup
 448:         if ticker.endswith("-USD") and ticker.startswith(("BTC", "ETH")):
 449:             inst_type = "crypto"
 450:         elif ticker.startswith(("XAG", "XAU")):
 451:             inst_type = "warrant"
 452:         else:
 453:             inst_type = "stock"
 454: 
 455:         # Get remaining session minutes
 456:         remaining = remaining_session_minutes(inst_type)
 457:         if remaining < 2:
 458:             continue
 459: 
 460:         # Estimate volatility from ATR
 461:         import math
 462:         vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)
 463: 
 464:         # ATR stop level
 465:         stop_price = entry_price * (1 - 2 * atr_pct / 100)
 466:         if stop_price <= 0:
 467:             stop_price = entry_price * 0.01
 468: 
 469:         # Simulate paths
 470:         paths = simulate_intraday_paths(
 471:             price=current_price,
 472:             volatility=vol,
 473:             drift=0.0,
 474:             remaining_minutes=int(remaining),
 475:             instrument_type=inst_type,
 476:             n_paths=2000,
 477:         )
 478: 
 479:         # Stop hit probability
 480:         session_min = np.min(paths[:, 1:], axis=1)
 481:         stop_hit_prob = float(np.mean(session_min <= stop_price))
 482: 
 483:         # Expected time to hit stop
 484:         hit_times = _first_hit_times(paths, stop_price, direction="below")
 485:         hitting = hit_times[hit_times > 0]
 486:         expected_hit_time = float(np.mean(hitting)) if len(hitting) > 0 else None
 487: 
 488:         result[ticker] = {
 489:             "stop_price_usd": round(stop_price, 4),
 490:             "stop_hit_prob": round(stop_hit_prob, 4),
 491:             "expected_hit_time_min": round(expected_hit_time, 1) if expected_hit_time else None,
 492:             "knockout_prob": None,
 493:             "current_price_usd": round(current_price, 4),
 494:         }
 495: 
 496:     return result
 497: 
 498: 
 499: def get_position_ages(portfolio: dict) -> dict:
 500:     """Calculate age of each position from first BUY transaction.
 501: 
 502:     Args:
 503:         portfolio: Full portfolio state dict (with "holdings" and "transactions").
 504: 
 505:     Returns:
 506:         dict keyed by ticker (only tickers currently held with shares > 0):
 507:             - age_hours: float
 508:             - age_days: float
 509:             - first_buy: str (ISO-8601 timestamp of first BUY)
 510:             - num_buys: int (total BUY transactions for this ticker)
 511:             - num_sells: int (total SELL transactions for this ticker)
 512:     """
 513:     holdings = portfolio.get("holdings", {})
 514:     transactions = portfolio.get("transactions", [])
 515:     now = datetime.datetime.now(datetime.UTC)
 516:     result = {}
 517: 
 518:     for ticker, pos in holdings.items():
 519:         shares = pos.get("shares", 0)
 520:         if shares <= 0:
 521:             continue
 522: 
 523:         # Find all BUY and SELL transactions for this ticker
 524:         first_buy_ts = None
 525:         num_buys = 0
 526:         num_sells = 0
 527: 
 528:         for tx in transactions:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import types, sys, importlib.util, pathlib
# fake file_utils for import
m=types.ModuleType('portfolio.file_utils')
m.load_json=lambda *a, **k: {}
sys.modules['portfolio.file_utils']=m
spec=importlib.util.spec_from_file_location('ks', pathlib.Path('portfolio/kelly_sizing.py'))
ks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)
transactions=[
 {'ticker':'ABC','action':'BUY','shares':1,'total_sek':100},
 {'ticker':'ABC','action':'SELL','shares':1,'total_sek':110},
 {'ticker':'ABC','action':'BUY','shares':1,'total_sek':200},
 {'ticker':'ABC','action':'SELL','shares':1,'total_sek':190},
]
print(ks._compute_trade_stats(transactions,'ABC'))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 674ms:
{'win_rate': 0.5, 'avg_win_pct': 26.666666666666668, 'avg_loss_pct': 26.666666666666668, 'total_trades': 2, 'wins': 1, 'losses': 1}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import types, sys, importlib.util, pathlib
m=types.ModuleType('portfolio.file_utils'); m.load_json=lambda *a, **k: {}; m.atomic_write_json=lambda *a, **k: None
sys.modules['portfolio.file_utils']=m
spec=importlib.util.spec_from_file_location('wp', pathlib.Path('portfolio/warrant_portfolio.py'))
wp=importlib.util.module_from_spec(spec); spec.loader.exec_module(wp)
holding={'units':100,'entry_price_sek':10,'leverage':5,'underlying_entry_price_usd':100}
print(wp.warrant_pnl(holding, 70, 10.5))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 818ms:
{'pnl_pct': -150.0, 'pnl_sek': -1500.0, 'current_implied_sek': -5.0, 'total_value_sek': -500.0, 'entry_value_sek': 1000, 'underlying_change_pct': -30.0, 'source': 'implied'}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from portfolio.trade_risk_classifier import classify_trade_risk
print(classify_trade_risk('BUY',0.8,10,'trending-up',0.8,existing_exposure_pct=24))
print(classify_trade_risk('BUY',0.8,10,'trending-up',0.8,existing_exposure_pct=26))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 670ms:
{'level': 'LOW', 'score': 1, 'factors': ['notable position (10.0% > 5%)']}
{'level': 'LOW', 'score': 2, 'factors': ['notable position (10.0% > 5%)', 'moderate concentration (26.0% > 25%)']}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio -File | Select-String -Pattern ' or 0\\.| or [0-9]+\\.| or 0\\b| or 1\\b|\\.get\\(["'^'"\\)]*\\) or ' -CaseSensitive | ForEach-Object { \"{0}:{1}: {2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 568ms:
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\equity_curve.py:344: fee = tx.get("fee_sek", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\equity_curve.py:363: sell_fee = tx.get("fee_sek", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\equity_curve.py:483: ts_str = t.get("sell_ts") or t.get("buy_ts")
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\kelly_sizing.py:186: per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\kelly_sizing.py:187: per_ticker_for_ticker = per_ticker_block.get(ticker, {}) or {}
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\kelly_sizing.py:209: per_ticker_sig = per_ticker_for_ticker.get(sig_name) or {}
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\kelly_sizing.py:278: per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo_risk.py:347: confidence: Confidence level (0.95 or 0.99 typically).
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo_risk.py:447: atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo.py:299: atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct") or _atr_default_for_ticker(ticker)
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo.py:360: action = extra.get("_weighted_action") or ticker_data.get("action", "HOLD")
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo.py:361: conf = extra.get("_weighted_confidence") or ticker_data.get("weighted_confidence", 0.5)
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_validator.py:77: total_sek = tx.get("total_sek", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_validator.py:99: shares = tx.get("shares", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_validator.py:231: tx_shares = tx.get("shares", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_validator.py:232: tx_price = tx.get("price_usd", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\risk_management.py:653: total_fees_from_state = portfolio.get("total_fees_sek", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\risk_management.py:663: fee = tx.get("fee_sek", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\risk_management.py:666: total_sek = tx.get("total_sek", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\warrant_portfolio.py:237: old_underlying = existing.get("underlying_entry_price_usd", 0) or 0
Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\warrant_portfolio.py:238: new_underlying = underlying_price_usd or 0

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import types, sys, importlib.util, pathlib, copy
store={}
mod=types.ModuleType('portfolio.file_utils')

def load_json(path, default=None):
    return copy.deepcopy(store.get(str(path), default))

def atomic_write_json(path, state):
    store[str(path)] = copy.deepcopy(state)

mod.load_json=load_json
mod.atomic_write_json=atomic_write_json
sys.modules['portfolio.file_utils']=mod
spec=importlib.util.spec_from_file_location('tg', pathlib.Path('portfolio/trade_guards.py'))
tg=importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)
# first buy opens position
print('initial warnings', tg.check_overtrading_guards('ABC','BUY','patient',{'holdings':{}},config=None))
tg.record_trade('ABC','BUY','patient')
# second buy is scale-in, not new position
warnings = tg.check_overtrading_guards('ABC','BUY','patient',{'holdings':{'ABC':{'shares':10}}},config=None)
print('after first buy', warnings)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-portfolio-risk
 succeeded in 598ms:
initial warnings []
after first buy [{'guard': 'ticker_cooldown', 'severity': 'block', 'message': 'ABC traded 0m ago by patient. Cooldown: 30m (base 30m × 1x). 30m remaining.', 'details': {'ticker': 'ABC', 'strategy': 'patient', 'elapsed_min': 0.0, 'cooldown_min': 30, 'multiplier': 1, 'remaining_min': 30.0}}, {'guard': 'position_rate_limit', 'severity': 'block', 'message': 'patient: 1 new position(s) in last 8h (limit: 1).', 'details': {'strategy': 'patient', 'recent_count': 1, 'limit': 1, 'window_hours': 8}}]

codex
Several core pieces of the new risk stack are either unusable or materially wrong: most modules cannot import on this branch, Kelly sizing miscomputes realized trade outcomes, stop-hit probabilities are understated, and multiple guard/risk calculations mis-handle common portfolio states. Those issues are sufficient to make the patch unsafe to rely on as-is.

Full review comments:

- [P0] Add the shared file_utils module before importing it — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_mgr.py:12-13
  Because this branch is compared against `empty-baseline`, `portfolio.file_utils` is not defined anywhere in the patch. Importing `portfolio_mgr` from this line currently raises `ModuleNotFoundError`, and the same missing helper also breaks `equity_curve`, `kelly_sizing`, `portfolio_validator`, `risk_management`, `trade_guards`, and `warrant_portfolio`, so most of the new portfolio/risk surface is unusable as-is.

- [P1] Match sells to their own entry lots in Kelly trade stats — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\kelly_sizing.py:95-103
  `_compute_trade_stats()` computes one blended buy price across every BUY for the ticker and then reuses that same basis for each SELL. When the history contains multiple round-trips or a full exit followed by a re-entry, realized wins/losses are rewritten against the wrong cost basis (for example `100→110` and `200→190` becomes `-26.7%` and `+26.7%`), which feeds materially incorrect win/loss inputs into `recommended_size()`.

- [P1] Compute stop-hit probability from path crossings — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo.py:216-219
  `probability_below()` only measures whether the simulated price finishes below the threshold at the horizon. `simulate_ticker()` exposes that value as `p_stop_hit_*`, but a stop is hit on any path that crosses the barrier and then rebounds before the close; those paths are all missed here. For volatile names this systematically understates stop risk and makes the Monte Carlo output look safer than the underlying path process implies.

- [P1] Record only opening buys in the position-rate guard — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\trade_guards.py:291-296
  `record_trade()` appends every BUY to `new_position_timestamps`, but Guard 3 is documented as a limit on *new positions*. Since `check_overtrading_guards()` never checks whether the ticker is already held, a scale-in BUY that happens after the per-ticker cooldown has expired still counts as a fresh position and can block all further adds for the rest of the 4h/8h window.

- [P1] Floor leveraged warrant value at zero — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\warrant_portfolio.py:95-103
  `warrant_pnl()` applies leveraged underlying returns directly to the entry price, so any move worse than `-100% / leverage` produces a negative instrument value and a loss greater than 100% of the capital invested. For example, a 30% drop on a 5x product yields `current_implied_sek = -5`; long warrants/certificates can go to zero, but they cannot become negative, so this will generate impossible portfolio valuations in large drawdowns.

- [P2] Score concentration from post-trade exposure, not current exposure — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\trade_risk_classifier.py:105-111
  The concentration factor is based on `existing_exposure_pct` alone. That misses risky entries like a 10% BUY on top of 24% existing exposure (true post-trade exposure 34%, scored as zero), and it also penalizes SELLs that reduce an overconcentrated book. The classifier therefore understates concentration risk for new buys while overstating risk for de-risking exits.

- [P2] Use total portfolio value for drawdown probabilities — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo_risk.py:392-399
  `drawdown_probability()` thresholds losses against invested exposure only (`sum(shares*price)`), even though `compute_portfolio_var()` receives the full `portfolio_state` including cash. In partially invested portfolios this reports the probability of losing 5% of the *risk sleeve*, not 5% of the portfolio—for example, a book that is 20% invested will trip `drawdown_5pct_prob` on a 5% holdings loss even though total portfolio drawdown is only 1%.
Several core pieces of the new risk stack are either unusable or materially wrong: most modules cannot import on this branch, Kelly sizing miscomputes realized trade outcomes, stop-hit probabilities are understated, and multiple guard/risk calculations mis-handle common portfolio states. Those issues are sufficient to make the patch unsafe to rely on as-is.

Full review comments:

- [P0] Add the shared file_utils module before importing it — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\portfolio_mgr.py:12-13
  Because this branch is compared against `empty-baseline`, `portfolio.file_utils` is not defined anywhere in the patch. Importing `portfolio_mgr` from this line currently raises `ModuleNotFoundError`, and the same missing helper also breaks `equity_curve`, `kelly_sizing`, `portfolio_validator`, `risk_management`, `trade_guards`, and `warrant_portfolio`, so most of the new portfolio/risk surface is unusable as-is.

- [P1] Match sells to their own entry lots in Kelly trade stats — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\kelly_sizing.py:95-103
  `_compute_trade_stats()` computes one blended buy price across every BUY for the ticker and then reuses that same basis for each SELL. When the history contains multiple round-trips or a full exit followed by a re-entry, realized wins/losses are rewritten against the wrong cost basis (for example `100→110` and `200→190` becomes `-26.7%` and `+26.7%`), which feeds materially incorrect win/loss inputs into `recommended_size()`.

- [P1] Compute stop-hit probability from path crossings — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo.py:216-219
  `probability_below()` only measures whether the simulated price finishes below the threshold at the horizon. `simulate_ticker()` exposes that value as `p_stop_hit_*`, but a stop is hit on any path that crosses the barrier and then rebounds before the close; those paths are all missed here. For volatile names this systematically understates stop risk and makes the Monte Carlo output look safer than the underlying path process implies.

- [P1] Record only opening buys in the position-rate guard — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\trade_guards.py:291-296
  `record_trade()` appends every BUY to `new_position_timestamps`, but Guard 3 is documented as a limit on *new positions*. Since `check_overtrading_guards()` never checks whether the ticker is already held, a scale-in BUY that happens after the per-ticker cooldown has expired still counts as a fresh position and can block all further adds for the rest of the 4h/8h window.

- [P1] Floor leveraged warrant value at zero — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\warrant_portfolio.py:95-103
  `warrant_pnl()` applies leveraged underlying returns directly to the entry price, so any move worse than `-100% / leverage` produces a negative instrument value and a loss greater than 100% of the capital invested. For example, a 30% drop on a 5x product yields `current_implied_sek = -5`; long warrants/certificates can go to zero, but they cannot become negative, so this will generate impossible portfolio valuations in large drawdowns.

- [P2] Score concentration from post-trade exposure, not current exposure — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\trade_risk_classifier.py:105-111
  The concentration factor is based on `existing_exposure_pct` alone. That misses risky entries like a 10% BUY on top of 24% existing exposure (true post-trade exposure 34%, scored as zero), and it also penalizes SELLs that reduce an overconcentrated book. The classifier therefore understates concentration risk for new buys while overstating risk for de-risking exits.

- [P2] Use total portfolio value for drawdown probabilities — Q:\finance-analyzer\.worktrees\adv-portfolio-risk\portfolio\monte_carlo_risk.py:392-399
  `drawdown_probability()` thresholds losses against invested exposure only (`sum(shares*price)`), even though `compute_portfolio_var()` receives the full `portfolio_state` including cash. In partially invested portfolios this reports the probability of losing 5% of the *risk sleeve*, not 5% of the portfolio—for example, a book that is 20% invested will trip `drawdown_5pct_prob` on a 5% holdings loss even though total portfolio drawdown is only 1%.
