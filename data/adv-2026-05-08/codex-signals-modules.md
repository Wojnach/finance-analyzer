OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-signals-modules
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e0834-123d-73f3-ab0b-96601d0b1f9f
--------
user
changes against 'empty-baseline'
  → Inspect diff against merge base
  • Analyze changed files for actionable bugs
  • Produce prioritized review findings JSON
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-08T15:28:18.136926Z ERROR codex_core::tools::router: error=Exit code: 1
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


 exited 1 in 562ms:
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
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Location; Get-ChildItem -Force' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Force Q:\\finance-analyzer; Get-ChildItem -Force Q:\\finance-analyzer\\.worktrees" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 660ms:

[32;1mPath[0m
[32;1m----[0m
Q:\finance-analyzer\.worktrees\adv-signals-modules

[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-signals-modules\portfoli
                      o
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-signals-modules
[32;1mPSChildName         : [0mportfolio
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mportfolio
[32;1mParent              : [0mQ:\finance-analyzer\.worktrees\adv-signals-modules
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\finance-analyzer\.worktrees\adv-signals-modules\portfolio
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-08 17:25:55
[32;1mCreationTimeUtc     : [0m2026-05-08 15:25:55
[32;1mLastAccessTime      : [0m2026-05-08 17:28:11
[32;1mLastAccessTimeUtc   : [0m2026-05-08 15:28:11
[32;1mLastWriteTime       : [0m2026-05-08 17:25:55
[32;1mLastWriteTimeUtc    : [0m2026-05-08 15:25:55
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mportfolio
[32;1mResolvedTarget      : [0mQ:\finance-analyzer\.worktrees\adv-signals-modules\portfolio
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m63
[32;1mName          : [0m.git


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m46
[32;1mName          : [0mBASELINE.md



 succeeded in 721ms:

    Directory: Q:\finance-analyzer

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    11:58                [44;1m.claude[0m
d----          2026-05-08    17:25                [44;1m.git[0m
d----          2026-03-17    11:03                [44;1m.openclaw[0m
d----          2026-03-11    13:18                [44;1m.playwright-cli[0m
d----          2026-05-04    17:07                [44;1m.playwright-mcp[0m
d----          2026-02-10    15:58                [44;1m.pytest_cache[0m
d----          2026-04-22    07:39                [44;1m.remember[0m
d----          2026-03-27    09:23                [44;1m.ruff_cache[0m
d----          2026-02-10    20:54                [44;1m.venv[0m
d----          2026-02-11    19:47                [44;1m.venv-train[0m
d----          2026-03-18    21:50                [44;1m.venv-unsloth[0m
d----          2026-03-29    10:00                [44;1m.worktree[0m
d----          2026-05-08    17:25                [44;1m.worktrees[0m
d----          2026-05-05    01:14                [44;1mdashboard[0m
d----          2026-05-08    17:28                [44;1mdata[0m
d----          2026-05-08    17:25                [44;1mdocs[0m
d----          2026-02-27    17:17                [44;1mllama.cpp[0m
d----          2026-05-02    01:40                [44;1mlogs[0m
d----          2026-03-18    00:01                [44;1mmemory[0m
d----          2026-02-13    02:16                [44;1mmodels[0m
d----          2026-03-11    13:09                [44;1moutput[0m
d----          2026-05-08    10:43                [44;1mportfolio[0m
d----          2026-05-03    19:58                [44;1mscripts[0m
d----          2026-05-08    10:43                [44;1mtests[0m
d----          2026-03-18    21:49                [44;1mtraining[0m
d----          2026-03-18    22:00                [44;1munsloth_compiled_cache[0m
d----          2026-03-04    09:19                [44;1muser_data[0m
d----          2026-05-07    18:43                [44;1mworktrees[0m
-a---          2026-02-19    15:14           3065 _check_positions.py
-a---          2026-02-19    12:48            115 [33;1m_check_tasks.ps1[0m
-a---          2026-02-19    12:49            594 _check.py
-a---          2026-02-18    00:49           3797 _checkin.py
-a---          2026-02-17    22:30            997 _diag.py
-a---          2026-02-17    22:30           1006 _diag2.py
-a---          2026-02-18    00:03           2043 _journal.py
-a---          2026-02-19    13:11           1806 _kill_all_loops.py
-a---          2026-02-19    13:10           1449 _kill_dupes.py
-a---          2026-02-18    11:49           3657 _l2_action.py
-a---          2026-02-18    01:04           3832 _l2_checkin.py
-a---          2026-02-16    07:18            463 _parse_log.py
-a---          2026-02-19    15:08            267 _run_analyze.py
-a---          2026-02-17    23:33           3040 _run.py
-a---          2026-02-18    10:34           1304 _send_tg.py
-a---          2026-02-19    13:14            456 _start_loop.py
-a---          2026-02-19    15:43            781 _test_claude_call.py
-a---          2026-02-19    18:01          11985 _test_integration.py
-a---          2026-02-19    15:48           1008 _test_watch_parse.py
-a---          2026-02-19    17:59            535 [33;1m_tmp_check.ps1[0m
-a---          2026-03-08    09:27            257 .git_commit_msg.txt
-a---          2026-02-10    15:48            285 .gitattributes
-a---          2026-05-04    01:23           5052 .gitignore
-a---          2026-02-10    17:19            682 .mcp.example.json
-a---          2026-03-31    10:32            485 .mcp.json
-a---          2026-05-02    02:01              0 0
-a---          2026-03-18    00:11           7874 AGENTS.md
-a---          2026-04-08    21:02            788 CLAUDE.local.md
-a---          2026-05-04    00:31          18656 CLAUDE.md
-a---          2026-03-27    07:40           4964 config.example.json
la---          2026-03-15    19:31              0 [36;1mconfig.json[0m -> C:\Users\Herc2\.config\finance-analyzer\config.json
-a---          2026-03-18    00:11            168 HEARTBEAT.md
-a---          2026-05-04    17:07         108128 home_phone_full.png
-a---          2026-03-18    00:11            636 IDENTITY.md
-a---          2026-03-03    21:23              0 nul
-a---          2026-05-04    01:49          75580 phone-assets-loaded.png
-a---          2026-05-04    01:48          19087 phone-assets.png
-a---          2026-05-04    01:48          54666 phone-avanza-live.png
-a---          2026-05-04    01:46          44490 phone-avanza.png
-a---          2026-05-03    23:13          37845 phone-bottomsheet.png
-a---          2026-05-03    23:14          46801 phone-dark.png
-a---          2026-05-03    23:12          88589 phone-decision-detail.png
-a---          2026-05-03    23:12          69794 phone-decisions.png
-a---          2026-05-03    23:13          40077 phone-health.png
-a---          2026-05-04    01:50          50937 phone-home-final.png
-a---          2026-05-03    23:12          46761 phone-home.png
-a---          2026-05-03    23:13          30953 phone-more.png
-a---          2026-05-04    01:48          70789 phone-prices.png
-a---          2026-05-04    01:49          46156 phone-refresh-clicked.png
-a---          2026-05-04    01:49          45690 phone-refresh-flash.png
-a---          2026-05-03    23:14          47303 phone-settings.png
-a---          2026-05-03    23:13          31516 phone-signals.png
-a---          2026-05-04    17:07          48161 portfolio_phone.png
-a---          2026-04-01    11:16           1206 pyproject.toml
-a---          2026-03-12    23:37           6736 README.md
-a---          2026-05-02    02:19            297 requirements.txt
-a---          2026-02-15    14:04           1384 send_telegram.py
-a---          2026-02-10    15:48           4653 SESSION_PLAN.md
-a---          2026-03-18    00:11           1673 SOUL.md
-a---          2026-02-18    01:25            169 [32;1mstart-loop.bat[0m
-a---          2026-02-18    01:17           3216 test_digest.py
-a---          2026-03-11    12:50          11392 TODO.md
-a---          2026-03-18    00:11            860 TOOLS.md
-a---          2026-02-10    16:01          81920 tradesv3.dryrun.sqlite
-a---          2026-03-18    00:11            477 USER.md
-a---          2026-03-18    01:00           1432 WORKTREE_README.md

    Directory: Q:\finance-analyzer\.worktrees

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    17:25                [44;1madv-avanza-api[0m
d----          2026-05-08    17:25                [44;1madv-data-external[0m
d----          2026-05-08    17:25                [44;1madv-infrastructure[0m
d----          2026-05-08    17:25                [44;1madv-metals-core[0m
d----          2026-05-08    17:25                [44;1madv-orchestration[0m
d----          2026-05-08    17:25                [44;1madv-portfolio-risk[0m
d----          2026-05-08    17:25                [44;1madv-signals-core[0m
d----          2026-05-08    17:25                [44;1madv-signals-modules[0m
d----          2026-05-08    17:24                [44;1madversarial-review-2026-05-08[0m
d----          2026-04-30    10:35                [44;1mauto-improve[0m
d----          2026-05-08    10:44                [44;1mauto-improve-2026-05-08[0m
d----          2026-03-05    23:44                [44;1mauto-session-2026-03-05[0m
d----          2026-03-09    19:53                [44;1mauto-session-2026-03-09[0m
d----          2026-03-13    09:10                [44;1mauto-session-2026-03-13[0m
d----          2026-03-20    09:21                [44;1mauto-session-2026-03-20[0m
d----          2026-03-25    09:48                [44;1mauto-session-2026-03-25[0m
d----          2026-04-28    15:27                [44;1mdashboard-ops-board[0m
d----          2026-03-09    23:55                [44;1mfix-metals-phase1[0m
d----          2026-05-05    01:03                [44;1mfix-t1-timeout-drift-2026-05-05[0m
d----          2026-03-10    00:21                [44;1mgolddigger-signal-upgrade[0m
d----          2026-05-04    16:45                [44;1mheartbeat-before-l2-2026-05-04[0m
d----          2026-05-05    10:26                [44;1mimprove-2026-05-05[0m
d----          2026-05-04    10:40                [44;1mimprove-auto-session-2026-05-04[0m
d----          2026-02-18    15:18                [44;1miskbets[0m
d----          2026-05-05    00:35                [44;1mlive-prices-charts-2026-05-05[0m
d----          2026-03-09    22:56                [44;1mlocal-llm-accuracy[0m
d----          2026-03-11    20:06                [44;1mmetals-execution-engine[0m
d----          2026-05-05    09:39                [44;1mnews-calendar-2026-05-05[0m
d----          2026-03-17    23:53                [44;1mprevent-loop-duplicates[0m
d----          2026-05-04    14:56                [44;1mregime-cache-l2-2026-05-04[0m
d----          2026-03-04    19:23                [44;1mwt-focus-analysis[0m
d----          2026-03-04    19:55                [44;1mwt-main-merge[0m


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content .git' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Force portfolio' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 601ms:
gitdir: Q:/finance-analyzer/.git/worktrees/adv-signals-modules

 succeeded in 637ms:

    Directory: Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    17:25                [44;1msignals[0m


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git --git-dir=Q:/finance-analyzer/.git/worktrees/adv-signals-modules --work-tree=Q:/finance-analyzer/.worktrees/adv-signals-modules status --short --branch' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 566ms:
## review/2026-05-08-signals-modules

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git --git-dir=Q:/finance-analyzer/.git/worktrees/adv-signals-modules --work-tree=Q:/finance-analyzer/.worktrees/adv-signals-modules diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 559ms:
 portfolio/signals/__init__.py                      |   4 +
 .../signals/__pycache__/__init__.cpython-312.pyc   | Bin 0 -> 300 bytes
 .../__pycache__/calendar_seasonal.cpython-312.pyc  | Bin 0 -> 14489 bytes
 .../__pycache__/candlestick.cpython-312.pyc        | Bin 0 -> 10987 bytes
 .../__pycache__/claude_fundamental.cpython-312.pyc | Bin 0 -> 41112 bytes
 .../complexity_gap_regime.cpython-312.pyc          | Bin 0 -> 9722 bytes
 .../__pycache__/copper_gold_ratio.cpython-312.pyc  | Bin 0 -> 10440 bytes
 .../__pycache__/cot_positioning.cpython-312.pyc    | Bin 0 -> 13511 bytes
 .../__pycache__/credit_spread.cpython-312.pyc      | Bin 0 -> 10783 bytes
 .../__pycache__/cross_asset_tsmom.cpython-312.pyc  | Bin 0 -> 8709 bytes
 .../__pycache__/crypto_cross_asset.cpython-312.pyc | Bin 0 -> 12808 bytes
 .../__pycache__/crypto_evrp.cpython-312.pyc        | Bin 0 -> 12547 bytes
 .../__pycache__/crypto_macro.cpython-312.pyc       | Bin 0 -> 8636 bytes
 .../__pycache__/drift_regime_gate.cpython-312.pyc  | Bin 0 -> 7226 bytes
 .../__pycache__/dxy_cross_asset.cpython-312.pyc    | Bin 0 -> 3496 bytes
 .../__pycache__/econ_calendar.cpython-312.pyc      | Bin 0 -> 11241 bytes
 .../signals/__pycache__/fibonacci.cpython-312.pyc  | Bin 0 -> 16325 bytes
 .../signals/__pycache__/forecast.cpython-312.pyc   | Bin 0 -> 38908 bytes
 .../__pycache__/futures_basis.cpython-312.pyc      | Bin 0 -> 9107 bytes
 .../__pycache__/futures_flow.cpython-312.pyc       | Bin 0 -> 10413 bytes
 .../gold_real_yield_paradox.cpython-312.pyc        | Bin 0 -> 13275 bytes
 .../__pycache__/hash_ribbons.cpython-312.pyc       | Bin 0 -> 10732 bytes
 .../__pycache__/heikin_ashi.cpython-312.pyc        | Bin 0 -> 19911 bytes
 .../__pycache__/hurst_regime.cpython-312.pyc       | Bin 0 -> 11286 bytes
 .../__pycache__/macro_regime.cpython-312.pyc       | Bin 0 -> 14741 bytes
 .../mahalanobis_turbulence.cpython-312.pyc         | Bin 0 -> 11479 bytes
 .../__pycache__/mean_reversion.cpython-312.pyc     | Bin 0 -> 19325 bytes
 .../__pycache__/metals_cross_asset.cpython-312.pyc | Bin 0 -> 14754 bytes
 .../signals/__pycache__/momentum.cpython-312.pyc   | Bin 0 -> 17053 bytes
 .../__pycache__/momentum_factors.cpython-312.pyc   | Bin 0 -> 18963 bytes
 .../__pycache__/network_momentum.cpython-312.pyc   | Bin 0 -> 14148 bytes
 .../signals/__pycache__/news_event.cpython-312.pyc | Bin 0 -> 22038 bytes
 .../__pycache__/orderbook_flow.cpython-312.pyc     | Bin 0 -> 6728 bytes
 .../__pycache__/oscillators.cpython-312.pyc        | Bin 0 -> 21033 bytes
 .../ovx_metals_spillover.cpython-312.pyc           | Bin 0 -> 8537 bytes
 .../__pycache__/realized_skewness.cpython-312.pyc  | Bin 0 -> 9524 bytes
 .../residual_pair_reversion.cpython-312.pyc        | Bin 0 -> 13297 bytes
 .../__pycache__/shannon_entropy.cpython-312.pyc    | Bin 0 -> 9486 bytes
 .../__pycache__/smart_money.cpython-312.pyc        | Bin 0 -> 17395 bytes
 .../statistical_jump_regime.cpython-312.pyc        | Bin 0 -> 9158 bytes
 .../signals/__pycache__/structure.cpython-312.pyc  | Bin 0 -> 9694 bytes
 .../treasury_risk_rotation.cpython-312.pyc         | Bin 0 -> 8543 bytes
 .../signals/__pycache__/trend.cpython-312.pyc      | Bin 0 -> 19892 bytes
 .../__pycache__/vix_term_structure.cpython-312.pyc | Bin 0 -> 7507 bytes
 .../__pycache__/vol_ratio_regime.cpython-312.pyc   | Bin 0 -> 11047 bytes
 .../signals/__pycache__/volatility.cpython-312.pyc | Bin 0 -> 17340 bytes
 .../__pycache__/volume_flow.cpython-312.pyc        | Bin 0 -> 15539 bytes
 .../__pycache__/williams_vix_fix.cpython-312.pyc   | Bin 0 -> 9065 bytes
 .../xtrend_equity_spillover.cpython-312.pyc        | Bin 0 -> 9464 bytes
 portfolio/signals/calendar_seasonal.py             | 492 +++++++++++
 portfolio/signals/candlestick.py                   | 401 +++++++++
 portfolio/signals/claude_fundamental.py            | 941 ++++++++++++++++++++
 portfolio/signals/complexity_gap_regime.py         | 277 ++++++
 portfolio/signals/copper_gold_ratio.py             | 275 ++++++
 portfolio/signals/cot_positioning.py               | 411 +++++++++
 portfolio/signals/credit_spread.py                 | 326 +++++++
 portfolio/signals/cross_asset_tsmom.py             | 233 +++++
 portfolio/signals/crypto_cross_asset.py            | 262 ++++++
 portfolio/signals/crypto_evrp.py                   | 353 ++++++++
 portfolio/signals/crypto_macro.py                  | 281 ++++++
 portfolio/signals/drift_regime_gate.py             | 186 ++++
 portfolio/signals/dxy_cross_asset.py               | 100 +++
 portfolio/signals/econ_calendar.py                 | 264 ++++++
 portfolio/signals/fibonacci.py                     | 539 ++++++++++++
 portfolio/signals/forecast.py                      | 975 +++++++++++++++++++++
 portfolio/signals/futures_basis.py                 | 267 ++++++
 portfolio/signals/futures_flow.py                  | 310 +++++++
 portfolio/signals/gold_real_yield_paradox.py       | 323 +++++++
 portfolio/signals/hash_ribbons.py                  | 302 +++++++
 portfolio/signals/heikin_ashi.py                   | 647 ++++++++++++++
 portfolio/signals/hurst_regime.py                  | 341 +++++++
 portfolio/signals/macro_regime.py                  | 419 +++++++++
 portfolio/signals/mahalanobis_turbulence.py        | 332 +++++++
 portfolio/signals/mean_reversion.py                | 600 +++++++++++++
 portfolio/signals/metals_cross_asset.py            | 454 ++++++++++
 portfolio/signals/momentum.py                      | 439 ++++++++++
 portfolio/signals/momentum_factors.py              | 514 +++++++++++
 portfolio/signals/network_momentum.py              | 386 ++++++++
 portfolio/signals/news_event.py                    | 610 +++++++++++++
 portfolio/signals/orderbook_flow.py                | 218 +++++
 portfolio/signals/oscillators.py                   | 589 +++++++++++++
 portfolio/signals/ovx_metals_spillover.py          | 228 +++++
 portfolio/signals/realized_skewness.py             | 240 +++++
 portfolio/signals/residual_pair_reversion.py       | 391 +++++++++
 portfolio/signals/shannon_entropy.py               | 307 +++++++
 portfolio/signals/smart_money.py                   | 562 ++++++++++++
 portfolio/signals/statistical_jump_regime.py       | 269 ++++++
 portfolio/signals/structure.py                     | 262 ++++++
 portfolio/signals/treasury_risk_rotation.py        | 206 +++++
 portfolio/signals/trend.py                         | 575 ++++++++++++
 portfolio/signals/vix_term_structure.py            | 197 +++++
 portfolio/signals/vol_ratio_regime.py              | 272 ++++++
 portfolio/signals/volatility.py                    | 458 ++++++++++
 portfolio/signals/volume_flow.py                   | 360 ++++++++
 portfolio/signals/williams_vix_fix.py              | 231 +++++
 portfolio/signals/xtrend_equity_spillover.py       | 253 ++++++
 96 files changed, 17882 insertions(+)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/signals/__init__.py' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 691ms:
"""Enhanced signal modules — grouped technical indicators.

Each module computes multiple sub-indicators and returns a composite vote.
"""

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib, pkgutil, sys
from pathlib import Path
sys.path.insert(0, r'Q:/finance-analyzer/.worktrees/adv-signals-modules')
mods=[]
for p in Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals').glob('*.py'):
    if p.name=='__init__.py':
        continue
    mods.append('portfolio.signals.'+p.stem)
for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        print('FAIL', m, type(e).__name__, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 2434ms:
FAIL portfolio.signals.calendar_seasonal ModuleNotFoundError No module named 'portfolio.fomc_dates'
OK portfolio.signals.candlestick
FAIL portfolio.signals.claude_fundamental ModuleNotFoundError No module named 'portfolio.file_utils'
FAIL portfolio.signals.complexity_gap_regime ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.copper_gold_ratio ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.cot_positioning ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.credit_spread ModuleNotFoundError No module named 'portfolio.file_utils'
FAIL portfolio.signals.cross_asset_tsmom ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.crypto_cross_asset ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.crypto_evrp ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.crypto_macro ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.drift_regime_gate ModuleNotFoundError No module named 'portfolio.signal_utils'
OK portfolio.signals.dxy_cross_asset
FAIL portfolio.signals.econ_calendar ModuleNotFoundError No module named 'portfolio.econ_dates'
FAIL portfolio.signals.fibonacci ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.forecast ModuleNotFoundError No module named 'portfolio.file_utils'
FAIL portfolio.signals.futures_basis ModuleNotFoundError No module named 'portfolio.api_utils'
FAIL portfolio.signals.futures_flow ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.gold_real_yield_paradox ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.hash_ribbons ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.heikin_ashi ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.hurst_regime ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.macro_regime ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.mahalanobis_turbulence ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.mean_reversion ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.metals_cross_asset ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.momentum ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.momentum_factors ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.network_momentum ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.news_event ModuleNotFoundError No module named 'portfolio.file_utils'
FAIL portfolio.signals.orderbook_flow ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.oscillators ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.ovx_metals_spillover ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.realized_skewness ModuleNotFoundError No module named 'scipy'
FAIL portfolio.signals.residual_pair_reversion ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.shannon_entropy ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.smart_money ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.statistical_jump_regime ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.structure ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.treasury_risk_rotation ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.trend ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.vix_term_structure ModuleNotFoundError No module named 'portfolio.shared_state'
FAIL portfolio.signals.volatility ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.volume_flow ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.vol_ratio_regime ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.williams_vix_fix ModuleNotFoundError No module named 'portfolio.signal_utils'
FAIL portfolio.signals.xtrend_equity_spillover ModuleNotFoundError No module named 'portfolio.shared_state'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content BASELINE.md' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 564ms:
# Empty baseline for adversarial review diffs

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Force Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Force Q:\\finance-analyzer\\portfolio\\signals | Select-Object -First 20" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 670ms:

    Directory: Q:\finance-analyzer\portfolio\signals

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    15:30                [44;1m__pycache__[0m
-a---          2026-02-19    17:14            141 __init__.py
-a---          2026-04-08    23:56          15883 calendar_seasonal.py
-a---          2026-03-09    09:20          13019 candlestick.py
-a---          2026-05-08    10:43          35933 claude_fundamental.py
-a---          2026-04-28    10:36           8847 complexity_gap_regime.py
-a---          2026-04-17    19:06           9523 copper_gold_ratio.py
-a---          2026-05-02    02:01          13816 cot_positioning.py
-a---          2026-05-08    10:43          11242 credit_spread.py
-a---          2026-04-15    18:53           7265 cross_asset_tsmom.py
-a---          2026-04-30    10:34          10486 crypto_cross_asset.py
-a---          2026-04-28    10:36          11866 crypto_evrp.py
-a---          2026-04-07    10:28           9215 crypto_macro.py
-a---          2026-04-28    18:50           6397 drift_regime_gate.py
-a---          2026-04-13    15:22           3434 dxy_cross_asset.py
-a---          2026-04-24    00:15           9600 econ_calendar.py
-a---          2026-05-08    10:43          18111 fibonacci.py
-a---          2026-05-08    10:43          40433 forecast.py
-a---          2026-04-10    18:48           9555 futures_basis.py
-a---          2026-05-02    02:01          11029 futures_flow.py


 succeeded in 701ms:

    Directory: Q:\finance-analyzer\portfolio

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    13:00                [44;1m__pycache__[0m
d----          2026-04-17    09:20                [44;1mavanza[0m
d----          2026-04-07    10:28                [44;1melongir[0m
d----          2026-05-08    10:43                [44;1mgolddigger[0m
d----          2026-05-04    00:31                [44;1mmstr_loop[0m
d----          2026-05-08    10:43                [44;1msignals[0m
d----          2026-04-07    10:28                [44;1mstrategies[0m
-a---          2026-02-11    15:03              0 __init__.py
-a---          2026-05-08    10:43          39336 accuracy_degradation.py
-a---          2026-05-05    01:14          83376 accuracy_stats.py
-a---          2026-05-08    10:43          60621 agent_invocation.py
-a---          2026-04-01    13:02           1976 alert_budget.py
-a---          2026-03-22    09:29          10478 alpha_vantage.py
-a---          2026-04-17    13:51          33262 analyze.py
-a---          2026-03-19    09:19           1750 api_utils.py
-a---          2026-04-15    18:58          30731 autonomous.py
-a---          2026-05-02    01:43          13438 avanza_client.py
-a---          2026-04-13    12:28          15330 avanza_control.py
-a---          2026-04-15    10:57           3643 avanza_order_lock.py
-a---          2026-05-02    02:45          16708 avanza_orders.py
-a---          2026-04-13    12:28           7995 avanza_resilient_page.py
-a---          2026-04-28    10:36          47835 avanza_session.py
-a---          2026-03-18    09:18           4184 avanza_tracker.py
-a---          2026-03-27    09:37          11162 backtester.py
-a---          2026-05-04    01:11          19504 bert_sentiment.py
-a---          2026-04-17    13:51          23136 bigbet.py
-a---          2026-05-02    01:30           5618 circuit_breaker.py
-a---          2026-04-28    10:36          26674 claude_gate.py
-a---          2026-03-19    09:19           2397 config_validator.py
-a---          2026-03-12    14:08           3856 cost_model.py
-a---          2026-04-07    10:28          16564 crypto_macro_data.py
-a---          2026-04-30    01:22          10797 crypto_precompute.py
-a---          2026-03-31    10:23          13326 crypto_scheduler.py
-a---          2026-03-19    09:19           6510 cumulative_tracker.py
-a---          2026-04-24    10:19          10475 daily_digest.py
-a---          2026-05-05    10:25          12435 data_collector.py
-a---          2026-03-19    09:19           2245 data_refresh.py
-a---          2026-03-19    09:19           3860 decision_outcome_tracker.py
-a---          2026-04-15    10:57          10270 digest.py
-a---          2026-04-09    00:02           7237 earnings_calendar.py
-a---          2026-04-28    15:43           9983 econ_dates.py
-a---          2026-05-08    10:43          22292 equity_curve.py
-a---          2026-03-31    10:23          27968 exit_optimizer.py
-a---          2026-03-31    19:44           3461 exposure_coach.py
-a---          2026-05-02    01:21           6928 fear_greed.py
-a---          2026-04-07    22:36           3646 feature_normalizer.py
-a---          2026-05-04    01:23          13061 file_utils.py
-a---          2026-04-20    17:12          45311 fin_evolve.py
-a---          2026-04-07    10:28          55299 fin_fish.py
-a---          2026-04-26    12:57          72320 fin_snipe_manager.py
-a---          2026-03-12    23:37          11196 fin_snipe.py
-a---          2026-04-07    10:28           7513 fish_instrument_finder.py
-a---          2026-04-14    13:50          32541 fish_monitor_smart.py
-a---          2026-04-06    10:29           8625 focus_analysis.py
-a---          2026-02-20    00:57           1829 fomc_dates.py
-a---          2026-05-01    01:45          16156 forecast_accuracy.py
-a---          2026-03-19    09:19          14751 forecast_signal.py
-a---          2026-05-02    01:54           2212 funding_rate.py
-a---          2026-04-09    11:29           7095 futures_data.py
-a---          2026-05-08    10:43           3573 fx_rates.py
-a---          2026-03-15    12:01           1047 gold_precompute.py
-a---          2026-05-04    14:28           9834 gpu_gate.py
-a---          2026-05-04    17:09          17573 health.py
-a---          2026-04-08    23:55           3719 http_retry.py
-a---          2026-05-02    01:21           9754 ic_computation.py
-a---          2026-04-21    10:58           8956 indicators.py
-a---          2026-04-01    00:50          10500 instrument_profile.py
-a---          2026-04-17    13:51          32447 iskbets.py
-a---          2026-03-19    09:19          12581 journal_index.py
-a---          2026-04-21    10:58          20192 journal.py
-a---          2026-04-06    18:54           9913 kelly_metals.py
-a---          2026-05-02    01:42          14795 kelly_sizing.py
-a---          2026-04-15    10:57           7140 linear_factor.py
-a---          2026-05-01    10:23          21192 llama_server.py
-a---          2026-05-03    22:12          25818 llm_batch.py
-a---          2026-04-21    14:38           8418 llm_calibration.py
-a---          2026-05-01    01:45          10729 llm_outcome_backfill.py
-a---          2026-04-30    23:51           9892 llm_probability_log.py
-a---          2026-05-01    01:18          17652 local_llm_report.py
-a---          2026-04-29    10:19          17839 log_rotation.py
-a---          2026-02-21    13:46           1341 logging_config.py
-a---          2026-05-08    10:43          85014 loop_contract.py
-a---          2026-05-03    21:55           8034 loop_health.py
-a---          2026-04-15    10:57          13239 macro_context.py
-a---          2026-05-05    01:02          60206 main.py
-a---          2026-04-14    13:50          17149 market_health.py
-a---          2026-04-17    10:24          11978 market_timing.py
-a---          2026-04-02    11:05          13655 memory_consolidation.py
-a---          2026-04-02    22:20           7791 message_store.py
-a---          2026-03-18    09:18           3338 message_throttle.py
-a---          2026-04-17    12:14          16356 meta_learner.py
-a---          2026-04-15    10:57          11159 metals_cross_assets.py
-a---          2026-03-25    09:33           6840 metals_ladder.py
-a---          2026-03-31    20:03           3788 metals_orderbook.py
-a---          2026-04-14    13:50          42303 metals_precompute.py
-a---          2026-04-17    10:24           8336 microstructure_state.py
-a---          2026-04-02    11:05           7633 microstructure.py
-a---          2026-04-07    10:28           3572 ministral_signal.py
-a---          2026-04-30    23:51           7943 ministral_trader.py
-a---          2026-03-19    09:19           5128 ml_signal.py
-a---          2026-03-19    09:19           6791 ml_trainer.py
-a---          2026-05-08    10:43          17312 monte_carlo_risk.py
-a---          2026-05-04    10:39          14980 monte_carlo.py
-a---          2026-04-30    01:22          10761 mstr_precompute.py
-a---          2026-04-17    13:51           9005 multi_agent_layer2.py
-a---          2026-04-28    13:20          12100 news_keywords.py
-a---          2026-03-31    10:23           1983 notification_text.py
-a---          2026-04-14    13:50          36481 oil_precompute.py
-a---          2026-05-02    02:02          12057 onchain_data.py
-a---          2026-03-19    09:19          21167 orb_backtest.py
-a---          2026-03-19    09:19          13972 orb_postmortem.py
-a---          2026-03-19    09:19          20564 orb_predictor.py
-a---          2026-05-08    10:43          20347 outcome_tracker.py
-a---          2026-03-19    09:19           2846 perception_gate.py
-a---          2026-04-28    10:36           6356 portfolio_mgr.py
-a---          2026-04-11    15:30          11941 portfolio_validator.py
-a---          2026-05-01    02:18          10257 price_source.py
-a---          2026-03-19    09:19          20060 price_targets.py
-a---          2026-03-31    10:23           2448 process_lock.py
-a---          2026-04-07    10:28          13250 prophecy.py
-a---          2026-04-16    10:34           7056 qwen3_signal.py
-a---          2026-04-30    23:51           7691 qwen3_trader.py
-a---          2026-03-19    09:19           7721 reflection.py
-a---          2026-03-19    09:19           6075 regime_alerts.py
-a---          2026-05-05    00:49          57168 reporting.py
-a---          2026-05-08    10:43          37263 risk_management.py
-a---          2026-04-02    11:05           2642 seasonality_updater.py
-a---          2026-04-15    10:57           4030 seasonality.py
-a---          2026-05-01    01:45          12513 sentiment_shadow_backfill.py
-a---          2026-04-28    13:20          39408 sentiment.py
-a---          2026-03-19    09:19           7221 session_calendar.py
-a---          2026-04-21    14:38           8276 shadow_registry.py
-a---          2026-05-04    10:39          16495 shared_state.py
-a---          2026-03-27    07:16           1808 short_horizon.py
-a---          2026-03-27    00:14          14797 signal_db.py
-a---          2026-05-01    10:23           5207 signal_decay_alert.py
-a---          2026-05-07    23:43         182767 signal_engine.py
-a---          2026-05-02    01:21           6839 signal_history.py
-a---          2026-04-02    11:05           8981 signal_postmortem.py
-a---          2026-05-07    19:03          17124 signal_registry.py
-a---          2026-05-05    00:49           2849 signal_state_since.py
-a---          2026-03-27    12:05           4383 signal_utils.py
-a---          2026-04-15    10:57           6091 signal_weight_optimizer.py
-a---          2026-03-27    00:14           4527 signal_weights.py
-a---          2026-03-15    12:01           1053 silver_precompute.py
-a---          2026-03-19    09:19           4718 social_sentiment.py
-a---          2026-03-18    09:18           3031 stats.py
-a---          2026-04-23    10:21          10730 subprocess_utils.py
-a---          2026-03-19    09:19           5570 telegram_notifications.py
-a---          2026-05-02    01:54          16794 telegram_poller.py
-a---          2026-05-02    01:42          11934 ticker_accuracy.py
-a---          2026-05-07    23:43          14402 tickers.py
-a---          2026-04-02    11:05           5529 tinylora_trainer.py
-a---          2026-05-05    10:25          14513 trade_guards.py
-a---          2026-04-01    13:02           3780 trade_risk_classifier.py
-a---          2026-03-19    09:19           3927 trade_validation.py
-a---          2026-04-15    10:57           5800 train_signal_weights.py
-a---          2026-04-23    00:04          20282 trigger.py
-a---          2026-03-18    09:18           8284 vector_memory.py
-a---          2026-05-02    01:55           9297 warrant_portfolio.py
-a---          2026-03-19    09:19          10477 weekly_digest.py


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
import filecmp
wt = Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals')
main = Path(r'Q:/finance-analyzer/portfolio/signals')
for p in sorted(wt.glob('*.py')):
    if p.name == '__init__.py':
        continue
    mp = main / p.name
    same = mp.exists() and filecmp.cmp(p, mp, shallow=False)
    print(p.name, 'SAME' if same else 'DIFF/MISSING')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 610ms:
calendar_seasonal.py SAME
candlestick.py SAME
claude_fundamental.py SAME
complexity_gap_regime.py SAME
copper_gold_ratio.py SAME
cot_positioning.py SAME
credit_spread.py SAME
cross_asset_tsmom.py SAME
crypto_cross_asset.py SAME
crypto_evrp.py SAME
crypto_macro.py SAME
drift_regime_gate.py SAME
dxy_cross_asset.py SAME
econ_calendar.py SAME
fibonacci.py SAME
forecast.py SAME
futures_basis.py SAME
futures_flow.py SAME
gold_real_yield_paradox.py SAME
hash_ribbons.py SAME
heikin_ashi.py SAME
hurst_regime.py SAME
macro_regime.py SAME
mahalanobis_turbulence.py SAME
mean_reversion.py SAME
metals_cross_asset.py SAME
momentum.py SAME
momentum_factors.py SAME
network_momentum.py SAME
news_event.py SAME
orderbook_flow.py SAME
oscillators.py SAME
ovx_metals_spillover.py SAME
realized_skewness.py SAME
residual_pair_reversion.py SAME
shannon_entropy.py SAME
smart_money.py SAME
statistical_jump_regime.py SAME
structure.py SAME
treasury_risk_rotation.py SAME
trend.py SAME
vix_term_structure.py SAME
vol_ratio_regime.py SAME
volatility.py SAME
volume_flow.py SAME
williams_vix_fix.py SAME
xtrend_equity_spillover.py SAME

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signal_utils.py" in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse Q:\\finance-analyzer\\tests | Select-String -Pattern 'signals\\.|portfolio\\.signals|signal_registry|compute_' | Select-Object -First 200" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 629ms:
"""Shared signal utility functions.

Canonical implementations of common technical-analysis helpers used across
multiple signal modules.  Import from here instead of duplicating locally.

All functions operate on ``pd.Series`` inputs and return ``pd.Series``
(or ``float`` for ``safe_float``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Returns NaN where insufficient data (min_periods = period).
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average using pandas ewm.

    Uses ``adjust=False`` for recursive EMA (standard in TA).
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI.

    Uses ``clip()`` for separating gains/losses (numerically stable).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def safe_float(val) -> float:
    """Convert *val* to float, returning ``NaN`` for non-finite / missing values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    try:
        f = float(val)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA / SMMA).

    Equivalent to EMA with ``alpha = 1 / period``.
    """
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average with linearly increasing weights.

    Weight_i = i + 1 for i in 0..period-1 (most recent bar has highest weight).
    """
    weights = np.arange(1, period + 1, dtype=float)

    def _apply_wma(x: np.ndarray) -> float:
        return np.dot(x, weights) / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(
        _apply_wma, raw=True,
    )


def majority_vote(votes: list, count_hold: bool = False) -> tuple:
    """Compute majority vote from a list of BUY/SELL/HOLD strings.

    HOLD votes are treated as abstentions — they do not block a directional
    result.  Only BUY vs SELL counts determine direction.  This means a
    composite with 5 HOLDs and 2 BUYs produces BUY (not HOLD), because the
    HOLDs are neutral, not opposing votes.

    Args:
        votes: List of "BUY", "SELL", or "HOLD" strings
        count_hold: If False (default), confidence = winner / active_voters (BUY+SELL only).
                    If True, confidence = winner / total_votes (including HOLD).

    Returns:
        (action, confidence) tuple where action is "BUY", "SELL", or "HOLD"
    """
    buy = sum(1 for v in votes if v == "BUY")
    sell = sum(1 for v in votes if v == "SELL")
    hold = sum(1 for v in votes if v == "HOLD")

    active = buy + sell
    total = buy + sell + hold

    if active == 0:
        return "HOLD", 0.0

    denom = total if count_hold else active

    if buy > sell:
        return "BUY", round(buy / denom, 4) if denom > 0 else 0.0
    elif sell > buy:
        return "SELL", round(sell / denom, 4) if denom > 0 else 0.0
    else:
        # Tie between buy and sell — no clear direction.
        # HOLD confidence is always 0.0 — it's the absence of a signal, not a
        # directional vote.
        return "HOLD", 0.0


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change: ``100 * (current - n_periods_ago) / n_periods_ago``."""
    shifted = series.shift(period)
    return 100.0 * (series - shifted) / shifted.replace(0, np.nan)

 succeeded in 715ms:

Q:\finance-analyzer\tests\_state_reset.py:109:        import [7mportfolio.signals[0m.forecast as fc
Q:\finance-analyzer\tests\test_3h_integration.py:28:        from portfolio.indicators import [7mcompute_[0mindicators
Q:\finance-analyzer\tests\test_3h_integration.py:32:        ind = [7mcompute_[0mindicators(df, horizon="3h")
Q:\finance-analyzer\tests\test_3h_integration.py:44:        from portfolio.indicators import [7mcompute_[0mindicators
Q:\finance-analyzer\tests\test_3h_integration.py:48:        ind = [7mcompute_[0mindicators(df, horizon="3h")
Q:\finance-analyzer\tests\test_3h_integration.py:55:        from portfolio.indicators import [7mcompute_[0mindicators
Q:\finance-analyzer\tests\test_3h_integration.py:59:        ind = [7mcompute_[0mindicators(df)
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:54:    def 
test_get_or_[7mcompute_[0maccuracy_serializes_on_miss(self, [0m[7m[0mmonkeypatch, [0m[7m[0mtmp_path):[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:63:        results, elapsed = self._race(lambda: 
acc_mod.get_or_[7mcompute_[0maccuracy("1d"))[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:78:    def test_get_or_[7mcompute_[0mrecent_serializes_on_miss(self, [0m
[7m[0mmonkeypatch, [0m[7m[0mtmp_path):[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:85:            lambda: 
acc_mod.get_or_[7mcompute_[0mrecent_accuracy("1d", [0m[7m[0mdays=7)[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:92:    def 
test_get_or_[7mcompute_[0mper_ticker_serializes_on_miss(self, [0m[7m[0mmonkeypatch, [0m[7m[0mtmp_path):[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:99:            lambda: 
acc_mod.get_or_[7mcompute_[0mper_ticker_accuracy("1d")[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:106:    def 
test_get_or_[7mcompute_[0mconsensus_serializes_on_miss(self, [0m[7m[0mmonkeypatch, [0m[7m[0mtmp_path):[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:115:            lambda: 
acc_mod.get_or_[7mcompute_[0mconsensus_accuracy("1d")[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:124:    def 
test_get_or_[7mcompute_[0mconsensus_returns_cache_on_hit(self, [0m[7m[0mmonkeypatch, [0m[7m[0mtmp_path):[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:138:        result = 
acc_mod.get_or_[7mcompute_[0mconsensus_accuracy("1d")[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:144:    def test_cache_hit_does_not_acquire_[7mcompute_[0mlock(self, [0m
[7m[0mmonkeypatch, [0m[7m[0mtmp_path):[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:156:        result = acc_mod.get_or_[7mcompute_[0maccuracy("1d")
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:269:        get_or_[7mcompute_[0m* [0m[7m[0mhelpers [0m[7m[0mshort-circuit [0m[7m[0mon [0m[7m[0mcache [0m
[7m[0mhit."""[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:508:    def test_releases_lock_even_on_[7mcompute_[0mexception(self, [0m
[7m[0mmonkeypatch, [0m[7m[0mtmp_path):[0m
Q:\finance-analyzer\tests\test_accuracy_compute_lock.py:509:        """If get_or_[7mcompute_[0m* [0m[7m[0mraises, [0m[7m[0mthe [0m[7m[0mfile [0m[7m[0mlock [0m[7m[0mmust [0m
[7m[0mstill [0m[7m[0mrelease[0m
Q:\finance-analyzer\tests\test_accuracy_snapshot_extras.py:123:        monkeypatch.setattr(forecast_mod, 
"[7mcompute_[0mforecast_accuracy", [0m[7m[0mstub)[0m
Q:\finance-analyzer\tests\test_accuracy_snapshot_extras.py:131:    def test_different_keys_[7mcompute_[0mseparately(self, [0m
[7m[0mmonkeypatch):[0m
Q:\finance-analyzer\tests\test_accuracy_snapshot_extras.py:139:        monkeypatch.setattr(forecast_mod, 
"[7mcompute_[0mforecast_accuracy", [0m[7m[0mstub)[0m
Q:\finance-analyzer\tests\test_accuracy_snapshot_extras.py:154:        monkeypatch.setattr(forecast_mod, 
"[7mcompute_[0mforecast_accuracy", [0m[7m[0mstub)[0m
Q:\finance-analyzer\tests\test_accuracy_stats.py:739:            "portfolio.accuracy_stats.get_or_[7mcompute_[0maccuracy", [0m
[7m[0mfake_acc,[0m
Q:\finance-analyzer\tests\test_accuracy_stats.py:742:            
"portfolio.accuracy_stats.get_or_[7mcompute_[0mconsensus_accuracy",[0m
Q:\finance-analyzer\tests\test_accuracy_stats.py:746:            
"portfolio.accuracy_stats.get_or_[7mcompute_[0mper_ticker_accuracy",[0m
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:195:        from portfolio.indicators import 
[7mcompute_[0mindicators[0m
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:204:        ind = [7mcompute_[0mindicators(df)
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:220:        from portfolio.indicators import 
[7mcompute_[0mindicators[0m
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:234:        ind = [7mcompute_[0mindicators(df)
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:242:            # (Action field is set when [7mcompute_[0mfn [0m
[7m[0mreturned [0m[7m[0ma [0m[7m[0mvalidated [0m[7m[0mresult[0m
Q:\finance-analyzer\tests\test_adversarial_review_round11.py:9:      <min_samples [7msignals.[0m
Q:\finance-analyzer\tests\test_agreement_rate.py:10:    _[7mcompute_[0magreement_rate,
Q:\finance-analyzer\tests\test_agreement_rate.py:11:    _[7mcompute_[0mdynamic_correlation_groups,
Q:\finance-analyzer\tests\test_agreement_rate.py:22:        rate, n = _[7mcompute_[0magreement_rate(a, b)
Q:\finance-analyzer\tests\test_agreement_rate.py:30:        rate, n = _[7mcompute_[0magreement_rate(a, b)
Q:\finance-analyzer\tests\test_agreement_rate.py:38:        rate, n = _[7mcompute_[0magreement_rate(a, b)
Q:\finance-analyzer\tests\test_agreement_rate.py:46:        rate, n = _[7mcompute_[0magreement_rate(a, b)
Q:\finance-analyzer\tests\test_agreement_rate.py:54:        rate, n = _[7mcompute_[0magreement_rate(a, b)
Q:\finance-analyzer\tests\test_agreement_rate.py:60:        rate, n = _[7mcompute_[0magreement_rate([], [])
Q:\finance-analyzer\tests\test_agreement_rate.py:68:        rate, n = _[7mcompute_[0magreement_rate(a, b)
Q:\finance-analyzer\tests\test_agreement_rate.py:112:        result = _[7mcompute_[0mdynamic_correlation_groups()
Q:\finance-analyzer\tests\test_agreement_rate.py:140:        result = _[7mcompute_[0mdynamic_correlation_groups()
Q:\finance-analyzer\tests\test_alpha_vantage.py:18:from [7mportfolio.signals[0m.claude_fundamental [0m[7m[0mimport [0m
[7m[0m_build_fundamentals_block[0m
Q:\finance-analyzer\tests\test_analyze.py:441:            "OVERALL: Mixed [7msignals.[0m"
Q:\finance-analyzer\tests\test_autonomous.py:301:        for sig in [7msignals.[0mvalues():
Q:\finance-analyzer\tests\test_batch2_defensive.py:17:        from [7mportfolio.signals[0m.news_event import _fetch_headlines
Q:\finance-analyzer\tests\test_batch2_defensive.py:22:        from [7mportfolio.signals[0m.news_event import _fetch_headlines
Q:\finance-analyzer\tests\test_batch2_defensive.py:26:    def test_[7mcompute_[0msignal_none_ticker_in_context(self):
Q:\finance-analyzer\tests\test_batch2_defensive.py:30:        from [7mportfolio.signals[0m.news_event [0m[7m[0mimport [0m
[7m[0mcompute_news_event_signal[0m
Q:\finance-analyzer\tests\test_batch2_defensive.py:33:        result = [7mcompute_[0mnews_event_signal(df, [0m
[7m[0mcontext={"ticker": [0m[7m[0mNone, [0m[7m[0m"config": [0m[7m[0m{}})[0m
Q:\finance-analyzer\tests\test_batch2_defensive.py:36:    def test_[7mcompute_[0msignal_no_context(self):
Q:\finance-analyzer\tests\test_batch2_defensive.py:40:        from [7mportfolio.signals[0m.news_event [0m[7m[0mimport [0m
[7m[0mcompute_news_event_signal[0m
Q:\finance-analyzer\tests\test_batch2_defensive.py:43:        result = [7mcompute_[0mnews_event_signal(df)
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:124:        from [7mportfolio.signals[0m.forecast [0m[7m[0mimport [0m
[7m[0mreset_circuit_breakers[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:131:        from [7mportfolio.signals[0m.forecast import (
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:177:        from [7mportfolio.signals[0m.forecast import (
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:236:        from [7mportfolio.signals[0m.calendar_seasonal [0m
[7m[0mimport [0m[7m[0mcompute_calendar_signal[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:249:        with 
patch("[7mportfolio.signals[0m.calendar_seasonal._day_of_week_effect",[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:251:            with caplog.at_level(logging.DEBUG, 
logger="[7mportfolio.signals[0m.calendar_seasonal"):[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:252:                result = [7mcompute_[0mcalendar_signal(df)
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:263:        from [7mportfolio.signals[0m.calendar_seasonal [0m
[7m[0mimport [0m[7m[0mcompute_calendar_signal[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:280:        patches_list = 
{f"[7mportfolio.signals[0m.calendar_seasonal.{s}": [0m[7m[0mValueError("fail")[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:283:        with caplog.at_level(logging.DEBUG, 
logger="[7mportfolio.signals[0m.calendar_seasonal"):[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:291:                result = [7mcompute_[0mcalendar_signal(df)
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:356:        import [7mportfolio.signals[0m.forecast as fmod
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:379:        from [7mportfolio.signals[0m.forecast [0m[7m[0mimport [0m
[7m[0m_PREDICTION_DEDUP_EVICT_AGE[0m
Q:\finance-analyzer\tests\test_bug_fixes_session_20260321.py:384:        from [7mportfolio.signals[0m.forecast [0m[7m[0mimport [0m
[7m[0m_forecast_lock[0m
Q:\finance-analyzer\tests\test_bug_fixes_session4.py:23:    """Dashboard /api/signal-heatmap should list all 19 
enhanced [7msignals.[0m"""[0m
Q:\finance-analyzer\tests\test_confidence_penalties.py:3:Covers apply_confidence_penalties() (4-stage cascade) and 
_[7mcompute_[0madx().[0m
Q:\finance-analyzer\tests\test_confidence_penalties.py:10:from portfolio.signal_engine import _[7mcompute_[0madx, [0m
[7m[0mapply_confidence_penalties[0m
Q:\finance-analyzer\tests\test_confidence_penalties.py:69:# TEST-6: _[7mcompute_[0madx tests
Q:\finance-analyzer\tests\test_confidence_penalties.py:73:    """Tests for _[7mcompute_[0madx()."""
Q:\finance-analyzer\tests\test_confidence_penalties.py:76:        assert _[7mcompute_[0madx(None) is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:79:        assert _[7mcompute_[0madx("not a dataframe") is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:80:        assert _[7mcompute_[0madx(42) is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:81:        assert _[7mcompute_[0madx([1, 2, 3]) is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:86:        assert _[7mcompute_[0madx(df) is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:90:        assert _[7mcompute_[0madx(df, period=14) is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:94:        result = _[7mcompute_[0madx(df, period=14)
Q:\finance-analyzer\tests\test_confidence_penalties.py:101:        adx = _[7mcompute_[0madx(df)
Q:\finance-analyzer\tests\test_confidence_penalties.py:108:        adx = _[7mcompute_[0madx(df)
Q:\finance-analyzer\tests\test_confidence_penalties.py:116:        adx = _[7mcompute_[0madx(df)
Q:\finance-analyzer\tests\test_confidence_penalties.py:127:        result = _[7mcompute_[0madx(df)
Q:\finance-analyzer\tests\test_confidence_penalties.py:132:        adx_up = _[7mcompute_[0madx(df_up)
Q:\finance-analyzer\tests\test_confidence_penalties.py:138:        adx = _[7mcompute_[0madx(df, period=7)
Q:\finance-analyzer\tests\test_confidence_penalties.py:144:        assert _[7mcompute_[0madx(df) is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:148:        assert _[7mcompute_[0madx(df) is None
Q:\finance-analyzer\tests\test_confidence_penalties.py:159:        result = _[7mcompute_[0madx(df)
Q:\finance-analyzer\tests\test_confidence_penalties.py:404:        """_[7mcompute_[0madx [0m[7m[0mresult [0m[7m[0mshould [0m[7m[0mbe [0m[7m[0mstored [0m[7m[0min [0m
[7m[0mextra_info['_adx']."""[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:7:from [7mportfolio.signals[0m.crypto_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_crypto_cross_asset_signal[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:17:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([100, [0m[7m[0m101]),[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:25:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]),[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:33:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:42:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:50:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:63:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([3500, [0m[7m[0m3675]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:72:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([3500, [0m[7m[0m3325]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:82:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:90:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:100:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:112:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:122:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:137:        out = 
[7mcompute_[0mcrypto_cross_asset_signal(_df([105000, [0m[7m[0m105100]), [0m[7m[0m{[0m
Q:\finance-analyzer\tests\test_crypto_data.py:134:    def test_[7mcompute_[0mpremium(self):
Q:\finance-analyzer\tests\test_crypto_data.py:135:        from crypto_data import [7mcompute_[0mmstr_btc_nav
Q:\finance-analyzer\tests\test_crypto_data.py:136:        result = [7mcompute_[0mmstr_btc_nav(mstr_price=300, [0m
[7m[0mbtc_price=67000)[0m
Q:\finance-analyzer\tests\test_crypto_data.py:145:        from crypto_data import [7mcompute_[0mmstr_btc_nav
Q:\finance-analyzer\tests\test_crypto_data.py:146:        assert [7mcompute_[0mmstr_btc_nav(0, 67000) is None
Q:\finance-analyzer\tests\test_crypto_data.py:147:        assert [7mcompute_[0mmstr_btc_nav(300, 0) is None
Q:\finance-analyzer\tests\test_crypto_data.py:148:        assert [7mcompute_[0mmstr_btc_nav(0, 0) is None
Q:\finance-analyzer\tests\test_crypto_macro.py:87:        from portfolio.crypto_macro_data import 
[7mcompute_[0mgold_btc_ratio[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:99:            result = [7mcompute_[0mgold_btc_ratio()
Q:\finance-analyzer\tests\test_crypto_macro.py:107:        from portfolio.crypto_macro_data import 
[7mcompute_[0mgold_btc_ratio[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:112:            result = [7mcompute_[0mgold_btc_ratio()
Q:\finance-analyzer\tests\test_crypto_macro.py:136:        from [7mportfolio.signals[0m.crypto_macro [0m[7m[0mimport [0m
[7m[0mcompute_crypto_macro_signal[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:138:        result = [7mcompute_[0mcrypto_macro_signal(
Q:\finance-analyzer\tests\test_crypto_macro.py:147:        from [7mportfolio.signals[0m.crypto_macro [0m[7m[0mimport [0m
[7m[0mcompute_crypto_macro_signal[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:149:        with patch("[7mportfolio.signals[0m.crypto_macro._cached", [0m
[7m[0mreturn_value=None):[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:150:            result = [7mcompute_[0mcrypto_macro_signal(
Q:\finance-analyzer\tests\test_crypto_macro.py:159:        from [7mportfolio.signals[0m.crypto_macro import _options_gravity
Q:\finance-analyzer\tests\test_crypto_macro.py:167:        from [7mportfolio.signals[0m.crypto_macro import _options_gravity
Q:\finance-analyzer\tests\test_crypto_macro.py:175:        from [7mportfolio.signals[0m.crypto_macro import _options_gravity
Q:\finance-analyzer\tests\test_crypto_macro.py:183:        from [7mportfolio.signals[0m.crypto_macro import _options_gravity
Q:\finance-analyzer\tests\test_crypto_macro.py:191:        from [7mportfolio.signals[0m.crypto_macro [0m[7m[0mimport [0m
[7m[0m_put_call_sentiment[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:199:        from [7mportfolio.signals[0m.crypto_macro [0m[7m[0mimport [0m
[7m[0m_put_call_sentiment[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:207:        from [7mportfolio.signals[0m.crypto_macro import _gold_rotation
Q:\finance-analyzer\tests\test_crypto_macro.py:215:        from [7mportfolio.signals[0m.crypto_macro import _gold_rotation
Q:\finance-analyzer\tests\test_crypto_macro.py:223:        from [7mportfolio.signals[0m.crypto_macro [0m[7m[0mimport [0m
[7m[0m_exchange_netflow_signal[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:235:        from [7mportfolio.signals[0m.crypto_macro [0m[7m[0mimport [0m
[7m[0m_exchange_netflow_signal[0m
Q:\finance-analyzer\tests\test_crypto_macro.py:247:        from [7mportfolio.signals[0m.crypto_macro import _expiry_proximity
Q:\finance-analyzer\tests\test_crypto_precompute.py:60:        result = cp.maybe_pre[7mcompute_[0mcrypto({})
Q:\finance-analyzer\tests\test_crypto_precompute.py:80:        result = cp.maybe_pre[7mcompute_[0mcrypto({})
Q:\finance-analyzer\tests\test_crypto_precompute.py:90:    assert callable(cp.maybe_pre[7mcompute_[0mcrypto)
Q:\finance-analyzer\tests\test_cumulative_tracker.py:82:# [7mcompute_[0mrolling_changes
Q:\finance-analyzer\tests\test_cumulative_tracker.py:86:    """Tests for [7mcompute_[0mrolling_changes()."""
Q:\finance-analyzer\tests\test_cumulative_tracker.py:89:        from portfolio.cumulative_tracker import 
[7mcompute_[0mrolling_changes[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:97:        result = [7mcompute_[0mrolling_changes(snapshots=snapshots)
Q:\finance-analyzer\tests\test_cumulative_tracker.py:107:        from portfolio.cumulative_tracker import 
[7mcompute_[0mrolling_changes[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:113:        result = [7mcompute_[0mrolling_changes(snapshots=snapshots)
Q:\finance-analyzer\tests\test_cumulative_tracker.py:117:        from portfolio.cumulative_tracker import 
[7mcompute_[0mrolling_changes[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:118:        result = [7mcompute_[0mrolling_changes(snapshots=[])
Q:\finance-analyzer\tests\test_cumulative_tracker.py:122:        from portfolio.cumulative_tracker import 
[7mcompute_[0mrolling_changes[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:124:        result = [7mcompute_[0mrolling_changes(snapshots=snapshots)
Q:\finance-analyzer\tests\test_cumulative_tracker.py:130:        from portfolio.cumulative_tracker import 
[7mcompute_[0mrolling_changes[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:136:        result = [7mcompute_[0mrolling_changes(tickers=["XAG-USD"], [0m
[7m[0msnapshots=snapshots)[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:141:        from portfolio.cumulative_tracker import 
[7mcompute_[0mrolling_changes[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:147:        result = [7mcompute_[0mrolling_changes(snapshots=snapshots)
Q:\finance-analyzer\tests\test_cumulative_tracker.py:152:        from portfolio.cumulative_tracker import 
[7mcompute_[0mrolling_changes[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:157:        result = [7mcompute_[0mrolling_changes(snapshots=snapshots)
Q:\finance-analyzer\tests\test_cumulative_tracker.py:210:    
@patch("portfolio.cumulative_tracker.[7mcompute_[0mrolling_changes")[0m
Q:\finance-analyzer\tests\test_cumulative_tracker.py:235:    
@patch("portfolio.cumulative_tracker.[7mcompute_[0mrolling_changes")[0m
Q:\finance-analyzer\tests\test_dashboard.py:433:        # The endpoint now uses get_or_[7mcompute_[0m* [0m[7m[0mwrappers [0m[7m[0m(2026-05-03 [0m
[7m[0mperf[0m
Q:\finance-analyzer\tests\test_dashboard.py:440:                
get_or_[7mcompute_[0maccuracy=MagicMock(return_value=mock_sa),[0m
Q:\finance-analyzer\tests\test_dashboard.py:441:                
get_or_[7mcompute_[0mconsensus_accuracy=MagicMock(return_value=mock_ca),[0m
Q:\finance-analyzer\tests\test_dashboard.py:442:                
get_or_[7mcompute_[0mper_ticker_accuracy=MagicMock(return_value=mock_ta),[0m
Q:\finance-analyzer\tests\test_dashboard.py:459:        [7mcompute_[0mcalls = {"n": 0}
Q:\finance-analyzer\tests\test_dashboard.py:462:            [7mcompute_[0mcalls["n"] += 1
Q:\finance-analyzer\tests\test_dashboard.py:467:                
get_or_[7mcompute_[0maccuracy=MagicMock(side_effect=_track_call),[0m
Q:\finance-analyzer\tests\test_dashboard.py:468:                
get_or_[7mcompute_[0mconsensus_accuracy=MagicMock(side_effect=_track_call),[0m
Q:\finance-analyzer\tests\test_dashboard.py:469:                
get_or_[7mcompute_[0mper_ticker_accuracy=MagicMock(side_effect=_track_call),[0m
Q:\finance-analyzer\tests\test_dashboard.py:481:        assert [7mcompute_[0mcalls["n"] == 12, (
Q:\finance-analyzer\tests\test_dashboard.py:482:            f"Expected 12 wrapper calls (one full fanout), got 
{[7mcompute_[0mcalls['n']}"[0m
Q:\finance-analyzer\tests\test_dashboard.py:498:                get_or_[7mcompute_[0maccuracy=MagicMock(return_value={}),
Q:\finance-analyzer\tests\test_dashboard.py:499:                
get_or_[7mcompute_[0mconsensus_accuracy=MagicMock(side_effect=_ca_for),[0m
Q:\finance-analyzer\tests\test_dashboard.py:500:                
get_or_[7mcompute_[0mper_ticker_accuracy=MagicMock(return_value={}),[0m
Q:\finance-analyzer\tests\test_dashboard.py:516:                get_or_[7mcompute_[0maccuracy=MagicMock(return_value=None),
Q:\finance-analyzer\tests\test_dashboard.py:517:                
get_or_[7mcompute_[0mconsensus_accuracy=MagicMock(return_value=None),[0m
Q:\finance-analyzer\tests\test_dashboard.py:518:                
get_or_[7mcompute_[0mper_ticker_accuracy=MagicMock(return_value=None),[0m
Q:\finance-analyzer\tests\test_data_collector.py:11:  - collect_timeframes: cache hit, cache miss, [7mcompute_[0mindicators [0m
[7m[0mreturning [0m[7m[0mNone [0m[7m[0m(insufficient[0m
Q:\finance-analyzer\tests\test_data_collector.py:565:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:581:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:596:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:618:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:636:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:638:        """When [7mcompute_[0mindicators [0m[7m[0mreturns [0m[7m[0mNone [0m[7m[0m(insufficient [0m
[7m[0mdata), [0m[7m[0mthe [0m[7m[0mtimeframe [0m[7m[0mis [0m[7m[0mskipped."""[0m
Q:\finance-analyzer\tests\test_data_collector.py:648:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:671:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:707:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:725:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_data_collector.py:740:    @patch("portfolio.data_collector.[7mcompute_[0mindicators")
Q:\finance-analyzer\tests\test_disabled_signal_rescue.py:12:    """Verify the rescue mechanism for globally disabled 
[7msignals.[0m"""[0m
Q:\finance-analyzer\tests\test_disabled_signal_rescue.py:47:        """Every override should reference a signal in 
DISABLED_[7mSIGNALS.[0m"""[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:1:"""Tests for [7mportfolio.signals[0m.dxy_cross_asset."""
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:22:        from [7mportfolio.signals[0m.dxy_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_dxy_cross_asset_signal[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:27:        result = 
[7mcompute_[0mdxy_cross_asset_signal(_make_df(), [0m[7m[0mticker="XAG-USD")[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:33:        from [7mportfolio.signals[0m.dxy_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_dxy_cross_asset_signal[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:38:        result = 
[7mcompute_[0mdxy_cross_asset_signal(_make_df(), [0m[7m[0mticker="XAG-USD")[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:43:        from [7mportfolio.signals[0m.dxy_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_dxy_cross_asset_signal[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:48:        result = 
[7mcompute_[0mdxy_cross_asset_signal(_make_df(), [0m[7m[0mticker="XAU-USD")[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:54:        from [7mportfolio.signals[0m.dxy_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_dxy_cross_asset_signal[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:59:        result = 
[7mcompute_[0mdxy_cross_asset_signal(_make_df(), [0m[7m[0mticker="XAG-USD")[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:65:        from [7mportfolio.signals[0m.dxy_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_dxy_cross_asset_signal[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:66:        result = 
[7mcompute_[0mdxy_cross_asset_signal(_make_df(), [0m[7m[0mticker="NVDA")[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:72:        from [7mportfolio.signals[0m.dxy_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_dxy_cross_asset_signal[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:73:        result = 
[7mcompute_[0mdxy_cross_asset_signal(_make_df(), [0m[7m[0mticker="BTC-USD")[0m
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:79:        from [7mportfolio.signals[0m.dxy_cross_asset [0m[7m[0mimport [0m
[7m[0mcompute_dxy_cross_asset_signal[0m


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\portfolio\\signal_registry.py -Pattern 'realized_skewness|portfolio.signals'" in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\requirements.txt" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 698ms:
# Core runtime
pandas
numpy
requests

# Machine Learning
scikit-learn
joblib

# Web Dashboard
flask
markdown  # used by dashboard/house_blueprint.py for the /house viewer

# Testing
pytest

# Optional: LLM fine-tuning (install separately in training venv)
# torch
# transformers
# datasets
# peft

 succeeded in 803ms:

Q:\finance-analyzer\portfolio\signal_registry.py:29:        module_path: Full module path (e.g., 
"[7mportfolio.signals[0m.trend")[0m
Q:\finance-analyzer\portfolio\signal_registry.py:110:        ("trend", "[7mportfolio.signals[0m.trend", [0m
[7m[0m"compute_trend_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:111:        ("momentum", "[7mportfolio.signals[0m.momentum", [0m
[7m[0m"compute_momentum_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:112:        ("volume_flow", "[7mportfolio.signals[0m.volume_flow", [0m
[7m[0m"compute_volume_flow_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:113:        ("volatility_sig", "[7mportfolio.signals[0m.volatility", [0m
[7m[0m"compute_volatility_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:114:        ("candlestick", "[7mportfolio.signals[0m.candlestick", [0m
[7m[0m"compute_candlestick_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:115:        ("structure", "[7mportfolio.signals[0m.structure", [0m
[7m[0m"compute_structure_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:116:        ("fibonacci", "[7mportfolio.signals[0m.fibonacci", [0m
[7m[0m"compute_fibonacci_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:117:        ("smart_money", "[7mportfolio.signals[0m.smart_money", [0m
[7m[0m"compute_smart_money_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:118:        ("oscillators", "[7mportfolio.signals[0m.oscillators", [0m
[7m[0m"compute_oscillator_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:119:        ("heikin_ashi", "[7mportfolio.signals[0m.heikin_ashi", [0m
[7m[0m"compute_heikin_ashi_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:120:        ("calendar", "[7mportfolio.signals[0m.calendar_seasonal", [0m
[7m[0m"compute_calendar_signal"),[0m
Q:\finance-analyzer\portfolio\signal_registry.py:125:    register_enhanced("mean_reversion", 
"[7mportfolio.signals[0m.mean_reversion",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:127:    register_enhanced("momentum_factors", 
"[7mportfolio.signals[0m.momentum_factors",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:130:    register_enhanced("macro_regime", 
"[7mportfolio.signals[0m.macro_regime",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:133:    register_enhanced("news_event", 
"[7mportfolio.signals[0m.news_event",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:135:    register_enhanced("econ_calendar", 
"[7mportfolio.signals[0m.econ_calendar",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:138:    register_enhanced("forecast", "[7mportfolio.signals[0m.forecast",
Q:\finance-analyzer\portfolio\signal_registry.py:141:    register_enhanced("claude_fundamental", 
"[7mportfolio.signals[0m.claude_fundamental",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:144:    register_enhanced("futures_flow", 
"[7mportfolio.signals[0m.futures_flow",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:147:    register_enhanced("crypto_macro", 
"[7mportfolio.signals[0m.crypto_macro",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:150:    register_enhanced("orderbook_flow", 
"[7mportfolio.signals[0m.orderbook_flow",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:153:    register_enhanced("metals_cross_asset", 
"[7mportfolio.signals[0m.metals_cross_asset",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:158:    register_enhanced("dxy_cross_asset", 
"[7mportfolio.signals[0m.dxy_cross_asset",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:161:    register_enhanced("cot_positioning", 
"[7mportfolio.signals[0m.cot_positioning",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:164:    register_enhanced("credit_spread_risk", 
"[7mportfolio.signals[0m.credit_spread",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:167:    register_enhanced("futures_basis", 
"[7mportfolio.signals[0m.futures_basis",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:170:    register_enhanced("hurst_regime", 
"[7mportfolio.signals[0m.hurst_regime",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:173:    register_enhanced("shannon_entropy", 
"[7mportfolio.signals[0m.shannon_entropy",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:176:    register_enhanced("vix_term_structure", 
"[7mportfolio.signals[0m.vix_term_structure",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:179:    register_enhanced("gold_real_yield_paradox", 
"[7mportfolio.signals[0m.gold_real_yield_paradox",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:182:    register_enhanced("cross_asset_tsmom", 
"[7mportfolio.signals[0m.cross_asset_tsmom",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:187:    register_enhanced("copper_gold_ratio", 
"[7mportfolio.signals[0m.copper_gold_ratio",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:192:    register_enhanced("statistical_jump_regime", 
"[7mportfolio.signals[0m.statistical_jump_regime",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:197:    register_enhanced("network_momentum", 
"[7mportfolio.signals[0m.network_momentum",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:203:    register_enhanced("ovx_metals_spillover", 
"[7mportfolio.signals[0m.ovx_metals_spillover",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:208:    register_enhanced("xtrend_equity_spillover", 
"[7mportfolio.signals[0m.xtrend_equity_spillover",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:213:    register_enhanced("complexity_gap_regime", 
"[7mportfolio.signals[0m.complexity_gap_regime",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:218:    register_enhanced("[7mrealized_skewness[0m", [0m
[7m[0m"portfolio.signals.realized_skewness",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:219:                      "compute_[7mrealized_skewness[0m_signal")
Q:\finance-analyzer\portfolio\signal_registry.py:223:    register_enhanced("mahalanobis_turbulence", 
"[7mportfolio.signals[0m.mahalanobis_turbulence",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:229:    register_enhanced("crypto_evrp", 
"[7mportfolio.signals[0m.crypto_evrp",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:235:    register_enhanced("hash_ribbons", 
"[7mportfolio.signals[0m.hash_ribbons",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:241:    register_enhanced("drift_regime_gate", 
"[7mportfolio.signals[0m.drift_regime_gate",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:247:    register_enhanced("vol_ratio_regime", 
"[7mportfolio.signals[0m.vol_ratio_regime",[0m
Q:\finance-analyzer\portfolio\signal_registry.py:254:                      "[7mportfolio.signals[0m.residual_pair_reversion",
Q:\finance-analyzer\portfolio\signal_registry.py:263:                      "[7mportfolio.signals[0m.williams_vix_fix",
Q:\finance-analyzer\portfolio\signal_registry.py:271:                      "[7mportfolio.signals[0m.treasury_risk_rotation",


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signal_registry.py -TotalCount 360" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 678ms:
"""Signal registry — plugin system for modular signal management.

Signals register themselves via @register_signal decorator. signal_engine.py
discovers all signals from the registry instead of hardcoded lists.
"""
import importlib
import logging
import time
from collections.abc import Callable

logger = logging.getLogger("portfolio.signal_registry")

# Registry storage
_CORE_SIGNALS: dict[str, dict] = {}
_ENHANCED_SIGNALS: dict[str, dict] = {}


def register_signal(name: str, signal_type: str = "enhanced",
                    module_path: str = None, func_name: str = None,
                    requires_macro: bool = False):
    """Register a signal in the global registry.

    Can be used as a decorator on compute functions, or called directly
    to register signals programmatically.

    Args:
        name: Signal name (e.g., "trend", "rsi")
        signal_type: "core" or "enhanced"
        module_path: Full module path (e.g., "portfolio.signals.trend")
        func_name: Function name to call (e.g., "compute_trend_signal")
        requires_macro: Whether this signal needs macro context (only macro_regime)
    """
    def decorator(func):
        entry = {
            "name": name,
            "type": signal_type,
            "module_path": module_path or func.__module__,
            "func_name": func_name or func.__name__,
            "requires_macro": requires_macro,
            "func": func,
        }
        if signal_type == "core":
            _CORE_SIGNALS[name] = entry
        else:
            _ENHANCED_SIGNALS[name] = entry
        return func
    return decorator


def register_enhanced(name: str, module_path: str, func_name: str,
                      requires_macro: bool = False,
                      requires_context: bool = False,
                      max_confidence: float = 1.0):
    """Programmatically register an enhanced signal module."""
    _ENHANCED_SIGNALS[name] = {
        "name": name,
        "type": "enhanced",
        "module_path": module_path,
        "func_name": func_name,
        "requires_macro": requires_macro,
        "requires_context": requires_context,
        "max_confidence": max_confidence,
        "func": None,  # lazy-loaded
    }


def get_enhanced_signals() -> dict[str, dict]:
    """Return all registered enhanced signals."""
    return dict(_ENHANCED_SIGNALS)


def get_signal_names() -> list:
    """Return all signal names (core + enhanced) in order."""
    core = list(_CORE_SIGNALS.keys())
    enhanced = list(_ENHANCED_SIGNALS.keys())
    return core + enhanced


_FAILED_IMPORT_SENTINEL = object()
_FAILED_IMPORT_COOLDOWN = 300  # retry broken imports after 5 min

def load_signal_func(entry: dict) -> Callable | None:
    """Lazy-load and cache the compute function for a signal.

    On import failure, caches the failure for _FAILED_IMPORT_COOLDOWN seconds
    so the warning is logged once, not 35× per cycle (5 tickers × 7 TFs).
    """
    cached = entry.get("func")
    if cached is not None and cached is not _FAILED_IMPORT_SENTINEL:
        return cached
    if cached is _FAILED_IMPORT_SENTINEL and time.monotonic() - entry.get("_fail_ts", 0) < _FAILED_IMPORT_COOLDOWN:
        return None
    try:
        mod = importlib.import_module(entry["module_path"])
        func = getattr(mod, entry["func_name"])
        entry["func"] = func
        entry.pop("_fail_ts", None)
        return func
    except Exception as e:
        logger.warning("Failed to load signal %s: %s", entry['name'], e)
        entry["func"] = _FAILED_IMPORT_SENTINEL
        entry["_fail_ts"] = time.monotonic()
        return None


# Register all enhanced signals (called at import time)
def _register_defaults():
    """Register the default set of enhanced signal modules."""
    defaults = [
        ("trend", "portfolio.signals.trend", "compute_trend_signal"),
        ("momentum", "portfolio.signals.momentum", "compute_momentum_signal"),
        ("volume_flow", "portfolio.signals.volume_flow", "compute_volume_flow_signal"),
        ("volatility_sig", "portfolio.signals.volatility", "compute_volatility_signal"),
        ("candlestick", "portfolio.signals.candlestick", "compute_candlestick_signal"),
        ("structure", "portfolio.signals.structure", "compute_structure_signal"),
        ("fibonacci", "portfolio.signals.fibonacci", "compute_fibonacci_signal"),
        ("smart_money", "portfolio.signals.smart_money", "compute_smart_money_signal"),
        ("oscillators", "portfolio.signals.oscillators", "compute_oscillator_signal"),
        ("heikin_ashi", "portfolio.signals.heikin_ashi", "compute_heikin_ashi_signal"),
        ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
    ]
    for name, mod_path, func_name in defaults:
        register_enhanced(name, mod_path, func_name)
    # mean_reversion and momentum_factors require context for seasonality detrending
    register_enhanced("mean_reversion", "portfolio.signals.mean_reversion",
                      "compute_mean_reversion_signal", requires_context=True)
    register_enhanced("momentum_factors", "portfolio.signals.momentum_factors",
                      "compute_momentum_factors_signal", requires_context=True)
    # macro_regime is special — requires_macro=True
    register_enhanced("macro_regime", "portfolio.signals.macro_regime",
                      "compute_macro_regime_signal", requires_macro=True)
    # news_event and econ_calendar require context (ticker, config); capped at 0.7
    register_enhanced("news_event", "portfolio.signals.news_event",
                      "compute_news_event_signal", requires_context=True, max_confidence=0.7)
    register_enhanced("econ_calendar", "portfolio.signals.econ_calendar",
                      "compute_econ_calendar_signal", requires_context=True, max_confidence=0.7)
    # forecast signal — Kronos + Chronos price direction prediction; capped at 0.7
    register_enhanced("forecast", "portfolio.signals.forecast",
                      "compute_forecast_signal", requires_context=True, max_confidence=0.7)
    # Claude fundamental — three-tier LLM cascade; capped at 0.7
    register_enhanced("claude_fundamental", "portfolio.signals.claude_fundamental",
                      "compute_claude_fundamental_signal", requires_context=True, max_confidence=0.7)
    # Futures flow — OI, LS ratios, funding history (crypto only); capped at 0.7
    register_enhanced("futures_flow", "portfolio.signals.futures_flow",
                      "compute_futures_flow_signal", requires_context=True, max_confidence=0.7)
    # Crypto macro — options max pain, gold-BTC rotation, exchange reserves (crypto only); capped at 0.7
    register_enhanced("crypto_macro", "portfolio.signals.crypto_macro",
                      "compute_crypto_macro_signal", requires_context=True, max_confidence=0.7)
    # Orderbook flow — microstructure metrics (metals + crypto); capped at 0.7
    register_enhanced("orderbook_flow", "portfolio.signals.orderbook_flow",
                      "compute_orderbook_flow_signal", requires_context=True, max_confidence=0.7)
    # Metals cross-asset — copper, GVZ, G/S ratio, SPY, oil (metals only); capped at 0.7
    register_enhanced("metals_cross_asset", "portfolio.signals.metals_cross_asset",
                      "compute_metals_cross_asset_signal", requires_context=True, max_confidence=0.7)
    # DXY cross-asset — intraday USD index inverse correlation (metals only); capped at 0.8
    # 2026-04-13: added standalone signal to capture DXY R²~0.6 vs silver at
    # 1-3h horizon. Complements macro_regime's daily DXY sub-indicator.
    register_enhanced("dxy_cross_asset", "portfolio.signals.dxy_cross_asset",
                      "compute_dxy_cross_asset_signal", requires_context=True, max_confidence=0.8)
    # COT positioning — CFTC speculative/commercial positioning, contrarian (metals only); capped at 0.7
    register_enhanced("cot_positioning", "portfolio.signals.cot_positioning",
                      "compute_cot_positioning_signal", requires_context=True, max_confidence=0.7)
    # Credit spread risk — HY OAS from FRED as cross-asset risk appetite gauge; capped at 0.7
    register_enhanced("credit_spread_risk", "portfolio.signals.credit_spread",
                      "compute_credit_spread_signal", requires_context=True, max_confidence=0.7)
    # Futures basis regime — mark-index spread, contango/backwardation detection; capped at 0.7
    register_enhanced("futures_basis", "portfolio.signals.futures_basis",
                      "compute_futures_basis_signal", requires_context=True, max_confidence=0.7)
    # Hurst regime detector — R/S analysis for trending/MR/random-walk classification
    register_enhanced("hurst_regime", "portfolio.signals.hurst_regime",
                      "compute_hurst_regime_signal", requires_context=True)
    # Shannon entropy — market noise/predictability filter; low entropy = trending
    register_enhanced("shannon_entropy", "portfolio.signals.shannon_entropy",
                      "compute_shannon_entropy_signal")
    # VIX term structure — contango/backwardation regime detection; capped at 0.7
    register_enhanced("vix_term_structure", "portfolio.signals.vix_term_structure",
                      "compute_vix_term_structure_signal", requires_context=True, max_confidence=0.7)
    # Gold real yield paradox — GYDI regime detector (metals only); capped at 0.7
    register_enhanced("gold_real_yield_paradox", "portfolio.signals.gold_real_yield_paradox",
                      "compute_gold_real_yield_paradox_signal", requires_context=True, max_confidence=0.7)
    # Cross-asset TSMOM — bond/equity momentum predicts target asset direction; capped at 0.7
    register_enhanced("cross_asset_tsmom", "portfolio.signals.cross_asset_tsmom",
                      "compute_cross_asset_tsmom_signal", requires_context=True, max_confidence=0.7)
    # Copper/gold ratio — intermarket regime indicator (all assets); capped at 0.7
    # 2026-04-17: cross-asset signal, inverts direction for metals (falling ratio = gold strength).
    # 94% recession prediction accuracy, 0.85 correlation with 10Y yields.
    register_enhanced("copper_gold_ratio", "portfolio.signals.copper_gold_ratio",
                      "compute_copper_gold_ratio_signal", requires_context=True, max_confidence=0.7)
    # Statistical Jump Model regime — jump detection + persistence penalty (all assets)
    # 2026-04-18: addresses failing macro_regime (46.6% at 1d, 30.3% metals).
    # Source: Shu, Yu, Mulvey 2024 (12 citations). Persistence penalty prevents whiplash.
    register_enhanced("statistical_jump_regime", "portfolio.signals.statistical_jump_regime",
                      "compute_statistical_jump_regime_signal")
    # Network momentum — cross-asset momentum spillover (all assets); capped at 0.7
    # 2026-04-19: Pu et al. 2023 (arXiv:2308.11294). Sharpe 1.511 across 64 futures.
    # Simplified: correlation-weighted peer momentum divergence instead of GNN.
    register_enhanced("network_momentum", "portfolio.signals.network_momentum",
                      "compute_network_momentum_signal", requires_context=True, max_confidence=0.7)
    # OVX metals spillover — oil implied volatility as metals predictor; capped at 0.7
    # 2026-04-20: OVX at extreme quantiles predicts precious metals returns via
    # contagion/inflation/dollar channels. Distinct from metals_cross_asset (oil PRICE).
    # Source: ScienceDirect OVX cross-asset quantile predictability papers.
    register_enhanced("ovx_metals_spillover", "portfolio.signals.ovx_metals_spillover",
                      "compute_ovx_metals_spillover_signal", requires_context=True, max_confidence=0.7)
    # Cross-asset equity trend spillover — SPY/QQQ TA predicts all assets; capped at 0.7
    # 2026-04-21: Fieberg et al. 2025, robust across 1.3M research designs.
    # Inverts for safe havens (XAU, XAG): bullish equities = risk-on = SELL metals.
    register_enhanced("xtrend_equity_spillover", "portfolio.signals.xtrend_equity_spillover",
                      "compute_xtrend_equity_spillover_signal", requires_context=True, max_confidence=0.7)
    # Complexity gap regime — RMT-based market structure/synchronization detector; capped at 0.7
    # 2026-04-22: Mukhia et al. 2026, arXiv:2604.19107. Complexity gap = norm_max_eigenvalue -
    # avg_pairwise_corr. Gap collapse = crisis synchronization. Inverts for safe havens.
    register_enhanced("complexity_gap_regime", "portfolio.signals.complexity_gap_regime",
                      "compute_complexity_gap_regime_signal", requires_context=True, max_confidence=0.7)
    # Realized skewness — 3rd moment directional signal (all assets)
    # 2026-04-23: Fernandez-Perez et al. 2018. Sharpe 0.79, 8.01% annual on
    # 27 commodity futures. Z-scored skewness + kurtosis confirmation.
    register_enhanced("realized_skewness", "portfolio.signals.realized_skewness",
                      "compute_realized_skewness_signal")
    # Mahalanobis turbulence — cross-asset regime detection via Mahalanobis distance
    # 2026-04-24: Kritzman & Li (2010). Sharpe 2.20 vs 1.0 B&H, max DD 6% vs 32%.
    # Measures statistical unusualness of multi-asset returns. Includes absorption ratio.
    register_enhanced("mahalanobis_turbulence", "portfolio.signals.mahalanobis_turbulence",
                      "compute_mahalanobis_turbulence_signal", requires_context=True, max_confidence=0.7)
    # Crypto eVRP — Expected Volatility Risk Premium (crypto only); capped at 0.7
    # 2026-04-25: Zarattini, Mele & Aziz (2025). eVRP = DVOL(30d) - RV(10d).
    # Options-derived signal uncorrelated with trend-following cluster.
    # Deribit public API, no auth. BTC + ETH only.
    register_enhanced("crypto_evrp", "portfolio.signals.crypto_evrp",
                      "compute_crypto_evrp_signal", requires_context=True, max_confidence=0.7)
    # Hash Ribbons BTC — miner capitulation detector (BTC-only); capped at 0.7
    # 2026-04-26: Charles Edwards / Capriole Investments. 89% win rate on 9 signals.
    # 30d/60d SMA hashrate crossover with price momentum confirmation.
    # blockchain.info API (free, no auth). Fires ~1/year but extreme conviction.
    register_enhanced("hash_ribbons", "portfolio.signals.hash_ribbons",
                      "compute_hash_ribbons_signal", requires_context=True, max_confidence=0.7)
    # Drift Regime Gate — positive-day fraction regime detector; capped at 0.7
    # 2026-04-28: arxiv:2511.12490 (2025). OOS Sharpe >13 on 20yr walk-forward.
    # Fraction of positive close-to-close days in 63-bar window detects drift regime.
    # Pure OHLCV, all 5 tickers. Directional via SMA distance.
    register_enhanced("drift_regime_gate", "portfolio.signals.drift_regime_gate",
                      "compute_drift_regime_gate_signal", max_confidence=0.7)
    # Vol ratio regime — GK/CC volatility ratio + VR test + ER as regime detector
    # 2026-04-29: Garman & Klass (1980), Lo & MacKinlay (1988), Kaufman ER.
    # Three orthogonal regime measures: GK/CC ratio, variance ratio, efficiency ratio.
    # Directional: mean-reversion in ranging, momentum in trending. All OHLCV, all assets.
    register_enhanced("vol_ratio_regime", "portfolio.signals.vol_ratio_regime",
                      "compute_vol_ratio_regime_signal", max_confidence=0.7)
    # Residual pair reversion — cointegration-based pairs trading (all assets)
    # 2026-04-30: Leung & Nguyen (2018), Amberdata crypto pairs (Sharpe 0.93).
    # Regime-neutral mean reversion on OLS residual: ETH~BTC, XAG~XAU, MSTR~BTC.
    # Specifically targets ETH-USD (weakest ticker) and XAG-USD (user focus).
    register_enhanced("residual_pair_reversion",
                      "portfolio.signals.residual_pair_reversion",
                      "compute_residual_pair_reversion_signal",
                      requires_context=True, max_confidence=0.7)
    # Williams VIX Fix — synthetic volatility bottom/capitulation detector (all assets)
    # 2026-05-01: Larry Williams (2007). Profit factor 2.0, 322 trades 1993-2024.
    # WVF = (highest_close_22 - low) / highest_close_22 * 100. Spikes above upper
    # Bollinger Band signal extreme fear = BUY. Directly targets system's BUY-side
    # accuracy weakness. Pure OHLCV, all 5 tickers.
    register_enhanced("williams_vix_fix",
                      "portfolio.signals.williams_vix_fix",
                      "compute_williams_vix_fix_signal",
                      max_confidence=0.7)
    # Treasury risk rotation — bond yield curve shape as cross-asset regime signal
    # 2026-05-07: Gayed (2014), SSRN 2431022. IEF vs TLT relative performance.
    # Steepening = risk-on, flattening = risk-off. Inverted for safe havens.
    # Only signal using bond market data — zero correlation with existing clusters.
    register_enhanced("treasury_risk_rotation",
                      "portfolio.signals.treasury_risk_rotation",
                      "compute_treasury_risk_rotation_signal",
                      requires_context=True, max_confidence=0.7)


_register_defaults()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\portfolio\\signal_engine.py -Pattern 'get_enhanced_signals|load_signal_func|for .*enhanced|_ENHANCED_SIGNALS' -Context 2,3" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 654ms:

  Q:\finance-analyzer\portfolio\signal_engine.py:21:    _cached_or_enqueue,
  Q:\finance-analyzer\portfolio\signal_engine.py:22:)
> Q:\finance-analyzer\portfolio\signal_engine.py:23:from portfolio.signal_registry import [7mget_enhanced_signals[0m, [0m
[7m[0mload_signal_func[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:24:from portfolio.signal_utils import true_range[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:25:from [0m[7m[0mportfolio.tickers [0m[7m[0mimport [0m[7m[0mCRYPTO_SYMBOLS, [0m[7m[0mDISABLED_SIGNALS, [0m
[7m[0mGPU_SIGNALS, [0m[7m[0mMETALS_SYMBOLS, [0m[7m[0mSIGNAL_NAMES, [0m[7m[0mSTOCK_SYMBOLS[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:26:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2988:    # --- Enhanced signal modules (composite indicators computed 
from raw OHLCV) ---
  Q:\finance-analyzer\portfolio\signal_engine.py:2989:    # Loaded from signal_registry — no hardcoded list needed 
here.
> Q:\finance-analyzer\portfolio\signal_engine.py:2990:    _enhanced_entries = [7mget_enhanced_signals[0m()[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2991:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2992: [0m[7m [0m[7m [0m[7m [0m[7m[0mif [0m[7m[0mdf [0m[7m[0mis [0m[7m[0mnot [0m[7m[0mNone [0m[7m[0mand [0m[7m[0misinstance(df, [0m[7m[0mpd.DataFrame) [0m[7m[0mand [0m
[7m[0mlen(df) [0m[7m[0m>= [0m[7m[0m26:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2993: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mFetch [0m[7m[0mmacro [0m[7m[0mcontext [0m[7m[0monce [0m[7m[0mfor [0m[7m[0many [0m[7m[0msignal [0m[7m[0mthat [0m[7m[0mrequires [0m
[7m[0mit[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2994:        macro_data = None
> Q:\finance-analyzer\portfolio\signal_engine.py:2995:        has_macro_signals = any(e.get("requires_macro") [7mfor [0m[7me [0m[7min [0m
[7m_enhanced[0m_entries.values())[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2996:        if has_macro_signals:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2997:            try:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2998: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mfrom [0m[7m[0mportfolio.macro_context [0m[7m[0mimport [0m[7m[0mget_dxy, [0m
[7m[0mget_fed_calendar, [0m[7m[0mget_treasury[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3028:
  Q:\finance-analyzer\portfolio\signal_engine.py:3029:        _signal_failures = []
> Q:\finance-analyzer\portfolio\signal_engine.py:3030:        [7mfor sig_name, entry in _enhanced[0m_entries.items():[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3031: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mBUG-178 [0m[7m[0mfix [0m[7m[0m(2026-04-10): [0m[7m[0mrespect [0m[7m[0mDISABLED_SIGNALS [0m
[7m[0min [0m[7m[0mthe [0m[7m[0mdispatch[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3032: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mloop. [0m[7m[0mPreviously [0m[7m[0mthis [0m[7m[0mloop [0m[7m[0miterated [0m[7m[0m*every* [0m
[7m[0mregistered [0m[7m[0menhanced[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3033: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0msignal [0m[7m[0mregardless [0m[7m[0mof [0m[7m[0mdisabled [0m[7m[0mstatus, [0m[7m[0mwhich [0m[7m[0mmeant [0m
[7m[0mthe [0m[7m[0mthree[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3047:                    try:
  Q:\finance-analyzer\portfolio\signal_engine.py:3048:                        _sig_t0 = time.monotonic()
> Q:\finance-analyzer\portfolio\signal_engine.py:3049:                        compute_fn = [7mload_signal_func[0m(entry)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3050:                        if compute_fn is not None:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3051:                            if ticker:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3052: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m_set_last_signal(ticker, [0m
[7m[0mf"shadow:{sig_name}")[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3079:            try:
  Q:\finance-analyzer\portfolio\signal_engine.py:3080:                _sig_t0 = time.monotonic()
> Q:\finance-analyzer\portfolio\signal_engine.py:3081:                compute_fn = [7mload_signal_func[0m(entry)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3082:                if compute_fn is None:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3083:                    votes[sig_name] = "HOLD"[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3084:                    continue[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3119:                )
  Q:\finance-analyzer\portfolio\signal_engine.py:3120:
> Q:\finance-analyzer\portfolio\signal_engine.py:3121:        # Persist signal health (single batch write [7mfor [0m[7mall [0m
[7menhanced [0m[7m[0msignals)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3122:        try:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3123: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mfrom [0m[7m[0mportfolio.health [0m[7m[0mimport [0m
[7m[0mupdate_signal_health_batch[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3124:            health_results = {[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3125:                sig_name: (sig_name not in _signal_failures)
> Q:\finance-analyzer\portfolio\signal_engine.py:3126:                [7mfor sig_name in _enhanced[0m_entries[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3127:            }[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3128:            update_signal_health_batch(health_results)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3129:        except Exception:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3130:            logger.debug("Signal health tracking failed", 
exc_info=True)
  Q:\finance-analyzer\portfolio\signal_engine.py:3131:    else:
> Q:\finance-analyzer\portfolio\signal_engine.py:3132:        [7mfor sig_name in _enhanced[0m_entries:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3133:            votes[sig_name] = "HOLD"[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3134:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3135: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mMSTR [0m[7m[0mBTC [0m[7m[0mcross-asset [0m[7m[0mproxy [0m[7m[0m(2026-04-29): [0m[7m[0mMSTR [0m[7m[0mis [0m[7m[0ma [0m[7m[0mBTC [0m
[7m[0mtreasury [0m[7m[0mcompany[0m


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast
from pathlib import Path
mods=Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals')
stdlib={'__future__','math','json','time','logging','datetime','statistics','typing','dataclasses','pathlib','collections','functools','threading','itertools','re','urllib','hashlib','os'}
for p in sorted(mods.glob('*.py')):
    tree=ast.parse(p.read_text(encoding='utf-8'))
    imports=[]
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imports.append(a.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.append(n.module.split('.')[0])
    ext=sorted(set(i for i in imports if i not in {'__future__','portfolio','typing','collections','datetime','logging','math','statistics','time','json','pathlib','threading'} and not i.startswith('_')))
    print(p.name, ext)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 846ms:
__init__.py []
calendar_seasonal.py ['calendar', 'pandas']
candlestick.py ['numpy', 'pandas']
claude_fundamental.py ['pandas', 're', 'yfinance']
complexity_gap_regime.py ['numpy', 'pandas', 'yfinance']
copper_gold_ratio.py ['numpy', 'pandas', 'yfinance']
cot_positioning.py ['numpy', 'requests']
credit_spread.py ['requests']
cross_asset_tsmom.py ['numpy', 'pandas', 'yfinance']
crypto_cross_asset.py ['pandas']
crypto_evrp.py ['numpy', 'pandas']
crypto_macro.py ['pandas']
drift_regime_gate.py ['numpy', 'pandas']
dxy_cross_asset.py []
econ_calendar.py ['pandas']
fibonacci.py ['numpy', 'pandas']
forecast.py ['concurrent', 'pandas', 'platform', 'subprocess']
futures_basis.py ['numpy']
futures_flow.py ['pandas']
gold_real_yield_paradox.py ['numpy', 'pandas']
hash_ribbons.py ['numpy', 'pandas', 'requests']
heikin_ashi.py ['numpy', 'pandas']
hurst_regime.py ['numpy', 'pandas']
macro_regime.py ['contextlib', 'numpy', 'pandas']
mahalanobis_turbulence.py ['numpy', 'pandas', 'yfinance']
mean_reversion.py ['numpy', 'pandas']
metals_cross_asset.py []
momentum.py ['numpy', 'pandas']
momentum_factors.py ['numpy', 'pandas']
network_momentum.py ['numpy', 'pandas', 'yfinance']
news_event.py ['os', 'pandas']
orderbook_flow.py []
oscillators.py ['numpy', 'pandas']
ovx_metals_spillover.py ['numpy', 'yfinance']
realized_skewness.py ['numpy', 'pandas', 'scipy']
residual_pair_reversion.py ['numpy', 'pandas', 'yfinance']
shannon_entropy.py ['numpy', 'pandas']
smart_money.py ['numpy', 'pandas']
statistical_jump_regime.py ['numpy', 'pandas']
structure.py ['numpy', 'pandas']
treasury_risk_rotation.py ['numpy', 'pandas', 'yfinance']
trend.py ['numpy', 'pandas']
vix_term_structure.py ['numpy', 'yfinance']
vol_ratio_regime.py ['numpy', 'pandas']
volatility.py ['numpy', 'pandas']
volume_flow.py ['numpy', 'pandas']
williams_vix_fix.py ['numpy', 'pandas']
xtrend_equity_spillover.py ['numpy', 'pandas', 'yfinance']

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\pyproject.toml,Q:\\finance-analyzer\\requirements.txt -Pattern 'yfinance|scipy'" in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse Q:\\finance-analyzer | Select-String -Pattern '"'^import yfinance|from yfinance|''^import scipy|from scipy'"' | Select-Object Path,LineNumber,Line | Format-Table -AutoSize" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 598ms:
2026-05-08T15:32:37.679627Z ERROR codex_core::tools::router: error=Exit code: 124
Wall time: 120.1 seconds
Output:
command timed out after 120126 milliseconds


 exited 124 in 120126ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "pytest -q Q:\\finance-analyzer\\tests\\test_dxy_cross_asset_signal.py Q:\\finance-analyzer\\tests\\test_crypto_cross_asset_signal.py Q:\\finance-analyzer\\tests\\test_crypto_macro.py Q:\\finance-analyzer\\tests\\test_bug_fixes_session_20260321.py" in Q:\finance-analyzer
2026-05-08T15:32:48.658433Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mpytest: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mpytest[0m -q Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py Q: …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mThe term 'pytest' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

 exited 1 in 676ms:
[31;1mpytest: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mpytest[0m -q Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py Q: …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mThe term 'pytest' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python -m pytest -q Q:\\finance-analyzer\\tests\\test_dxy_cross_asset_signal.py Q:\\finance-analyzer\\tests\\test_crypto_cross_asset_signal.py Q:\\finance-analyzer\\tests\\test_crypto_macro.py Q:\\finance-analyzer\\tests\\test_bug_fixes_session_20260321.py" in Q:\finance-analyzer
2026-05-08T15:32:52.065008Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
C:\Users\Herc2\AppData\Local\Programs\Python\Python312\python.exe: No module named pytest

 exited 1 in 543ms:
C:\Users\Herc2\AppData\Local\Programs\Python\Python312\python.exe: No module named pytest

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -m pytest -q Q:\\finance-analyzer\\tests\\test_dxy_cross_asset_signal.py Q:\\finance-analyzer\\tests\\test_crypto_cross_asset_signal.py Q:\\finance-analyzer\\tests\\test_crypto_macro.py Q:\\finance-analyzer\\tests\\test_bug_fixes_session_20260321.py" in Q:\finance-analyzer
2026-05-08T15:32:57.761722Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 2.2 seconds
Total output lines: 2091
Output:
EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE      [100%]
=================================== ERRORS ====================================
_____ ERROR at setup of TestDxyCrossAssetSignal.test_dxy_down_buys_silver _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_____ ERROR at setup of TestDxyCrossAssetSignal.test_dxy_up_sells_silver ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestDxyCrossAssetSignal.test_dxy_down_buys_gold ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestDxyCrossAssetSignal.test_below_threshold_holds _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_non_metals_returns_hold_without_fetching _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_non_metals_btc_returns_hold __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___ ERROR at setup of TestDxyCrossAssetSignal.test_missing_dxy_returns_hold ___

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_missing_change_1h_returns_hold _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_confidence_scales_with_move __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestDxyCrossAssetSignal.test_confidence_caps_at_one ____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestDxyCrossAssetSignal.test_indicators_include_source ___

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_____ ERROR at setup of TestDxyCrossAssetSignal.test_sub_signals_present ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5…19806 tokens truncated…d once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCalendarExceptionLogging.test_sub_signal_failure_logged _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCalendarExceptionLogging.test_all_sub_signals_fail_returns_hold _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestFxAlertRouting.test_fallback_alert_uses_error_category _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestFxAlertRouting.test_stale_rate_alert_uses_error_category _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___ ERROR at setup of TestForecastDedupEviction.test_stale_entries_evicted ____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestForecastDedupEviction.test_eviction_constant_exists __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestForecastDedupEviction.test_forecast_lock_exists ____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
============================== warnings summary ===============================
.venv\Lib\site-packages\_pytest\cacheprovider.py:475
  Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\cacheprovider.py:475: PytestCacheWarning: could not create cache path Q:\finance-analyzer\.pytest_cache\v\cache\nodeids: [WinError 5] Access is denied: 'Q:\\finance-analyzer\\.pytest_cache\\v\\cache'
    config.cache.set("cache/nodeids", sorted(self.cached_nodeids))

.venv\Lib\site-packages\_pytest\cacheprovider.py:429
  Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\cacheprovider.py:429: PytestCacheWarning: could not create cache path Q:\finance-analyzer\.pytest_cache\v\cache\lastfailed: [WinError 5] Access is denied: 'Q:\\finance-analyzer\\.pytest_cache\\v\\cache'
    config.cache.set("cache/lastfailed", self.lastfailed)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_dxy_down_buys_silver
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_dxy_up_sells_silver
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_dxy_down_buys_gold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_below_threshold_holds
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_non_metals_returns_hold_without_fetching
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_non_metals_btc_returns_hold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_missing_dxy_returns_hold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_missing_change_1h_returns_hold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_confidence_scales_with_move
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_confidence_caps_at_one
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_indicators_include_source
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_sub_signals_present
ERROR tests/test_dxy_cross_asset_signal.py::TestMacroContextIntradayDxy::test_primary_source_used_when_available
ERROR tests/test_dxy_cross_asset_signal.py::TestMacroContextIntradayDxy::test_falls_back_to_eurusd_synth_on_empty
ERROR tests/test_dxy_cross_asset_signal.py::TestMacroContextIntradayDxy::test_all_sources_fail_returns_none
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[XAU-USD]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[XAG-USD]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[MSTR]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[AAPL]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[]
ERROR tests/test_crypto_cross_asset_signal.py::TestNoData::test_btc_no_cross_asset_returns_hold
ERROR tests/test_crypto_cross_asset_signal.py::TestFearGreed::test_extreme_fear_buys
ERROR tests/test_crypto_cross_asset_signal.py::TestFearGreed::test_extreme_greed_sells
ERROR tests/test_crypto_cross_asset_signal.py::TestFearGreed::test_neutral_holds
ERROR tests/test_crypto_cross_asset_signal.py::TestEthBtcRatio::test_eth_outperformance_bullish_for_eth
ERROR tests/test_crypto_cross_asset_signal.py::TestEthBtcRatio::test_eth_underperformance_sells_for_eth
ERROR tests/test_crypto_cross_asset_signal.py::TestDxy::test_dxy_up_bearish_for_crypto
ERROR tests/test_crypto_cross_asset_signal.py::TestDxy::test_dxy_down_bullish
ERROR tests/test_crypto_cross_asset_signal.py::TestSpy::test_spy_strong_up_bullish
ERROR tests/test_crypto_cross_asset_signal.py::TestGoldBtcRatio::test_gold_strong_btc_weak_bearish
ERROR tests/test_crypto_cross_asset_signal.py::TestConfidenceBounds::test_confidence_capped_at_max
ERROR tests/test_crypto_cross_asset_signal.py::TestConfidenceBounds::test_confidence_zero_on_hold
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_max_pain_simple - ...
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_put_call_ratio_calculation
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_empty_response - P...
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_no_open_interest
ERROR tests/test_crypto_macro.py::TestGoldBtcRatio::test_ratio_computed - Per...
ERROR tests/test_crypto_macro.py::TestGoldBtcRatio::test_missing_prices - Per...
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_non_crypto_returns_hold
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_btc_returns_signal
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_below_max_pain
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_above_max_pain
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_near_max_pain
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_far_expiry
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_pcr_high_contrarian_buy
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_pcr_low_contrarian_sell
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_gold_rotation_btc_outperforming
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_gold_rotation_gold_outperforming
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_netflow_accumulation
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_netflow_distribution
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_quarterly_expiry_day
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_dirty_flag_cleared_after_successful_write
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_dirty_flag_remains_on_write_failure
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_retry_after_failure
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_not_dirty_skips_write
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastThreadSafety::test_concurrent_trip_and_check
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastThreadSafety::test_log_health_reset_under_lock
ERROR tests/test_bug_fixes_session_20260321.py::TestPortfolioPnlZeroDivGuard::test_zero_initial_value_uses_fallback
ERROR tests/test_bug_fixes_session_20260321.py::TestPortfolioPnlZeroDivGuard::test_missing_initial_value_uses_fallback
ERROR tests/test_bug_fixes_session_20260321.py::TestPortfolioPnlZeroDivGuard::test_valid_initial_value_used_directly
ERROR tests/test_bug_fixes_session_20260321.py::TestCalendarExceptionLogging::test_sub_signal_failure_logged
ERROR tests/test_bug_fixes_session_20260321.py::TestCalendarExceptionLogging::test_all_sub_signals_fail_returns_hold
ERROR tests/test_bug_fixes_session_20260321.py::TestFxAlertRouting::test_fallback_alert_uses_error_category
ERROR tests/test_bug_fixes_session_20260321.py::TestFxAlertRouting::test_stale_rate_alert_uses_error_category
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastDedupEviction::test_stale_entries_evicted
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastDedupEviction::test_eviction_constant_exists
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastDedupEviction::test_forecast_lock_exists
2 warnings, 67 errors in 0.77s

 exited 1 in 2172ms:
EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE      [100%]
=================================== ERRORS ====================================
_____ ERROR at setup of TestDxyCrossAssetSignal.test_dxy_down_buys_silver _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_____ ERROR at setup of TestDxyCrossAssetSignal.test_dxy_up_sells_silver ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestDxyCrossAssetSignal.test_dxy_down_buys_gold ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestDxyCrossAssetSignal.test_below_threshold_holds _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_non_metals_returns_hold_without_fetching _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_non_metals_btc_returns_hold __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___ ERROR at setup of TestDxyCrossAssetSignal.test_missing_dxy_returns_hold ___

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_missing_change_1h_returns_hold _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestDxyCrossAssetSignal.test_confidence_scales_with_move __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestDxyCrossAssetSignal.test_confidence_caps_at_one ____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestDxyCrossAssetSignal.test_indicators_include_source ___

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_____ ERROR at setup of TestDxyCrossAssetSignal.test_sub_signals_present ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestMacroContextIntradayDxy.test_primary_source_used_when_available _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestMacroContextIntradayDxy.test_falls_back_to_eurusd_synth_on_empty _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestMacroContextIntradayDxy.test_all_sources_fail_returns_none _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestNonCryptoTickers.test_returns_hold[XAU-USD] ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestNonCryptoTickers.test_returns_hold[XAG-USD] ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_______ ERROR at setup of TestNonCryptoTickers.test_returns_hold[MSTR] ________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_______ ERROR at setup of TestNonCryptoTickers.test_returns_hold[AAPL] ________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_________ ERROR at setup of TestNonCryptoTickers.test_returns_hold[] __________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestNoData.test_btc_no_cross_asset_returns_hold ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___________ ERROR at setup of TestFearGreed.test_extreme_fear_buys ____________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__________ ERROR at setup of TestFearGreed.test_extreme_greed_sells ___________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_____________ ERROR at setup of TestFearGreed.test_neutral_holds ______________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestEthBtcRatio.test_eth_outperformance_bullish_for_eth __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestEthBtcRatio.test_eth_underperformance_sells_for_eth __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__________ ERROR at setup of TestDxy.test_dxy_up_bearish_for_crypto ___________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_______________ ERROR at setup of TestDxy.test_dxy_down_bullish _______________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____________ ERROR at setup of TestSpy.test_spy_strong_up_bullish _____________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestGoldBtcRatio.test_gold_strong_btc_weak_bearish _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestConfidenceBounds.test_confidence_capped_at_max _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_____ ERROR at setup of TestConfidenceBounds.test_confidence_zero_on_hold _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__________ ERROR at setup of TestDeribitOptions.test_max_pain_simple __________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestDeribitOptions.test_put_call_ratio_calculation _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__________ ERROR at setup of TestDeribitOptions.test_empty_response ___________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_________ ERROR at setup of TestDeribitOptions.test_no_open_interest __________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___________ ERROR at setup of TestGoldBtcRatio.test_ratio_computed ____________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___________ ERROR at setup of TestGoldBtcRatio.test_missing_prices ____________

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestCryptoMacroSignal.test_non_crypto_returns_hold _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_______ ERROR at setup of TestCryptoMacroSignal.test_btc_returns_signal _______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCryptoMacroSignal.test_options_gravity_below_max_pain _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCryptoMacroSignal.test_options_gravity_above_max_pain _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCryptoMacroSignal.test_options_gravity_near_max_pain __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___ ERROR at setup of TestCryptoMacroSignal.test_options_gravity_far_expiry ___

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestCryptoMacroSignal.test_pcr_high_contrarian_buy _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestCryptoMacroSignal.test_pcr_low_contrarian_sell _____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCryptoMacroSignal.test_gold_rotation_btc_outperforming _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCryptoMacroSignal.test_gold_rotation_gold_outperforming _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestCryptoMacroSignal.test_netflow_accumulation ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestCryptoMacroSignal.test_netflow_distribution ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
______ ERROR at setup of TestCryptoMacroSignal.test_quarterly_expiry_day ______

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestSentimentFlushCrashSafety.test_dirty_flag_cleared_after_successful_write _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestSentimentFlushCrashSafety.test_dirty_flag_remains_on_write_failure _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestSentimentFlushCrashSafety.test_retry_after_failure ___

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestSentimentFlushCrashSafety.test_not_dirty_skips_write __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestForecastThreadSafety.test_concurrent_trip_and_check __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestForecastThreadSafety.test_log_health_reset_under_lock _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestPortfolioPnlZeroDivGuard.test_zero_initial_value_uses_fallback _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestPortfolioPnlZeroDivGuard.test_missing_initial_value_uses_fallback _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestPortfolioPnlZeroDivGuard.test_valid_initial_value_used_directly _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCalendarExceptionLogging.test_sub_signal_failure_logged _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestCalendarExceptionLogging.test_all_sub_signals_fail_returns_hold _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestFxAlertRouting.test_fallback_alert_uses_error_category _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
_ ERROR at setup of TestFxAlertRouting.test_stale_rate_alert_uses_error_category _

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
___ ERROR at setup of TestForecastDedupEviction.test_stale_entries_evicted ____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
__ ERROR at setup of TestForecastDedupEviction.test_eviction_constant_exists __

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
____ ERROR at setup of TestForecastDedupEviction.test_forecast_lock_exists ____

tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0x000001642DFEB920>, _basetemp=None, _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="session", autouse=True)
    def _redirect_signal_utility_disk_cache(tmp_path_factory):
        """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
        pytest suite NEVER touches the production data/signal_utility_cache.json.
    
        Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
        now deletes the disk file, and the per-test in-memory clearing fixture below
        was indiscriminately wiping the production L2 cache every test run. With
        this session-level monkeypatch, the production file path is replaced once
        at session start with a tmpdir that gets cleaned up automatically when
        pytest exits.
    
        Session-scoped because file path doesn't need per-test isolation (the
        in-memory clear below handles that), and a per-test redirect would burn
        a tmpdir per test for thousands of tests.
        """
        try:
            import portfolio.accuracy_stats as acc_mod
        except ImportError:
            yield
            return
>       tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\pytest-of-Herc2'

tests\conftest.py:54: PermissionError
============================== warnings summary ===============================
.venv\Lib\site-packages\_pytest\cacheprovider.py:475
  Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\cacheprovider.py:475: PytestCacheWarning: could not create cache path Q:\finance-analyzer\.pytest_cache\v\cache\nodeids: [WinError 5] Access is denied: 'Q:\\finance-analyzer\\.pytest_cache\\v\\cache'
    config.cache.set("cache/nodeids", sorted(self.cached_nodeids))

.venv\Lib\site-packages\_pytest\cacheprovider.py:429
  Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\cacheprovider.py:429: PytestCacheWarning: could not create cache path Q:\finance-analyzer\.pytest_cache\v\cache\lastfailed: [WinError 5] Access is denied: 'Q:\\finance-analyzer\\.pytest_cache\\v\\cache'
    config.cache.set("cache/lastfailed", self.lastfailed)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_dxy_down_buys_silver
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_dxy_up_sells_silver
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_dxy_down_buys_gold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_below_threshold_holds
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_non_metals_returns_hold_without_fetching
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_non_metals_btc_returns_hold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_missing_dxy_returns_hold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_missing_change_1h_returns_hold
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_confidence_scales_with_move
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_confidence_caps_at_one
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_indicators_include_source
ERROR tests/test_dxy_cross_asset_signal.py::TestDxyCrossAssetSignal::test_sub_signals_present
ERROR tests/test_dxy_cross_asset_signal.py::TestMacroContextIntradayDxy::test_primary_source_used_when_available
ERROR tests/test_dxy_cross_asset_signal.py::TestMacroContextIntradayDxy::test_falls_back_to_eurusd_synth_on_empty
ERROR tests/test_dxy_cross_asset_signal.py::TestMacroContextIntradayDxy::test_all_sources_fail_returns_none
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[XAU-USD]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[XAG-USD]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[MSTR]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[AAPL]
ERROR tests/test_crypto_cross_asset_signal.py::TestNonCryptoTickers::test_returns_hold[]
ERROR tests/test_crypto_cross_asset_signal.py::TestNoData::test_btc_no_cross_asset_returns_hold
ERROR tests/test_crypto_cross_asset_signal.py::TestFearGreed::test_extreme_fear_buys
ERROR tests/test_crypto_cross_asset_signal.py::TestFearGreed::test_extreme_greed_sells
ERROR tests/test_crypto_cross_asset_signal.py::TestFearGreed::test_neutral_holds
ERROR tests/test_crypto_cross_asset_signal.py::TestEthBtcRatio::test_eth_outperformance_bullish_for_eth
ERROR tests/test_crypto_cross_asset_signal.py::TestEthBtcRatio::test_eth_underperformance_sells_for_eth
ERROR tests/test_crypto_cross_asset_signal.py::TestDxy::test_dxy_up_bearish_for_crypto
ERROR tests/test_crypto_cross_asset_signal.py::TestDxy::test_dxy_down_bullish
ERROR tests/test_crypto_cross_asset_signal.py::TestSpy::test_spy_strong_up_bullish
ERROR tests/test_crypto_cross_asset_signal.py::TestGoldBtcRatio::test_gold_strong_btc_weak_bearish
ERROR tests/test_crypto_cross_asset_signal.py::TestConfidenceBounds::test_confidence_capped_at_max
ERROR tests/test_crypto_cross_asset_signal.py::TestConfidenceBounds::test_confidence_zero_on_hold
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_max_pain_simple - ...
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_put_call_ratio_calculation
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_empty_response - P...
ERROR tests/test_crypto_macro.py::TestDeribitOptions::test_no_open_interest
ERROR tests/test_crypto_macro.py::TestGoldBtcRatio::test_ratio_computed - Per...
ERROR tests/test_crypto_macro.py::TestGoldBtcRatio::test_missing_prices - Per...
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_non_crypto_returns_hold
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_btc_returns_signal
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_below_max_pain
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_above_max_pain
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_near_max_pain
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_options_gravity_far_expiry
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_pcr_high_contrarian_buy
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_pcr_low_contrarian_sell
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_gold_rotation_btc_outperforming
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_gold_rotation_gold_outperforming
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_netflow_accumulation
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_netflow_distribution
ERROR tests/test_crypto_macro.py::TestCryptoMacroSignal::test_quarterly_expiry_day
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_dirty_flag_cleared_after_successful_write
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_dirty_flag_remains_on_write_failure
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_retry_after_failure
ERROR tests/test_bug_fixes_session_20260321.py::TestSentimentFlushCrashSafety::test_not_dirty_skips_write
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastThreadSafety::test_concurrent_trip_and_check
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastThreadSafety::test_log_health_reset_under_lock
ERROR tests/test_bug_fixes_session_20260321.py::TestPortfolioPnlZeroDivGuard::test_zero_initial_value_uses_fallback
ERROR tests/test_bug_fixes_session_20260321.py::TestPortfolioPnlZeroDivGuard::test_missing_initial_value_uses_fallback
ERROR tests/test_bug_fixes_session_20260321.py::TestPortfolioPnlZeroDivGuard::test_valid_initial_value_used_directly
ERROR tests/test_bug_fixes_session_20260321.py::TestCalendarExceptionLogging::test_sub_signal_failure_logged
ERROR tests/test_bug_fixes_session_20260321.py::TestCalendarExceptionLogging::test_all_sub_signals_fail_returns_hold
ERROR tests/test_bug_fixes_session_20260321.py::TestFxAlertRouting::test_fallback_alert_uses_error_category
ERROR tests/test_bug_fixes_session_20260321.py::TestFxAlertRouting::test_stale_rate_alert_uses_error_category
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastDedupEviction::test_stale_entries_evicted
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastDedupEviction::test_eviction_constant_exists
ERROR tests/test_bug_fixes_session_20260321.py::TestForecastDedupEviction::test_forecast_lock_exists
2 warnings, 67 errors in 0.77s

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$env:PYTHONPATH='"'Q:\\finance-analyzer'; Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -m pytest -q Q:\\finance-analyzer\\tests\\test_dxy_cross_asset_signal.py --basetemp=Q:\\finance-analyzer\\.worktrees\\adv-signals-modules\\.pytest_tmp -p no:cacheprovider" in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-08T15:33:06.530248Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 1.6 seconds
Output:
EEEEEEEEEEEEEEE                                                          [100%]Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pytest\__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
                     ^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 223, in console_main
    code = main()
           ^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 353, in wrap_session
    config.hook.pytest_sessionfinish(
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\logging.py", line 873, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\terminal.py", line 960, in pytest_sessionfinish
    result = yield
             ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\warnings.py", line 118, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\tmpdir.py", line 304, in pytest_sessionfinish
    cleanup_dead_symlinks(basetemp)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\pathlib.py", line 357, in cleanup_dead_symlinks
    for left_dir in root.iterdir():
                    ^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1056, in iterdir
    for name in os.listdir(self):
                ^^^^^^^^^^^^^^^^
PermissionError: [WinError 5] Access is denied: 'Q:\\finance-analyzer\\.worktrees\\adv-signals-modules\\.pytest_tmp'

 exited 1 in 1565ms:
EEEEEEEEEEEEEEE                                                          [100%]Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pytest\__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
                     ^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 223, in console_main
    code = main()
           ^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 353, in wrap_session
    config.hook.pytest_sessionfinish(
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\logging.py", line 873, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\terminal.py", line 960, in pytest_sessionfinish
    result = yield
             ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\warnings.py", line 118, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\tmpdir.py", line 304, in pytest_sessionfinish
    cleanup_dead_symlinks(basetemp)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\pathlib.py", line 357, in cleanup_dead_symlinks
    for left_dir in root.iterdir():
                    ^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1056, in iterdir
    for name in os.listdir(self):
                ^^^^^^^^^^^^^^^^
PermissionError: [WinError 5] Access is denied: 'Q:\\finance-analyzer\\.worktrees\\adv-signals-modules\\.pytest_tmp'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Force Q:\\finance-analyzer\\.worktrees\\adv-signals-modules" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 568ms:

    Directory: Q:\finance-analyzer\.worktrees\adv-signals-modules

[32;1mMode   [0m[32;1m              LastWriteTime[0m [32;1;3m        Length[0m[32;1m Name[0m
[32;1m----   [0m [32;1m             -------------[0m [32;1m        ------[0m [32;1m----[0m
d----          2026-05-08    17:33                [44;1m.pytest_tmp[0m
d----          2026-05-08    17:25                [44;1mportfolio[0m
-a-h-          2026-05-08    17:25             63 .git
-a---          2026-05-08    17:25             46 BASELINE.md


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Force Q:\\finance-analyzer\\.worktrees\\adv-signals-modules\\.pytest_tmp -Recurse" in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-08T15:33:13.532491Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mGet-ChildItem: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mGet-ChildItem -Force Q:\finance-analyzer\.worktrees\adv-signals-modul[0m …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mAccess to the path 'Q:\finance-analyzer\.worktrees\adv-signals-modules\.pytest_tmp' is denied.[0m

 exited 1 in 821ms:
[31;1mGet-ChildItem: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mGet-ChildItem -Force Q:\finance-analyzer\.worktrees\adv-signals-modul[0m …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mAccess to the path 'Q:\finance-analyzer\.worktrees\adv-signals-modules\.pytest_tmp' is denied.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$env:PYTHONPATH='"'Q:\\finance-analyzer'; Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -m pytest -q Q:\\finance-analyzer\\tests\\test_dxy_cross_asset_signal.py --basetemp=C:\\Users\\Herc2\\.codex\\memories\\pytest_tmp -p no:cacheprovider" in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-08T15:33:19.830745Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 1.5 seconds
Output:
EEEEEEEEEEEEEEE                                                          [100%]Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pytest\__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
                     ^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 223, in console_main
    code = main()
           ^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 353, in wrap_session
    config.hook.pytest_sessionfinish(
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\logging.py", line 873, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\terminal.py", line 960, in pytest_sessionfinish
    result = yield
             ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\warnings.py", line 118, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\tmpdir.py", line 304, in pytest_sessionfinish
    cleanup_dead_symlinks(basetemp)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\pathlib.py", line 357, in cleanup_dead_symlinks
    for left_dir in root.iterdir():
                    ^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1056, in iterdir
    for name in os.listdir(self):
                ^^^^^^^^^^^^^^^^
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\.codex\\memories\\pytest_tmp'

 exited 1 in 1528ms:
EEEEEEEEEEEEEEE                                                          [100%]Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pytest\__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
                     ^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 223, in console_main
    code = main()
           ^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\config\__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\main.py", line 353, in wrap_session
    config.hook.pytest_sessionfinish(
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 167, in _multicall
    raise exception
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\logging.py", line 873, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\terminal.py", line 960, in pytest_sessionfinish
    result = yield
             ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\warnings.py", line 118, in pytest_sessionfinish
    return (yield)
            ^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\pluggy\_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\tmpdir.py", line 304, in pytest_sessionfinish
    cleanup_dead_symlinks(basetemp)
  File "Q:\finance-analyzer\.venv\Lib\site-packages\_pytest\pathlib.py", line 357, in cleanup_dead_symlinks
    for left_dir in root.iterdir():
                    ^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1056, in iterdir
    for name in os.listdir(self):
                ^^^^^^^^^^^^^^^^
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\.codex\\memories\\pytest_tmp'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
import re
for p in sorted(Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals').glob('*.py')):
    text=p.read_text(encoding='utf-8')
    for m in re.finditer(r'requests\\.(get|post)\\(', text):
        start=text.rfind('\\n',0,m.start())+1
        end=text.find('\\n',m.start())
        print(p.name, text[start:end].strip())
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 586ms:
cot_positioning.py resp = requests.get(url, timeout=_CFTC_TIMEOUT)
credit_spread.py return requests.get(url, **kwargs)
hash_ribbons.py r = requests.get(url, timeout=kw.get("timeout", 20))

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/signals/credit_spread.py' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 628ms:
"""Credit spread risk appetite signal — cross-asset macro indicator.

Signal #35.  Uses ICE BofA High Yield Option-Adjusted Spread (HY OAS)
from FRED API as a measure of institutional credit risk appetite.
Combines 4 sub-indicators via majority vote:
    1. OAS Level Z-Score:    where current spread sits vs 252d history
    2. OAS 5d Momentum:      rate of spread widening/tightening
    3. OAS Momentum Accel:   acceleration of momentum (early warning)
    4. Crisis Level Gate:    hard threshold at 500bp (every bear market)

Direction depends on asset class:
- Gold/Silver (safe haven):  risk-off → BUY, risk-on → SELL
- Crypto/Stocks (risk-on):   risk-off → SELL, risk-on → BUY

Data: FRED API series BAMLH0A0HYM2 (daily, free with API key).
Cached for 4 hours since data updates daily.

Requires context dict with keys: ticker, config (for FRED API key).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from portfolio.file_utils import load_json
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.credit_spread")

# ---------------------------------------------------------------------------
# Asset-class classification for directional interpretation
# ---------------------------------------------------------------------------
_SAFE_HAVEN = {"XAU-USD", "XAG-USD"}
_RISK_ASSETS = {"BTC-USD", "ETH-USD", "MSTR"}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_ZSCORE_RISK_OFF = 1.5      # OAS z-score above this = risk-off
_ZSCORE_RISK_ON = -1.0      # OAS z-score below this = risk-on (complacent)
_MOM_5D_THRESHOLD = 0.20    # 20bp 5d change = meaningful momentum
_ACCEL_THRESHOLD = 0.15     # 15bp acceleration = early warning
_CRISIS_LEVEL = 5.0         # 500bp = crisis threshold
_HISTORY_LIMIT = 300        # FRED observations to fetch (covers ~252 trading days)
_FRED_SERIES = "BAMLH0A0HYM2"
_FRED_TIMEOUT = 15
_CACHE_TTL = 4 * 3600       # 4 hours

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_oas_cache: dict = {}


def _fetch_hy_oas(fred_api_key: str) -> list[float] | None:
    """Fetch HY OAS history from FRED.  Returns list of floats (newest first).

    Cached for 4 hours since data is daily.
    """
    now = time.time()
    if (
        _oas_cache.get("key") == fred_api_key
        and _oas_cache.get("data")
        and now - _oas_cache.get("time", 0) < _CACHE_TTL
    ):
        return _oas_cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key — cannot fetch HY OAS")
        return _oas_cache.get("data")

    try:
        from portfolio.http_retry import fetch_with_retry
    except ImportError:
        import requests

        class _Shim:
            @staticmethod
            def __call__(url, **kwargs):
                return requests.get(url, **kwargs)
        fetch_with_retry = _Shim()

    try:
        resp = fetch_with_retry(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": _FRED_SERIES,
                "api_key": fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": _HISTORY_LIMIT,
            },
            timeout=_FRED_TIMEOUT,
        )
        if hasattr(resp, "json"):
            data = resp.json()
        else:
            import json
            data = json.loads(resp)

        observations = data.get("observations", [])
        values = []
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue

        if values:
            _oas_cache["key"] = fred_api_key
            _oas_cache["data"] = values
            _oas_cache["time"] = now
            logger.debug("HY OAS fetched: %d observations, current=%.2f", len(values), values[0])
            return values

    except Exception:
        logger.warning("FRED HY OAS fetch failed", exc_info=True)

    return _oas_cache.get("data")


def _get_fred_key(context: dict | None) -> str:
    """Extract FRED API key from context -> config."""
    if not context:
        return ""
    cfg = context.get("config")
    if not cfg:
        return ""
    if isinstance(cfg, dict):
        return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
    return getattr(cfg, "fred_api_key", "") or getattr(
        getattr(cfg, "golddigger", None), "fred_api_key", ""
    ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""


def _is_safe_haven(ticker: str) -> bool:
    """Return True if ticker is a safe-haven asset (gold/silver)."""
    return ticker in _SAFE_HAVEN


# ---------------------------------------------------------------------------
# Sub-indicator 1: OAS Level Z-Score
# ---------------------------------------------------------------------------

def _oas_zscore_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """Z-score of current OAS vs 252d history."""
    if len(values) < 50:
        return "HOLD", {"oas_zscore": 0.0}

    lookback = min(252, len(values))
    history = values[:lookback]
    current = values[0]
    mean = sum(history) / len(history)
    variance = sum((v - mean) ** 2 for v in history) / len(history)
    std = variance ** 0.5

    if std < 0.01:
        return "HOLD", {"oas_zscore": 0.0}

    zscore = (current - mean) / std

    action = "HOLD"
    if zscore > _ZSCORE_RISK_OFF:
        # Risk-off: spreads unusually wide
        action = "BUY" if safe_haven else "SELL"
    elif zscore < _ZSCORE_RISK_ON:
        # Risk-on: spreads unusually tight (complacent)
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_zscore": round(zscore, 3), "oas_current": round(current, 2),
                     "oas_mean_252d": round(mean, 2)}


# ---------------------------------------------------------------------------
# Sub-indicator 2: OAS 5d Momentum
# ---------------------------------------------------------------------------

def _oas_momentum_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """5-day change in OAS (widening vs tightening)."""
    if len(values) < 6:
        return "HOLD", {"oas_mom_5d": 0.0}

    mom = values[0] - values[5]

    action = "HOLD"
    if mom > _MOM_5D_THRESHOLD:
        # Spreads widening rapidly → risk-off
        action = "BUY" if safe_haven else "SELL"
    elif mom < -_MOM_5D_THRESHOLD:
        # Spreads tightening → risk-on
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_mom_5d": round(mom, 3)}


# ---------------------------------------------------------------------------
# Sub-indicator 3: Momentum Acceleration
# ---------------------------------------------------------------------------

def _oas_acceleration_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """Acceleration of OAS momentum (early warning of regime shift)."""
    if len(values) < 11:
        return "HOLD", {"oas_accel": 0.0}

    mom_current = values[0] - values[5]
    mom_prev = values[5] - values[10]
    accel = mom_current - mom_prev

    action = "HOLD"
    if accel > _ACCEL_THRESHOLD:
        # Accelerating widening → risk-off intensifying
        action = "BUY" if safe_haven else "SELL"
    elif accel < -_ACCEL_THRESHOLD:
        # Accelerating tightening → risk-on intensifying
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_accel": round(accel, 3)}


# ---------------------------------------------------------------------------
# Sub-indicator 4: Crisis Level Gate
# ---------------------------------------------------------------------------

def _crisis_level_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """Hard threshold: OAS above 500bp = crisis mode."""
    if not values:
        return "HOLD", {"oas_crisis": False}

    current = values[0]
    crisis = current >= _CRISIS_LEVEL

    action = "HOLD"
    if crisis:
        # Full crisis → strong safe-haven bid
        action = "BUY" if safe_haven else "SELL"
    elif current <= 2.5:
        # Extreme complacency → contrarian warning
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_crisis": crisis, "oas_level": round(current, 2)}


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_credit_spread_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute credit spread risk appetite signal.

    Args:
        df: OHLCV DataFrame (unused — credit spread data fetched from FRED).
        context: dict with keys {ticker, config, asset_class, regime}.

    Returns:
        dict with keys: action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))

    if not ticker:
        return empty

    # All tickers can use this signal
    if ticker not in _SAFE_HAVEN and ticker not in _RISK_ASSETS:
        return empty

    safe_haven = _is_safe_haven(ticker)

    # Get FRED API key from config
    fred_key = _get_fred_key(context)

    # Also try loading from config.json directly as fallback
    if not fred_key:
        try:
            cfg = load_json("config.json", default={}) or {}
            fred_key = cfg.get("golddigger", {}).get("fred_api_key", "") or ""
        except Exception:
            logger.debug("config.json fallback read failed", exc_info=True)

    if not fred_key:
        logger.debug("No FRED API key available for credit spread signal")
        return empty

    # Fetch HY OAS data
    values = _fetch_hy_oas(fred_key)
    if not values or len(values) < 20:
        return empty

    # Compute sub-indicators
    zscore_action, zscore_ind = _oas_zscore_signal(values, safe_haven)
    mom_action, mom_ind = _oas_momentum_signal(values, safe_haven)
    accel_action, accel_ind = _oas_acceleration_signal(values, safe_haven)
    crisis_action, crisis_ind = _crisis_level_signal(values, safe_haven)

    votes = [zscore_action, mom_action, accel_action, crisis_action]
    action, confidence = majority_vote(votes, count_hold=False)

    # Merge all indicators
    indicators = {}
    indicators.update(zscore_ind)
    indicators.update(mom_ind)
    indicators.update(accel_ind)
    indicators.update(crisis_ind)
    indicators["safe_haven_mode"] = safe_haven

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "oas_zscore": zscore_action,
            "oas_momentum": mom_action,
            "oas_acceleration": accel_action,
            "crisis_level": crisis_action,
        },
        "indicators": indicators,
    }

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path, pattern in [
    (r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals/claude_fundamental.py', 'import yfinance'),
    (r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals/realized_skewness.py', 'from scipy'),
]:
    p=Path(path)
    for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if pattern in line:
            print(p.name, i, line)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 586ms:
claude_fundamental.py 172             import yfinance as yf
realized_skewness.py 25 from scipy import stats

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/signals/claude_fundamental.py -TotalCount 220 | Select-Object -Last 70' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 676ms:

def _get_earnings_calendar():
    """Fetch next earnings dates from yfinance for all stock tickers.

    Cached for 12h to avoid excessive Yahoo Finance requests.
    Returns dict of {ticker: {date, eps_estimate, days_until}}.
    """
    now = time.time()
    if now - _earnings_cache["ts"] < _EARNINGS_CACHE_TTL and _earnings_cache["data"]:
        return _earnings_cache["data"]

    from datetime import datetime

    try:
        from portfolio.tickers import STOCK_SYMBOLS
    except ImportError:
        return {}

    result = {}
    for ticker in STOCK_SYMBOLS:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None or (hasattr(cal, "empty") and cal.empty):
                continue
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
                eps_avg = cal.get("Earnings Average")
            elif hasattr(cal, "loc"):
                dates = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else []
                eps_avg = cal.loc["Earnings Average"] if "Earnings Average" in cal.index else None
            else:
                continue

            if dates:
                next_date = dates[0] if isinstance(dates, list) else dates
                if hasattr(next_date, "date"):
                    next_date = next_date.date()
                today = datetime.now(UTC).date()
                days_until = (next_date - today).days
                result[ticker] = {
                    "date": str(next_date),
                    "eps_estimate": float(eps_avg) if eps_avg is not None else None,
                    "days_until": days_until,
                }
        except Exception:
            logger.debug("Earnings fetch failed for ticker", exc_info=True)
            continue

    _earnings_cache["data"] = result
    _earnings_cache["ts"] = now
    return result


def _build_fundamentals_block(ticker, fundamentals, tier="haiku", earnings=None):
    """Build a fundamentals data string for a ticker based on tier detail level.

    Args:
        ticker: Stock ticker symbol
        fundamentals: Dict of all fundamentals from alpha_vantage cache
        tier: "haiku" (one-liner), "sonnet"/"opus" (detailed block)

    Returns:
        Formatted string with fundamentals data, or empty string if none available.
    """
    fund = fundamentals.get(ticker) if fundamentals else None
    if not fund:
        return ""


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast
from pathlib import Path
mods=Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals')
for p in sorted(mods.glob('*.py')):
    tree=ast.parse(p.read_text(encoding='utf-8'))
    top=[]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] in {'yfinance','scipy'}:
                    top.append((a.name,node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in {'yfinance','scipy'}:
                top.append((node.module,node.lineno))
    if top:
        print(p.name, top)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 762ms:
realized_skewness.py [('scipy', 25)]

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for p in sorted(Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals').glob('*.py')):
    text=p.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(text,1):
        if 'import yfinance as yf' in line:
            start=max(1,i-4); end=min(len(text),i+6)
            print('###',p.name,i)
            for j in range(start,end+1):
                print(f'{j}: {text[j-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 558ms:
### claude_fundamental.py 172
168: 
169:     result = {}
170:     for ticker in STOCK_SYMBOLS:
171:         try:
172:             import yfinance as yf
173:             t = yf.Ticker(ticker)
174:             cal = t.calendar
175:             if cal is None or (hasattr(cal, "empty") and cal.empty):
176:                 continue
177:             if isinstance(cal, dict):
178:                 dates = cal.get("Earnings Date", [])
### complexity_gap_regime.py 68
64:     Cached for 1 hour via shared_state.
65:     """
66:     def _do_fetch():
67:         try:
68:             import yfinance as yf
69: 
70:             data = yf.download(
71:                 _YF_TICKERS, period="4mo", progress=False, threads=True
72:             )
73:             if data is None or data.empty:
74:                 return None
### copper_gold_ratio.py 82
78:         copper_df = _download("HG=F", period="1y", interval="1d")
79:         gold_df = _download("GC=F", period="1y", interval="1d")
80:     except Exception:
81:         try:
82:             import yfinance as yf
83:             copper_df = yf.download("HG=F", period="1y", interval="1d",
84:                                     progress=False, auto_adjust=True)
85:             gold_df = yf.download("GC=F", period="1y", interval="1d",
86:                                   progress=False, auto_adjust=True)
87:         except Exception as exc:
88:             logger.warning("copper_gold_ratio: data fetch failed: %s", exc)
### cross_asset_tsmom.py 62
58:     or None on failure.  Cached for 1 hour.
59:     """
60:     def _do_fetch():
61:         try:
62:             import yfinance as yf
63: 
64:             tickers = list(_YF_TICKERS)
65:             data = yf.download(tickers, period="13mo", progress=False, threads=True)
66:             if data is None or data.empty:
67:                 return None
68: 
### mahalanobis_turbulence.py 75
71:     Cached for 1 hour via shared_state.
72:     """
73:     def _do_fetch():
74:         try:
75:             import yfinance as yf
76: 
77:             data = yf.download(
78:                 _YF_TICKERS, period="14mo", progress=False, threads=True
79:             )
80:             if data is None or data.empty:
81:                 return None
### network_momentum.py 78
74:     Cached for 1 hour.
75:     """
76:     def _do_fetch():
77:         try:
78:             import yfinance as yf
79: 
80:             tickers = list(_YF_DOWNLOAD_TICKERS)
81:             data = yf.download(
82:                 tickers, period="4mo", progress=False, threads=True
83:             )
84:             if data is None or data.empty:
### ovx_metals_spillover.py 61
57: 
58: def _fetch_ovx_data() -> dict | None:
59:     """Fetch OVX historical data via yfinance."""
60:     try:
61:         import yfinance as yf
62: 
63:         ovx = yf.download("^OVX", period="15mo", progress=False)
64:         if ovx is None or len(ovx) < _MIN_HISTORY:
65:             return None
66: 
67:         if hasattr(ovx.columns, "levels") and len(ovx.columns.levels) > 1:
### residual_pair_reversion.py 79
75:     Cached for 1 hour to avoid redundant API calls.
76:     """
77:     def _do_fetch():
78:         try:
79:             import yfinance as yf
80:             data = yf.download(
81:                 yf_ticker, period="1y", interval="1d",
82:                 progress=False, auto_adjust=True,
83:             )
84:             if data is None or data.empty:
85:                 return None
### treasury_risk_rotation.py 52
48:     """Fetch IEF and TLT daily closes via yfinance.  Cached 1 hour."""
49: 
50:     def _do_fetch():
51:         try:
52:             import yfinance as yf
53: 
54:             data = yf.download(
55:                 ["IEF", "TLT"], period="14mo", progress=False, threads=True,
56:             )
57:             if data is None or data.empty:
58:                 return None
### vix_term_structure.py 46
42: 
43: def _fetch_vix_data() -> dict | None:
44:     """Fetch VIX and VIX3M closing prices via yfinance."""
45:     try:
46:         import yfinance as yf
47: 
48:         vix = yf.download("^VIX", period="2mo", progress=False)
49:         vix3m = yf.download("^VIX3M", period="2mo", progress=False)
50: 
51:         if vix is None or vix3m is None or len(vix) < _LOOKBACK or len(vix3m) < _LOOKBACK:
52:             return None
### xtrend_equity_spillover.py 57
53:     Cached for 1 hour.
54:     """
55:     def _do_fetch():
56:         try:
57:             import yfinance as yf
58: 
59:             data = yf.download(
60:                 ["SPY", "QQQ"], period="6mo", progress=False, threads=True,
61:             )
62:             if data is None or data.empty:
63:                 return None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\tests\\test_dxy_cross_asset_signal.py" in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/signals/dxy_cross_asset.py' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 600ms:
"""DXY cross-asset signal — inverse correlation with metals on intraday.

Why this exists: DXY (US Dollar Index) has R² ~0.6 inverse correlation
with silver/gold on 1-3h horizons — arguably the single strongest
short-term directional driver for metals. The existing macro_regime
signal has DXY as 1 of 6 sub-indicators at daily resolution, diluting
its vote. This standalone signal captures DXY directly at 60m resolution
so intraday dollar moves translate to a real BUY/SELL vote for metals.

Metals-only (XAU-USD, XAG-USD). Crypto + equities get HOLD.

Direction: DXY up → USD strong → metals weak → SELL. DXY down → BUY.
Threshold: 0.15% 1h move. DXY typical 1h range is 0.05-0.10% — a 0.15%
move is a clearly directional hour. Confidence scales linearly with move
magnitude up to a 0.5% cap (a ~5 sigma move saturates confidence).

Complements rather than replaces macro_regime's daily DXY vote. Keep both.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("portfolio.signals.dxy_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

# 1h DXY move threshold. Typical 1h DXY range is ~0.05-0.10%; 0.15% is a
# ~2-sigma move. Below this, signal votes HOLD.
_DXY_1H_THRESHOLD_PCT = 0.15

# Above this 1h move magnitude, confidence saturates at 1.0. A 0.5% 1h
# move in DXY is exceptional — think major macro data release.
_DXY_1H_CONFIDENCE_CAP_PCT = 0.5


def compute_dxy_cross_asset_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute DXY-direction vote for a metals ticker.

    Args:
        df: OHLCV DataFrame (unused — DXY data fetched separately).
        context: dict with ``ticker`` key. Other keys ignored.

    Returns:
        dict with action, confidence, sub_signals, indicators.
        HOLD with 0 confidence if ticker is not metals or DXY data is
        unavailable.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    if ticker not in _METALS_TICKERS:
        return empty

    try:
        from portfolio.macro_context import get_dxy_intraday
    except ImportError:
        logger.debug("macro_context.get_dxy_intraday not available")
        return empty

    data = get_dxy_intraday()
    if data is None:
        return empty

    change_1h = data.get("change_1h_pct")
    if change_1h is None:
        return empty

    if change_1h < -_DXY_1H_THRESHOLD_PCT:
        # Weak USD → strong metals → BUY
        action = "BUY"
        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    elif change_1h > _DXY_1H_THRESHOLD_PCT:
        # Strong USD → weak metals → SELL
        action = "SELL"
        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    else:
        action = "HOLD"
        confidence = 0.0

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {"dxy_1h": action},
        "indicators": {
            "dxy_value": data.get("value"),
            "dxy_change_1h_pct": round(change_1h, 3),
            "dxy_change_3h_pct": (
                round(data["change_3h_pct"], 3)
                if data.get("change_3h_pct") is not None else None
            ),
            "source": data.get("source", "unknown"),
        },
    }

 succeeded in 604ms:
"""Tests for portfolio.signals.dxy_cross_asset."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def _make_df(n=30):
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1000.0] * n,
    })


class TestDxyCrossAssetSignal:
    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_dxy_down_buys_silver(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.4,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "BUY"
        assert result["confidence"] > 0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_dxy_up_sells_silver(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": 0.25, "change_3h_pct": 0.4,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "SELL"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_dxy_down_buys_gold(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.3, "change_3h_pct": -0.5,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
        assert result["action"] == "BUY"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_below_threshold_holds(self, mock_dxy):
        """0.1% is below the 0.15% threshold → HOLD."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.1, "change_3h_pct": -0.15,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_non_metals_returns_hold_without_fetching(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="NVDA")
        assert result["action"] == "HOLD"
        mock_dxy.assert_not_called()

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_non_metals_btc_returns_hold(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="BTC-USD")
        assert result["action"] == "HOLD"
        mock_dxy.assert_not_called()

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_missing_dxy_returns_hold(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = None
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_missing_change_1h_returns_hold(self, mock_dxy):
        """If DXY data exists but 1h field is None, vote HOLD."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": None, "change_3h_pct": -0.3,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "HOLD"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_confidence_scales_with_move(self, mock_dxy):
        """Confidence scales linearly with |change_1h| up to 0.5% cap."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal

        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.5,
            "source": "DX-Y.NYB",
        }
        r1 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")

        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.5, "change_3h_pct": -0.8,
            "source": "DX-Y.NYB",
        }
        r2 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")

        assert r2["confidence"] > r1["confidence"]
        # A 0.5% move saturates at 1.0
        assert r2["confidence"] == 1.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_confidence_caps_at_one(self, mock_dxy):
        """Enormous DXY moves saturate at confidence=1.0."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -2.0, "change_3h_pct": -3.0,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["confidence"] == 1.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_indicators_include_source(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.2, "change_3h_pct": -0.3,
            "source": "EURUSD=X-synth",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["indicators"]["source"] == "EURUSD=X-synth"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_sub_signals_present(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.3,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["sub_signals"]["dxy_1h"] == "BUY"


class TestMacroContextIntradayDxy:
    """Tests for macro_context._fetch_dxy_intraday fallback logic."""

    @patch("yfinance.download")
    def test_primary_source_used_when_available(self, mock_dl):
        import pandas as pd

        from portfolio.macro_context import _fetch_dxy_intraday
        # Generate 5 hours of DX-Y.NYB bars
        idx = pd.date_range("2026-04-13 09:00", periods=5, freq="h")
        df = pd.DataFrame(
            {"Close": [104.0, 104.1, 104.2, 104.3, 104.4]}, index=idx,
        )
        mock_dl.return_value = df
        result = _fetch_dxy_intraday()
        assert result is not None
        assert result["source"] == "DX-Y.NYB"
        # change_1h: (104.4 / 104.3 - 1) * 100 ≈ +0.096%
        assert abs(result["change_1h_pct"] - 0.0959) < 0.01
        # change_3h: (104.4 / 104.1 - 1) * 100 ≈ +0.288%
        assert abs(result["change_3h_pct"] - 0.2882) < 0.01

    @patch("yfinance.download")
    def test_falls_back_to_eurusd_synth_on_empty(self, mock_dl):
        import pandas as pd

        from portfolio.macro_context import _fetch_dxy_intraday

        idx = pd.date_range("2026-04-13 09:00", periods=5, freq="h")
        # First call (DX-Y.NYB) returns empty; second (EURUSD=X) returns data
        empty_df = pd.DataFrame()
        eurusd_df = pd.DataFrame(
            {"Close": [1.170, 1.171, 1.172, 1.173, 1.174]}, index=idx,
        )
        mock_dl.side_effect = [empty_df, eurusd_df]
        result = _fetch_dxy_intraday()
        assert result is not None
        assert result["source"] == "EURUSD=X-synth"
        # Synth: 58 * eurusd^-0.576 — verify direction: eurusd went UP → synth DXY should go DOWN
        assert result["change_1h_pct"] < 0

    @patch("yfinance.download")
    def test_all_sources_fail_returns_none(self, mock_dl):
        import pandas as pd

        from portfolio.macro_context import _fetch_dxy_intraday
        mock_dl.return_value = pd.DataFrame()  # both DX-Y.NYB and EURUSD empty
        result = _fetch_dxy_intraday()
        assert result is None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/signals/vix_term_structure.py -TotalCount 120' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 528ms:
"""VIX term structure signal — contango/backwardation regime detection.

Uses VIX/VIX3M ratio to detect risk regime shifts. Backwardation
(VIX > VIX3M) occurs <20% of the time and signals market stress.
Deep contango (ratio < 0.85) signals complacency — contrarian BUY
on recovery.

Backtested strongest on metals (SLV 55%/64%/73% at 1d/3d/5d,
GLD 55%/61% at 1d/3d). Weaker on BTC/equities (~40-50%). Applied
to all assets; per-ticker accuracy gating auto-disables for assets
where it underperforms.

Sub-indicators:
    1. Backwardation flag     — ratio >= 1.0 = stress
    2. Contango depth         — how far below 1.0 (deep = complacent)
    3. VIX ratio z-score      — any deviation from 20d mean
    4. Ratio slope (5d)       — rate of change signals transitions

Data: yfinance ^VIX and ^VIX3M (free, no API key).
"""
from __future__ import annotations

import logging

import numpy as np

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.vix_term_structure")

_MAX_CONFIDENCE = 0.7

_BACKWARDATION_THRESHOLD = 1.0
_STRONG_BACKWARDATION = 1.05
_DEEP_CONTANGO = 0.85
_Z_THRESHOLD = 0.0  # any deviation votes; backtested: z=0.0 >> z=1.0
_SLOPE_THRESHOLD_PCT = 2.0
_LOOKBACK = 20
_CACHE_TTL = 900


def _fetch_vix_data() -> dict | None:
    """Fetch VIX and VIX3M closing prices via yfinance."""
    try:
        import yfinance as yf

        vix = yf.download("^VIX", period="2mo", progress=False)
        vix3m = yf.download("^VIX3M", period="2mo", progress=False)

        if vix is None or vix3m is None or len(vix) < _LOOKBACK or len(vix3m) < _LOOKBACK:
            return None

        close_col = "Close"
        if hasattr(vix.columns, "levels") and len(vix.columns.levels) > 1:
            vix = vix.droplevel(level=1, axis=1)
        if hasattr(vix3m.columns, "levels") and len(vix3m.columns.levels) > 1:
            vix3m = vix3m.droplevel(level=1, axis=1)

        vix_close = vix[close_col].dropna()
        vix3m_close = vix3m[close_col].dropna()

        if len(vix_close) < _LOOKBACK or len(vix3m_close) < _LOOKBACK:
            return None

        idx = vix_close.index.intersection(vix3m_close.index)
        if len(idx) < _LOOKBACK:
            return None

        vix_vals = vix_close.loc[idx].values.astype(float)
        vix3m_vals = vix3m_close.loc[idx].values.astype(float)

        mask = (vix3m_vals > 0) & np.isfinite(vix_vals) & np.isfinite(vix3m_vals)
        vix_vals = vix_vals[mask]
        vix3m_vals = vix3m_vals[mask]

        if len(vix_vals) < _LOOKBACK:
            return None

        ratio = vix_vals / vix3m_vals

        return {
            "vix_current": float(vix_vals[-1]),
            "vix3m_current": float(vix3m_vals[-1]),
            "ratio_current": float(ratio[-1]),
            "ratio_series": ratio.tolist(),
        }
    except Exception as e:
        logger.debug("VIX term structure fetch failed: %s", e)
        return None


def _backwardation_flag(ratio: float) -> str:
    if ratio >= _BACKWARDATION_THRESHOLD:
        return "SELL"
    if ratio < _DEEP_CONTANGO:
        return "BUY"
    return "HOLD"


def _contango_depth(ratio: float) -> str:
    depth = 1.0 - ratio
    if depth > 0.10:
        return "BUY"
    if depth < 0.0:
        return "SELL"
    return "HOLD"


def _ratio_zscore(ratio_series: list[float]) -> tuple[float, str]:
    arr = np.array(ratio_series[-_LOOKBACK:])
    if len(arr) < _LOOKBACK:
        return 0.0, "HOLD"
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-8:
        return 0.0, "HOLD"
    z = (arr[-1] - mean) / std
    if z > _Z_THRESHOLD:
        return float(z), "SELL"

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast
from pathlib import Path
for p in sorted(Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals').glob('*.py')):
    if p.name=='__init__.py': continue
    tree=ast.parse(p.read_text(encoding='utf-8'))
    funcs=[n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('compute_')]
    print(p.stem, funcs)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 629ms:
calendar_seasonal ['compute_calendar_signal']
candlestick ['compute_candlestick_signal']
claude_fundamental ['compute_claude_fundamental_signal']
complexity_gap_regime ['compute_complexity_gap_regime_signal']
copper_gold_ratio ['compute_copper_gold_ratio_signal']
cot_positioning ['compute_cot_positioning_signal']
credit_spread ['compute_credit_spread_signal']
cross_asset_tsmom ['compute_cross_asset_tsmom_signal']
crypto_cross_asset ['compute_crypto_cross_asset_signal']
crypto_evrp ['compute_crypto_evrp_signal']
crypto_macro ['compute_crypto_macro_signal']
drift_regime_gate ['compute_drift_regime_gate_signal']
dxy_cross_asset ['compute_dxy_cross_asset_signal']
econ_calendar ['compute_econ_calendar_signal']
fibonacci ['compute_fibonacci_signal']
forecast ['compute_forecast_signal']
futures_basis ['compute_futures_basis_signal']
futures_flow ['compute_futures_flow_signal']
gold_real_yield_paradox ['compute_gold_real_yield_paradox_signal']
hash_ribbons ['compute_hash_ribbons_signal']
heikin_ashi ['compute_heikin_ashi_signal']
hurst_regime ['compute_hurst_regime_signal']
macro_regime ['compute_macro_regime_signal']
mahalanobis_turbulence ['compute_mahalanobis_turbulence_signal']
mean_reversion ['compute_mean_reversion_signal']
metals_cross_asset ['compute_metals_cross_asset_signal']
momentum ['compute_momentum_signal']
momentum_factors ['compute_momentum_factors_signal']
network_momentum ['compute_network_momentum_signal']
news_event ['compute_news_event_signal']
orderbook_flow ['compute_orderbook_flow_signal']
oscillators ['compute_oscillator_signal']
ovx_metals_spillover ['compute_ovx_metals_spillover_signal']
realized_skewness ['compute_realized_skewness_signal']
residual_pair_reversion ['compute_residual_pair_reversion_signal']
shannon_entropy ['compute_shannon_entropy_signal']
smart_money ['compute_smart_money_signal']
statistical_jump_regime ['compute_statistical_jump_regime_signal']
structure ['compute_structure_signal']
treasury_risk_rotation ['compute_treasury_risk_rotation_signal']
trend ['compute_trend_signal']
vix_term_structure ['compute_vix_term_structure_signal']
vol_ratio_regime ['compute_vol_ratio_regime_signal']
volatility ['compute_volatility_signal']
volume_flow ['compute_volume_flow_signal']
williams_vix_fix ['compute_williams_vix_fix_signal']
xtrend_equity_spillover ['compute_xtrend_equity_spillover_signal']

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import pandas as pd
from portfolio.signals.crypto_cross_asset import compute_crypto_cross_asset_signal

def _df(prices):
    return pd.DataFrame({
        'open': prices,
        'high': [p+1 for p in prices],
        'low': [p-1 for p in prices],
        'close': prices,
        'volume': [1000]*len(prices),
    })

cases=[
    ('noncrypto', compute_crypto_cross_asset_signal(_df([100,101]), {'ticker':'XAU-USD'})),
    ('btc no ctx', compute_crypto_cross_asset_signal(_df([105000,105100]), {'ticker':'BTC-USD'})),
    ('fear', compute_crypto_cross_asset_signal(_df([105000,105100]), {'ticker':'BTC-USD','fear_greed':10})),
    ('greed', compute_crypto_cross_asset_signal(_df([105000,105100]), {'ticker':'BTC-USD','fear_greed':90})),
    ('eth bull', compute_crypto_cross_asset_signal(_df([3500,3675]), {'ticker':'ETH-USD','eth_btc_ratio_change_24h_pct':6})),
]
for name,res in cases:
    print(name,res)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 959ms:
noncrypto {'signal': 'HOLD', 'confidence': 0.0, 'sub_signals': {}, 'reason': 'non-crypto ticker'}
btc no ctx {'signal': 'HOLD', 'confidence': 0.0, 'sub_signals': {}, 'reason': 'no cross-asset data available'}
fear {'signal': 'HOLD', 'confidence': 0.0, 'sub_signals': {}, 'reason': 'no cross-asset data available'}
greed {'signal': 'HOLD', 'confidence': 0.0, 'sub_signals': {}, 'reason': 'no cross-asset data available'}
eth bull {'signal': 'HOLD', 'confidence': 0.0, 'sub_signals': {}, 'reason': 'no cross-asset data available'}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\tests\\test_crypto_cross_asset_signal.py" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 580ms:
"""Tests for portfolio/signals/crypto_cross_asset.py."""
from __future__ import annotations

import pandas as pd
import pytest

from portfolio.signals.crypto_cross_asset import compute_crypto_cross_asset_signal


def _df(closes):
    return pd.DataFrame({"Close": closes})


class TestNonCryptoTickers:
    @pytest.mark.parametrize("ticker", ["XAU-USD", "XAG-USD", "MSTR", "AAPL", ""])
    def test_returns_hold(self, ticker):
        out = compute_crypto_cross_asset_signal(_df([100, 101]),
                                                {"ticker": ticker})
        assert out["signal"] == "HOLD"
        assert "non-crypto" in out["reason"]


class TestNoData:
    def test_btc_no_cross_asset_returns_hold(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]),
                                                {"ticker": "BTC-USD"})
        assert out["signal"] == "HOLD"
        assert "no cross-asset" in out["reason"]


class TestFearGreed:
    def test_extreme_fear_buys(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 15,
                                            "classification": "Extreme Fear"}},
        })
        assert out["signal"] == "BUY"
        assert out["sub_signals"]["fear_greed"]["decision"] == "BUY"

    def test_extreme_greed_sells(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 90,
                                            "classification": "Extreme Greed"}},
        })
        assert out["signal"] == "SELL"

    def test_neutral_holds(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 50,
                                            "classification": "Neutral"}},
        })
        assert out["signal"] == "HOLD"


class TestEthBtcRatio:
    def test_eth_outperformance_bullish_for_eth(self):
        # ETH +5%, BTC flat -> ratio rises -> BUY for ETH
        eth = _df([3500.0, 3675.0])
        btc = _df([100000.0, 100000.0])
        out = compute_crypto_cross_asset_signal(_df([3500, 3675]), {
            "ticker": "ETH-USD",
            "cross_asset": {"eth_history": eth, "btc_history": btc},
        })
        assert out["sub_signals"]["eth_btc_ratio"]["decision"] == "BUY"

    def test_eth_underperformance_sells_for_eth(self):
        eth = _df([3500.0, 3325.0])  # -5%
        btc = _df([100000.0, 100000.0])
        out = compute_crypto_cross_asset_signal(_df([3500, 3325]), {
            "ticker": "ETH-USD",
            "cross_asset": {"eth_history": eth, "btc_history": btc},
        })
        assert out["sub_signals"]["eth_btc_ratio"]["decision"] == "SELL"


class TestDxy:
    def test_dxy_up_bearish_for_crypto(self):
        dxy = _df([100.0, 101.0])  # +1% — crypto headwind
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"dxy_history": dxy},
        })
        assert out["sub_signals"]["dxy"]["decision"] == "SELL"

    def test_dxy_down_bullish(self):
        dxy = _df([100.0, 99.0])
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"dxy_history": dxy},
        })
        assert out["sub_signals"]["dxy"]["decision"] == "BUY"


class TestSpy:
    def test_spy_strong_up_bullish(self):
        spy = _df([580.0, 585.0])
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"spy_history": spy},
        })
        assert out["sub_signals"]["spy"]["decision"] == "BUY"


class TestGoldBtcRatio:
    def test_gold_strong_btc_weak_bearish(self):
        # Gold rises, BTC flat -> ratio rises -> SELL crypto
        gold = _df([4500.0, 4600.0])
        btc = _df([100000.0, 100000.0])
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"gold_history": gold, "btc_history": btc},
        })
        assert out["sub_signals"]["gold_btc_ratio"]["decision"] == "SELL"


class TestConfidenceBounds:
    def test_confidence_capped_at_max(self):
        # All five sub-indicators voting BUY
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {
                "fear_greed": {"value": 10, "classification": "Extreme Fear"},
                "dxy_history": _df([100.0, 99.0]),  # DXY down -> BUY
                "spy_history": _df([580.0, 585.0]),  # SPY up -> BUY
                "eth_history": _df([3500.0, 3500.0]),
                "btc_history": _df([100000.0, 100000.0]),
                "gold_history": _df([4500.0, 4400.0]),  # gold down -> BUY (gold/btc ratio falls)
            },
        })
        assert out["signal"] == "BUY"
        assert 0 < out["confidence"] <= 0.7  # _MAX_CONFIDENCE

    def test_confidence_zero_on_hold(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 50,
                                            "classification": "Neutral"}},
        })
        assert out["signal"] == "HOLD"
        assert out["confidence"] == 0.0

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import pandas as pd
from portfolio.signals.crypto_cross_asset import compute_crypto_cross_asset_signal

def _df(closes):
    return pd.DataFrame({'Close': closes})

cases = [
    ('extreme fear', compute_crypto_cross_asset_signal(_df([105000,105100]), {'ticker':'BTC-USD','cross_asset': {'fear_greed': {'value':15,'classification':'Extreme Fear'}}})),
    ('extreme greed', compute_crypto_cross_asset_signal(_df([105000,105100]), {'ticker':'BTC-USD','cross_asset': {'fear_greed': {'value':90,'classification':'Extreme Greed'}}})),
    ('eth bull', compute_crypto_cross_asset_signal(_df([3500,3675]), {'ticker':'ETH-USD','cross_asset': {'eth_history': _df([3500.,3675.]), 'btc_history': _df([100000.,100000.])}})),
    ('dxy up', compute_crypto_cross_asset_signal(_df([105000,105100]), {'ticker':'BTC-USD','cross_asset': {'dxy_history': _df([100.,101.])}})),
    ('spy up', compute_crypto_cross_asset_signal(_df([105000,105100]), {'ticker':'BTC-USD','cross_asset': {'spy_history': _df([580.,585.])}})),
]
for name, res in cases:
    print(name, res)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 972ms:
extreme fear {'signal': 'BUY', 'confidence': 0.55, 'sub_signals': {'eth_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no ratio data'}}, 'fear_greed': {'decision': 'BUY', 'indicators': {'fear_greed': 15, 'classification': 'Extreme Fear'}}, 'dxy': {'decision': 'HOLD', 'indicators': {'reason': 'no DXY'}}, 'spy': {'decision': 'HOLD', 'indicators': {'reason': 'no SPY'}}, 'gold_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no gold/BTC data'}}}, 'reason': "votes={'HOLD': 4, 'BUY': 1}, active=1"}
extreme greed {'signal': 'SELL', 'confidence': 0.55, 'sub_signals': {'eth_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no ratio data'}}, 'fear_greed': {'decision': 'SELL', 'indicators': {'fear_greed': 90, 'classification': 'Extreme Greed'}}, 'dxy': {'decision': 'HOLD', 'indicators': {'reason': 'no DXY'}}, 'spy': {'decision': 'HOLD', 'indicators': {'reason': 'no SPY'}}, 'gold_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no gold/BTC data'}}}, 'reason': "votes={'HOLD': 4, 'SELL': 1}, active=1"}
eth bull {'signal': 'BUY', 'confidence': 0.55, 'sub_signals': {'eth_btc_ratio': {'decision': 'BUY', 'indicators': {'eth_btc_change_pct': 5.0}}, 'fear_greed': {'decision': 'HOLD', 'indicators': {'reason': 'no fear/greed'}}, 'dxy': {'decision': 'HOLD', 'indicators': {'reason': 'no DXY'}}, 'spy': {'decision': 'HOLD', 'indicators': {'reason': 'no SPY'}}, 'gold_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no gold/BTC data'}}}, 'reason': "votes={'BUY': 1, 'HOLD': 4}, active=1"}
dxy up {'signal': 'SELL', 'confidence': 0.55, 'sub_signals': {'eth_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no ratio data'}}, 'fear_greed': {'decision': 'HOLD', 'indicators': {'reason': 'no fear/greed'}}, 'dxy': {'decision': 'SELL', 'indicators': {'dxy_change_pct': 1.0}}, 'spy': {'decision': 'HOLD', 'indicators': {'reason': 'no SPY'}}, 'gold_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no gold/BTC data'}}}, 'reason': "votes={'HOLD': 4, 'SELL': 1}, active=1"}
spy up {'signal': 'BUY', 'confidence': 0.55, 'sub_signals': {'eth_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no ratio data'}}, 'fear_greed': {'decision': 'HOLD', 'indicators': {'reason': 'no fear/greed'}}, 'dxy': {'decision': 'HOLD', 'indicators': {'reason': 'no DXY'}}, 'spy': {'decision': 'BUY', 'indicators': {'spy_change_pct': 0.862}}, 'gold_btc_ratio': {'decision': 'HOLD', 'indicators': {'reason': 'no gold/BTC data'}}}, 'reason': "votes={'HOLD': 4, 'BUY': 1}, active=1"}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, inspect, traceback, sys
from pathlib import Path

def load(path):
    name = Path(path).stem + '_manual'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

paths = [
    r'Q:/finance-analyzer/tests/test_dxy_cross_asset_signal.py',
    r'Q:/finance-analyzer/tests/test_crypto_cross_asset_signal.py',
    r'Q:/finance-analyzer/tests/test_crypto_macro.py',
]
for path in paths:
    mod = load(path)
    print('MODULE', Path(path).name)
    total=passed=0
    for name,obj in vars(mod).items():
        if inspect.isclass(obj) and name.startswith('Test'):
            inst=obj()
            for meth_name,meth in inspect.getmembers(inst, predicate=callable):
                if meth_name.startswith('test_'):
                    sig=inspect.signature(meth)
                    # bound method signature excludes self
                    if len(sig.parameters)==0:
                        total+=1
                        try:
                            meth()
                            passed+=1
                            print(' PASS', name, meth_name)
                        except Exception as e:
                            print(' FAIL', name, meth_name, type(e).__name__, e)
                            traceback.print_exc(limit=2)
                    else:
                        print(' SKIP', name, meth_name, sig)
    print('SUMMARY', passed, '/', total)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 1113ms:
MODULE test_dxy_cross_asset_signal.py
 SKIP TestDxyCrossAssetSignal test_below_threshold_holds (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_confidence_caps_at_one (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_confidence_scales_with_move (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_dxy_down_buys_gold (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_dxy_down_buys_silver (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_dxy_up_sells_silver (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_indicators_include_source (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_missing_change_1h_returns_hold (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_missing_dxy_returns_hold (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_non_metals_btc_returns_hold (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_non_metals_returns_hold_without_fetching (mock_dxy)
 SKIP TestDxyCrossAssetSignal test_sub_signals_present (mock_dxy)
 SKIP TestMacroContextIntradayDxy test_all_sources_fail_returns_none (mock_dl)
 SKIP TestMacroContextIntradayDxy test_falls_back_to_eurusd_synth_on_empty (mock_dl)
 SKIP TestMacroContextIntradayDxy test_primary_source_used_when_available (mock_dl)
SUMMARY 0 / 0
MODULE test_crypto_cross_asset_signal.py
 SKIP TestNonCryptoTickers test_returns_hold (ticker)
 PASS TestNoData test_btc_no_cross_asset_returns_hold
 PASS TestFearGreed test_extreme_fear_buys
 PASS TestFearGreed test_extreme_greed_sells
 PASS TestFearGreed test_neutral_holds
 PASS TestEthBtcRatio test_eth_outperformance_bullish_for_eth
 PASS TestEthBtcRatio test_eth_underperformance_sells_for_eth
 PASS TestDxy test_dxy_down_bullish
 PASS TestDxy test_dxy_up_bearish_for_crypto
 PASS TestSpy test_spy_strong_up_bullish
 PASS TestGoldBtcRatio test_gold_strong_btc_weak_bearish
 PASS TestConfidenceBounds test_confidence_capped_at_max
 PASS TestConfidenceBounds test_confidence_zero_on_hold
SUMMARY 12 / 12
MODULE test_crypto_macro.py
 PASS TestDeribitOptions test_empty_response
 PASS TestDeribitOptions test_max_pain_simple
 PASS TestDeribitOptions test_no_open_interest
 PASS TestDeribitOptions test_put_call_ratio_calculation
 PASS TestGoldBtcRatio test_missing_prices
 PASS TestGoldBtcRatio test_ratio_computed
 PASS TestCryptoMacroSignal test_btc_returns_signal
 PASS TestCryptoMacroSignal test_gold_rotation_btc_outperforming
 PASS TestCryptoMacroSignal test_gold_rotation_gold_outperforming
 PASS TestCryptoMacroSignal test_netflow_accumulation
 PASS TestCryptoMacroSignal test_netflow_distribution
 PASS TestCryptoMacroSignal test_non_crypto_returns_hold
 PASS TestCryptoMacroSignal test_options_gravity_above_max_pain
 PASS TestCryptoMacroSignal test_options_gravity_below_max_pain
 PASS TestCryptoMacroSignal test_options_gravity_far_expiry
 PASS TestCryptoMacroSignal test_options_gravity_near_max_pain
 PASS TestCryptoMacroSignal test_pcr_high_contrarian_buy
 PASS TestCryptoMacroSignal test_pcr_low_contrarian_sell
 PASS TestCryptoMacroSignal test_quarterly_expiry_day
SUMMARY 19 / 19

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, sys
from pathlib import Path
path = r'Q:/finance-analyzer/tests/test_dxy_cross_asset_signal.py'
spec = importlib.util.spec_from_file_location('tmpmod', path)
mod = importlib.util.module_from_spec(spec)
sys.modules['tmpmod']=mod
spec.loader.exec_module(mod)
obj = mod.TestDxyCrossAssetSignal()
for name in ['test_dxy_down_buys_silver','test_non_metals_returns_hold_without_fetching']:
    meth = getattr(obj, name)
    print(name, meth, meth.__code__.co_argcount)
    try:
        meth()
        print('PASS')
    except Exception as e:
        print('FAIL', type(e).__name__, e)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 1066ms:
test_dxy_down_buys_silver <bound method TestDxyCrossAssetSignal.test_dxy_down_buys_silver of <tmpmod.TestDxyCrossAssetSignal object at 0x00000226F5AA71A0>> 0
PASS
test_non_metals_returns_hold_without_fetching <bound method TestDxyCrossAssetSignal.test_non_metals_returns_hold_without_fetching of <tmpmod.TestDxyCrossAssetSignal object at 0x00000226F5AA71A0>> 0
PASS

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, inspect, traceback, sys
from pathlib import Path

def load(path):
    name = Path(path).stem + '_manual2'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

paths = [
    r'Q:/finance-analyzer/tests/test_dxy_cross_asset_signal.py',
    r'Q:/finance-analyzer/tests/test_bug_fixes_session_20260321.py',
    r'Q:/finance-analyzer/tests/test_alpha_vantage.py',
]
for path in paths:
    mod = load(path)
    print('MODULE', Path(path).name)
    total=passed=0
    for name,obj in vars(mod).items():
        if inspect.isclass(obj) and name.startswith('Test'):
            inst=obj()
            for meth_name,meth in inspect.getmembers(inst, predicate=callable):
                if meth_name.startswith('test_'):
                    argc = meth.__code__.co_argcount
                    if argc==0:
                        total+=1
                        try:
                            meth()
                            passed+=1
                            print(' PASS', name, meth_name)
                        except Exception as e:
                            print(' FAIL', name, meth_name, type(e).__name__, e)
                            traceback.print_exc(limit=2)
                    else:
                        print(' SKIP', name, meth_name, 'argc', argc)
    print('SUMMARY', passed, '/', total)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 1467ms:
MODULE test_dxy_cross_asset_signal.py
 PASS TestDxyCrossAssetSignal test_below_threshold_holds
 PASS TestDxyCrossAssetSignal test_confidence_caps_at_one
 PASS TestDxyCrossAssetSignal test_confidence_scales_with_move
 PASS TestDxyCrossAssetSignal test_dxy_down_buys_gold
 PASS TestDxyCrossAssetSignal test_dxy_down_buys_silver
 PASS TestDxyCrossAssetSignal test_dxy_up_sells_silver
 PASS TestDxyCrossAssetSignal test_indicators_include_source
 PASS TestDxyCrossAssetSignal test_missing_change_1h_returns_hold
 PASS TestDxyCrossAssetSignal test_missing_dxy_returns_hold
 PASS TestDxyCrossAssetSignal test_non_metals_btc_returns_hold
 PASS TestDxyCrossAssetSignal test_non_metals_returns_hold_without_fetching
 PASS TestDxyCrossAssetSignal test_sub_signals_present
 PASS TestMacroContextIntradayDxy test_all_sources_fail_returns_none
 PASS TestMacroContextIntradayDxy test_falls_back_to_eurusd_synth_on_empty
 PASS TestMacroContextIntradayDxy test_primary_source_used_when_available
SUMMARY 15 / 15
MODULE test_bug_fixes_session_20260321.py
 SKIP TestSentimentFlushCrashSafety test_dirty_flag_cleared_after_successful_write argc 2
 SKIP TestSentimentFlushCrashSafety test_dirty_flag_remains_on_write_failure argc 2
 SKIP TestSentimentFlushCrashSafety test_not_dirty_skips_write argc 1
 SKIP TestSentimentFlushCrashSafety test_retry_after_failure argc 2
 SKIP TestForecastThreadSafety test_concurrent_trip_and_check argc 1
 SKIP TestForecastThreadSafety test_log_health_reset_under_lock argc 1
 SKIP TestPortfolioPnlZeroDivGuard test_missing_initial_value_uses_fallback argc 1
 SKIP TestPortfolioPnlZeroDivGuard test_valid_initial_value_used_directly argc 1
 SKIP TestPortfolioPnlZeroDivGuard test_zero_initial_value_uses_fallback argc 1
 SKIP TestCalendarExceptionLogging test_all_sub_signals_fail_returns_hold argc 2
 SKIP TestCalendarExceptionLogging test_sub_signal_failure_logged argc 2
 SKIP TestFxAlertRouting test_fallback_alert_uses_error_category argc 1
 SKIP TestFxAlertRouting test_stale_rate_alert_uses_error_category argc 1
 SKIP TestForecastDedupEviction test_eviction_constant_exists argc 1
 SKIP TestForecastDedupEviction test_forecast_lock_exists argc 1
 SKIP TestForecastDedupEviction test_stale_entries_evicted argc 1
SUMMARY 0 / 0
MODULE test_alpha_vantage.py
 SKIP TestNormalizeOverview test_analyst_ratings argc 1
 SKIP TestNormalizeOverview test_dash_field argc 1
 SKIP TestNormalizeOverview test_empty_response argc 1
 SKIP TestNormalizeOverview test_empty_string_field argc 1
 SKIP TestNormalizeOverview test_error_response argc 1
 SKIP TestNormalizeOverview test_fetched_at_is_recent argc 1
 SKIP TestNormalizeOverview test_missing_symbol argc 1
 SKIP TestNormalizeOverview test_none_string_fields argc 1
 SKIP TestNormalizeOverview test_normal_response argc 1
 SKIP TestNormalizeOverview test_rate_limit_note argc 1
 SKIP TestCache test_cache_age_hours_none argc 1
 SKIP TestCache test_cache_age_hours_recent argc 1
 SKIP TestCache test_fresh_detection argc 1
 SKIP TestCache test_get_all_fundamentals_empty argc 1
 SKIP TestCache test_get_all_fundamentals_returns_copy argc 1
 SKIP TestCache test_get_fundamentals_hit argc 1
 SKIP TestCache test_get_fundamentals_miss argc 1
 SKIP TestCache test_load_persistent_cache_corrupt argc 2
 SKIP TestCache test_load_persistent_cache_missing_file argc 1
 SKIP TestCache test_load_persistent_cache_valid argc 2
 SKIP TestCache test_stale_detection argc 1
 SKIP TestBatchRefresh test_all_fresh_no_refresh argc 1
 PASS TestBatchRefresh test_budget_exhaustion
 PASS TestBatchRefresh test_circuit_breaker_blocks
 FAIL TestBatchRefresh test_circuit_breaker_trips AssertionError 
 SKIP TestBatchRefresh test_disabled_returns_zero argc 1
 SKIP TestBatchRefresh test_no_api_key_returns_zero argc 1
 FAIL TestBatchRefresh test_partial_failure AssertionError 
 PASS TestBatchRefresh test_skips_qqq
 FAIL TestBatchRefresh test_successful_refresh AssertionError 
 SKIP TestShouldBatchRefresh test_disabled argc 1
 SKIP TestShouldBatchRefresh test_needs_refresh_empty_cache argc 1
 SKIP TestShouldBatchRefresh test_no_api_key argc 1
 SKIP TestShouldBatchRefresh test_no_refresh_all_fresh argc 1
 SKIP TestBuildFundamentalsBlock test_empty_fundamentals argc 1
 SKIP TestBuildFundamentalsBlock test_haiku_one_liner argc 1
 SKIP TestBuildFundamentalsBlock test_missing_ticker argc 1
 SKIP TestBuildFundamentalsBlock test_none_fundamentals argc 1
 SKIP TestBuildFundamentalsBlock test_opus_same_as_sonnet argc 1
 SKIP TestBuildFundamentalsBlock test_sonnet_detailed argc 1
SUMMARY 3 / 6
Circuit breaker 'alpha_vantage': CLOSED -> OPEN (threshold 3 reached)
Traceback (most recent call last):
  File "<stdin>", line 30, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py", line 1396, in patched
    return func(*newargs, **newkeywargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Traceback (most recent call last):
  File "<stdin>", line 30, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py", line 1396, in patched
    return func(*newargs, **newkeywargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Traceback (most recent call last):
  File "<stdin>", line 30, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py", line 1396, in patched
    return func(*newargs, **newkeywargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import os, re, importlib.util, inspect, traceback, sys
from pathlib import Path
root = Path(r'Q:/finance-analyzer/tests')
files=[]
for p in root.glob('test_*.py'):
    txt = p.read_text(encoding='utf-8', errors='ignore')
    if 'portfolio.signals.' in txt:
        files.append(p)
print('files', len(files))
failures=[]
for p in files:
    name = p.stem + '_manual3'
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print('IMPORT FAIL', p.name, e)
        failures.append((str(p),'import',repr(e)))
        continue
    for cname,obj in vars(mod).items():
        if inspect.isclass(obj) and cname.startswith('Test'):
            inst=obj()
            for mname,meth in inspect.getmembers(inst, predicate=callable):
                if mname.startswith('test_') and meth.__code__.co_argcount==0:
                    try:
                        meth()
                    except Exception as e:
                        failures.append((str(p), f'{cname}.{mname}', repr(e)))
                        print('FAIL', p.name, cname, mname, type(e).__name__, e)
print('TOTAL FAILURES', len(failures))
for f in failures[:50]:
    print(f)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 4876ms:
files 68
FAIL test_alpha_vantage.py TestBatchRefresh test_circuit_breaker_trips AssertionError 
FAIL test_alpha_vantage.py TestBatchRefresh test_partial_failure AssertionError 
FAIL test_alpha_vantage.py TestBatchRefresh test_successful_refresh AssertionError 
FAIL test_forecast_circuit_breaker.py TestKronosCircuitBreaker test_does_not_trip_on_success AssertionError 
FAIL test_forecast_circuit_breaker.py TestKronosCircuitBreaker test_skips_when_tripped AssertionError 
FAIL test_forecast_circuit_breaker.py TestKronosCircuitBreaker test_trips_on_empty_results AssertionError 
FAIL test_forecast_circuit_breaker.py TestKronosCircuitBreaker test_trips_on_exception AssertionError 
FAIL test_forecast_circuit_breaker.py TestKronosCircuitBreaker test_trips_on_subprocess_failure AssertionError 
FAIL test_forecast_circuit_breaker.py TestForecastFullPathEnabled test_full_path_loads_candles_and_calls_models AssertionError Expected '_load_candles_ohlcv' to be called once. Called 0 times.
FAIL test_forecast_circuit_breaker.py TestForecastFullPathEnabled test_full_path_majority_buy AssertionError 
FAIL test_forecast_timeout.py TestPredictionDedup test_dedup_prevents_double_logging AssertionError 
FAIL test_forecast_timeout.py TestPredictionDedup test_different_tickers_not_deduped AssertionError 
FAIL test_forecast_timeout.py TestAccuracyMetadata test_predictions_accuracy_null_when_no_data AssertionError 
FAIL test_forecast_timeout.py TestAccuracyMetadata test_predictions_include_accuracy_metadata AssertionError 
FAIL test_futures_flow.py TestComputeFuturesFlowSignal test_confidence_cap TypeError TestComputeFuturesFlowSignal.test_confidence_cap() missing 1 required positional argument: 'sample_df'
FAIL test_futures_flow.py TestComputeFuturesFlowSignal test_fetch_failure_hold TypeError TestComputeFuturesFlowSignal.test_fetch_failure_hold() missing 1 required positional argument: 'sample_df'
FAIL test_futures_flow.py TestComputeFuturesFlowSignal test_result_structure TypeError TestComputeFuturesFlowSignal.test_result_structure() missing 1 required positional argument: 'sample_df'
FAIL test_kronos_stdout_fix.py TestRunKronosContaminatedStdout test_error_diagnostic_logged_on_failure TypeError TestRunKronosContaminatedStdout.test_error_diagnostic_logged_on_failure() missing 1 required positional argument: 'caplog'
FAIL test_kronos_stdout_fix.py TestRunKronosContaminatedStdout test_stdout_clean_json_still_works AssertionError 
FAIL test_kronos_stdout_fix.py TestRunKronosContaminatedStdout test_stdout_multi_line_warnings_then_json AssertionError 
FAIL test_kronos_stdout_fix.py TestRunKronosContaminatedStdout test_stdout_with_prefix_text_then_json AssertionError 
FAIL test_metals_cross_asset_signal.py TestGetCrossAssetContextHealth test_degraded_intraday_source_logs_warning TypeError TestGetCrossAssetContextHealth.test_degraded_intraday_source_logs_warning() missing 1 required positional argument: 'caplog'
TOTAL FAILURES 22
('Q:\\finance-analyzer\\tests\\test_alpha_vantage.py', 'TestBatchRefresh.test_circuit_breaker_trips', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_alpha_vantage.py', 'TestBatchRefresh.test_partial_failure', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_alpha_vantage.py', 'TestBatchRefresh.test_successful_refresh', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py', 'TestKronosCircuitBreaker.test_does_not_trip_on_success', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py', 'TestKronosCircuitBreaker.test_skips_when_tripped', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py', 'TestKronosCircuitBreaker.test_trips_on_empty_results', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py', 'TestKronosCircuitBreaker.test_trips_on_exception', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py', 'TestKronosCircuitBreaker.test_trips_on_subprocess_failure', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py', 'TestForecastFullPathEnabled.test_full_path_loads_candles_and_calls_models', 'AssertionError("Expected \'_load_candles_ohlcv\' to be called once. Called 0 times.")')
('Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py', 'TestForecastFullPathEnabled.test_full_path_majority_buy', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_timeout.py', 'TestPredictionDedup.test_dedup_prevents_double_logging', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_timeout.py', 'TestPredictionDedup.test_different_tickers_not_deduped', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_timeout.py', 'TestAccuracyMetadata.test_predictions_accuracy_null_when_no_data', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_forecast_timeout.py', 'TestAccuracyMetadata.test_predictions_include_accuracy_metadata', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_futures_flow.py', 'TestComputeFuturesFlowSignal.test_confidence_cap', 'TypeError("TestComputeFuturesFlowSignal.test_confidence_cap() missing 1 required positional argument: \'sample_df\'")')
('Q:\\finance-analyzer\\tests\\test_futures_flow.py', 'TestComputeFuturesFlowSignal.test_fetch_failure_hold', 'TypeError("TestComputeFuturesFlowSignal.test_fetch_failure_hold() missing 1 required positional argument: \'sample_df\'")')
('Q:\\finance-analyzer\\tests\\test_futures_flow.py', 'TestComputeFuturesFlowSignal.test_result_structure', 'TypeError("TestComputeFuturesFlowSignal.test_result_structure() missing 1 required positional argument: \'sample_df\'")')
('Q:\\finance-analyzer\\tests\\test_kronos_stdout_fix.py', 'TestRunKronosContaminatedStdout.test_error_diagnostic_logged_on_failure', 'TypeError("TestRunKronosContaminatedStdout.test_error_diagnostic_logged_on_failure() missing 1 required positional argument: \'caplog\'")')
('Q:\\finance-analyzer\\tests\\test_kronos_stdout_fix.py', 'TestRunKronosContaminatedStdout.test_stdout_clean_json_still_works', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_kronos_stdout_fix.py', 'TestRunKronosContaminatedStdout.test_stdout_multi_line_warnings_then_json', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_kronos_stdout_fix.py', 'TestRunKronosContaminatedStdout.test_stdout_with_prefix_text_then_json', 'AssertionError()')
('Q:\\finance-analyzer\\tests\\test_metals_cross_asset_signal.py', 'TestGetCrossAssetContextHealth.test_degraded_intraday_source_logs_warning', 'TypeError("TestGetCrossAssetContextHealth.test_degraded_intraday_source_logs_warning() missing 1 required positional argument: \'caplog\'")')
Circuit breaker 'alpha_vantage': CLOSED -> OPEN (threshold 3 reached)
metals_cross_asset: intraday source(s) unavailable ['oil'] — sub-signals for these will vote HOLD this cycle
Claude fundamental haiku bg-refresh failed: API down
Claude fundamental sonnet bg-refresh failed: API down
Claude fundamental opus bg-refresh failed: API down

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\tests\\test_forecast_circuit_breaker.py" in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/signals/forecast.py' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 684ms:
"""Tests for forecast signal circuit breaker.

Verifies that after one GPU failure, subsequent calls skip instantly
instead of waiting for timeouts on every ticker.
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio.signals.forecast import (
    _chronos_circuit_open,
    _health_weighted_vote,
    _kronos_circuit_open,
    _run_chronos,
    _run_kronos,
    compute_forecast_signal,
    reset_circuit_breakers,
)


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Reset circuit breakers, enable Kronos, and disable models_disabled before and after each test."""
    import portfolio.signals.forecast as mod
    orig_kronos = mod._KRONOS_ENABLED
    orig_disabled = mod._FORECAST_MODELS_DISABLED
    mod._KRONOS_ENABLED = True
    mod._FORECAST_MODELS_DISABLED = False
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()
    mod._KRONOS_ENABLED = orig_kronos
    mod._FORECAST_MODELS_DISABLED = orig_disabled


# --- Kronos disabled by default ---

class TestKronosDisabled:
    def test_kronos_disabled_returns_none(self):
        """When _KRONOS_ENABLED is False, _run_kronos returns None immediately."""
        import portfolio.signals.forecast as mod
        mod._KRONOS_ENABLED = False
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert not _kronos_circuit_open()  # should NOT trip breaker


# --- Kronos circuit breaker ---

class TestKronosCircuitBreaker:
    def test_initially_closed(self):
        assert not _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_trips_on_subprocess_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="CUDA error")
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_trips_on_exception(self, mock_run):
        mock_run.side_effect = TimeoutError("timed out")
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_trips_on_empty_results(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"method": "none", "results": {}}'
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.gpu_gate.get_vram_usage", return_value=None)
    @patch("portfolio.signals.forecast.subprocess.run")
    def test_skips_when_tripped(self, mock_run, _mock_vram):
        """After tripping, subsequent calls should NOT invoke subprocess.
        get_vram_usage is mocked to prevent it from also calling subprocess.run."""
        mock_run.side_effect = TimeoutError("timed out")
        _run_kronos([{"close": 100}] * 50)  # trips breaker
        assert mock_run.call_count == 1

        # Second call should be instant skip
        result = _run_kronos([{"close": 200}] * 50)
        assert result is None
        assert mock_run.call_count == 1  # NOT called again

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_does_not_trip_on_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"method": "kronos", "results": {"1h": {"direction": "up", "pct_move": 0.5, "confidence": 0.6}}}'
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is not None
        assert not _kronos_circuit_open()

    def test_resets_after_ttl(self):
        """Breaker should auto-reset after TTL expires."""
        import portfolio.signals.forecast as mod
        mod._kronos_tripped_until = time.monotonic() + 0.01  # trip for 10ms
        assert _kronos_circuit_open()
        time.sleep(0.02)
        assert not _kronos_circuit_open()


# --- Chronos circuit breaker ---

class TestChronosCircuitBreaker:
    def test_initially_closed(self):
        assert not _chronos_circuit_open()

    def test_trips_on_exception(self):
        mock_mod = MagicMock()
        mock_mod.forecast_chronos.side_effect = RuntimeError("CUDA error")
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            result = _run_chronos([100.0] * 50)
        assert result is None
        assert _chronos_circuit_open()

    def test_skips_when_tripped(self):
        """After tripping, subsequent calls should NOT attempt import."""
        import portfolio.signals.forecast as mod
        mod._chronos_tripped_until = time.monotonic() + 60
        result = _run_chronos([100.0] * 50)
        assert result is None

    def test_resets_after_ttl(self):
        import portfolio.signals.forecast as mod
        mod._chronos_tripped_until = time.monotonic() + 0.01
        assert _chronos_circuit_open()
        time.sleep(0.02)
        assert not _chronos_circuit_open()


# --- Integration: compute_forecast_signal with breakers ---

class TestComputeForecastWithBreaker:
    @patch("portfolio.signals.forecast._run_chronos", return_value=None)
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_returns_hold_when_both_tripped(self, mock_candles, mock_kronos, mock_chronos):
        mock_candles.return_value = [{"close": 100.0}] * 60
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.forecast._run_chronos", return_value=None)
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_circuit_indicators_reported(self, mock_candles, mock_kronos, mock_chronos):
        """Circuit breaker state should be visible in indicators."""
        import portfolio.signals.forecast as mod
        mod._kronos_tripped_until = time.monotonic() + 60
        mod._chronos_tripped_until = time.monotonic() + 60

        mock_candles.return_value = [{"close": 100.0}] * 60
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["indicators"]["kronos_circuit_open"] is True
        assert result["indicators"]["chronos_circuit_open"] is True

    @patch("portfolio.signals.forecast._run_chronos", return_value=None)
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_circuit_closed_indicators(self, mock_candles, mock_kronos, mock_chronos):
        mock_candles.return_value = [{"close": 100.0}] * 60
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["indicators"]["kronos_circuit_open"] is False
        assert result["indicators"]["chronos_circuit_open"] is False


class TestResetCircuitBreakers:
    def test_reset_clears_both(self):
        import portfolio.signals.forecast as mod
        mod._kronos_tripped_until = time.monotonic() + 999
        mod._chronos_tripped_until = time.monotonic() + 999
        assert _kronos_circuit_open()
        assert _chronos_circuit_open()

        reset_circuit_breakers()
        assert not _kronos_circuit_open()
        assert not _chronos_circuit_open()


# --- Forecast models disabled (top-level kill switch) ---

class TestForecastModelsDisabled:
    def test_returns_hold_immediately_when_disabled(self):
        """When _FORECAST_MODELS_DISABLED is True, returns HOLD with no model work."""
        import portfolio.signals.forecast as mod
        mod._FORECAST_MODELS_DISABLED = True
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert result["indicators"]["models_disabled"] is True
        # Sub-signals should all be HOLD (default)
        for v in result["sub_signals"].values():
            assert v == "HOLD"

    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_no_candle_fetch_when_disabled(self, mock_candles):
        """When disabled, should not even attempt to load candles."""
        import portfolio.signals.forecast as mod
        mod._FORECAST_MODELS_DISABLED = True
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        mock_candles.assert_not_called()
        assert result["indicators"]["models_disabled"] is True


class TestForecastFullPathEnabled:
    """Verify that when models are enabled, the full code path runs (candles + both models)."""

    @staticmethod
    def _bypass_cache(key, ttl, fn, *args):
        """Bypass _cached so mocks are called directly."""
        return fn(*args)

    @patch("portfolio.signals.forecast._gate_subsignal_votes_by_accuracy",
           side_effect=lambda raw, ticker, config_forecast=None: (dict(raw), {}))
    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos")
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_full_path_loads_candles_and_calls_models(self, mock_candles, mock_kronos, mock_chronos, mock_cached, mock_gate):
        """With _FORECAST_MODELS_DISABLED=False, candles are loaded and both models called.

        2026-04-17: patch _gate_subsignal_votes_by_accuracy to pass raw
        votes through — the accuracy gate was added after this test was
        written and would otherwise force sub-signals to HOLD when the
        test's mocked ticker has no historical accuracy data.
        """
        mock_cached.side_effect = self._bypass_cache
        mock_candles.return_value = [{"close": float(100 + i)} for i in range(80)]
        mock_kronos.return_value = {
            "method": "kronos",
            "results": {
                "1h": {"direction": "up", "pct_move": 0.3, "confidence": 0.6},
                "24h": {"direction": "down", "pct_move": -0.5, "confidence": 0.5},
            },
        }
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.4, "confidence": 0.55},
            "24h": {"action": "SELL", "pct_move": -0.6, "confidence": 0.5},
        }
        df = pd.DataFrame({"close": [100.0] * 80})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})

        # Candles should have been loaded
        mock_candles.assert_called_once_with("BTC-USD")
        # Both models should have been invoked
        mock_kronos.assert_called_once()
        mock_chronos.assert_called_once()
        # models_disabled should NOT be in indicators
        assert "models_disabled" not in result["indicators"]
        # Sub-signals should reflect model outputs
        assert result["sub_signals"]["kronos_1h"] == "BUY"
        assert result["sub_signals"]["kronos_24h"] == "SELL"
        assert result["sub_signals"]["chronos_1h"] == "BUY"
        assert result["sub_signals"]["chronos_24h"] == "SELL"
        # 1h gets 2x weight: [BUY,BUY,SELL, BUY,BUY,SELL] → 4 BUY vs 2 SELL → BUY
        assert result["action"] == "BUY"

    @patch("portfolio.signals.forecast._gate_subsignal_votes_by_accuracy",
           side_effect=lambda raw, ticker, config_forecast=None: (dict(raw), {}))
    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos")
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_full_path_majority_buy(self, mock_candles, mock_kronos, mock_chronos, mock_cached, mock_gate):
        """When 3/4 sub-signals are BUY, composite action should be BUY.

        2026-04-17: patch _gate_subsignal_votes_by_accuracy (same
        rationale as test_full_path_loads_candles_and_calls_models).
        """
        mock_cached.side_effect = self._bypass_cache
        mock_candles.return_value = [{"close": float(100 + i)} for i in range(80)]
        mock_kronos.return_value = {
            "method": "kronos",
            "results": {
                "1h": {"direction": "up", "pct_move": 0.3, "confidence": 0.6},
                "24h": {"direction": "up", "pct_move": 0.5, "confidence": 0.5},
            },
        }
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.4, "confidence": 0.55},
            "24h": {"action": "SELL", "pct_move": -0.2, "confidence": 0.3},
        }
        df = pd.DataFrame({"close": [100.0] * 80})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})

        assert result["action"] == "BUY"
        assert result["confidence"] <= 0.7  # capped
        assert result["sub_signals"]["kronos_1h"] == "BUY"
        assert result["sub_signals"]["kronos_24h"] == "BUY"
        assert result["sub_signals"]["chronos_1h"] == "BUY"
        assert result["sub_signals"]["chronos_24h"] == "SELL"


# --- Health-weighted vote (excludes dead models) ---

class TestHealthWeightedVote:
    """Test _health_weighted_vote — the core fix for the HOLD-always bug.

    When Kronos is dead, its 2 permanent HOLD sub-signals dilute the
    4-vote majority. _health_weighted_vote excludes dead models.
    """

    def test_chronos_only_buy(self):
        """When Kronos dead, Chronos 2 BUY -> BUY."""
        sub = {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
               "chronos_1h": "BUY", "chronos_24h": "BUY"}
        action, conf = _health_weighted_vote(sub, kronos_ok=False, chronos_ok=True)
        assert action == "BUY"
        assert conf > 0

    def test_chronos_only_sell(self):
        """When Kronos dead, Chronos 2 SELL -> SELL."""
        sub = {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
               "chronos_1h": "SELL", "chronos_24h": "SELL"}
        action, conf = _health_weighted_vote(sub, kronos_ok=False, chronos_ok=True)
        assert action == "SELL"
        assert conf > 0

    def test_chronos_only_split(self):
        """When Kronos dead, Chronos BUY+SELL -> BUY (1h gets 2x weight)."""
        sub = {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
               "chronos_1h": "BUY", "chronos_24h": "SELL"}
        action, conf = _health_weighted_vote(sub, kronos_ok=False, chronos_ok=True)
        # 1h gets 2x weight: [BUY, BUY, SELL] → BUY wins
        assert action == "BUY"

    def test_both_alive_majority(self):
        """Both alive, 3 BUY 1 SELL -> BUY."""
        sub = {"kronos_1h": "BUY", "kronos_24h": "BUY",
               "chronos_1h": "BUY", "chronos_24h": "SELL"}
        action, conf = _health_weighted_vote(sub, kronos_ok=True, chronos_ok=True)
        assert action == "BUY"
        assert conf > 0

    def test_both_dead(self):
        """Both dead -> HOLD with 0 confidence."""
        sub = {"kronos_1h": "BUY", "kronos_24h": "BUY",
               "chronos_1h": "BUY", "chronos_24h": "BUY"}
        action, conf = _health_weighted_vote(sub, kronos_ok=False, chronos_ok=False)
        assert action == "HOLD"
        assert conf == 0.0

    def test_kronos_only(self):
        """Chronos dead, Kronos 2 BUY -> BUY."""
        sub = {"kronos_1h": "BUY", "kronos_24h": "BUY",
               "chronos_1h": "HOLD", "chronos_24h": "HOLD"}
        action, conf = _health_weighted_vote(sub, kronos_ok=True, chronos_ok=False)
        assert action == "BUY"
        assert conf > 0

 succeeded in 725ms:
"""Forecast signal — price direction prediction from time-series models.

Combines Kronos (K-line foundation model), Chronos (Amazon), and Prophet
into a majority-vote composite signal with four sub-signals:
  1. kronos_1h  — Kronos 1-hour prediction
  2. kronos_24h — Kronos 24-hour prediction
  3. chronos_1h — Chronos 1-hour prediction
  4. chronos_24h — Chronos 24-hour prediction

Reuses candle loading from portfolio.forecast_signal. Confidence capped at 0.7.
Registered as enhanced signal #28 with weight=0 (shadow mode) initially.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from portfolio.file_utils import atomic_append_jsonl
from portfolio.gpu_gate import gpu_gate
from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.forecast")

# Cache TTL — forecasts don't change fast
_FORECAST_TTL = 300  # 5 minutes

# Confidence cap (same as news_event, econ_calendar)
_MAX_CONFIDENCE = 0.7

# Default Chronos timeout (seconds) — reduced from 120 to avoid long hangs
_CHRONOS_TIMEOUT = 60

# Default Kronos subprocess timeout (seconds) — reduced from 90; fails fast
_KRONOS_TIMEOUT = 30

# Forecast models master switch. Set to True to disable all model calls (early-return HOLD).
# Circuit breakers remain as secondary protection — auto-trip on failure, 5min TTL.
_FORECAST_MODELS_DISABLED = False

# Kronos inference — UN-RETIRED 2026-04-21 afternoon. The morning's retire
# was premature. The real problem was not Kronos per se but the shadow-mode
# implementation: when _KRONOS_SHADOW=True, forecast.py forced Kronos's
# composite-vote sub-signal to HOLD (see lines ~811/820) while still counting
# those HOLD votes inside _health_weighted_vote's majority tally — so when
# Chronos voted BUY/BUY/BUY and Kronos (silently HOLD) also "voted" H/H/H,
# the result was a 3-3 tie that broke to HOLD and polluted Chronos's verdict.
#
# The correct structure (this session):
#   - Kronos subprocess runs normally, real prediction captured in
#     raw_sub_signals["kronos_*"] for accuracy backfill.
#   - In shadow mode the Kronos sub-signal is EXCLUDED FROM THE COMPOSITE
#     VOTE POOL entirely (see _health_weighted_vote below) — it does not
#     contribute HOLD weights, does not dilute Chronos.
#   - Subprocess reliability (59 % success → 90 % target) remains a separate
#     work stream. The shadow-logging path tolerates failure gracefully;
#     un-retirement does not require the reliability fix to land first.
#
# Config re-read here so operators can flip kronos_enabled=true|"shadow"|false
# without a code edit, matching the pre-retire behavior.
_KRONOS_ENABLED = False
_KRONOS_SHADOW = False


def _init_kronos_enabled():
    """Read kronos_enabled from config.json at import time.

    Values:
      - `true`  → active voter (raw prediction contributes to composite vote)
      - `"shadow"` → inference runs, raw prediction is logged, but the
        sub-signal is EXCLUDED from the composite vote pool (see
        _health_weighted_vote). Zero composite pollution.
      - `false` (default) → subprocess skipped entirely.
    """
    global _KRONOS_ENABLED, _KRONOS_SHADOW
    try:
        from portfolio.file_utils import load_json as _load_json
        _cfg = _load_json(
            str(Path(__file__).resolve().parent.parent.parent / "config.json"),
            {},
        )
        val = _cfg.get("forecast", {}).get("kronos_enabled", False)
        if val == "shadow":
            _KRONOS_ENABLED = True
            _KRONOS_SHADOW = True
        else:
            _KRONOS_ENABLED = bool(val)
            _KRONOS_SHADOW = False
    except Exception as e:
        logger.debug("Kronos init from config: %s", e)


_init_kronos_enabled()

if platform.system() == "Windows":
    _KRONOS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    _KRONOS_SCRIPT = r"Q:\models\kronos_infer.py"
else:
    _KRONOS_PYTHON = "/home/deck/models/.venv/bin/python"
    _KRONOS_SCRIPT = "/home/deck/models/kronos_infer.py"

# Prediction log
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PREDICTIONS_FILE = _DATA_DIR / "forecast_predictions.jsonl"
_HEALTH_FILE = _DATA_DIR / "forecast_health.jsonl"

# Circuit breaker — after first failure, skip remaining tickers in this loop cycle.
# Prevents 27 x 6s GPU timeouts when CUDA is broken.
_CIRCUIT_BREAKER_TTL = 30  # 30 seconds before retry
_kronos_tripped_until = 0.0  # monotonic timestamp when breaker resets
_chronos_tripped_until = 0.0

# BUG-102: Lock protects circuit breaker state and dedup cache from ThreadPoolExecutor races.
# The read-check-write pattern in _log_health() is not atomic without a lock.
_forecast_lock = threading.Lock()

# Prediction dedup — track last logged timestamp per ticker to avoid
# logging cached replays. Key: ticker, value: ISO-8601 timestamp.
_PREDICTION_DEDUP_TTL = 60  # seconds — don't re-log within this window
_PREDICTION_DEDUP_EVICT_AGE = 600  # BUG-106: evict entries older than 10 minutes
_last_prediction_ts: dict[str, float] = {}  # ticker -> monotonic timestamp


def _extract_json_from_stdout(stdout: str | None) -> dict | None:
    """Extract JSON from potentially contaminated subprocess stdout.

    HuggingFace's from_pretrained() prints to stdout during model loading,
    which contaminates the subprocess output before the JSON result.
    This function handles that by finding the first '{' and parsing from there.

    Returns parsed dict on success, None on failure.
    """
    if not stdout:
        return None

    text = stdout.strip()
    if not text:
        return None

    # Fast path: stdout starts with '{' — clean JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Slow path: find first '{' and try parsing from there
    brace_idx = text.find("{")
    if brace_idx > 0:
        try:
            parsed = json.loads(text[brace_idx:])
            logger.debug("JSON extracted via brace-offset fallback (offset=%d, len=%d)", brace_idx, len(text))
            return parsed
        except json.JSONDecodeError:
            pass

    # Last resort: scan lines in reverse for a JSON line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                logger.debug("JSON extracted via reverse-line-scan fallback (len=%d)", len(text))
                return parsed
            except json.JSONDecodeError:
                continue

    logger.debug("JSON extraction failed — all 3 strategies exhausted (len=%d)", len(text) if text else 0)
    return None


def _kronos_circuit_open() -> bool:
    with _forecast_lock:
        return time.monotonic() < _kronos_tripped_until


def _trip_kronos():
    global _kronos_tripped_until
    with _forecast_lock:
        _kronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
    logger.warning("Kronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)


def _chronos_circuit_open() -> bool:
    with _forecast_lock:
        return time.monotonic() < _chronos_tripped_until


def _trip_chronos():
    global _chronos_tripped_until
    with _forecast_lock:
        _chronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
    logger.warning("Chronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)


def reset_circuit_breakers():
    """Reset both circuit breakers (for testing or manual recovery)."""
    global _kronos_tripped_until, _chronos_tripped_until
    with _forecast_lock:
        _kronos_tripped_until = 0.0
        _chronos_tripped_until = 0.0


def _log_health(model: str, ticker: str, success: bool, duration_ms: int, error: str = ""):
    """Append a line to forecast_health.jsonl for persistent success/failure tracking.

    On success, auto-resets the relevant circuit breaker so recovered models
    resume immediately instead of waiting for the full TTL (BUG-56 fix).
    """
    global _kronos_tripped_until, _chronos_tripped_until
    try:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "model": model,
            "ticker": ticker,
            "ok": success,
            "ms": duration_ms,
        }
        if error:
            entry["error"] = error[:200]
        atomic_append_jsonl(_HEALTH_FILE, entry)
    except Exception as e:
        logger.debug("Forecast health logging failed: %s", e)

    # Auto-reset circuit breaker on success — faster recovery from transient failures
    # BUG-102: Use lock to make read-check-write atomic
    if success:
        with _forecast_lock:
            if model == "kronos" and _kronos_tripped_until > 0:
                _kronos_tripped_until = 0.0
                logger.info("Kronos circuit breaker RESET on successful %s", ticker)
            elif model == "chronos" and _chronos_tripped_until > 0:
                _chronos_tripped_until = 0.0
                logger.info("Chronos circuit breaker RESET on successful %s", ticker)


def _load_candles_ohlcv(ticker: str, periods: int = 168,
                        interval: str = "1h") -> list[dict] | None:
    """Load recent OHLCV candles as list of dicts.

    Args:
        ticker: Instrument ticker (e.g., "BTC-USD")
        periods: Number of candles to fetch
        interval: Candle interval ("1h", "5m", "15m", etc.)
    """
    from portfolio.tickers import SYMBOLS

    source_info = SYMBOLS.get(ticker, {})

    # Determine the data source — needed to apply source-specific interval constraints
    if "binance" in source_info:
        source = "binance"
    elif "binance_fapi" in source_info:
        source = "binance_fapi"
    elif "alpaca" in source_info:
        source = "alpaca"
    else:
        source = None

    # Alpaca minimum supported interval is 15m — fall back if configured interval is smaller.
    # alpaca_klines() does its own mapping; pass the raw internal interval directly.
    if source == "alpaca" and interval in ("1m", "3m", "5m"):
        logger.debug(
            "Alpaca does not support %s interval for %s — falling back to 15m", interval, ticker
        )
        interval = "15m"

    try:
        if source == "binance":
            from portfolio.data_collector import binance_klines
            symbol = source_info["binance"]
            df = binance_klines(symbol, interval=interval, limit=periods)
        elif source == "binance_fapi":
            from portfolio.data_collector import binance_fapi_klines
            symbol = source_info["binance_fapi"]
            df = binance_fapi_klines(symbol, interval=interval, limit=periods)
        elif source == "alpaca":
            # Pass the raw internal interval — alpaca_klines() handles the mapping itself.
            from portfolio.data_collector import alpaca_klines
            symbol = source_info["alpaca"]
            df = alpaca_klines(symbol, interval=interval, limit=periods)
        else:
            return None

        if df is not None and len(df) > 30:
            candles = []
            for _, row in df.iterrows():
                candles.append({
                    "open": float(row.get("open", row.get("close", 0))),
                    "high": float(row.get("high", row.get("close", 0))),
                    "low": float(row.get("low", row.get("close", 0))),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })
            return candles
    except Exception as e:
        logger.debug("OHLCV fetch failed for %s (interval=%s): %s", ticker, interval, e)

    return None


def _run_kronos(candles: list[dict], horizons: tuple = (1, 24), _ticker: str = "") -> dict | None:
    """Run Kronos inference via subprocess with GPU gating."""
    if not _KRONOS_ENABLED:
        return None
    if _kronos_circuit_open():
        return None
    t0 = time.time()
    try:
        with gpu_gate("kronos", timeout=90) as acquired:
            if not acquired:
                logger.warning("GPU gate timeout for Kronos %s", _ticker)
                return None
            return _run_kronos_inner(candles, horizons, _ticker, t0)
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        logger.warning("Kronos GPU gate error: %s", e)
        _log_health("kronos", _ticker, False, ms, str(e)[:200])
        _trip_kronos()
        return None


def _run_kronos_inner(candles, horizons, _ticker, t0):
    """Kronos inference (called inside GPU gate)."""
    try:
        # Read tunable params from config
        try:
            from portfolio.file_utils import load_json
            cfg = load_json(str(Path(__file__).resolve().parent.parent.parent / "config.json"), {})
            fc = cfg.get("forecast", {})
        except Exception:
            logger.debug("Forecast config load failed, using defaults", exc_info=True)
            fc = {}

        input_data = json.dumps({
            "candles": candles,
            "prices_close": [c["close"] for c in candles],
            "temperature": fc.get("kronos_temperature", 1.0),
            "top_p": fc.get("kronos_top_p", 0.9),
            "sample_count": fc.get("kronos_samples", 3),
        })
        proc = subprocess.run(
            [_KRONOS_PYTHON, _KRONOS_SCRIPT,
             "--horizons", ",".join(str(h) for h in horizons)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=_KRONOS_TIMEOUT,
        )
        ms = round((time.time() - t0) * 1000)
        if proc.returncode != 0:
            err = proc.stderr[:200]
            logger.warning("Kronos subprocess failed: %s", err)
            _log_health("kronos", _ticker, False, ms, err)
            _trip_kronos()
            return None
        if not proc.stdout or not proc.stdout.strip():
            _log_health("kronos", _ticker, False, ms, "empty_stdout")
            _trip_kronos()
            return None
        result = _extract_json_from_stdout(proc.stdout)
        if result is None:
            # JSON extraction failed — log actual stdout for diagnostics
            preview = repr(proc.stdout[:200])
            logger.warning("Kronos stdout not valid JSON for %s: %s", _ticker, preview)
            _log_health("kronos", _ticker, False, ms, f"json_extract_failed: {preview[:150]}")
            _trip_kronos()
            return None
        if not result or not result.get("results"):
            _log_health("kronos", _ticker, False, ms, "empty_results")
            _trip_kronos()
            return None
        _log_health("kronos", _ticker, True, ms)
        return result
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        logger.warning("Kronos subprocess error (v2): %s", e)
        _log_health("kronos", _ticker, False, ms, str(e)[:200])
        _trip_kronos()
        return None


def _run_chronos(prices: list[float], horizons: tuple = (1, 24), _ticker: str = "",
                 timeout: int | None = None) -> dict | None:
    """Run Chronos forecast (in-process, lazy-loaded) with GPU gating and timeout."""
    if _chronos_circuit_open():
        return None

    with gpu_gate("chronos", timeout=120) as acquired:
        if not acquired:
            logger.warning("GPU gate timeout for Chronos %s", _ticker)
            return None
        return _run_chronos_inner(prices, horizons, _ticker, timeout)


def _run_chronos_inner(prices, horizons, _ticker, timeout):
    """Chronos inference (called inside GPU gate)."""
    t0 = time.time()
    _timeout = timeout or _CHRONOS_TIMEOUT
    try:
        from portfolio.forecast_signal import forecast_chronos

        # Run in thread with timeout to prevent hangs
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(forecast_chronos, "", prices, horizons=horizons)
            try:
                result = future.result(timeout=_timeout)
            except FuturesTimeout:
                ms = round((time.time() - t0) * 1000)
                logger.warning("Chronos timed out after %ds for %s", _timeout, _ticker)
                _log_health("chronos", _ticker, False, ms, f"timeout_{_timeout}s")
                _trip_chronos()
                return None

        ms = round((time.time() - t0) * 1000)
        if result is None:
            _log_health("chronos", _ticker, False, ms, "returned_none")
            _trip_chronos()
        else:
            _log_health("chronos", _ticker, True, ms)
        return result
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        logger.warning("Chronos failed: %s", e)
        _log_health("chronos", _ticker, False, ms, str(e)[:200])
        _trip_chronos()
        return None


def _health_weighted_vote(sub_signals, kronos_ok, chronos_ok):
    """Vote only using sub-signals from healthy (working) models.

    When Kronos is dead (high failure rate), its 2 permanent HOLD votes
    dilute the 4-vote majority and make the signal always return HOLD.
    This function excludes dead models from the vote.

    **Shadow exclusion (2026-04-21)**: when `_KRONOS_SHADOW` is True, Kronos's
    sub-signal is EXCLUDED from the vote pool regardless of `kronos_ok`. The
    shadow mode is for logging + accuracy backfill only — adding HOLD weights
    to the composite would tie against Chronos's verdicts (observed 3668
    times in the previous shadow-mode run, all forced to HOLD). The raw
    Kronos prediction is still captured in `raw_sub_signals["kronos_*"]` for
    downstream accuracy tracking.

    1h horizon gets 2x weight (counted twice) because short-term predictions
    are more actionable and Chronos 24h predictions are less reliable.
    """
    alive_votes = []
    # Shadow mode: Kronos runs, logs its raw prediction, but does NOT
    # contribute to the composite vote. Only vote when the model is both
    # healthy AND live (not shadow).
    if kronos_ok and not _KRONOS_SHADOW:
        # 1h gets double weight
        alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("kronos_24h", "HOLD"))
    if chronos_ok:
        # 1h gets double weight
        alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("chronos_24h", "HOLD"))

    if not alive_votes:
        return "HOLD", 0.0

    return majority_vote(alive_votes)


# Per-ticker accuracy cache TTL
_ACCURACY_CACHE_TTL = 1800  # 30 minutes

# Default thresholds for accuracy gating
_HOLD_THRESHOLD = 0.55        # Below this: force HOLD (signal can't predict)
_MIN_SAMPLES = 10             # Below this: use raw vote (insufficient data)

# Volatility gate — force HOLD when ATR% exceeds threshold
# Chronos predicts negligible moves (~0.1% avg), so high-volatility environments
# where actual moves are 3-5% make the signal useless.
_VOL_GATE_CRYPTO = 0.03       # 3% ATR for crypto
_VOL_GATE_DEFAULT = 0.02      # 2% ATR for metals/stocks

# Regime-aware confidence discount — Chronos has a mean-reversion bias
# (predicts small moves back to mean). In trending markets this is wrong.
_REGIME_DISCOUNT_TRENDING = 0.5   # Halve confidence in trending regimes
_REGIME_DISCOUNT_HIGH_VOL = 0.6   # Reduce confidence in high-vol regimes
_REGIME_NEUTRAL = 1.0             # No discount in ranging/neutral regimes


def _compute_atr_pct(close_prices: list[float], period: int = 14) -> float | None:
    """Compute ATR% from close prices (approximation using close-to-close).

    Returns ATR as fraction of current price (e.g. 0.03 = 3%), or None
    if insufficient data.
    """
    if not close_prices or len(close_prices) < period + 1:
        return None
    # Approximate true range from close-to-close changes
    trs = [abs(close_prices[i] - close_prices[i - 1]) for i in range(1, len(close_prices))]
    if len(trs) < period:
        return None
    # EMA-smoothed ATR over last `period` values
    recent_trs = trs[-period * 2:]  # use more data for EMA warmup
    atr = recent_trs[0]
    alpha = 2.0 / (period + 1)
    for tr in recent_trs[1:]:
        atr = alpha * tr + (1 - alpha) * atr
    current = close_prices[-1]
    if current <= 0:
        return None
    return atr / current


def _is_crypto_ticker(ticker: str) -> bool:
    """Check if ticker is crypto (BTC-USD, ETH-USD)."""
    try:
        from portfolio.tickers import CRYPTO_SYMBOLS
        return ticker in CRYPTO_SYMBOLS
    except ImportError:
        return ticker in {"BTC-USD", "ETH-USD"}


def _load_forecast_accuracy(cache_ttl=None):
    """Load per-ticker forecast accuracy, cached via _cached().

    Returns dict: {ticker: {accuracy, samples}} or empty dict on error.
    """
    ttl = cache_ttl or _ACCURACY_CACHE_TTL

    def _fetch():
        try:
            from portfolio.forecast_accuracy import get_all_ticker_accuracies
            return get_all_ticker_accuracies(horizon="24h", days=14)
        except Exception as e:
            logger.debug("Failed to load forecast accuracy: %s", e)
            return {}

    return _cached("forecast_ticker_accuracy", ttl, _fetch)


def _load_forecast_subsignal_accuracy(cache_ttl=None, days=30):
    """Load raw sub-signal accuracy for 1h and 24h forecast votes."""
    ttl = cache_ttl or _ACCURACY_CACHE_TTL
    cache_key = f"forecast_subsignal_accuracy_{days}"

    def _fetch():
        try:
            from portfolio.forecast_accuracy import compute_forecast_accuracy

            return {
                "1h": compute_forecast_accuracy(
                    horizon="1h", days=days, use_raw_sub_signals=True
                ),
                "24h": compute_forecast_accuracy(
                    horizon="24h", days=days, use_raw_sub_signals=True
                ),
            }
        except Exception as e:
            logger.debug("Failed to load forecast sub-signal accuracy: %s", e)
            return {}

    return _cached(cache_key, ttl, _fetch)


def _gate_subsignal_votes_by_accuracy(sub_signals, ticker, config_forecast=None):
    """Gate individual forecast sub-signals using raw historical accuracy."""
    cfg = config_forecast or {}
    hold_threshold = cfg.get("subsignal_hold_threshold", cfg.get("hold_threshold", _HOLD_THRESHOLD))
    min_samples = cfg.get("subsignal_min_samples", cfg.get("min_samples", _MIN_SAMPLES))
    lookback_days = cfg.get("subsignal_accuracy_days", 30)
    cache_ttl = cfg.get("subsignal_accuracy_cache_ttl", _ACCURACY_CACHE_TTL)

    gated = dict(sub_signals)
    info = {}
    if not ticker:
        return gated, info

    accuracy_matrix = _load_forecast_subsignal_accuracy(cache_ttl=cache_ttl, days=lookback_days)
    for sub_name, vote in sub_signals.items():
        if vote == "HOLD":
            continue

        horizon = "1h" if sub_name.endswith("_1h") else "24h"
        horizon_stats = ((accuracy_matrix or {}).get(horizon) or {}).get(sub_name) or {}
        ticker_stats = (horizon_stats.get("by_ticker") or {}).get(ticker)

        accuracy = None
        samples = 0
        source = None
        if ticker_stats and ticker_stats.get("total", 0) >= min_samples:
            accuracy = float(ticker_stats["accuracy"])
            samples = int(ticker_stats["total"])
            source = "ticker"
        elif horizon_stats.get("total", 0) >= min_samples:
            accuracy = float(horizon_stats["accuracy"])
            samples = int(horizon_stats["total"])
            source = "global"

        gating = "insufficient_data"
        if accuracy is not None:
            gating = "held" if accuracy < hold_threshold else "raw"
            if gating == "held":
                gated[sub_name] = "HOLD"

        info[sub_name] = {
            "gating": gating,
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "samples": samples,
            "source": source,
        }

    return gated, info


def _regime_discount(regime: str, config_forecast: dict | None = None) -> float:
    """Return confidence multiplier based on market regime.

    Chronos has a mean-reversion bias — it predicts small moves back to mean.
    In trending markets, this is wrong, so we discount confidence.
    """
    cfg = config_forecast or {}
    if not regime:
        return _REGIME_NEUTRAL
    r = regime.lower()
    if r in ("trending-up", "trending-down", "breakout"):
        return cfg.get("regime_discount_trending", _REGIME_DISCOUNT_TRENDING)
    elif r in ("high-vol", "capitulation"):
        return cfg.get("regime_discount_high_vol", _REGIME_DISCOUNT_HIGH_VOL)
    else:
        # range-bound, neutral — mean-reversion is appropriate
        return _REGIME_NEUTRAL


def _accuracy_weighted_vote(sub_signals, kronos_ok, chronos_ok, ticker="",
                            config_forecast=None, atr_pct=None, regime=None):
    """Vote with per-ticker accuracy gating, volatility gate, and regime discount.

    Extends _health_weighted_vote with:
    - Volatility gate: high ATR% → force HOLD (Chronos can't predict big moves)
    - Regime discount: trending markets → reduce confidence (mean-reversion bias)
    - Accuracy gate: accuracy < hold_threshold → force HOLD
    - Good accuracy: use raw vote, scale confidence by accuracy
    - Insufficient samples: use raw vote (not enough data to judge)

    Bad tickers abstain (HOLD) rather than invert — inversion games the
    accuracy metric without fixing the underlying prediction quality.

    Returns (action, confidence, gating_info) where gating_info is a dict with
    accuracy metadata for logging.
    """
    cfg = config_forecast or {}
    hold_thresh = cfg.get("hold_threshold", _HOLD_THRESHOLD)
    min_samples = cfg.get("min_samples", _MIN_SAMPLES)

    # Start with health-weighted vote as baseline
    base_action, base_conf = _health_weighted_vote(
        sub_signals, kronos_ok, chronos_ok
    )

    gating_info = {
        "forecast_accuracy": None,
        "forecast_samples": 0,
        "forecast_gating": "raw",
        "forecast_inverted": False,
        "base_action": base_action,
        "base_confidence": base_conf,
        "atr_pct": atr_pct,
    }

    if not ticker:
        return base_action, base_conf, gating_info

    # Volatility gate — Chronos predicts negligible moves (~0.1% avg),
    # so high-vol environments make the signal useless
    if atr_pct is not None:
        vol_thresh = cfg.get("vol_gate_crypto", _VOL_GATE_CRYPTO) \
            if _is_crypto_ticker(ticker) \
            else cfg.get("vol_gate_default", _VOL_GATE_DEFAULT)
        if atr_pct > vol_thresh:
            gating_info["forecast_gating"] = "vol_gated"
            return "HOLD", 0.0, gating_info

    # Load per-ticker accuracy
    all_acc = _load_forecast_accuracy(cfg.get("accuracy_cache_ttl"))
    ticker_acc = all_acc.get(ticker) if all_acc else None

    if ticker_acc is None or ticker_acc.get("samples", 0) < min_samples:
        gating_info["forecast_gating"] = "insufficient_data"
        if ticker_acc:
            gating_info["forecast_accuracy"] = ticker_acc["accuracy"]
            gating_info["forecast_samples"] = ticker_acc["samples"]
        return base_action, base_conf, gating_info

    acc = ticker_acc["accuracy"]
    samples = ticker_acc["samples"]
    gating_info["forecast_accuracy"] = acc
    gating_info["forecast_samples"] = samples

    if acc < hold_thresh:
        # Below threshold — signal can't predict this ticker, abstain
        gating_info["forecast_gating"] = "held"
        return "HOLD", 0.0, gating_info

    else:
        # Good accuracy — use raw vote, scale confidence by accuracy
        gating_info["forecast_gating"] = "raw"
        # Apply regime discount (trending → lower confidence for mean-reversion bias)
        r_discount = _regime_discount(regime, cfg)
        gating_info["regime_discount"] = r_discount
        scaled_conf = base_conf * acc * r_discount
        return base_action, min(scaled_conf, _MAX_CONFIDENCE), gating_info


def _direction_to_action(direction: str) -> str:
    """Convert direction string to action."""
    if direction in ("up", "BUY"):
        return "BUY"
    if direction in ("down", "SELL"):
        return "SELL"
    return "HOLD"


def compute_forecast_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the composite forecast signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (used as fallback if candle fetch fails).
    context : dict, optional
        Dict with keys: ticker, config, macro.

    Returns
    -------
    dict
        action, confidence, sub_signals, indicators
    """
    result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "kronos_1h": "HOLD",
            "kronos_24h": "HOLD",
            "chronos_1h": "HOLD",
            "chronos_24h": "HOLD",
        },
        "indicators": {},
    }

    # Bulletproof early return — skip ALL work when models are disabled
    if _FORECAST_MODELS_DISABLED:
        result["indicators"]["models_disabled"] = True
        return result

    ticker = (context or {}).get("ticker", "")
    if not ticker:
        return result

    # Apply Chronos model config if specified
    chronos_model = (context or {}).get("config", {}).get("forecast", {}).get("chronos_model")
    if chronos_model:
        try:
            from portfolio.forecast_signal import set_chronos_model
            set_chronos_model(chronos_model)
        except Exception as e:
            logger.debug("Chronos model config override failed: %s", e)

    config_forecast = (context or {}).get("config", {}).get("forecast", {})

    # Load candles (1h for Chronos, optionally 5m for Kronos)
    cache_key = f"forecast_candles_{ticker}"
    candles = _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv, ticker)

    # Load 5m candles for Kronos if configured (more granular context)
    kronos_interval = config_forecast.get("kronos_interval", "1h")
    if kronos_interval != "1h" and _KRONOS_ENABLED:
        kronos_periods = config_forecast.get("kronos_periods", 500)
        kronos_cache_key = f"forecast_candles_{ticker}_{kronos_interval}"
        kronos_candles = _cached(kronos_cache_key, _FORECAST_TTL,
                                  _load_candles_ohlcv, ticker, kronos_periods,
                                  kronos_interval)
    else:
        kronos_candles = None

    if not candles or len(candles) < 50:
        # Fallback to df close prices if available
        if df is not None and len(df) >= 50 and "close" in df.columns:
            close_prices = df["close"].values.tolist()
        else:
            result["indicators"]["error"] = "insufficient_candle_data"
            return result
    else:
        close_prices = [c["close"] for c in candles]

    # If Kronos-specific candle fetch failed but df has full OHLCV data, build candle dicts
    # from the DataFrame so Kronos still gets richer data than just close prices.
    if kronos_candles is None and df is not None and len(df) >= 50:
        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        if ohlcv_cols.issubset(df.columns):
            try:
                kronos_candles = [
                    {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                    for _, row in df.iterrows()
                ]
                logger.debug(
                    "Kronos candle fallback from df for %s (%d candles)", ticker, len(kronos_candles)
                )
                result["indicators"]["kronos_candles_source"] = "df_fallback"
            except Exception as e:
                logger.debug("Kronos df candle fallback failed for %s: %s", ticker, e)
                kronos_candles = None

    current_price = close_prices[-1]
    result["indicators"]["current_price"] = current_price
    result["indicators"]["candle_count"] = len(close_prices)
    result["indicators"]["kronos_circuit_open"] = _kronos_circuit_open()
    result["indicators"]["chronos_circuit_open"] = _chronos_circuit_open()

    # 2026-05-03 ordering: Chronos BEFORE Kronos. Chronos is in-process and
    # fast (~50ms warm, ~1.7s cold model load). Kronos is a subprocess that
    # holds the GPU file-lock for the duration of model load + inference,
    # which on cold-start was ~210s — long enough that 4 concurrent ticker
    # threads all timed out their Chronos gate (120s) waiting behind one
    # Kronos. Running Chronos first lets all 4 tickers pipeline through the
    # GPU in seconds; Kronos (shadow-only by current config) then runs and
    # whichever threads can't grab the gate within 90s skip silently — fine
    # because the live consensus uses Chronos, not Kronos.
    #
    # Run Chronos (skip entirely if circuit breaker is open)
    t0 = time.time()
    chronos_key = f"chronos_forecast_{ticker}"
    chronos = _cached(chronos_key, _FORECAST_TTL, _run_chronos, close_prices, (1, 24), ticker)
    chronos_ms = round((time.time() - t0) * 1000)
    result["indicators"]["chronos_time_ms"] = chronos_ms

    if chronos:
        if "1h" in chronos:
            result["sub_signals"]["chronos_1h"] = chronos["1h"].get("action", "HOLD")
            result["indicators"]["chronos_1h_pct"] = chronos["1h"].get("pct_move", 0)
            result["indicators"]["chronos_1h_conf"] = chronos["1h"].get("confidence", 0)

        if "24h" in chronos:
            result["sub_signals"]["chronos_24h"] = chronos["24h"].get("action", "HOLD")
            result["indicators"]["chronos_24h_pct"] = chronos["24h"].get("pct_move", 0)
            result["indicators"]["chronos_24h_conf"] = chronos["24h"].get("confidence", 0)

    # Run Kronos — use 5m candles if available, otherwise 1h
    t0 = time.time()
    kronos_key = f"kronos_forecast_{ticker}"
    kronos_input = kronos_candles if kronos_candles and len(kronos_candles) >= 50 else (candles or [])
    kronos = _cached(kronos_key, _FORECAST_TTL, _run_kronos, kronos_input, (1, 24), ticker)
    if kronos_candles and len(kronos_candles) >= 50:
        result["indicators"]["kronos_interval"] = kronos_interval
    kronos_ms = round((time.time() - t0) * 1000)
    result["indicators"]["kronos_time_ms"] = kronos_ms

    if kronos and kronos.get("results"):
        kr = kronos["results"]
        result["indicators"]["kronos_method"] = kronos.get("method", "unknown")
        result["indicators"]["kronos_shadow"] = _KRONOS_SHADOW

        if "1h" in kr:
            k1h_action = _direction_to_action(kr["1h"].get("direction", "neutral"))
            # 2026-04-21 shadow fix: always record the REAL prediction in
            # sub_signals. The vote-pool filter in _health_weighted_vote
            # excludes the sub-signal when _KRONOS_SHADOW is True, so the
            # composite is never polluted. This change means the
            # forecast_predictions.jsonl log captures Kronos's true verdict
            # (not a forced HOLD), which unblocks meaningful accuracy
            # backfill. Pre-fix: `"HOLD" if _KRONOS_SHADOW else k1h_action`
            # — left zero statistical signal in 3668 logged predictions.
            result["sub_signals"]["kronos_1h"] = k1h_action
            result["indicators"]["kronos_1h_raw"] = k1h_action
            result["indicators"]["kronos_1h_pct"] = kr["1h"].get("pct_move", 0)
            result["indicators"]["kronos_1h_conf"] = kr["1h"].get("confidence", 0)
            result["indicators"]["kronos_1h_range_pct"] = kr["1h"].get("predicted_range_pct", 0)
            result["indicators"]["kronos_1h_range_skew"] = kr["1h"].get("range_skew", 0)

        if "24h" in kr:
            k24h_action = _direction_to_action(kr["24h"].get("direction", "neutral"))
            result["sub_signals"]["kronos_24h"] = k24h_action
            result["indicators"]["kronos_24h_raw"] = k24h_action
            result["indicators"]["kronos_24h_pct"] = kr["24h"].get("pct_move", 0)
            result["indicators"]["kronos_24h_conf"] = kr["24h"].get("confidence", 0)
            result["indicators"]["kronos_24h_range_pct"] = kr["24h"].get("predicted_range_pct", 0)
            result["indicators"]["kronos_24h_range_skew"] = kr["24h"].get("range_skew", 0)
            result["indicators"]["kronos_24h_predicted_high"] = kr["24h"].get("predicted_high", 0)
            result["indicators"]["kronos_24h_predicted_low"] = kr["24h"].get("predicted_low", 0)

    raw_sub_signals = dict(result["sub_signals"])
    gated_sub_signals, subsignal_gating = _gate_subsignal_votes_by_accuracy(
        raw_sub_signals, ticker, config_forecast=config_forecast
    )
    result["sub_signals"] = gated_sub_signals
    result["indicators"]["forecast_subsignal_gating"] = subsignal_gating

    # Accuracy-weighted vote — per-ticker accuracy gating + health exclusion
    kronos_ok = kronos is not None and bool(kronos.get("results"))
    chronos_ok = chronos is not None
    result["indicators"]["kronos_ok"] = kronos_ok
    result["indicators"]["chronos_ok"] = chronos_ok

    # Compute ATR% for volatility gate
    atr_pct = _compute_atr_pct(close_prices)
    result["indicators"]["forecast_atr_pct"] = round(atr_pct, 4) if atr_pct else None

    regime = (context or {}).get("regime", "")
    result["action"], result["confidence"], gating_info = _accuracy_weighted_vote(
        result["sub_signals"], kronos_ok, chronos_ok,
        ticker=ticker, config_forecast=config_forecast,
        atr_pct=atr_pct, regime=regime,
    )

    # Store gating metadata in indicators
    result["indicators"]["forecast_accuracy"] = gating_info.get("forecast_accuracy")
    result["indicators"]["forecast_samples"] = gating_info.get("forecast_samples", 0)
    result["indicators"]["forecast_gating"] = gating_info.get("forecast_gating", "raw")
    result["indicators"]["forecast_inverted"] = gating_info.get("forecast_inverted", False)

    # Cap confidence (already capped inside _accuracy_weighted_vote, but belt-and-suspenders)
    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)

    # Log prediction for accuracy tracking (with dedup)
    # BUG-102: Lock protects _last_prediction_ts from concurrent ThreadPoolExecutor access
    # BUG-106: Evict stale entries to prevent unbounded dict growth
    try:
        now_mono = time.monotonic()
        with _forecast_lock:
            last_ts = _last_prediction_ts.get(ticker, 0.0)
            should_log = now_mono - last_ts >= _PREDICTION_DEDUP_TTL
        if should_log:
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "ticker": ticker,
                "current_price": current_price,
                "sub_signals": result["sub_signals"],
                "raw_sub_signals": raw_sub_signals,
                "subsignal_gating": subsignal_gating,
                "action": result["action"],
                "confidence": result["confidence"],
                "per_ticker_accuracy": gating_info.get("forecast_accuracy"),
                "gating_action": gating_info.get("forecast_gating", "raw"),
            }
            if kronos and kronos.get("results"):
                entry["kronos"] = kronos["results"]
            if chronos:
                entry["chronos"] = chronos
            atomic_append_jsonl(_PREDICTIONS_FILE, entry)
            with _forecast_lock:
                _last_prediction_ts[ticker] = now_mono
                # BUG-106: Evict stale entries older than 10 minutes
                stale = [k for k, v in _last_prediction_ts.items()
                         if now_mono - v > _PREDICTION_DEDUP_EVICT_AGE]
                for k in stale:
                    del _last_prediction_ts[k]
    except Exception:
        logger.debug("Failed to log forecast prediction", exc_info=True)

    return result

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\tests\\test_forecast_timeout.py" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 612ms:
"""Tests for Chronos timeout, prediction dedup, and accuracy metadata.

Covers:
- Chronos timeout behavior (mock slow Chronos)
- Kronos reduced timeout constant
- Prediction deduplication (same ticker within TTL not re-logged)
- Accuracy metadata in logged predictions
"""

import json
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio.signals.forecast import (
    _CHRONOS_TIMEOUT,
    _KRONOS_TIMEOUT,
    _PREDICTION_DEDUP_TTL,
    _chronos_circuit_open,
    _run_chronos,
    compute_forecast_signal,
    reset_circuit_breakers,
)


@pytest.fixture(autouse=True)
def _reset(tmp_path):
    import portfolio.signals.forecast as mod
    orig_kronos = mod._KRONOS_ENABLED
    orig_disabled = mod._FORECAST_MODELS_DISABLED
    orig_pred_file = mod._PREDICTIONS_FILE
    mod._KRONOS_ENABLED = False
    mod._FORECAST_MODELS_DISABLED = False
    mod._last_prediction_ts.clear()
    # Isolate predictions file per test (avoids cross-worker conflicts in parallel)
    mod._PREDICTIONS_FILE = tmp_path / "forecast_predictions.jsonl"
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()
    mod._KRONOS_ENABLED = orig_kronos
    mod._FORECAST_MODELS_DISABLED = orig_disabled
    mod._PREDICTIONS_FILE = orig_pred_file
    mod._last_prediction_ts.clear()


# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------

class TestTimeoutConstants:
    def test_chronos_timeout_default(self):
        assert _CHRONOS_TIMEOUT == 60

    def test_kronos_timeout_reduced(self):
        """Kronos timeout should be lower than the old 120s."""
        assert _KRONOS_TIMEOUT == 30
        assert _KRONOS_TIMEOUT < 120


# ---------------------------------------------------------------------------
# Chronos timeout behavior
# ---------------------------------------------------------------------------

class TestChronosTimeout:
    def test_timeout_returns_none(self):
        """When Chronos takes too long, should return None and trip breaker."""
        def slow_chronos(*args, **kwargs):
            time.sleep(5)
            return {"1h": {"action": "BUY", "pct_move": 0.3}}

        mock_mod = MagicMock()
        mock_mod.forecast_chronos = slow_chronos
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            result = _run_chronos([100.0] * 50, timeout=1)
        assert result is None
        assert _chronos_circuit_open()

    def test_fast_chronos_succeeds(self):
        """When Chronos completes within timeout, should return result."""
        fast_result = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "SELL", "pct_move": -0.2, "confidence": 0.4},
        }
        mock_mod = MagicMock()
        mock_mod.forecast_chronos.return_value = fast_result
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            result = _run_chronos([100.0] * 50, timeout=10)
        assert result is not None
        assert result == fast_result
        assert not _chronos_circuit_open()

    def test_timeout_parameter_used(self):
        """Custom timeout parameter should be used."""
        def medium_chronos(*args, **kwargs):
            time.sleep(0.5)
            return {"1h": {"action": "BUY", "pct_move": 0.3}}

        mock_mod = MagicMock()
        mock_mod.forecast_chronos = medium_chronos
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            # 0.2s timeout should trigger timeout
            result = _run_chronos([100.0] * 50, timeout=0.2)
        assert result is None


# ---------------------------------------------------------------------------
# Prediction deduplication
# ---------------------------------------------------------------------------

class TestPredictionDedup:
    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_dedup_prevents_double_logging(self, mock_candles, mock_kronos,
                                            mock_chronos, mock_cached):
        """Second call within dedup TTL should not append to predictions file."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        # Clear the file
        with open(pred_file, "w") as f:
            pass

        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value={}):
            df = pd.DataFrame({"close": [100.0] * 60})

            # First call
            compute_forecast_signal(df, context={"ticker": "BTC-USD"})
            lines_after_first = pred_file.read_text(encoding="utf-8").strip().split("\n")
            # Filter out empty lines
            lines_after_first = [l for l in lines_after_first if l.strip()]

            # Second call immediately — should be deduped
            compute_forecast_signal(df, context={"ticker": "BTC-USD"})
            lines_after_second = pred_file.read_text(encoding="utf-8").strip().split("\n")
            lines_after_second = [l for l in lines_after_second if l.strip()]

        assert len(lines_after_first) == 1
        assert len(lines_after_second) == 1  # No duplicate

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_different_tickers_not_deduped(self, mock_candles, mock_kronos,
                                            mock_chronos, mock_cached):
        """Different tickers should not be deduped against each other."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        with open(pred_file, "w") as f:
            pass

        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value={}):
            df = pd.DataFrame({"close": [100.0] * 60})
            compute_forecast_signal(df, context={"ticker": "BTC-USD"})
            compute_forecast_signal(df, context={"ticker": "ETH-USD"})

        lines = [l for l in pred_file.read_text(encoding="utf-8").strip().split("\n")
                 if l.strip()]
        assert len(lines) == 2  # Both logged

    def test_dedup_ttl_constant(self):
        """Dedup TTL should be 60 seconds."""
        assert _PREDICTION_DEDUP_TTL == 60


# ---------------------------------------------------------------------------
# Accuracy metadata in predictions
# ---------------------------------------------------------------------------

class TestAccuracyMetadata:
    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_predictions_include_accuracy_metadata(self, mock_candles, mock_kronos,
                                                     mock_chronos, mock_cached):
        """Logged predictions should include per_ticker_accuracy and gating_action."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        with open(pred_file, "w") as f:
            pass

        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 25}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy",
                    return_value=acc_data):
            df = pd.DataFrame({"close": [100.0] * 60})
            compute_forecast_signal(df, context={"ticker": "AMZN"})

        lines = [l for l in pred_file.read_text(encoding="utf-8").strip().split("\n")
                 if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["per_ticker_accuracy"] == 0.90
        assert entry["gating_action"] == "raw"
        assert entry["ticker"] == "AMZN"

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_predictions_accuracy_null_when_no_data(self, mock_candles, mock_kronos,
                                                      mock_chronos, mock_cached):
        """When no accuracy data, per_ticker_accuracy should be None."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        with open(pred_file, "w") as f:
            pass

        with patch("portfolio.signals.forecast._load_forecast_accuracy",
                    return_value={}):
            df = pd.DataFrame({"close": [100.0] * 60})
            compute_forecast_signal(df, context={"ticker": "NEW-TICKER"})

        lines = [l for l in pred_file.read_text(encoding="utf-8").strip().split("\n")
                 if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["per_ticker_accuracy"] is None
        assert entry["gating_action"] == "insufficient_data"

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from unittest.mock import patch, MagicMock
import pandas as pd
import portfolio.signals.forecast as mod

# Setup like fixture
orig_kronos = mod._KRONOS_ENABLED
orig_disabled = mod._FORECAST_MODELS_DISABLED
mod._KRONOS_ENABLED = True
mod._FORECAST_MODELS_DISABLED = False
mod.reset_circuit_breakers()
mod._last_prediction_ts.clear()


def bypass_cache(key, ttl, fn, *args):
    return fn(*args)

with patch('portfolio.signals.forecast._gate_subsignal_votes_by_accuracy', side_effect=lambda raw, ticker, config_forecast=None: (dict(raw), {})), \\
     patch('portfolio.signals.forecast._cached') as mock_cached, \\
     patch('portfolio.signals.forecast._run_chronos') as mock_chronos, \\
     patch('portfolio.signals.forecast._run_kronos') as mock_kronos, \\
     patch('portfolio.signals.forecast._load_candles_ohlcv') as mock_candles:
    mock_cached.side_effect = bypass_cache
    mock_candles.return_value = [{'close': float(100+i)} for i in range(80)]
    mock_kronos.return_value = {
        'method':'kronos',
        'results':{
            '1h': {'direction':'up','pct_move':0.3,'confidence':0.6},
            '24h': {'direction':'down','pct_move':-0.5,'confidence':0.5},
        },
    }
    mock_chronos.return_value = {
        '1h': {'action':'BUY','pct_move':0.4,'confidence':0.55},
        '24h': {'action':'SELL','pct_move':-0.6,'confidence':0.5},
    }
    df = pd.DataFrame({'close':[100.0]*80})
    res = mod.compute_forecast_signal(df, context={'ticker':'BTC-USD'})
    print('res action', res['action'])
    print('candles calls', mock_candles.call_args_list)
    print('cached calls', mock_cached.call_args_list)
    print('sub', res['sub_signals'])
    print('ind', {k:res['indicators'].get(k) for k in ['forecast_gating','kronos_ok','chronos_ok','kronos_circuit_open','chronos_circuit_open','error']})

mod._KRONOS_ENABLED = orig_kronos
mod._FORECAST_MODELS_DISABLED = orig_disabled
mod.reset_circuit_breakers()
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 1310ms:
res action BUY
candles calls [call('BTC-USD')]
cached calls [call('forecast_candles_BTC-USD', 300, <MagicMock name='_load_candles_ohlcv' id='1524167875120'>, 'BTC-USD'),
 call('chronos_forecast_BTC-USD', 300, <MagicMock name='_run_chronos' id='1524167720160'>, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0], (1, 24), 'BTC-USD'),
 call('kronos_forecast_BTC-USD', 300, <MagicMock name='_run_kronos' id='1524167871424'>, [{'close': 100.0}, {'close': 101.0}, {'close': 102.0}, {'close': 103.0}, {'close': 104.0}, {'close': 105.0}, {'close': 106.0}, {'close': 107.0}, {'close': 108.0}, {'close': 109.0}, {'close': 110.0}, {'close': 111.0}, {'close': 112.0}, {'close': 113.0}, {'close': 114.0}, {'close': 115.0}, {'close': 116.0}, {'close': 117.0}, {'close': 118.0}, {'close': 119.0}, {'close': 120.0}, {'close': 121.0}, {'close': 122.0}, {'close': 123.0}, {'close': 124.0}, {'close': 125.0}, {'close': 126.0}, {'close': 127.0}, {'close': 128.0}, {'close': 129.0}, {'close': 130.0}, {'close': 131.0}, {'close': 132.0}, {'close': 133.0}, {'close': 134.0}, {'close': 135.0}, {'close': 136.0}, {'close': 137.0}, {'close': 138.0}, {'close': 139.0}, {'close': 140.0}, {'close': 141.0}, {'close': 142.0}, {'close': 143.0}, {'close': 144.0}, {'close': 145.0}, {'close': 146.0}, {'close': 147.0}, {'close': 148.0}, {'close': 149.0}, {'close': 150.0}, {'close': 151.0}, {'close': 152.0}, {'close': 153.0}, {'close': 154.0}, {'close': 155.0}, {'close': 156.0}, {'close': 157.0}, {'close': 158.0}, {'close': 159.0}, {'close': 160.0}, {'close': 161.0}, {'close': 162.0}, {'close': 163.0}, {'close': 164.0}, {'close': 165.0}, {'close': 166.0}, {'close': 167.0}, {'close': 168.0}, {'close': 169.0}, {'close': 170.0}, {'close': 171.0}, {'close': 172.0}, {'close': 173.0}, {'close': 174.0}, {'close': 175.0}, {'close': 176.0}, {'close': 177.0}, {'close': 178.0}, {'close': 179.0}], (1, 24), 'BTC-USD'),
 call('forecast_ticker_accuracy', 1800, <function _load_forecast_accuracy.<locals>._fetch at 0x00000162DF7AB7E0>)]
sub {'kronos_1h': 'BUY', 'kronos_24h': 'SELL', 'chronos_1h': 'BUY', 'chronos_24h': 'SELL'}
ind {'forecast_gating': 'insufficient_data', 'kronos_ok': True, 'chronos_ok': True, 'kronos_circuit_open': False, 'chronos_circuit_open': False, 'error': None}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\tests\\*.py -Pattern 'realized_skewness'" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 694ms:

Q:\finance-analyzer\tests\test_signal_engine.py:1142:    def test_mstr_[7mrealized_skewness[0m_disabled(self):
Q:\finance-analyzer\tests\test_signal_engine.py:1143:        # [7mrealized_skewness[0m 36.0% 1d on MSTR (50 sam).
Q:\finance-analyzer\tests\test_signal_engine.py:1145:        assert "[7mrealized_skewness[0m" [0m[7m[0min [0m
[7m[0m_TICKER_DISABLED_SIGNALS["MSTR"][0m
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:1:"""Tests for [7mrealized_skewness[0m signal module."""
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:5:from portfolio.signals.[7mrealized_skewness [0m[7m[0mimport [0m
[7m[0mcompute_realized_skewness_signal[0m
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:54:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:63:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:69:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:75:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:81:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:87:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:93:        result = compute_[7mrealized_skewness[0m_signal(df, [0m
[7m[0mcontext=ctx)[0m
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:98:        result = compute_[7mrealized_skewness[0m_signal(None)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:108:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:115:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:125:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:135:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:140:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:145:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:155:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:162:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:180:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:186:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:193:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:210:        result = compute_[7mrealized_skewness[0m_signal(df)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:217:        result = compute_[7mrealized_skewness[0m_signal(df)


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
mods=['yfinance','scipy']
for m in mods:
    try:
        __import__(m)
        print(m,'installed')
    except Exception as e:
        print(m,'missing',e)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 1212ms:
yfinance installed
scipy installed

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p = Path(r'Q:/finance-analyzer/.worktrees/adv-signals-modules/portfolio/signals/forecast.py')
for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
    if '_KRONOS_PYTHON' in line or '_KRONOS_SCRIPT' in line:
        print(i, line)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 575ms:
107     _KRONOS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
108     _KRONOS_SCRIPT = r"Q:\models\kronos_infer.py"
110     _KRONOS_PYTHON = "/home/deck/models/.venv/bin/python"
111     _KRONOS_SCRIPT = "/home/deck/models/kronos_infer.py"
354             [_KRONOS_PYTHON, _KRONOS_SCRIPT,

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio/signals/*.py | Select-String -Pattern 'Q:\\\\|/home/deck|C:/' | Format-Table Path,LineNumber,Line -AutoSize" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 608ms:

[32;1mPath                                                                            [0m[32;1m LineNumber[0m[32;1m Line[0m
[32;1m----                                                                            [0m [32;1m----------[0m [32;1m----[0m
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py        107     _KRONOS_PYTHON = r"Q:\…
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py        108     _KRONOS_SCRIPT = r"Q:\…
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py        110     _KRONOS_PYTHON = "/hom…
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py        111     _KRONOS_SCRIPT = "/hom…


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse Q:\\finance-analyzer\\portfolio -Filter *.py | Select-String -Pattern 'import yfinance as yf|from scipy import stats|import scipy' | Select-Object Path,LineNumber,Line | Format-Table -AutoSize" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 614ms:

[32;1mPath                                                            [0m[32;1m LineNumber[0m[32;1m Line[0m
[32;1m----                                                            [0m [32;1m----------[0m [32;1m----[0m
Q:\finance-analyzer\portfolio\crypto_precompute.py                      203         import yfinance as yf
Q:\finance-analyzer\portfolio\data_collector.py                         171         import yfinance as yf
Q:\finance-analyzer\portfolio\data_collector.py                         213     import yfinance as yf
Q:\finance-analyzer\portfolio\earnings_calendar.py                      100         import yfinance as yf
Q:\finance-analyzer\portfolio\fear_greed.py                             128     import yfinance as yf
Q:\finance-analyzer\portfolio\mstr_precompute.py                        142         import yfinance as yf
Q:\finance-analyzer\portfolio\outcome_tracker.py                        262         import yfinance as yf
Q:\finance-analyzer\portfolio\price_source.py                           132     import yfinance as yf
Q:\finance-analyzer\portfolio\sentiment.py                              132     import yfinance as yf
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py             172             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py           68             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py               82             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py               62             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py          75             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\network_momentum.py                78             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py            61         import yfinance as yf
Q:\finance-analyzer\portfolio\signals\realized_skewness.py               25 from scipy import stats
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py         79             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py          52             import yfinance as yf
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py              46         import yfinance as yf
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py         57             import yfinance as yf


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/signals/realized_skewness.py -TotalCount 120' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 539ms:
"""Realized skewness directional signal module.

Computes 4 sub-indicators based on the 3rd moment of daily returns and returns
a majority-vote composite BUY/SELL/HOLD signal with confidence score.

Sub-indicators:
    1. Skewness Z-Score      — z-scored realized skewness vs rolling baseline
    2. Skewness Momentum     — 5-bar delta of skewness (acceleration)
    3. Kurtosis Confirmation — high kurtosis + negative skew = stronger signal
    4. Rolling Skew Regime   — short-window skew vs long-window skew divergence

Academic evidence:
    - Fernandez-Perez et al (2018): Sharpe 0.79, 8.01% annual on 27 commodity
      futures. Low-skew assets outperform high-skew assets.
    - ScienceDirect (2024): negative cross-sectional relationship between
      skewness risk and future crypto returns.

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 60 rows of data (for z-score normalization).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from portfolio.signal_utils import majority_vote, safe_float

# Minimum rows required.  We need at least ``SKEW_LOOKBACK`` returns for a
# reliable skewness estimate, but we fall back to ``MIN_ROWS`` when the full
# lookback is not yet available.
MIN_ROWS = 60
SKEW_LOOKBACK = 252       # ~1 year of daily data
NORM_WINDOW = 60          # z-score normalisation window
Z_BUY = -1.5             # negative skew = mean-reversion opportunity
Z_SELL = 1.5              # positive skew = momentum exhaustion
SKEW_MOM_PERIOD = 5       # bars for skewness momentum (acceleration)
SHORT_SKEW_WINDOW = 20    # short-window skew for regime divergence


def _compute_rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    """Compute rolling skewness using scipy.stats.skew (Fisher definition)."""
    return returns.rolling(window=window, min_periods=max(window // 2, 20)).apply(
        lambda x: stats.skew(x, nan_policy="omit") if len(x.dropna()) >= 20 else np.nan,
        raw=False,
    )


def _sub_skew_zscore(returns: pd.Series) -> tuple[float, str, dict]:
    """Sub-signal 1: z-scored realized skewness.

    Negative skew (fat left tail) historically precedes mean-reversion rallies.
    Positive skew (fat right tail) signals momentum exhaustion.
    """
    lookback = min(SKEW_LOOKBACK, len(returns))
    skew_val = stats.skew(returns.iloc[-lookback:].dropna())
    if np.isnan(skew_val):
        return 0.0, "HOLD", {"raw_skewness": np.nan, "skew_z": np.nan}

    # Compute rolling skewness for z-scoring
    rolling_skew = _compute_rolling_skewness(returns, lookback)
    recent = rolling_skew.iloc[-NORM_WINDOW:]
    mean_skew = recent.mean()
    std_skew = recent.std()

    if std_skew < 1e-8 or np.isnan(std_skew):
        return 0.0, "HOLD", {"raw_skewness": safe_float(skew_val), "skew_z": 0.0}

    z = (skew_val - mean_skew) / std_skew

    indicators = {"raw_skewness": safe_float(skew_val), "skew_z": safe_float(z)}

    if z < Z_BUY:
        confidence = min(abs(z) / 3.0, 1.0)
        return confidence, "BUY", indicators
    elif z > Z_SELL:
        confidence = min(abs(z) / 3.0, 1.0)
        return confidence, "SELL", indicators
    return 0.0, "HOLD", indicators


def _sub_skew_momentum(returns: pd.Series) -> tuple[float, str, dict]:
    """Sub-signal 2: skewness momentum (acceleration).

    If skewness is falling rapidly (becoming more negative), the asset is
    developing a fat left tail — contrarian BUY.  Rising skewness toward
    positive = SELL.
    """
    lookback = min(SKEW_LOOKBACK, len(returns))
    rolling_skew = _compute_rolling_skewness(returns, lookback)
    valid = rolling_skew.dropna()

    if len(valid) < SKEW_MOM_PERIOD + 1:
        return 0.0, "HOLD", {"skew_momentum": np.nan}

    current = valid.iloc[-1]
    past = valid.iloc[-(SKEW_MOM_PERIOD + 1)]
    delta = current - past

    indicators = {"skew_momentum": safe_float(delta)}

    # Falling skew (becoming more negative) = BUY
    if delta < -0.3:
        return min(abs(delta) / 1.0, 0.8), "BUY", indicators
    # Rising skew (becoming more positive) = SELL
    elif delta > 0.3:
        return min(abs(delta) / 1.0, 0.8), "SELL", indicators
    return 0.0, "HOLD", indicators


def _sub_kurtosis_confirm(returns: pd.Series) -> tuple[float, str, dict]:
    """Sub-signal 3: kurtosis confirmation.

    High kurtosis (fat tails) + negative skew = strong mean-reversion BUY.
    High kurtosis + positive skew = strong momentum exhaustion SELL.
    Low kurtosis = no extreme tail structure = HOLD.
    """
    lookback = min(SKEW_LOOKBACK, len(returns))
    window = returns.iloc[-lookback:].dropna()
    if len(window) < 20:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\portfolio\\*.py,Q:\\finance-analyzer\\portfolio\\**\\*.py,Q:\\finance-analyzer\\tests\\*.py -Pattern 'compute_crypto_cross_asset_signal' | Format-Table Path,LineNumber,Line -AutoSize" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 650ms:

[32;1mPath                                                       [0m[32;1m LineNumber[0m[32;1m Line[0m
[32;1m----                                                       [0m [32;1m----------[0m [32;1m----[0m
Q:\finance-analyzer\portfolio\signals\crypto_cross_asset.py        168 def compute_crypto_cross_asset_signal(df: pd.Da…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py          7 from portfolio.signals.crypto_cross_asset impor…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         17         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         25         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         33         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         42         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         50         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         63         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         72         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         82         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py         90         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py        100         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py        112         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py        122         out = compute_crypto_cross_asset_signal…
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py        137         out = compute_crypto_cross_asset_signal…


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, sys, inspect, traceback
from pathlib import Path
path = r'Q:/finance-analyzer/tests/test_signal_realized_skewness.py'
name='trs'
spec=importlib.util.spec_from_file_location(name,path)
mod=importlib.util.module_from_spec(spec)
sys.modules[name]=mod
spec.loader.exec_module(mod)
passed=total=0
for cname,obj in vars(mod).items():
    if inspect.isclass(obj) and cname.startswith('Test'):
        inst=obj()
        for mname,meth in inspect.getmembers(inst, predicate=callable):
            if mname.startswith('test_') and meth.__code__.co_argcount==0:
                total += 1
                try:
                    meth()
                    passed += 1
                    print('PASS',cname,mname)
                except Exception as e:
                    print('FAIL',cname,mname,type(e).__name__,e)
                    traceback.print_exc(limit=2)
print('summary',passed,total)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 1763ms:
summary 0 0

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\tests\\test_signal_realized_skewness.py -TotalCount 260" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 550ms:
"""Tests for realized_skewness signal module."""
import numpy as np
import pandas as pd

from portfolio.signals.realized_skewness import compute_realized_skewness_signal


def _make_df(n=200):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_skewed_df(n=300, skew_direction="negative"):
    """Create a DataFrame with intentionally skewed returns."""
    np.random.seed(123)
    if skew_direction == "negative":
        # Fat left tail: mostly small positive returns with occasional large drops
        returns = np.where(
            np.random.rand(n) > 0.9,
            -np.random.exponential(0.03, n),  # large negative
            np.random.exponential(0.005, n),   # small positive
        )
    else:
        # Fat right tail: mostly small negative returns with occasional large jumps
        returns = np.where(
            np.random.rand(n) > 0.9,
            np.random.exponential(0.03, n),   # large positive
            -np.random.exponential(0.005, n),  # small negative
        )

    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.003)),
        "low": close * (1 - abs(np.random.randn(n) * 0.003)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_realized_skewness_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_realized_skewness_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_realized_skewness_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_none_dataframe_returns_hold(self):
        result = compute_realized_skewness_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestSubSignals:
    """Test individual sub-signal behavior."""

    def test_sub_signals_have_expected_keys(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        expected_keys = {"skew_zscore", "skew_momentum", "kurtosis_confirm",
                         "skew_regime_divergence"}
        assert expected_keys == set(result["sub_signals"].keys())

    def test_all_sub_signals_are_valid_actions(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        for key, value in result["sub_signals"].items():
            assert value in ("BUY", "SELL", "HOLD"), f"{key} has invalid value {value}"


class TestIndicators:
    """Test indicator values are present and reasonable."""

    def test_raw_skewness_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "raw_skewness" in result["indicators"]
        skew = result["indicators"]["raw_skewness"]
        # Skewness should be a finite number
        assert np.isfinite(skew)
        # For random data, skewness should be roughly between -3 and 3
        assert -5.0 < skew < 5.0

    def test_skew_z_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "skew_z" in result["indicators"]

    def test_kurtosis_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "kurtosis" in result["indicators"]

    def test_skew_divergence_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "skew_divergence" in result["indicators"]


class TestDirectionality:
    """Test that the signal produces expected directions for skewed data."""

    def test_negative_skew_tends_toward_buy(self):
        """Negative skew (fat left tail) should lean toward BUY."""
        df = _make_skewed_df(n=300, skew_direction="negative")
        result = compute_realized_skewness_signal(df)
        # At minimum, the raw skewness should be negative
        assert result["indicators"]["raw_skewness"] < 0

    def test_positive_skew_tends_toward_sell(self):
        """Positive skew (fat right tail) should lean toward SELL."""
        df = _make_skewed_df(n=300, skew_direction="positive")
        result = compute_realized_skewness_signal(df)
        # At minimum, the raw skewness should be positive
        assert result["indicators"]["raw_skewness"] > 0


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_constant_price_returns_hold(self):
        """Constant price = zero returns = zero skewness = HOLD."""
        n = 100
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        result = compute_realized_skewness_signal(df)
        assert result["action"] == "HOLD"

    def test_minimum_viable_rows(self):
        """Test with exactly MIN_ROWS (60) rows."""
        df = _make_df(n=60)
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert isinstance(result["confidence"], float)

    def test_large_dataset(self):
        """Test with a large dataset (1000 rows)."""
        df = _make_df(n=1000)
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_high_volatility_data(self):
        """Test with extremely volatile data."""
        np.random.seed(99)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 5.0)  # 10x normal vol
        close = np.maximum(close, 1.0)  # prevent negative prices
        df = pd.DataFrame({
            "open": close,
            "high": close * 1.05,
            "low": close * 0.95,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        })
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_all_volume_zero(self):
        """Volume=0 should not crash."""
        df = _make_df()
        df["volume"] = 0.0
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, sys, inspect, traceback
from pathlib import Path
path = r'Q:/finance-analyzer/tests/test_signal_realized_skewness.py'
name='trs2'
spec=importlib.util.spec_from_file_location(name,path)
mod=importlib.util.module_from_spec(spec)
sys.modules[name]=mod
spec.loader.exec_module(mod)
passed=total=0
for cname,obj in vars(mod).items():
    if inspect.isclass(obj) and cname.startswith('Test'):
        inst=obj()
        for mname,meth in inspect.getmembers(inst, predicate=callable):
            if mname.startswith('test_'):
                total += 1
                try:
                    meth()
                    passed += 1
                    print('PASS',cname,mname)
                except Exception as e:
                    print('FAIL',cname,mname,type(e).__name__,e)
                    traceback.print_exc(limit=2)
print('summary',passed,total)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 4099ms:
PASS TestSignalInterface test_empty_dataframe_returns_hold
PASS TestSignalInterface test_has_indicators
PASS TestSignalInterface test_has_sub_signals
PASS TestSignalInterface test_insufficient_rows_returns_hold
PASS TestSignalInterface test_nan_handling
PASS TestSignalInterface test_none_dataframe_returns_hold
PASS TestSignalInterface test_returns_dict_with_required_keys
PASS TestSignalInterface test_with_context
PASS TestSubSignals test_all_sub_signals_are_valid_actions
PASS TestSubSignals test_sub_signals_have_expected_keys
PASS TestIndicators test_kurtosis_in_indicators
PASS TestIndicators test_raw_skewness_in_indicators
PASS TestIndicators test_skew_divergence_in_indicators
PASS TestIndicators test_skew_z_in_indicators
PASS TestDirectionality test_negative_skew_tends_toward_buy
PASS TestDirectionality test_positive_skew_tends_toward_sell
PASS TestEdgeCases test_all_volume_zero
PASS TestEdgeCases test_constant_price_returns_hold
PASS TestEdgeCases test_high_volatility_data
PASS TestEdgeCases test_large_dataset
PASS TestEdgeCases test_minimum_viable_rows
summary 21 21

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\tests\\*.py -Pattern '_get_earnings_calendar|Earnings Date|claude_fundamental' | Format-Table Path,LineNumber,Line -AutoSize" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 648ms:

[32;1mPath                                                        [0m[32;1m LineNumber[0m[32;1m Line[0m
[32;1m----                                                        [0m [32;1m----------[0m [32;1m----[0m
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py          215         """Direct pin: with MSTR's _default bl…
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py          226         assert "claude_fundamental" in _TICKER…
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py          239         cf_action = extra.get("claude_fundamen…
Q:\finance-analyzer\tests\test_alpha_vantage.py                      18 from portfolio.signals.claude_fundamental impo…
Q:\finance-analyzer\tests\test_bug_fixes_session4.py                  3 BUG-30: dashboard signal heatmap missing 3 sig…
Q:\finance-analyzer\tests\test_bug_fixes_session4.py                 31     def test_enhanced_signals_includes_claude_…
Q:\finance-analyzer\tests\test_bug_fixes_session4.py                 32         """claude_fundamental signal (#29) mus…
Q:\finance-analyzer\tests\test_bug_fixes_session4.py                 35         assert '"claude_fundamental"' in sourc…
Q:\finance-analyzer\tests\test_dashboard_system_status.py           220             {"scope": "signal", "key": "claude…
Q:\finance-analyzer\tests\test_dashboard_system_status.py           491                     "claude_fundamental": {"to…
Q:\finance-analyzer\tests\test_dashboard_system_status.py           498         cf = next(m for m in models if m["key"…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py         35         """Batch 1 trim: MSTR default has clau…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py         37         assert mstr == frozenset({"claude_fund…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py         45         assert result == frozenset({"claude_fu…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py         51             "claude_fundamental", "credit_spre…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py         59             "claude_fundamental", "credit_spre…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        122         """Default (MSTR claude_fundamental) a…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        124             "claude_fundamental": "BUY",
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        134             # claude_fundamental should be HOL…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        161             "claude_fundamental": "BUY",
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        168         # claude_fundamental NOT gated (no tic…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        203         expected_default = {"claude_fundamenta…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        221         expected_default = {"claude_fundamenta…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        255     def test_mstr_claude_fundamental_is_hold_a…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        259         gating, then assert claude_fundamental…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        302             # claude_fundamental is in _defaul…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        303             cf = snapshot.get("claude_fundamen…
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py        305                 f"At horizon={horizon}, claude…
Q:\finance-analyzer\tests\test_instrument_profile.py                 44         assert "claude_fundamental" in trusted
Q:\finance-analyzer\tests\test_layer2_journal_contract.py           225     claude_fundamental, bigbet, iskbets, goldd…
Q:\finance-analyzer\tests\test_llm_outcome_backfill.py              194     on claude_fundamental BTC-USD before this …
Q:\finance-analyzer\tests\test_llm_outcome_backfill.py              206         "signal": "claude_fundamental",
Q:\finance-analyzer\tests\test_llm_outcome_backfill.py              390         "signal": "claude_fundamental",
Q:\finance-analyzer\tests\test_llm_probability_log.py               116         "claude_fundamental", "MSTR", _valid_p…
Q:\finance-analyzer\tests\test_llm_probability_log.py               132     assert mod.is_llm_signal("claude_fundament…
Q:\finance-analyzer\tests\test_loop_contract_grace.py                57         # false-negatived by unrelated claude_…
Q:\finance-analyzer\tests\test_macro_window_gating.py               179         """When macro window is inactive, clau…
Q:\finance-analyzer\tests\test_macro_window_gating.py               181         votes = {"claude_fundamental": "BUY", …
Q:\finance-analyzer\tests\test_macro_window_gating.py               183             "claude_fundamental": {"accuracy":…
Q:\finance-analyzer\tests\test_macro_window_gating.py               191     def test_macro_active_force_holds_claude_f…
Q:\finance-analyzer\tests\test_macro_window_gating.py               192         """During a macro window, claude_funda…
Q:\finance-analyzer\tests\test_macro_window_gating.py               195         # claude_fundamental BUY would normall…
Q:\finance-analyzer\tests\test_macro_window_gating.py               197         votes = {"claude_fundamental": "BUY", …
Q:\finance-analyzer\tests\test_macro_window_gating.py               199             "claude_fundamental": {"accuracy":…
Q:\finance-analyzer\tests\test_outcome_tracker_core.py              438             "econ_calendar", "forecast", "clau…
Q:\finance-analyzer\tests\test_signal_engine_core.py                244         # Use claude_fundamental as opponent —…
Q:\finance-analyzer\tests\test_signal_engine_core.py                245         votes = {"rsi": "BUY", "claude_fundame…
Q:\finance-analyzer\tests\test_signal_engine_core.py                248             "claude_fundamental": {"accuracy":…
Q:\finance-analyzer\tests\test_signal_engine_core.py                251         # rsi weight = 0.6 * 1.5(ranging) = 0.…
Q:\finance-analyzer\tests\test_signal_engine_core.py                296         # Use claude_fundamental — not in mome…
Q:\finance-analyzer\tests\test_signal_engine_core.py                297         votes = {"rsi": "BUY", "claude_fundame…
Q:\finance-analyzer\tests\test_signal_engine_core.py                300             "claude_fundamental": {"accuracy":…
Q:\finance-analyzer\tests\test_signal_engine_core.py                304             "claude_fundamental": {"normalized…
Q:\finance-analyzer\tests\test_signal_engine_core.py                308         # claude_fundamental: 0.6 * 1.0(rangin…
Q:\finance-analyzer\tests\test_signal_engine_core.py               1697                     "momentum_factors", "claud…
Q:\finance-analyzer\tests\test_signal_engine_core.py               1731     def test_trending_down_gates_bb_and_claude…
Q:\finance-analyzer\tests\test_signal_engine_core.py               1735         assert "claude_fundamental" in gated
Q:\finance-analyzer\tests\test_signal_engine_core.py               1745     def test_trending_down_3h_gates_bb_and_cla…
Q:\finance-analyzer\tests\test_signal_engine_core.py               1746         """At 3h, bb and claude_fundamental re…
Q:\finance-analyzer\tests\test_signal_engine_core.py               1750         assert "claude_fundamental" in gated
Q:\finance-analyzer\tests\test_signal_engine.py                     380     def test_claude_fundamental_sell_gated(sel…
Q:\finance-analyzer\tests\test_signal_engine.py                     381         """claude_fundamental: BUY=65.7% fine,…
Q:\finance-analyzer\tests\test_signal_engine.py                     384         votes = {"claude_fundamental": "SELL",…
Q:\finance-analyzer\tests\test_signal_engine.py                     386             "claude_fundamental": {"accuracy":…
Q:\finance-analyzer\tests\test_signal_engine.py                     393         # claude_fundamental SELL gated, only …
Q:\finance-analyzer\tests\test_signal_engine.py                     909     """2026-04-27: claude_fundamental 40.5% at…
Q:\finance-analyzer\tests\test_signal_engine.py                     912     def test_claude_fundamental_gated_at_defau…
Q:\finance-analyzer\tests\test_signal_engine.py                     915         assert "claude_fundamental" in gated_d…
Q:\finance-analyzer\tests\test_signal_engine.py                     917     def test_claude_fundamental_per_ticker_gat…
Q:\finance-analyzer\tests\test_signal_engine.py                     918         """Metals have no earnings/guidance — …
Q:\finance-analyzer\tests\test_signal_engine.py                     920         assert "claude_fundamental" in _TICKER…
Q:\finance-analyzer\tests\test_signal_engine.py                     922     def test_claude_fundamental_per_ticker_gat…
Q:\finance-analyzer\tests\test_signal_engine.py                     923         """Metals have no earnings/guidance — …
Q:\finance-analyzer\tests\test_signal_engine.py                     925         assert "claude_fundamental" in _TICKER…
Q:\finance-analyzer\tests\test_signal_engine.py                    1084     MSTR's +8.4% W16 rally. Kept: claude_funda…
Q:\finance-analyzer\tests\test_signal_engine.py                    1116     def test_mstr_claude_fundamental_still_dis…
Q:\finance-analyzer\tests\test_signal_engine.py                    1119         assert "claude_fundamental" in _TICKER…
Q:\finance-analyzer\tests\test_signal_engine.py                    1167                  "claude_fundamental": "BUY"}
Q:\finance-analyzer\tests\test_signal_engine.py                    1172             "claude_fundamental": {"accuracy":…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py          1 """Tests for portfolio.signals.claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py         11 from portfolio.signals.claude_fundamental impo…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py         22     compute_claude_fundamental_signal,
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        285     post-merge, effectively deactivating claud…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        292         from portfolio.signals import claude_f…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        296         log_path = tmp_path / "claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        341         from portfolio.signals import claude_f…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        343         log_path = tmp_path / "claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        386         from portfolio.signals import claude_f…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        388         log_path = tmp_path / "claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        403         from portfolio.signals import claude_f…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        405         log_path = tmp_path / "claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        422         from portfolio.signals import claude_f…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        424         log_path = tmp_path / "claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        440         from portfolio.signals import claude_f…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        442         log_path = tmp_path / "claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        460         from portfolio.signals import claude_f…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        462         log_path = tmp_path / "claude_fundamen…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        490         result = compute_claude_fundamental_si…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        496         result = compute_claude_fundamental_si…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        500                 "config": {"claude_fundamental…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        506     @mock.patch("portfolio.signals.claude_fund…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        522         result = compute_claude_fundamental_si…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        524             context={"ticker": "BTC-USD", "con…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        530     @mock.patch("portfolio.signals.claude_fund…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        545         ctx = {"ticker": "BTC-USD", "config": …
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        547         result1 = compute_claude_fundamental_s…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        548         result2 = compute_claude_fundamental_s…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        559         ctx = {"ticker": "BTC-USD", "config": …
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        560         compute_claude_fundamental_signal(df, …
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        566     @mock.patch("portfolio.signals.claude_fund…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        570         ctx = {"ticker": "BTC-USD", "config": …
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        571         result = compute_claude_fundamental_si…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        589         ctx = {"ticker": "BTC-USD", "config": …
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        590         result = compute_claude_fundamental_si…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        619         ctx = {"ticker": "NVDA", "config": {"c…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        620         result = compute_claude_fundamental_si…
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        640             caller="claude_fundamental_haiku",
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py        666             caller="claude_fundamental_sonnet",
Q:\finance-analyzer\tests\test_validate_signal_result.py            233                   "claude_fundamental", "futur…


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\tests\\test_signals_claude_fundamental.py -Pattern 'earnings|calendar' -Context 2,2" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 575ms:

  Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:166:                },
  Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:167:                "conviction": 0.8,
> Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:168:                "reasoning": "HBM supply pre-sold, 
[7mearnings [0m[7m[0mtrajectory [0m[7m[0mstrong",[0m
[7m[0m  Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:169:                "contrarian_flag": True,[0m
[7m[0m  Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:170:            }[0m


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import yfinance, inspect
import yfinance.scrapers.quote as q
import re
src = inspect.getsource(q.Quote.calendar)
print(src)
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
2026-05-08T15:39:27.546438Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 1.3 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 1285, in getsource
    lines, lnum = getsourcelines(object)
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 1267, in getsourcelines
    lines, lnum = findsource(object)
                  ^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 1078, in findsource
    file = getsourcefile(object)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 955, in getsourcefile
    filename = getfile(object)
               ^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 935, in getfile
    raise TypeError('module, class, method, function, traceback, frame, or '
TypeError: module, class, method, function, traceback, frame, or code object was expected, got property

 exited 1 in 1297ms:
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 1285, in getsource
    lines, lnum = getsourcelines(object)
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 1267, in getsourcelines
    lines, lnum = findsource(object)
                  ^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 1078, in findsource
    file = getsourcefile(object)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 955, in getsourcefile
    filename = getfile(object)
               ^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\inspect.py", line 935, in getfile
    raise TypeError('module, class, method, function, traceback, frame, or '
TypeError: module, class, method, function, traceback, frame, or code object was expected, got property

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import inspect
from yfinance.scrapers.quote import Quote
print(inspect.getsource(Quote.calendar.fget))
'@ | Q:\\finance-analyzer\\.venv\\Scripts\\python.exe -" in Q:\finance-analyzer
 succeeded in 1164ms:
    @property
    def calendar(self) -> dict:
        if self._calendar is None:
            self._fetch_calendar()
        return self._calendar


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path Q:\\finance-analyzer\\portfolio\\signal_engine.py,Q:\\finance-analyzer\\portfolio\\data_collector.py,Q:\\finance-analyzer\\portfolio\\main.py,Q:\\finance-analyzer\\portfolio\\analyze.py -Pattern 'compute_forecast_signal|timeframe|horizon' -Context 1,2" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 637ms:

  Q:\finance-analyzer\portfolio\signal_engine.py:376:#
> Q:\finance-analyzer\portfolio\signal_engine.py:377:# 2026-04-16 (Batch 4): [7mhorizon[0m-specific per-ticker blacklists via[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:378:# _TICKER_DISABLED_BY_[7mHORIZON[0m. Structure:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:379:#   {"3h": {ticker: frozenset(bad_signals_at_3h)},[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:380:#    "1d": {ticker: frozenset(bad_signals_at_1d)},[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:381:#    "_default": {ticker: frozenset(bad_at_ALL_[7mhorizon[0ms)}}[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:382:#[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:383:# [0m[7m[0mCompute-time [0m[7m[0m(signal [0m[7m[0mdispatch [0m[7m[0mloop): [0m[7m[0muses [0m[7m[0mthe [0m[7m[0m_default [0m[7m[0mlist [0m
[7m[0monly. [0m[7m[0mSignals[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:384:# compute once per ticker per cycle and their vote is reused 
across [7mhorizon[0ms,[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:385:# so disabling at compute time requires the signal to be bad at 
EVERY [7mhorizon[0m.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:386:# [7mHorizon[0m-specific [0m[7m[0mentries [0m[7m[0mdo [0m[7m[0mNOT [0m[7m[0mskip [0m[7m[0mcompute [0m[7m[0m(the [0m[7m[0mvote [0m[7m[0mstill [0m
[7m[0mexists, [0m[7m[0mbut[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:387:# is force-HOLD'd per-[7mhorizon[0m at consensus time).[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:388:#[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:389:# Consensus-time: when building consensus for [7mhorizon[0m H, apply
> Q:\finance-analyzer\portfolio\signal_engine.py:390:# (_default[ticker] | _TICKER_DISABLED_BY_[7mHORIZON[0m[H][ticker]).[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:391:#[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:392:# [0m[7m[0mWhy [0m[7m[0mthis [0m[7m[0mstructure: [0m[7m[0mthe [0m[7m[0mApr [0m[7m[0m14 [0m[7m[0mMSTR [0m[7m[0mblacklist [0m[7m[0mwas [0m[7m[0mbuilt [0m[7m[0mfrom [0m[7m[0m3h [0m
[7m[0maccuracy[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:394:# to 21.9%). 5 of 7 blacklisted MSTR signals were 66-81% accurate 
at 1d.
> Q:\finance-analyzer\portfolio\signal_engine.py:395:# Batch 1 trimmed the list to 2 entries; Batch 4 (this) enables 
per-[7mhorizon[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:396:# [0m[7m[0mentries [0m[7m[0mso [0m[7m[0mfuture [0m[7m[0maudits [0m[7m[0mcan [0m[7m[0msay [0m[7m[0m"bad [0m[7m[0mat [0m[7m[0m3h, [0m[7m[0mfine [0m[7m[0mat [0m[7m[0m1d" [0m
[7m[0mwithout [0m[7m[0mglobal[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:397:# penalty.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:398:_TICKER_DISABLED_BY_[7mHORIZON[0m: dict[str, dict[str, frozenset]] = {
> Q:\finance-analyzer\portfolio\signal_engine.py:399:    # Disabled at ALL [7mhorizon[0ms [0m[7m[0m— [0m[7m[0mbad [0m[7m[0meverywhere, [0m[7m[0msafe [0m[7m[0mto [0m[7m[0mskip [0m
[7m[0meven [0m[7m[0mat [0m[7m[0mcompute.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:400:    "_default": {[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:401: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m2026-04-15 [0m[7m[0maudit: [0m[7m[0mper-ticker [0m[7m[0m3h [0m[7m[0maccuracy [0m[7m[0mgating, [0m
[7m[0mretained [0m[7m[0mpending[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:402:        # per-[7mhorizon[0m audit of 1d/3d/5d behaviors.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:403: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m2026-04-24 [0m[7m[0mafter-hours [0m[7m[0maudit: [0m[7m[0madded [0m[7m[0mstructure [0m[7m[0m(metals), [0m
[7m[0mcredit_spread_risk[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:404: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mand [0m[7m[0mmacro_regime [0m[7m[0m(XAU), [0m[7m[0mema [0m[7m[0m(crypto/metals), [0m
[7m[0mfutures_flow [0m[7m[0m(crypto).[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:471:        # These signals are already auto-gated by the blended 
accuracy gate,
> Q:\finance-analyzer\portfolio\signal_engine.py:472:        # but per-[7mhorizon[0m blacklists provide defense-in-depth.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:473:        "BTC-USD": frozenset({"news_event", "forecast",[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:474: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m"econ_calendar", [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m1.8% [0m[7m[0m1d [0m
[7m[0m(113 [0m[7m[0msam)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:477:                              "calendar",            # 30.8% 
1d_recent (712 sam), 100% BUY-only
> Q:\finance-analyzer\portfolio\signal_engine.py:478:                              "momentum_factors",    # 32.7% 
1d_recent (910 sam), 60.1% at 3h — [7mhorizon [0m[7m[0mdivergence[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:479: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m"volume_flow", [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m35.8% [0m
[7m[0m1d_recent [0m[7m[0m(924 [0m[7m[0msam)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:480: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m"heikin_ashi", [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m38.2% [0m
[7m[0m1d_recent [0m[7m[0m(709 [0m[7m[0msam)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:524:
> Q:\finance-analyzer\portfolio\signal_engine.py:525:# P2-H (2026-04-17): module-load validation of 
_TICKER_DISABLED_BY_[7mHORIZON[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:526:# shape. Catches structural errors (missing _default, invalid 
[7mhorizon [0m[7m[0mkeys,[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:527:# [0m[7m[0mnon-frozenset [0m[7m[0mvalues) [0m[7m[0mat [0m[7m[0mimport [0m[7m[0mtime [0m[7m[0mrather [0m[7m[0mthan [0m[7m[0msilently [0m[7m[0mat [0m
[7m[0mruntime.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:528:_VALID_[7mHORIZON[0m_KEYS [0m[7m[0m= [0m[7m[0mfrozenset({"_default", [0m[7m[0m"3h", [0m[7m[0m"4h", [0m[7m[0m"12h", [0m
[7m[0m"1d", [0m[7m[0m"3d", [0m[7m[0m"5d", [0m[7m[0m"10d"})[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:529:assert "_default" in _TICKER_DISABLED_BY_[7mHORIZON[0m, (
> Q:\finance-analyzer\portfolio\signal_engine.py:530:    "_TICKER_DISABLED_BY_[7mHORIZON[0m missing required '_default' key")
> Q:\finance-analyzer\portfolio\signal_engine.py:531:for _k, _inner in _TICKER_DISABLED_BY_[7mHORIZON[0m.items():
> Q:\finance-analyzer\portfolio\signal_engine.py:532:    assert _k in _VALID_[7mHORIZON[0m_KEYS, (
> Q:\finance-analyzer\portfolio\signal_engine.py:533:        f"_TICKER_DISABLED_BY_[7mHORIZON [0m[7m[0mhas [0m[7m[0munknown [0m[7m[0mhorizon [0m[7m[0mkey [0m
[7m[0m{_k!r}; [0m[7m[0m"[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:534:        f"valid keys: {sorted(_VALID_[7mHORIZON[0m_KEYS)}")[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:535:    assert isinstance(_inner, dict), ([0m
> Q:\finance-analyzer\portfolio\signal_engine.py:536:        f"_TICKER_DISABLED_BY_[7mHORIZON[0m[{_k!r}] must be a dict")[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:537:    for _tk, _sigs in _inner.items():[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:538:        assert isinstance(_sigs, frozenset), ([0m
> Q:\finance-analyzer\portfolio\signal_engine.py:539:            f"_TICKER_DISABLED_BY_[7mHORIZON[0m[{_k!r}][{_tk!r}] [0m[7m[0mmust [0m
[7m[0mbe [0m[7m[0ma [0m[7m[0m"[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:540:            f"frozenset (got {type(_sigs).__name__})")[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:541:del _k, _inner[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:543:
> Q:\finance-analyzer\portfolio\signal_engine.py:544:def _get_[7mhorizon[0m_disabled_signals(ticker: [0m[7m[0mstr [0m[7m[0m| [0m[7m[0mNone, [0m[7m[0mhorizon: [0m
[7m[0mstr [0m[7m[0m| [0m[7m[0mNone) [0m[7m[0m-> [0m[7m[0mfrozenset:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:545:    """Return signals to force-HOLD for (ticker, [7mhorizon[0m). [0m[7m[0mUnion [0m
[7m[0mof [0m[7m[0mdefault [0m[7m[0m+ [0m[7m[0mhorizon-specific.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:546:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:547: [0m[7m [0m[7m [0m[7m [0m[7m[0mP3-1 [0m[7m[0m(2026-04-17): [0m[7m[0muses [0m[7m[0m.get('_default', [0m[7m[0m{}) [0m[7m[0mdefensively [0m
[7m[0minstead [0m[7m[0mof [0m[7m[0m[][0m
  Q:\finance-analyzer\portfolio\signal_engine.py:553:        return frozenset()
> Q:\finance-analyzer\portfolio\signal_engine.py:554:    default_map = _TICKER_DISABLED_BY_[7mHORIZON[0m.get("_default", {})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:555:    default_set = default_map.get(ticker, frozenset())[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:556:    if not [7mhorizon[0m:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:557:        return default_set[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:558:    [7mhorizon[0m_set [0m[7m[0m= [0m[7m[0m_TICKER_DISABLED_BY_HORIZON.get(horizon, [0m
[7m[0m{}).get(ticker, [0m[7m[0mfrozenset())[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:559:    return default_set | [7mhorizon[0m_set[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:560:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:561:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:562:# Backward-compat alias: the compute-time (signal dispatch) gate. 
Equal to the
> Q:\finance-analyzer\portfolio\signal_engine.py:563:# _default list — the minimum set of signals that are bad at 
every [7mhorizon[0m.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:564:# [0m[7m[0mExisting [0m[7m[0mcallers [0m[7m[0mreference [0m[7m[0mthis [0m[7m[0mname; [0m[7m[0mkeep [0m[7m[0mit [0m[7m[0mas [0m[7m[0ma [0m[7m[0mview [0m[7m[0mof [0m
[7m[0m_default.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:565:_TICKER_DISABLED_SIGNALS = _TICKER_DISABLED_BY_[7mHORIZON[0m["_default"][0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:566:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:567:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:579:# macro window, then auto-reverts when the window passes. It 
composes
> Q:\finance-analyzer\portfolio\signal_engine.py:580:# multiplicatively with the existing regime/[7mhorizon[0m weight chain.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:581:MACRO_WINDOW_DOWNWEIGHT_SIGNALS = frozenset({[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:582:    "sentiment", "momentum_factors", "structure",[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:599:
> Q:\finance-analyzer\portfolio\signal_engine.py:600:# --- Signal (full 32-signal for "Now" [7mtimeframe[0m) ---[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:601:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:602:MIN_VOTERS_CRYPTO [0m[7m[0m= [0m[7m[0m3 [0m[7m [0m[7m[0m# [0m[7m[0mcrypto [0m[7m[0mhas [0m[7m[0m30 [0m[7m[0msignals [0m[7m[0m(8 [0m[7m[0mcore [0m[7m[0m+ [0m[7m[0m22 [0m
[7m[0menhanced; [0m[7m[0mml [0m[7m[0mdisabled) [0m[7m[0m— [0m[7m[0mneed [0m[7m[0m3[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:769:# Regime-gated signals: completely silenced (forced HOLD) in 
certain regimes
> Q:\finance-analyzer\portfolio\signal_engine.py:770:# because they produce negative alpha.  [7mHorizon[0m-aware [0m[7m[0msince [0m
[7m[0m2026-03-29:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:771:# [0m[7m[0mBUG-149: [0m[7m[0mtrend [0m[7m[0mhas [0m[7m[0m61.6% [0m[7m[0maccuracy [0m[7m[0mon [0m[7m[0m3h [0m[7m[0meven [0m[7m[0min [0m[7m[0mranging [0m[7m[0m— [0m
[7m[0mshort-term [0m[7m[0mtrends[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:772:# exist within range-bound markets, so only gate on longer 
[7mhorizon[0ms.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:773:# Structure: {regime: {[7mhorizon[0m: frozenset(signals), ...}}
> Q:\finance-analyzer\portfolio\signal_engine.py:774:# "_default" key applies to [7mhorizon[0ms not explicitly listed.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:775:REGIME_GATED_SIGNALS: dict[str, dict[str, frozenset[str]]] = {[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:776:    "ranging": {[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:811:        }),
> Q:\finance-analyzer\portfolio\signal_engine.py:812:        # 3h: news_event 58.5%, smart_money 53.1% — decent at 
short [7mhorizon[0ms.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:813: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mvolatility_sig [0m[7m[0m47.2%, [0m[7m[0mforecast [0m[7m[0m47.2% [0m[7m[0m— [0m[7m[0mmarginal, [0m[7m[0mlet [0m
[7m[0maccuracy [0m[7m[0mgate[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:814:        # handle them dynamically at 3h.[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:816:        # 2026-04-11: sentiment added — 33.8% at 3h_recent (3629 
sam). The 0.5x
> Q:\finance-analyzer\portfolio\signal_engine.py:817:        # [7mhorizon [0m[7m[0mweight [0m[7m[0mis [0m[7m[0minsufficient; [0m[7m[0mthis [0m[7m[0msignal [0m[7m[0mactively [0m
[7m[0mharms [0m[7m[0m3h [0m[7m[0mconsensus.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:818: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m2026-04-25: [0m[7m[0mclaude_fundamental [0m[7m[0madded [0m[7m[0m— [0m[7m[0m0 [0m[7m[0maccuracy [0m
[7m[0msamples [0m[7m[0mat [0m[7m[0m3h [0m[7m[0mso[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:819: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mthe [0m[7m[0maccuracy [0m[7m[0mgate [0m[7m[0mdefaults [0m[7m[0mto [0m[7m[0m0.5 [0m[7m[0mand [0m[7m[0mpasses [0m[7m[0mthe [0m[7m[0m47% [0m
[7m[0mgate. [0m[7m[0mAt [0m[7m[0m1d [0m[7m[0mthe[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:838:        }),
> Q:\finance-analyzer\portfolio\signal_engine.py:839:        # mean_reversion 3h_recent=45.5% — gate on short [7mhorizon[0ms[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:840: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mSELL-biased [0m[7m[0msignals [0m[7m[0mwork [0m[7m[0mshort-term [0m[7m[0meven [0m[7m[0min [0m[7m[0muptrends [0m[7m[0m— [0m
[7m[0mdo [0m[7m[0mNOT [0m[7m[0mgate [0m[7m[0mat [0m[7m[0m3h[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:841: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m2026-04-13: [0m[7m[0msentiment [0m[7m[0m33.8% [0m[7m[0mat [0m[7m[0m3h [0m[7m[0m(3629 [0m[7m[0msam) [0m[7m[0m— [0m
[7m[0mdestructive [0m[7m[0mat [0m[7m[0m3h [0m[7m[0min [0m[7m[0mALL [0m[7m[0mregimes[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:859:            # 2026-04-27: sentiment 40.1% at 1d_recent (202 sam), 
BUY-only bias.
> Q:\finance-analyzer\portfolio\signal_engine.py:860:            # Was only gated at 3h/4h; actively harmful at longer 
[7mhorizon[0ms [0m[7m[0mtoo.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:861:            "sentiment",[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:862:        }),[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:882:
> Q:\finance-analyzer\portfolio\signal_engine.py:883:def _get_regime_gated(regime: str, [7mhorizon[0m: [0m[7m[0mstr [0m[7m[0m| [0m[7m[0mNone [0m[7m[0m= [0m[7m[0mNone) [0m[7m[0m-> [0m
[7m[0mfrozenset[str]:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:884:    """Get the set of signals to gate for a regime+[7mhorizon [0m
[7m[0mcombination.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:885:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:886:    Intentional semantics: [7mhorizon[0m-specific [0m[7m[0moverride [0m[7m[0mREPLACES [0m
[7m[0m`_default`,[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:887:    NOT unions with it. This is by design (BUG-149, 2026-03-29):[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:888:    `_default` lists signals that are bad at long [7mhorizon[0ms [0m
[7m[0m(1d/3d/5d) [0m[7m[0min[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:889:    a regime, while a [7mhorizon [0m[7m[0moverride [0m[7m[0m(3h/4h) [0m[7m[0mis [0m[7m[0mthe [0m
[7m[0mFINER-grained [0m[7m[0mlist[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:890:    of what should still be gated at that intraday [7mhorizon[0m. [0m
[7m[0mExample:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:891: [0m[7m [0m[7m [0m[7m [0m[7m[0m`trend` [0m[7m[0mhas [0m[7m[0m40.7% [0m[7m[0maccuracy [0m[7m[0mat [0m[7m[0m1d [0m[7m[0mranging [0m[7m[0m(gate [0m[7m[0mvia [0m[7m[0m_default) [0m
[7m[0mbut[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:892:    61.6% at 3h ranging (allow via no-mention in 3h override).[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:894:    2026-05-02 audit: 04-24 P0-1 / 04-29 SC-P1-1 / 05-01 P0 
(carryover)
> Q:\finance-analyzer\portfolio\signal_engine.py:895:    framed this as a "union bug" by analogy to 
`_get_[7mhorizon[0m_disabled_signals`.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:896: [0m[7m [0m[7m [0m[7m [0m[7m[0mRe-reading [0m[7m[0mthe [0m[7m[0mdocstring [0m[7m[0mat [0m[7m[0mline [0m[7m[0m762-767 [0m[7m[0mand [0m[7m[0mthe [0m[7m[0mper-signal [0m
[7m[0mcomments[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:897: [0m[7m [0m[7m [0m[7m [0m[7m[0min [0m[7m[0mREGIME_GATED_SIGNALS [0m[7m[0mconfirms [0m[7m[0mthe [0m[7m[0mintent [0m[7m[0mis [0m
[7m[0mreplace-semantics.[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:904:        return frozenset()
> Q:\finance-analyzer\portfolio\signal_engine.py:905:    if [7mhorizon[0m and horizon in regime_dict:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:906:        return regime_dict[[7mhorizon[0m][0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:907:    return regime_dict.get("_default", frozenset())[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:908:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:909:# [7mHorizon[0m-specific signal weight multipliers.
> Q:\finance-analyzer\portfolio\signal_engine.py:910:# Signals with >15pp accuracy divergence between [7mhorizon[0ms [0m[7m[0mget [0m
[7m[0madjusted.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:911:# Updated: 2026-04-27 accuracy audit (3h_recent vs 1d_recent).[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:912:[7mHORIZON[0m_SIGNAL_WEIGHTS: dict[str, dict[str, float]] = {[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:913:    "3h": {[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:914:        "news_event": 1.4,      # 70.0% at 3h_recent (1762 sam)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:983:
> Q:\finance-analyzer\portfolio\signal_engine.py:984:# Dynamic [7mhorizon[0m weight computation settings[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:985:_DYNAMIC_[7mHORIZON[0m_WEIGHT_TTL = 3600  # 1 hour cache
> Q:\finance-analyzer\portfolio\signal_engine.py:986:_DYNAMIC_[7mHORIZON[0m_MIN_SAMPLES [0m[7m[0m= [0m[7m[0m50 [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mneed [0m[7m[0menough [0m[7m[0mdata [0m[7m[0mper [0m[7m[0msignal [0m
[7m[0mper [0m[7m[0mhorizon[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:987:_DYNAMIC_[7mHORIZON[0m_CLAMP_LOW = 0.4    # minimum multiplier
> Q:\finance-analyzer\portfolio\signal_engine.py:988:_DYNAMIC_[7mHORIZON[0m_CLAMP_HIGH = 1.5   # maximum multiplier
> Q:\finance-analyzer\portfolio\signal_engine.py:989:_DYNAMIC_[7mHORIZON[0m_DEADBAND [0m[7m[0m= [0m[7m[0m0.1 [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mignore [0m[7m[0mmultipliers [0m[7m[0mwithin [0m
[7m[0m±10% [0m[7m[0mof [0m[7m[0m1.0[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:990:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:991:# Cross-[7mhorizon [0m[7m[0mpairs: [0m[7m[0mfor [0m[7m[0ma [0m[7m[0mgiven [0m[7m[0mhorizon, [0m[7m[0mwhich [0m[7m[0mother [0m[7m[0mhorizons [0m
[7m[0mto [0m[7m[0mcompare [0m[7m[0magainst[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:992:_CROSS_[7mHORIZON[0m_PAIRS = {[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:993:    "3h": ["1d"],[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:994:    "4h": ["1d"],[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:998:
> Q:\finance-analyzer\portfolio\signal_engine.py:999:def _compute_dynamic_[7mhorizon[0m_weights(horizon: [0m[7m[0mstr) [0m[7m[0m-> [0m[7m[0mdict[str, [0m
[7m[0mfloat]:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1000:    """Compute [7mhorizon[0m-specific [0m[7m[0msignal [0m[7m[0mweight [0m[7m[0mmultipliers [0m[7m[0mfrom [0m
[7m[0maccuracy [0m[7m[0mcache.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1001:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1002:    For each signal, computes the ratio of its accuracy on this 
[7mhorizon [0m[7m[0mvs[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1003:    the comparison [7mhorizon[0m(s). [0m[7m[0mSignals [0m[7m[0mthat [0m[7m[0mperform [0m[7m[0mmuch [0m[7m[0mbetter [0m
[7m[0mon [0m[7m[0mthis[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1004:    [7mhorizon [0m[7m[0mget [0m[7m[0mboosted; [0m[7m[0msignals [0m[7m[0mthat [0m[7m[0mperform [0m[7m[0mmuch [0m[7m[0mworse [0m[7m[0mget [0m
[7m[0mpenalized.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1005:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1006: [0m[7m [0m[7m [0m[7m [0m[7m[0mReturns [0m[7m[0ma [0m[7m[0mdict [0m[7m[0mof [0m[7m[0m{signal_name: [0m[7m[0mmultiplier} [0m[7m[0mfor [0m[7m[0mmultipliers [0m
[7m[0moutside[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1007:    the deadband (i.e., > 1.1 or < 0.9). Falls back to static
> Q:\finance-analyzer\portfolio\signal_engine.py:1008:    [7mHORIZON[0m_SIGNAL_WEIGHTS if accuracy cache is unavailable.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1009:    """[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1010:    try:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1013:        if not cache:
> Q:\finance-analyzer\portfolio\signal_engine.py:1014:            return [7mHORIZON[0m_SIGNAL_WEIGHTS.get(horizon, {})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1015:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1016:        # Get recent accuracy for this [7mhorizon [0m[7m[0mand [0m[7m[0mcomparison [0m
[7m[0mhorizons[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1017:        this_key = f"{[7mhorizon[0m}_recent"[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1018:        this_data = cache.get(this_key, {})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1019:        if not this_data:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1020:            return [7mHORIZON[0m_SIGNAL_WEIGHTS.get(horizon, {})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1021:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1022:        cross_[7mhorizon[0ms = _CROSS_HORIZON_PAIRS.get(horizon, [])
> Q:\finance-analyzer\portfolio\signal_engine.py:1023:        if not cross_[7mhorizon[0ms:
> Q:\finance-analyzer\portfolio\signal_engine.py:1024:            return [7mHORIZON[0m_SIGNAL_WEIGHTS.get(horizon, {})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1025:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1026:        # Gather comparison accuracies (true mean across 
comparison [7mhorizon[0ms)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1027:        cross_sum: dict[str, float] = {}[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1028:        cross_count: dict[str, int] = {}[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1029:        for ch in cross_[7mhorizon[0ms:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1030:            ch_key = f"{ch}_recent"[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1031:            ch_acc = cache.get(ch_key, {})[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1032:            for sig, stats in ch_acc.items():
> Q:\finance-analyzer\portfolio\signal_engine.py:1033:                if stats.get("total", 0) >= 
_DYNAMIC_[7mHORIZON[0m_MIN_SAMPLES:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1034:                    acc = stats.get("accuracy", 0.5)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1035: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mcross_sum[sig] [0m[7m[0m= [0m[7m[0mcross_sum.get(sig, [0m[7m[0m0.0) [0m[7m[0m+ [0m
[7m[0macc[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1046:            samples = stats.get("total", 0)
> Q:\finance-analyzer\portfolio\signal_engine.py:1047:            if samples < _DYNAMIC_[7mHORIZON[0m_MIN_SAMPLES:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1048:                continue[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1049:            this_acc = stats.get("accuracy", 0.5)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1053:
> Q:\finance-analyzer\portfolio\signal_engine.py:1054:            # Ratio of this-[7mhorizon [0m[7m[0maccuracy [0m[7m[0mto [0m[7m[0mcross-horizon [0m
[7m[0maccuracy[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1055:            ratio = this_acc / cross_acc[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1056:            # Clamp[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1057:            ratio = max(_DYNAMIC_[7mHORIZON[0m_CLAMP_LOW, [0m
[7m[0mmin(_DYNAMIC_HORIZON_CLAMP_HIGH, [0m[7m[0mratio))[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1058: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mDeadband: [0m[7m[0monly [0m[7m[0minclude [0m[7m[0mif [0m[7m[0mmeaningfully [0m[7m[0mdifferent [0m
[7m[0mfrom [0m[7m[0m1.0[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1059:            if abs(ratio - 1.0) > _DYNAMIC_[7mHORIZON[0m_DEADBAND:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1060:                weights[sig] = round(ratio, 2)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1061:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1062:        return weights if weights else 
[7mHORIZON[0m_SIGNAL_WEIGHTS.get(horizon, [0m[7m[0m{})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1063:    except Exception:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1064:        logger.debug("Dynamic [7mhorizon [0m[7m[0mweights [0m[7m[0munavailable, [0m[7m[0musing [0m
[7m[0mstatic [0m[7m[0mfallback", [0m[7m[0mexc_info=True)[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1065:        return [7mHORIZON[0m_SIGNAL_WEIGHTS.get(horizon, {})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1066:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1067:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1068:def _get_[7mhorizon[0m_weights(horizon: [0m[7m[0mstr [0m[7m[0m| [0m[7m[0mNone) [0m[7m[0m-> [0m[7m[0mdict[str, [0m
[7m[0mfloat]:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1069:    """Get [7mhorizon[0m-specific [0m[7m[0msignal [0m[7m[0mweight [0m[7m[0mmultipliers, [0m
[7m[0mpreferring [0m[7m[0mdynamic [0m[7m[0mcomputation.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1070:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1071: [0m[7m [0m[7m [0m[7m [0m[7m[0mUses [0m[7m[0mcached [0m[7m[0mdynamic [0m[7m[0mweights [0m[7m[0mwhen [0m[7m[0mavailable, [0m[7m[0mfalling [0m[7m[0mback [0m[7m[0mto [0m
[7m[0mstatic [0m[7m[0mdict.[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1072:    """
> Q:\finance-analyzer\portfolio\signal_engine.py:1073:    if not [7mhorizon[0m:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1074:        return {}[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1075:    cache_key = f"dynamic_[7mhorizon[0m_weights_{horizon}"
> Q:\finance-analyzer\portfolio\signal_engine.py:1076:    return _cached(cache_key, _DYNAMIC_[7mHORIZON[0m_WEIGHT_TTL, [0m
[7m[0mlambda: [0m[7m[0m_compute_dynamic_horizon_weights(horizon))[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1077:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1078:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1097:            continue
> Q:\finance-analyzer\portfolio\signal_engine.py:1098:        # Per-ticker blacklist: check _default [7mhorizon [0m[7m[0mfor [0m
[7m[0msignals [0m[7m[0mbad [0m[7m[0mat [0m[7m[0mall [0m[7m[0mhorizons[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1099:        if sig in _TICKER_DISABLED_SIGNALS.get(ticker, ()):[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1100:            continue[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1665:
> Q:\finance-analyzer\portfolio\signal_engine.py:1666:def _get_ic_data([7mhorizon[0m: str) -> dict | None:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1667:    """Load IC data for the given [7mhorizon[0m, [0m[7m[0mcomputing [0m[7m[0mif [0m[7m[0mcache [0m[7m[0mis [0m
[7m[0mstale.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1668:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1669: [0m[7m [0m[7m [0m[7m [0m[7m[0mReturns [0m[7m[0mthe [0m[7m[0mfull [0m[7m[0mcache [0m[7m[0mdict [0m[7m[0m{"global": [0m[7m[0m{...}, [0m[7m[0m"per_ticker": [0m
[7m[0m{...}}[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1673:    with _ic_data_lock:
> Q:\finance-analyzer\portfolio\signal_engine.py:1674:        cached = _ic_data_cache.get([7mhorizon[0m)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1675: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mif [0m[7m[0mcached [0m[7m[0mand [0m[7m[0mnow [0m[7m[0m- [0m[7m[0mcached.get("_loaded_at", [0m[7m[0m0) [0m[7m[0m< [0m
[7m[0m_IC_DATA_TTL:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1676:            return cached[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1679:        from portfolio.ic_computation import 
compute_and_cache_ic, load_cached_ic
> Q:\finance-analyzer\portfolio\signal_engine.py:1680:        cache = load_cached_ic([7mhorizon[0m)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1681:        if cache is None:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1682:            cache = compute_and_cache_ic([7mhorizon[0m)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1683:        if cache:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1684:            cache["_loaded_at"] = now[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1685:            with _ic_data_lock:
> Q:\finance-analyzer\portfolio\signal_engine.py:1686:                _ic_data_cache[[7mhorizon[0m] = cache[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1687:            return cache[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1688:    except Exception:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1689:        logger.debug("IC data unavailable for %s", [7mhorizon[0m, [0m
[7m[0mexc_info=True)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1690:    return None[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1691:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1737:def _weighted_consensus(votes, accuracy_data, regime, 
activation_rates=None,
> Q:\finance-analyzer\portfolio\signal_engine.py:1738:                        accuracy_gate=None, max_signals=None, 
[7mhorizon[0m=None,[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1739:                        regime_gated_override=None, ticker=None):[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1740: [0m[7m [0m[7m [0m[7m [0m[7m[0m"""Compute [0m[7m[0mweighted [0m[7m[0mconsensus [0m[7m[0musing [0m[7m[0maccuracy, [0m[7m[0mIC, [0m[7m[0mregime, [0m
[7m[0mand [0m[7m[0mactivation [0m[7m[0mfrequency.[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1742:    Weight per signal = accuracy_weight * ic_mult * regime_mult 
* normalized_weight
> Q:\finance-analyzer\portfolio\signal_engine.py:1743:                        * [7mhorizon[0m_mult * activity_cap[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1744: [0m[7m [0m[7m [0m[7m [0m[7m[0mwhere [0m[7m[0mnormalized_weight [0m[7m[0m= [0m[7m[0mrarity_bonus [0m[7m[0m* [0m[7m[0mbias_penalty [0m[7m[0m(from [0m
[7m[0mactivation [0m[7m[0mrates).[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1745: [0m[7m [0m[7m [0m[7m [0m[7m[0mRare, [0m[7m[0mbalanced [0m[7m[0msignals [0m[7m[0mget [0m[7m[0mmore [0m[7m[0mweight; [0m[7m[0mnoisy/biased [0m[7m[0msignals [0m
[7m[0mget [0m[7m[0mless.[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1755:
> Q:\finance-analyzer\portfolio\signal_engine.py:1756:    [7mHorizon[0m-specific [0m[7m[0mweights: [0m[7m[0msignals [0m[7m[0mwith [0m[7m[0mdivergent [0m[7m[0maccuracy [0m
[7m[0macross [0m[7m[0mhorizons[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1757:    get boosted or penalized via [7mHORIZON[0m_SIGNAL_WEIGHTS.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1758:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1759: [0m[7m [0m[7m [0m[7m [0m[7m[0mActivity [0m[7m[0mrate [0m[7m[0mcap: [0m[7m[0msignals [0m[7m[0mwith [0m[7m[0m>70% [0m[7m[0mactivation [0m[7m[0mrate [0m[7m[0mget [0m
[7m[0m0.5x [0m[7m[0mpenalty[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1771:    activation_rates = activation_rates or {}
> Q:\finance-analyzer\portfolio\signal_engine.py:1772:    [7mhorizon[0m_mults = _get_horizon_weights(horizon)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1773:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1774: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mCodex [0m[7m[0mround [0m[7m[0m10/11/12 [0m[7m[0m(2026-04-17 [0m[7m[0mfollow-up): [0m[7m[0mdeep-sanitize [0m
[7m[0maccuracy_data[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1855:    # Regime gating: force-HOLD signals that produce negative 
alpha in this regime.
> Q:\finance-analyzer\portfolio\signal_engine.py:1856:    # BUG-149: now [7mhorizon[0m-aware [0m[7m[0m— [0m[7m[0me.g., [0m[7m[0mtrend [0m[7m[0mworks [0m[7m[0mat [0m[7m[0m3h [0m[7m[0min [0m
[7m[0mranging [0m[7m[0m(61.6%)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1857: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mSC-I-001: [0m[7m[0mwhen [0m[7m[0mcaller [0m[7m[0mprovides [0m[7m[0mregime_gated_override [0m[7m[0m(with [0m
[7m[0mBUG-158 [0m[7m[0mper-ticker[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1858: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mexemptions [0m[7m[0malready [0m[7m[0mapplied), [0m[7m[0muse [0m[7m[0mit [0m[7m[0minstead [0m[7m[0mof [0m[7m[0mrecomputing [0m
[7m[0mfrom [0m[7m[0mscratch.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1859:    regime_gated = regime_gated_override if 
regime_gated_override is not None else _get_regime_gated(regime, [7mhorizon[0m)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1860: [0m[7m [0m[7m [0m[7m [0m[7m[0mvotes [0m[7m[0m= [0m[7m[0m{k: [0m[7m[0m("HOLD" [0m[7m[0mif [0m[7m[0mk [0m[7m[0min [0m[7m[0mregime_gated [0m[7m[0melse [0m[7m[0mv) [0m[7m[0mfor [0m[7m[0mk, [0m[7m[0mv [0m[7m[0min [0m
[7m[0mvotes.items()}[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1861:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1862:    # [7mHorizon[0m-specific [0m[7m[0mper-ticker [0m[7m[0mblacklist [0m[7m[0m(2026-04-16, [0m[7m[0mBatch [0m
[7m[0m4). [0m[7m[0mExtends [0m[7m[0mthe[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1863:    # compute-time _default blacklist with [7mhorizon[0m-specific [0m
[7m[0mentries. [0m[7m[0mCompute [0m[7m[0mtime[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1864:    # can't see [7mhorizon [0m[7m[0m(one [0m[7m[0mvote [0m[7m[0mreused [0m[7m[0macross [0m
[7m[0m3h/4h/12h/1d/3d/5d/10d [0m[7m[0mconsensus),[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1865:    # so per-[7mhorizon[0m gating must happen here.
> Q:\finance-analyzer\portfolio\signal_engine.py:1866:    [7mhorizon[0m_disabled [0m[7m[0m= [0m[7m[0m_get_horizon_disabled_signals(ticker, [0m
[7m[0mhorizon)[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:1867:    if [7mhorizon[0m_disabled:
> Q:\finance-analyzer\portfolio\signal_engine.py:1868:        votes = {k: ("HOLD" if k in [7mhorizon[0m_disabled [0m[7m[0melse [0m[7m[0mv) [0m[7m[0mfor [0m
[7m[0mk, [0m[7m[0mv [0m[7m[0min [0m[7m[0mvotes.items()}[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1869:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:1870: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mMacro-window [0m[7m[0mforce-HOLD [0m[7m[0mpre-pass [0m[7m[0m(2026-04-28). [0m[7m[0mWhen [0m[7m[0ma [0m
[7m[0mhigh-impact[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:1873:    # The downweight branch for the other macro-fragile signals 
lives in
> Q:\finance-analyzer\portfolio\signal_engine.py:1874:    # the weight loop below so it composes with regime/[7mhorizon [0m
[7m[0mmultipliers.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1875:    macro_active = _is_macro_window_cached()[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:1876:    if macro_active and MACRO_WINDOW_FORCE_HOLD_SIGNALS:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2063:    # or None if IC computation is unavailable.
> Q:\finance-analyzer\portfolio\signal_engine.py:2064:    ic_cache = _get_ic_data([7mhorizon[0m) if horizon else None[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2065:    ic_global = ic_cache.get("global", {}) if ic_cache else {}[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2066: [0m[7m [0m[7m [0m[7m [0m[7m[0mic_per_ticker [0m[7m[0m= [0m[7m[0mic_cache.get("per_ticker", [0m[7m[0m{}) [0m[7m[0mif [0m[7m[0mic_cache [0m
[7m[0melse [0m[7m[0m{}[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2173:        weight *= regime_mults.get(signal_name, 1.0)
> Q:\finance-analyzer\portfolio\signal_engine.py:2174:        # [7mHorizon[0m-specific weight adjustment[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:2175:        if signal_name in [7mhorizon[0m_mults:
> Q:\finance-analyzer\portfolio\signal_engine.py:2176:            weight *= [7mhorizon[0m_mults[signal_name][0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2177: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mMacro-window [0m[7m[0mdownweight [0m[7m[0m(2026-04-28). [0m[7m[0mComposes [0m[7m[0mwith [0m
[7m[0mregime/[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:2178:        # [7mhorizon [0m[7m[0mmultipliers [0m[7m[0m— [0m[7m[0me.g., [0m[7m[0mduring [0m[7m[0ma [0m[7m[0mmacro [0m[7m[0mwindow [0m[7m[0min [0m
[7m[0mranging[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:2179:        # regime, sentiment hits 0.5 (macro) × 0.X (regime) × Y 
([7mhorizon[0m).[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2180:        # Only applies to MACRO_WINDOW_DOWNWEIGHT_SIGNALS — the[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2181: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mFORCE_HOLD [0m[7m[0msignals [0m[7m[0mwere [0m[7m[0malready [0m[7m[0mmutated [0m[7m[0mto [0m[7m[0mHOLD [0m[7m[0mabove [0m
[7m[0mand [0m[7m[0mwon't[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2251:
> Q:\finance-analyzer\portfolio\signal_engine.py:2252:def _time_of_day_factor([7mhorizon[0m=None):[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2253:    hour = datetime.now(UTC).hour[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:2254:    if [7mhorizon[0m in ("3h", "4h"):
> Q:\finance-analyzer\portfolio\signal_engine.py:2255:        from portfolio.short_[7mhorizon[0m import time_of_day_scale_3h[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2256:        return time_of_day_scale_3h(hour)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2257:    # Default 1d behavior[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2262:
> Q:\finance-analyzer\portfolio\signal_engine.py:2263:def _load_local_model_accuracy(signal_name, [7mhorizon[0m="1d", [0m
[7m[0mdays=None, [0m[7m[0mcache_ttl=None):[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2264:    """Load per-ticker accuracy for a local model signal."""[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2265: [0m[7m [0m[7m [0m[7m [0m[7m[0mlookback_days [0m[7m[0m= [0m[7m[0mdays [0m[7m[0mif [0m[7m[0mdays [0m[7m[0mis [0m[7m[0mnot [0m[7m[0mNone [0m[7m[0melse [0m
[7m[0m_LOCAL_MODEL_LOOKBACK_DAYS[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2266:    ttl = cache_ttl or _LOCAL_MODEL_ACCURACY_TTL
> Q:\finance-analyzer\portfolio\signal_engine.py:2267:    cache_key = 
f"local_model_accuracy_{signal_name}_{[7mhorizon[0m}_{lookback_days}"[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2268:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2269:    def _fetch():[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2272:
> Q:\finance-analyzer\portfolio\signal_engine.py:2273:            return accuracy_by_signal_ticker(signal_name, 
[7mhorizon[0m=horizon, [0m[7m[0mdays=lookback_days)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2274:        except Exception:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2275: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mlogger.warning("Failed [0m[7m[0mto [0m[7m[0mload [0m[7m[0m%s [0m[7m[0maccuracy", [0m
[7m[0msignal_name, [0m[7m[0mexc_info=True)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2280:
> Q:\finance-analyzer\portfolio\signal_engine.py:2281:def _build_llm_context(ticker, ind, [7mtimeframe[0ms, extra_info):[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2282: [0m[7m [0m[7m [0m[7m [0m[7m[0m"""Build [0m[7m[0mshared [0m[7m[0mcontext [0m[7m[0mdict [0m[7m[0mfor [0m[7m[0mlocal [0m[7m[0mLLM [0m[7m[0msignals [0m
[7m[0m(Ministral, [0m[7m[0mQwen3)."""[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2283:    tf_summary = ""[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:2284:    if [7mtimeframe[0ms:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2285:        parts = [][0m
> Q:\finance-analyzer\portfolio\signal_engine.py:2286:        for label, entry in [7mtimeframe[0ms:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2287: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mif [0m[7m[0misinstance(entry, [0m[7m[0mdict) [0m[7m[0mand [0m[7m[0m"action" [0m[7m[0min [0m[7m[0mentry [0m[7m[0mand [0m
[7m[0mentry["action"]:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2288:                ti = entry.get("indicators", {})[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2314:        "funding_rate": extra_info.get("funding_action", "N/A"),
> Q:\finance-analyzer\portfolio\signal_engine.py:2315:        "[7mtimeframe[0m_summary": tf_summary,[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2316:        "headlines": "",[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2317:    }[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2336:    accuracy_data = _load_local_model_accuracy(
> Q:\finance-analyzer\portfolio\signal_engine.py:2337:        signal_name, [7mhorizon[0m=cfg.get("horizon", [0m[7m[0m"1d"), [0m
[7m[0mdays=days, [0m[7m[0mcache_ttl=cache_ttl[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2338:    )[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2339:    ticker_stats = (accuracy_data or {}).get(ticker)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2478:    # --- Stage 3: Trap detection ---
> Q:\finance-analyzer\portfolio\signal_engine.py:2479:    # NOTE: df must be the "Now" [7mtimeframe [0m[7m[0m(15m [0m[7m[0mcandles, [0m[7m[0m100 [0m
[7m[0mbars [0m[7m[0m≈ [0m[7m[0m25h).[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2480: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mLast [0m[7m[0m5 [0m[7m[0mbars [0m[7m[0m= [0m[7m[0m75 [0m[7m[0mminutes [0m[7m[0m— [0m[7m[0mappropriate [0m[7m[0mfor [0m[7m[0mintraday [0m[7m[0mtrap [0m
[7m[0mdetection.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2481: [0m[7m [0m[7m [0m[7m [0m[7m[0mif [0m[7m[0maction [0m[7m[0m!= [0m[7m[0m"HOLD" [0m[7m[0mand [0m[7m[0mdf [0m[7m[0mis [0m[7m[0mnot [0m[7m[0mNone [0m[7m[0mand [0m[7m[0misinstance(df, [0m
[7m[0mpd.DataFrame) [0m[7m[0mand [0m[7m[0mlen(df) [0m[7m[0m>= [0m[7m[0m5:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2520:    # When all signals agree, the move is often already priced 
in.
> Q:\finance-analyzer\portfolio\signal_engine.py:2521:    # 90%+ confidence has 28-32% actual accuracy across all 
[7mhorizon[0ms.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2522:    if action != "HOLD" and conf > 0.0:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2523:        buy_count = extra_info.get("_buy_count", 0)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2584:
> Q:\finance-analyzer\portfolio\signal_engine.py:2585:def generate_signal(ind, ticker=None, config=None, 
[7mtimeframe[0ms=None, [0m[7m[0mdf=None, [0m[7m[0mhorizon=None):[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2586: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mCRITICAL-2 [0m[7m[0mguard [0m[7m[0m(2026-04-17 [0m[7m[0madversarial [0m[7m[0mreview): [0m
[7m[0mempty/None [0m[7m[0mticker[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2587: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mslipped [0m[7m[0mthrough [0m[7m[0mscattered [0m[7m[0m`if [0m[7m[0mticker:` [0m[7m[0mchecks [0m[7m[0min [0m
[7m[0mproduction [0m[7m[0mbefore.[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2632:    # RSI — only votes at extremes (adaptive thresholds from 
rolling percentiles)
> Q:\finance-analyzer\portfolio\signal_engine.py:2633:    if [7mhorizon[0m in ("3h", "4h"):[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2634: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m3h: [0m[7m[0mRSI(7) [0m[7m[0mis [0m[7m[0mmore [0m[7m[0msensitive [0m[7m[0m— [0m[7m[0muse [0m[7m[0mfixed [0m[7m[0m25/75 [0m
[7m[0mthresholds[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2635:        rsi_lower = 25[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2795:
> Q:\finance-analyzer\portfolio\signal_engine.py:2796:    # ML Classifier — disabled: 28.2% accuracy (1,027 samples, 
1d [7mhorizon[0m).[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2797: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mWorse [0m[7m[0mthan [0m[7m[0mcoin [0m[7m[0mflip; [0m[7m[0mactively [0m[7m[0mharmful [0m[7m[0mto [0m[7m[0mconsensus. [0m[7m[0mStill [0m
[7m[0mtracked [0m[7m[0mfor[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2798:    # accuracy monitoring but never votes.[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2801:    # Funding Rate — 29.9% accuracy at 1d but 74.2% at 3h (535 
samples).
> Q:\finance-analyzer\portfolio\signal_engine.py:2802:    # Re-enabled 2026-04-09: [7mhorizon[0m-gated [0m[7m[0mvia [0m
[7m[0mREGIME_GATED_SIGNALS [0m[7m[0m(_default[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:2803:    # gates it at 1d across all regimes; active only at 3h/4h 
[7mhorizon[0ms).[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2804: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mCrypto-only [0m[7m[0m(BTC, [0m[7m[0mETH). [0m[7m[0mThe [0m[7m[0mregime [0m[7m[0mgate [0m[7m[0mhandles [0m
[7m[0msuppression [0m[7m[0mat [0m[7m[0m1d.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2805:    votes["funding"] = "HOLD"[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2904:
> Q:\finance-analyzer\portfolio\signal_engine.py:2905:            ctx = _build_llm_context(ticker, ind, [7mtimeframe[0ms, [0m
[7m[0mextra_info)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2906: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m2026-04-10 [0m[7m[0m(perf/llama-swap-reduction): [0m[7m[0mgate [0m[7m[0mthe [0m
[7m[0menqueue [0m[7m[0mwith[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:2907: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mrotation [0m[7m[0mpredicate. [0m[7m[0mWhen [0m[7m[0mministral [0m[7m[0mis [0m[7m[0moff-cycle [0m
[7m[0mthis [0m[7m[0mcycle,[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:2951:
> Q:\finance-analyzer\portfolio\signal_engine.py:2952:            ctx = _build_llm_context(ticker, ind, [7mtimeframe[0ms, [0m
[7m[0mextra_info)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2953:            # Qwen3 gets asset_type for prompt diversification[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:2954:            if ticker in CRYPTO_SYMBOLS:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3196:                tier = indicators.get("tier")
> Q:\finance-analyzer\portfolio\signal_engine.py:3197:            # Default [7mhorizon [0m[7m[0mto [0m[7m[0m"1d" [0m[7m[0mwhen [0m[7m[0mcaller [0m[7m[0mpasses [0m[7m[0mNone [0m[7m[0mso [0m
[7m[0mthe [0m[7m[0mlog [0m[7m[0mis[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3198:            # queryable by [7mhorizon [0m[7m[0mwithout [0m[7m[0mnull-handling [0m[7m[0mat [0m
[7m[0mevery [0m[7m[0mjoin [0m[7m[0msite.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3199:            log_vote([0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3200:                sig_name, ticker or "", probs,[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3201:                [7mhorizon[0m=horizon [0m[7m[0mor [0m[7m[0m"1d", [0m[7m[0mchosen=action, [0m
[7m[0mconfidence=conf, [0m[7m[0mtier=tier,[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3202:            )[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3203:    except Exception:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3215:
> Q:\finance-analyzer\portfolio\signal_engine.py:3216:    # 3h [7mhorizon[0m: [0m[7m[0mgate [0m[7m[0mslow [0m[7m[0msignals [0m[7m[0mthat [0m[7m[0mare [0m[7m[0mnoise [0m[7m[0mat [0m[7m[0mshort [0m
[7m[0mtimeframes[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3217:    if [7mhorizon[0m in ("3h", "4h"):
> Q:\finance-analyzer\portfolio\signal_engine.py:3218:        from portfolio.short_[7mhorizon[0m import is_slow_signal_3h[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3219:        for sig_name in list(votes.keys()):[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3220: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mif [0m[7m[0mis_slow_signal_3h(sig_name) [0m[7m[0mand [0m[7m[0mvotes[sig_name] [0m
[7m[0m!= [0m[7m[0m"HOLD":[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3226:    # (idempotent — gating HOLD→HOLD is a no-op).
> Q:\finance-analyzer\portfolio\signal_engine.py:3227:    # BUG-149: now [7mhorizon[0m-aware via _get_regime_gated()[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3228: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mBUG-158: [0m[7m[0mPer-ticker [0m[7m[0mexemption [0m[7m[0m— [0m[7m[0mif [0m[7m[0ma [0m[7m[0msignal [0m[7m[0mhas [0m[7m[0m≥60% [0m
[7m[0maccuracy [0m[7m[0mwith [0m[7m[0m≥50[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3229: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0msamples [0m[7m[0mon [0m[7m[0mTHIS [0m[7m[0mticker, [0m[7m[0mexempt [0m[7m[0mit [0m[7m[0mfrom [0m[7m[0mregime [0m[7m[0mgating. [0m
[7m[0mfear_greed [0m[7m[0mis [0m[7m[0m93.8%[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3230:    # on XAG-USD but globally gated in ranging — this recovers 
that alpha.
> Q:\finance-analyzer\portfolio\signal_engine.py:3231:    regime_gated = _get_regime_gated(regime, [7mhorizon[0m)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3232:    _ticker_acc_data = {}[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3233:    try:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3234:        from portfolio.accuracy_stats import 
accuracy_by_ticker_signal_cached
> Q:\finance-analyzer\portfolio\signal_engine.py:3235:        acc_[7mhorizon [0m[7m[0m= [0m[7m[0mhorizon [0m[7m[0mif [0m[7m[0mhorizon [0m[7m[0min [0m[7m[0m("3h", [0m[7m[0m"4h", [0m[7m[0m"12h") [0m
[7m[0melse [0m[7m[0m"1d"[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3236:        _ticker_acc_data = 
(accuracy_by_ticker_signal_cached(acc_[7mhorizon[0m) [0m[7m[0mor [0m[7m[0m{}).get(ticker, [0m[7m[0m{})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3237:    except Exception:[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3238: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mlogger.debug("Per-ticker [0m[7m[0maccuracy [0m[7m[0munavailable [0m[7m[0mfor [0m[7m[0mregime [0m
[7m[0mgating [0m[7m[0mexemption", [0m[7m[0mexc_info=True)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3250:        from portfolio.accuracy_stats import 
get_or_compute_recent_accuracy
> Q:\finance-analyzer\portfolio\signal_engine.py:3251:        # get_or_compute_recent_accuracy expects the base 
[7mhorizon[0m, [0m[7m[0mnot [0m[7m[0mthe [0m[7m[0mcache [0m[7m[0mkey[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3252:        base_hz = "3h" if [7mhorizon[0m in ("3h", "4h") else "1d"[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3253: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m_recent_acc_data [0m[7m[0m= [0m
[7m[0mget_or_compute_recent_accuracy(base_hz) [0m[7m[0mor [0m[7m[0m{}[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3254:    except Exception:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3284:
> Q:\finance-analyzer\portfolio\signal_engine.py:3285:    # P1-B (2026-04-17 adversarial review): apply 
[7mhorizon[0m-specific [0m[7m[0mblacklist[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3286: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mHERE, [0m[7m[0mbefore [0m[7m[0mbuy/sell [0m[7m[0mcounting, [0m[7m[0mso [0m[7m[0m`active_voters` [0m
[7m[0mreflects [0m[7m[0mthe [0m[7m[0mpost-[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3287:    # [7mhorizon[0m-disable [0m[7m[0mstate. [0m[7m[0mPreviously [0m[7m[0mthis [0m[7m[0mgating [0m[7m[0monly [0m
[7m[0mhappened [0m[7m[0minside[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3288: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m_weighted_consensus, [0m[7m[0mleaving [0m[7m[0m`extra_info["_voters"]` [0m[7m[0mstale [0m
[7m[0m- [0m[7m[0mStage [0m[7m[0m4's[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3289: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mdynamic_min [0m[7m[0mcheck [0m[7m[0mcould [0m[7m[0mpass [0m[7m[0ma [0m[7m[0m5-voter [0m[7m[0mcount [0m[7m[0meven [0m[7m[0mthough [0m
[7m[0monly [0m[7m[0m2 [0m[7m[0mvoters[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3292:    # in depth for callers that bypass generate_signal.
> Q:\finance-analyzer\portfolio\signal_engine.py:3293:    [7mhorizon[0m_disabled_effective [0m[7m[0m= [0m
[7m[0m_get_horizon_disabled_signals(ticker, [0m[7m[0mhorizon)[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3294:    for sig_name in [7mhorizon[0m_disabled_effective:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3295:        if sig_name in votes and votes[sig_name] != "HOLD":[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3296:            votes[sig_name] = "HOLD"[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3302:    # only core voter (e.g. claude_fundamental) was suppressed 
by macro.
> Q:\finance-analyzer\portfolio\signal_engine.py:3303:    # Mirrors the existing regime_gate / [7mhorizon[0m_disabled [0m
[7m[0mmutation[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3304:    # pattern above.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3305:    macro_active_effective = _is_macro_window_cached()[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3357:    activation_rates = {}
> Q:\finance-analyzer\portfolio\signal_engine.py:3358:    # H3: Define acc_[7mhorizon [0m[7m[0mbefore [0m[7m[0mthe [0m[7m[0mtry/except [0m[7m[0mso [0m[7m[0mthe [0m[7m[0mexcept [0m
[7m[0mblock [0m[7m[0mand[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3359:    # subsequent code can reference it even if the import fails.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3360:    acc_[7mhorizon [0m[7m[0m= [0m[7m[0mhorizon [0m[7m[0mif [0m[7m[0mhorizon [0m[7m[0min [0m[7m[0m("3h", [0m[7m[0m"4h", [0m[7m[0m"12h") [0m[7m[0melse [0m
[7m[0m"1d"[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3361:    try:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3362:        from portfolio.accuracy_stats import ([0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3374:        # See accuracy_stats.py for the lock rationale.
> Q:\finance-analyzer\portfolio\signal_engine.py:3375:        alltime = get_or_compute_accuracy(acc_[7mhorizon[0m)[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3376:        recent = get_or_compute_recent_accuracy(acc_[7mhorizon[0m, [0m
[7m[0mdays=7)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3377: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mBUG-164 [0m[7m[0mlazy-populate [0m[7m[0mper-ticker [0m[7m[0mconsensus [0m[7m[0maccuracy [0m[7m[0m— [0m
[7m[0m_ptc_key[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3378: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mconvention [0m[7m[0mpreserved [0m[7m[0mby [0m
[7m[0mget_or_compute_per_ticker_accuracy.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3379:        _ptc_data = 
get_or_compute_per_ticker_accuracy(acc_[7mhorizon[0m)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3380: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mRES-2026-04-17: [0m[7m[0mPass [0m[7m[0mper-ticker [0m[7m[0mconsensus [0m[7m[0maccuracy [0m
[7m[0minto [0m[7m[0mextra_info[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3381: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mso [0m[7m[0mapply_confidence_penalties [0m[7m[0mcan [0m[7m[0mpenalize [0m[7m[0mtickers [0m
[7m[0mwhere [0m[7m[0mthe [0m[7m[0mconsensus[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3418:            from portfolio.accuracy_stats import 
get_or_compute_regime_accuracy
> Q:\finance-analyzer\portfolio\signal_engine.py:3419:            # BUG-134: Use acc_[7mhorizon [0m[7m[0m(not [0m[7m[0mhardcoded [0m[7m[0m"1d") [0m[7m[0mso [0m
[7m[0mregime [0m[7m[0maccuracy[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3420:            # matches the prediction [7mhorizon[0m (3h/4h/12h/1d).[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3421: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0m2026-05-04: [0m[7m[0mswitched [0m[7m[0mfrom [0m[7m[0mmanual [0m[7m[0mL2-only [0m[7m[0mdance [0m[7m[0mto [0m
[7m[0mL1+L2 [0m[7m[0mwrapper.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3422: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mThe [0m[7m[0mprevious [0m[7m[0mcode [0m[7m[0mre-read [0m[7m[0mdisk [0m[7m[0mon [0m[7m[0mevery [0m[7m[0mticker [0m
[7m[0mcall [0m[7m[0m(~10-50ms[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3427:            # failure preserves the pre-existing fall-through 
behavior.
> Q:\finance-analyzer\portfolio\signal_engine.py:3428:            regime_acc = 
get_or_compute_regime_accuracy(acc_[7mhorizon[0m)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3429:            current_regime_data = regime_acc.get(regime, {})[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3430:            for sig_name, rdata in current_regime_data.items():[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3472:    # Utility boost: scale accuracy weight by return-based 
utility score.
> Q:\finance-analyzer\portfolio\signal_engine.py:3473:    # Utility boost and best-[7mhorizon[0m overlay.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3474: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mH3: [0m[7m[0mSkip [0m[7m[0mwhen [0m[7m[0mprimary [0m[7m[0mload [0m[7m[0mfailed [0m[7m[0mto [0m[7m[0mpreserve [0m[7m[0mfail-closed [0m
[7m[0mgate.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3475:    if not _accuracy_failed:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3477:            from portfolio.accuracy_stats import signal_utility
> Q:\finance-analyzer\portfolio\signal_engine.py:3478:            # BUG-135: Use acc_[7mhorizon [0m[7m[0m(not [0m[7m[0mhardcoded [0m[7m[0m"1d") [0m[7m[0mso [0m
[7m[0mutility [0m[7m[0mboost[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3479:            # reflects the actual prediction [7mhorizon[0m's [0m[7m[0mreturn [0m
[7m[0mprofile.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3480:            utility_data = signal_utility(acc_[7mhorizon[0m)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3481:            for sig_name in list(accuracy_data.keys()):[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3482:                u = utility_data.get(sig_name, {})[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3501:
> Q:\finance-analyzer\portfolio\signal_engine.py:3502:    # Multi-[7mhorizon[0m: [0m[7m[0moptionally [0m[7m[0muse [0m[7m[0meach [0m[7m[0msignal's [0m[7m[0mbest [0m[7m[0mhorizon [0m
[7m[0maccuracy.[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3503: [0m[7m [0m[7m [0m[7m [0m[7m[0m# [0m[7m[0mH3: [0m[7m[0mSkip [0m[7m[0mwhen [0m[7m[0mprimary [0m[7m[0mload [0m[7m[0mfailed [0m[7m[0mto [0m[7m[0mpreserve [0m[7m[0mfail-closed [0m
[7m[0mgate.[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3504:    sig_cfg = (config or {}).get("signals", {})[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3505:    if not _accuracy_failed and sig_cfg.get("use_best_[7mhorizon[0m", [0m
[7m[0mFalse):[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3506:        try:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3507:            from portfolio.accuracy_stats import 
signal_best_[7mhorizon[0m_accuracy[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3508:            best_hz = 
signal_best_[7mhorizon[0m_accuracy(min_samples=50)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3509:            for sig_name, bh_data in best_hz.items():[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3510:                if bh_data.get("total", 0) >= 30:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3511:                    # Only override if best-[7mhorizon [0m[7m[0maccuracy [0m[7m[0mis [0m
[7m[0mmeaningfully [0m[7m[0mbetter[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3512: [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m[0mcurrent [0m[7m[0m= [0m[7m[0maccuracy_data.get(sig_name, [0m
[7m[0m{}).get("accuracy", [0m[7m[0m0.5)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3513:                    if bh_data["accuracy"] > current + 0.03:[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3515:        except Exception:
> Q:\finance-analyzer\portfolio\signal_engine.py:3516:            logger.debug("Best-[7mhorizon [0m[7m[0maccuracy [0m[7m[0munavailable", [0m
[7m[0mexc_info=True)[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\signal_engine.py:3517: [0m[7m [0m[7m [0m[7m [0m[7m[0maccuracy_gate [0m[7m[0m= [0m[7m[0msig_cfg.get("accuracy_gate_threshold", [0m
[7m[0mACCURACY_GATE_THRESHOLD)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3518:    max_signals = sig_cfg.get("max_active_signals")[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3548:        max_signals=max_signals,
> Q:\finance-analyzer\portfolio\signal_engine.py:3549:        [7mhorizon[0m=horizon,[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3550:        regime_gated_override=regime_gated_effective,[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3551:        ticker=ticker,[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3568:    # Time-of-day confidence adjustment
> Q:\finance-analyzer\portfolio\signal_engine.py:3569:    tod_factor = _time_of_day_factor([7mhorizon[0m=horizon)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3570:    conf *= tod_factor[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3571:    weighted_conf *= tod_factor[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3586:    extra_info["_regime"] = regime
> Q:\finance-analyzer\portfolio\signal_engine.py:3587:    if [7mhorizon[0m:[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3588:        extra_info["_[7mhorizon[0m"] = horizon[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3589:    extra_info["_weighted_action"] = weighted_action[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3590:    extra_info["_weighted_confidence"] = weighted_conf[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3682:
> Q:\finance-analyzer\portfolio\signal_engine.py:3683:            # H1: Match the [7mhorizon[0m-scoped key written above.[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3684:            _ptc_acc = 
load_cached_accuracy(f"per_ticker_consensus_{acc_[7mhorizon[0m}")[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3685:            if _ptc_acc:[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3686:                _ptc_stats = _ptc_acc.get(ticker, {})[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3704:    # Global confidence cap — calibration data shows >80% 
confidence is
> Q:\finance-analyzer\portfolio\signal_engine.py:3705:    # anti-correlated with accuracy at every [7mhorizon [0m[7m[0m(70-80% [0m
[7m[0mbucket [0m[7m[0mis [0m[7m[0mthe[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3706:    # best performing at 57-59% actual accuracy)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3707:    conf = min(conf, 0.80)[0m
  Q:\finance-analyzer\portfolio\signal_engine.py:3708:
> Q:\finance-analyzer\portfolio\signal_engine.py:3709:    # 3h [7mhorizon[0m: [0m[7m[0mcap [0m[7m[0mconfidence [0m[7m[0mto [0m[7m[0mprevent [0m[7m[0moverconfident [0m
[7m[0mshort-term [0m[7m[0mpredictions[0m
> Q:\finance-analyzer\portfolio\signal_engine.py:3710:    if [7mhorizon[0m in ("3h", "4h"):
> Q:\finance-analyzer\portfolio\signal_engine.py:3711:        from portfolio.short_[7mhorizon[0m import CONFIDENCE_CAP_3H[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3712:        conf = min(conf, CONFIDENCE_CAP_3H)[0m
[7m[0m  Q:\finance-analyzer\portfolio\signal_engine.py:3713:[0m
> Q:\finance-analyzer\portfolio\data_collector.py:1:"""Data collection — Binance, Alpaca, yfinance kline fetchers + 
multi-[7mtimeframe [0m[7m[0mcollector."""[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:2:[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:3:import logging[0m
  Q:\finance-analyzer\portfolio\data_collector.py:23:
> Q:\finance-analyzer\portfolio\data_collector.py:24:# BUG-179: Timeout for parallel [7mtimeframe[0m fetches (seconds)[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:25:_TF_POOL_TIMEOUT = 60[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:26:ALPACA_INTERVAL_MAP = {[0m
  Q:\finance-analyzer\portfolio\data_collector.py:42:
> Q:\finance-analyzer\portfolio\data_collector.py:43:# Multi-[7mtimeframe [0m[7m[0manalysis [0m[7m[0m— [0m[7m[0m(label, [0m[7m[0mbinance_interval, [0m
[7m[0mnum_candles, [0m[7m[0mcache_ttl_seconds)[0m
> Q:\finance-analyzer\portfolio\data_collector.py:44:[7mTIMEFRAME[0mS = [[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:45:    ("Now", "15m", 100, 0),  # ~25h data, refresh every cycle[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:46:    ("12h", "1h", 100, 300),  # ~4d data, cache 5min[0m
  Q:\finance-analyzer\portfolio\data_collector.py:53:
> Q:\finance-analyzer\portfolio\data_collector.py:54:STOCK_[7mTIMEFRAME[0mS = [[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:55:    ("Now", "15m", 100, 0),[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:56:    ("12h", "1h", 100, 300),[0m
  Q:\finance-analyzer\portfolio\data_collector.py:129:            params={
> Q:\finance-analyzer\portfolio\data_collector.py:130:                "[7mtimeframe[0m": alpaca_tf,[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:131:                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:132:                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),[0m
  Q:\finance-analyzer\portfolio\data_collector.py:270:
> Q:\finance-analyzer\portfolio\data_collector.py:271:# --- Multi-[7mtimeframe[0m collector ---[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:272:[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:273:[0m
  Q:\finance-analyzer\portfolio\data_collector.py:279:
> Q:\finance-analyzer\portfolio\data_collector.py:280:def _fetch_one_[7mtimeframe[0m(source, [0m[7m[0msource_key, [0m[7m[0mlabel, [0m[7m[0minterval, [0m
[7m[0mlimit, [0m[7m[0mttl):[0m
> Q:\finance-analyzer\portfolio\data_collector.py:281:    """Fetch and process a single [7mtimeframe[0m. Thread-safe."""[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:282:    cache_key = f"tf_{source_key}_{label}"[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:283:    if ttl > 0:[0m
  Q:\finance-analyzer\portfolio\data_collector.py:314:
> Q:\finance-analyzer\portfolio\data_collector.py:315:def collect_[7mtimeframe[0ms(source):[0m
> Q:\finance-analyzer\portfolio\data_collector.py:316:    """Collect all [7mtimeframe[0ms [0m[7m[0mfor [0m[7m[0ma [0m[7m[0msource, [0m[7m[0mfetching [0m[7m[0min [0m
[7m[0mparallel."""[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:317:    is_stock = "alpaca" in source[0m
> Q:\finance-analyzer\portfolio\data_collector.py:318:    tfs = STOCK_[7mTIMEFRAME[0mS if is_stock else TIMEFRAMES[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\data_collector.py:319: [0m[7m [0m[7m [0m[7m [0m[7m[0msource_key [0m[7m[0m= [0m[7m[0msource.get("alpaca") [0m[7m[0mor [0m[7m[0msource.get("binance") [0m
[7m[0mor [0m[7m[0msource.get("binance_fapi")[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:320:[0m
> Q:\finance-analyzer\portfolio\data_collector.py:321:    # BUG-179: Submit all [7mtimeframe [0m[7m[0mfetches [0m[7m[0mwith [0m[7m[0mtimeout [0m[7m[0mto [0m
[7m[0mprevent [0m[7m[0mhangs[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\data_collector.py:322: [0m[7m [0m[7m [0m[7m [0m[7m[0mwith [0m[7m[0mThreadPoolExecutor(max_workers=len(tfs), [0m
[7m[0mthread_name_prefix=f"tf_{source_key}") [0m[7m[0mas [0m[7m[0mpool:[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:323:        futures = {[0m
> Q:\finance-analyzer\portfolio\data_collector.py:324:            pool.submit(_fetch_one_[7mtimeframe[0m, [0m[7m[0msource, [0m
[7m[0msource_key, [0m[7m[0mlabel, [0m[7m[0minterval, [0m[7m[0mlimit, [0m[7m[0mttl): [0m[7m[0mlabel[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:325:            for label, interval, limit, ttl in tfs[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:326:        }[0m
  Q:\finance-analyzer\portfolio\data_collector.py:334:            stuck = [lbl for f, lbl in futures.items() if not 
f.done()]
> Q:\finance-analyzer\portfolio\data_collector.py:335:            logger.error("BUG-179: [7mTimeframe [0m[7m[0mpool [0m[7m[0mtimeout [0m[7m[0mfor [0m
[7m[0m%s. [0m[7m[0mStuck: [0m[7m[0m%s",[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:336:                         source_key, stuck)[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:337:            for f in futures:[0m
  Q:\finance-analyzer\portfolio\data_collector.py:339:
> Q:\finance-analyzer\portfolio\data_collector.py:340:    # Maintain original [7mtimeframe[0m order[0m
[7m [0m[7m [0m[7m[0mQ:\finance-analyzer\portfolio\data_collector.py:341: [0m[7m [0m[7m [0m[7m [0m[7m[0mtf_order [0m[7m[0m= [0m[7m[0m{label: [0m[7m[0mi [0m[7m[0mfor [0m[7m[0mi, [0m[7m[0m(label, [0m[7m[0m_, [0m[7m[0m_, [0m[7m[0m_) [0m[7m[0min [0m
[7m[0menumerate(tfs)}[0m
[7m[0m  Q:\finance-analyzer\portfolio\data_collector.py:342:    raw_results.sort(key=lambda x: tf_order.get(x[0], 999))[0m
  Q:\finance-analyzer\portfolio\main.py:133:    BINANCE_FAPI_BASE,
> Q:\finance-analyzer\portfolio\main.py:134:    STOCK_[7mTIMEFRAME[0mS,[0m
> Q:\finance-analyzer\portfolio\main.py:135:    [7mTIMEFRAME[0mS,[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:136:    _fetch_klines,[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:137:    alpaca_klines,[0m
  Q:\finance-analyzer\portfolio\main.py:139:    binance_klines,
> Q:\finance-analyzer\portfolio\main.py:140:    collect_[7mtimeframe[0ms,[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:141:    yfinance_klines,[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:142:)[0m
  Q:\finance-analyzer\portfolio\main.py:308:        logger.warning("accuracy degradation import failed: %s", e_deg)
> Q:\finance-analyzer\portfolio\main.py:309:    # 2026-05-04: pre-warm the four dashboard [7mhorizon[0ms (1d/3d/5d/10d)[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:310:    # for /api/accuracy so the first request after a dashboard restart[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:311:    # doesn't spend seconds scanning the signal log. Self-gates to once[0m
  Q:\finance-analyzer\portfolio\main.py:472:    # --- Fully parallel: data collection + signal generation per ticker 
---
> Q:\finance-analyzer\portfolio\main.py:473:    # Each ticker: fetch [7mtimeframe[0ms, [0m[7m[0mcompute [0m[7m[0mindicators, [0m[7m[0mgenerate [0m[7m[0msignals [0m
[7m[0m— [0m[7m[0mall [0m[7m[0mthreaded.[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:474:    # Rate limiters, cache locks, and GPU gate are already thread-safe.[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:475:    from concurrent.futures import ThreadPoolExecutor, as_completed[0m
  Q:\finance-analyzer\portfolio\main.py:482:            t0 = time.monotonic()
> Q:\finance-analyzer\portfolio\main.py:483:            tfs = collect_[7mtimeframe[0ms(source)[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:484:            tf_elapsed = time.monotonic() - t0[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:485:[0m
  Q:\finance-analyzer\portfolio\main.py:502:            action, conf, extra = generate_signal(
> Q:\finance-analyzer\portfolio\main.py:503:                ind, ticker=name, config=config, [7mtimeframe[0ms=tfs, df=now_df[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:504:            )[0m
[7m[0m  Q:\finance-analyzer\portfolio\main.py:505:            sig_elapsed = time.monotonic() - sig_start[0m
  Q:\finance-analyzer\portfolio\analyze.py:142:
> Q:\finance-analyzer\portfolio\analyze.py:143:    # [7mTimeframe[0ms[0m
> Q:\finance-analyzer\portfolio\analyze.py:144:    tf_list = summary.get("[7mtimeframe[0ms", {}).get(ticker, [])[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:145:    if tf_list:[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:146:        lines.append("")[0m
> Q:\finance-analyzer\portfolio\analyze.py:147:        lines.append("[7mTimeframe[0ms:")[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:148:        header = "       "[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:149:        row = f"  {ticker[:5]:<5} "[0m
  Q:\finance-analyzer\portfolio\analyze.py:151:        for tf in tf_list:
> Q:\finance-analyzer\portfolio\analyze.py:152:            h = tf["[7mhorizon[0m"][0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:153:            a = tf.get("action", "HOLD")[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:154:            header += f"{h:>4} "[0m
  Q:\finance-analyzer\portfolio\analyze.py:385:    extra = sig.get("extra", {})
> Q:\finance-analyzer\portfolio\analyze.py:386:    tfs = summary.get("[7mtimeframe[0ms", {}).get(ticker, [])[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:387:    sell_tfs = sum(1 for tf in tfs if tf.get("action") == "SELL")[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:388:    buy_tfs = sum(1 for tf in tfs if tf.get("action") == "BUY")[0m
  Q:\finance-analyzer\portfolio\analyze.py:475:
> Q:\finance-analyzer\portfolio\analyze.py:476:        # [7mTimeframe[0ms[0m
> Q:\finance-analyzer\portfolio\analyze.py:477:        tfs = summary.get("[7mtimeframe[0ms", {}).get(ticker, [])[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:478:        if tfs:[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:479:            tf_tags = " ".join([0m
> Q:\finance-analyzer\portfolio\analyze.py:480:                f"{tf['[7mhorizon[0m']}={'B' [0m[7m[0mif [0m[7m[0mtf.get('action')=='BUY' [0m[7m[0melse [0m
[7m[0m'S' [0m[7m[0mif [0m[7m[0mtf.get('action')=='SELL' [0m[7m[0melse [0m[7m[0m'H'}"[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:481:                for tf in tfs[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:482:            )[0m
  Q:\finance-analyzer\portfolio\analyze.py:517:        "- Weigh higher-accuracy signals more heavily (accuracy % shown 
in parens).",
> Q:\finance-analyzer\portfolio\analyze.py:518:        "- Consider regime + [7mtimeframe [0m[7m[0malignment. [0m[7m[0mALL [0m[7m[0mTFs [0m[7m[0mSELL [0m[7m[0m= [0m[7m[0mget [0m
[7m[0mout.",[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:519:        "",[0m
[7m[0m  Q:\finance-analyzer\portfolio\analyze.py:520:        "For EACH position, respond with exactly this format:",[0m


  ✓ Inspect diff against merge base
  ✓ Analyze changed files for actionable bugs
  → Produce prioritized review findings JSON
codex
The patch introduces signal modules that are unavailable in common deployment setups: `realized_skewness` requires an undeclared SciPy dependency, and Kronos forecasting only works on two hard-coded filesystem layouts when enabled. Those are functional issues, so the patch should not be considered correct as-is.

Full review comments:

- [P2] Avoid host-specific Kronos subprocess paths — Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py:107-111
  If `forecast.kronos_enabled` (or `"shadow"`) is enabled on any machine that is not laid out exactly like `Q:\finance-analyzer`/`Q:\models` or `/home/deck/models`, `_run_kronos()` will spawn a non-existent interpreter/script, trip the breaker, and the forecast signal will never use or backfill Kronos outputs. Because this module explicitly advertises config-only enablement, these paths need to be derived or configured rather than hard-coded to two developer hosts.

- [P2] Declare SciPy for the realized_skewness signal — Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\realized_skewness.py:25-25
  In a clean install built from this repo's declared dependencies, `scipy` is not installed, but this module imports it at import time. Since `realized_skewness` is a registered enhanced signal, `load_signal_func()` will fail to import the module and that signal will be stuck at `HOLD` indefinitely instead of ever contributing a vote.
The patch introduces signal modules that are unavailable in common deployment setups: `realized_skewness` requires an undeclared SciPy dependency, and Kronos forecasting only works on two hard-coded filesystem layouts when enabled. Those are functional issues, so the patch should not be considered correct as-is.

Full review comments:

- [P2] Avoid host-specific Kronos subprocess paths — Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py:107-111
  If `forecast.kronos_enabled` (or `"shadow"`) is enabled on any machine that is not laid out exactly like `Q:\finance-analyzer`/`Q:\models` or `/home/deck/models`, `_run_kronos()` will spawn a non-existent interpreter/script, trip the breaker, and the forecast signal will never use or backfill Kronos outputs. Because this module explicitly advertises config-only enablement, these paths need to be derived or configured rather than hard-coded to two developer hosts.

- [P2] Declare SciPy for the realized_skewness signal — Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\realized_skewness.py:25-25
  In a clean install built from this repo's declared dependencies, `scipy` is not installed, but this module imports it at import time. Since `realized_skewness` is a registered enhanced signal, `load_signal_func()` will fail to import the module and that signal will be stuck at `HOLD` indefinitely instead of ever contributing a vote.
