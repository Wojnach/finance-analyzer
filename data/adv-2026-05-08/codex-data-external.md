OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-data-external
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e0834-23f0-7981-8361-677f1e111af6
--------
user
changes against 'empty-baseline'
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel; git status --short; git diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-data-external
2026-05-08T15:28:23.327186Z ERROR codex_core::tools::router: error=Exit code: 1
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


 exited 1 in 627ms:
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


2026-05-08T15:28:23.352762Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-data-external'
'Q:/finance-analyzer/.worktrees/adv-data-external/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-data-external
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-data-external'
'Q:/finance-analyzer/.worktrees/adv-data-external/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-data-external
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


 exited 1 in 653ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-data-external'
'Q:/finance-analyzer/.worktrees/adv-data-external/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-data-external
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-data-external'
'Q:/finance-analyzer/.worktrees/adv-data-external/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-data-external
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
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-data-external rev-parse --show-toplevel; git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-data-external status --short; git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-data-external diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-data-external diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 657ms:
portfolio/alpha_vantage.py
portfolio/bert_sentiment.py
portfolio/crypto_macro_data.py
portfolio/data_collector.py
portfolio/earnings_calendar.py
portfolio/fear_greed.py
portfolio/futures_data.py
portfolio/fx_rates.py
portfolio/macro_context.py
portfolio/market_health.py
portfolio/news_keywords.py
portfolio/onchain_data.py
portfolio/sentiment.py
portfolio/social_sentiment.py

 succeeded in 693ms:
Q:/finance-analyzer/.worktrees/adv-data-external
 portfolio/alpha_vantage.py     | 321 ++++++++++++++
 portfolio/bert_sentiment.py    | 471 ++++++++++++++++++++
 portfolio/crypto_macro_data.py | 461 ++++++++++++++++++++
 portfolio/data_collector.py    | 343 +++++++++++++++
 portfolio/earnings_calendar.py | 216 +++++++++
 portfolio/fear_greed.py        | 191 ++++++++
 portfolio/futures_data.py      | 234 ++++++++++
 portfolio/fx_rates.py          |  89 ++++
 portfolio/macro_context.py     | 403 +++++++++++++++++
 portfolio/market_health.py     | 512 ++++++++++++++++++++++
 portfolio/news_keywords.py     | 353 +++++++++++++++
 portfolio/onchain_data.py      | 343 +++++++++++++++
 portfolio/sentiment.py         | 964 +++++++++++++++++++++++++++++++++++++++++
 portfolio/social_sentiment.py  | 137 ++++++
 14 files changed, 5038 insertions(+)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -File | Select-Object -ExpandProperty FullName' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\data_collector.py -TotalCount 250" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\sentiment.py -TotalCount 250" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 687ms:
"""Data collection — Binance, Alpaca, yfinance kline fetchers + multi-timeframe collector."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import pandas as pd

import portfolio.shared_state as _ss
from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
from portfolio.circuit_breaker import CircuitBreaker
from portfolio.http_retry import fetch_with_retry
from portfolio.indicators import compute_indicators, technical_signal

logger = logging.getLogger("portfolio.data_collector")

# --- Circuit breakers for each data source ---

binance_spot_cb = CircuitBreaker("binance_spot", failure_threshold=5, recovery_timeout=60)
binance_fapi_cb = CircuitBreaker("binance_fapi", failure_threshold=5, recovery_timeout=60)
alpaca_cb = CircuitBreaker("alpaca", failure_threshold=5, recovery_timeout=60)

# BUG-179: Timeout for parallel timeframe fetches (seconds)
_TF_POOL_TIMEOUT = 60
ALPACA_INTERVAL_MAP = {
    "15m": ("15Min", 5),
    "1h": ("1Hour", 10),
    "1d": ("1Day", 365),
    "1w": ("1Week", 730),
    "1M": ("1Month", 1825),
}

# yfinance interval mapping: our interval → (yf_interval, yf_period)
_YF_INTERVAL_MAP = {
    "15m": ("15m", "5d"),       # yfinance max for intraday <=60d
    "1h": ("1h", "30d"),
    "1d": ("1d", "365d"),
    "1w": ("1wk", "730d"),
    "1M": ("1mo", "1825d"),
}

# Multi-timeframe analysis — (label, binance_interval, num_candles, cache_ttl_seconds)
TIMEFRAMES = [
    ("Now", "15m", 100, 0),  # ~25h data, refresh every cycle
    ("12h", "1h", 100, 300),  # ~4d data, cache 5min
    ("2d", "4h", 100, 900),  # ~17d data, cache 15min
    ("7d", "1d", 100, 3600),  # ~100d data, cache 1hr
    ("1mo", "3d", 100, 14400),  # ~300d data, cache 4hr
    ("3mo", "1w", 100, 43200),  # ~2yr data, cache 12hr
    ("6mo", "1M", 48, 86400),  # ~4yr data, cache 24hr
]

STOCK_TIMEFRAMES = [
    ("Now", "15m", 100, 0),
    ("12h", "1h", 100, 300),
    ("2d", "1h", 48, 900),
    ("7d", "1d", 30, 3600),
    ("1mo", "1d", 100, 3600),
    ("3mo", "1w", 100, 43200),
    ("6mo", "1M", 48, 86400),
]


# --- Binance API ---

_BINANCE_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_buy_vol",
    "taker_buy_quote_vol", "ignore",
]


def _binance_fetch(base_url, cb, label, symbol, interval="5m", limit=100):
    """Shared Binance kline fetcher for spot and FAPI endpoints."""
    if not cb.allow_request():
        logger.warning("Binance %s circuit OPEN — skipping %s", label, symbol)
        raise ConnectionError(f"Binance {label} circuit open for {symbol}")
    try:
        r = fetch_with_retry(
            f"{base_url}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r is None:
            raise ConnectionError(f"Binance {label} klines request failed for {symbol}")
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ConnectionError(
                f"Binance {label} returned empty data for {symbol} {interval}"
            )
        df = pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        cb.record_success()
        return df
    except Exception:
        cb.record_failure()
        raise


def binance_klines(symbol, interval="5m", limit=100):
    return _binance_fetch(BINANCE_BASE, binance_spot_cb, "spot", symbol, interval, limit)


def binance_fapi_klines(symbol, interval="5m", limit=100):
    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
    return _binance_fetch(BINANCE_FAPI_BASE, binance_fapi_cb, "FAPI", symbol, interval, limit)


# --- Alpaca API ---


def alpaca_klines(ticker, interval="1d", limit=100):
    if interval not in ALPACA_INTERVAL_MAP:
        raise ValueError(f"Unsupported Alpaca interval: {interval}")
    if not alpaca_cb.allow_request():
        logger.warning("Alpaca circuit OPEN — skipping %s", ticker)
        raise ConnectionError(f"Alpaca circuit open for {ticker}")
    try:
        alpaca_tf, lookback_days = ALPACA_INTERVAL_MAP[interval]
        end = datetime.now(UTC)
        start = end - pd.Timedelta(days=lookback_days)
        r = fetch_with_retry(
            f"{ALPACA_BASE}/stocks/{ticker}/bars",
            headers=get_alpaca_headers(),
            params={
                "timeframe": alpaca_tf,
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": "iex",
                "adjustment": "split",
            },
            timeout=10,
        )
        if r is None:
            raise ConnectionError(f"Alpaca request failed for {ticker}")
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if not bars:
            raise ValueError(f"No Alpaca data for {ticker} interval={interval}")
        df = pd.DataFrame(bars)
        df = df.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "t": "time",
            }
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["time"])
        alpaca_cb.record_success()
        return df.tail(limit)
    except Exception:
        alpaca_cb.record_failure()
        raise


# --- yfinance API ---


def fetch_vix():
    """Fetch current VIX level via yfinance. Returns dict or None."""
    try:
        import yfinance as yf

        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist is None or hist.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        current = float(last["Close"])
        prev_close = float(prev["Close"])
        change_pct = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # VIX regime classification
        if current >= 30:
            regime_hint = "high-vol"
        elif current >= 20:
            regime_hint = "elevated"
        elif current >= 15:
            regime_hint = "normal"
        else:
            regime_hint = "complacent"

        return {
            "value": round(current, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": round(change_pct, 2),
            "regime_hint": regime_hint,
        }
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)
        return None


def yfinance_klines(ticker, interval="1d", limit=100):
    """Fetch candles via yfinance with extended-hours data (prepost=True).

    Returns a DataFrame matching alpaca_klines() format:
    columns: open, high, low, close, volume, time
    """
    import yfinance as yf

    from portfolio.tickers import YF_MAP

    yf_ticker = YF_MAP.get(ticker, ticker)
    if interval not in _YF_INTERVAL_MAP:
        raise ValueError(f"Unsupported yfinance interval: {interval}")
    yf_interval, yf_period = _YF_INTERVAL_MAP[interval]

    df = yf.download(
        yf_ticker,
        period=yf_period,
        interval=yf_interval,
        prepost=True,
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise ValueError(f"No yfinance data for {yf_ticker} interval={interval}")

    # yfinance returns MultiIndex columns when downloading single ticker too
    # (e.g. ('Close', 'NVDA')); flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df["time"] = df.index
    df = df.reset_index(drop=True)
    return df.tail(limit)


# --- Kline dispatcher ---

 succeeded in 704ms:
Q:\finance-analyzer\.worktrees\adv-data-external\BASELINE.md
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\alpha_vantage.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\crypto_macro_data.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\earnings_calendar.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\fear_greed.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\futures_data.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\fx_rates.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\macro_context.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\market_health.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\news_keywords.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\onchain_data.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\social_sentiment.py

 succeeded in 699ms:
"""Sentiment analysis — multi-model A/B testing with FinGPT, CryptoBERT, and TradingHero.

Runs multiple sentiment models in shadow mode and logs results for comparison.
The primary (voting) model is the legacy model; shadow models are logged to
data/sentiment_ab_log.jsonl for accuracy tracking. Once a shadow model proves
superior (>60% on 200+ samples), it can be promoted.

Models:
  - CryptoBERT: crypto headlines (legacy primary)
  - Trading-Hero-LLM: stock headlines (legacy primary)
  - FinGPT (Finance-Llama-8B / FinGPT-MT): GGUF via llama-cpp-python (shadow)
  - FinBERT: CPU fallback (shadow)

Phase 3B: Cumulative headline clustering — groups related headlines and scores
them as a batch for richer "drumbeat effect" detection.
"""

import json
import logging
import platform
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_json

logger = logging.getLogger("portfolio.sentiment")

CRYPTO_TICKERS = {"BTC", "ETH"}

if platform.system() == "Windows":
    MODELS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
    TRADING_HERO_SCRIPT = r"Q:\models\trading_hero_infer.py"
    FINBERT_SCRIPT = r"Q:\models\finbert_infer.py"
else:
    MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
    TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"
    FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AB_LOG_FILE = DATA_DIR / "sentiment_ab_log.jsonl"

TICKER_CATEGORIES = {
    "BTC": "BTC",
    "ETH": "ETH",
    "XAU": "GOLD",
    "XAG": "SILVER",
    "PLTR": "TECHNOLOGY",
    "NVDA": "TECHNOLOGY",
    "AMD": "TECHNOLOGY",
    "GOOGL": "TECHNOLOGY",
    "AMZN": "TECHNOLOGY",
    "AAPL": "TECHNOLOGY",
    "AVGO": "TECHNOLOGY",
    "META": "TECHNOLOGY",
    "MU": "TECHNOLOGY",
    "SOUN": "TECHNOLOGY",
    "SMCI": "TECHNOLOGY",
    "TSM": "TECHNOLOGY",
    "TTWO": "TECHNOLOGY",
    "VRT": "TECHNOLOGY",
    "LMT": "TECHNOLOGY",
    "MSTR": "TECHNOLOGY",
}

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"


def _is_crypto(ticker):
    return ticker.upper().replace("-USD", "") in CRYPTO_TICKERS


def _fetch_crypto_headlines(ticker="BTC", limit=20, *, cryptocompare_api_key=None):
    category = TICKER_CATEGORIES.get(ticker.upper(), ticker.upper())
    url = f"{CRYPTOCOMPARE_URL}&categories={category}"
    headers = {"User-Agent": "Mozilla/5.0"}
    if cryptocompare_api_key:
        headers["Authorization"] = f"Apikey {cryptocompare_api_key}"
    data = fetch_json(
        url,
        headers=headers,
        timeout=15,
        label="crypto_headlines",
    )
    if data is None:
        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
    if isinstance(data, dict) and data.get("Response") == "Error":
        logger.warning("[CryptoCompare] API error: %s", data.get("Message", "unknown"))
        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
    raw = data.get("Data", [])
    articles = list(raw)[:limit] if isinstance(raw, list) else []
    parsed = [
        {
            "title": a["title"],
            "source": a.get("source", "unknown"),
            "published": datetime.fromtimestamp(
                a["published_on"], tz=UTC
            ).isoformat(),
        }
        for a in articles
    ]
    if not parsed:
        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
    return parsed


# Mapping from short crypto ticker to yfinance symbol for fallback
_CRYPTO_YFINANCE_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD"}


def _fetch_crypto_headlines_yahoo_fallback(ticker, limit=20):
    """Fallback: fetch crypto headlines via yfinance when CryptoCompare fails."""
    yf_symbol = _CRYPTO_YFINANCE_MAP.get(ticker.upper())
    if not yf_symbol:
        return []
    try:
        articles = _fetch_yahoo_headlines(yf_symbol, limit=limit)
        if articles:
            logger.info("[CryptoCompare] fallback to Yahoo Finance for %s: %d articles",
                        ticker, len(articles))
        return articles
    except Exception as e:
        logger.debug("[Yahoo News] crypto fallback error for %s: %s", ticker, e)
        return []


def _fetch_yahoo_headlines(ticker, limit=10):
    import yfinance as yf

    stock = yf.Ticker(ticker)
    news = stock.news or []
    articles = []
    for item in news[:limit]:
        content = item.get("content", item)
        title = content.get("title", "")
        if not title:
            continue
        pub = content.get("pubDate") or content.get("displayTime", "")
        provider = content.get("provider", {})
        source = (
            provider.get("displayName", "Yahoo Finance")
            if isinstance(provider, dict)
            else "Yahoo Finance"
        )
        articles.append(
            {
                "title": title,
                "source": source,
                "published": pub or datetime.now(UTC).isoformat(),
            }
        )
    return articles


def _fetch_newsapi_headlines(ticker, api_key, limit=10, query=None):
    """Fetch headlines from NewsAPI with optional custom search query."""
    search_q = query or ticker
    data = fetch_json(
        "https://newsapi.org/v2/everything",
        params={"q": search_q, "language": "en", "sortBy": "publishedAt",
                "pageSize": limit},
        headers={"User-Agent": "Mozilla/5.0", "X-Api-Key": api_key},
        timeout=15,
        label=f"newsapi:{ticker}",
    )
    if data is None:
        return []
    articles = data.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", "unknown"),
            "published": a.get("publishedAt", datetime.now(UTC).isoformat()),
        }
        for a in articles
        if a.get("title")
    ]


def _fetch_newsapi_with_tracking(ticker, api_key, limit=10, query=None):
    """Fetch from NewsAPI and track the call against daily quota.

    H9/DC-R3-2: only count against budget when the fetch actually returned data
    (not on empty responses or errors), preventing spurious budget exhaustion.
    """
    from portfolio.shared_state import newsapi_track_call
    result = _fetch_newsapi_headlines(ticker, api_key, limit=limit, query=query)
    if result:  # only count against budget when we actually got data
        newsapi_track_call()
    return result


def _fetch_stock_headlines(ticker, newsapi_key=None, limit=20):
    """Fetch stock headlines. NewsAPI for priority tickers (metals), Yahoo for the rest."""
    from portfolio.shared_state import (
        _cached,
        newsapi_quota_ok,
        newsapi_search_query,
        newsapi_ttl_for_ticker,
    )

    articles = []

    # NewsAPI: only for priority tickers during active hours (metals get 20-min refresh)
    ttl = newsapi_ttl_for_ticker(ticker) if newsapi_key else None
    if ttl is not None and newsapi_key and newsapi_quota_ok():
        try:
            query = newsapi_search_query(ticker)
            cached_newsapi = _cached(
                f"newsapi_{ticker}",
                ttl,
                _fetch_newsapi_with_tracking,
                ticker,
                newsapi_key,
                limit,
                query,
            )
            if cached_newsapi:
                articles.extend(cached_newsapi)
        except Exception as e:
            logger.debug("[NewsAPI] error for %s: %s", ticker, e)

    # Yahoo Finance: fallback for metals, primary for everything else
    if len(articles) < limit:
        try:
            remaining = max(0, limit - len(articles))
            if remaining > 0:
                yahoo_articles = _fetch_yahoo_headlines(ticker, limit=remaining)
                seen_titles = {a.get("title", "").lower() for a in articles}
                for ya in yahoo_articles:
                    if ya.get("title", "").lower() not in seen_titles:
                        articles.append(ya)
                        seen_titles.add(ya.get("title", "").lower())
        except Exception as e:
            logger.debug("[Yahoo News] error for %s: %s", ticker, e)

    newsapi_count = len([a for a in articles if a.get("source", "") != "Yahoo Finance"])
    yahoo_count = len(articles) - newsapi_count
    if articles:
        logger.debug("[Headlines %s] %d NewsAPI + %d Yahoo = %d total",
                     ticker, newsapi_count, yahoo_count, len(articles))

    return articles[:limit]


# 2026-04-09 (fix/bert-inproc-gpu): map subprocess script paths to in-process

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\sentiment.py | Select-Object -Skip 250 -First 250" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\data_collector.py | Select-Object -Skip 250 -First 250" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 677ms:


def _fetch_klines(source, interval, limit):
    if "binance_fapi" in source:
        _ss._binance_limiter.wait()
        return binance_fapi_klines(source["binance_fapi"], interval=interval, limit=limit)
    elif "binance" in source:
        _ss._binance_limiter.wait()
        return binance_klines(source["binance"], interval=interval, limit=limit)
    elif "alpaca" in source:
        ticker = source["alpaca"]
        if _ss._current_market_state in ("closed", "weekend", "holiday"):
            logger.debug("%s: using yfinance (market %s)", ticker, _ss._current_market_state)
            _ss._yfinance_limiter.wait()
            return yfinance_klines(ticker, interval=interval, limit=limit)
        _ss._alpaca_limiter.wait()
        return alpaca_klines(ticker, interval=interval, limit=limit)
    raise ValueError(f"Unknown source: {source}")


# --- Multi-timeframe collector ---


# yfinance is not thread-safe; serialize calls with a shared lock.
# H11/DC-R3-4: use the module-level lock from shared_state so all modules
# (fear_greed, golddigger/data_provider, data_collector) share one lock.
from portfolio.shared_state import yfinance_lock as _yfinance_lock


def _fetch_one_timeframe(source, source_key, label, interval, limit, ttl):
    """Fetch and process a single timeframe. Thread-safe."""
    cache_key = f"tf_{source_key}_{label}"
    if ttl > 0:
        with _ss._cache_lock:
            cached = _ss._tool_cache.get(cache_key)
            if cached and time.time() - cached["time"] < ttl:
                return (label, cached["data"])
    try:
        # yfinance is not thread-safe — serialize its calls
        if "alpaca" in source and _ss._current_market_state in ("closed", "weekend", "holiday"):
            with _yfinance_lock:
                df = _fetch_klines(source, interval, limit)
        else:
            df = _fetch_klines(source, interval, limit)
        ind = compute_indicators(df)
        if ind is None:
            logger.debug("%s/%s: insufficient data (%d rows), skipping",
                         source_key, label, len(df) if df is not None else 0)
            return None
        if label == "Now":
            action, conf = None, None
        else:
            action, conf = technical_signal(ind)
        entry = {"indicators": ind, "action": action, "confidence": conf}
        if label == "Now":
            entry["_df"] = df  # preserve raw DataFrame for enhanced signals
        if ttl > 0:
            with _ss._cache_lock:
                _ss._tool_cache[cache_key] = {"data": entry, "time": time.time()}
        return (label, entry)
    except Exception as e:
        return (label, {"error": str(e)})


def collect_timeframes(source):
    """Collect all timeframes for a source, fetching in parallel."""
    is_stock = "alpaca" in source
    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
    source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")

    # BUG-179: Submit all timeframe fetches with timeout to prevent hangs
    with ThreadPoolExecutor(max_workers=len(tfs), thread_name_prefix=f"tf_{source_key}") as pool:
        futures = {
            pool.submit(_fetch_one_timeframe, source, source_key, label, interval, limit, ttl): label
            for label, interval, limit, ttl in tfs
        }
        raw_results = []
        try:
            for future in as_completed(futures, timeout=_TF_POOL_TIMEOUT):
                result = future.result()
                if result is not None:
                    raw_results.append(result)
        except TimeoutError:
            stuck = [lbl for f, lbl in futures.items() if not f.done()]
            logger.error("BUG-179: Timeframe pool timeout for %s. Stuck: %s",
                         source_key, stuck)
            for f in futures:
                f.cancel()

    # Maintain original timeframe order
    tf_order = {label: i for i, (label, _, _, _) in enumerate(tfs)}
    raw_results.sort(key=lambda x: tf_order.get(x[0], 999))
    return raw_results

 succeeded in 703ms:
# model names so _run_model can try the fast in-process path first and fall
# back to the old subprocess path on failure. See portfolio/bert_sentiment.py
# for the full rationale — short version: subprocess cold-load was ~3-10s per
# call, in-process on GPU is ~50-200ms per call, ~20-60x speedup with the
# same output shape.
_INPROC_BERT_MAP = {
    CRYPTOBERT_SCRIPT: "CryptoBERT",
    TRADING_HERO_SCRIPT: "Trading-Hero-LLM",
    FINBERT_SCRIPT: "FinBERT",
}


def _run_model(script, texts):
    """Run a sentiment model.

    Tries the in-process BERT cache first (portfolio.bert_sentiment) because
    it avoids the ~3-10 s subprocess spawn + cold-load cost and runs on GPU
    if available. Falls back to the legacy subprocess pattern on any failure
    so the main loop stays up even if torch/transformers break or a model
    cache dir is missing.
    """
    model_name = _INPROC_BERT_MAP.get(script)
    if model_name is not None:
        try:
            from portfolio.bert_sentiment import predict as _bert_predict
            return _bert_predict(model_name, texts)
        except Exception as e:
            # Log once per (model, exception class) to keep the log clean if
            # we end up stuck on the subprocess fallback. sentiment.py already
            # has its own logger configured.
            logger.warning(
                "In-process BERT %s failed, falling back to subprocess: %s",
                model_name, e,
            )

    # Legacy subprocess path (also used if script is not one of the three
    # known BERT models, though that doesn't happen today).
    proc = subprocess.run(
        [MODELS_PYTHON, script],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Model failed: {proc.stderr}")
    return json.loads(proc.stdout)


# ── Deferred fingpt A/B buffering ──────────────────────────────────────────
# Fingpt is a SHADOW sentiment signal — it never votes. Its output lands in
# data/sentiment_ab_log.jsonl alongside the primary model's vote (CryptoBERT
# for crypto, Trading-Hero-LLM for stocks) for accuracy comparison.
#
# Historical note (2026-04-09, feat/fingpt-in-llmbatch):
# - v1: inline subprocess per call, cold-loading the GGUF every time (70-90s
#   GPU lock holds, broke the cycle budget)
# - v2: warm NDJSON daemon (scripts/fingpt_daemon.py) on GPU full offload
#   (OOM'd with llama-server also resident)
# - v3: warm NDJSON daemon on CPU (60-150s/cycle inference, forced
#   _FINGPT_REQUEST_TIMEOUT_S 60→180 and _TICKER_POOL_TIMEOUT 120→500 hotfix
#   bumps; worked but ugly)
# - v4 (current): fingpt runs in portfolio.llm_batch as Phase 3 of the
#   shared llama_server rotation on port 8787, with full -ngl 99 GPU
#   offload like ministral3 and qwen3. Retires ~250 LOC of daemon +
#   client code.
#
# Because fingpt now runs post-cycle in a batched phase, its results arrive
# AFTER get_sentiment() has already returned to the signal engine. The
# primary model and FinBERT shadow are still computed inline, but their A/B
# log write is DEFERRED: get_sentiment() stashes the primary + finbert shadow
# + the raw headlines + the enqueued fingpt sub_keys into
# _pending_ab_entries[ab_key] and returns. Once flush_llm_batch() completes
# in main.py, sentiment.flush_ab_log() walks the pending entries, merges the
# batched fingpt results into each, and writes the final A/B log rows.
#
# This preserves the EXACT schema of sentiment_ab_log.jsonl that downstream
# accuracy tracking consumes: one row per get_sentiment() call, with a
# shadow[] array containing fingpt per-headline + cumulative + finbert.
#
# Known open issue: sentiment_ab_log.jsonl shows fingpt returning constant
# "neutral, 0.7 confidence" for every real headline — see the
# project_fingpt_parser_defaulting_neutral memory. That is a parser / prompt
# bug in /mnt/q/models/fingpt_infer.py, NOT a problem with this migration.
# Scheduled as the immediate follow-up after this PR merges.

_pending_ab_entries: dict[str, dict] = {}
_pending_ab_lock = threading.Lock()


def _stash_ab_context(
    ab_key: str,
    ticker: str,
    primary_result: dict,
    all_articles: list[dict],
    diss_mult: float,
) -> None:
    """Store the inline portion of an A/B entry until the batched fingpt
    results arrive in flush_ab_log(). Called from get_sentiment().

    Thread-safe — multiple ThreadPoolExecutor workers call this concurrently.

    2026-04-28: cryptobert_shadow slot added; CryptoBERT was demoted from
    crypto primary to shadow. See get_sentiment docstring for rationale.
    """
    with _pending_ab_lock:
        _pending_ab_entries[ab_key] = {
            "ticker": ticker,
            "primary_result": primary_result,
            "finbert_shadow": None,  # filled in below by get_sentiment
            "cryptobert_shadow": None,  # filled in below for crypto tickers (2026-04-28)
            "all_articles": all_articles,
            "diss_mult": diss_mult,
            "fingpt_headlines_raw": None,  # filled in by Phase 3
            "fingpt_cumulatives_raw": {},  # sub_key → raw dict, filled in by Phase 3
        }


def _stash_finbert_shadow(ab_key: str, finbert_shadow: dict | None) -> None:
    """Attach the inline FinBERT shadow result to a pending A/B entry."""
    with _pending_ab_lock:
        entry = _pending_ab_entries.get(ab_key)
        if entry is not None:
            entry["finbert_shadow"] = finbert_shadow


def _stash_cryptobert_shadow(ab_key: str, cryptobert_shadow: dict | None) -> None:
    """Attach the inline CryptoBERT shadow result to a pending A/B entry.

    Added 2026-04-28 when CryptoBERT was demoted from crypto primary to
    shadow. Symmetric with _stash_finbert_shadow.
    """
    with _pending_ab_lock:
        entry = _pending_ab_entries.get(ab_key)
        if entry is not None:
            entry["cryptobert_shadow"] = cryptobert_shadow


def _stash_fingpt_result(ab_key: str, sub_key: str, result) -> None:
    """Called from portfolio.llm_batch._flush_fingpt_phase with the parsed
    fingpt result for one (ab_key, sub_key) tuple.

    sub_key is either "headlines" (result is a list of per-headline dicts)
    or "cumul:<N>" (result is a single cumulative dict).
    """
    with _pending_ab_lock:
        entry = _pending_ab_entries.get(ab_key)
        if entry is None:
            # get_sentiment was never called for this key this cycle — can
            # happen if enqueue_fingpt ran but the parent get_sentiment
            # raised before _stash_ab_context. Drop silently.
            return
        if sub_key == "headlines":
            entry["fingpt_headlines_raw"] = result
        elif sub_key.startswith("cumul:"):
            entry["fingpt_cumulatives_raw"][sub_key] = result


def flush_ab_log() -> None:
    """Walk _pending_ab_entries, merge batched fingpt results into shadow
    arrays, write one JSONL row per entry, and clear the buffer.

    Called once per cycle by main.py immediately after flush_llm_batch()
    finishes Phase 3. Safe to call even if some fingpt results are missing
    (the server returned None for that prompt) — those slots just get
    dropped from the shadow array, same as the daemon-era behavior of
    logging a fingpt:error entry.

    Thread-safe: acquires _pending_ab_lock for the entry snapshot, then
    clears the buffer under the same lock so no subsequent cycle can see
    leftover state.
    """
    with _pending_ab_lock:
        entries_snapshot = dict(_pending_ab_entries)
        _pending_ab_entries.clear()

    if not entries_snapshot:
        return

    for ab_key, entry in entries_snapshot.items():
        try:
            shadow: list[dict] = []

            # Fingpt per-headline → aggregate via _aggregate_sentiments the
            # same way the old inline path did. If the raw list is missing
            # (server returned nothing), skip the entry silently.
            fingpt_raw = entry.get("fingpt_headlines_raw")
            if fingpt_raw:
                # Filter out None entries (per-prompt failures).
                usable = [r for r in fingpt_raw if r is not None]
                if usable:
                    try:
                        fg_overall, fg_avg = _aggregate_sentiments(
                            usable,
                            headlines=entry["all_articles"],
                            dissemination_mult=entry.get("diss_mult", 1.0),
                        )
                        shadow.append({
                            "model": usable[0].get("model", "fingpt:finance-llama-8b"),
                            "sentiment": fg_overall,
                            "confidence": round(fg_avg[fg_overall], 4),
                            "avg_scores": {k: round(v, 4) for k, v in fg_avg.items()},
                        })
                    except Exception:
                        logger.debug(
                            "fingpt headlines aggregation failed for %s", ab_key,
                            exc_info=True,
                        )

            # Fingpt cumulative clusters → one shadow entry per cluster.
            for _sub_key in sorted(entry.get("fingpt_cumulatives_raw", {})):
                cum = entry["fingpt_cumulatives_raw"][_sub_key]
                if cum is None:
                    continue
                shadow.append({
                    "model": cum.get("model", "fingpt:cumulative"),
                    "sentiment": cum.get("sentiment", "neutral"),
                    "confidence": cum.get("confidence", 0.0),
                    "headline_count": cum.get("headline_count", 0),
                })

            # FinBERT shadow (already aggregated inline during get_sentiment).
            finbert = entry.get("finbert_shadow")
            if finbert is not None:
                shadow.append(finbert)

            # CryptoBERT shadow (added 2026-04-28 — was the primary; demoted
            # to shadow due to 99.1% neutral output on press-wire input).
            # Crypto tickers only; entry stays None for stocks.
            cryptobert = entry.get("cryptobert_shadow")
            if cryptobert is not None:
                shadow.append(cryptobert)

            if shadow:
                _log_ab_result(entry["ticker"], entry["primary_result"], shadow)
        except Exception:
            logger.debug("flush_ab_log: entry %s failed", ab_key, exc_info=True)


def _run_finbert(texts):
    """Run FinBERT sentiment inference.

    2026-04-09 (fix/bert-inproc-gpu): routes through _run_model so FinBERT
    also benefits from the in-process GPU cache. _run_model's _INPROC_BERT_MAP
    knows that FINBERT_SCRIPT -> "FinBERT" and will hit bert_sentiment.predict
    first, falling back to the old subprocess path on any exception.
    """
    return _run_model(FINBERT_SCRIPT, texts)



exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content BASELINE.md' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 552ms:
# Empty baseline for adversarial review diffs

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python -m py_compile portfolio\\*.py" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem portfolio -File | Select-Object Name,Length' in Q:\finance-analyzer\.worktrees\adv-data-external
2026-05-08T15:29:03.171602Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[Errno 22] Invalid argument: 'portfolio\\*.py'
 exited 1 in 605ms:
[Errno 22] Invalid argument: 'portfolio\\*.py'
 succeeded in 606ms:

[32;1mName                [0m[32;1m Length[0m
[32;1m----                [0m [32;1m------[0m
alpha_vantage.py      10478
bert_sentiment.py     19504
crypto_macro_data.py  16564
data_collector.py     12435
earnings_calendar.py   7237
fear_greed.py          6928
futures_data.py        7095
fx_rates.py            3573
macro_context.py      13239
market_health.py      17149
news_keywords.py      12100
onchain_data.py       12057
sentiment.py          39408
social_sentiment.py    4718


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'python -m compileall portfolio' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 621ms:
Listing 'portfolio'...
Compiling 'portfolio\\alpha_vantage.py'...
Compiling 'portfolio\\bert_sentiment.py'...
Compiling 'portfolio\\crypto_macro_data.py'...
Compiling 'portfolio\\data_collector.py'...
Compiling 'portfolio\\earnings_calendar.py'...
Compiling 'portfolio\\fear_greed.py'...
Compiling 'portfolio\\futures_data.py'...
Compiling 'portfolio\\fx_rates.py'...
Compiling 'portfolio\\macro_context.py'...
Compiling 'portfolio\\market_health.py'...
Compiling 'portfolio\\news_keywords.py'...
Compiling 'portfolio\\onchain_data.py'...
Compiling 'portfolio\\sentiment.py'...
Compiling 'portfolio\\social_sentiment.py'...

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib
mods = [
    'portfolio.data_collector',
    'portfolio.sentiment',
    'portfolio.market_health',
]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, 'OK')
    except Exception as e:
        print(m, type(e).__name__, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 1249ms:
portfolio.data_collector ModuleNotFoundError No module named 'portfolio.shared_state'
portfolio.sentiment ModuleNotFoundError No module named 'portfolio.file_utils'
portfolio.market_health ModuleNotFoundError No module named 'portfolio.file_utils'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, os, pathlib
root = pathlib.Path('portfolio')
files = sorted(root.glob('*.py'))
all_files = {p.stem for p in files}
missing = {}
for p in files:
    tree = ast.parse(p.read_text(encoding='utf-8'))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('portfolio.'):
                mod = node.module.split('.',1)[1].split('.')[0]
                if mod not in all_files:
                    mods.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('portfolio.'):
                    mod = alias.name.split('.',1)[1].split('.')[0]
                    if mod not in all_files:
                        mods.add(alias.name)
    if mods:
        missing[p.name]=sorted(mods)
for k,v in missing.items():
    print(k, v)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 645ms:
alpha_vantage.py ['portfolio.circuit_breaker', 'portfolio.file_utils', 'portfolio.http_retry', 'portfolio.shared_state', 'portfolio.tickers']
crypto_macro_data.py ['portfolio.file_utils', 'portfolio.http_retry', 'portfolio.shared_state']
data_collector.py ['portfolio.api_utils', 'portfolio.circuit_breaker', 'portfolio.http_retry', 'portfolio.indicators', 'portfolio.shared_state', 'portfolio.tickers']
earnings_calendar.py ['portfolio.api_utils', 'portfolio.http_retry', 'portfolio.shared_state', 'portfolio.tickers']
fear_greed.py ['portfolio.file_utils', 'portfolio.http_retry', 'portfolio.shared_state']
futures_data.py ['portfolio.api_utils', 'portfolio.http_retry', 'portfolio.shared_state']
fx_rates.py ['portfolio.api_utils', 'portfolio.http_retry', 'portfolio.message_store']
macro_context.py ['portfolio.api_utils', 'portfolio.fomc_dates', 'portfolio.golddigger.data_provider', 'portfolio.http_retry', 'portfolio.price_source', 'portfolio.shared_state', 'portfolio.tickers']
market_health.py ['portfolio.file_utils', 'portfolio.price_source', 'portfolio.shared_state']
onchain_data.py ['portfolio.api_utils', 'portfolio.file_utils', 'portfolio.http_retry', 'portfolio.shared_state']
sentiment.py ['portfolio.file_utils', 'portfolio.http_retry', 'portfolio.llm_batch', 'portfolio.shared_state']

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, pathlib
for p in sorted(pathlib.Path('portfolio').glob('*.py')):
    print('\\n###', p.name)
    tree = ast.parse(p.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            print('def', node.name, 'line', node.lineno)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id.isupper() or t.id.startswith('_'):
                        pass
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 684ms:

### alpha_vantage.py
def load_persistent_cache line 36
def _save_persistent_cache line 49
def get_fundamentals line 60
def get_all_fundamentals line 66
def _normalize_overview line 72
def _fetch_overview line 129
def _check_budget line 157
def _is_stale line 171
def _cache_age_hours line 188
def refresh_fundamentals_batch line 204
def should_batch_refresh line 302

### bert_sentiment.py
def _resolve_cache_dir line 121
def _resolve_finbert_snapshot line 125
def _has_meta_tensor line 134
def _accelerate_version line 150
def _model_load_kwargs line 163
def _load_model line 185
def _get_model line 301
def predict line 322
def _predict_batched line 373
def _predict_per_text line 408
def available_models line 456
def is_loaded line 461
def _reset_for_tests line 466

### crypto_macro_data.py
def _fetch_deribit_options line 46
def get_deribit_options line 192
def compute_gold_btc_ratio line 202
def _load_ratio_history line 268
def _append_ratio_history line 292
def get_exchange_netflow_trend line 320
def _load_netflow_history line 390
def _append_netflow_history line 414
def get_crypto_macro_data line 433

### data_collector.py
def _binance_fetch line 74
def binance_klines line 104
def binance_fapi_klines line 108
def alpaca_klines line 116
def fetch_vix line 168
def yfinance_klines line 207
def _fetch_klines line 253
def _fetch_one_timeframe line 280
def collect_timeframes line 315

### earnings_calendar.py
def _fetch_earnings_alpha_vantage line 31
def _fetch_earnings_yfinance line 97
def _fetch_earnings_date line 141
def get_earnings_proximity line 155
def should_gate_earnings line 182
def get_all_earnings_proximity line 199
def clear_cache line 213

### fear_greed.py
def get_sustained_fear_days line 20
def update_fear_streak line 34
def _classify line 83
def get_crypto_fear_greed line 95
def get_stock_fear_greed line 126
def get_fear_greed line 174

### futures_data.py
def _fetch_json line 27
def get_open_interest line 33
def get_open_interest_history line 58
def get_long_short_ratio line 86
def get_top_trader_position_ratio line 115
def get_top_trader_account_ratio line 144
def get_funding_rate_history line 173
def get_all_futures_data line 201

### fx_rates.py
def fetch_usd_sek line 23
def _fx_alert_telegram line 69

### macro_context.py
def _alpaca_headers line 20
def _fetch_dxy line 31
def get_dxy line 65
def _dxy_features_from_close line 78
def _fetch_dxy_intraday line 107
def get_dxy_intraday line 147
def _fetch_klines line 152
def _fetch_volume_signal line 219
def get_volume_signal line 262
def _fred_10y_fallback line 269
def _fetch_treasury line 303
def get_treasury line 356
def get_fed_calendar line 360

### market_health.py
def _fetch_index_data line 53
def count_distribution_days line 98
def detect_ftd_state line 166
def compute_breadth_score line 290
def _classify_zone line 368
def _compute_market_health line 377
def get_market_health line 459
def maybe_refresh_market_health line 473
def get_confidence_penalty line 491

### news_keywords.py
def score_headline line 139
def keyword_severity line 162
def is_credible_source line 174
def dissemination_score line 182
def get_sector_impact line 252
def _ticker_synonym_pattern line 306
def is_relevant_headline line 322

### onchain_data.py
def _coerce_epoch line 29
def _load_config_token line 81
def _save_onchain_cache line 96
def _load_onchain_cache line 105
def _api_get line 124
def _fetch_mvrv line 136
def _fetch_sopr line 147
def _fetch_nupl line 155
def _fetch_realized_price line 163
def _fetch_exchange_netflow line 171
def _fetch_liquidations line 180
def _safe_float line 196
def _fetch_all_onchain line 206
def get_onchain_data line 244
def interpret_onchain line 291

### sentiment.py
def _is_crypto line 73
def _fetch_crypto_headlines line 77
def _fetch_crypto_headlines_yahoo_fallback line 115
def _fetch_yahoo_headlines line 131
def _fetch_newsapi_headlines line 159
def _fetch_newsapi_with_tracking line 184
def _fetch_stock_headlines line 197
def _run_model line 263
def _stash_ab_context line 341
def _stash_finbert_shadow line 369
def _stash_cryptobert_shadow line 377
def _stash_fingpt_result line 389
def flush_ab_log line 409
def _run_finbert line 490
def _cluster_headlines line 505
def _compute_weights line 590
def _aggregate_sentiments line 604
def _majority_label line 656
def _filter_relevant_headlines line 682
def _log_ab_result line 726
def get_sentiment line 748
def get_crypto_sentiment line 935

### social_sentiment.py
def _fetch_subreddit line 29
def _search_subreddit line 59
def get_reddit_posts line 91

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\alpha_vantage.py -TotalCount 250" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\alpha_vantage.py | Select-Object -Skip 250 -First 200" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 625ms:
"""Alpha Vantage fundamentals — daily-refresh stock fundamentals cache.

Fetches OVERVIEW endpoint data for stock tickers and caches persistently.
Free tier: 25 requests/day, 5 requests/minute.

Not used for crypto or metals (no OVERVIEW data available).
"""

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from portfolio.circuit_breaker import CircuitBreaker
from portfolio.file_utils import load_json
from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import _alpha_vantage_limiter
from portfolio.tickers import STOCK_SYMBOLS

logger = logging.getLogger("portfolio.alpha_vantage")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "fundamentals_cache.json"

AV_BASE_URL = "https://www.alphavantage.co/query"

# Module-level state
_cache = {}  # ticker -> normalized fundamentals dict
_cache_lock = threading.Lock()
_daily_budget_used = 0
_budget_reset_date = ""  # ISO date string for budget tracking
_cb = CircuitBreaker("alpha_vantage", failure_threshold=3, recovery_timeout=300)


def load_persistent_cache():
    """Load fundamentals cache from disk on startup."""
    global _cache
    data = load_json(CACHE_FILE)
    if data is None:
        logger.info("No fundamentals cache found at %s", CACHE_FILE)
        return
    if isinstance(data, dict):
        with _cache_lock:
            _cache = data
        logger.info("Loaded fundamentals for %d tickers from cache", len(data))


def _save_persistent_cache():
    """Write current cache to disk atomically."""
    try:
        from portfolio.file_utils import atomic_write_json
        with _cache_lock:
            snapshot = dict(_cache)
        atomic_write_json(CACHE_FILE, snapshot)
    except Exception as e:
        logger.warning("Failed to save fundamentals cache: %s", e)


def get_fundamentals(ticker):
    """Return cached fundamentals for a ticker, or None if not available."""
    with _cache_lock:
        return _cache.get(ticker)


def get_all_fundamentals():
    """Return all cached fundamentals (for reporting)."""
    with _cache_lock:
        return dict(_cache)


def _normalize_overview(raw):
    """Convert Alpha Vantage OVERVIEW response to clean format.

    AV returns "None" strings for missing values and all values as strings.
    """
    def _float(val, default=None):
        if val is None or val == "None" or val == "-" or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _int(val, default=None):
        if val is None or val == "None" or val == "-" or val == "":
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    # Check for error responses
    if "Error Message" in raw or "Note" in raw:
        return None

    # Require at least Symbol to be present
    if not raw.get("Symbol"):
        return None

    result = {
        "pe_ratio": _float(raw.get("PERatio")),
        "forward_pe": _float(raw.get("ForwardPE")),
        "peg_ratio": _float(raw.get("PEGRatio")),
        "eps": _float(raw.get("EPS")),
        "revenue_growth_yoy": _float(raw.get("QuarterlyRevenueGrowthYOY")),
        "earnings_growth_yoy": _float(raw.get("QuarterlyEarningsGrowthYOY")),
        "profit_margin": _float(raw.get("ProfitMargin")),
        "market_cap": _int(raw.get("MarketCapitalization")),
        "sector": raw.get("Sector") if raw.get("Sector") != "None" else None,
        "industry": raw.get("Industry") if raw.get("Industry") != "None" else None,
        "dividend_yield": _float(raw.get("DividendYield")),
        "analyst_target": _float(raw.get("AnalystTargetPrice")),
        "analyst_ratings": {
            "strong_buy": _int(raw.get("AnalystRatingStrongBuy"), 0),
            "buy": _int(raw.get("AnalystRatingBuy"), 0),
            "hold": _int(raw.get("AnalystRatingHold"), 0),
            "sell": _int(raw.get("AnalystRatingSell"), 0),
            "strong_sell": _int(raw.get("AnalystRatingStrongSell"), 0),
        },
        "beta": _float(raw.get("Beta")),
        "w52_high": _float(raw.get("52WeekHigh")),
        "w52_low": _float(raw.get("52WeekLow")),
        "_fetched_at": datetime.now(UTC).isoformat(),
    }
    return result


def _fetch_overview(ticker, api_key):
    """Fetch OVERVIEW data for a single ticker from Alpha Vantage."""
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key,
    }
    resp = fetch_with_retry(
        AV_BASE_URL,
        params=params,
        timeout=15,
        retries=2,
    )
    if resp is None:
        return None
    try:
        data = resp.json()
    except (ValueError, AttributeError):
        return None

    # Rate limit info check
    if isinstance(data, dict) and "Note" in data:
        logger.warning("Alpha Vantage rate limit hit: %s", data["Note"][:100])
        return None

    return data


def _check_budget():
    """Check and reset daily budget counter. Returns current usage count.

    BUG-108: Protected by _cache_lock for thread safety.
    """
    global _daily_budget_used, _budget_reset_date
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    with _cache_lock:
        if _budget_reset_date != today:
            _daily_budget_used = 0
            _budget_reset_date = today
        return _daily_budget_used


def _is_stale(ticker, max_stale_days=5):
    """Check if cached data for ticker is older than max_stale_days."""
    with _cache_lock:
        entry = _cache.get(ticker)
    if not entry:
        return True
    fetched_at = entry.get("_fetched_at")
    if not fetched_at:
        return True
    try:
        fetched_time = datetime.fromisoformat(fetched_at)
        age_seconds = (datetime.now(UTC) - fetched_time).total_seconds()
        return age_seconds > max_stale_days * 86400
    except (ValueError, TypeError):
        return True


def _cache_age_hours(ticker):
    """Return age of cached data in hours, or None if not cached."""
    with _cache_lock:
        entry = _cache.get(ticker)
    if not entry:
        return None
    fetched_at = entry.get("_fetched_at")
    if not fetched_at:
        return None
    try:
        fetched_time = datetime.fromisoformat(fetched_at)
        return (datetime.now(UTC) - fetched_time).total_seconds() / 3600
    except (ValueError, TypeError):
        return None


def refresh_fundamentals_batch(config):
    """Batch-refresh fundamentals for all stock tickers.

    Respects rate limits (5/min) and daily budget (25/day).
    Returns number of successfully refreshed tickers.
    """
    global _daily_budget_used

    av_config = config.get("alpha_vantage", {})
    if not av_config.get("enabled", True):
        logger.info("Alpha Vantage disabled in config")
        return 0

    api_key = av_config.get("api_key", "")
    if not api_key:
        logger.warning("Alpha Vantage API key not configured")
        return 0

    daily_budget = av_config.get("daily_budget", 25)
    skip_tickers = set(av_config.get("skip_tickers", []))
    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)

    # Check circuit breaker
    if not _cb.allow_request():
        logger.info("Alpha Vantage circuit breaker active (%s)", _cb.state.value)
        return 0

    budget_used = _check_budget()
    if budget_used >= daily_budget:
        logger.info("Alpha Vantage daily budget exhausted (%d/%d)", budget_used, daily_budget)
        return 0

    # Build refresh list: stock tickers not recently cached, not skipped
    tickers_to_refresh = []
    for ticker in sorted(STOCK_SYMBOLS):
        if ticker in skip_tickers:
            continue
        age = _cache_age_hours(ticker)
        if age is None or age > cache_ttl_hours:
            tickers_to_refresh.append(ticker)

    if not tickers_to_refresh:
        logger.info("All fundamentals fresh, nothing to refresh")
        return 0

    # Cap by remaining budget
    remaining_budget = daily_budget - budget_used

 succeeded in 647ms:
    tickers_to_refresh = tickers_to_refresh[:remaining_budget]

    logger.info(
        "Refreshing fundamentals for %d tickers (budget: %d/%d used)",
        len(tickers_to_refresh), budget_used, daily_budget,
    )

    success_count = 0
    for ticker in tickers_to_refresh:
        # Rate limit
        _alpha_vantage_limiter.wait()

        try:
            raw = _fetch_overview(ticker, api_key)
            if raw is None:
                _cb.record_failure()
                if not _cb.allow_request():
                    break
                continue

            normalized = _normalize_overview(raw)
            if normalized is None:
                logger.warning("Alpha Vantage: empty/error response for %s", ticker)
                _cb.record_failure()
                if not _cb.allow_request():
                    break
                continue

            with _cache_lock:
                _cache[ticker] = normalized
                _daily_budget_used += 1  # BUG-108: increment under lock
            _cb.record_success()
            success_count += 1
            logger.info("Refreshed fundamentals for %s (PE=%.1f, sector=%s)",
                        ticker,
                        normalized.get("pe_ratio") or 0,
                        normalized.get("sector", "?"))

        except Exception as e:
            logger.warning("Alpha Vantage fetch failed for %s: %s", ticker, e)
            _cb.record_failure()
            if not _cb.allow_request():
                break

    if success_count > 0:
        _save_persistent_cache()
        logger.info("Fundamentals refresh complete: %d/%d succeeded", success_count, len(tickers_to_refresh))

    return success_count


def should_batch_refresh(config):
    """Check if a batch refresh should run now.

    Only runs when cache is >24h old. Designed to be called from the main loop.
    """
    av_config = config.get("alpha_vantage", {})
    if not av_config.get("enabled", True) or not av_config.get("api_key", ""):
        return False

    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)

    # Check if any stock ticker needs refresh
    for ticker in STOCK_SYMBOLS:
        if ticker in set(av_config.get("skip_tickers", [])):
            continue
        age = _cache_age_hours(ticker)
        if age is None or age > cache_ttl_hours:
            return True

    return False

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\bert_sentiment.py -TotalCount 250" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\bert_sentiment.py | Select-Object -Skip 250 -First 250" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 645ms:
"""In-process BERT sentiment inference (CryptoBERT, Trading-Hero-LLM, FinBERT).

2026-04-09 (fix/bert-inproc-gpu): this module replaces the subprocess.run
pattern that portfolio/sentiment.py was using to call three inference scripts
under /mnt/q/models/ — cryptobert_infer.py, trading_hero_infer.py, and
finbert_infer.py.

Why this exists
---------------
The old path spawned a fresh Python subprocess for every sentiment call, and
each subprocess had to:
  1. Start a Python interpreter (~500 ms)
  2. Import torch + transformers (~1.5-2.5 s)
  3. Cold-load the 125M BERT weights from disk (~1-3 s)
  4. Run inference on CPU (~100-300 ms per headline)
That was ~3-10 s per call, of which >90% was fixed spawn overhead. With 10
BERT calls per cycle (2 per ticker * 5 active tickers), we burned 30-100 s
per cycle just in BERT sentiment. And none of the three scripts ever called
`.to("cuda")` despite the main venv having `torch 2.6.0+cu124` with CUDA
available, so everything ran on CPU.

The fix
-------
Load each model lazily on first use, cache (tokenizer, model, device, lock)
in a module-level dict, move the model to CUDA if available, and run forward
passes directly in the caller's process. Subsequent calls skip the load and
just run inference (~5-20 ms per headline on GPU).

Per-model threading.Lock serializes CUDA kernel launches because main.py's
ThreadPoolExecutor(8 workers) can call predict() concurrently. The lock is
per model so CryptoBERT and FinBERT can run in parallel, but two threads
asking for the same model serialize. That's fine — forward pass is ~20 ms
and the subprocess cold-load it replaces was ~3-10 s per thread.

Fallback contract
-----------------
portfolio/sentiment.py wraps every call to this module in try/except and
falls back to the old subprocess path on any exception. That means: if torch
import fails, if a model cache dir is missing, if CUDA OOMs on the load,
we don't crash the main loop — we just lose the speedup for that call and
log a warning.

VRAM budget
-----------
RTX 3080 10 GB. llama-server (ministral3 / qwen3 / fingpt) holds ~5 GB when
its current model is loaded. Three BERT models = ~1.5 GB resident. Total:
~6.5 GB under load. Kronos/Chronos uses its own gpu_gate and is not
co-resident with llama-server. Plenty of headroom.

Not in scope
------------
- Retiring the 3 /mnt/q/models/*_infer.py scripts: they stay as the fallback
  path and as CLI debugging tools. Cost is zero (unchanged, not imported
  unless the subprocess path is triggered).
- GPU gate acquisition around the forward pass: BERT forward pass is ~20 ms
  and llama-server's phase runs post-ticker, so there's no co-residency
  conflict. If that changes, wrap predict() in a gpu_gate context.
- Eager load at module import: keeps import side-effect-free; first call
  absorbs the ~5-10 s cold-load cost.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from typing import Any

logger = logging.getLogger("portfolio.bert_sentiment")


# --- Model configuration ---------------------------------------------------
#
# Each entry mirrors the hardcoded config of the matching subprocess script
# under /mnt/q/models/. Label maps use the same ordering as those scripts so
# the output shape is a drop-in replacement for _run_model's subprocess path.

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "CryptoBERT": {
        "hf_name": "ElKulako/cryptobert",
        "cache_dir_win": r"Q:\models\cryptobert",
        "cache_dir_linux": "/home/deck/models/cryptobert",
        "max_length": 128,
        # CryptoBERT's native labels: {0: Bearish, 1: Neutral, 2: Bullish}.
        # sentiment.py expects positive/negative/neutral, so map at read time.
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
        "local_files_only": True,
    },
    "Trading-Hero-LLM": {
        "hf_name": "fuchenru/Trading-Hero-LLM",
        "cache_dir_win": r"Q:\models\trading-hero-llm",
        "cache_dir_linux": "/home/deck/models/trading-hero-llm",
        "max_length": 512,
        # Trading-Hero-LLM's labels: {0: neutral, 1: positive, 2: negative}.
        "label_map": {0: "neutral", 1: "positive", 2: "negative"},
        "local_files_only": True,
    },
    "FinBERT": {
        "hf_name": "ProsusAI/finbert",
        "cache_dir_win": r"Q:\models\finbert",
        "cache_dir_linux": "/home/deck/models/finbert",
        "max_length": 512,
        # ProsusAI/finbert's labels: {0: positive, 1: negative, 2: neutral}.
        # This matches the hardcoded LABELS list in finbert_infer.py.
        "label_map": {0: "positive", 1: "negative", 2: "neutral"},
        # FinBERT uses a snapshot dir layout (models--ProsusAI--finbert/snapshots/<hash>)
        # instead of the Hub cache layout, so we resolve the path differently.
        "local_files_only": False,
        "snapshot_subdir": "models--ProsusAI--finbert",
    },
}


# Module-level cache: model_name -> (tokenizer, model, device, lock).
# Populated lazily by _get_model. Protected by _init_lock during load.
_models: dict[str, tuple[Any, Any, str, threading.Lock]] = {}
_init_lock = threading.Lock()


def _resolve_cache_dir(config: dict) -> str:
    return config["cache_dir_win"] if platform.system() == "Windows" else config["cache_dir_linux"]


def _resolve_finbert_snapshot(cache_dir: str, subdir: str) -> str | None:
    """FinBERT is saved under cache_dir/models--ProsusAI--finbert/snapshots/<hash>/.
    Return the first snapshot path, or None if the layout doesn't match.
    """
    import glob
    snapshots = glob.glob(os.path.join(cache_dir, subdir, "snapshots", "*"))
    return snapshots[0] if snapshots else None


def _has_meta_tensor(model: Any) -> bool:
    """Walk both `parameters()` and `buffers()` for the meta-tensor check.

    BERT models keep LayerNorm running mean/var (and a few other tensors)
    as buffers, not parameters. The race-induced meta corruption we
    observed at 2026-05-03 23:38 surfaces as "Tensor on device meta..."
    during the forward pass — that error fires whether the offender is
    a parameter OR a buffer. Checking parameters() alone would miss the
    buffer case, so do both.
    """
    return (
        any(p.is_meta for p in model.parameters())
        or any(b.is_meta for b in model.buffers())
    )


def _accelerate_version() -> str | None:
    """Return the installed accelerate version string for diagnostic logging,
    or None if accelerate isn't importable. Used in the meta-tensor retry
    error message so future regressions can be correlated with library
    upgrades.
    """
    try:
        import accelerate
        return getattr(accelerate, "__version__", "unknown")
    except ImportError:
        return None


def _model_load_kwargs(name: str, config: dict, cache_dir: str) -> tuple[str, dict]:
    """Resolve the from_pretrained() positional path + kwargs for `name`.

    Returns (load_path, kwargs). FinBERT uses a snapshot path (no
    cache_dir/local_files_only); others use cache_dir + local_files_only.
    Falls back to the Hub name for FinBERT if no local snapshot exists.

    Extracted 2026-05-04 (fix/bert-meta-tensor) so the same dispatch can
    be reused by the meta-tensor recovery retry path without duplicating
    the FinBERT-vs-others branching.
    """
    if name == "FinBERT":
        snapshot = _resolve_finbert_snapshot(cache_dir, config["snapshot_subdir"])
        if snapshot is not None:
            return snapshot, {}
        return config["hf_name"], {}
    return config["hf_name"], {
        "cache_dir": cache_dir,
        "local_files_only": config.get("local_files_only", False),
    }


def _load_model(name: str) -> tuple[Any, Any, str, threading.Lock]:
    """Load a BERT model + tokenizer. Called under _init_lock.

    Returns (tokenizer, model, device, per_model_lock) where device is
    "cuda" or "cpu".
    """
    # Lazy import so the main loop doesn't pay the torch import cost at
    # startup if no ticker ever calls get_sentiment this run.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = _MODEL_CONFIGS[name]
    cache_dir = _resolve_cache_dir(config)
    hf_name = config["hf_name"]
    load_path, load_kwargs = _model_load_kwargs(name, config, cache_dir)

    if name == "FinBERT" and load_path == hf_name:
        logger.info("Loading BERT model %s via hub name %s (no local snapshot found)", name, hf_name)
    elif name == "FinBERT":
        logger.info("Loading BERT model %s from snapshot %s", name, load_path)
    else:
        logger.info("Loading BERT model %s from %s", name, cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(load_path, **{
        k: v for k, v in load_kwargs.items()
        if k in ("cache_dir", "local_files_only")
    })
    model = AutoModelForSequenceClassification.from_pretrained(load_path, **load_kwargs)

    # 2026-05-04 (fix/bert-meta-tensor): defensive meta-tensor recovery.
    #
    # Race between Chronos's CUDA load and concurrent BERT loads (commit
    # 789cc91c, 2026-05-03 21:08, swapped Chronos/Kronos order so Chronos
    # now loads on the first ticker's forecast call concurrent with the
    # sentiment phase's BERT loads via main.py's ThreadPoolExecutor) can
    # leave some FinBERT weights on the `meta` device when accelerate's
    # lazy init interleaves with CUDA init on another thread. Without
    # this guard, predict-time forward passes silently fail per-text
    # ("Tensor on device meta is not on the expected device cpu!") and
    # the per-text fallback writes a zero-confidence neutral placeholder
    # for every headline, polluting sentiment_ab_log.jsonl until the
    # next process restart.
    #
    # Detection at load time + one retry with eager-init kwargs flips
    # this from silent A/B-log corruption into either a self-healed
    # cycle (warning + clean reload) or a loud RuntimeError that the
    # caller's _get_model() doesn't cache, so subsequent predict calls
    # try again from scratch instead of compounding the corruption.
    #
    # Cost: one is_meta walk over ~200 parameters per load (<1ms).
    # Triggered: only when accelerate's race actually leaves meta tensors,
    # which is rare and load-time-only — never during steady-state
    # inference.
    if _has_meta_tensor(model):
        logger.warning(
            "BERT %s loaded with meta tensors (likely accelerate race with "
            "concurrent CUDA load); retrying with eager init",
            name,
        )
        eager_kwargs = {
            **load_kwargs,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": False,
        }
        model = AutoModelForSequenceClassification.from_pretrained(
            load_path, **eager_kwargs,

 succeeded in 645ms:
        )
        if _has_meta_tensor(model):
            raise RuntimeError(
                f"BERT {name} still has meta tensors after retry "
                f"(accelerate version: {_accelerate_version() or 'not installed'}, "
                f"load_path={load_path!r})"
            )

    # Put the model into inference mode. finbert_infer.py historically
    # uses the equivalent .train(False) spelling — same effect, and we
    # prefer it here because the other spelling collides with an unrelated
    # security-scanner false positive on a substring match.
    model.train(False)

    # 2026-04-09 (hotfix): BERT models now stay on CPU by default.
    #
    # Initial deployment tried to move BERT models to CUDA for ~5-20x per-call
    # inference speedup, but that created a VRAM contention problem with
    # llama-server's model swap phase (LLM batch Phase 1/2/3). The budget:
    #   BERT (3 models) ~1.5 GB + Chronos-2 ~3.5 GB + llama-server 5 GB
    #   = ~10 GB = the entire RTX 3080 10GB budget, no margin.
    # With BERT + Chronos resident, llama-server's 5 GB finance-llama-8b load
    # was timing out / retrying for 200+ s per swap, making cycles LONGER
    # than the pre-migration subprocess baseline. See portfolio.log for the
    # 21:30 (262s) and 21:48 (429s) cycles on 2026-04-09.
    #
    # The main architectural win — removing ~30-60 s/cycle of subprocess
    # spawn + cold-load overhead — does NOT depend on GPU inference. CPU
    # forward pass for a 125M BERT is ~100-300 ms per headline, vs ~5-20 ms
    # on GPU: the GPU speedup only saves ~2-3 s/cycle on top. Not worth the
    # VRAM contention.
    #
    # Set BERT_SENTIMENT_USE_GPU=1 in the environment to opt back in to GPU
    # (e.g. for testing if VRAM pressure has eased by retiring Chronos or
    # similar). Default: CPU.
    use_gpu = os.environ.get("BERT_SENTIMENT_USE_GPU", "").strip() in ("1", "true", "TRUE", "yes")
    device = "cpu"
    if use_gpu and torch.cuda.is_available():
        try:
            model = model.to("cuda")
            device = "cuda"
            logger.info("BERT model %s moved to CUDA (BERT_SENTIMENT_USE_GPU=1)", name)
        except Exception as e:
            logger.warning("BERT model %s failed to move to CUDA, staying on CPU: %s", name, e)
    else:
        logger.info("BERT model %s staying on CPU (default, avoids VRAM contention with llama-server)", name)

    return tokenizer, model, device, threading.Lock()


def _get_model(name: str) -> tuple[Any, Any, str, threading.Lock]:
    """Thread-safe lazy accessor. Loads the model on first call, returns the
    cached tuple on subsequent calls.
    """
    if name not in _MODEL_CONFIGS:
        raise KeyError(f"Unknown BERT model: {name!r}. Known: {list(_MODEL_CONFIGS)}")

    # Fast path: already loaded. Avoid holding _init_lock during forward pass.
    entry = _models.get(name)
    if entry is not None:
        return entry

    # Slow path: acquire init lock, double-check, load.
    with _init_lock:
        entry = _models.get(name)
        if entry is None:
            entry = _load_model(name)
            _models[name] = entry
        return entry


def predict(model_name: str, texts: list[str]) -> list[dict]:
    """Run BERT sentiment inference on a list of texts.

    Returns a list of dicts matching the legacy subprocess output shape:
        [{"text": <str>, "sentiment": "positive"|"negative"|"neutral",
          "confidence": <float>, "scores": {"positive": .., "negative": ..,
          "neutral": ..}}, ...]

    2026-04-09 (hotfix 2): uses BATCHED tokenize + forward pass. The three
    legacy subprocess scripts (cryptobert_infer.py / trading_hero_infer.py
    / finbert_infer.py) all pass the full text list to the tokenizer in
    one call, which gives one forward pass over a padded tensor instead
    of N sequential passes. On CPU the speedup is ~5-10x per call because
    the BERT kernel launch overhead is amortized across the batch.

    If the batched path fails (e.g. OOM on a huge batch, or tokenizer
    edge case), we fall back to a per-text loop so the caller still gets
    one result per input. A final safety net emits a zero-confidence
    neutral placeholder if even the per-text path fails.
    """
    if not texts:
        return []

    # Lazy torch import. If this fails, caller (sentiment.py) catches and
    # falls back to subprocess. Don't try to guard here - let the exception
    # propagate.
    import torch

    tokenizer, model, device, lock = _get_model(model_name)
    config = _MODEL_CONFIGS[model_name]
    max_length = config["max_length"]
    label_map = config["label_map"]

    # Hold the per-model lock for the whole batch. Batched forward pass
    # takes ~100-500 ms on CPU for N<=30 headlines, much better than the
    # N sequential passes the earlier version of this code did.
    with lock:
        try:
            return _predict_batched(
                texts, tokenizer, model, device, max_length, label_map, torch,
            )
        except Exception as e:
            logger.warning(
                "BERT %s batched predict failed, falling back to per-text loop: %s",
                model_name, e,
            )
            return _predict_per_text(
                texts, tokenizer, model, device, max_length, label_map, torch, model_name,
            )


def _predict_batched(texts, tokenizer, model, device, max_length, label_map, torch):
    """Single tokenizer + forward pass over the whole batch."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)  # shape [N, num_labels]

    results: list[dict] = []
    num_labels = len(label_map)
    for i, text in enumerate(texts):
        row = probs[i]
        label_idx = int(torch.argmax(row).item())
        sentiment = label_map[label_idx]
        scores = {
            label_map[j]: float(row[j].item())
            for j in range(num_labels)
        }
        confidence = float(row[label_idx].item())
        results.append({
            "text": text[:100],
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": scores,
        })
    return results


def _predict_per_text(texts, tokenizer, model, device, max_length, label_map, torch, model_name):
    """Fallback: one forward pass per text. Slower but more resilient to
    edge-case failures in the batched path (e.g. OOM on a huge batch or
    tokenizer error on one odd input).
    """
    results: list[dict] = []
    num_labels = len(label_map)
    for text in texts:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            label_idx = int(torch.argmax(probs).item())
            sentiment = label_map[label_idx]
            scores = {
                label_map[i]: float(probs[i].item())
                for i in range(num_labels)
            }
            confidence = float(probs[label_idx].item())
            results.append({
                "text": text[:100],
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": scores,
            })
        except Exception as e:
            logger.warning(
                "BERT %s per-text predict failed for %r: %s",
                model_name, text[:60], e,
            )
            results.append({
                "text": text[:100],
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            })
    return results


def available_models() -> list[str]:
    """Return the list of supported BERT model names (for tests + debugging)."""
    return list(_MODEL_CONFIGS)


def is_loaded(name: str) -> bool:
    """Check whether a model has been lazy-loaded yet (for tests)."""
    return name in _models


def _reset_for_tests() -> None:
    """Drop the model cache. Tests only - don't call this in production code.
    Does not unload torch-held GPU memory; only clears the Python dict.
    """
    with _init_lock:
        _models.clear()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\market_health.py -TotalCount 260" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\market_health.py | Select-Object -Skip 260 -First 260" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 597ms:
"""Market health module — distribution days, FTD detection, breadth score.

Provides market-level context that the signal engine uses to penalize BUY
confidence in unhealthy markets.  All data comes from yfinance (SPY/QQQ),
cached hourly to avoid rate limits.

Key concepts:
- Distribution day (O'Neil): index closes down >=0.2% on higher volume
- Follow-Through Day: >=1.25% gain on day 4+ of rally on higher volume
- Breadth score: composite 0-100 from distribution days, FTD state, SMAs, trend
"""

import logging
from datetime import UTC, datetime

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.market_health")

# Cache TTL: 1 hour — market health doesn't need minute-level freshness
MARKET_HEALTH_TTL = 3600

# Distribution day thresholds (O'Neil standard)
DIST_DAY_PRICE_DROP_PCT = -0.002  # >=0.2% decline
DIST_DAY_ROLLING_WINDOW = 25  # trading days
STALLING_UPPER_RANGE_PCT = 0.25  # top 25% of daily range

# FTD thresholds
FTD_CORRECTION_PCT = -0.05  # 5% drop from high = correction
FTD_MIN_RALLY_DAYS = 4  # FTD cannot occur before day 4
FTD_MIN_GAIN_PCT = 0.0125  # 1.25% gain minimum
FTD_FAILURE_WINDOW = 10  # days after FTD to watch for failure

# Breadth score component weights (sum = 100)
_WEIGHT_DIST_DAYS = 25
_WEIGHT_FTD_STATE = 25
_WEIGHT_SMA200 = 20
_WEIGHT_SMA50 = 15
_WEIGHT_TREND_10D = 15

# Zone thresholds
ZONE_DANGER = 30
ZONE_CAUTION = 50

# State file for FTD state machine persistence
import pathlib

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_STATE_FILE = _DATA_DIR / "market_health_state.json"


def _fetch_index_data(symbol: str, period: str = "90d") -> dict | None:
    """Fetch daily OHLCV for an index via Alpaca (primary) or yfinance (fallback).

    Returns dict with keys: closes, volumes, highs, lows, opens
    as lists of floats (oldest first).
    """
    # Primary: Alpaca — already integrated with rate limiters and circuit breaker
    try:
        from portfolio.data_collector import alpaca_klines
        df = alpaca_klines(symbol, interval="1d", limit=90)
        if df is not None and len(df) >= 10:
            return {
                "closes": df["close"].tolist(),
                "volumes": df["volume"].tolist(),
                "highs": df["high"].tolist(),
                "lows": df["low"].tolist(),
                "opens": df["open"].tolist(),
            }
    except Exception:
        logger.debug("Alpaca fetch failed for %s, trying yfinance fallback", symbol)

    # Fallback path via the canonical price-source router. Handles
    # Alpaca retry + yfinance emergency fallback internally.
    # 2026-04-14: swapped direct yfinance call for price_source router.
    try:
        from portfolio.price_source import fetch_klines

        _LIMIT = {"1d": 2, "5d": 10, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        limit = _LIMIT.get(period, 90)
        hist = fetch_klines(symbol, interval="1d", limit=limit, period=period)
        if hist is None or hist.empty or len(hist) < 10:
            return None

        return {
            "closes": hist["close"].tolist(),
            "volumes": hist["volume"].tolist(),
            "highs": hist["high"].tolist(),
            "lows": hist["low"].tolist(),
            "opens": hist["open"].tolist(),
        }
    except Exception:
        logger.warning("Failed to fetch %s data", symbol, exc_info=True)
        return None


def count_distribution_days(
    closes: list[float],
    volumes: list[float],
    highs: list[float],
    lows: list[float],
    window: int = DIST_DAY_ROLLING_WINDOW,
) -> dict:
    """Count distribution days in the trailing window.

    A distribution day: price closes down >=0.2% AND volume >= previous day.
    A stalling day: price closes in upper 25% of range on higher volume
    (institutions selling into strength — counts as distribution).

    Returns dict with distribution_days count, stalling_days, and detail list.
    """
    if len(closes) < 2:
        return {"distribution_days": 0, "stalling_days": 0, "details": []}

    n = len(closes)
    lookback_start = max(1, n - window)

    dist_days = 0
    stall_days = 0
    details = []

    for i in range(lookback_start, n):
        pct_change = (closes[i] / closes[i - 1]) - 1
        vol_higher = volumes[i] >= volumes[i - 1]
        day_range = highs[i] - lows[i]

        # Distribution day: down >=0.2% on higher volume
        if pct_change <= DIST_DAY_PRICE_DROP_PCT and vol_higher:
            dist_days += 1
            details.append({
                "day_offset": i - n + 1,  # negative offset from today
                "type": "distribution",
                "pct_change": round(pct_change * 100, 2),
            })
        # Stalling day: closes in upper 25% of range on higher volume
        # (but price didn't meaningfully advance — <0.2% gain)
        elif (
            vol_higher
            and day_range > 0
            and 0 <= pct_change < 0.002
            and (closes[i] - lows[i]) / day_range >= (1 - STALLING_UPPER_RANGE_PCT)
        ):
            stall_days += 1
            details.append({
                "day_offset": i - n + 1,
                "type": "stalling",
                "pct_change": round(pct_change * 100, 2),
            })

    return {
        "distribution_days": dist_days,
        "stalling_days": stall_days,
        "total_pressure": dist_days + stall_days,
        "details": details,
    }


# FTD state machine states
STATE_CORRECTING = "correcting"
STATE_RALLY_ATTEMPT = "rally_attempt"
STATE_FTD_CONFIRMED = "ftd_confirmed"
STATE_CONFIRMED_UPTREND = "confirmed_uptrend"


def detect_ftd_state(
    closes: list[float],
    volumes: list[float],
    prev_state: dict | None = None,
) -> dict:
    """Track Follow-Through Day state machine.

    States:
    - correcting: index is in a correction (down >=5% from recent high)
    - rally_attempt: first up day after correction, counting rally days
    - ftd_confirmed: FTD occurred (day 4+ of rally, >=1.25% gain, higher volume)
    - confirmed_uptrend: FTD has held (not undercut rally low within 10 days)

    Args:
        closes: daily close prices (oldest first)
        volumes: daily volumes (oldest first)
        prev_state: previous state dict for continuity (optional)

    Returns:
        dict with state, rally_day, rally_low, ftd_day_offset, etc.
    """
    if len(closes) < 20:
        return {
            "state": STATE_CORRECTING,
            "rally_day": 0,
            "rally_low": 0,
            "recent_high": 0,
            "ftd_day_offset": None,
        }

    # Use last 60 days for analysis
    n = len(closes)

    # Initialize state
    if prev_state:
        state = prev_state.get("state", STATE_CORRECTING)
        rally_day = prev_state.get("rally_day", 0)
        rally_low = prev_state.get("rally_low", 0)
        recent_high = prev_state.get("recent_high", 0)
        ftd_day_offset = prev_state.get("ftd_day_offset")
    else:
        state = STATE_CORRECTING
        rally_day = 0
        rally_low = min(closes[-20:])
        recent_high = max(closes[-60:]) if len(closes) >= 60 else max(closes)
        ftd_day_offset = None

    # Process the most recent day
    today_close = closes[-1]
    today_vol = volumes[-1]
    yesterday_close = closes[-2]
    yesterday_vol = volumes[-2]

    pct_change = (today_close / yesterday_close) - 1

    # Update recent high
    if today_close > recent_high:
        recent_high = today_close

    # Check correction from high
    drawdown = (today_close / recent_high) - 1 if recent_high > 0 else 0

    if state == STATE_CONFIRMED_UPTREND:
        # Check if uptrend is broken (new correction)
        if drawdown <= FTD_CORRECTION_PCT:
            state = STATE_CORRECTING
            rally_day = 0
            rally_low = today_close
            ftd_day_offset = None

    elif state == STATE_FTD_CONFIRMED:
        # Check if FTD holds or fails
        if today_close < rally_low:
            # FTD failed — undercut rally low
            state = STATE_CORRECTING
            rally_day = 0
            rally_low = today_close
            ftd_day_offset = None
        elif ftd_day_offset is not None and (n - 1 - ftd_day_offset) > FTD_FAILURE_WINDOW:
            # FTD has held past the failure window — confirmed uptrend
            state = STATE_CONFIRMED_UPTREND

    elif state == STATE_RALLY_ATTEMPT:
        if today_close < rally_low:
            # Rally failed — undercut rally low
            state = STATE_CORRECTING
            rally_day = 0
            rally_low = today_close
            ftd_day_offset = None
        elif pct_change > 0:
            rally_day += 1
            # Check for FTD
            if (
                rally_day >= FTD_MIN_RALLY_DAYS
                and pct_change >= FTD_MIN_GAIN_PCT

 succeeded in 632ms:
                and today_vol > yesterday_vol
            ):
                state = STATE_FTD_CONFIRMED
                ftd_day_offset = n - 1
        else:
            # Down day during rally — reset rally count but stay in rally_attempt
            # unless we undercut the low (handled above)
            pass

    elif state == STATE_CORRECTING:
        # Update rally low
        if today_close < rally_low or rally_low == 0:
            rally_low = today_close
        # Check for rally attempt start (first up day)
        if pct_change > 0 and drawdown <= FTD_CORRECTION_PCT:
            state = STATE_RALLY_ATTEMPT
            rally_day = 1
            rally_low = min(rally_low, yesterday_close)

    return {
        "state": state,
        "rally_day": rally_day,
        "rally_low": round(rally_low, 2),
        "recent_high": round(recent_high, 2),
        "ftd_day_offset": ftd_day_offset,
        "drawdown_pct": round(drawdown * 100, 2),
    }


def compute_breadth_score(
    dist_data: dict,
    ftd_state: dict,
    closes: list[float],
) -> dict:
    """Compute composite market breadth score (0-100).

    Components:
    - Distribution day severity (25 pts)
    - FTD state (25 pts)
    - SPY vs 200-SMA (20 pts)
    - SPY vs 50-SMA (15 pts)
    - 10-day return direction (15 pts)
    """
    components = {}

    # Component 1: Distribution day severity (fewer = better)
    total_pressure = dist_data.get("total_pressure", 0)
    if total_pressure <= 1:
        components["distribution"] = _WEIGHT_DIST_DAYS
    elif total_pressure <= 3:
        components["distribution"] = int(_WEIGHT_DIST_DAYS * 0.6)
    elif total_pressure <= 5:
        components["distribution"] = int(_WEIGHT_DIST_DAYS * 0.2)
    else:
        components["distribution"] = 0

    # Component 2: FTD state
    state = ftd_state.get("state", STATE_CORRECTING)
    ftd_scores = {
        STATE_CONFIRMED_UPTREND: _WEIGHT_FTD_STATE,
        STATE_FTD_CONFIRMED: int(_WEIGHT_FTD_STATE * 0.8),
        STATE_RALLY_ATTEMPT: int(_WEIGHT_FTD_STATE * 0.4),
        STATE_CORRECTING: 0,
    }
    components["ftd_state"] = ftd_scores.get(state, 0)

    # Component 3: Price vs 200-SMA
    if len(closes) >= 200:
        sma200 = sum(closes[-200:]) / 200
        components["sma200"] = _WEIGHT_SMA200 if closes[-1] > sma200 else 0
    elif len(closes) >= 50:
        # Fallback: use longest available SMA
        sma_n = len(closes)
        sma_val = sum(closes) / sma_n
        components["sma200"] = _WEIGHT_SMA200 if closes[-1] > sma_val else 0
    else:
        components["sma200"] = int(_WEIGHT_SMA200 * 0.5)  # neutral if insufficient data

    # Component 4: Price vs 50-SMA
    if len(closes) >= 50:
        sma50 = sum(closes[-50:]) / 50
        components["sma50"] = _WEIGHT_SMA50 if closes[-1] > sma50 else 0
    else:
        components["sma50"] = int(_WEIGHT_SMA50 * 0.5)

    # Component 5: 10-day return
    if len(closes) >= 11:
        ret_10d = (closes[-1] / closes[-11]) - 1
        if ret_10d > 0.02:
            components["trend_10d"] = _WEIGHT_TREND_10D
        elif ret_10d > 0:
            components["trend_10d"] = int(_WEIGHT_TREND_10D * 0.67)
        elif ret_10d > -0.02:
            components["trend_10d"] = int(_WEIGHT_TREND_10D * 0.33)
        else:
            components["trend_10d"] = 0
    else:
        components["trend_10d"] = int(_WEIGHT_TREND_10D * 0.5)

    score = sum(components.values())

    return {
        "score": score,
        "components": components,
    }


def _classify_zone(score: int) -> str:
    """Classify market health score into zones."""
    if score < ZONE_DANGER:
        return "danger"
    if score < ZONE_CAUTION:
        return "caution"
    return "healthy"


def _compute_market_health() -> dict | None:
    """Full market health computation from live data.

    Fetches SPY and QQQ, computes distribution days, FTD state,
    and breadth score.  Returns the complete health snapshot.
    """
    spy_data = _fetch_index_data("SPY", "90d")
    qqq_data = _fetch_index_data("QQQ", "90d")

    if not spy_data:
        logger.warning("SPY data unavailable — cannot compute market health")
        return None

    # Distribution days for both indices
    spy_dist = count_distribution_days(
        spy_data["closes"], spy_data["volumes"],
        spy_data["highs"], spy_data["lows"],
    )
    qqq_dist = (
        count_distribution_days(
            qqq_data["closes"], qqq_data["volumes"],
            qqq_data["highs"], qqq_data["lows"],
        )
        if qqq_data
        else {"distribution_days": 0, "stalling_days": 0, "total_pressure": 0}
    )

    # FTD state machine (use SPY as primary, persist state)
    prev_ftd = load_json(_STATE_FILE, default={}).get("ftd_state")
    ftd = detect_ftd_state(spy_data["closes"], spy_data["volumes"], prev_ftd)

    # Breadth score (based on SPY — the broad market)
    breadth = compute_breadth_score(spy_dist, ftd, spy_data["closes"])
    score = breadth["score"]
    zone = _classify_zone(score)

    # 10-day return for context
    closes = spy_data["closes"]
    ret_10d = ((closes[-1] / closes[-11]) - 1) * 100 if len(closes) >= 11 else 0

    # SMA status
    spy_above_200 = False
    spy_above_50 = False
    if len(closes) >= 200:
        spy_above_200 = closes[-1] > sum(closes[-200:]) / 200
    if len(closes) >= 50:
        spy_above_50 = closes[-1] > sum(closes[-50:]) / 50

    result = {
        "score": score,
        "zone": zone,
        "distribution_days_spy": spy_dist["distribution_days"],
        "distribution_days_qqq": qqq_dist["distribution_days"],
        "stalling_days_spy": spy_dist["stalling_days"],
        "total_pressure_spy": spy_dist["total_pressure"],
        "total_pressure_qqq": qqq_dist["total_pressure"],
        "ftd_state": ftd["state"],
        "ftd_rally_day": ftd["rally_day"],
        "ftd_drawdown_pct": ftd["drawdown_pct"],
        "spy_above_200sma": spy_above_200,
        "spy_above_50sma": spy_above_50,
        "spy_return_10d_pct": round(ret_10d, 2),
        "components": breadth["components"],
        "updated_at": datetime.now(UTC).isoformat(),
    }

    # Persist FTD state for continuity across restarts
    state_to_save = {
        "ftd_state": {
            "state": ftd["state"],
            "rally_day": ftd["rally_day"],
            "rally_low": ftd["rally_low"],
            "recent_high": ftd["recent_high"],
            "ftd_day_offset": ftd["ftd_day_offset"],
        },
        "last_updated": result["updated_at"],
    }
    atomic_write_json(_STATE_FILE, state_to_save)

    return result


def get_market_health(force: bool = False) -> dict | None:
    """Get cached market health snapshot.  Refreshes hourly.

    Args:
        force: bypass cache and recompute

    Returns:
        Market health dict or None on failure.
    """
    if force:
        return _compute_market_health()
    return _cached("market_health", MARKET_HEALTH_TTL, _compute_market_health)


def maybe_refresh_market_health() -> None:
    """Post-cycle hook: refresh market health if stale.

    Called from main.py's _run_post_cycle().  The _cached() call
    internally skips if data is fresh.
    """
    try:
        health = get_market_health()
        if health:
            logger.debug(
                "Market health: score=%d zone=%s dist_spy=%d ftd=%s",
                health["score"], health["zone"],
                health["distribution_days_spy"], health["ftd_state"],
            )
    except Exception:
        logger.warning("market health refresh failed", exc_info=True)


def get_confidence_penalty(action: str, health: dict | None = None) -> float:
    """Return confidence multiplier based on market health.

    Only penalizes BUY signals.  SELL and HOLD are unaffected.

    Returns:
        Multiplier in range [0.6, 1.1].  1.0 = no change.
    """
    if action != "BUY":
        return 1.0

    if health is None:
        return 1.0  # no data = no penalty

    score = health.get("score", 50)
    if score < ZONE_DANGER:
        return 0.6  # harsh penalty — danger zone
    if score < ZONE_CAUTION:
        return 0.8  # moderate penalty — caution zone
    if score >= 70:
        return 1.1  # slight boost — very healthy
    return 1.0  # healthy — no change

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\fear_greed.py" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\fx_rates.py" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\macro_context.py -TotalCount 260" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 823ms:
"""USD/SEK exchange rate fetching with caching and staleness alerts.

NOTE: Not migrated to shared_state._cached() because this module has custom
staleness behaviour: Telegram alerting when stale, hardcoded 10.50 fallback,
and ConnectionError semantics. These would be lost with _cached().
"""

import logging
import threading
import time

from portfolio.api_utils import load_config as _load_config
from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.fx_rates")

# BUG-215: Thread-safe FX cache. Accessed from 8-worker ThreadPoolExecutor.
_fx_lock = threading.Lock()
_fx_cache = {"rate": None, "time": 0}
_FX_STALE_THRESHOLD = 7200  # 2 hours — warn if FX rate hasn't been refreshed


def fetch_usd_sek():
    now = time.time()
    with _fx_lock:
        cached_rate = _fx_cache["rate"]
        cached_time = _fx_cache["time"]
    if cached_rate and now - cached_time < 900:  # 15 min cache
        return cached_rate
    try:
        r = fetch_with_retry(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "SEK"},
            timeout=10,
        )
        if r is None:
            raise ConnectionError("FX rate request failed after retries")
        r.raise_for_status()
        rate = float(r.json()["rates"]["SEK"])
        # BUG-117: Sanity check — SEK/USD should be in 7-15 range historically.
        # If outside this range, the API may be returning bad data.
        if not (7.0 <= rate <= 15.0):
            logger.error("FX rate %.4f SEK/USD outside sane bounds (7-15) — ignoring", rate)
        else:
            with _fx_lock:
                _fx_cache["rate"] = rate
                _fx_cache["time"] = now
            return rate
    except Exception as e:
        logger.warning("FX rate fetch failed: %s", e)
    # Fallback to stale cached value
    with _fx_lock:
        cached_rate = _fx_cache["rate"]
        cached_time = _fx_cache["time"]
    if cached_rate:
        age_secs = now - cached_time
        if age_secs > _FX_STALE_THRESHOLD:
            logger.warning("Using stale FX rate (%.1fh old)", age_secs / 3600)
            _fx_alert_telegram(age_secs)
        return cached_rate
    # Last resort: hardcoded fallback
    # BUG-117: Use ERROR level — hardcoded rate may be severely stale.
    # Portfolio valuations using this rate could be off by 10-15% if SEK has moved.
    logger.error("Using hardcoded FX fallback rate 10.50 SEK — no cached or live rate available")
    _fx_alert_telegram(None)
    return 10.50


def _fx_alert_telegram(age_secs):
    """Send a one-shot Telegram alert about FX rate issues. Fires at most once per 4h."""
    with _fx_lock:
        last_alert = _fx_cache.get("_last_fx_alert", 0)
    now = time.time()
    if now - last_alert < 14400:  # 4h cooldown between alerts
        return
    try:
        config = _load_config()
        if age_secs is not None:
            msg = f"_FX WARNING: USD/SEK rate is {age_secs / 3600:.1f}h stale. API may be down._"
        else:
            msg = "_FX WARNING: Using hardcoded fallback rate 10.50 SEK. No live or cached rate available._"
        # BUG-105: Route via message store with "error" category so it reaches Telegram.
        # Previously used "fx_alert" which was save-only — user never saw FX warnings.
        from portfolio.message_store import send_or_store
        send_or_store(msg, config, category="error")
        with _fx_lock:
            _fx_cache["_last_fx_alert"] = now
    except Exception as e:
        logger.debug("FX Telegram alert failed: %s", e)

 succeeded in 872ms:
import logging
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.http_retry import fetch_json

logger = logging.getLogger(__name__)

CRYPTO_TICKERS = {"BTC", "ETH", "BTC-USD", "ETH-USD"}

# Sustained fear/greed tracking — used by signal_engine to gate contrarian
# signals during prolonged extreme sentiment regimes (e.g., 46-day fear streaks).
EXTREME_FEAR_THRESHOLD = 20
EXTREME_GREED_THRESHOLD = 80
_BASE_DIR = Path(__file__).resolve().parent.parent
_STREAK_FILE = _BASE_DIR / "data" / "fear_greed_streak.json"


def get_sustained_fear_days() -> int:
    """Return consecutive days the Fear & Greed index has been <= EXTREME_FEAR_THRESHOLD.

    Returns 0 if not in an extreme fear streak, or if tracking data is unavailable.
    """
    try:
        data = load_json(_STREAK_FILE)
        if data and data.get("streak_type") == "extreme_fear":
            return data.get("streak_days", 0)
    except Exception:
        logger.debug("Could not read fear streak file", exc_info=True)
    return 0


def update_fear_streak(fg_value: int) -> dict:
    """Update the sustained fear/greed streak tracker.

    Called after each successful F&G fetch. Persists streak state to disk
    so it survives process restarts.

    H26 fix: Only increments streak_days once per calendar day, not per fetch.
    Previously, 1440 fetches/day inflated streak_days by 1440x.
    """
    data = load_json(_STREAK_FILE, default={}) or {}

    now = datetime.now(UTC)
    now_str = now.isoformat()
    today_str = now.strftime("%Y-%m-%d")
    prev_type = data.get("streak_type", "neutral")
    prev_days = data.get("streak_days", 0)
    last_date = data.get("last_date", "")

    # H26: Only increment streak on date change (not every fetch)
    is_new_day = today_str != last_date

    if fg_value <= EXTREME_FEAR_THRESHOLD:
        if prev_type == "extreme_fear":
            if is_new_day:
                data["streak_days"] = prev_days + 1
        else:
            data = {"streak_type": "extreme_fear", "streak_days": 1,
                    "streak_started": now_str}
    elif fg_value >= EXTREME_GREED_THRESHOLD:
        if prev_type == "extreme_greed":
            if is_new_day:
                data["streak_days"] = prev_days + 1
        else:
            data = {"streak_type": "extreme_greed", "streak_days": 1,
                    "streak_started": now_str}
    else:
        data = {"streak_type": "neutral", "streak_days": 0,
                "streak_started": now_str}

    data["last_value"] = fg_value
    data["last_updated"] = now_str
    data["last_date"] = today_str
    try:
        atomic_write_json(_STREAK_FILE, data)
    except Exception:
        logger.debug("Could not write fear streak file", exc_info=True)
    return data


def _classify(value):
    if value <= 20:
        return "Extreme Fear"
    if value <= 40:
        return "Fear"
    if value <= 60:
        return "Neutral"
    if value <= 80:
        return "Greed"
    return "Extreme Greed"


def get_crypto_fear_greed() -> dict:
    body = fetch_json("https://api.alternative.me/fng/", timeout=10,
                      label="crypto_fear_greed")
    if body is None:
        return None
    # 2026-05-02 (adversarial review 05-01 P1-13 / 04-29 DE-P1-2):
    # alternative.me returns {"data": []} during maintenance windows. The
    # previous unguarded `body["data"][0]` raised IndexError, which then
    # crashed every cycle's fear-greed signal computation silently.
    # Belt-and-braces: also guard against missing/malformed inner fields.
    data_list = body.get("data") if isinstance(body, dict) else None
    if not data_list:
        logger.debug("crypto_fear_greed: API returned no data (maintenance?)")
        return None
    data = data_list[0]
    if not isinstance(data, dict):
        logger.debug("crypto_fear_greed: data[0] not a dict, got %r", type(data))
        return None
    try:
        return {
            "value": int(data["value"]),
            "classification": data["value_classification"],
            "timestamp": datetime.fromtimestamp(
                int(data["timestamp"]), tz=UTC
            ).isoformat(),
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.debug("crypto_fear_greed: malformed entry: %s", e)
        return None


def get_stock_fear_greed() -> dict:
    import pandas as pd
    import yfinance as yf

    from portfolio.shared_state import yfinance_lock

    # H11/DC-R3-4: yfinance is not thread-safe; hold the shared lock
    with yfinance_lock:
        vix = yf.Ticker("^VIX")
        h = vix.history(period="5d")
    if h.empty:
        return None
    # A-DE-4 (2026-04-11): Newer yfinance versions sometimes return columns
    # as a MultiIndex even for single-ticker Ticker.history() calls
    # (especially for index symbols like ^VIX). Without this flatten,
    # `h["Close"]` returns a DataFrame instead of a Series, .iloc[-1]
    # then returns a row of values, and `float(...)` raises TypeError —
    # which the calling code swallows, leaving the stock F&G signal
    # silently dead. Defensive flatten: if MultiIndex, take the level-0
    # names (Open/High/Low/Close/Volume).
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    close_series = h["Close"]
    # Belt-and-suspenders: if Close is somehow still 2-D (duplicate column
    # after the flatten — possible if yfinance returned overlapping levels),
    # squeeze to 1-D before .iloc[-1].
    if hasattr(close_series, "iloc") and getattr(close_series, "ndim", 1) > 1:
        close_series = close_series.iloc[:, 0]
    vix_val = float(close_series.iloc[-1])
    if vix_val >= 40:
        value = 5
    elif vix_val >= 30:
        value = int(5 + (40 - vix_val) * 1.5)
    elif vix_val >= 20:
        value = int(20 + (30 - vix_val) * 3)
    elif vix_val >= 15:
        value = int(50 + (20 - vix_val) * 6)
    else:
        value = int(80 + (15 - vix_val) * 4)
    value = max(0, min(100, value))
    return {
        "value": value,
        "classification": _classify(value),
        "timestamp": datetime.now(UTC).isoformat(),
        "vix": round(vix_val, 2),
    }


def get_fear_greed(ticker=None) -> dict:
    if ticker is None or ticker.upper().replace("-USD", "") in CRYPTO_TICKERS:
        return get_crypto_fear_greed()
    return get_stock_fear_greed()


if __name__ == "__main__":
    print("=== Crypto Fear & Greed ===")
    result = get_crypto_fear_greed()
    print(f"  Value: {result['value']} ({result['classification']})")
    print(f"  Timestamp: {result['timestamp']}")
    print("\n=== Stock Fear & Greed (VIX) ===")
    result = get_stock_fear_greed()
    if result:
        print(f"  Value: {result['value']} ({result['classification']})")
        print(f"  VIX: {result['vix']}")
    else:
        print("  Failed to fetch VIX data")

 succeeded in 858ms:
import logging
from pathlib import Path

import pandas as pd

from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import VOLUME_TTL as _VOLUME_TTL
from portfolio.shared_state import _alpaca_limiter, _binance_limiter, _cached

logger = logging.getLogger("portfolio.macro_context")

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

from datetime import UTC

from portfolio.tickers import TICKER_SOURCE_MAP as TICKER_MAP


def _alpaca_headers():
    return get_alpaca_headers()


DXY_TTL = 3600
# 2026-04-13: Intraday DXY cache is 3 min — 60m bars refresh at each hour
# boundary and we want to re-query shortly after the new bar closes.
DXY_INTRADAY_TTL = 180
TREASURY_TTL_VAL = 3600


def _fetch_dxy():
    """Fetch DXY data.

    2026-04-14: routed via price_source — DXY (DX-Y.NYB) is in the
    yfinance allowed-fallback list (no free real-time alternative),
    but this preserves a single upgrade point for the day a real-time
    DXY feed becomes available.
    """
    from portfolio.price_source import fetch_klines

    h = fetch_klines("DX-Y.NYB", interval="1d", limit=30, period="30d")
    if h is None or h.empty:
        return None

    close = h["close"]
    current = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    pct_5d = (
        float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
    )

    if current > sma20:
        trend = "strong"
    else:
        trend = "weak"

    return {
        "value": round(current, 2),
        "sma20": round(sma20, 2),
        "trend": trend,
        "change_5d_pct": round(pct_5d, 2),
    }


def get_dxy():
    return _cached("dxy", DXY_TTL, _fetch_dxy)


# --- Intraday DXY (2026-04-13) ---------------------------------------------
# The daily _fetch_dxy above feeds a daily-resolution sub-signal inside
# macro_regime. For 1-3h metals prediction (silver had 46.4% consensus
# accuracy at 3h despite DXY's R² ~0.6 inverse correlation with silver),
# we need 60m-bar DXY data. Primary source is yfinance DX-Y.NYB intraday;
# fallback is EURUSD=X which makes up ~57.6% of DXY weight and gives us
# a usable synth when the primary pseudo-ticker's intraday feed hiccups.


def _dxy_features_from_close(close, *, source: str) -> dict | None:
    """Build the intraday DXY payload from a pandas Close series."""
    import math

    if close is None or len(close) < 2:
        return None
    last = float(close.iloc[-1])
    if math.isnan(last):
        return None

    def _pct(periods: int) -> float:
        if len(close) < periods + 1:
            return float("nan")
        prior = float(close.iloc[-1 - periods])
        if prior == 0 or math.isnan(prior):
            return float("nan")
        return (last / prior - 1) * 100

    change_1h = _pct(1)
    change_3h = _pct(3)

    return {
        "value": round(last, 4),
        "change_1h_pct": round(change_1h, 4) if not math.isnan(change_1h) else None,
        "change_3h_pct": round(change_3h, 4) if not math.isnan(change_3h) else None,
        "source": source,
    }


def _fetch_dxy_intraday():
    """Fetch intraday DXY (60m bars). Fallback chain: primary index → EURUSD synth.

    2026-04-14: routed via price_source — DXY and EURUSD=X are both in
    the yfinance allowed-fallback list (no free real-time alternative).
    The router preserves the same fallback logic but centralizes it.
    """
    from portfolio.price_source import fetch_klines

    def _download(ticker: str):
        try:
            df = fetch_klines(ticker, interval="60m", limit=120, period="5d")
            if df is None or df.empty or "close" not in df.columns:
                return None
            return df["close"].dropna()
        except Exception as exc:
            logger.debug("price_source intraday fetch failed for %s: %s", ticker, exc)
            return None

    # Primary: DX-Y.NYB intraday 60m
    close = _download("DX-Y.NYB")
    result = _dxy_features_from_close(close, source="DX-Y.NYB")
    if result is not None and result.get("change_1h_pct") is not None:
        return result

    # Fallback: synthesize from EURUSD=X spot.
    # DXY weights EUR at ~57.6%; the single-factor approximation
    # DXY ≈ c × EURUSD^(-0.576) captures the bulk of DXY's directional
    # variance. The constant 58.0 does NOT match real DXY levels (~99) —
    # it is arbitrary. Only ``change_1h_pct`` / ``change_3h_pct`` from
    # this synth path are usable — the ``value`` field is meaningless.
    # Downstream consumers (signals/dxy_cross_asset.py) only read the
    # change fields, so this is safe.
    eurusd = _download("EURUSD=X")
    if eurusd is None or len(eurusd) == 0:
        return None
    synth = 58.0 * (eurusd ** -0.576)
    return _dxy_features_from_close(synth, source="EURUSD=X-synth")


def get_dxy_intraday():
    """Cached accessor for intraday DXY features."""
    return _cached("dxy_intraday", DXY_INTRADAY_TTL, _fetch_dxy_intraday)


def _fetch_klines(ticker):
    source_type, symbol = TICKER_MAP.get(ticker, (None, None))
    if source_type in ("binance", "binance_fapi"):
        base_url = BINANCE_FAPI_BASE if source_type == "binance_fapi" else BINANCE_BASE
        _binance_limiter.wait()
        r = fetch_with_retry(
            f"{base_url}/klines",
            params={"symbol": symbol, "interval": "15m", "limit": 100},
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_vol",
                "trades",
                "tb",
                "tq",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    elif source_type == "alpaca":
        from datetime import datetime

        _alpaca_limiter.wait()
        end = datetime.now(UTC)
        start = end - pd.Timedelta(days=5)
        r = fetch_with_retry(
            f"{ALPACA_BASE}/stocks/{symbol}/bars",
            headers=_alpaca_headers(),
            params={
                "timeframe": "15Min",
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": "iex",
            },
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df = df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    return None


def _fetch_volume_signal(ticker):
    """Compute volume signal from klines for a single ticker."""
    klines_df = _fetch_klines(ticker)
    if klines_df is None or klines_df.empty:
        return None
    vol = klines_df["volume"].astype(float)
    close = klines_df["close"].astype(float)
    if len(vol) < 2:
        return None
    last_vol = float(vol.iloc[-2])
    avg20 = (
        float(vol.iloc[:-1].rolling(20).mean().iloc[-1])
        if len(vol) >= 22
        else float(vol.iloc[:-1].mean())
    )
    ratio = last_vol / avg20 if avg20 > 0 else 1.0

    # Price direction over last 3 completed candles
    if len(close) >= 5:
        price_change = float(close.iloc[-2] / close.iloc[-5] - 1)
    else:
        price_change = 0.0

    # Volume spike (>1.5x avg) confirms direction
    # No spike = abstain (HOLD)
    if ratio > 1.5:
        if price_change > 0:
            action = "BUY"
        elif price_change < 0:
            action = "SELL"
        else:
            action = "HOLD"
    else:
        action = "HOLD"

    return {
        "ratio": round(ratio, 2),
        "spike": ratio > 1.5,
        "price_change_3": round(price_change * 100, 2),
        "action": action,
    }


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\macro_context.py | Select-Object -Skip 260 -First 200" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 703ms:

def get_volume_signal(ticker):
    return _cached(f"vol_{ticker}", _VOLUME_TTL, _fetch_volume_signal, ticker)


from portfolio.fomc_dates import FOMC_DATES_ISO as FOMC_DATES


def _fred_10y_fallback():
    """FRED DGS10 fallback when yfinance ^TNX fails.

    Added 2026-04-09 after yfinance ^TNX fetch started returning None for
    extended periods (16h stale), triggering `TypeError('NoneType' object is
    not subscriptable')` from this function's callers.

    Returns a dict in the same shape as the yfinance path would for the "10y"
    key — {yield_pct, change_5d} — or None if FRED is also unavailable.
    Reuses `portfolio.golddigger.data_provider.fetch_us10y`, which already
    has its own 1h cache + circuit breaker.
    """
    try:
        import json as _json

        from portfolio.golddigger.data_provider import fetch_us10y
        fred_key = ""
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                fred_key = _json.load(f).get("golddigger", {}).get("fred_api_key", "") or ""
        except Exception:
            return None
        if not fred_key:
            return None
        yield_decimal = fetch_us10y(fred_key, series_id="DGS10")
        if yield_decimal is None:
            return None
        # fetch_us10y returns decimal (0.0425); yfinance path uses pct (4.25).
        return {"yield_pct": round(yield_decimal * 100, 3), "change_5d": 0.0}
    except Exception:
        logger.warning("FRED fallback failed for 10y", exc_info=True)
        return None


def _fetch_treasury():
    """Fetch treasury yield data, with FRED fallback for 10y.

    2026-04-14: routed via price_source. Treasury tickers (^TNX, ^TYX)
    are CBOE-style indices with no free intraday alternative; the router
    sends them through yfinance. 2YY=F is a futures pseudo-ticker also
    with no free alternative.
    """
    from portfolio.price_source import fetch_klines

    tickers = {"10y": "^TNX", "2y": "2YY=F", "30y": "^TYX"}
    result = {}
    for label, sym in tickers.items():
        try:
            h = fetch_klines(sym, interval="1d", limit=30, period="30d")
            if h is None or h.empty:
                continue
            close = h["close"]
            current = float(close.iloc[-1])
            pct_5d = (
                float((close.iloc[-1] / close.iloc[-5] - 1) * 100)
                if len(close) >= 5
                else 0
            )
            result[label] = {
                "yield_pct": round(current, 3),
                "change_5d": round(pct_5d, 2),
            }
        except Exception:
            logger.warning("Treasury fetch failed for %s", label, exc_info=True)

    # FRED fallback for 10y when yfinance ^TNX is down (common symptom:
    # No data / NoneType errors). Other maturities don't have a clean FRED
    # fallback via this helper, so they stay yfinance-only.
    if "10y" not in result:
        fallback = _fred_10y_fallback()
        if fallback is not None:
            logger.info("Treasury 10y: using FRED fallback (%.3f%%)", fallback["yield_pct"])
            result["10y"] = fallback

    if "10y" in result and "2y" in result:
        spread = result["10y"]["yield_pct"] - result["2y"]["yield_pct"]
        result["spread_2s10s"] = round(spread, 3)
        if spread < 0:
            result["curve"] = "inverted"
        elif spread < 0.2:
            result["curve"] = "flat"
        else:
            result["curve"] = "normal"

    return result or None


def get_treasury():
    return _cached("treasury", TREASURY_TTL_VAL, _fetch_treasury)


def get_fed_calendar():
    from datetime import datetime, timedelta

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    upcoming = [d for d in FOMC_DATES if d >= today]
    if not upcoming:
        return None

    next_date = upcoming[0]
    days_until = (
        datetime.strptime(next_date, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")
    ).days

    is_meeting_day = today in FOMC_DATES
    is_day_before = any(
        (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        == today
        for d in FOMC_DATES
    )

    result = {
        "next_fomc": next_date,
        "days_until": days_until,
        "meetings_remaining": len(upcoming) // 2,
    }
    if is_meeting_day:
        result["warning"] = "FOMC meeting TODAY — expect volatility"
    elif is_day_before:
        result["warning"] = "FOMC meeting TOMORROW — positioning risk"
    elif days_until <= 7:
        result["warning"] = f"FOMC in {days_until} days — pre-meeting drift possible"

    return result


if __name__ == "__main__":
    dxy = get_dxy()
    print(f"DXY: {dxy}")
    treasury = get_treasury()
    print(f"Treasury: {treasury}")
    fed = get_fed_calendar()
    print(f"Fed: {fed}")
    for t in list(TICKER_MAP.keys()):
        print(f"{t}: {get_volume_signal(t)}")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\earnings_calendar.py" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\social_sentiment.py" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\futures_data.py" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 739ms:
"""Binance FAPI futures data — open interest, long/short ratios, funding history.

Fetches public endpoints for crypto tickers (BTC-USD, ETH-USD only).
Uses existing infrastructure: fetch_json, _cached, _binance_limiter.
"""

import logging
import time

from portfolio.api_utils import BINANCE_FAPI_BASE, BINANCE_FUTURES_DATA
from portfolio.http_retry import fetch_json
from portfolio.shared_state import _binance_limiter, _cached

logger = logging.getLogger("portfolio.futures_data")

SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

# Cache TTLs
_OI_TTL = 300        # 5 min
_LS_TTL = 300        # 5 min
_FUNDING_TTL = 900   # 15 min


def _fetch_json(url, params=None, timeout=10):
    """Fetch JSON from Binance FAPI with rate limiting and retry."""
    _binance_limiter.wait()
    return fetch_json(url, params=params, timeout=timeout, label="binance_fapi")


def get_open_interest(ticker):
    """Current open interest for a crypto ticker.

    Returns: {oi, oi_usdt, symbol, time} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FAPI_BASE}/openInterest",
            params={"symbol": symbol},
        )
        if data is None:
            return None
        return {
            "oi": float(data["openInterest"]),
            "symbol": data["symbol"],
            "time": data.get("time", int(time.time() * 1000)),
        }

    return _cached(f"futures_oi_{ticker}", _OI_TTL, _fetch)


def get_open_interest_history(ticker, period="5m", limit=30):
    """Historical open interest snapshots.

    Returns: list of {oi, oi_usdt, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/openInterestHist",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "oi": float(d["sumOpenInterest"]),
                "oi_usdt": float(d["sumOpenInterestValue"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_oi_hist_{ticker}_{period}", _OI_TTL, _fetch)


def get_long_short_ratio(ticker, period="5m", limit=30):
    """Global long/short account ratio.

    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_ls_{ticker}_{period}", _LS_TTL, _fetch)


def get_top_trader_position_ratio(ticker, period="5m", limit=30):
    """Top trader long/short position ratio.

    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/topLongShortPositionRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_top_pos_{ticker}_{period}", _LS_TTL, _fetch)


def get_top_trader_account_ratio(ticker, period="5m", limit=30):
    """Top trader long/short account ratio.

    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/topLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_top_acct_{ticker}_{period}", _LS_TTL, _fetch)


def get_funding_rate_history(ticker, limit=100):
    """Historical funding rates.

    Returns: list of {fundingRate, fundingTime, symbol} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FAPI_BASE}/fundingRate",
            params={"symbol": symbol, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "fundingRate": float(d["fundingRate"]),
                "fundingTime": d["fundingTime"],
                "symbol": d.get("symbol", symbol),
            }
            for d in data
        ]

    return _cached(f"futures_funding_hist_{ticker}", _FUNDING_TTL, _fetch)


def get_all_futures_data(ticker):
    """Fetch all futures data for a ticker. Each sub-key can be None on failure.

    Returns: dict with keys: open_interest, oi_history, ls_ratio,
             top_position_ratio, top_account_ratio, funding_history.
    """
    if ticker not in SYMBOL_MAP:
        return None

    return {
        "open_interest": get_open_interest(ticker),
        "oi_history": get_open_interest_history(ticker),
        "ls_ratio": get_long_short_ratio(ticker),
        "top_position_ratio": get_top_trader_position_ratio(ticker),
        "top_account_ratio": get_top_trader_account_ratio(ticker),
        "funding_history": get_funding_rate_history(ticker),
    }


if __name__ == "__main__":
    import json
    for t in ["BTC-USD", "ETH-USD"]:
        print(f"\n=== {t} ===")
        result = get_all_futures_data(t)
        if result is None:
            print("  Not a crypto ticker — skipped")
        else:
            for k, v in result.items():
                if v is None:
                    print(f"  {k}: None (fetch failed)")
                elif isinstance(v, list):
                    print(f"  {k}: {len(v)} entries, latest={v[-1] if v else 'empty'}")
                else:
                    print(f"  {k}: {json.dumps(v, indent=2)}")

 succeeded in 758ms:
"""Earnings calendar — proximity gate to prevent trading near earnings.

Fetches upcoming earnings dates from yfinance for US stock tickers.
When a ticker has earnings within GATE_DAYS, BUY signals are gated to HOLD
to avoid unpredictable binary events.

Cache: per-ticker, 24h TTL (earnings dates don't change intraday).
"""

import logging
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime

from portfolio.tickers import STOCK_SYMBOLS

logger = logging.getLogger("portfolio.earnings_calendar")

# Gate window: force HOLD if earnings within this many calendar days
GATE_DAYS = 2

# Cache TTL: 24 hours (earnings dates are stable)
EARNINGS_CACHE_TTL = 86400

# Per-ticker cache: {ticker: {"data": {...}, "time": epoch}}
_earnings_cache: dict[str, dict] = {}
_earnings_lock = threading.Lock()


def _fetch_earnings_alpha_vantage(ticker: str) -> dict | None:
    """Fetch next earnings date from Alpha Vantage EARNINGS endpoint.

    Uses the already-configured AV API key and rate limiter.
    """
    try:
        from portfolio.api_utils import load_config
        from portfolio.http_retry import fetch_with_retry
        from portfolio.shared_state import _alpha_vantage_limiter

        config = load_config()
        # C9/DC-R3-1: key lives under config["alpha_vantage"]["api_key"], not
        # the flat "alpha_vantage_key" key that doesn't exist.
        api_key = config.get("alpha_vantage", {}).get("api_key", "")
        if not api_key:
            return None

        _alpha_vantage_limiter.wait()
        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
        # because there is no public increment function exported from that module.
        # Known limitation — earnings fetches consume 1 AV call each but are not
        # reflected in the budget tracker.  Each ticker only fetches once per 24h.
        r = fetch_with_retry(
            "https://www.alphavantage.co/query",
            params={
                "function": "EARNINGS",
                "symbol": ticker,
                "apikey": api_key,
            },
            timeout=10,
        )
        if r is None:
            return None
        data = r.json()

        # AV EARNINGS returns quarterlyEarnings and annualEarnings
        quarterly = data.get("quarterlyEarnings", [])
        if not quarterly:
            return None

        today = datetime.now(UTC).date()
        # Find the next upcoming earnings (reportedDate in the future or very recent)
        for q in quarterly:
            rd = q.get("reportedDate")
            if not rd or rd == "None":
                continue
            try:
                from datetime import date as _date
                ed = _date.fromisoformat(rd)
                days_until = (ed - today).days
                if days_until >= -1:
                    return {
                        "earnings_date": ed.isoformat(),
                        "days_until": days_until,
                        "gate_active": 0 <= days_until <= GATE_DAYS,
                        "timing": "unknown",
                    }
            except (ValueError, TypeError):
                continue

        return None
    except Exception:
        logger.debug("Alpha Vantage earnings fetch failed for %s", ticker, exc_info=True)
        return None


def _fetch_earnings_yfinance(ticker: str) -> dict | None:
    """Fallback: fetch next earnings date from yfinance."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        with suppress(Exception):
            cal = t.calendar
            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                    if isinstance(earnings_date, list) and earnings_date:
                        earnings_date = earnings_date[0]
                else:
                    if "Earnings Date" in cal.index:
                        earnings_date = cal.loc["Earnings Date"].iloc[0]
                    else:
                        earnings_date = None

                if earnings_date is not None:
                    if hasattr(earnings_date, "date"):
                        ed = earnings_date.date()
                    elif isinstance(earnings_date, str):
                        ed = datetime.fromisoformat(earnings_date).date()
                    else:
                        ed = None

                    if ed:
                        today = datetime.now(UTC).date()
                        days_until = (ed - today).days
                        if days_until < -5:
                            return None
                        return {
                            "earnings_date": ed.isoformat(),
                            "days_until": days_until,
                            "gate_active": 0 <= days_until <= GATE_DAYS,
                            "timing": "unknown",
                        }
        return None
    except Exception:
        logger.debug("yfinance earnings fetch failed for %s", ticker, exc_info=True)
        return None


def _fetch_earnings_date(ticker: str) -> dict | None:
    """Fetch next earnings date — Alpha Vantage primary, yfinance fallback.

    Returns dict with earnings_date, days_until, timing, or None.
    """
    # Primary: Alpha Vantage (already have API key + rate limiter)
    result = _fetch_earnings_alpha_vantage(ticker)
    if result:
        return result

    # Fallback: yfinance
    return _fetch_earnings_yfinance(ticker)


def get_earnings_proximity(ticker: str) -> dict | None:
    """Get cached earnings proximity for a ticker.

    Only works for STOCK_SYMBOLS — returns None for crypto/metals.

    Returns:
        dict with earnings_date, days_until, gate_active, timing
        or None if no upcoming earnings found.
    """
    if ticker not in STOCK_SYMBOLS:
        return None

    now = time.time()
    with _earnings_lock:
        cached = _earnings_cache.get(ticker)
        if cached and now - cached["time"] < EARNINGS_CACHE_TTL:
            return cached["data"]

    # Fetch fresh
    data = _fetch_earnings_date(ticker)

    with _earnings_lock:
        _earnings_cache[ticker] = {"data": data, "time": now}

    return data


def should_gate_earnings(ticker: str) -> bool:
    """Check if ticker should be gated to HOLD due to nearby earnings.

    Returns True if:
    - Ticker is a US stock (not crypto/metals)
    - Earnings are within GATE_DAYS calendar days
    """
    if ticker not in STOCK_SYMBOLS:
        return False

    prox = get_earnings_proximity(ticker)
    if prox is None:
        return False

    return prox.get("gate_active", False)


def get_all_earnings_proximity() -> dict:
    """Get earnings proximity for all stock tickers.

    Returns dict keyed by ticker with proximity info.
    Used by reporting.py to enrich agent_summary.
    """
    result = {}
    for ticker in STOCK_SYMBOLS:
        prox = get_earnings_proximity(ticker)
        if prox:
            result[ticker] = prox
    return result


def clear_cache() -> None:
    """Clear the earnings cache (for testing)."""
    with _earnings_lock:
        _earnings_cache.clear()

 succeeded in 755ms:
"""Social media sentiment — Reddit headline fetcher.

Uses Reddit's public JSON API, no authentication needed.
"""

import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime

USER_AGENT = "finance-analyzer/1.0 (portfolio intelligence bot)"

# (subreddit, dedicated) — dedicated: keep all posts; general: filter by keywords
TICKER_SUBREDDITS = {
    "BTC": [("Bitcoin", True), ("CryptoCurrency", False)],
    "ETH": [("ethereum", True), ("CryptoCurrency", False)],
    "PLTR": [("PLTR", True), ("wallstreetbets", False)],
    "NVDA": [("wallstreetbets", False), ("stocks", False)],
}

TICKER_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "PLTR": ["palantir", "pltr"],
    "NVDA": ["nvidia", "nvda"],
}


def _fetch_subreddit(sub, keywords, dedicated, per_sub):
    posts = []
    url = f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub + 5}&raw_json=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        title = post.get("title", "").strip()
        if not title or post.get("stickied"):
            continue
        if not dedicated and not any(kw in title.lower() for kw in keywords):
            continue
        created = post.get("created_utc", 0)
        posts.append(
            {
                "title": title,
                "source": f"reddit/r/{sub}",
                "published": (
                    datetime.fromtimestamp(created, tz=UTC).isoformat()
                    if created
                    else datetime.now(UTC).isoformat()
                ),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
            }
        )
    return posts


def _search_subreddit(sub, keywords, limit=10):
    query = urllib.parse.quote(" OR ".join(keywords))
    url = (
        f"https://www.reddit.com/r/{sub}/search.json"
        f"?q={query}&sort=new&restrict_sr=on&limit={limit}&raw_json=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        title = post.get("title", "").strip()
        if not title:
            continue
        created = post.get("created_utc", 0)
        posts.append(
            {
                "title": title,
                "source": f"reddit/r/{sub}",
                "published": (
                    datetime.fromtimestamp(created, tz=UTC).isoformat()
                    if created
                    else datetime.now(UTC).isoformat()
                ),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
            }
        )
    return posts


def get_reddit_posts(ticker, limit=20):
    short = ticker.upper().replace("-USD", "")
    subreddits = TICKER_SUBREDDITS.get(short, [])
    keywords = TICKER_KEYWORDS.get(short, [short.lower()])
    if not subreddits:
        return []

    posts = []
    seen = set()
    per_sub = max(5, limit // len(subreddits))

    for sub, dedicated in subreddits:
        try:
            fetched = _fetch_subreddit(sub, keywords, dedicated, per_sub)
            for p in fetched:
                if p["title"] not in seen:
                    seen.add(p["title"])
                    posts.append(p)
        except Exception as e:
            print(f"    [Reddit r/{sub}] error: {e}")

    # Fallback: if keyword filtering yielded nothing, try search
    if not posts:
        for sub, _dedicated in subreddits[:1]:
            try:
                fetched = _search_subreddit(sub, keywords, limit=per_sub)
                for p in fetched:
                    if p["title"] not in seen:
                        seen.add(p["title"])
                        posts.append(p)
            except Exception as e:
                print(f"    [Reddit search r/{sub}] error: {e}")

    posts.sort(key=lambda p: p.get("score", 0), reverse=True)
    return posts[:limit]


if __name__ == "__main__":
    for ticker in ["BTC", "ETH", "PLTR", "NVDA"]:
        print(f"\n{'='*60}")
        print(f"  Reddit posts for {ticker}")
        print(f"{'='*60}")
        posts = get_reddit_posts(ticker)
        print(f"  Found {len(posts)} posts")
        for p in posts[:5]:
            score = p.get("score", 0)
            print(f"  [{score:>5}] {p['source']:>25}  {p['title'][:70]}")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\crypto_macro_data.py -TotalCount 260" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\crypto_macro_data.py | Select-Object -Skip 260 -First 240" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\onchain_data.py -TotalCount 260" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 818ms:
"""Crypto macro data — options, exchange reserves, ETF flows, gold-BTC ratio.

Fetches data from free, no-auth-required APIs:
  - Deribit public REST API (options: max pain, OI, put/call ratio)
  - BGeometrics exchange netflow trend (already fetched, we track history)
  - Gold-BTC ratio (computed from existing price data)

Cache: 15 min for options (markets move), 1h for ratios.
All fetches use http_retry for resilience.

Usage:
    from portfolio.crypto_macro_data import get_crypto_macro_data
    data = get_crypto_macro_data("BTC-USD")
"""

import json
import logging
import time
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_json
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.crypto_macro_data")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Cache TTLs
OPTIONS_TTL = 900       # 15 min — options data changes slowly
RATIO_TTL = 3600        # 1h — gold/btc ratio
NETFLOW_HIST_TTL = 3600 # 1h — netflow history

# Deribit public API (no auth required)
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

# Persistent netflow history file
NETFLOW_HISTORY_FILE = DATA_DIR / "exchange_netflow_history.jsonl"


# ---------------------------------------------------------------------------
# Deribit Options (free, no auth)
# ---------------------------------------------------------------------------

def _fetch_deribit_options(currency="BTC"):
    """Fetch options book summary from Deribit public API.

    Returns dict with max_pain, total_oi, put_call_ratio, nearest_expiry,
    or None on failure.
    """
    url = f"{DERIBIT_BASE}/get_book_summary_by_currency"
    params = {"currency": currency, "kind": "option"}
    data = fetch_json(url, params=params, timeout=20, retries=2,
                      label="deribit:options_summary")
    if not data:
        return None

    result_list = data.get("result")
    if not result_list or not isinstance(result_list, list):
        return None

    # Parse instrument names: BTC-28MAR26-70000-C
    # Format: {currency}-{expiry}-{strike}-{C|P}
    import datetime
    from collections import defaultdict

    expiry_data = defaultdict(lambda: {"calls": defaultdict(float),
                                        "puts": defaultdict(float),
                                        "total_call_oi": 0.0,
                                        "total_put_oi": 0.0})

    for item in result_list:
        name = item.get("instrument_name", "")
        oi = item.get("open_interest", 0) or 0
        if oi <= 0:
            continue

        parts = name.split("-")
        if len(parts) != 4:
            continue

        _, expiry_str, strike_str, option_type = parts
        try:
            strike = float(strike_str)
        except (ValueError, TypeError):
            continue

        ed = expiry_data[expiry_str]
        if option_type == "C":
            ed["calls"][strike] += oi
            ed["total_call_oi"] += oi
        elif option_type == "P":
            ed["puts"][strike] += oi
            ed["total_put_oi"] += oi

    if not expiry_data:
        return None

    # Find nearest expiry by parsing date strings
    def _parse_expiry(s):
        """Parse Deribit expiry like '28MAR26' to a date."""
        try:
            return datetime.datetime.strptime(s, "%d%b%y").date()
        except ValueError:
            return None

    now = datetime.date.today()
    nearest_expiry = None
    nearest_date = None
    for exp_str in expiry_data:
        d = _parse_expiry(exp_str)
        if d and d >= now and (nearest_date is None or d < nearest_date):
            nearest_date = d
            nearest_expiry = exp_str

    if not nearest_expiry:
        # Fall back to first expiry with most OI
        nearest_expiry = max(expiry_data,
                             key=lambda e: expiry_data[e]["total_call_oi"] +
                                           expiry_data[e]["total_put_oi"])

    ed = expiry_data[nearest_expiry]
    all_strikes = sorted(set(list(ed["calls"].keys()) + list(ed["puts"].keys())))

    if not all_strikes:
        return None

    # Compute max pain: strike where total loss for option buyers is maximized
    # For each candidate strike price:
    #   Call loss = sum of call_oi * max(0, strike - candidate) for all strikes
    #   Put loss = sum of put_oi * max(0, candidate - strike) for all strikes
    #   Total pain = call_loss + put_loss
    # Max pain = strike with highest total pain

    max_pain_strike = None
    max_pain_value = -1

    for candidate in all_strikes:
        total_pain = 0
        for strike in all_strikes:
            call_oi = ed["calls"].get(strike, 0)
            put_oi = ed["puts"].get(strike, 0)
            # Call holder loses if price < strike (ITM calls lose nothing)
            # Actually: option BUYER pain = how much the option expires worthless
            # Max pain for call buyers at candidate: if candidate < strike, call expires OTM
            # Call buyer loss = call_oi * max(0, strike - candidate) ... no wait
            # Standard max pain:
            # For calls at strike K: if expiry price P < K, call expires worthless,
            #   pain = 0 (buyer already lost premium, not counted)
            # For calls at strike K: if P >= K, call is ITM, pain to SELLERS
            # Actually the standard approach:
            # For each candidate expiry price P:
            #   call_pain = sum(call_oi[K] * max(0, P - K)) for all K
            #   put_pain = sum(put_oi[K] * max(0, K - P)) for all K
            #   total = call_pain + put_pain (intrinsic value = money paid out)
            # Max pain = P that MINIMIZES total payout (i.e., max pain for buyers)
            call_pain = call_oi * max(0, candidate - strike)
            put_pain = put_oi * max(0, strike - candidate)
            total_pain += call_pain + put_pain

        # Max pain for buyers = strike where payout is MINIMIZED
        # So we want the candidate with MINIMUM total_pain
        if max_pain_strike is None or total_pain < max_pain_value:
            max_pain_value = total_pain
            max_pain_strike = candidate

    total_call_oi = ed["total_call_oi"]
    total_put_oi = ed["total_put_oi"]
    put_call_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

    # Also compute total OI across ALL expiries
    grand_call_oi = sum(e["total_call_oi"] for e in expiry_data.values())
    grand_put_oi = sum(e["total_put_oi"] for e in expiry_data.values())
    grand_pcr = round(grand_put_oi / grand_call_oi, 3) if grand_call_oi > 0 else None

    days_to_expiry = (nearest_date - now).days if nearest_date else None

    return {
        "max_pain": max_pain_strike,
        "nearest_expiry": nearest_expiry,
        "days_to_expiry": days_to_expiry,
        "nearest_call_oi": total_call_oi,
        "nearest_put_oi": total_put_oi,
        "nearest_pcr": put_call_ratio,
        "total_call_oi": grand_call_oi,
        "total_put_oi": grand_put_oi,
        "total_pcr": grand_pcr,
    }


def get_deribit_options(currency="BTC"):
    """Get Deribit options data with caching."""
    return _cached(f"deribit_options_{currency}", OPTIONS_TTL,
                   _fetch_deribit_options, currency)


# ---------------------------------------------------------------------------
# Gold-BTC Ratio (computed from existing price data)
# ---------------------------------------------------------------------------

def compute_gold_btc_ratio():
    """Compute Gold/BTC ratio from latest agent_summary prices.

    Returns dict with current ratio, 7d/14d/30d history from price snapshots,
    and trend direction.
    """
    try:
        from portfolio.file_utils import load_json
        summary = load_json(DATA_DIR / "agent_summary_compact.json")
        if not summary:
            return None

        signals = summary.get("signals", {})
        btc_price = signals.get("BTC-USD", {}).get("price_usd")
        gold_price = signals.get("XAU-USD", {}).get("price_usd")

        if not btc_price or not gold_price or btc_price <= 0:
            return None

        current_ratio = gold_price / btc_price

        # Try to get historical ratios from price snapshots
        history = _load_ratio_history()
        _append_ratio_history(current_ratio, gold_price, btc_price)

        # Compute trend from history
        trend = "flat"
        ratio_7d = None
        ratio_14d = None

        if history and len(history) >= 2:
            # Get ratio from ~7 days ago (assuming hourly snapshots)
            now = time.time()
            for entry in reversed(history):
                age_days = (now - entry.get("ts", 0)) / 86400
                if age_days >= 7 and ratio_7d is None:
                    ratio_7d = entry.get("ratio")
                if age_days >= 14 and ratio_14d is None:
                    ratio_14d = entry.get("ratio")
                    break

            if ratio_7d:
                change_7d = (current_ratio - ratio_7d) / ratio_7d
                if change_7d > 0.02:
                    trend = "gold_outperforming"  # gold gaining vs BTC
                elif change_7d < -0.02:
                    trend = "btc_outperforming"   # BTC gaining vs gold = rotation
                else:
                    trend = "flat"

        return {
            "gold_btc_ratio": round(current_ratio, 6),
            "gold_price": gold_price,
            "btc_price": btc_price,
            "ratio_7d_ago": ratio_7d,
            "ratio_14d_ago": ratio_14d,
            "trend": trend,
        }
    except Exception:

 succeeded in 809ms:
"""BGeometrics on-chain data integration for Bitcoin.

Fetches MVRV Z-Score, SOPR, NUPL, realized price, exchange netflow,
and liquidation data from the free BGeometrics API (bitcoin-data.com).

Free tier: 8 requests/hour, 15 requests/day.
Budget: 6 metrics x 2 refreshes/day = 12 requests.
Cache: 12 hours per metric (on-chain data doesn't change fast).

Usage:
    from portfolio.onchain_data import get_onchain_data, interpret_onchain
    data = get_onchain_data()  # returns dict or None
    interp = interpret_onchain(data)  # returns interpretation dict
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from portfolio.api_utils import load_config as _load_config
from portfolio.file_utils import atomic_write_json, load_json
from portfolio.http_retry import fetch_json
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.onchain_data")


def _coerce_epoch(value) -> float:
    """A-DE-5 (2026-04-11): Coerce a cache timestamp to a unix-epoch float.

    Older versions of onchain_cache.json stored "ts" as an ISO 8601 string
    instead of an epoch number. The seeding code below does
    `time.time() - cache_ts < TTL` which crashes with TypeError when ts is
    a string. This helper accepts:
        - int / float           → returned as float
        - "1712345678"          → parsed as float
        - "2026-04-11T..."      → parsed via datetime.fromisoformat
        - anything else / fail  → 0.0 (treated as ancient → cache miss)

    Returning 0 on failure is safe: it forces a cache miss, which costs
    one extra API call but never silently breaks the on-chain voter.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            pass
        try:
            # Handle "Z" suffix for UTC zulu time
            iso = value.replace("Z", "+00:00")
            return datetime.fromisoformat(iso).timestamp()
        except (ValueError, TypeError):
            pass
    # P1-14 (2026-05-02): defensive fall-through. Returning 0.0 silently
    # forces a cache miss, which costs one extra API call but never breaks
    # the on-chain voter. Log at DEBUG so operators investigating "why is
    # the BGeometrics 15-req/day budget burning every restart?" can see
    # this firing in the logs without breaking the safe-default behaviour.
    logger.debug(
        "_coerce_epoch: unparseable value type=%s repr=%.50r — returning 0.0",
        type(value).__name__,
        value,
    )
    return 0.0

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "onchain_cache.json"

API_BASE = "https://bitcoin-data.com"
ONCHAIN_TTL = 43200  # 12 hours


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config_token():
    """Load BGeometrics API token from config.json."""
    try:
        config = _load_config()
        token = config.get("bgeometrics", {}).get("api_token", "")
        return token if token else None
    except Exception as e:
        logger.warning("BGeometrics token load failed: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Persistent cache (survives restarts)
# ---------------------------------------------------------------------------

def _save_onchain_cache(data):
    """Save on-chain data to persistent cache file."""
    try:
        # BUG-125: Use atomic_write_json to prevent corrupt cache on crash
        atomic_write_json(CACHE_FILE, data, ensure_ascii=False)
    except Exception:
        logger.warning("Failed to write onchain cache", exc_info=True)


def _load_onchain_cache(max_age_seconds=ONCHAIN_TTL):
    """Load on-chain data from persistent cache if fresh enough."""
    data = load_json(CACHE_FILE)
    if data is None:
        return None
    try:
        ts = data.get("ts", 0)
        if time.time() - ts > max_age_seconds:
            return None
        return data
    except Exception as e:
        logger.warning("On-chain cache load failed: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Individual metric fetchers
# ---------------------------------------------------------------------------

def _api_get(endpoint, token, params=None):
    """Make authenticated GET request to BGeometrics API.

    Skips retries on failure since retrying just burns more of the
    8 req/hour free tier budget.
    """
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}
    return fetch_json(url, headers=headers, params=params, timeout=15,
                      retries=0, label=f"bgeometrics:{endpoint}")


def _fetch_mvrv(token):
    """Fetch latest MVRV and MVRV Z-Score."""
    data = _api_get("/v1/mvrv/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {
        "mvrv": data.get("mvrv"),
        "mvrv_zscore": data.get("mvrvZScore"),
    }


def _fetch_sopr(token):
    """Fetch latest SOPR (Spent Output Profit Ratio)."""
    data = _api_get("/v1/sopr/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {"sopr": data.get("sopr")}


def _fetch_nupl(token):
    """Fetch latest NUPL (Net Unrealized Profit/Loss)."""
    data = _api_get("/v1/nupl/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {"nupl": data.get("nupl")}


def _fetch_realized_price(token):
    """Fetch latest realized price."""
    data = _api_get("/v1/realized-price/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {"realized_price": data.get("realizedPrice")}


def _fetch_exchange_netflow(token):
    """Fetch latest exchange netflow (negative = accumulation)."""
    data = _api_get("/v1/exchange-netflow", token, params={"size": 1})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    latest = data[0] if isinstance(data[0], dict) else data[-1]
    return {"netflow": latest.get("netflow")}


def _fetch_liquidations(token):
    """Fetch latest BTC liquidation data."""
    data = _api_get("/v1/btc-liquidations", token, params={"size": 1})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    latest = data[0] if isinstance(data[0], dict) else data[-1]
    return {
        "long_liquidations": latest.get("longLiquidations"),
        "short_liquidations": latest.get("shortLiquidations"),
    }


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def _safe_float(val):
    """Convert API value to float, handling strings and None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fetch_all_onchain(token):
    """Fetch all 6 on-chain metrics and aggregate into a single dict.

    Adds 1s delay between requests to respect free tier rate limits (8 req/hr).
    """
    result = {"ts": time.time()}

    fetchers = [
        ("mvrv", _fetch_mvrv),
        ("sopr", _fetch_sopr),
        ("nupl", _fetch_nupl),
        ("realized_price", _fetch_realized_price),
        ("exchange_netflow", _fetch_exchange_netflow),
        ("liquidations", _fetch_liquidations),
    ]

    any_success = False
    for i, (name, fetcher) in enumerate(fetchers):
        if i > 0:
            time.sleep(1)  # Rate limit: space out requests
        try:
            data = fetcher(token)
            if data:
                # Convert string values to float
                result.update({k: _safe_float(v) if k != "ts" else v
                              for k, v in data.items()})
                any_success = True
        except Exception:
            logger.warning("BGeometrics %s fetch failed", name, exc_info=True)

    if not any_success:
        return None

    # Save to persistent cache
    _save_onchain_cache(result)
    return result


def get_onchain_data():
    """Get on-chain data for BTC, using in-memory + persistent cache.

    Returns dict with all available metrics, or None if unavailable.
    """
    # H12/DC-R3-5: On restart, seed the in-memory cache from the persistent cache
    # if it is still fresh. This avoids firing 6 BGeometrics API calls immediately
    # after every process restart (burns the 15 req/day budget).
    persistent = load_json(CACHE_FILE, default={})
    if persistent:
        # A-DE-5 (2026-04-11): Defensive timestamp parse. Older versions of
        # this cache stored "ts" as an ISO 8601 string instead of an epoch
        # float. The arithmetic on line below crashes with TypeError when
        # ts is a string, which then propagates up and silently disables
        # the on-chain BTC voter on the next restart. Detect and convert.
        raw_ts = persistent.get("ts", 0) or persistent.get("_fetched_at", 0)
        cache_ts = _coerce_epoch(raw_ts)

 succeeded in 841ms:
        logger.warning("Failed to compute gold/BTC ratio", exc_info=True)
        return None


RATIO_HISTORY_FILE = DATA_DIR / "gold_btc_ratio_history.jsonl"


def _load_ratio_history(max_age_days=30):
    """Load gold/BTC ratio history from JSONL file."""
    try:
        if not RATIO_HISTORY_FILE.exists():
            return []
        cutoff = time.time() - (max_age_days * 86400)
        entries = []
        with open(RATIO_HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("ts", 0) >= cutoff:
                        entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        return entries
    except Exception as e:
        logger.warning("Gold/BTC ratio history load failed: %s", e, exc_info=True)
        return []


def _append_ratio_history(ratio, gold_price, btc_price):
    """Append current ratio to history file (at most once per hour)."""
    try:
        # Check if we already have a recent entry (within 1 hour)
        history = _load_ratio_history(max_age_days=1)
        if history:
            latest_ts = history[-1].get("ts", 0)
            if time.time() - latest_ts < 3600:
                return  # Already have a recent entry

        entry = {
            "ts": time.time(),
            "ratio": round(ratio, 6),
            "gold": gold_price,
            "btc": btc_price,
        }
        atomic_append_jsonl(RATIO_HISTORY_FILE, entry)
    except Exception:
        logger.warning("Failed to append ratio history", exc_info=True)


# ---------------------------------------------------------------------------
# Exchange Netflow Trend (from BGeometrics data we already collect)
# ---------------------------------------------------------------------------

NETFLOW_HISTORY_MAX_DAYS = 30


def get_exchange_netflow_trend():
    """Analyze exchange netflow trend from on-chain data.

    Uses the BGeometrics netflow we already fetch in onchain_data.py.
    Tracks history in a JSONL file to detect multi-day accumulation/distribution.

    Returns dict with trend direction and strength.
    """
    try:
        from portfolio.onchain_data import get_onchain_data
        onchain = get_onchain_data()

        netflow = None
        if onchain:
            netflow = onchain.get("netflow")
            if netflow is not None:
                _append_netflow_history(netflow)

        # Load history and compute trend
        history = _load_netflow_history()
        if not history or len(history) < 3:
            return {
                "current_netflow": netflow,
                "trend": "insufficient_data",
                "consecutive_negative": 0,
                "sum_7d": None,
            }

        now = time.time()
        recent_7d = [e for e in history
                     if now - e.get("ts", 0) < 7 * 86400]
        recent_14d = [e for e in history
                      if now - e.get("ts", 0) < 14 * 86400]

        # Count consecutive negative netflows (accumulation)
        consecutive_neg = 0
        for entry in reversed(history):
            if entry.get("netflow", 0) < 0:
                consecutive_neg += 1
            else:
                break

        sum_7d = sum(e.get("netflow", 0) for e in recent_7d) if recent_7d else None
        sum_14d = sum(e.get("netflow", 0) for e in recent_14d) if recent_14d else None

        # Determine trend
        trend = "neutral"
        if sum_7d is not None:
            if sum_7d < -1000:  # > 1000 BTC net outflow in 7d
                trend = "strong_accumulation"
            elif sum_7d < -100:
                trend = "accumulation"
            elif sum_7d > 1000:
                trend = "strong_distribution"
            elif sum_7d > 100:
                trend = "distribution"

        return {
            "current_netflow": netflow,
            "trend": trend,
            "consecutive_negative": consecutive_neg,
            "sum_7d": round(sum_7d, 1) if sum_7d else None,
            "sum_14d": round(sum_14d, 1) if sum_14d else None,
            "data_points_7d": len(recent_7d),
        }
    except Exception:
        logger.warning("Failed to get netflow trend", exc_info=True)
        return None


def _load_netflow_history():
    """Load netflow history from JSONL file."""
    try:
        if not NETFLOW_HISTORY_FILE.exists():
            return []
        cutoff = time.time() - (NETFLOW_HISTORY_MAX_DAYS * 86400)
        entries = []
        with open(NETFLOW_HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("ts", 0) >= cutoff:
                        entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        return entries
    except Exception as e:
        logger.warning("Exchange netflow history load failed: %s", e, exc_info=True)
        return []


def _append_netflow_history(netflow):
    """Append netflow data point (at most once per 6 hours)."""
    try:
        history = _load_netflow_history()
        if history:
            latest_ts = history[-1].get("ts", 0)
            if time.time() - latest_ts < 21600:  # 6h
                return

        entry = {"ts": time.time(), "netflow": netflow}
        atomic_append_jsonl(NETFLOW_HISTORY_FILE, entry)
    except Exception:
        logger.warning("Failed to append netflow history", exc_info=True)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def get_crypto_macro_data(ticker="BTC-USD"):
    """Get all crypto macro data for a ticker.

    Returns dict with options, gold_btc_ratio, netflow_trend, or None on failure.
    """
    currency = "BTC" if "BTC" in ticker else "ETH" if "ETH" in ticker else None
    if not currency:
        return None

    result = {}

    # Options data (BTC and ETH both available on Deribit)
    options = get_deribit_options(currency)
    if options:
        result["options"] = options

    # Gold-BTC ratio (only meaningful for BTC, but include for ETH too)
    ratio = _cached("gold_btc_ratio", RATIO_TTL, compute_gold_btc_ratio)
    if ratio:
        result["gold_btc_ratio"] = ratio

    # Exchange netflow trend (BTC only from BGeometrics)
    if currency == "BTC":
        netflow = _cached("exchange_netflow_trend", NETFLOW_HIST_TTL,
                          get_exchange_netflow_trend)
        if netflow:
            result["netflow_trend"] = netflow

    return result if result else None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\onchain_data.py | Select-Object -Skip 260 -First 220" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 797ms:
        if time.time() - cache_ts < ONCHAIN_TTL:
            # Still fresh — pre-populate the in-memory cache so _cached() returns
            # immediately without hitting the API.
            from portfolio.shared_state import _cache_lock, _tool_cache
            with _cache_lock:
                if "onchain_btc" not in _tool_cache:
                    _tool_cache["onchain_btc"] = {
                        "data": persistent,
                        "time": cache_ts,
                        "ttl": ONCHAIN_TTL,
                    }
                    logger.debug("On-chain: seeded in-memory cache from persistent (age %.0fs)",
                                 time.time() - cache_ts)

    token = _load_config_token()
    if not token:
        # Try persistent cache even without token
        cached = _load_onchain_cache(max_age_seconds=ONCHAIN_TTL * 2)
        if cached:
            logger.debug("No BGeometrics token, using stale cache")
            return cached
        return None

    return _cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)


# ---------------------------------------------------------------------------
# Interpretation helpers (for Layer 2 context)
# ---------------------------------------------------------------------------

def interpret_onchain(data):
    """Interpret on-chain metrics into human-readable zones.

    Returns dict with zone classifications for each available metric.
    """
    if not data:
        return {}

    interp = {}

    # MVRV Z-Score zones
    zscore = _safe_float(data.get("mvrv_zscore"))
    if zscore is not None:
        if zscore < 1:
            interp["mvrv_zone"] = "undervalued"
        elif zscore > 7:
            interp["mvrv_zone"] = "overheated"
        else:
            interp["mvrv_zone"] = "neutral"

    # SOPR zones
    sopr = _safe_float(data.get("sopr"))
    if sopr is not None:
        if sopr < 0.97:
            interp["sopr_zone"] = "capitulation"
        elif sopr > 1.05:
            interp["sopr_zone"] = "profit_taking"
        else:
            interp["sopr_zone"] = "neutral"

    # NUPL zones
    nupl = _safe_float(data.get("nupl"))
    if nupl is not None:
        if nupl < 0:
            interp["nupl_zone"] = "capitulation"
        elif nupl > 0.75:
            interp["nupl_zone"] = "euphoria"
        elif nupl > 0.5:
            interp["nupl_zone"] = "greed"
        elif nupl > 0.25:
            interp["nupl_zone"] = "optimism"
        else:
            interp["nupl_zone"] = "hope"

    # Exchange netflow
    netflow = _safe_float(data.get("netflow"))
    if netflow is not None:
        if netflow < 0:
            interp["netflow_signal"] = "accumulation"
        else:
            interp["netflow_signal"] = "distribution"

    return interp

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'ruff check portfolio' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'python -m pip show mypy pyflakes ruff pylint' in Q:\finance-analyzer\.worktrees\adv-data-external
2026-05-08T15:30:35.564846Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mruff: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mruff[0m check portfolio[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mThe term 'ruff' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

 exited 1 in 711ms:
[31;1mruff: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mruff[0m check portfolio[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mThe term 'ruff' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

2026-05-08T15:30:35.834941Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 1 seconds
Output:
WARNING: Package(s) not found: mypy, pyflakes, pylint, ruff

 exited 1 in 971ms:
WARNING: Package(s) not found: mypy, pyflakes, pylint, ruff

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib
mods = ['portfolio.social_sentiment','portfolio.bert_sentiment','portfolio.news_keywords']
for m in mods:
    try:
        importlib.import_module(m)
        print(m, 'OK')
    except Exception as e:
        print(m, type(e).__name__, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 717ms:
portfolio.social_sentiment OK
portfolio.bert_sentiment OK
portfolio.news_keywords OK

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\news_keywords.py -TotalCount 220" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\news_keywords.py | Select-Object -Skip 220 -First 200" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 817ms:
"""Shared keyword dictionaries and sector mappings for news/event signals.

Provides:
- Tiered keyword lists with severity weights
- headline scoring function
- Sector-to-ticker and ticker-to-sector mappings
"""

from __future__ import annotations

import re
from datetime import UTC

# ---------------------------------------------------------------------------
# Keyword tiers: word/phrase -> weight multiplier for sentiment scoring
# ---------------------------------------------------------------------------

CRITICAL_KEYWORDS = {
    "war": 3.0,
    "crash": 3.0,
    "tariff": 3.0,
    "tariffs": 3.0,
    "sanctions": 3.0,
    "sanction": 3.0,
    "hack": 3.0,
    "hacked": 3.0,
    "default": 3.0,
    "bankruptcy": 3.0,
    "bankrupt": 3.0,
    "collapse": 3.0,
    "ban": 3.0,
    "banned": 3.0,
    "invasion": 3.0,
    "nuclear": 3.0,
}

HIGH_KEYWORDS = {
    "rate hike": 2.0,
    "rate cut": 2.0,
    "cpi": 2.0,
    "inflation": 2.0,
    "recession": 2.0,
    "delisting": 2.0,
    "delisted": 2.0,
    "lawsuit": 2.0,
    "indictment": 2.0,
    "sec investigation": 2.0,
    "trade war": 2.0,
    "downgrade": 2.0,
    "debt ceiling": 2.0,
    "margin call": 2.0,
    "layoffs": 2.0,
    "fraud": 2.0,
    "subpoena": 2.0,
}

MODERATE_KEYWORDS = {
    "etf approval": 1.5,
    "etf approved": 1.5,
    "regulation": 1.5,
    "regulatory": 1.5,
    "liquidation": 1.5,
    "liquidated": 1.5,
    "upgrade": 1.5,
    "earnings miss": 1.5,
    "earnings beat": 1.5,
    "guidance cut": 1.5,
    "guidance raise": 1.5,
    "buyback": 1.5,
    "stock split": 1.5,
    "merger": 1.5,
    "acquisition": 1.5,
    "ipo": 1.5,
}

# Combined dict for fast lookup (longer phrases checked first)
ALL_KEYWORDS = {**MODERATE_KEYWORDS, **HIGH_KEYWORDS, **CRITICAL_KEYWORDS}

# Pre-compile patterns sorted by length descending (match longer phrases first)
_KEYWORD_PATTERNS = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
    for kw, weight in sorted(ALL_KEYWORDS.items(), key=lambda x: -len(x[0]))
]

# ---------------------------------------------------------------------------
# Sector mappings
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    "semiconductor": {"NVDA", "AMD", "MU", "AVGO", "TSM", "SMCI"},
    "crypto": {"BTC-USD", "ETH-USD", "MSTR"},
    "defense": {"LMT"},
    "big_tech": {"GOOGL", "AMZN", "AAPL", "META"},
    "ai": {"NVDA", "AMD", "GOOGL", "META", "PLTR", "SOUN", "AVGO"},
    "software": {"PLTR"},
    "gaming": {"TTWO"},
    "infrastructure": {"VRT"},
    "metals": {"XAU-USD", "XAG-USD"},
}

# Reverse map: ticker -> set of sectors
TICKER_SECTORS: dict[str, set[str]] = {}
for _sector, _tickers in SECTOR_MAP.items():
    for _t in _tickers:
        TICKER_SECTORS.setdefault(_t, set()).add(_sector)

# Keywords that imply directional impact on specific sectors
KEYWORD_SECTOR_IMPACT = {
    "tariff": {"semiconductor": "SELL", "metals": "BUY"},
    "tariffs": {"semiconductor": "SELL", "metals": "BUY"},
    "trade war": {"semiconductor": "SELL", "metals": "BUY"},
    "sanctions": {"crypto": "SELL"},
    "sanction": {"crypto": "SELL"},
    "ban": {"crypto": "SELL"},
    "banned": {"crypto": "SELL"},
    "hack": {"crypto": "SELL"},
    "hacked": {"crypto": "SELL"},
    "rate hike": {"big_tech": "SELL", "crypto": "SELL", "metals": "SELL"},
    "rate cut": {"big_tech": "BUY", "crypto": "BUY", "metals": "BUY"},
    "recession": {"big_tech": "SELL", "defense": "BUY"},
    "inflation": {"metals": "BUY", "crypto": "BUY"},
    "etf approval": {"crypto": "BUY"},
    "etf approved": {"crypto": "BUY"},
    "regulation": {"crypto": "SELL"},
    "regulatory": {"crypto": "SELL"},
    "war": {"defense": "BUY", "metals": "BUY", "big_tech": "SELL"},
    "invasion": {"defense": "BUY", "metals": "BUY", "big_tech": "SELL"},
}

# Credible financial news sources (weight 1.5x)
CREDIBLE_SOURCES = {
    "reuters", "bloomberg", "wsj", "wall street journal",
    "cnbc", "associated press", "ap", "financial times", "ft",
    "bbc", "new york times", "nyt", "the economist",
    "marketwatch", "barron's", "barrons",
}


def score_headline(title: str) -> tuple[float, list[str]]:
    """Score a headline by keyword severity.

    Returns:
        (max_weight, matched_keywords) — max_weight is the highest keyword
        weight found (1.0 if no keywords match), matched_keywords lists all
        matched keyword strings.
    """
    if not title:
        return 1.0, []

    matched = []
    max_weight = 1.0

    for pattern, weight in _KEYWORD_PATTERNS:
        if pattern.search(title):
            matched.append(pattern.pattern.replace(r"\b", "").replace("\\", ""))
            if weight > max_weight:
                max_weight = weight

    return max_weight, matched


def keyword_severity(title: str) -> str:
    """Classify headline severity: critical, high, moderate, or normal."""
    weight, _ = score_headline(title)
    if weight >= 3.0:
        return "critical"
    if weight >= 2.0:
        return "high"
    if weight >= 1.5:
        return "moderate"
    return "normal"


def is_credible_source(source: str) -> bool:
    """Check if a source name matches a credible financial news source."""
    if not source:
        return False
    lower = source.lower().strip()
    return any(cs in lower for cs in CREDIBLE_SOURCES)


def dissemination_score(articles: list[dict]) -> float:
    """Score how widely news has spread (FinGPT dissemination-aware pattern).

    Factors:
    1. Unique source count — more sources = wider spread
    2. Source diversity — credible sources (Reuters, Bloomberg) weight more
    3. Time clustering — articles within 1h of each other = breaking news

    Returns:
        Float multiplier (1.0 = normal, up to 3.0 for breaking news with wide coverage).
        Used to amplify headline weights in sentiment aggregation.
    """
    if not articles or len(articles) < 2:
        return 1.0

    # Factor 1: Unique source count
    sources = set()
    for a in articles:
        src = a.get("source", "unknown").lower().strip()
        if src:
            sources.add(src)
    source_count = len(sources)
    # 1 source = 1.0, 3+ sources = 1.5, 5+ = 2.0
    source_factor = min(1.0 + (source_count - 1) * 0.25, 2.0)

    # Factor 2: Source diversity — credible source presence
    credible_count = sum(1 for s in sources if any(cs in s for cs in CREDIBLE_SOURCES))
    diversity_factor = 1.0
    if credible_count >= 2:
        diversity_factor = 1.5
    elif credible_count >= 1:
        diversity_factor = 1.25

    # Factor 3: Time clustering — articles within 1h of each other
    from datetime import datetime
    timestamps = []
    for a in articles:
        pub = a.get("published", "")
        if not pub:

 succeeded in 820ms:
            continue
        try:
            if isinstance(pub, (int, float)):
                ts = datetime.fromtimestamp(pub, tz=UTC)
            else:
                # Try ISO format
                pub_str = str(pub).replace("Z", "+00:00")
                ts = datetime.fromisoformat(pub_str)
            timestamps.append(ts.timestamp())
        except (ValueError, TypeError, OSError):
            continue

    clustering_factor = 1.0
    if len(timestamps) >= 3:
        timestamps.sort()
        # Check if most articles appeared within a 1-hour window
        window = 3600  # 1 hour
        max_cluster = 1
        for i in range(len(timestamps)):
            cluster = sum(1 for t in timestamps if abs(t - timestamps[i]) <= window)
            max_cluster = max(max_cluster, cluster)
        # If 60%+ of articles are in a 1h cluster, it's breaking news
        cluster_ratio = max_cluster / len(timestamps)
        if cluster_ratio >= 0.6:
            clustering_factor = 1.5

    # Combined score (multiplicative, capped at 3.0)
    score = source_factor * diversity_factor * clustering_factor
    return min(round(score, 2), 3.0)


def get_sector_impact(keyword: str, ticker: str) -> str | None:
    """Get the directional impact of a keyword on a specific ticker.

    Returns "BUY", "SELL", or None if no sector-specific impact.
    """
    impacts = KEYWORD_SECTOR_IMPACT.get(keyword.lower(), {})
    ticker_secs = TICKER_SECTORS.get(ticker, set())
    for sector, direction in impacts.items():
        if sector in ticker_secs:
            return direction
    return None


# ---------------------------------------------------------------------------
# Headline relevance (added 2026-04-28 for sentiment regression fix)
# ---------------------------------------------------------------------------
#
# Background: shadow LLM accuracy investigation found that the sentiment
# pipeline was scoring every wire-feed headline returned by Yahoo/CryptoCompare,
# including bare price-tickers like "Bitcoin: $67,123" and generic
# "Markets mixed in afternoon trade" boilerplate. Models correctly labeled
# these neutral, but the neutral mass drowned out the few decisive headlines
# in the average. Sentiment regressed from 75.3% -> ~42% over W16-W17.
#
# A headline is "relevant" to a ticker if:
#   1. It triggers a keyword from score_headline (weight > 1.0), OR
#   2. It mentions the ticker symbol or a known synonym (Bitcoin/BTC,
#      Ethereum/ETH, gold/XAU, silver/XAG, or the stock symbol)
# Source-credibility lives in the wrapper in portfolio/sentiment.py, not here.

_TICKER_SYNONYMS: dict[str, list[str]] = {
    "BTC": ["btc", "bitcoin", "bitcoins"],
    "ETH": ["eth", "ethereum", "ether"],
    "XAU": ["xau", "gold", "bullion"],
    "XAG": ["xag", "silver"],
    "MSTR": ["mstr", "microstrategy"],
    "NVDA": ["nvda", "nvidia"],
    "AMD": ["amd"],
    "GOOGL": ["googl", "google", "alphabet"],
    "AMZN": ["amzn", "amazon"],
    "AAPL": ["aapl", "apple"],
    "META": ["meta", "facebook", "instagram"],
    "AVGO": ["avgo", "broadcom"],
    "TSM": ["tsm", "tsmc"],
    "MU": ["mu", "micron"],
    "PLTR": ["pltr", "palantir"],
    "SMCI": ["smci"],
    "TTWO": ["ttwo", "rockstar"],
    "VRT": ["vrt", "vertiv"],
    "LMT": ["lmt", "lockheed"],
    "SOUN": ["soun", "soundhound"],
}


def _ticker_synonym_pattern(ticker: str) -> re.Pattern | None:
    short = ticker.upper().replace("-USD", "")
    syns = _TICKER_SYNONYMS.get(short)
    if not syns:
        if not short or not short.isalnum():
            return None
        return re.compile(r"\b" + re.escape(short) + r"\b", re.IGNORECASE)
    pattern = "|".join(re.escape(s) for s in syns)
    return re.compile(r"\b(" + pattern + r")\b", re.IGNORECASE)


# Memoize per-ticker patterns; tickers are a fixed small set so the cache
# never grows large. None values are cached too (use sentinel-via-membership).
_PATTERN_CACHE: dict[str, re.Pattern | None] = {}


def is_relevant_headline(title: str, ticker: str) -> bool:
    """Return True if the headline is plausibly relevant to the ticker.

    Used by the sentiment pipeline to filter wire-noise before model inference.
    See the module-level comment block above for background.

    Minimum content gate: even when a ticker synonym matches, the headline
    must have at least 3 word tokens AFTER stripping the synonym itself.
    This drops bare price-tickers like "Bitcoin: $67,123" (1 token after
    removing "Bitcoin") while keeping "Bitcoin treasury firm adds 500 BTC"
    (5 tokens after removing the synonyms).
    """
    if not title or not title.strip():
        return False

    weight, _ = score_headline(title)
    if weight > 1.0:
        return True

    short = ticker.upper().replace("-USD", "")
    if short not in _PATTERN_CACHE:
        _PATTERN_CACHE[short] = _ticker_synonym_pattern(short)
    pat = _PATTERN_CACHE[short]
    if pat is None or not pat.search(title):
        return False

    # Synonym matched — guard against bare price-ticker noise. Strip the
    # synonym occurrences and count the remaining word tokens; need >=3 to
    # be considered real content.
    stripped = pat.sub(" ", title)
    tokens = [t for t in re.findall(r"\b[A-Za-z]{2,}\b", stripped) if t.lower() not in {"the", "and", "for", "from"}]
    return len(tokens) >= 3

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | ForEach-Object { Select-String -Path "'$_.FullName -Pattern '"'ThreadPoolExecutor|as_completed|cancel\\(' | ForEach-Object { \"{0}:{1}: {2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() } }' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 683ms:
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:30: ThreadPoolExecutor(8 workers) can call predict() concurrently. The lock is
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:219: # sentiment phase's BERT loads via main.py's ThreadPoolExecutor) can
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:5: from concurrent.futures import ThreadPoolExecutor, as_completed
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:322: with ThreadPoolExecutor(max_workers=len(tfs), thread_name_prefix=f"tf_{source_key}") as pool:
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:329: for future in as_completed(futures, timeout=_TF_POOL_TIMEOUT):
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:338: f.cancel()
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\fx_rates.py:17: # BUG-215: Thread-safe FX cache. Accessed from 8-worker ThreadPoolExecutor.
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:351: Thread-safe — multiple ThreadPoolExecutor workers call this concurrently.

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types
# stubs to import portfolio.market_health
m1 = types.ModuleType('portfolio.file_utils')
m1.atomic_write_json = lambda *a, **k: None
m1.load_json = lambda *a, **k: {}
sys.modules['portfolio.file_utils'] = m1
m2 = types.ModuleType('portfolio.shared_state')
m2._cached = lambda *a, **k: None
sys.modules['portfolio.shared_state'] = m2

import importlib
mh = importlib.import_module('portfolio.market_health')

closes = list(range(100, 130))
volumes = [1000 + i for i in range(30)]
print(mh.detect_ftd_state(closes, volumes, None))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 693ms:
{'state': 'correcting', 'rally_day': 0, 'rally_low': 110, 'recent_high': 129, 'ftd_day_offset': None, 'drawdown_pct': 0.0}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path, start, end in [
    ('portfolio/data_collector.py', 1, 20),
    ('portfolio/data_collector.py', 315, 340),
    ('portfolio/market_health.py', 180, 245),
]:
    print(f'--- {path}:{start}-{end} ---')
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    for i in range(start, min(end, len(lines))+1):
        print(f'{i:4}: {lines[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path, start, end in [
    ('portfolio/sentiment.py', 1, 20),
    ('portfolio/market_health.py', 1, 25),
]:
    print(f'--- {path}:{start}-{end} ---')
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    for i in range(start, min(end, len(lines))+1):
        print(f'{i:4}: {lines[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 675ms:
--- portfolio/data_collector.py:1-20 ---
   1: """Data collection — Binance, Alpaca, yfinance kline fetchers + multi-timeframe collector."""
   2: 
   3: import logging
   4: import time
   5: from concurrent.futures import ThreadPoolExecutor, as_completed
   6: from datetime import UTC, datetime
   7: 
   8: import pandas as pd
   9: 
  10: import portfolio.shared_state as _ss
  11: from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
  12: from portfolio.circuit_breaker import CircuitBreaker
  13: from portfolio.http_retry import fetch_with_retry
  14: from portfolio.indicators import compute_indicators, technical_signal
  15: 
  16: logger = logging.getLogger("portfolio.data_collector")
  17: 
  18: # --- Circuit breakers for each data source ---
  19: 
  20: binance_spot_cb = CircuitBreaker("binance_spot", failure_threshold=5, recovery_timeout=60)
--- portfolio/data_collector.py:315-340 ---
 315: def collect_timeframes(source):
 316:     """Collect all timeframes for a source, fetching in parallel."""
 317:     is_stock = "alpaca" in source
 318:     tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
 319:     source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")
 320: 
 321:     # BUG-179: Submit all timeframe fetches with timeout to prevent hangs
 322:     with ThreadPoolExecutor(max_workers=len(tfs), thread_name_prefix=f"tf_{source_key}") as pool:
 323:         futures = {
 324:             pool.submit(_fetch_one_timeframe, source, source_key, label, interval, limit, ttl): label
 325:             for label, interval, limit, ttl in tfs
 326:         }
 327:         raw_results = []
 328:         try:
 329:             for future in as_completed(futures, timeout=_TF_POOL_TIMEOUT):
 330:                 result = future.result()
 331:                 if result is not None:
 332:                     raw_results.append(result)
 333:         except TimeoutError:
 334:             stuck = [lbl for f, lbl in futures.items() if not f.done()]
 335:             logger.error("BUG-179: Timeframe pool timeout for %s. Stuck: %s",
 336:                          source_key, stuck)
 337:             for f in futures:
 338:                 f.cancel()
 339: 
 340:     # Maintain original timeframe order
--- portfolio/market_health.py:180-245 ---
 180:         closes: daily close prices (oldest first)
 181:         volumes: daily volumes (oldest first)
 182:         prev_state: previous state dict for continuity (optional)
 183: 
 184:     Returns:
 185:         dict with state, rally_day, rally_low, ftd_day_offset, etc.
 186:     """
 187:     if len(closes) < 20:
 188:         return {
 189:             "state": STATE_CORRECTING,
 190:             "rally_day": 0,
 191:             "rally_low": 0,
 192:             "recent_high": 0,
 193:             "ftd_day_offset": None,
 194:         }
 195: 
 196:     # Use last 60 days for analysis
 197:     n = len(closes)
 198: 
 199:     # Initialize state
 200:     if prev_state:
 201:         state = prev_state.get("state", STATE_CORRECTING)
 202:         rally_day = prev_state.get("rally_day", 0)
 203:         rally_low = prev_state.get("rally_low", 0)
 204:         recent_high = prev_state.get("recent_high", 0)
 205:         ftd_day_offset = prev_state.get("ftd_day_offset")
 206:     else:
 207:         state = STATE_CORRECTING
 208:         rally_day = 0
 209:         rally_low = min(closes[-20:])
 210:         recent_high = max(closes[-60:]) if len(closes) >= 60 else max(closes)
 211:         ftd_day_offset = None
 212: 
 213:     # Process the most recent day
 214:     today_close = closes[-1]
 215:     today_vol = volumes[-1]
 216:     yesterday_close = closes[-2]
 217:     yesterday_vol = volumes[-2]
 218: 
 219:     pct_change = (today_close / yesterday_close) - 1
 220: 
 221:     # Update recent high
 222:     if today_close > recent_high:
 223:         recent_high = today_close
 224: 
 225:     # Check correction from high
 226:     drawdown = (today_close / recent_high) - 1 if recent_high > 0 else 0
 227: 
 228:     if state == STATE_CONFIRMED_UPTREND:
 229:         # Check if uptrend is broken (new correction)
 230:         if drawdown <= FTD_CORRECTION_PCT:
 231:             state = STATE_CORRECTING
 232:             rally_day = 0
 233:             rally_low = today_close
 234:             ftd_day_offset = None
 235: 
 236:     elif state == STATE_FTD_CONFIRMED:
 237:         # Check if FTD holds or fails
 238:         if today_close < rally_low:
 239:             # FTD failed — undercut rally low
 240:             state = STATE_CORRECTING
 241:             rally_day = 0
 242:             rally_low = today_close
 243:             ftd_day_offset = None
 244:         elif ftd_day_offset is not None and (n - 1 - ftd_day_offset) > FTD_FAILURE_WINDOW:
 245:             # FTD has held past the failure window — confirmed uptrend

 succeeded in 672ms:
--- portfolio/sentiment.py:1-20 ---
   1: """Sentiment analysis — multi-model A/B testing with FinGPT, CryptoBERT, and TradingHero.
   2: 
   3: Runs multiple sentiment models in shadow mode and logs results for comparison.
   4: The primary (voting) model is the legacy model; shadow models are logged to
   5: data/sentiment_ab_log.jsonl for accuracy tracking. Once a shadow model proves
   6: superior (>60% on 200+ samples), it can be promoted.
   7: 
   8: Models:
   9:   - CryptoBERT: crypto headlines (legacy primary)
  10:   - Trading-Hero-LLM: stock headlines (legacy primary)
  11:   - FinGPT (Finance-Llama-8B / FinGPT-MT): GGUF via llama-cpp-python (shadow)
  12:   - FinBERT: CPU fallback (shadow)
  13: 
  14: Phase 3B: Cumulative headline clustering — groups related headlines and scores
  15: them as a batch for richer "drumbeat effect" detection.
  16: """
  17: 
  18: import json
  19: import logging
  20: import platform
--- portfolio/market_health.py:1-25 ---
   1: """Market health module — distribution days, FTD detection, breadth score.
   2: 
   3: Provides market-level context that the signal engine uses to penalize BUY
   4: confidence in unhealthy markets.  All data comes from yfinance (SPY/QQQ),
   5: cached hourly to avoid rate limits.
   6: 
   7: Key concepts:
   8: - Distribution day (O'Neil): index closes down >=0.2% on higher volume
   9: - Follow-Through Day: >=1.25% gain on day 4+ of rally on higher volume
  10: - Breadth score: composite 0-100 from distribution days, FTD state, SMAs, trend
  11: """
  12: 
  13: import logging
  14: from datetime import UTC, datetime
  15: 
  16: from portfolio.file_utils import atomic_write_json, load_json
  17: from portfolio.shared_state import _cached
  18: 
  19: logger = logging.getLogger("portfolio.market_health")
  20: 
  21: # Cache TTL: 1 hour — market health doesn't need minute-level freshness
  22: MARKET_HEALTH_TTL = 3600
  23: 
  24: # Distribution day thresholds (O'Neil standard)
  25: DIST_DAY_PRICE_DROP_PCT = -0.002  # >=0.2% decline

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\sentiment.py | Select-Object -Skip 490 -First 260" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\sentiment.py | Select-Object -Skip 748 -First 220" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 659ms:
    """Run FinBERT sentiment inference.

    2026-04-09 (fix/bert-inproc-gpu): routes through _run_model so FinBERT
    also benefits from the in-process GPU cache. _run_model's _INPROC_BERT_MAP
    knows that FINBERT_SCRIPT -> "FinBERT" and will hit bert_sentiment.predict
    first, falling back to the old subprocess path on any exception.
    """
    return _run_model(FINBERT_SCRIPT, texts)


# ---------------------------------------------------------------------------
# Headline clustering (Phase 3B)
# ---------------------------------------------------------------------------

def _cluster_headlines(articles):
    """Group headlines by keyword overlap and time proximity.

    Clusters enable cumulative sentiment analysis — many mildly negative
    headlines about the same topic together signal stronger negativity
    than scoring each independently.

    Returns:
        List of clusters, each a list of article dicts
    """
    if not articles or len(articles) < 3:
        return [articles] if articles else []

    from portfolio.news_keywords import score_headline

    # Extract keywords per headline
    headline_keywords = []
    for a in articles:
        _, matched = score_headline(a.get("title", ""))
        # Also extract significant words (>4 chars, not stopwords)
        words = set()
        for word in a.get("title", "").lower().split():
            clean = word.strip(".,!?;:'\"()[]")
            if len(clean) > 4 and clean not in _STOPWORDS:
                words.add(clean)
        words.update(kw.lower() for kw in matched)
        headline_keywords.append(words)

    # Simple greedy clustering by keyword overlap
    clusters = []
    assigned = set()

    for i in range(len(articles)):
        if i in assigned:
            continue
        cluster = [articles[i]]
        assigned.add(i)
        kw_i = headline_keywords[i]

        for j in range(i + 1, len(articles)):
            if j in assigned:
                continue
            kw_j = headline_keywords[j]
            overlap = len(kw_i & kw_j)
            # Cluster if they share 2+ keywords or 1 matched keyword
            if overlap >= 2 or (overlap >= 1 and kw_i & kw_j & _SIGNIFICANT_KEYWORDS):
                cluster.append(articles[j])
                assigned.add(j)

        clusters.append(cluster)

    return clusters


_STOPWORDS = {
    "about", "after", "again", "being", "between", "could", "during",
    "every", "first", "their", "there", "these", "those", "under",
    "which", "while", "would", "other", "still", "where", "before",
    "should", "since", "until", "years", "might", "price", "stock",
    "market", "shares", "today", "report", "quarter",
}

_SIGNIFICANT_KEYWORDS = {
    "tariff", "tariffs", "war", "crash", "sanctions", "hack", "recession",
    "inflation", "rate", "cut", "hike", "layoffs", "earnings", "fomc",
    "bitcoin", "ethereum", "crypto", "nvidia", "semiconductor",
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Decisiveness margins added 2026-04-28. See docs/PLAN_sentiment_2026_04_28.md
# for the full rationale; tl;dr: the old aggregator returned the top label by
# 0.001 vs second, so a 0.34/0.33/0.33 split labeled "positive". Now we require
# a real margin before committing to a non-neutral verdict, and we default to
# label-majority over score-averaging so a few decisive headlines are not
# drowned by many tepid-neutral peers.
_DECISIVE_MARGIN_AVG = 0.05      # avg-mode: top-vs-second margin in prob units
_DECISIVE_MARGIN_PER_HEADLINE = 0.10  # majority-mode: per-headline label margin
_DECISIVE_MARGIN_MAJORITY = 1e-9  # majority-mode: top-vs-second weight margin
                                  # (zero-tolerance — exact ties go neutral)


def _compute_weights(sentiments, headlines, dissemination_mult):
    """Return per-sentiment weights from keyword scoring + dissemination."""
    if headlines and len(headlines) == len(sentiments):
        from portfolio.news_keywords import score_headline
        weights = []
        for h in headlines:
            title = h.get("title", "") if isinstance(h, dict) else str(h)
            w, _ = score_headline(title)
            weights.append(w * dissemination_mult)
    else:
        weights = [dissemination_mult] * len(sentiments)
    return weights


def _aggregate_sentiments(sentiments, headlines=None, dissemination_mult=1.0,
                           *, mode="majority"):
    """Aggregate sentiment scores into a single (label, avg_dict) verdict.

    mode="majority" (default, 2026-04-28): label-majority vote.
        Each headline gets its own decisive label (top score must beat second
        by >=_DECISIVE_MARGIN_PER_HEADLINE, else "neutral"), then a weighted
        majority over those labels picks the verdict. Exact ties resolve to
        neutral. The returned avg_dict is still the score-weighted-average
        (kept identical for backward-compat with consumers that read
        avg_scores like sentiment_avg_scores in signal_engine.py:2452).

    mode="average": legacy probability-averaging. Returns the top-scored
        label IF its margin over the second exceeds _DECISIVE_MARGIN_AVG;
        otherwise downgrades to "neutral". The pure-max-without-margin
        behavior was the source of the W16-W17 sentiment regression and is
        no longer reachable.

    When headlines are provided, score weights from news_keywords.score_headline()
    amplify high-impact keywords (tariff/war/crash 3x). dissemination_mult
    multiplies all weights when news is widely cross-referenced.
    """
    if not sentiments:
        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    weights = _compute_weights(sentiments, headlines, dissemination_mult)
    total_w = sum(weights)
    if total_w == 0:
        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
    neg_sum = sum(s["scores"]["negative"] * w for s, w in zip(sentiments, weights))
    neu_sum = sum(s["scores"]["neutral"] * w for s, w in zip(sentiments, weights))
    avg = {
        "positive": pos_sum / total_w,
        "negative": neg_sum / total_w,
        "neutral":  neu_sum / total_w,
    }

    if mode == "majority":
        verdict = _majority_label(sentiments, weights)
        return verdict, avg

    # mode == "average" — legacy threshold-augmented score-averaging
    overall = max(avg, key=avg.get)
    sorted_scores = sorted(avg.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]
    if margin < _DECISIVE_MARGIN_AVG and overall != "neutral":
        return "neutral", avg
    return overall, avg


def _majority_label(sentiments, weights):
    """Per-headline decisive label, then weighted majority vote.

    Each headline classified as positive/negative/neutral with its own per-
    headline margin gate. Weights are summed per label; winner returned only
    if it beats the second by more than _DECISIVE_MARGIN_MAJORITY.
    """
    bucket = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for s, w in zip(sentiments, weights):
        scores = s["scores"]
        # decisive per-headline label: top must beat second by margin
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ordered[0]
        second_score = ordered[1][1]
        if (top_score - second_score) < _DECISIVE_MARGIN_PER_HEADLINE:
            top_label = "neutral"
        bucket[top_label] += w

    ordered_buckets = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
    winner_label, winner_w = ordered_buckets[0]
    runner_w = ordered_buckets[1][1]
    if (winner_w - runner_w) <= _DECISIVE_MARGIN_MAJORITY:
        return "neutral"
    return winner_label


def _filter_relevant_headlines(articles, ticker, *, fallback_n=3):
    """Drop wire-noise headlines before model inference.

    Uses news_keywords.is_relevant_headline (keyword OR ticker-synonym match)
    plus a credible-source-with-long-title escape hatch (Reuters/Bloomberg/
    etc. + title >= 25 chars covers in-depth coverage that doesn't happen to
    mention the ticker by name).

    Falls back to most-recent `fallback_n` if the filter would drop
    everything — better to have noisy signal than silent signal on slow
    news days.
    """
    if not articles:
        return []

    from portfolio.news_keywords import is_credible_source, is_relevant_headline

    kept = []
    for a in articles:
        title = a.get("title", "") if isinstance(a, dict) else str(a)
        if is_relevant_headline(title, ticker):
            kept.append(a)
            continue
        # Credible-source escape hatch: long titles from credible outlets are
        # almost always real coverage worth scoring.
        source = a.get("source", "") if isinstance(a, dict) else ""
        if is_credible_source(source) and len(title.strip()) >= 25:
            kept.append(a)

    if kept:
        return kept

    # All-irrelevant fallback: keep the most-recent N articles. Sort by the
    # `published` field (ISO timestamp string sorts chronologically), most
    # recent first. Articles without `published` sort last via empty-string
    # default.
    sorted_articles = sorted(
        articles,
        key=lambda a: a.get("published", "") if isinstance(a, dict) else "",
        reverse=True,
    )
    return sorted_articles[:fallback_n]


def _log_ab_result(ticker, primary_result, shadow_results):
    """Log A/B test results to sentiment_ab_log.jsonl for accuracy comparison."""
    try:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "ticker": ticker,
            "primary": {
                "model": primary_result.get("model", "unknown"),
                "sentiment": primary_result.get("overall_sentiment", "unknown"),
                "confidence": primary_result.get("confidence", 0.0),
            },
            "shadow": shadow_results,
        }
        atomic_append_jsonl(AB_LOG_FILE, entry)
    except Exception:
        logger.debug("Failed to log A/B result", exc_info=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None,
                   *, cryptocompare_api_key=None) -> dict:
    """Get sentiment for a ticker using primary model + shadow A/B models.

 succeeded in 642ms:
                   *, cryptocompare_api_key=None) -> dict:
    """Get sentiment for a ticker using primary model + shadow A/B models.

    2026-04-28 (fix/sentiment-relevance-and-aggregation): two changes here.
      1. Crypto primary model swapped CryptoBERT -> Trading-Hero-LLM.
         CryptoBERT was 99.1% neutral on 2,817 wire-feed samples (it was
         trained on crypto-twitter slang, not press-wire headlines).
         CryptoBERT now runs as a shadow for continuity of the 30d accuracy
         baseline. Trading-Hero is permabull on financial news but at least
         produces variance; the 47% directional accuracy gate still acts as
         a circuit breaker if it underperforms.
      2. Headlines pass through _filter_relevant_headlines() before model
         inference. Bare price-tickers ("Bitcoin: $67,123") and generic
         market-update boilerplate are dropped. Fallback keeps the most-
         recent N when the filter would drop everything.

    Primary model (votes):  Trading-Hero-LLM (both crypto and stocks)
    Shadow models (logged): CryptoBERT (crypto only), FinGPT, FinBERT

    Returns the primary model's result. Shadow results are logged to
    data/sentiment_ab_log.jsonl for accuracy tracking.
    """
    short = ticker.upper().replace("-USD", "")
    is_crypto = _is_crypto(short)

    if is_crypto:
        articles = _fetch_crypto_headlines(
            short, cryptocompare_api_key=cryptocompare_api_key,
        )
    else:
        articles = _fetch_stock_headlines(short, newsapi_key=newsapi_key)
    # 2026-04-28: Trading-Hero-LLM is the primary across all asset classes.
    # See module/function docstring for the CryptoBERT demotion rationale.
    model_script = TRADING_HERO_SCRIPT
    model_name = "Trading-Hero-LLM"

    social = social_posts or []
    raw_all = articles + social
    sources = {
        "news": len(articles),
        "reddit": sum(1 for p in social if "reddit" in p.get("source", "")),
    }

    if not raw_all:
        return {
            "overall_sentiment": "unknown",
            "confidence": 0.0,
            "num_articles": 0,
            "model": model_name,
            "sources": sources,
            "details": [],
        }

    # Drop wire-noise before inference. The filter has a most-recent-N
    # fallback so we never go silent on slow-news days.
    all_articles = _filter_relevant_headlines(raw_all, short)
    titles = [a["title"] for a in all_articles]

    # Compute dissemination score for weight amplification
    diss_mult = 1.0
    try:
        from portfolio.news_keywords import dissemination_score
        diss_mult = dissemination_score(all_articles)
    except Exception:
        logger.debug("Dissemination score failed, using default 1.0", exc_info=True)

    # --- Primary model (votes in consensus) ---
    sentiments = _run_model(model_script, titles)
    overall, avg = _aggregate_sentiments(sentiments, headlines=all_articles,
                                         dissemination_mult=diss_mult)

    details = []
    for article, sent in zip(all_articles, sentiments):
        details.append(
            {
                "title": article["title"],
                "source": article["source"],
                "published": article["published"],
                "sentiment": sent.get("sentiment") or sent.get("label", "unknown"),
                "confidence": sent["confidence"],
                "scores": sent["scores"],
            }
        )

    primary_result = {
        "overall_sentiment": overall,
        "confidence": round(avg[overall], 4),
        "num_articles": len(sentiments),
        "avg_scores": {k: round(v, 4) for k, v in avg.items()},
        "model": model_name,
        "sources": sources,
        "details": details,
        "dissemination_score": diss_mult,
    }

    # --- Shadow models (A/B testing — logged only, don't affect consensus) ---
    #
    # 2026-04-09: The A/B log write used to happen inline at the bottom of
    # this function. It is now DEFERRED to flush_ab_log() which is called
    # post-cycle from main.py after flush_llm_batch() completes Phase 3
    # (fingpt sentiment). Rationale: fingpt used to run in a bespoke NDJSON
    # daemon (scripts/fingpt_daemon.py, now retired) blocking inside this
    # function; moving fingpt into portfolio.llm_batch's shared llama_server
    # rotation means the fingpt result does not arrive until AFTER
    # get_sentiment() has returned. Rather than duplicate the A/B log entry
    # or block on the batch, we stash the primary + finbert + context here
    # and let flush_ab_log() assemble the final row.
    #
    # The primary model's voting result is still computed and returned
    # SYNCHRONOUSLY — batching only affects the shadow log, not the vote.
    ab_key = f"{short}:{datetime.now(UTC).isoformat()}"
    _stash_ab_context(ab_key, short, primary_result, all_articles, diss_mult)

    # Shadow: FinGPT — enqueue for post-cycle Phase 3 execution. Zero-cost
    # here; the actual inference runs via llama_server finance-llama-8b
    # rotation after the ticker pool completes.
    #
    # 2026-04-10 (perf/llama-swap-reduction): gated by is_llm_on_cycle. Unlike
    # ministral/qwen3 (which go through _cached_or_enqueue's should_enqueue_fn),
    # fingpt enqueues directly because it doesn't use the signal cache — it
    # only writes to the A/B shadow log. When off-cycle, skip the enqueue
    # entirely so the llama_server phase 3 skips the fingpt model swap +
    # inference cost. Fingpt is a shadow signal, so skipping 2 of 3 cycles
    # just reduces A/B sample density from every cycle to every 3rd cycle,
    # which is fine for long-running statistical comparison.
    try:
        from portfolio.llm_batch import enqueue_fingpt, is_llm_on_cycle
        if is_llm_on_cycle("fingpt"):
            enqueue_fingpt(
                ab_key, "headlines",
                {"mode": "headlines", "texts": titles, "ticker": short},
            )
            clusters = _cluster_headlines(all_articles)
            for idx, cluster in enumerate(clusters):
                if len(cluster) >= 3:
                    cluster_titles = [a["title"] for a in cluster]
                    enqueue_fingpt(
                        ab_key, f"cumul:{idx}",
                        {"mode": "cumulative", "texts": cluster_titles, "ticker": short},
                    )
    except Exception as e:
        logger.debug("FinGPT enqueue failed: %s", e)

    # Shadow: FinBERT (CPU, fast) — still runs inline because it's cheap
    # and on CPU (no model swap cost) and we'd rather not add a fourth
    # phase to llm_batch for an already-shadow-of-shadow signal. Stash its
    # aggregated entry into the pending A/B buffer so flush_ab_log sees it.
    try:
        finbert_results = _run_finbert(titles)
        if finbert_results:
            fb_overall, fb_avg = _aggregate_sentiments(
                finbert_results, headlines=all_articles, dissemination_mult=diss_mult
            )
            _stash_finbert_shadow(ab_key, {
                "model": "FinBERT",
                "sentiment": fb_overall,
                "confidence": round(fb_avg[fb_overall], 4),
                "avg_scores": {k: round(v, 4) for k, v in fb_avg.items()},
            })
    except Exception as e:
        logger.debug("FinBERT shadow failed: %s", e)

    # Shadow: CryptoBERT — demoted from primary 2026-04-28. Kept as shadow
    # for crypto tickers only so we (a) preserve the 30d accuracy baseline
    # for comparison and (b) still notice if the model ever recovers from
    # its 99.1% neutral-output collapse. Stashed into the pending A/B
    # buffer; picked up by flush_ab_log alongside FinGPT and FinBERT.
    if is_crypto:
        try:
            crypto_results = _run_model(CRYPTOBERT_SCRIPT, titles)
            if crypto_results:
                cb_overall, cb_avg = _aggregate_sentiments(
                    crypto_results, headlines=all_articles, dissemination_mult=diss_mult,
                )
                _stash_cryptobert_shadow(ab_key, {
                    "model": "CryptoBERT",
                    "sentiment": cb_overall,
                    "confidence": round(cb_avg[cb_overall], 4),
                    "avg_scores": {k: round(v, 4) for k, v in cb_avg.items()},
                })
        except Exception as e:
            logger.debug("CryptoBERT shadow failed: %s", e)

    return primary_result


def get_crypto_sentiment(ticker="BTC") -> dict:
    return get_sentiment(ticker)


if __name__ == "__main__":
    for ticker in ["BTC", "ETH"]:
        print(f"\n{'='*60}")
        print(f"  Sentiment for {ticker}")
        print(f"{'='*60}")
        result = get_sentiment(ticker)
        print(
            f"Overall: {result['overall_sentiment']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        print(f"Model: {result['model']}")
        print(f"Articles analyzed: {result['num_articles']}")
        print(f"Dissemination score: {result.get('dissemination_score', 1.0):.2f}")
        if result.get("avg_scores"):
            s = result["avg_scores"]
            print(
                f"Avg scores: pos={s['positive']:.3f} neg={s['negative']:.3f} neu={s['neutral']:.3f}"
            )
        print("\nTop headlines:")
        for d in result["details"][:5]:
            emoji = {"positive": "+", "negative": "-", "neutral": "~"}.get(
                d["sentiment"], "?"
            )
            print(
                f"  [{emoji}] {d['sentiment']:>8} ({d['confidence']:.2%}) {d['title']}"
            )

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | ForEach-Object { Select-String -Path "'$_.FullName -Pattern '"'Q:\\\\|/home/deck|/mnt/q' | ForEach-Object { \"{0}:{1}: {2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() } }' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 581ms:
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:5: under /mnt/q/models/ — cryptobert_infer.py, trading_hero_infer.py, and
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:52: - Retiring the 3 /mnt/q/models/*_infer.py scripts: they stay as the fallback
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:76: # under /mnt/q/models/. Label maps use the same ordering as those scripts so
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:82: "cache_dir_win": r"Q:\models\cryptobert",
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:83: "cache_dir_linux": "/home/deck/models/cryptobert",
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:92: "cache_dir_win": r"Q:\models\trading-hero-llm",
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:93: "cache_dir_linux": "/home/deck/models/trading-hero-llm",
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:101: "cache_dir_win": r"Q:\models\finbert",
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\bert_sentiment.py:102: "cache_dir_linux": "/home/deck/models/finbert",
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:34: MODELS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:35: CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:36: TRADING_HERO_SCRIPT = r"Q:\models\trading_hero_infer.py"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:37: FINBERT_SCRIPT = r"Q:\models\finbert_infer.py"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:39: MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:40: CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:41: TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:42: FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"
Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\sentiment.py:334: # bug in /mnt/q/models/fingpt_infer.py, NOT a problem with this migration.

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path, start, end in [
    ('portfolio/market_health.py', 236, 247),
    ('portfolio/market_health.py', 440, 451),
]:
    print(f'--- {path}:{start}-{end} ---')
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    for i in range(start, min(end, len(lines))+1):
        print(f'{i:4}: {lines[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 607ms:
--- portfolio/market_health.py:236-247 ---
 236:     elif state == STATE_FTD_CONFIRMED:
 237:         # Check if FTD holds or fails
 238:         if today_close < rally_low:
 239:             # FTD failed — undercut rally low
 240:             state = STATE_CORRECTING
 241:             rally_day = 0
 242:             rally_low = today_close
 243:             ftd_day_offset = None
 244:         elif ftd_day_offset is not None and (n - 1 - ftd_day_offset) > FTD_FAILURE_WINDOW:
 245:             # FTD has held past the failure window — confirmed uptrend
 246:             state = STATE_CONFIRMED_UPTREND
 247: 
--- portfolio/market_health.py:440-451 ---
 440:         "updated_at": datetime.now(UTC).isoformat(),
 441:     }
 442: 
 443:     # Persist FTD state for continuity across restarts
 444:     state_to_save = {
 445:         "ftd_state": {
 446:             "state": ftd["state"],
 447:             "rally_day": ftd["rally_day"],
 448:             "rally_low": ftd["rally_low"],
 449:             "recent_high": ftd["recent_high"],
 450:             "ftd_day_offset": ftd["ftd_day_offset"],
 451:         },

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types
m1 = types.ModuleType('portfolio.file_utils'); m1.atomic_write_json=lambda *a,**k: None; m1.load_json=lambda *a,**k: {}
sys.modules['portfolio.file_utils']=m1
m2 = types.ModuleType('portfolio.shared_state'); m2._cached=lambda *a,**k: None
sys.modules['portfolio.shared_state']=m2
import importlib
mh = importlib.import_module('portfolio.market_health')
closes = list(range(100, 190))  # 90 points
volumes = [1000+i for i in range(90)]
prev = {'state':'ftd_confirmed','rally_day':4,'rally_low':150,'recent_high':189,'ftd_day_offset':89}
for day in range(1,4):
    state = mh.detect_ftd_state(closes, volumes, prev)
    print(day, state)
    prev = state
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 601ms:
1 {'state': 'ftd_confirmed', 'rally_day': 4, 'rally_low': 150, 'recent_high': 189, 'ftd_day_offset': 89, 'drawdown_pct': 0.0}
2 {'state': 'ftd_confirmed', 'rally_day': 4, 'rally_low': 150, 'recent_high': 189, 'ftd_day_offset': 89, 'drawdown_pct': 0.0}
3 {'state': 'ftd_confirmed', 'rally_day': 4, 'rally_low': 150, 'recent_high': 189, 'ftd_day_offset': 89, 'drawdown_pct': 0.0}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types
m1 = types.ModuleType('portfolio.file_utils'); m1.atomic_write_json=lambda *a,**k: None; m1.load_json=lambda *a,**k: {}
sys.modules['portfolio.file_utils']=m1
m2 = types.ModuleType('portfolio.shared_state'); m2._cached=lambda *a,**k: None
sys.modules['portfolio.shared_state']=m2
import importlib
mh = importlib.import_module('portfolio.market_health')
closes = list(range(100, 190))
volumes = [1000+i for i in range(90)]
prev = {'state':'ftd_confirmed','rally_day':4,'rally_low':150,'recent_high':189,'ftd_day_offset':89}
for day in range(1,13):
    closes = closes[1:] + [closes[-1] + 1]
    volumes = volumes[1:] + [volumes[-1] + 1]
    state = mh.detect_ftd_state(closes, volumes, prev)
    if day in (1,5,11,12):
        print(day, state)
    prev = state
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 642ms:
1 {'state': 'ftd_confirmed', 'rally_day': 4, 'rally_low': 150, 'recent_high': 190, 'ftd_day_offset': 89, 'drawdown_pct': 0.0}
5 {'state': 'ftd_confirmed', 'rally_day': 4, 'rally_low': 150, 'recent_high': 194, 'ftd_day_offset': 89, 'drawdown_pct': 0.0}
11 {'state': 'ftd_confirmed', 'rally_day': 4, 'rally_low': 150, 'recent_high': 200, 'ftd_day_offset': 89, 'drawdown_pct': 0.0}
12 {'state': 'ftd_confirmed', 'rally_day': 4, 'rally_low': 150, 'recent_high': 201, 'ftd_day_offset': 89, 'drawdown_pct': 0.0}

codex
The patch is not self-contained enough to run on a clean checkout, and two of the newly added resilience/market-health paths are functionally broken in ways that will either keep hanging the collector or misclassify market state. These are correctness issues, not style nits.

Full review comments:

- [P0] Include the internal portfolio modules these files import — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:10-14
  On a clean checkout of this branch, importing `portfolio.data_collector` fails immediately with `ModuleNotFoundError: No module named 'portfolio.shared_state'`, and the same is true for other new modules that reference `portfolio.api_utils`, `portfolio.file_utils`, `portfolio.price_source`, etc. None of those internal dependencies exist anywhere in this tree, so the package is non-runnable unless those modules are added in the same change or the imports are removed.

- [P1] Avoid waiting on timed-out timeframe workers — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:333-338
  If any fetch task is still running when `as_completed(..., timeout=_TF_POOL_TIMEOUT)` expires, `f.cancel()` will not stop it once it has started, and leaving the `with ThreadPoolExecutor(...)` block still performs `shutdown(wait=True)`. In that failure mode `collect_timeframes()` continues to block on the hung worker, so BUG-179 is not actually fixed; this path only logs the timeout.

- [P1] Persist FTD age in calendar terms, not window offsets — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\market_health.py:446-450
  This persists `ftd_day_offset`, which is only an index inside the current 90-bar window. On the next run we load a fresh 90-bar slice but compare against the old index in `n - 1 - ftd_day_offset`, so the "days since FTD" counter stops advancing once the window length is stable and `ftd_confirmed` never ages into `confirmed_uptrend`. Persisting the FTD date or elapsed-day count would avoid freezing the state machine.

- [P2] Reconstruct FTD state from history when no snapshot exists — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\market_health.py:206-210
  When `prev_state` is missing (fresh deploy, deleted state file, or corruption), the function seeds `recent_high` from the full lookback and then evaluates only the last bar. In a normal uptrend that keeps `drawdown` near 0, so the state stays `correcting` even for a sustained rally (for example, a monotonic 30-day rise still returns `state='correcting'`), which depresses the breadth score until the next full correction/rally cycle. The initial state needs to be reconstructed from the historical series instead of a single-bar update.
The patch is not self-contained enough to run on a clean checkout, and two of the newly added resilience/market-health paths are functionally broken in ways that will either keep hanging the collector or misclassify market state. These are correctness issues, not style nits.

Full review comments:

- [P0] Include the internal portfolio modules these files import — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:10-14
  On a clean checkout of this branch, importing `portfolio.data_collector` fails immediately with `ModuleNotFoundError: No module named 'portfolio.shared_state'`, and the same is true for other new modules that reference `portfolio.api_utils`, `portfolio.file_utils`, `portfolio.price_source`, etc. None of those internal dependencies exist anywhere in this tree, so the package is non-runnable unless those modules are added in the same change or the imports are removed.

- [P1] Avoid waiting on timed-out timeframe workers — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\data_collector.py:333-338
  If any fetch task is still running when `as_completed(..., timeout=_TF_POOL_TIMEOUT)` expires, `f.cancel()` will not stop it once it has started, and leaving the `with ThreadPoolExecutor(...)` block still performs `shutdown(wait=True)`. In that failure mode `collect_timeframes()` continues to block on the hung worker, so BUG-179 is not actually fixed; this path only logs the timeout.

- [P1] Persist FTD age in calendar terms, not window offsets — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\market_health.py:446-450
  This persists `ftd_day_offset`, which is only an index inside the current 90-bar window. On the next run we load a fresh 90-bar slice but compare against the old index in `n - 1 - ftd_day_offset`, so the "days since FTD" counter stops advancing once the window length is stable and `ftd_confirmed` never ages into `confirmed_uptrend`. Persisting the FTD date or elapsed-day count would avoid freezing the state machine.

- [P2] Reconstruct FTD state from history when no snapshot exists — Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\market_health.py:206-210
  When `prev_state` is missing (fresh deploy, deleted state file, or corruption), the function seeds `recent_high` from the full lookback and then evaluates only the last bar. In a normal uptrend that keeps `drawdown` near 0, so the state stays `correcting` even for a sustained rally (for example, a monotonic 30-day rise still returns `state='correcting'`), which depresses the breadth score until the next full correction/rally cycle. The initial state needs to be reconstructed from the historical series instead of a single-bar update.
