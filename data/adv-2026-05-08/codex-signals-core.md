OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-signals-core
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e0833-6dc2-78e2-a082-0650cef0c375
--------
user
changes against 'empty-baseline'
  → Inspect diff against merge base
  • Analyze changed files for correctness issues
  • Return prioritized findings in JSON
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git diff 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-08T15:27:39.703506Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
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


 exited 1 in 658ms:
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


2026-05-08T15:27:39.723096Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
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


 exited 1 in 676ms:
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


 exited 1 in 674ms:
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


2026-05-08T15:27:39.724624Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
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


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Location; Get-ChildItem -Force' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 563ms:

[32;1mPath[0m
[32;1m----[0m
Q:\finance-analyzer\.worktrees\adv-signals-core

[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\finance-analyzer\.worktrees\adv-signals-core
[32;1mPSChildName         : [0mportfolio
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mportfolio
[32;1mParent              : [0mQ:\finance-analyzer\.worktrees\adv-signals-core
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\finance-analyzer\.worktrees\adv-signals-core\portfolio
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-08 17:25:55
[32;1mCreationTimeUtc     : [0m2026-05-08 15:25:55
[32;1mLastAccessTime      : [0m2026-05-08 17:27:02
[32;1mLastAccessTimeUtc   : [0m2026-05-08 15:27:02
[32;1mLastWriteTime       : [0m2026-05-08 17:25:55
[32;1mLastWriteTimeUtc    : [0m2026-05-08 15:25:55
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mportfolio
[32;1mResolvedTarget      : [0mQ:\finance-analyzer\.worktrees\adv-signals-core\portfolio
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m60
[32;1mName          : [0m.git


[32;1mLastWriteTime : [0m2026-05-08 17:25:55
[32;1mLength        : [0m46
[32;1mName          : [0mBASELINE.md



exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content .git' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 565ms:
gitdir: Q:/finance-analyzer/.git/worktrees/adv-signals-core

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel; git status --short' in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-08T15:27:49.223173Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-signals-core'
'Q:/finance-analyzer/.worktrees/adv-signals-core/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-signals-core
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-signals-core'
'Q:/finance-analyzer/.worktrees/adv-signals-core/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-signals-core

 exited 1 in 562ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-signals-core'
'Q:/finance-analyzer/.worktrees/adv-signals-core/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-signals-core
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-signals-core'
'Q:/finance-analyzer/.worktrees/adv-signals-core/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-signals-core

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-signals-core diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-signals-core diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-signals-core rev-parse --show-toplevel' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 603ms:
Q:/finance-analyzer/.worktrees/adv-signals-core

 succeeded in 640ms:
 portfolio/accuracy_degradation.py    | 1016 ++++++++++
 portfolio/accuracy_stats.py          | 2068 +++++++++++++++++++
 portfolio/forecast_accuracy.py       |  472 +++++
 portfolio/outcome_tracker.py         |  577 ++++++
 portfolio/signal_db.py               |  388 ++++
 portfolio/signal_engine.py           | 3722 ++++++++++++++++++++++++++++++++++
 portfolio/signal_history.py          |  215 ++
 portfolio/signal_postmortem.py       |  266 +++
 portfolio/signal_registry.py         |  276 +++
 portfolio/signal_utils.py            |  132 ++
 portfolio/signal_weight_optimizer.py |  170 ++
 portfolio/signal_weights.py          |  120 ++
 portfolio/ticker_accuracy.py         |  343 ++++
 13 files changed, 9765 insertions(+)

 succeeded in 638ms:
portfolio/accuracy_degradation.py
portfolio/accuracy_stats.py
portfolio/forecast_accuracy.py
portfolio/outcome_tracker.py
portfolio/signal_db.py
portfolio/signal_engine.py
portfolio/signal_history.py
portfolio/signal_postmortem.py
portfolio/signal_registry.py
portfolio/signal_utils.py
portfolio/signal_weight_optimizer.py
portfolio/signal_weights.py
portfolio/ticker_accuracy.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem portfolio | Select-Object Name,Length,LastWriteTime' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 603ms:

[32;1mName                      [0m[32;1m Length[0m[32;1m LastWriteTime[0m
[32;1m----                      [0m [32;1m------[0m [32;1m-------------[0m
accuracy_degradation.py     39336 2026-05-08 17:25:55
accuracy_stats.py           83376 2026-05-08 17:25:55
forecast_accuracy.py        16156 2026-05-08 17:25:55
outcome_tracker.py          20347 2026-05-08 17:25:55
signal_db.py                14797 2026-05-08 17:25:55
signal_engine.py           182767 2026-05-08 17:25:55
signal_history.py            6839 2026-05-08 17:25:55
signal_postmortem.py         8981 2026-05-08 17:25:55
signal_registry.py          17124 2026-05-08 17:25:55
signal_utils.py              4383 2026-05-08 17:25:55
signal_weight_optimizer.py   6091 2026-05-08 17:25:55
signal_weights.py            4527 2026-05-08 17:25:55
ticker_accuracy.py          11934 2026-05-08 17:25:55


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python -m py_compile portfolio\\*.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-08T15:28:03.743361Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[Errno 22] Invalid argument: 'portfolio\\*.py'
 exited 1 in 595ms:
[Errno 22] Invalid argument: 'portfolio\\*.py'
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python - <<'PY'
import compileall
ok = compileall.compile_dir('portfolio', force=True, quiet=1)
print('ok', ok)
PY" in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-08T15:28:07.227653Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m python - <[36;1m<[0m'PY'[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m           ~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mMissing file specification after redirection operator.[0m

 exited 1 in 775ms:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m python - <[36;1m<[0m'PY'[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m           ~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mMissing file specification after redirection operator.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import compileall
ok = compileall.compile_dir('portfolio', force=True, quiet=1)
print('ok', ok)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 688ms:
ok True

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_db.py -First 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_registry.py -First 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_utils.py -First 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_history.py -First 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 750ms:
"""SQLite storage for signal snapshots — replaces signal_log.jsonl for reads.

Schema:
- snapshots: one row per invocation (ts, trigger_reasons, fx_rate)
- ticker_signals: one row per ticker per snapshot (price, consensus, votes, signals JSON)
- outcomes: one row per ticker per horizon per snapshot (backfilled prices + change_pct)

Usage:
    from portfolio.signal_db import SignalDB
    db = SignalDB()           # uses default path data/signal_log.db
    db.insert_snapshot(entry) # dict in same format as signal_log.jsonl line
    entries = db.load_entries()  # returns list[dict] matching JSONL format
"""

import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "signal_log.db"


class SignalDB:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._ensure_schema()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=10)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL UNIQUE,
                trigger_reasons TEXT,
                fx_rate REAL
            );

            CREATE TABLE IF NOT EXISTS ticker_signals (
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                price_usd REAL,
                consensus TEXT,
                buy_count INTEGER,
                sell_count INTEGER,
                total_voters INTEGER,
                signals TEXT,
                regime TEXT DEFAULT 'unknown',
                PRIMARY KEY (snapshot_id, ticker),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                horizon TEXT NOT NULL,
                price_usd REAL,
                change_pct REAL,
                outcome_ts TEXT,
                PRIMARY KEY (snapshot_id, ticker, horizon),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts);
            CREATE INDEX IF NOT EXISTS idx_ticker_signals_ticker ON ticker_signals(ticker);
            CREATE INDEX IF NOT EXISTS idx_outcomes_horizon ON outcomes(horizon);
        """)
        conn.commit()

        # Migration: add regime column to existing DBs that predate this field
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(ticker_signals)").fetchall()}
        if "regime" not in existing_cols:
            conn.execute("ALTER TABLE ticker_signals ADD COLUMN regime TEXT DEFAULT 'unknown'")
            conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Write ---

    def insert_snapshot(self, entry):
        """Insert a signal_log entry (same dict format as JSONL line).

        Skips silently if ts already exists (idempotent for migration).
        """
        conn = self._get_conn()
        ts = entry["ts"]
        trigger_reasons = json.dumps(entry.get("trigger_reasons", []))
        fx_rate = entry.get("fx_rate")

        try:
            cur = conn.execute(
                "INSERT INTO snapshots (ts, trigger_reasons, fx_rate) VALUES (?, ?, ?)",
                (ts, trigger_reasons, fx_rate),
            )
        except sqlite3.IntegrityError:
            return  # duplicate ts, skip

        snapshot_id = cur.lastrowid

        tickers = entry.get("tickers", {})
        for ticker, tdata in tickers.items():
            conn.execute(
                """INSERT INTO ticker_signals
                   (snapshot_id, ticker, price_usd, consensus, buy_count, sell_count, total_voters, signals, regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    ticker,
                    tdata.get("price_usd"),
                    tdata.get("consensus"),
                    tdata.get("buy_count"),
                    tdata.get("sell_count"),
                    tdata.get("total_voters"),
                    json.dumps(tdata.get("signals", {})),
                    tdata.get("regime", "unknown"),
                ),
            )

        outcomes = entry.get("outcomes", {})
        for ticker, horizons in outcomes.items():
            if not isinstance(horizons, dict):
                continue
            for horizon, odata in horizons.items():
                if odata is None:
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO outcomes
                       (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        snapshot_id,
                        ticker,
                        horizon,
                        odata.get("price_usd"),
                        odata.get("change_pct"),
                        odata.get("ts"),
                    ),
                )

        conn.commit()

    def update_outcome(self, ts, ticker, horizon, price_usd, change_pct, outcome_ts):
        """Update a single outcome cell. Used by backfill."""
        conn = self._get_conn()
        row = conn.execute("SELECT id FROM snapshots WHERE ts = ?", (ts,)).fetchone()
        if not row:
            return False
        snapshot_id = row["id"]
        conn.execute(
            """INSERT OR REPLACE INTO outcomes
               (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts),
        )
        conn.commit()
        return True

    # --- Read ---

    def load_entries(self):
        """Load all snapshots as list[dict] matching JSONL format.

        Compatible drop-in replacement for accuracy_stats.load_entries().
        """
        conn = self._get_conn()
        snapshots = conn.execute("SELECT * FROM snapshots ORDER BY ts").fetchall()

        entries = []
        for snap in snapshots:
            sid = snap["id"]

            tickers = {}
            for row in conn.execute(
                "SELECT * FROM ticker_signals WHERE snapshot_id = ?", (sid,)
            ):
                tickers[row["ticker"]] = {
                    "price_usd": row["price_usd"],
                    "consensus": row["consensus"],
                    "buy_count": row["buy_count"],
                    "sell_count": row["sell_count"],
                    "total_voters": row["total_voters"],
                    "signals": json.loads(row["signals"]) if row["signals"] else {},
                    "regime": row["regime"] if row["regime"] is not None else "unknown",
                }

            outcomes = {}
            for row in conn.execute(
                "SELECT * FROM outcomes WHERE snapshot_id = ?", (sid,)
            ):
                if row["ticker"] not in outcomes:
                    outcomes[row["ticker"]] = {}
                outcomes[row["ticker"]][row["horizon"]] = {
                    "price_usd": row["price_usd"],
                    "change_pct": row["change_pct"],
                    "ts": row["outcome_ts"],
                }

            entries.append({
                "ts": snap["ts"],
                "trigger_reasons": json.loads(snap["trigger_reasons"]) if snap["trigger_reasons"] else [],
                "fx_rate": snap["fx_rate"],
                "tickers": tickers,
                "outcomes": outcomes,
            })

        return entries

    def snapshot_count(self):
        """Return total number of snapshots."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM snapshots").fetchone()
        return row["cnt"]

    def entries_missing_outcomes(self, horizon):
        """Find snapshot timestamps that are missing a specific horizon outcome.

        Returns list of (ts, ticker, price_usd) tuples for entries needing backfill.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT s.ts, ts2.ticker, ts2.price_usd
               FROM snapshots s
               JOIN ticker_signals ts2 ON s.id = ts2.snapshot_id
               LEFT JOIN outcomes o ON s.id = o.snapshot_id
                   AND ts2.ticker = o.ticker AND o.horizon = ?
               WHERE o.snapshot_id IS NULL
               ORDER BY s.ts""",
            (horizon,),
        ).fetchall()
        return [(r["ts"], r["ticker"], r["price_usd"]) for r in rows]

    def signal_accuracy(self, horizon="1d"):
        """Compute per-signal accuracy directly via SQL.

        Returns dict matching accuracy_stats.signal_accuracy() format.
        """
        conn = self._get_conn()
        from portfolio.tickers import SIGNAL_NAMES

        stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

        rows = conn.execute(
            """SELECT ts2.signals, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL""",
            (horizon,),

 succeeded in 769ms:
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

 succeeded in 754ms:
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

 succeeded in 758ms:
"""Track signal voting history for flip-flop detection and persistence scoring.

Maintains a rolling window of the last N votes per signal per ticker.
Signals that maintain direction for 3+ consecutive checks are "persistent".
Signals that flip every 1-2 checks are "noisy".
"""

import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_jsonl, load_jsonl
from portfolio.tickers import SIGNAL_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "signal_history.jsonl"

MAX_ENTRIES_PER_TICKER = 50

# 2026-05-02 (adversarial review 05-01 P0-3): the main loop's
# ThreadPoolExecutor (8 workers) calls update_history() concurrently for
# 5 tickers. update_history is a read-modify-write of HISTORY_FILE; without
# a lock, last-writer-wins and 4/5 ticker updates per cycle are silently
# discarded. Persistence scores, streaks, and noisy-signal lists are then
# computed from a corrupted history. The atomic_write_jsonl call inside
# only guarantees the FILE write is atomic — not that the read+modify+write
# sequence is atomic across threads.
_history_lock = threading.Lock()


def _load_history():
    """Load all history entries from JSONL file.

    Returns:
        list[dict]: All history entries.
    """
    return load_jsonl(HISTORY_FILE)


def _save_history(entries):
    """Write all history entries back to JSONL file.

    Args:
        entries: List of history entry dicts.
    """
    atomic_write_jsonl(HISTORY_FILE, entries)


def _entries_for_ticker(entries, ticker):
    """Filter entries for a specific ticker.

    Args:
        entries: All history entries.
        ticker: Ticker symbol.

    Returns:
        list[dict]: Entries for this ticker, ordered by time.
    """
    return [e for e in entries if e.get("ticker") == ticker]


def update_history(ticker, votes_dict):
    """Append current signal votes to history file for a ticker.

    Trims to keep only the last MAX_ENTRIES_PER_TICKER entries per ticker.

    Thread-safe (2026-05-02): the read-modify-write of HISTORY_FILE is
    serialized with `_history_lock` so concurrent ThreadPoolExecutor
    workers don't lose each other's writes.

    Args:
        ticker: Ticker symbol (e.g. "BTC-USD").
        votes_dict: Dict mapping signal_name -> vote ("BUY"/"SELL"/"HOLD").
    """
    with _history_lock:
        entries = _load_history()

        new_entry = {
            "ts": datetime.now(UTC).isoformat(),
            "ticker": ticker,
            "votes": {sig: votes_dict.get(sig, "HOLD") for sig in SIGNAL_NAMES},
        }
        entries.append(new_entry)

        # Trim: keep only last MAX_ENTRIES_PER_TICKER per ticker
        by_ticker = defaultdict(list)
        for e in entries:
            by_ticker[e.get("ticker", "unknown")].append(e)

        trimmed = []
        for _t, t_entries in by_ticker.items():
            trimmed.extend(t_entries[-MAX_ENTRIES_PER_TICKER:])

        # Sort by timestamp for stable ordering
        trimmed.sort(key=lambda e: e.get("ts", ""))
        _save_history(trimmed)


def get_persistence_scores(ticker):
    """Compute persistence score for each signal for a ticker.

    Persistence score measures how often a signal maintains the same direction
    across consecutive checks. Score of 1.0 means perfectly persistent (never
    flips), 0.0 means flips every single check.

    The score is computed as: 1 - (flip_count / (N - 1)) where N is the number
    of non-HOLD votes and flip_count is the number of direction changes.
    Only considers non-HOLD votes (BUY/SELL) since HOLD is absence of signal.

    Args:
        ticker: Ticker symbol.

    Returns:
        dict: Mapping signal_name -> persistence score (0.0 to 1.0).
              Signals with fewer than 3 non-HOLD votes return 0.5 (neutral).
    """
    entries = _entries_for_ticker(_load_history(), ticker)
    if not entries:
        return {sig: 0.5 for sig in SIGNAL_NAMES}

    scores = {}
    for sig in SIGNAL_NAMES:
        # Extract non-HOLD votes in order
        active_votes = []
        for e in entries:
            vote = e.get("votes", {}).get(sig, "HOLD")
            if vote in ("BUY", "SELL"):
                active_votes.append(vote)

        if len(active_votes) < 3:
            scores[sig] = 0.5  # insufficient data
            continue

        # Count direction flips
        flips = 0
        for i in range(1, len(active_votes)):
            if active_votes[i] != active_votes[i - 1]:
                flips += 1

        max_flips = len(active_votes) - 1
        scores[sig] = round(1.0 - (flips / max_flips), 4) if max_flips > 0 else 1.0

    return scores


def get_noisy_signals(ticker, threshold=0.3):
    """Get list of signals that flip too often for a ticker.

    Args:
        ticker: Ticker symbol.
        threshold: Persistence score below this is considered noisy.

    Returns:
        list[str]: Signal names with persistence below threshold.
    """
    scores = get_persistence_scores(ticker)
    return [sig for sig, score in scores.items() if score < threshold]


def get_signal_streaks(ticker):
    """Get current voting streak for each signal for a ticker.

    A streak is the number of consecutive checks a signal has maintained the
    same vote (including HOLD). Useful for detecting sustained signals.

    Args:
        ticker: Ticker symbol.

    Returns:
        dict: Mapping signal_name -> {"current_vote": str, "streak_count": int}.
    """
    entries = _entries_for_ticker(_load_history(), ticker)
    if not entries:
        return {sig: {"current_vote": "HOLD", "streak_count": 0} for sig in SIGNAL_NAMES}

    streaks = {}
    for sig in SIGNAL_NAMES:
        votes = [e.get("votes", {}).get(sig, "HOLD") for e in entries]
        if not votes:
            streaks[sig] = {"current_vote": "HOLD", "streak_count": 0}
            continue

        current = votes[-1]
        count = 0
        for v in reversed(votes):
            if v == current:
                count += 1
            else:
                break

        streaks[sig] = {"current_vote": current, "streak_count": count}

    return streaks


def get_summary(ticker):
    """Get a combined summary of persistence, noise, and streaks for a ticker.

    Convenience function that calls all three analysis functions.

    Args:
        ticker: Ticker symbol.

    Returns:
        dict with keys: persistence_scores, noisy_signals, streaks, entries_count.
    """
    entries = _entries_for_ticker(_load_history(), ticker)
    return {
        "entries_count": len(entries),
        "persistence_scores": get_persistence_scores(ticker),
        "noisy_signals": get_noisy_signals(ticker),
        "streaks": get_signal_streaks(ticker),
    }

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_db.py -TotalCount 400 | Select-Object -Skip 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 641ms:
        ).fetchall()

        for row in rows:
            signals = json.loads(row["signals"]) if row["signals"] else {}
            change_pct = row["change_pct"]
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                stats[sig_name]["total"] += 1
                if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
                    stats[sig_name]["correct"] += 1

        result = {}
        for sig_name in SIGNAL_NAMES:
            s = stats[sig_name]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            result[sig_name] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
            }
        return result

    def consensus_accuracy(self, horizon="1d"):
        """Compute consensus accuracy directly via SQL."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ts2.consensus, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL
                 AND ts2.consensus != 'HOLD'""",
            (horizon,),
        ).fetchall()

        correct = 0
        total = 0
        for row in rows:
            total += 1
            if (row["consensus"] == "BUY" and row["change_pct"] > 0) or \
               (row["consensus"] == "SELL" and row["change_pct"] < 0):
                correct += 1

        acc = correct / total if total > 0 else 0.0
        return {
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "pct": round(acc * 100, 1),
        }

    def per_ticker_accuracy(self, horizon="1d"):
        """Compute per-ticker consensus accuracy via SQL."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ts2.ticker, ts2.consensus, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL
                 AND ts2.consensus != 'HOLD'""",
            (horizon,),
        ).fetchall()

        from collections import defaultdict
        stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in rows:
            stats[row["ticker"]]["total"] += 1
            if (row["consensus"] == "BUY" and row["change_pct"] > 0) or \
               (row["consensus"] == "SELL" and row["change_pct"] < 0):
                stats[row["ticker"]]["correct"] += 1

        result = {}
        for ticker, s in stats.items():
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            result[ticker] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
            }
        return result

    def ticker_signal_accuracy(self, horizon="1d", min_samples=0):
        """Per-ticker per-signal accuracy cross-tabulation via SQL.

        Returns: {ticker: {signal_name: {correct, total, accuracy, pct}}}
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ts2.ticker, ts2.signals, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL""",
            (horizon,),
        ).fetchall()

        from collections import defaultdict
        stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        for row in rows:
            signals = json.loads(row["signals"]) if row["signals"] else {}
            change_pct = row["change_pct"]
            ticker = row["ticker"]
            for sig_name, vote in signals.items():
                if vote == "HOLD":
                    continue
                stats[ticker][sig_name]["total"] += 1
                if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
                    stats[ticker][sig_name]["correct"] += 1

        result = {}
        for ticker, sig_stats in stats.items():
            ticker_result = {}
            for sig_name, s in sig_stats.items():
                if s["total"] < min_samples:
                    continue
                acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
                ticker_result[sig_name] = {
                    "correct": s["correct"],
                    "total": s["total"],
                    "accuracy": acc,
                    "pct": round(acc * 100, 1),
                }
            if ticker_result:
                result[ticker] = ticker_result
        return result

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -File | Select-Object -Expand FullName' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 575ms:
Q:\finance-analyzer\.worktrees\adv-signals-core\BASELINE.md
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\accuracy_degradation.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\accuracy_stats.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\forecast_accuracy.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\outcome_tracker.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_db.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_engine.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_history.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_postmortem.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_registry.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_utils.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_weight_optimizer.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\signal_weights.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\ticker_accuracy.py
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\accuracy_degradation.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\accuracy_stats.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\forecast_accuracy.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\outcome_tracker.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_db.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_engine.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_history.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_postmortem.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_registry.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_utils.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_weight_optimizer.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\signal_weights.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\__pycache__\ticker_accuracy.cpython-312.pyc

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-signals-core ls-tree -r --name-only HEAD' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 640ms:
BASELINE.md
portfolio/accuracy_degradation.py
portfolio/accuracy_stats.py
portfolio/forecast_accuracy.py
portfolio/outcome_tracker.py
portfolio/signal_db.py
portfolio/signal_engine.py
portfolio/signal_history.py
portfolio/signal_postmortem.py
portfolio/signal_registry.py
portfolio/signal_utils.py
portfolio/signal_weight_optimizer.py
portfolio/signal_weights.py
portfolio/ticker_accuracy.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_engine.py -First 120" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 585ms:
"""Signal generation engine — 32-signal voting system with weighted consensus."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.indicators import detect_regime
from portfolio.shared_state import (
    FEAR_GREED_TTL,
    FUNDING_RATE_TTL,
    MINISTRAL_TTL,
    ONCHAIN_TTL,
    SENTIMENT_TTL,
    VOLUME_TTL,
    _cached,
    _cached_or_enqueue,
)
from portfolio.signal_registry import get_enhanced_signals, load_signal_func
from portfolio.signal_utils import true_range
from portfolio.tickers import CRYPTO_SYMBOLS, DISABLED_SIGNALS, GPU_SIGNALS, METALS_SYMBOLS, SIGNAL_NAMES, STOCK_SYMBOLS

logger = logging.getLogger("portfolio.signal_engine")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOCAL_MODEL_ACCURACY_TTL = 1800

# ADX computation cache — keyed by id(df) so each DataFrame is computed at most once.
# Naturally expires when DataFrames are garbage-collected between cycles.
_adx_cache: dict[int, float | None] = {}
_adx_lock = threading.Lock()  # BUG-86: protect concurrent access from ThreadPoolExecutor
_ADX_CACHE_MAX = 200  # prevent unbounded growth

# BUG-178 diagnostics: per-ticker last-signal tracker.
# Updated right before each enhanced signal's compute_fn() is called so that
# when the BUG-178 ticker pool timeout fires, main.py can ask which signal
# each stuck ticker was running. Surfaces silent hangs (signals that never
# complete and therefore never trip the [SLOW] >1s logger).
# Added 2026-04-10 after a 49-event BUG-178 audit traced silent hangs to the
# disabled-signals dispatch path. Cheap (single dict write per signal call).
_last_signal_per_ticker: dict[str, tuple[str, float]] = {}
_last_signal_lock = threading.Lock()


def _set_last_signal(ticker: str, sig_name: str) -> None:
    """Record the signal currently being computed for a ticker (BUG-178 diag)."""
    with _last_signal_lock:
        _last_signal_per_ticker[ticker] = (sig_name, time.monotonic())


def get_last_signal(ticker: str) -> tuple[str, float] | None:
    """Return (sig_name, elapsed_seconds) for the most recent signal start
    on this ticker, or None if no signal has been recorded.

    Used by main.py's BUG-178 timeout handler to identify which signal hung.
    """
    with _last_signal_lock:
        entry = _last_signal_per_ticker.get(ticker)
    if entry is None:
        return None
    sig_name, started = entry
    return sig_name, time.monotonic() - started


# BUG-178 phase log (added 2026-04-15): records per-ticker phase durations
# inside generate_signal()'s post-dispatch code. The __post_dispatch__ marker
# above was too coarse — it collapsed 7+ distinct post-dispatch operations
# (accuracy load, weighted consensus, penalties, linear factor, etc.) into
# a single "after dispatch" bucket, so slow cycles with elapsed_since_set
# ~170s out of a 180s pool timeout gave us zero signal about which phase
# was actually slow.
#
# Each phase records (phase_name, duration_seconds). main.py's slow-cycle
# diagnostic reads this log when the pool timeout fires so we can see the
# full phase breakdown retrospectively. Bounded per-ticker (replaced on
# each generate_signal call) so memory is constant.
_phase_log_per_ticker: dict[str, list[tuple[str, float]]] = {}
_phase_log_lock = threading.Lock()

_PHASE_WARN_THRESHOLD_S = 2.0

# Defensive bound on the number of distinct ticker keys kept. In production
# this is 5 (Tier-1 symbols); tests and probes may pass arbitrary names and
# slowly grow the dict. When we exceed the cap, prune the least-recently-
# used entries (reset cycles refresh them, so LRU by last-reset is fine).
# Prior callers were silently leaking one small list per unique ticker name.
_PHASE_LOG_MAX_TICKERS = 64


def _reset_phase_log(ticker: str) -> None:
    """Clear the phase log for a ticker at the start of generate_signal.

    Also enforces _PHASE_LOG_MAX_TICKERS by pruning older entries when the
    dict grows past the cap — cheap per-call O(n) but n is bounded and the
    prune happens at most once per generate_signal invocation.
    """
    if not ticker:
        return
    with _phase_log_lock:
        if len(_phase_log_per_ticker) >= _PHASE_LOG_MAX_TICKERS and ticker not in _phase_log_per_ticker:
            # Evict oldest half — we don't need true LRU, just bounded memory.
            # `iter(dict)` yields insertion order in CPython 3.7+; dropping
            # the first half gives us amortized O(1) per call.
            evict_count = len(_phase_log_per_ticker) // 2
            for old_key in list(_phase_log_per_ticker)[:evict_count]:
                del _phase_log_per_ticker[old_key]
        _phase_log_per_ticker[ticker] = []


def _record_phase(ticker: str, phase: str, start_mono: float) -> float:
    """Record a phase completion for a ticker. Returns the phase duration.

    Logs WARNING if duration > _PHASE_WARN_THRESHOLD_S so that slow
    individual phases (e.g., cold accuracy_stats load, lock contention)
    are visible in portfolio.log without waiting for a BUG-178 timeout.
    """

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content BASELINE.md' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 619ms:
# Empty baseline for adversarial review diffs

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, pathlib, json
for path in pathlib.Path('portfolio').glob('*.py'):
    tree = ast.parse(path.read_text())
    defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defs.append((node.name, node.lineno))
        elif isinstance(node, ast.ClassDef):
            defs.append((f'class {node.name}', node.lineno))
    print(f'## {path.name}')
    for name, line in defs:
        print(f'{line}: {name}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 704ms:
## accuracy_degradation.py
90: _load_alert_state
101: _save_alert_state
105: _load_snapshot_state
113: _save_snapshot_state
119: save_full_accuracy_snapshot
203: _per_ticker_recent
240: _compact_per_ticker
265: _load_snapshots
270: _find_baseline_snapshot
277: _snapshot_age_days
287: _is_econ_blackout
311: _make_violation
328: check_degradation
400: _hydrate_cached_violations
410: _violation_to_dict
419: _dict_to_violation
430: _diff_against_baseline
558: _binomial_diff_se_pp
576: _maybe_alert
620: _classify_severity
631: _alerts_to_violations
663: filter_alerts_by_cooldown
693: maybe_save_daily_snapshot
762: _record_snapshot_writer_silent_failure
819: maybe_send_degradation_summary
886: build_daily_summary
968: _summary_diffs
1010: _format_summary_row
## accuracy_stats.py
98: _load_signal_utility_disk
115: _write_signal_utility_disk
144: load_entries
170: _vote_correct
189: signal_accuracy
280: signal_accuracy_recent
291: signal_accuracy_ewma
388: signal_accuracy_cost_adjusted
457: consensus_accuracy
516: per_ticker_accuracy
554: accuracy_by_signal_ticker
620: signal_utility
672: _compute_signal_utility
733: invalidate_signal_utility_cache
763: best_worst_signals
778: signal_activation_rates
861: blend_accuracy_data
973: load_cached_activation_rates
992: load_cached_accuracy
1008: write_accuracy_cache
1027: get_or_compute_accuracy
1046: get_or_compute_recent_accuracy
1062: get_or_compute_per_ticker_accuracy
1082: get_or_compute_consensus_accuracy
1123: _load_prewarm_ts_from_disk
1132: _save_prewarm_ts_to_disk
1143: maybe_prewarm_dashboard_accuracy
1238: _count_entries_with_outcomes
1249: print_accuracy_report
1312: signal_accuracy_by_regime
1378: load_cached_regime_accuracy
1395: write_regime_accuracy_cache
1428: get_or_compute_regime_accuracy
1484: invalidate_regime_accuracy_cache
1528: save_accuracy_snapshot
1568: _load_accuracy_snapshots
1583: _find_snapshot_near
1610: check_accuracy_changes
1685: format_accuracy_alerts
1704: signal_best_horizon_accuracy
1805: accuracy_by_ticker_signal
1883: top_signals_for_ticker
1912: load_cached_ticker_accuracy
1930: write_ticker_accuracy_cache
1944: _filter_min_samples
1956: accuracy_by_ticker_signal_cached
1982: probability_calibration
## forecast_accuracy.py
35: cached_forecast_accuracy
58: invalidate_forecast_accuracy_cache
69: load_predictions
86: load_health_stats
118: compute_forecast_accuracy
215: get_ticker_accuracy
245: get_all_ticker_accuracies
275: backfill_forecast_outcomes
353: _lookup_price_at_time
399: _write_predictions
405: get_forecast_accuracy_summary
445: print_forecast_accuracy_report
## outcome_tracker.py
29: _derive_signal_vote
112: log_signal_snapshot
171: _fetch_current_price
211: _fetch_historical_price
281: _signal_log_lock_path
293: _hold_signal_log_lock
342: backfill_outcomes
## signal_db.py
24: class SignalDB
## signal_engine.py
49: _set_last_signal
55: get_last_signal
94: _reset_phase_log
114: _record_phase
131: get_phase_log
263: _apply_persistence_filter
544: _get_horizon_disabled_signals
665: _load_prev_sentiments
685: _get_prev_sentiment
691: _set_prev_sentiment
700: flush_sentiment_state
883: _get_regime_gated
999: _compute_dynamic_horizon_weights
1068: _get_horizon_weights
1085: _compute_applicable_count
1123: _validate_signal_result
1173: _compute_agreement_rate
1193: _compute_dynamic_correlation_groups
1284: _get_correlation_groups
1387: _safe_accuracy
1407: _safe_sample_count
1421: _count_active_voters_at_gate
1475: _normalize_regime
1497: _dynamic_min_voters_for_regime
1519: _compute_gate_relaxation
1637: _compute_ic_mult
1666: _get_ic_data
1698: _is_macro_window_cached
1737: _weighted_consensus
2236: _confluence_score
2252: _time_of_day_factor
2263: _load_local_model_accuracy
2281: _build_llm_context
2320: _gate_local_model_vote
2358: _compute_adx
2415: apply_confidence_penalties
2585: generate_signal
## signal_history.py
33: _load_history
42: _save_history
51: _entries_for_ticker
64: update_history
101: get_persistence_scores
148: get_noisy_signals
162: get_signal_streaks
198: get_summary
## signal_postmortem.py
32: compute_regime_insights
87: compute_signal_health_report
121: compute_vote_correlation
185: generate_postmortem
252: get_postmortem_context
## signal_registry.py
18: register_signal
50: register_enhanced
67: get_enhanced_signals
72: get_signal_names
82: load_signal_func
107: _register_defaults
## signal_utils.py
16: sma
24: ema
32: rsi
46: true_range
55: safe_float
66: rma
75: wma
90: majority_vote
129: roc
## signal_weights.py
25: class SignalWeightManager
## signal_weight_optimizer.py
32: class WalkForwardResult
52: walk_forward_optimize
157: save_results
164: load_results
## ticker_accuracy.py
16: accuracy_by_ticker_signal
85: direction_probability
177: direction_probability_with_forecast
254: get_focus_probabilities
320: _extract_votes

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_weights.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_weight_optimizer.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\ticker_accuracy.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 852ms:
"""Multiplicative Weight Updates (MWU) for online signal learning.

Each signal maintains a persistent weight that is multiplied up on correct
outcomes and down on wrong outcomes.  The result is a classic Hedge algorithm:
signals that are consistently wrong rapidly approach zero weight (floor 0.01)
while consistently correct signals grow to dominate the aggregation.

Weights are persisted to JSON via the same atomic I/O used across the project.
"""

import logging
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.signal_weights")

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _BASE_DIR / "data" / "signal_weights.json"

_DEFAULT_ETA = 0.1   # learning rate — 10% multiplicative update per outcome
_WEIGHT_FLOOR = 0.01  # never reaches zero


class SignalWeightManager:
    """Manages MWU weights for all trading signals.

    Thread-safety note: this class is not internally thread-safe.  In the
    current system it is only called from the single-threaded outcome backfill
    path, so no locking is required.  Add a threading.Lock if that changes.
    """

    def __init__(self, path=None, eta=None):
        self._path = Path(path) if path is not None else _DEFAULT_PATH
        self._eta = eta if eta is not None else _DEFAULT_ETA
        self._weights: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weight(self, signal_name: str) -> float:
        """Return the current weight for *signal_name*.

        Defaults to 1.0 for unknown signals (no prior history).
        """
        return self._weights.get(signal_name, 1.0)

    def update(self, signal_name: str, correct: bool) -> float:
        """Update the weight for *signal_name* after one outcome.

        Correct prediction  → multiply by (1 + eta)
        Incorrect prediction → multiply by (1 - eta)

        The weight is clamped to the floor [_WEIGHT_FLOOR, +∞).

        Returns the new weight.
        """
        current = self._weights.get(signal_name, 1.0)
        if correct:
            new_weight = current * (1.0 + self._eta)
        else:
            new_weight = current * (1.0 - self._eta)
        new_weight = max(new_weight, _WEIGHT_FLOOR)
        self._weights[signal_name] = new_weight
        return new_weight

    def batch_update(self, outcomes: dict) -> None:
        """Update multiple signals at once then persist to disk.

        Args:
            outcomes: ``{signal_name: bool}`` — True means correct prediction.
        """
        for signal_name, correct in outcomes.items():
            self.update(signal_name, correct)
        self.save()

    def get_normalized_weights(self, signal_names) -> dict:
        """Return weights normalised so their average equals 1.0.

        Only considers signals in *signal_names*.  If the list is empty or all
        weights are zero, returns a uniform dict with all values set to 1.0.

        This means the total magnitude of the consensus is preserved — signals
        above 1.0 are stronger than average, below 1.0 weaker.
        """
        signal_names = list(signal_names)
        if not signal_names:
            return {}
        raw = {name: self.get_weight(name) for name in signal_names}
        avg = sum(raw.values()) / len(raw)
        if avg == 0.0:
            return {name: 1.0 for name in signal_names}
        return {name: w / avg for name, w in raw.items()}

    def save(self) -> None:
        """Persist weights to JSON atomically."""
        payload = {
            "eta": self._eta,
            "weights": self._weights,
        }
        atomic_write_json(self._path, payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load weights from disk.  No-ops silently if the file is missing."""
        data = load_json(self._path, default=None)
        if data is None:
            return
        if isinstance(data, dict):
            self._weights = {
                k: float(v)
                for k, v in data.get("weights", {}).items()
            }
            # Honour stored eta only if caller did not override it
            # (caller passes None → _DEFAULT_ETA, so we preserve stored value)

 succeeded in 876ms:
"""Walk-forward signal weight optimizer.

Retrains signal weights using rolling windows to prevent overfitting
and adapt to changing market regimes. Uses the LinearFactorModel for
per-window ridge regression.

Walk-forward method:
    1. Split history into train/test windows (e.g. 30d train, 7d test)
    2. Train model on each window, score on out-of-sample test period
    3. Track per-signal weight stability and out-of-sample performance
    4. Output: recommended weights and stability metrics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.linear_factor import LinearFactorModel

logger = logging.getLogger("portfolio.signal_weight_optimizer")

_BASE_DIR = Path(__file__).resolve().parent.parent
_RESULTS_FILE = _BASE_DIR / "data" / "models" / "walkforward_results.json"


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward optimization run."""
    n_windows: int = 0
    avg_r_squared: float = 0.0
    avg_oos_corr: float = 0.0  # out-of-sample correlation
    weight_stability: dict[str, float] = field(default_factory=dict)
    recommended_weights: dict[str, float] = field(default_factory=dict)
    signal_rankings: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "avg_r_squared": self.avg_r_squared,
            "avg_oos_corr": self.avg_oos_corr,
            "weight_stability": self.weight_stability,
            "recommended_weights": self.recommended_weights,
            "signal_rankings": self.signal_rankings,
        }


def walk_forward_optimize(
    signals_df: pd.DataFrame,
    returns: pd.Series,
    train_window: int = 720,   # 720 hours = 30 days
    test_window: int = 168,    # 168 hours = 7 days
    step_size: int = 168,      # step by 7 days
    alpha: float = 1.0,
    min_train_samples: int = 100,
) -> WalkForwardResult:
    """Run walk-forward optimization across rolling windows.

    Args:
        signals_df: DataFrame of signal values (columns=signals, rows=time).
        returns: Series of forward returns aligned with signals_df.
        train_window: Number of rows for training period.
        test_window: Number of rows for test period.
        step_size: Step size between windows.
        alpha: Ridge regularization strength.
        min_train_samples: Minimum training samples per window.

    Returns:
        WalkForwardResult with averaged metrics and recommended weights.
    """
    common = signals_df.index.intersection(returns.index)
    signals_df = signals_df.loc[common]
    returns = returns.loc[common]
    n = len(common)

    if n < train_window + test_window:
        logger.warning("Insufficient data for walk-forward: %d < %d",
                      n, train_window + test_window)
        return WalkForwardResult()

    all_weights: list[dict[str, float]] = []
    r_squared_scores: list[float] = []
    oos_correlations: list[float] = []

    start = 0
    while start + train_window + test_window <= n:
        train_end = start + train_window
        test_end = train_end + test_window

        train_X = signals_df.iloc[start:train_end]
        train_y = returns.iloc[start:train_end]
        test_X = signals_df.iloc[train_end:test_end]
        test_y = returns.iloc[train_end:test_end]

        model = LinearFactorModel(alpha=alpha)
        if not model.fit(train_X, train_y, min_samples=min_train_samples):
            start += step_size
            continue

        r_squared_scores.append(model.r_squared)
        all_weights.append(model.weights)

        # Out-of-sample prediction correlation
        predictions = []
        for _, row in test_X.iterrows():
            predictions.append(model.predict(row.to_dict()))
        if len(predictions) > 1 and test_y.std() > 1e-10:
            corr = float(np.corrcoef(predictions, test_y.values)[0, 1])
            if not np.isnan(corr):
                oos_correlations.append(corr)

        start += step_size

    if not all_weights:
        return WalkForwardResult()

    # Compute weight stability: std of each weight across windows / mean of abs
    all_signals = set()
    for w in all_weights:
        all_signals.update(w.keys())

    weight_stability = {}
    recommended_weights = {}
    for sig in all_signals:
        values = [w.get(sig, 0.0) for w in all_weights]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        mean_abs = float(np.mean(np.abs(values)))
        # Stability = 1 - (std / mean_abs). High = consistent direction.
        stability = 1.0 - (std_val / mean_abs) if mean_abs > 1e-10 else 0.0
        weight_stability[sig] = round(max(0.0, stability), 4)
        recommended_weights[sig] = round(mean_val, 6)

    # Rank signals by |mean_weight| * stability
    signal_rankings = sorted(
        [(sig, round(abs(recommended_weights[sig]) * weight_stability.get(sig, 0), 6))
         for sig in all_signals],
        key=lambda x: x[1],
        reverse=True,
    )

    result = WalkForwardResult(
        n_windows=len(all_weights),
        avg_r_squared=round(float(np.mean(r_squared_scores)), 4),
        avg_oos_corr=round(float(np.mean(oos_correlations)), 4) if oos_correlations else 0.0,
        weight_stability=weight_stability,
        recommended_weights=recommended_weights,
        signal_rankings=signal_rankings,
    )
    return result


def save_results(result: WalkForwardResult, path: Path | None = None) -> None:
    """Persist walk-forward results to JSON."""
    path = path or _RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, result.to_dict())


def load_results(path: Path | None = None) -> WalkForwardResult | None:
    """Load walk-forward results from JSON."""
    path = path or _RESULTS_FILE
    data = load_json(path)
    if not data:
        return None
    return WalkForwardResult(**data)

 succeeded in 933ms:
"""Per-ticker per-signal accuracy and directional probability engine.

Computes P(up) at multiple horizons for focus instruments using accuracy-weighted
signal votes. This is the core engine for Mode B probability-based notifications.
"""

import logging
import math
from datetime import UTC

from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES

logger = logging.getLogger("portfolio.ticker_accuracy")


def accuracy_by_ticker_signal(ticker, horizon="1d", days=None):
    """Per-signal accuracy for a specific ticker.

    Queries signal_log data (SQLite preferred, JSONL fallback) and computes
    accuracy grouped by signal name for a single ticker.

    Args:
        ticker: Instrument ticker (e.g., "XAG-USD").
        horizon: Outcome horizon ("3h", "1d", "3d", "5d", "10d").
        days: If set, only include entries from the last N days.

    Returns:
        dict: {signal_name: {"accuracy": float, "samples": int, "correct": int}}
    """
    # P0-1 (2026-05-02 adversarial follow-up): use accuracy_stats._vote_correct
    # so the neutral-outcome filter (|change_pct| < _MIN_CHANGE_PCT, change_pct
    # is None) matches signal_accuracy() / per_ticker_accuracy() / consensus_
    # accuracy() everywhere else. Without this, per-ticker accuracy was
    # overstated — Mode B Telegram and Kelly sizing both feed off it.
    from portfolio.accuracy_stats import _vote_correct, load_entries

    entries = load_entries()

    cutoff = None
    if days is not None:
        from datetime import datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        if cutoff and entry.get("ts", "") < cutoff:
            continue

        tdata = entry.get("tickers", {}).get(ticker)
        if not tdata:
            continue

        outcome = entry.get("outcomes", {}).get(ticker, {}).get(horizon)
        if not outcome:
            continue

        change_pct = outcome.get("change_pct")
        signals = tdata.get("signals", {})

        for sig_name in SIGNAL_NAMES:
            vote = signals.get(sig_name, "HOLD")
            if vote == "HOLD":
                continue
            result_val = _vote_correct(vote, change_pct)
            if result_val is None:
                continue  # neutral outcome — don't count
            stats[sig_name]["total"] += 1
            if result_val:
                stats[sig_name]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        if s["total"] == 0:
            continue
        result[sig_name] = {
            "accuracy": s["correct"] / s["total"],
            "samples": s["total"],
            "correct": s["correct"],
        }
    return result


def direction_probability(ticker, current_votes, horizon="1d", days=7, min_samples=5):
    """Convert signal votes + per-ticker accuracy into P(up) at given horizon.

    Algorithm:
        For each active signal voting BUY or SELL for this ticker:
          acc = that signal's accuracy for THIS ticker at THIS horizon
          if signal says BUY:  p_up = acc       (71% accurate BUY → 71% P(up))
          if signal says SELL: p_up = 1 - acc    (71% accurate SELL → 29% P(up))
        Weighted average of all p_up values (weight = sqrt(sample_count))

    Args:
        ticker: Instrument ticker.
        current_votes: dict {signal_name: "BUY"|"SELL"|"HOLD"} for current cycle.
        horizon: Outcome horizon for accuracy lookup.
        days: Lookback window for accuracy computation (None = all time).
        min_samples: Minimum samples required to use a signal's accuracy.

    Returns:
        dict: {
            "direction": "up"|"down"|"neutral",
            "probability": float (0.0-1.0),
            "signals_used": int,
            "total_samples": int,
            "signal_details": [{name, vote, accuracy, samples, p_up, weight}]
        }
    """
    acc_data = accuracy_by_ticker_signal(ticker, horizon=horizon, days=days)

    weighted_sum = 0.0
    weight_total = 0.0
    signals_used = 0
    total_samples = 0
    details = []

    for sig_name, vote in current_votes.items():
        if vote == "HOLD" or sig_name in DISABLED_SIGNALS:
            continue

        sig_acc = acc_data.get(sig_name)
        if not sig_acc or sig_acc["samples"] < min_samples:
            continue

        accuracy = sig_acc["accuracy"]
        samples = sig_acc["samples"]

        # Compute P(up) based on vote direction and accuracy
        if vote == "BUY":
            p_up = accuracy
        else:  # SELL
            p_up = 1.0 - accuracy

        weight = math.sqrt(samples)
        weighted_sum += p_up * weight
        weight_total += weight
        signals_used += 1
        total_samples += samples

        details.append({
            "name": sig_name,
            "vote": vote,
            "accuracy": round(accuracy, 3),
            "samples": samples,
            "p_up": round(p_up, 3),
            "weight": round(weight, 2),
        })

    if weight_total == 0:
        return {
            "direction": "neutral",
            "probability": 0.5,
            "signals_used": 0,
            "total_samples": 0,
            "signal_details": [],
        }

    p_up_final = weighted_sum / weight_total

    direction = "neutral"
    if p_up_final > 0.52:
        direction = "up"
    elif p_up_final < 0.48:
        direction = "down"

    return {
        "direction": direction,
        "probability": round(p_up_final, 3),
        "signals_used": signals_used,
        "total_samples": total_samples,
        "signal_details": details,
    }


def direction_probability_with_forecast(ticker, current_votes, forecast_data=None,
                                        horizon="1d", days=7, min_samples=5):
    """Enhanced probability with Chronos forecast blending.

    Takes the base signal-accuracy probability and blends in Chronos forecast
    predictions, weighted by Chronos's per-ticker accuracy.

    Args:
        ticker: Instrument ticker.
        current_votes: dict {signal_name: vote} for current cycle.
        forecast_data: dict from forecast signal indicators, e.g.:
            {"chronos_24h_pct": 0.5, "chronos_24h_conf": 0.6,
             "chronos_1h_pct": 0.3, "chronos_1h_conf": 0.55,
             "chronos_ok": True}
        horizon: Outcome horizon.
        days: Lookback window.
        min_samples: Minimum samples for signal accuracy.

    Returns:
        dict: Same as direction_probability() but with added fields:
            "forecast_pct_move": float (Chronos predicted % move)
            "forecast_confidence": float (Chronos confidence)
            "forecast_blended": bool (whether forecast was blended in)
    """
    base = direction_probability(ticker, current_votes, horizon=horizon,
                                 days=days, min_samples=min_samples)

    # Add forecast data if available
    if not forecast_data or not forecast_data.get("chronos_ok"):
        base["forecast_blended"] = False
        return base

    # Map horizon to chronos key
    horizon_map = {"1h": "1h", "3h": "1h", "1d": "24h", "3d": "24h", "24h": "24h"}
    chronos_h = horizon_map.get(horizon, "24h")

    pct_key = f"chronos_{chronos_h}_pct"
    conf_key = f"chronos_{chronos_h}_conf"

    pct_move = forecast_data.get(pct_key, 0) or 0
    confidence = forecast_data.get(conf_key, 0) or 0

    base["forecast_pct_move"] = round(pct_move, 3) if pct_move else 0
    base["forecast_confidence"] = round(confidence, 3) if confidence else 0

    # Blend forecast into probability if we have meaningful data
    if abs(pct_move) > 0.1 and confidence > 0.1:
        # Convert Chronos prediction to P(up): positive pct = higher P(up)
        forecast_p_up = 0.5 + min(max(pct_move * 0.1, -0.3), 0.3)  # scale, cap at 0.2-0.8

        # Weight by confidence (scaled down since forecast is one input among many)
        forecast_weight = confidence * 2.0  # comparable to sqrt(samples) for ~4 samples

        # Blend with existing probability
        existing_weight = sum(d.get("weight", 0) for d in base.get("signal_details", []))
        total_weight = existing_weight + forecast_weight

        if total_weight > 0:
            blended_p = (base["probability"] * existing_weight +
                         forecast_p_up * forecast_weight) / total_weight
            base["probability"] = round(blended_p, 3)

            # Update direction
            if blended_p > 0.52:
                base["direction"] = "up"
            elif blended_p < 0.48:
                base["direction"] = "down"
            else:
                base["direction"] = "neutral"

        base["forecast_blended"] = True
    else:
        base["forecast_blended"] = False

    return base


def get_focus_probabilities(tickers, current_data, horizons=None, days=7):
    """Compute probabilities for focus instruments across multiple horizons.

    Args:
        tickers: List of focus ticker names (e.g., ["XAG-USD", "BTC-USD"]).
        current_data: dict {ticker: {signals dict from signal_engine output}}.
            Each ticker entry needs an "extra" dict with "_votes" or
            a "signals" dict mapping signal_name -> vote.
        horizons: List of horizon strings. Defaults to ["3h", "1d", "3d"].
        days: Lookback window for accuracy.

    Returns:
        dict: {
            ticker: {
                "3h": {"direction": "up", "probability": 0.72, ...},
                "1d": {"direction": "up", "probability": 0.68, ...},
                "3d": {"direction": "up", "probability": 0.55, ...},
                "accuracy_1d": 0.71,
                "accuracy_samples": 89,
            }
        }
    """
    if horizons is None:
        horizons = ["3h", "1d", "3d"]

    result = {}

    for ticker in tickers:
        ticker_data = current_data.get(ticker, {})

        # Extract current votes from the signal data
        votes = _extract_votes(ticker_data)
        if not votes:
            continue

        ticker_result = {}
        for h in horizons:
            prob = direction_probability(ticker, votes, horizon=h, days=days)
            ticker_result[h] = prob

        # Overall accuracy summary (1d horizon as primary)
        acc_1d = accuracy_by_ticker_signal(ticker, horizon="1d", days=days)
        total_samples = sum(s["samples"] for s in acc_1d.values())
        if acc_1d:
            # Weighted average accuracy across active signals
            weighted_acc = 0.0
            weight_sum = 0.0
            for sig_name, sig_data in acc_1d.items():
                if sig_name in DISABLED_SIGNALS:
                    continue
                w = math.sqrt(sig_data["samples"])
                weighted_acc += sig_data["accuracy"] * w
                weight_sum += w
            if weight_sum > 0:
                ticker_result["accuracy_1d"] = round(weighted_acc / weight_sum, 3)
            else:
                ticker_result["accuracy_1d"] = 0.0
        else:
            ticker_result["accuracy_1d"] = 0.0

        ticker_result["accuracy_samples"] = total_samples
        result[ticker] = ticker_result

    return result


def _extract_votes(ticker_data):
    """Extract signal votes from various data formats.

    Handles:
    - {"extra": {"_votes": {signal: vote}}} (from signal_engine)
    - {"signals": {signal: vote}} (from signal_log)
    - Direct {signal: vote} dict
    """
    # From signal_engine output
    extra = ticker_data.get("extra", {})
    votes = extra.get("_votes")
    if votes:
        return votes

    # From signal_log format
    signals = ticker_data.get("signals", {})
    if signals and any(v in ("BUY", "SELL") for v in signals.values()):
        return signals

    # Direct dict
    if any(k in SIGNAL_NAMES for k in ticker_data):
        return {k: v for k, v in ticker_data.items() if k in SIGNAL_NAMES}

    return {}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_weights.py -TotalCount 220 | Select-Object -Skip 80" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 589ms:

        Only considers signals in *signal_names*.  If the list is empty or all
        weights are zero, returns a uniform dict with all values set to 1.0.

        This means the total magnitude of the consensus is preserved — signals
        above 1.0 are stronger than average, below 1.0 weaker.
        """
        signal_names = list(signal_names)
        if not signal_names:
            return {}
        raw = {name: self.get_weight(name) for name in signal_names}
        avg = sum(raw.values()) / len(raw)
        if avg == 0.0:
            return {name: 1.0 for name in signal_names}
        return {name: w / avg for name, w in raw.items()}

    def save(self) -> None:
        """Persist weights to JSON atomically."""
        payload = {
            "eta": self._eta,
            "weights": self._weights,
        }
        atomic_write_json(self._path, payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load weights from disk.  No-ops silently if the file is missing."""
        data = load_json(self._path, default=None)
        if data is None:
            return
        if isinstance(data, dict):
            self._weights = {
                k: float(v)
                for k, v in data.get("weights", {}).items()
            }
            # Honour stored eta only if caller did not override it
            # (caller passes None → _DEFAULT_ETA, so we preserve stored value)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_weights.py | Select-Object -Last 40" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 604ms:

        Only considers signals in *signal_names*.  If the list is empty or all
        weights are zero, returns a uniform dict with all values set to 1.0.

        This means the total magnitude of the consensus is preserved — signals
        above 1.0 are stronger than average, below 1.0 weaker.
        """
        signal_names = list(signal_names)
        if not signal_names:
            return {}
        raw = {name: self.get_weight(name) for name in signal_names}
        avg = sum(raw.values()) / len(raw)
        if avg == 0.0:
            return {name: 1.0 for name in signal_names}
        return {name: w / avg for name, w in raw.items()}

    def save(self) -> None:
        """Persist weights to JSON atomically."""
        payload = {
            "eta": self._eta,
            "weights": self._weights,
        }
        atomic_write_json(self._path, payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load weights from disk.  No-ops silently if the file is missing."""
        data = load_json(self._path, default=None)
        if data is None:
            return
        if isinstance(data, dict):
            self._weights = {
                k: float(v)
                for k, v in data.get("weights", {}).items()
            }
            # Honour stored eta only if caller did not override it
            # (caller passes None → _DEFAULT_ETA, so we preserve stored value)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p = Path('portfolio/signal_weights.py')
text = p.read_text()
print(repr(text[-200:]))
print('len', len(text.splitlines()))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 692ms:
' v in data.get("weights", {}).items()\n            }\n            # Honour stored eta only if caller did not override it\n            # (caller passes None → _DEFAULT_ETA, so we preserve stored value)\n'
len 120

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'SignalWeightManager\\('" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 599ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\forecast_accuracy.py -First 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\forecast_accuracy.py -TotalCount 500 | Select-Object -Skip 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\outcome_tracker.py -First 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\outcome_tracker.py -TotalCount 500 | Select-Object -Skip 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 681ms:
"""Forecast sub-signal accuracy tracker.

Reads forecast_predictions.jsonl, backfills actual prices at 1h/24h horizons,
and computes per-model per-ticker per-horizon accuracy statistics.
"""

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_write_jsonl

logger = logging.getLogger("portfolio.forecast_accuracy")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PREDICTIONS_FILE = DATA_DIR / "forecast_predictions.jsonl"
HEALTH_FILE = DATA_DIR / "forecast_health.jsonl"

# BUG-178/W15-W16 follow-up (2026-04-16): per-cycle in-memory cache for
# compute_forecast_accuracy(). The accuracy_degradation tracker calls this
# every cycle (throttled to every ~hour for the full check); each call walks
# forecast_predictions.jsonl. Cache the result for 1h so the hot path is a
# dict lookup. Mirrors the signal_utility cache pattern in accuracy_stats.py:
# lock guards the (timestamp, value) swap, NOT the compute, so two threads
# racing past TTL each recompute once but neither blocks the other.
_FORECAST_ACCURACY_CACHE_TTL = 3600.0
_forecast_accuracy_cache: dict[tuple, tuple[float, dict]] = {}
_forecast_accuracy_cache_lock = threading.Lock()


def cached_forecast_accuracy(horizon="24h", days=7, use_raw_sub_signals=True):
    """Cached wrapper around compute_forecast_accuracy().

    Keyed by (horizon, days, use_raw_sub_signals). 1h TTL — long enough to
    keep cycle-level callers fast, short enough to surface fresh outcomes
    after the daily backfill. Tests and callers that need immediately-fresh
    data can call compute_forecast_accuracy() directly.
    """
    key = (horizon, days, use_raw_sub_signals)
    now = time.time()
    with _forecast_accuracy_cache_lock:
        cached = _forecast_accuracy_cache.get(key)
        if cached and now - cached[0] < _FORECAST_ACCURACY_CACHE_TTL:
            return cached[1]
    # Compute outside the lock so other horizons aren't blocked.
    result = compute_forecast_accuracy(
        horizon=horizon, days=days, use_raw_sub_signals=use_raw_sub_signals,
    )
    with _forecast_accuracy_cache_lock:
        _forecast_accuracy_cache[key] = (now, result)
    return result


def invalidate_forecast_accuracy_cache():
    """Clear the in-memory forecast accuracy cache.

    Called by tests or by code paths that just wrote new outcomes (e.g.
    after a manual backfill). Production code rarely needs this — the 1h
    TTL handles natural staleness.
    """
    with _forecast_accuracy_cache_lock:
        _forecast_accuracy_cache.clear()


def load_predictions(predictions_file=None):
    """Load all forecast predictions from JSONL file."""
    path = predictions_file or PREDICTIONS_FILE
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def load_health_stats(health_file=None):
    """Load forecast health stats (success/failure rates per model)."""
    path = health_file or HEALTH_FILE
    if not path.exists():
        return {}
    stats = defaultdict(lambda: {"ok": 0, "fail": 0, "total": 0})
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            model = entry.get("model", "unknown")
            if entry.get("ok"):
                stats[model]["ok"] += 1
            else:
                stats[model]["fail"] += 1
            stats[model]["total"] += 1
        except json.JSONDecodeError:
            continue

    result = {}
    for model, s in stats.items():
        result[model] = {
            "ok": s["ok"],
            "fail": s["fail"],
            "total": s["total"],
            "success_rate": round(s["ok"] / s["total"], 3) if s["total"] else 0.0,
        }
    return result


def compute_forecast_accuracy(ticker=None, horizon="24h", days=None,
                              predictions_file=None, use_raw_sub_signals=False):
    """Compute accuracy of forecast sub-signals.

    For each prediction entry that has an actual outcome (backfilled),
    check if the predicted direction matched actual price movement.

    Args:
        ticker: Filter to specific ticker (None = all).
        horizon: Which horizon to evaluate ("1h" or "24h").
        days: Only include entries from last N days (None = all).
        predictions_file: Override predictions file path (for testing).

    Returns:
        dict: {
            model_name: {
                "accuracy": float,
                "correct": int,
                "total": int,
                "by_ticker": {ticker: {"accuracy": float, "correct": int, "total": int}}
            }
        }
    """
    entries = load_predictions(predictions_file)

    cutoff = None
    if days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    # Track per-model stats
    model_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for entry in entries:
        if cutoff and entry.get("ts", "") < cutoff:
            continue

        entry_ticker = entry.get("ticker", "")
        if ticker and entry_ticker != ticker:
            continue

        # Need actual outcome
        outcome = entry.get("outcome", {}).get(horizon)
        if outcome is None:
            continue

        actual_change = outcome.get("change_pct", 0)

        # Check each sub-signal
        sub_signals = entry.get("sub_signals", {})
        if use_raw_sub_signals and entry.get("raw_sub_signals"):
            sub_signals = entry.get("raw_sub_signals", {})
        for sub_name, vote in sub_signals.items():
            if vote == "HOLD":
                continue

            # Only check sub-signals matching the requested horizon
            if "_" in sub_name:
                sub_horizon = sub_name.split("_", 1)[1]
                if sub_horizon != horizon:
                    continue

            predicted_up = vote == "BUY"
            actual_up = actual_change > 0

            correct = (predicted_up and actual_up) or (not predicted_up and not actual_up)

            model_stats[sub_name][entry_ticker]["total"] += 1
            if correct:
                model_stats[sub_name][entry_ticker]["correct"] += 1

    # Aggregate
    result = {}
    for sub_name, ticker_stats in model_stats.items():
        total_correct = 0
        total_count = 0
        by_ticker = {}

        for t, s in ticker_stats.items():
            total_correct += s["correct"]
            total_count += s["total"]
            if s["total"] > 0:
                by_ticker[t] = {
                    "accuracy": round(s["correct"] / s["total"], 3),
                    "correct": s["correct"],
                    "total": s["total"],
                }

        result[sub_name] = {
            "accuracy": round(total_correct / total_count, 3) if total_count else 0.0,
            "correct": total_correct,
            "total": total_count,
            "by_ticker": by_ticker,
        }

    return result


def get_ticker_accuracy(ticker, horizon="24h", days=7, predictions_file=None):
    """Get forecast accuracy for a single ticker.

    Lightweight wrapper around compute_forecast_accuracy filtered to one ticker.

    Returns:
        dict: {"accuracy": float, "samples": int} or None if no data.
    """
    result = compute_forecast_accuracy(
        ticker=ticker, horizon=horizon, days=days,
        predictions_file=predictions_file,
    )
    # Aggregate across all sub-signals for this ticker's horizon
    total_correct = 0
    total_count = 0
    for sub_name, stats in result.items():
        # Only include sub-signals matching the requested horizon
        if "_" in sub_name and sub_name.split("_", 1)[1] == horizon:
            total_correct += stats["correct"]
            total_count += stats["total"]

    if total_count == 0:
        return None

    return {
        "accuracy": round(total_correct / total_count, 3),
        "samples": total_count,
    }


def get_all_ticker_accuracies(horizon="24h", days=7, predictions_file=None):
    """Get per-ticker forecast accuracy for all tickers.

    Returns:
        dict: {ticker: {"accuracy": float, "samples": int}}
    """
    result = compute_forecast_accuracy(
        horizon=horizon, days=days, predictions_file=predictions_file,
    )

    # Aggregate per-ticker across sub-signals
    ticker_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for sub_name, stats in result.items():
        # Only include sub-signals matching the requested horizon
        if "_" not in sub_name or sub_name.split("_", 1)[1] != horizon:
            continue

 succeeded in 717ms:
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.api_utils import BINANCE_BASE, BINANCE_FAPI_BASE
from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import _yfinance_limiter

logger = logging.getLogger("portfolio.outcome_tracker")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"

HORIZONS = {"3h": 10800, "4h": 14400, "12h": 43200, "1d": 86400, "3d": 259200, "5d": 432000, "10d": 864000}
import contextlib

from portfolio.tickers import (
    BINANCE_FAPI_MAP,
    BINANCE_SPOT_MAP,
    SIGNAL_NAMES,
    YF_MAP,
)


def _derive_signal_vote(name, indicators, extra):
    if name == "rsi":
        rsi = indicators.get("rsi")
        if rsi is None:
            return "HOLD"
        rsi_lower = indicators.get("rsi_p20", 30)
        rsi_upper = indicators.get("rsi_p80", 70)
        rsi_lower = max(rsi_lower, 15)
        rsi_upper = min(rsi_upper, 85)
        if rsi < rsi_lower:
            return "BUY"
        if rsi > rsi_upper:
            return "SELL"
        return "HOLD"

    if name == "macd":
        hist = indicators.get("macd_hist")
        hist_prev = indicators.get("macd_hist_prev")
        if hist is None or hist_prev is None:
            return "HOLD"
        if hist > 0 and hist_prev <= 0:
            return "BUY"
        if hist < 0 and hist_prev >= 0:
            return "SELL"
        return "HOLD"

    if name == "ema":
        ema9 = indicators.get("ema9")
        ema21 = indicators.get("ema21")
        if ema9 is None or ema21 is None:
            return "HOLD"
        ema_gap_pct = abs(ema9 - ema21) / ema21 * 100 if ema21 != 0 else 0
        if ema_gap_pct < 0.5:
            return "HOLD"
        return "BUY" if ema9 > ema21 else "SELL"

    if name == "bb":
        pos = indicators.get("price_vs_bb")
        if pos == "below_lower":
            return "BUY"
        if pos == "above_upper":
            return "SELL"
        return "HOLD"

    if name == "fear_greed":
        fg = extra.get("fear_greed")
        if fg is None:
            return "HOLD"
        if fg <= 20:
            return "BUY"
        if fg >= 80:
            return "SELL"
        return "HOLD"

    if name == "sentiment":
        sent = extra.get("sentiment")
        conf = extra.get("sentiment_conf", 0)
        if sent == "positive" and conf > 0.4:
            return "BUY"
        if sent == "negative" and conf > 0.4:
            return "SELL"
        return "HOLD"

    if name == "ministral":
        return extra.get("ministral_action", "HOLD")

    if name == "ml":
        return extra.get("ml_action", "HOLD")

    if name == "funding":
        return extra.get("funding_action", "HOLD")

    if name == "volume":
        return extra.get("volume_action", "HOLD")

    if name == "qwen3":
        return extra.get("qwen3_action", "HOLD")

    # custom_lora removed — signal disabled, no longer tracked

    return "HOLD"


def log_signal_snapshot(signals_dict, prices_usd, fx_rate, trigger_reasons):
    ts = datetime.now(UTC).isoformat()
    tickers = {}

    for ticker, sig_data in signals_dict.items():
        indicators = sig_data.get("indicators", {})
        extra = sig_data.get("extra", {})
        price = prices_usd.get(ticker, indicators.get("close"))

        # CROSS-001: use _raw_votes (pre-gate) so accuracy accumulates for
        # regime-gated signals, breaking the dead-signal trap.
        passed_votes = extra.get("_raw_votes", extra.get("_votes"))
        if passed_votes:
            signals = {name: passed_votes.get(name, "HOLD") for name in SIGNAL_NAMES}
        else:
            signals = {}
            for name in SIGNAL_NAMES:
                signals[name] = _derive_signal_vote(name, indicators, extra)

        buy_count = sum(1 for v in signals.values() if v == "BUY")
        sell_count = sum(1 for v in signals.values() if v == "SELL")

        consensus = sig_data.get("action", "HOLD")
        total_voters = buy_count + sell_count

        regime = extra.get("_regime", "unknown")

        tickers[ticker] = {
            "price_usd": price,
            "consensus": consensus,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_voters": total_voters,
            "signals": signals,
            "regime": regime,
        }

    entry = {
        "ts": ts,
        "trigger_reasons": trigger_reasons,
        "fx_rate": fx_rate,
        "tickers": tickers,
        "outcomes": {},
    }

    atomic_append_jsonl(SIGNAL_LOG, entry)

    # Dual-write to SQLite
    try:
        from portfolio.signal_db import SignalDB
        db = SignalDB()
        db.insert_snapshot(entry)
        db.close()
    except Exception as e:
        logger.warning("SQLite snapshot write failed (JSONL has data, SQLite may lag): %s", e)

    return entry


def _fetch_current_price(ticker):
    if ticker in BINANCE_FAPI_MAP:
        symbol = BINANCE_FAPI_MAP[ticker]
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        if r is None:
            return None
        r.raise_for_status()
        return float(r.json()["price"])

    if ticker in BINANCE_SPOT_MAP:
        symbol = BINANCE_SPOT_MAP[ticker]
        r = fetch_with_retry(
            f"{BINANCE_BASE}/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        if r is None:
            return None
        r.raise_for_status()
        return float(r.json()["price"])

    if ticker in YF_MAP:
        # 2026-04-14: route through canonical price_source (Alpaca for
        # stocks, yfinance emergency fallback). Previously bypassed
        # Alpaca entirely — added a 15-min delay to every stock outcome
        # backfill call.
        from portfolio.price_source import fetch_klines

        h = fetch_klines(YF_MAP[ticker], interval="1d", limit=5, period="5d")
        if h is None or h.empty:
            return None
        return float(h["close"].iloc[-1])

    return None


def _fetch_historical_price(ticker, target_ts):
    if ticker in BINANCE_FAPI_MAP:
        symbol = BINANCE_FAPI_MAP[ticker]
        start_ms = int(target_ts * 1000)
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/klines",
            params={
                "symbol": symbol,
                "interval": "1h",
                "startTime": start_ms,
                "limit": 1,
            },
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return float(data[0][4])

    if ticker in BINANCE_SPOT_MAP:
        symbol = BINANCE_SPOT_MAP[ticker]
        start_ms = int(target_ts * 1000)
        r = fetch_with_retry(
            f"{BINANCE_BASE}/klines",
            params={
                "symbol": symbol,
                "interval": "1h",
                "startTime": start_ms,
                "limit": 1,
            },
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return float(data[0][4])

    if ticker in YF_MAP:
        # 2026-04-14: historical price-at-timestamp needs a date window
        # that price_source.fetch_klines's `limit`-based contract can't
        # express precisely. Keep direct yfinance here — the call is
        # only for outcome backfill (non-hot-path; 15-min delay is
        # acceptable for an outcome at target_ts already in the past).
        # To migrate, add start/end support to price_source and update

 succeeded in 725ms:
        # this block.
        import yfinance as yf

        _yfinance_limiter.wait()
        target_dt = datetime.fromtimestamp(target_ts, tz=UTC)
        start_date = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        t = yf.Ticker(YF_MAP[ticker])
        h = t.history(start=start_date, end=end_date)
        if h.empty:
            return None
        target_date = target_dt.date()
        candidates = h[h.index.date <= target_date]
        if candidates.empty:
            return float(h["Close"].iloc[0])
        return float(candidates["Close"].iloc[-1])

    return None


def _signal_log_lock_path():
    """Path of the sidecar lockfile shared with atomic_append_jsonl.

    Same convention as portfolio.file_utils.atomic_append_jsonl:
        ``<dir>/.<filename>.lock`` (e.g. ``data/.signal_log.jsonl.lock``).
    Held to coordinate read-modify-rewrite of SIGNAL_LOG with concurrent
    log_signal_snapshot() appenders. SC-P1-3 (2026-05-02 follow-up).
    """
    return SIGNAL_LOG.parent / f".{SIGNAL_LOG.name}.lock"


@contextlib.contextmanager
def _hold_signal_log_lock():
    """Acquire the sidecar lockfile that atomic_append_jsonl uses.

    Cross-platform pattern lifted from portfolio/file_utils.py:
      - Windows: msvcrt.locking on a 1-byte range
      - POSIX: fcntl.flock LOCK_EX
    The lockfile is pre-created with a single null byte if missing so
    locking never fails on a size-0 file.
    """
    try:
        import msvcrt as _msvcrt  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - non-Windows
        _msvcrt = None  # type: ignore[assignment]
    try:
        import fcntl as _fcntl  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - Windows
        _fcntl = None  # type: ignore[assignment]

    import os as _os
    lock_path = _signal_log_lock_path()
    if not lock_path.exists():
        try:
            with open(lock_path, "ab") as lf:
                if lf.tell() == 0:
                    lf.write(b"\0")
        except OSError:
            pass

    with open(lock_path, "rb+") as lock_f:
        lfd = lock_f.fileno()
        win_locked = False
        try:
            if _msvcrt is not None:
                _os.lseek(lfd, 0, _os.SEEK_SET)
                _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
                win_locked = True
            elif _fcntl is not None:
                _fcntl.flock(lfd, _fcntl.LOCK_EX)
            yield
        finally:
            if win_locked and _msvcrt is not None:
                try:
                    _os.lseek(lfd, 0, _os.SEEK_SET)
                    _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            # fcntl.flock releases on close.


def backfill_outcomes(max_entries=2000):
    """Backfill price outcomes for signal log entries.

    Memory-optimized: only parses the last ``max_entries`` lines as JSON.
    Head entries are streamed as raw bytes during rewrite (BUG-112).

    SC-P1-3 (2026-05-02 follow-up): coordinates with concurrent
    log_signal_snapshot() appenders via the sidecar lock used by
    atomic_append_jsonl. Pattern:
      1. Snapshot phase (lock held briefly): record file size at
         backfill start, then read+parse entries from disk.
      2. Process phase (lock RELEASED): make slow HTTP calls. Live
         appenders run normally during this window.
      3. Rewrite phase (lock re-acquired): re-stat the file. Any bytes
         past the snapshot size are concurrent appends — copy them
         verbatim into the rewritten tmp file AFTER the processed tail,
         then os.replace under the lock so the rename is atomic w.r.t.
         any further appender. Without this, every entry appended in
         the process window was clobbered by the os.replace.

    Args:
        max_entries: Only process the last N entries to limit memory usage.
            Older entries are assumed to be fully backfilled already.
    """
    if not SIGNAL_LOG.exists():
        return 0

    # ---- Phase 1: snapshot read (lock held) ----
    with _hold_signal_log_lock():
        file_size = SIGNAL_LOG.stat().st_size
        if file_size == 0:
            return 0

        # Phase 1a: Count total lines (fast binary scan, no JSON parsing)
        total_lines = 0
        with open(SIGNAL_LOG, "rb") as f:
            for _ in f:
                total_lines += 1

        head_count = max(0, total_lines - max_entries) if max_entries else 0

        # Phase 1b: Skip head lines, parse only the tail as JSON
        head_end_offset = 0
        entries = []
        with open(SIGNAL_LOG, "rb") as f:
            for _ in range(head_count):
                f.readline()  # skip without JSON parsing
            head_end_offset = f.tell()

            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    try:
                        entries.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
        # snapshot_size pins the byte boundary of "what we read".
        # Anything appended past this offset during processing must be
        # preserved verbatim during the rewrite.
        snapshot_size = file_size
        # ---- lock released here for the slow processing window ----

    now = datetime.now(UTC)
    now_ts = now.timestamp()
    price_cache = {}
    updated = 0

    # Tickers we can actually fetch prices for — skip unknown/removed tickers
    known_tickers = set(BINANCE_SPOT_MAP) | set(BINANCE_FAPI_MAP) | set(YF_MAP)

    # Open SignalDB once for all dual-writes (avoids per-outcome open/close)
    _db = None
    try:
        from portfolio.signal_db import SignalDB
        _db = SignalDB()
    except Exception as e:
        logger.debug("SignalDB open failed: %s", e)

    for entry in entries:
        entry_ts = datetime.fromisoformat(entry["ts"]).timestamp()
        tickers = entry.get("tickers", {})
        outcomes = entry.get("outcomes", {})

        all_filled = True
        for ticker in tickers:
            if ticker not in outcomes:
                outcomes[ticker] = {h: None for h in HORIZONS}
            for h_key in HORIZONS:
                if outcomes[ticker].get(h_key) is None:
                    all_filled = False

        if all_filled and all(
            all(outcomes[t].get(h) is not None for h in HORIZONS) for t in tickers
        ):
            continue

        entry_updated = False
        for ticker in tickers:
            if ticker not in known_tickers:
                continue  # skip removed/unknown tickers (e.g. AI)
            if ticker not in outcomes:
                outcomes[ticker] = {h: None for h in HORIZONS}

            base_price = tickers[ticker].get("price_usd")
            if not base_price or base_price <= 0:
                continue  # BUG-220: skip — no base price to compute change_pct
            for h_key, h_seconds in HORIZONS.items():
                if outcomes[ticker].get(h_key) is not None:
                    continue
                target_ts = entry_ts + h_seconds
                if now_ts < target_ts:
                    continue

                cache_key = (ticker, int(target_ts // 3600))
                if cache_key not in price_cache:
                    try:
                        price_cache[cache_key] = _fetch_historical_price(
                            ticker, target_ts
                        )
                    except Exception:
                        price_cache[cache_key] = None

                hist_price = price_cache[cache_key]
                if hist_price is None:
                    continue

                change_pct = round(
                    ((hist_price - base_price) / base_price) * 100, 2
                )

                outcome_ts_str = datetime.fromtimestamp(
                    target_ts, tz=UTC
                ).isoformat()
                outcomes[ticker][h_key] = {
                    "price_usd": round(hist_price, 2),
                    "change_pct": change_pct,
                    "ts": outcome_ts_str,
                }
                entry_updated = True

                # Dual-write outcome to SQLite
                if _db is not None:
                    try:
                        _db.update_outcome(
                            entry["ts"], ticker, h_key,
                            round(hist_price, 2), change_pct, outcome_ts_str,
                        )
                    except Exception as e:
                        logger.debug("SQLite outcome write failed: %s", e)

        entry["outcomes"] = outcomes

        # C6: MWU weight update removed — SignalWeightManager.batch_update()
        # wrote to data/signal_weights.json but signal_engine.py never read it.
        # The entire MWU adaptation path was dead code producing disk I/O and
        # CPU burn for zero effect. See Adversarial Review C6 for details.

        if entry_updated:
            updated += 1

 succeeded in 745ms:
        for t, t_data in stats.get("by_ticker", {}).items():
            ticker_stats[t]["correct"] += t_data["correct"]
            ticker_stats[t]["total"] += t_data["total"]

    return {
        t: {
            "accuracy": round(s["correct"] / s["total"], 3) if s["total"] else 0.0,
            "samples": s["total"],
        }
        for t, s in ticker_stats.items()
        if s["total"] > 0
    }


def backfill_forecast_outcomes(max_entries=500, predictions_file=None,
                               snapshot_file=None):
    """Backfill actual price outcomes into forecast predictions.

    For each prediction without an outcome, check if enough time has
    passed for the horizon, then look up the actual price and compute
    the change percentage.

    Returns number of entries updated.
    """
    path = predictions_file or PREDICTIONS_FILE
    entries = load_predictions(path)
    if not entries:
        return 0

    updated = 0
    modified_entries = []

    for entry in entries:
        if "outcome" not in entry:
            entry["outcome"] = {}

        ts_str = entry.get("ts", "")
        if not ts_str:
            modified_entries.append(entry)
            continue

        try:
            entry_time = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            modified_entries.append(entry)
            continue

        current_price = entry.get("current_price", 0)
        if not current_price:
            modified_entries.append(entry)
            continue

        entry_ticker = entry.get("ticker", "")
        if not entry_ticker:
            modified_entries.append(entry)
            continue

        now = datetime.now(UTC)

        for horizon_key, hours in [("1h", 1), ("24h", 24)]:
            if horizon_key in entry["outcome"]:
                continue  # already backfilled

            horizon_time = entry_time + timedelta(hours=hours)
            if now < horizon_time:
                continue  # not enough time passed

            # Look up actual price at horizon time
            actual_price = _lookup_price_at_time(
                entry_ticker, horizon_time, snapshot_file=snapshot_file
            )
            if actual_price is not None:
                change_pct = (actual_price - current_price) / current_price * 100
                entry["outcome"][horizon_key] = {
                    "actual_price": round(actual_price, 6),
                    "change_pct": round(change_pct, 4),
                    "backfilled_at": now.isoformat(),
                }
                updated += 1

        modified_entries.append(entry)

        if updated >= max_entries:
            break

    # Write back
    if updated > 0:
        _write_predictions(modified_entries, path)

    return updated


def _lookup_price_at_time(ticker, target_time, snapshot_file=None,
                           tolerance_hours: float = 2.0):
    """Look up the actual price for a ticker at a specific time.

    Uses hourly price snapshots from data/price_snapshots_hourly.jsonl
    and finds the closest entry within `tolerance_hours` of target_time.

    The default `tolerance_hours=2.0` preserves the original behavior used
    by `forecast_accuracy.compute_forecast_accuracy` and any other consumer
    that expects tight tolerance. Callers backfilling LLM/sentiment outcomes
    can pass a wider value (8h for crypto/metals' 24/7 markets, 24-72h for
    stocks where after-hours and weekend gaps are structural).

    2026-05-01 (deferred research / fix/missing-backfill-outcomes): added
    tolerance_hours parameter. Previously hardcoded to 2h, which silently
    dropped 2,200+ LLM probability rows whose target_time fell into either:
      - a loop-downtime gap (4-8h, all tickers symmetrically), or
      - the structural MSTR overnight gap (12-72h, US stock only).
    Documented in docs/PLAN_missing_backfills_20260501.md.
    """
    path = snapshot_file or (DATA_DIR / "price_snapshots_hourly.jsonl")
    if not path.exists():
        return None

    best_price = None
    best_delta = timedelta(hours=tolerance_hours)

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            snap = json.loads(line)
            snap_time = datetime.fromisoformat(snap.get("ts", ""))
            delta = abs(snap_time - target_time)
            if delta < best_delta:
                prices = snap.get("prices", {})
                if ticker in prices:
                    best_price = prices[ticker]
                    best_delta = delta
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return best_price


def _write_predictions(entries, predictions_file=None):
    """Write predictions back to JSONL file."""
    path = predictions_file or PREDICTIONS_FILE
    atomic_write_jsonl(path, entries)


def get_forecast_accuracy_summary(focus_tickers=None, days=7):
    """Get a compact accuracy summary for Layer 2 consumption.

    Args:
        focus_tickers: List of tickers to include (None = all with data).
        days: Lookback window.

    Returns:
        dict: {
            "health": {model: success_rate},
            "accuracy": {sub_signal: {accuracy, samples, by_ticker}},
        }
    """
    health = load_health_stats()
    accuracy = compute_forecast_accuracy(days=days)

    result = {
        "health": health,
        "accuracy": {},
    }

    for sub_name, stats in accuracy.items():
        entry = {
            "accuracy": stats["accuracy"],
            "samples": stats["total"],
        }
        if focus_tickers:
            by_ticker = {t: stats["by_ticker"][t]
                         for t in focus_tickers
                         if t in stats["by_ticker"]}
            if by_ticker:
                entry["by_ticker"] = by_ticker
        else:
            if stats["by_ticker"]:
                entry["by_ticker"] = stats["by_ticker"]
        result["accuracy"][sub_name] = entry

    return result


def print_forecast_accuracy_report():
    """Print a human-readable forecast accuracy report."""
    health = load_health_stats()

    print("=== Forecast Model Health ===")
    if not health:
        print("  No health data available yet.")
    else:
        for model, stats in sorted(health.items()):
            rate = stats["success_rate"] * 100
            print(f"  {model:10s}: {stats['ok']}/{stats['total']} ({rate:.1f}% success)")

    print("\n=== Forecast Sub-Signal Accuracy ===")
    accuracy = compute_forecast_accuracy()

    if not accuracy:
        print("  No outcome data available yet. Run --forecast-outcomes to backfill.")
        return

    for sub_name, stats in sorted(accuracy.items()):
        if stats["total"] == 0:
            continue
        acc = stats["accuracy"] * 100
        print(f"\n  {sub_name}:")
        print(f"    Overall: {acc:.1f}% ({stats['correct']}/{stats['total']})")
        for t, t_stats in sorted(stats["by_ticker"].items()):
            t_acc = t_stats["accuracy"] * 100
            print(f"    {t:10s}: {t_acc:.1f}% ({t_stats['correct']}/{t_stats['total']})")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\outcome_tracker.py | Select-Object -Last 80" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 642ms:

        if entry_updated:
            updated += 1

    if _db is not None:
        try:
            _db.close()
        except Exception as e:
            logger.debug("SignalDB close failed: %s", e)

    import os
    import tempfile

    # ---- Phase 3: rewrite (lock re-acquired) ----
    # SC-P1-3: hold the sidecar lock across the rewrite so that:
    #   1. We can stat the file and detect concurrent appends past
    #      snapshot_size without racing a partial append.
    #   2. The os.replace is atomic w.r.t. any subsequent appender —
    #      no appender can land bytes between our copy and our rename.
    with _hold_signal_log_lock():
        # Re-stat to find any new bytes appended during processing.
        try:
            current_size = SIGNAL_LOG.stat().st_size
        except FileNotFoundError:
            current_size = snapshot_size  # nothing to preserve
        concurrent_tail_bytes = max(0, current_size - snapshot_size)

        fd, tmp = tempfile.mkstemp(dir=SIGNAL_LOG.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f_out:
                # Stream head bytes verbatim from original file (no JSON parsing)
                if head_end_offset > 0:
                    with open(SIGNAL_LOG, "rb") as f_in:
                        remaining = head_end_offset
                        while remaining > 0:
                            chunk = f_in.read(min(65536, remaining))
                            if not chunk:
                                break
                            f_out.write(chunk)
                            remaining -= len(chunk)
                # Write modified tail entries
                for entry in entries:
                    f_out.write((json.dumps(entry) + "\n").encode("utf-8"))
                # SC-P1-3: copy bytes appended after our snapshot verbatim
                # so concurrent log_signal_snapshot() appends survive the
                # rewrite. We never parse them as JSON — preserves every
                # byte exactly as the appender wrote it (including the
                # trailing newline atomic_append_jsonl always emits).
                if concurrent_tail_bytes > 0:
                    with open(SIGNAL_LOG, "rb") as f_in:
                        f_in.seek(snapshot_size)
                        remaining = concurrent_tail_bytes
                        while remaining > 0:
                            chunk = f_in.read(min(65536, remaining))
                            if not chunk:
                                break
                            f_out.write(chunk)
                            remaining -= len(chunk)
            os.replace(tmp, SIGNAL_LOG)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # Invalidate signal utility cache so the next cycle picks up fresh
    # accuracy data immediately rather than waiting for the 300s TTL.
    if updated > 0:
        try:
            from portfolio.accuracy_stats import invalidate_signal_utility_cache
            invalidate_signal_utility_cache()
            logger.info("Signal utility cache invalidated after backfill (%d entries)", updated)
        except Exception:
            logger.debug("Could not invalidate signal utility cache", exc_info=True)

    return updated


if __name__ == "__main__":
    count = backfill_outcomes()
    print(f"Backfilled {count} entries")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, pathlib
mods = {}
for path in pathlib.Path('portfolio').glob('*.py'):
    tree = ast.parse(path.read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(('from', node.module, [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                imports.append(('import', a.name, []))
    print('\\n##', path.name)
    for kind, mod, names in imports:
        if mod.startswith('portfolio'):
            print(kind, mod, names)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 710ms:

## accuracy_degradation.py
from portfolio.file_utils ['atomic_write_json', 'load_json']
from portfolio.tickers ['SIGNAL_NAMES']
from portfolio.accuracy_stats ['accuracy_by_ticker_signal_cached', 'consensus_accuracy', 'load_entries', 'save_accuracy_snapshot', 'signal_accuracy']
from portfolio.forecast_accuracy ['cached_forecast_accuracy']
from portfolio.accuracy_stats ['accuracy_by_signal_ticker', 'load_entries']
from portfolio.accuracy_stats ['_load_accuracy_snapshots']
from portfolio.accuracy_stats ['_find_snapshot_near']
from portfolio.loop_contract ['Violation']
from portfolio.accuracy_stats ['consensus_accuracy', 'load_entries', 'signal_accuracy']
from portfolio.forecast_accuracy ['cached_forecast_accuracy']
from portfolio.accuracy_stats ['ACCURACY_SNAPSHOTS_FILE']
from portfolio.econ_dates ['events_within_hours', 'recent_high_impact_events']
from portfolio.claude_gate ['record_critical_error']
from portfolio.message_store ['send_or_store']

## accuracy_stats.py
from portfolio.file_utils ['atomic_write_json']
from portfolio.file_utils ['load_json', 'load_jsonl', 'load_jsonl_tail']
from portfolio.tickers ['DISABLED_SIGNALS', 'SIGNAL_NAMES']
from portfolio.file_utils ['atomic_append_jsonl']
from portfolio.signal_db ['SignalDB']
from portfolio.process_lock ['acquire_lock_file', 'release_lock_file']

## forecast_accuracy.py
from portfolio.file_utils ['atomic_write_jsonl']

## outcome_tracker.py
from portfolio.api_utils ['BINANCE_BASE', 'BINANCE_FAPI_BASE']
from portfolio.file_utils ['atomic_append_jsonl']
from portfolio.http_retry ['fetch_with_retry']
from portfolio.shared_state ['_yfinance_limiter']
from portfolio.tickers ['BINANCE_FAPI_MAP', 'BINANCE_SPOT_MAP', 'SIGNAL_NAMES', 'YF_MAP']
from portfolio.signal_db ['SignalDB']
from portfolio.price_source ['fetch_klines']
from portfolio.signal_db ['SignalDB']
from portfolio.accuracy_stats ['invalidate_signal_utility_cache']

## signal_db.py
from portfolio.tickers ['SIGNAL_NAMES']

## signal_engine.py
from portfolio.indicators ['detect_regime']
from portfolio.shared_state ['FEAR_GREED_TTL', 'FUNDING_RATE_TTL', 'MINISTRAL_TTL', 'ONCHAIN_TTL', 'SENTIMENT_TTL', 'VOLUME_TTL', '_cached', '_cached_or_enqueue']
from portfolio.signal_registry ['get_enhanced_signals', 'load_signal_func']
from portfolio.signal_utils ['true_range']
from portfolio.tickers ['CRYPTO_SYMBOLS', 'DISABLED_SIGNALS', 'GPU_SIGNALS', 'METALS_SYMBOLS', 'SIGNAL_NAMES', 'STOCK_SYMBOLS']
from portfolio.market_timing ['should_skip_gpu']
from portfolio.file_utils ['atomic_write_json']
from portfolio.file_utils ['load_json']
from portfolio.accuracy_stats ['load_entries']
from portfolio.tickers ['SIGNAL_NAMES']
from portfolio.ic_computation ['compute_and_cache_ic', 'load_cached_ic']
from portfolio.short_horizon ['time_of_day_scale_3h']
from portfolio.fear_greed ['get_fear_greed', 'get_sustained_fear_days', 'update_fear_streak']
from portfolio.llm_probability_log ['derive_probs_from_result', 'llm_signals', 'log_vote']
from portfolio.short_horizon ['is_slow_signal_3h']
from portfolio.accuracy_stats ['accuracy_by_ticker_signal_cached']
from portfolio.accuracy_stats ['get_or_compute_recent_accuracy']
from portfolio.accuracy_stats ['blend_accuracy_data', 'get_or_compute_accuracy', 'get_or_compute_per_ticker_accuracy', 'get_or_compute_recent_accuracy', 'load_cached_activation_rates']
from portfolio.market_health ['get_confidence_penalty', 'get_market_health']
from portfolio.linear_factor ['LinearFactorModel']
from portfolio.short_horizon ['CONFIDENCE_CAP_3H']
from portfolio.file_utils ['load_json']
from portfolio.tickers ['ALL_TICKERS']
from portfolio.econ_dates ['is_macro_window']
from portfolio.accuracy_stats ['accuracy_by_signal_ticker']
from portfolio.social_sentiment ['get_reddit_posts']
from portfolio.sentiment ['get_sentiment']
from portfolio.funding_rate ['get_funding_rate']
from portfolio.onchain_data ['get_onchain_data']
from portfolio.macro_context ['get_volume_signal']
from portfolio.llm_batch ['enqueue_ministral', 'is_llm_on_cycle']
from portfolio.llm_batch ['enqueue_qwen3', 'is_llm_on_cycle']
from portfolio.health ['update_signal_health_batch']
from portfolio.accuracy_stats ['get_or_compute_regime_accuracy']
from portfolio.accuracy_stats ['signal_utility']
from portfolio.accuracy_stats ['signal_best_horizon_accuracy']
from portfolio.earnings_calendar ['should_gate_earnings']
from portfolio.accuracy_stats ['load_cached_accuracy']
from portfolio.macro_context ['get_dxy', 'get_fed_calendar', 'get_treasury']
from portfolio.seasonality ['get_profile']

## signal_history.py
from portfolio.file_utils ['atomic_write_jsonl', 'load_jsonl']
from portfolio.tickers ['SIGNAL_NAMES']

## signal_postmortem.py
from portfolio.file_utils ['atomic_write_json', 'load_json']
from portfolio.accuracy_stats ['load_cached_accuracy', 'load_cached_regime_accuracy', 'signal_accuracy', 'signal_accuracy_by_regime']
from portfolio.accuracy_stats ['load_entries']

## signal_registry.py

## signal_utils.py

## signal_weights.py
from portfolio.file_utils ['atomic_write_json', 'load_json']

## signal_weight_optimizer.py
from portfolio.file_utils ['atomic_write_json', 'load_json']
from portfolio.linear_factor ['LinearFactorModel']

## ticker_accuracy.py
from portfolio.tickers ['DISABLED_SIGNALS', 'SIGNAL_NAMES']
from portfolio.accuracy_stats ['_vote_correct', 'load_entries']

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path_str, start, end in [('portfolio/forecast_accuracy.py',300,335), ('portfolio/signal_weights.py',100,125)]:
    path=Path(path_str)
    for i,line in enumerate(path.read_text().splitlines(), start=1):
        if start <= i <= end:
            print(f'{path_str}:{i}: {line}')
    print('---')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 676ms:
portfolio/forecast_accuracy.py:300:             continue
portfolio/forecast_accuracy.py:301: 
portfolio/forecast_accuracy.py:302:         try:
portfolio/forecast_accuracy.py:303:             entry_time = datetime.fromisoformat(ts_str)
portfolio/forecast_accuracy.py:304:         except (ValueError, TypeError):
portfolio/forecast_accuracy.py:305:             modified_entries.append(entry)
portfolio/forecast_accuracy.py:306:             continue
portfolio/forecast_accuracy.py:307: 
portfolio/forecast_accuracy.py:308:         current_price = entry.get("current_price", 0)
portfolio/forecast_accuracy.py:309:         if not current_price:
portfolio/forecast_accuracy.py:310:             modified_entries.append(entry)
portfolio/forecast_accuracy.py:311:             continue
portfolio/forecast_accuracy.py:312: 
portfolio/forecast_accuracy.py:313:         entry_ticker = entry.get("ticker", "")
portfolio/forecast_accuracy.py:314:         if not entry_ticker:
portfolio/forecast_accuracy.py:315:             modified_entries.append(entry)
portfolio/forecast_accuracy.py:316:             continue
portfolio/forecast_accuracy.py:317: 
portfolio/forecast_accuracy.py:318:         now = datetime.now(UTC)
portfolio/forecast_accuracy.py:319: 
portfolio/forecast_accuracy.py:320:         for horizon_key, hours in [("1h", 1), ("24h", 24)]:
portfolio/forecast_accuracy.py:321:             if horizon_key in entry["outcome"]:
portfolio/forecast_accuracy.py:322:                 continue  # already backfilled
portfolio/forecast_accuracy.py:323: 
portfolio/forecast_accuracy.py:324:             horizon_time = entry_time + timedelta(hours=hours)
portfolio/forecast_accuracy.py:325:             if now < horizon_time:
portfolio/forecast_accuracy.py:326:                 continue  # not enough time passed
portfolio/forecast_accuracy.py:327: 
portfolio/forecast_accuracy.py:328:             # Look up actual price at horizon time
portfolio/forecast_accuracy.py:329:             actual_price = _lookup_price_at_time(
portfolio/forecast_accuracy.py:330:                 entry_ticker, horizon_time, snapshot_file=snapshot_file
portfolio/forecast_accuracy.py:331:             )
portfolio/forecast_accuracy.py:332:             if actual_price is not None:
portfolio/forecast_accuracy.py:333:                 change_pct = (actual_price - current_price) / current_price * 100
portfolio/forecast_accuracy.py:334:                 entry["outcome"][horizon_key] = {
portfolio/forecast_accuracy.py:335:                     "actual_price": round(actual_price, 6),
---
portfolio/signal_weights.py:100:             "eta": self._eta,
portfolio/signal_weights.py:101:             "weights": self._weights,
portfolio/signal_weights.py:102:         }
portfolio/signal_weights.py:103:         atomic_write_json(self._path, payload)
portfolio/signal_weights.py:104: 
portfolio/signal_weights.py:105:     # ------------------------------------------------------------------
portfolio/signal_weights.py:106:     # Internal helpers
portfolio/signal_weights.py:107:     # ------------------------------------------------------------------
portfolio/signal_weights.py:108: 
portfolio/signal_weights.py:109:     def _load(self) -> None:
portfolio/signal_weights.py:110:         """Load weights from disk.  No-ops silently if the file is missing."""
portfolio/signal_weights.py:111:         data = load_json(self._path, default=None)
portfolio/signal_weights.py:112:         if data is None:
portfolio/signal_weights.py:113:             return
portfolio/signal_weights.py:114:         if isinstance(data, dict):
portfolio/signal_weights.py:115:             self._weights = {
portfolio/signal_weights.py:116:                 k: float(v)
portfolio/signal_weights.py:117:                 for k, v in data.get("weights", {}).items()
portfolio/signal_weights.py:118:             }
portfolio/signal_weights.py:119:             # Honour stored eta only if caller did not override it
portfolio/signal_weights.py:120:             # (caller passes None → _DEFAULT_ETA, so we preserve stored value)
---

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path=Path('portfolio/forecast_accuracy.py')
for i,line in enumerate(path.read_text().splitlines(), start=1):
    if 333 <= i <= 360:
        print(f'{path}:{i}: {line}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 926ms:
portfolio\forecast_accuracy.py:333:                 change_pct = (actual_price - current_price) / current_price * 100
portfolio\forecast_accuracy.py:334:                 entry["outcome"][horizon_key] = {
portfolio\forecast_accuracy.py:335:                     "actual_price": round(actual_price, 6),
portfolio\forecast_accuracy.py:336:                     "change_pct": round(change_pct, 4),
portfolio\forecast_accuracy.py:337:                     "backfilled_at": now.isoformat(),
portfolio\forecast_accuracy.py:338:                 }
portfolio\forecast_accuracy.py:339:                 updated += 1
portfolio\forecast_accuracy.py:340: 
portfolio\forecast_accuracy.py:341:         modified_entries.append(entry)
portfolio\forecast_accuracy.py:342: 
portfolio\forecast_accuracy.py:343:         if updated >= max_entries:
portfolio\forecast_accuracy.py:344:             break
portfolio\forecast_accuracy.py:345: 
portfolio\forecast_accuracy.py:346:     # Write back
portfolio\forecast_accuracy.py:347:     if updated > 0:
portfolio\forecast_accuracy.py:348:         _write_predictions(modified_entries, path)
portfolio\forecast_accuracy.py:349: 
portfolio\forecast_accuracy.py:350:     return updated
portfolio\forecast_accuracy.py:351: 
portfolio\forecast_accuracy.py:352: 
portfolio\forecast_accuracy.py:353: def _lookup_price_at_time(ticker, target_time, snapshot_file=None,
portfolio\forecast_accuracy.py:354:                            tolerance_hours: float = 2.0):
portfolio\forecast_accuracy.py:355:     """Look up the actual price for a ticker at a specific time.
portfolio\forecast_accuracy.py:356: 
portfolio\forecast_accuracy.py:357:     Uses hourly price snapshots from data/price_snapshots_hourly.jsonl
portfolio\forecast_accuracy.py:358:     and finds the closest entry within `tolerance_hours` of target_time.
portfolio\forecast_accuracy.py:359: 
portfolio\forecast_accuracy.py:360:     The default `tolerance_hours=2.0` preserves the original behavior used

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem portfolio\\*.py | Select-String -Pattern 'if .*>= max_entries|break'" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 671ms:

portfolio\accuracy_stats.py:1245:                [7mbreak[0m
portfolio\forecast_accuracy.py:343:        [7mif updated >= max_entries[0m:
portfolio\forecast_accuracy.py:344:            [7mbreak[0m
portfolio\outcome_tracker.py:122:        # regime-gated signals, [7mbreak[0ming the dead-signal trap.
portfolio\outcome_tracker.py:535:                                [7mbreak[0m
portfolio\outcome_tracker.py:553:                                [7mbreak[0m
portfolio\signal_engine.py:79:# full phase [7mbreak[0mdown retrospectively. Bounded per-ticker (replaced on
portfolio\signal_engine.py:132:    """Return the phase [7mbreak[0mdown for a ticker's last generate_signal call.
portfolio\signal_engine.py:207:# regime that [7mbreak[0ms fundamental assumptions (e.g., wartime, systemic crisis).
portfolio\signal_engine.py:357:# Voter-count circuit [7mbreak[0mer (2026-04-16, Batch 2 of accuracy gating reconfig).
portfolio\signal_engine.py:606:# circuit-[7mbreak[0mer precondition. Placing here (after MIN_VOTERS_*) keeps the
portfolio\signal_engine.py:627:    f"otherwise the circuit [7mbreak[0mer requires more candidates than it can "
portfolio\signal_engine.py:632:    "circuit-[7mbreak[0mer step-count math).")
portfolio\signal_engine.py:829:        # Gating at 1d prevents false SELL consensus during [7mbreak[0mouts.
portfolio\signal_engine.py:1436:    # negative edge — circuit-[7mbreak[0mer relaxation must not promote it back to
portfolio\signal_engine.py:1461:        # Directional gate is not relaxed by the circuit [7mbreak[0mer — those gates
portfolio\signal_engine.py:1499:    the circuit [7mbreak[0mer and apply_confidence_penalties.
portfolio\signal_engine.py:1502:    emitting a non-HOLD action. The circuit [7mbreak[0mer uses it to size its
portfolio\signal_engine.py:1521:    """Compute circuit-[7mbreak[0mer relaxation to preserve voter diversity.
portfolio\signal_engine.py:1531:                                                or a genuine regime [7mbreak[0m
portfolio\signal_engine.py:1610:    # Regime [7mbreak[0m: relaxation recovers nothing beyond baseline. Keep the
portfolio\signal_engine.py:1997:    # trend-following [7mbreak[0ms and mean-reversion becomes more reliable.
portfolio\signal_engine.py:2042:    # Voter-count circuit [7mbreak[0mer (Batch 2 of 2026-04-16 accuracy gating reconfig).
portfolio\signal_engine.py:2056:            "Circuit [7mbreak[0mer: relaxing accuracy gate by %.0fpp "
portfolio\signal_engine.py:2094:        # circuit-[7mbreak[0mer relaxation must not let it back in. The standard
portfolio\signal_engine.py:2504:    # circuit [7mbreak[0mer's recovery-floor logic. Same semantic as before.
portfolio\signal_engine.py:3207:    # This allows accuracy tracking for regime-gated signals, [7mbreak[0ming the
portfolio\signal_history.py:191:                [7mbreak[0m
portfolio\signal_postmortem.py:215:        # Regime [7mbreak[0mdown


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path_str, ranges in [('portfolio/outcome_tracker.py', [(520,560)]), ('portfolio/accuracy_stats.py', [(1228,1252)])]:
    path=Path(path_str)
    for start,end in ranges:
        for i,line in enumerate(path.read_text().splitlines(), start=1):
            if start <= i <= end:
                print(f'{path}:{i}: {line}')
        print('---')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 590ms:
portfolio\outcome_tracker.py:520:             current_size = SIGNAL_LOG.stat().st_size
portfolio\outcome_tracker.py:521:         except FileNotFoundError:
portfolio\outcome_tracker.py:522:             current_size = snapshot_size  # nothing to preserve
portfolio\outcome_tracker.py:523:         concurrent_tail_bytes = max(0, current_size - snapshot_size)
portfolio\outcome_tracker.py:524: 
portfolio\outcome_tracker.py:525:         fd, tmp = tempfile.mkstemp(dir=SIGNAL_LOG.parent, suffix=".tmp")
portfolio\outcome_tracker.py:526:         try:
portfolio\outcome_tracker.py:527:             with os.fdopen(fd, "wb") as f_out:
portfolio\outcome_tracker.py:528:                 # Stream head bytes verbatim from original file (no JSON parsing)
portfolio\outcome_tracker.py:529:                 if head_end_offset > 0:
portfolio\outcome_tracker.py:530:                     with open(SIGNAL_LOG, "rb") as f_in:
portfolio\outcome_tracker.py:531:                         remaining = head_end_offset
portfolio\outcome_tracker.py:532:                         while remaining > 0:
portfolio\outcome_tracker.py:533:                             chunk = f_in.read(min(65536, remaining))
portfolio\outcome_tracker.py:534:                             if not chunk:
portfolio\outcome_tracker.py:535:                                 break
portfolio\outcome_tracker.py:536:                             f_out.write(chunk)
portfolio\outcome_tracker.py:537:                             remaining -= len(chunk)
portfolio\outcome_tracker.py:538:                 # Write modified tail entries
portfolio\outcome_tracker.py:539:                 for entry in entries:
portfolio\outcome_tracker.py:540:                     f_out.write((json.dumps(entry) + "\n").encode("utf-8"))
portfolio\outcome_tracker.py:541:                 # SC-P1-3: copy bytes appended after our snapshot verbatim
portfolio\outcome_tracker.py:542:                 # so concurrent log_signal_snapshot() appends survive the
portfolio\outcome_tracker.py:543:                 # rewrite. We never parse them as JSON — preserves every
portfolio\outcome_tracker.py:544:                 # byte exactly as the appender wrote it (including the
portfolio\outcome_tracker.py:545:                 # trailing newline atomic_append_jsonl always emits).
portfolio\outcome_tracker.py:546:                 if concurrent_tail_bytes > 0:
portfolio\outcome_tracker.py:547:                     with open(SIGNAL_LOG, "rb") as f_in:
portfolio\outcome_tracker.py:548:                         f_in.seek(snapshot_size)
portfolio\outcome_tracker.py:549:                         remaining = concurrent_tail_bytes
portfolio\outcome_tracker.py:550:                         while remaining > 0:
portfolio\outcome_tracker.py:551:                             chunk = f_in.read(min(65536, remaining))
portfolio\outcome_tracker.py:552:                             if not chunk:
portfolio\outcome_tracker.py:553:                                 break
portfolio\outcome_tracker.py:554:                             f_out.write(chunk)
portfolio\outcome_tracker.py:555:                             remaining -= len(chunk)
portfolio\outcome_tracker.py:556:             os.replace(tmp, SIGNAL_LOG)
portfolio\outcome_tracker.py:557:         except BaseException:
portfolio\outcome_tracker.py:558:             with contextlib.suppress(OSError):
portfolio\outcome_tracker.py:559:                 os.unlink(tmp)
portfolio\outcome_tracker.py:560:             raise
---
portfolio\accuracy_stats.py:1228:         _save_prewarm_ts_to_disk(t)
portfolio\accuracy_stats.py:1229:         return True
portfolio\accuracy_stats.py:1230:     except Exception:
portfolio\accuracy_stats.py:1231:         logger.debug("maybe_prewarm_dashboard_accuracy failed", exc_info=True)
portfolio\accuracy_stats.py:1232:         return False
portfolio\accuracy_stats.py:1233:     finally:
portfolio\accuracy_stats.py:1234:         if release_lock_file and fh != "noop":
portfolio\accuracy_stats.py:1235:             release_lock_file(fh)
portfolio\accuracy_stats.py:1236: 
portfolio\accuracy_stats.py:1237: 
portfolio\accuracy_stats.py:1238: def _count_entries_with_outcomes(entries, horizon):
portfolio\accuracy_stats.py:1239:     count = 0
portfolio\accuracy_stats.py:1240:     for entry in entries:
portfolio\accuracy_stats.py:1241:         outcomes = entry.get("outcomes", {})
portfolio\accuracy_stats.py:1242:         for _ticker, horizons in outcomes.items():
portfolio\accuracy_stats.py:1243:             if horizons.get(horizon):
portfolio\accuracy_stats.py:1244:                 count += 1
portfolio\accuracy_stats.py:1245:                 break
portfolio\accuracy_stats.py:1246:     return count
portfolio\accuracy_stats.py:1247: 
portfolio\accuracy_stats.py:1248: 
portfolio\accuracy_stats.py:1249: def print_accuracy_report():
portfolio\accuracy_stats.py:1250:     entries = load_entries()
portfolio\accuracy_stats.py:1251:     if not entries:
portfolio\accuracy_stats.py:1252:         print("No signal log data found.")
---

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'raw_sub_signals|sub_signals|chronos_'" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 691ms:

portfolio\accuracy_degradation.py:179:            horizon="24h", days=days, use_[7mraw_sub_signals[0m=True,
portfolio\accuracy_degradation.py:509:            use_[7mraw_sub_signals[0m=True,
portfolio\forecast_accuracy.py:35:def cached_forecast_accuracy(horizon="24h", days=7, use_[7mraw_sub_signals[0m=True):
portfolio\forecast_accuracy.py:38:    Keyed by (horizon, days, use_[7mraw_sub_signals[0m). 1h TTL — long enough to
portfolio\forecast_accuracy.py:43:    key = (horizon, days, use_[7mraw_sub_signals[0m)
portfolio\forecast_accuracy.py:51:        horizon=horizon, days=days, use_[7mraw_sub_signals[0m=use_raw_sub_signals,
portfolio\forecast_accuracy.py:119:                              predictions_file=None, use_[7mraw_sub_signals[0m=False):
portfolio\forecast_accuracy.py:166:        [7msub_signals[0m = entry.get("sub_signals", {})
portfolio\forecast_accuracy.py:167:        if use_[7mraw_sub_signals[0m and entry.get("raw_sub_signals"):
portfolio\forecast_accuracy.py:168:            [7msub_signals[0m = entry.get("raw_sub_signals", {})
portfolio\forecast_accuracy.py:169:        for sub_name, vote in [7msub_signals[0m.items():
portfolio\signal_engine.py:1127:    and [7msub_signals[0m is a dict. Returns a clean dict, always.
portfolio\signal_engine.py:1130:        return {"action": "HOLD", "confidence": 0.0, "[7msub_signals[0m": {}}
portfolio\signal_engine.py:1149:    [7msub_signals[0m = result.get("sub_signals")
portfolio\signal_engine.py:1150:    if not isinstance([7msub_signals[0m, dict):
portfolio\signal_engine.py:1151:        [7msub_signals[0m = {}
portfolio\signal_engine.py:1156:        "[7msub_signals[0m": sub_signals,
portfolio\signal_engine.py:3105:                extra_info[f"{sig_name}_[7msub_signals[0m"] = validated["sub_signals"]
portfolio\ticker_accuracy.py:188:            {"[7mchronos_[0m24h_pct": 0.5, "chronos_24h_conf": 0.6,
portfolio\ticker_accuracy.py:189:             "[7mchronos_[0m1h_pct": 0.3, "chronos_1h_conf": 0.55,
portfolio\ticker_accuracy.py:190:             "[7mchronos_[0mok": True}
portfolio\ticker_accuracy.py:205:    if not forecast_data or not forecast_data.get("[7mchronos_[0mok"):
portfolio\ticker_accuracy.py:211:    [7mchronos_[0mh = horizon_map.get(horizon, "24h")
portfolio\ticker_accuracy.py:213:    pct_key = f"[7mchronos_[0m{chronos_h}_pct"
portfolio\ticker_accuracy.py:214:    conf_key = f"[7mchronos_[0m{chronos_h}_conf"
portfolio\__pycache__\accuracy_degradation.cpython-312.pyc:63:per_tickerz0Lifetime per-ticker accuracy snapshot 
failed: %s�r3   r@   �per_ticker_recentz.Recent per-ticker accuracy snapshot failed: 
%s�24hT��horizonr3   �use_[7mraw_sub_signals[0m�forecast_recentz%Forecast [0m[7m[0maccuracy [0m[7m[0msnapshot [0m[7m[0mfailed: [0m[7m[0m%s�	[0m
[7m[0mconsensus�consensus_recentz&Consensus [0m[7m[0maccuracy [0m[7m[0msnapshot [0m[7m[0mfailed: [0m[7m[0m%s)�extras)r   r   �portfolio.accuracy_statsr6  [0m
[7m[0m r7   r8   r9   r:   �portfolio.forecast_accuracyr<   �nowr   �	[0m[7m[0misoformat�get�items�	[0m[7m[0mException�logger�warning�_[0m
[7m[0mcompact_per_ticker�_per_ticker_recent)r3   �_tdr6   r7   r8   r9   r:   r<   rO   �cutoff�all_entries�e�recent_e[0m
[7m[0mntries�recent�namer   �per_ticker_lifetimerG   rL   �statss    [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m
[7m[0mr$   �save_full_accuracy_snapshotrd   w   sh  � � *�� � E��F��l�l�3��#�4�.�0�[0m
portfolio\__pycache__\forecast_accuracy.cpython-312.pyc:15:    Keyed by (horizon, days, use_[7mraw_sub_signals[0m). [0m[7m[0m1h [0m[7m[0mTTL [0m[7m[0m— [0m
[7m[0mlong [0m[7m[0menough [0m[7m[0mto[0m
portfolio\__pycache__\forecast_accuracy.cpython-312.pyc:19:    r   �   N)�horizon�days�use_[7mraw_sub_signals[0m)�tim[0m
[7m[0me�_forecast_accuracy_cache_lockr   �get�_FORECAST_ACCURACY_CACHE_TTL�compute_forecast_accuracy)r   r   r   �k[0m
[7m[0mey�now�cached�results    [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m[7m [0m
[7m[0m�portfolio\forecast_accuracy.py�cached_forecast_accuracyr   #   s�   � � �D�-�[0m
portfolio\__pycache__\forecast_accuracy.cpython-312.pyc:88:change_pctr   �[7msub_signals[0m�raw_sub_signals�HOLD�_r
portfolio\__pycache__\signal_engine.cpython-312.pyc:181:    and [7msub_signals[0m is a dict. Returns a clean dict, always.
portfolio\__pycache__\signal_engine.cpython-312.pyc:183:confidence�[7msub_signals[0mr  z8Signal [0m[7m[0m%s [0m[7m[0mreturned [0m[7m[0minvalid [0m
[7m[0maction=%r, [0m[7m[0mdefaulting [0m[7m[0mto [0m[7m[0mHOLDr  z>Signal [0m[7m[0m%s [0m[7m[0mreturned [0m[7m[0mnon-finite [0m[7m[0mconfidence=%r, [0m[7m[0mdefaulting [0m[7m[0mto [0m[7m[0m0.0r  �[0m
portfolio\__pycache__\signal_engine.cpython-312.pyc:709:qwen3_samples�qwen3_gating�qwen3_confidence�qwen3_signal�  
 c              3   �>   K  � | ]  }|j                  d �      �� � y�w)�requires_macroN�r&   )r�
  rL  s     r#   r�  z"generate_signal.<locals>.<genexpr>�  s   � �� �\�A[�A����&6� 
7�A[�s   �)�get_dxy�get_fed_calendar�get_treasury�dxyrP   �treasury�fedzMacro context fetch 
failedr�   )�get_profilez&Seasonality profile load failed for 
%s)r   r�  �macror�   �seasonality_profilezshadow:�requires_context)�contextrk  )rt  z[SLOW-SHADOW] 
%s/%s: %.1fsr  )r   r  �_action�_confidence�shadow_zShadow signal %s failed: %sr  z[SLOW] %s/%s: 
%.1fsr  �_[7msub_signals[0mr  �_indicatorszSignal [0m[7m[0m%s [0m[7m[0mfailed: [0m[7m[0m%s�_signal_failuresrH   z"%s: [0m[7m[0m%d [0m[7m[0menhanced [0m[7m[0msignals [0m
[7m[0mfailed: [0m[7m[0m%sz, [0m[7m[0m)�update_signal_health_batchzSignal [0m[7m[0mhealth [0m[7m[0mtracking [0m[7m[0mfailedr�   r2  �	[0m[7m[0mbtc_proxy�btc_proxy_actionr8   [0m
[7m[0m�btc_proxy_confidence�cross_ticker_cache�btc_proxy_source�__post_dispatch__)�derive_probs_from_result�llm_signal[0m
[7m[0ms�log_vote)r  rV   r  )r  r~   �tierr�   )r�   �chosenr  r�  zllm [0m[7m[0mprobability [0m[7m[0mlogging [0m[7m[0mfailed�[0m
portfolio\__pycache__\ticker_accuracy.cpython-312.pyc:120:            {"[7mchronos_[0m24h_pct": 0.5, "chronos_24h_conf": 0.6,
portfolio\__pycache__\ticker_accuracy.cpython-312.pyc:121:             "[7mchronos_[0m1h_pct": 0.3, "chronos_1h_conf": 0.55,
portfolio\__pycache__\ticker_accuracy.cpython-312.pyc:122:             "[7mchronos_[0mok": True}
portfolio\__pycache__\ticker_accuracy.cpython-312.pyc:133:[7mchronos_[0mokF�forecast_blended�1h�24h)rQ   �3h�1d�3drR  [0m
[7m[0m �chronos_�_pct�_confr   r/   �forecast_pct_move�forecast_confidenceg�������?r5   g333333ӿg333333�?g       @c   [0m
[7m[0m           3   �@   K  � | ]  }|j                  d d�      �� � y�w)r3   r   N)r   )�.0�ds   [0m
[7m[0m [0m[7m [0m[7m[0mr)   �	[0m[7m[0m<genexpr>z6direction_probability_with_forecast.<locals>.<genexpr>�   s   � �� �Y�:X�Q�a�e�e�H�a�0�:X�s   �[0m
[7m[0mr:   r7   r;   r<   r6   r=   r>   r4   T)rL   r   rC   �abs�min�max�sum)r   rD   �[0m
portfolio\__pycache__\ticker_accuracy.cpython-312.pyc:134:forecast_datar   r   rE   �base�horizon_map�	
[7mchronos_[0mh�pct_key�conf_key�pct_move�[0m


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\signal_postmortem.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 649ms:
"""Signal postmortem — analyze WHY signals fail by regime, ticker, and time.

Reads accuracy data broken down by regime and identifies patterns:
- Which signals work in which regimes (and fail in others)
- Which tickers are unpredictable for specific signals
- Signal correlation clusters (vote agreement rates)

Output goes to data/signal_postmortem.json for Layer 2 context and
periodic review by the after-hours research agent.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.signal_postmortem")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSTMORTEM_FILE = DATA_DIR / "signal_postmortem.json"

# Minimum samples for reliable analysis
MIN_SAMPLES = 15

# Thresholds for classification
STRONG_THRESHOLD = 0.60  # >=60% = signal works
WEAK_THRESHOLD = 0.45    # <45% = signal is noise/harmful
DIVERGENCE_THRESHOLD = 0.15  # 15pp regime divergence = regime-dependent


def compute_regime_insights(regime_accuracy: dict) -> list[dict]:
    """Identify signals that perform very differently across regimes.

    Finds signals where accuracy in one regime is >15pp different from another.
    These are regime-dependent signals that should be gated or boosted.

    Args:
        regime_accuracy: Output of accuracy_stats.signal_accuracy_by_regime()

    Returns:
        List of insight dicts with signal, best_regime, worst_regime, spread.
    """
    if not regime_accuracy:
        return []

    # Collect per-signal accuracy across regimes
    signal_regimes: dict[str, dict[str, dict]] = {}
    for regime, sig_map in regime_accuracy.items():
        for sig, stats in sig_map.items():
            if stats.get("total", 0) < MIN_SAMPLES:
                continue
            signal_regimes.setdefault(sig, {})[regime] = stats

    insights = []
    for sig, regimes in signal_regimes.items():
        if len(regimes) < 2:
            continue

        accs = {r: s["accuracy"] for r, s in regimes.items()}
        best_regime = max(accs, key=accs.get)
        worst_regime = min(accs, key=accs.get)
        spread = accs[best_regime] - accs[worst_regime]

        if spread >= DIVERGENCE_THRESHOLD:
            insights.append({
                "signal": sig,
                "type": "regime_dependent",
                "best_regime": best_regime,
                "best_accuracy": round(accs[best_regime] * 100, 1),
                "best_samples": regimes[best_regime]["total"],
                "worst_regime": worst_regime,
                "worst_accuracy": round(accs[worst_regime] * 100, 1),
                "worst_samples": regimes[worst_regime]["total"],
                "spread_pp": round(spread * 100, 1),
                "recommendation": (
                    f"Gate {sig} in {worst_regime} regime "
                    f"({accs[worst_regime]*100:.0f}%) — "
                    f"it works in {best_regime} ({accs[best_regime]*100:.0f}%)"
                ),
            })

    insights.sort(key=lambda x: x["spread_pp"], reverse=True)
    return insights


def compute_signal_health_report(accuracy_data: dict) -> list[dict]:
    """Classify signals into strong, weak, and marginal categories.

    Args:
        accuracy_data: Standard accuracy dict {signal: {accuracy, total, ...}}

    Returns:
        List of signal health dicts sorted by accuracy.
    """
    report = []
    for sig, stats in accuracy_data.items():
        total = stats.get("total", 0)
        if total < MIN_SAMPLES:
            continue

        acc = stats.get("accuracy", 0.5)
        if acc >= STRONG_THRESHOLD:
            category = "strong"
        elif acc < WEAK_THRESHOLD:
            category = "weak"
        else:
            category = "marginal"

        report.append({
            "signal": sig,
            "accuracy_pct": round(acc * 100, 1),
            "samples": total,
            "category": category,
        })

    report.sort(key=lambda x: x["accuracy_pct"], reverse=True)
    return report


def compute_vote_correlation(entries: list[dict] | None = None) -> list[dict]:
    """Compute pairwise signal vote agreement rates.

    Analyzes signal_log entries to find which signals frequently agree.
    High agreement (>80%) suggests redundancy — one signal adds no
    information beyond what the other provides.

    Args:
        entries: Pre-loaded signal_log entries. If None, loads from disk.

    Returns:
        List of correlated pairs sorted by agreement rate.
    """
    if entries is None:
        try:
            from portfolio.accuracy_stats import load_entries
            entries = load_entries()
        except Exception:
            logger.warning("Could not load signal_log entries for correlation analysis")
            return []

    if not entries:
        return []

    # Count pairwise agreement
    from collections import defaultdict
    pair_agree = defaultdict(int)
    pair_total = defaultdict(int)

    for entry in entries:
        tickers = entry.get("tickers", {})
        for _ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            # Only count signals that are actually voting (non-HOLD)
            active = {s: v for s, v in signals.items() if v != "HOLD"}
            active_names = sorted(active.keys())

            for i, s1 in enumerate(active_names):
                for s2 in active_names[i + 1:]:
                    pair = (s1, s2)
                    pair_total[pair] += 1
                    if active[s1] == active[s2]:
                        pair_agree[pair] += 1

    # Compute agreement rates
    correlations = []
    for pair, total in pair_total.items():
        if total < 30:  # need enough co-occurrences
            continue
        agree = pair_agree.get(pair, 0)
        rate = agree / total
        if rate >= 0.70:  # only report high correlations
            correlations.append({
                "signal_a": pair[0],
                "signal_b": pair[1],
                "agreement_rate": round(rate, 3),
                "co_occurrences": total,
                "agrees": agree,
            })

    correlations.sort(key=lambda x: x["agreement_rate"], reverse=True)
    return correlations


def generate_postmortem() -> dict:
    """Generate a complete signal postmortem report.

    Combines regime insights, health classification, and correlation analysis.
    Writes to data/signal_postmortem.json.
    """
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "regime_insights": [],
        "signal_health": [],
        "correlations": [],
        "summary": {},
    }

    # Regime-dependent analysis
    try:
        from portfolio.accuracy_stats import (
            load_cached_accuracy,
            load_cached_regime_accuracy,
            signal_accuracy,
            signal_accuracy_by_regime,
        )

        # Overall accuracy
        acc = load_cached_accuracy("1d")
        if not acc:
            acc = signal_accuracy("1d")
        if acc:
            report["signal_health"] = compute_signal_health_report(acc)

        # Regime breakdown
        regime_acc = load_cached_regime_accuracy("1d")
        if not regime_acc:
            regime_acc = signal_accuracy_by_regime("1d")
        if regime_acc:
            report["regime_insights"] = compute_regime_insights(regime_acc)

    except Exception:
        logger.warning("Accuracy data unavailable for postmortem", exc_info=True)

    # Correlation analysis
    try:
        report["correlations"] = compute_vote_correlation()
    except Exception:
        logger.warning("Correlation analysis failed", exc_info=True)

    # Summary
    strong = [s for s in report["signal_health"] if s["category"] == "strong"]
    weak = [s for s in report["signal_health"] if s["category"] == "weak"]
    report["summary"] = {
        "strong_signals": len(strong),
        "weak_signals": len(weak),
        "regime_dependent": len(report["regime_insights"]),
        "correlated_pairs": len(report["correlations"]),
        "top_3_strong": [s["signal"] for s in strong[:3]],
        "top_3_weak": [s["signal"] for s in weak[:3]],
    }

    atomic_write_json(POSTMORTEM_FILE, report)
    logger.info(
        "Signal postmortem: %d strong, %d weak, %d regime-dependent, %d correlated pairs",
        len(strong), len(weak), len(report["regime_insights"]), len(report["correlations"]),
    )

    return report


def get_postmortem_context() -> dict | None:
    """Load cached postmortem for inclusion in agent_summary.

    Returns compact version suitable for Layer 2 context.
    """
    data = load_json(POSTMORTEM_FILE)
    if not data:
        return None

    # Return compact version — just summary + top insights
    return {
        "summary": data.get("summary", {}),
        "top_regime_insights": data.get("regime_insights", [])[:5],
        "top_correlations": data.get("correlations", [])[:5],
    }

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\accuracy_stats.py -First 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\accuracy_stats.py -TotalCount 520 | Select-Object -Skip 260" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\accuracy_stats.py -TotalCount 1040 | Select-Object -Skip 520" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\accuracy_stats.py -TotalCount 1560 | Select-Object -Skip 1040" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content portfolio\\accuracy_stats.py | Select-Object -Skip 1560" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 771ms:
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("portfolio.accuracy_stats")

from datetime import UTC

from portfolio.file_utils import atomic_write_json as _atomic_write_json
from portfolio.file_utils import load_json, load_jsonl, load_jsonl_tail
from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES

# C2: Protect all read-modify-write cache operations from concurrent ticker threads
_accuracy_write_lock = threading.Lock()

# BUG-178 (2026-04-16): thundering-herd protection. The disk-backed accuracy
# caches expire on a 1h TTL; on the first cycle after expiry, all 5 ticker
# threads race through load_cached_accuracy() → None → signal_accuracy() and
# each pays the 7s+ cost of loading 50,000 signal-log entries from SQLite.
# Wall time was measured at 215s for a 5-thread race vs 7s single-threaded —
# 30x amplification driven by GIL + DB + file-I/O serialization. The
# get_or_compute_*() helpers below use double-checked locking: cache hits
# take the fast path with no lock acquisition; only the first miss-thread
# computes, and the others wait on _accuracy_compute_lock and then read the
# freshly-populated cache. The lock is held THROUGH the compute (unlike the
# signal_utility cache below) because cache-miss is rare (~once per hour
# per horizon) and serializing 4 threads through a 7s wait is far cheaper
# than 4 redundant 50000-entry SQL scans.
_accuracy_compute_lock = threading.Lock()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
ACCURACY_CACHE_FILE = DATA_DIR / "accuracy_cache.json"
BEST_HORIZON_CACHE_FILE = DATA_DIR / "best_horizon_cache.json"
ACCURACY_CACHE_TTL = 3600
HORIZONS = ["3h", "4h", "12h", "1d", "3d", "5d", "10d"]

# In-memory cache for signal_utility (added 2026-04-15, BUG-178 mitigation).
# signal_utility() walks every entry in the signal log (~6320 snapshots / ~92K
# ticker rows as of this writing) and costs ~3.6s cold and <50ms hot. It's
# called from generate_signal() on every ticker, every cycle, with NO
# disk-backed cache — so when the OS file cache is cold (memory pressure,
# fresh process, antivirus scan) 5 concurrent ticker threads each pay the
# 3-4s cold read, which can compound under file-cache page-in contention.
#
# Invalidation is pure TTL (300s) — there is NO mtime check against
# signal_log.db, so a backfill that writes new outcomes mid-cycle is only
# visible to signal_utility after the TTL expires. This is an explicit
# trade: outcome backfill runs daily at 18:00 local via the PF-OutcomeCheck
# scheduled task (see docs/operational-runbook.md), so a 5-minute staleness
# window is dominated by the 24-hour write cadence. Code paths that need
# immediately-fresh utility (tests, outcome_tracker, post-backfill reports)
# must either pass entries= explicitly (which bypasses the cache) or call
# invalidate_signal_utility_cache() after the write.
#
# 300s TTL matches the shortest LLM rotation period and is well below the
# 3600s ACCURACY_CACHE_TTL used for the disk-backed caches. The lock
# guards the (timestamp, value) tuple so two threads racing to refresh
# can't corrupt the dict. Dogpile behavior: the lock is held ONLY for the
# swap, NOT for the compute — the slow signal_utility() call happens
# outside the lock, so other threads waiting on the lock see the fresh
# value the moment the first thread returns. Two threads that both miss
# on a TTL-boundary race will each recompute once (one wasted walk), but
# neither blocks the other. This is cheaper than holding a global lock
# through a 3.6s disk scan and funneling every ticker thread through it.
_SIGNAL_UTILITY_CACHE_TTL = 300.0
_signal_utility_cache: dict[str, tuple[float, dict]] = {}
_signal_utility_cache_lock = threading.Lock()

# L2 disk-backed cache (added 2026-05-03, BUG-178 follow-up).
# Survives process restart so the first cycle after a `schtasks /run` doesn't
# pay the ~49s parallel-cold-compute cost we measured under 4-thread
# contention (PASS 4 of scripts/perf/profile_utility_overlay.py). Mirrors the
# existing pattern at regime_accuracy_cache.json: single "time" key gates TTL,
# per-horizon data persists across writes via load-merge-write.
#
# 1-hour TTL matches ACCURACY_CACHE_TTL and is appropriate because outcome
# backfill runs daily — a fresh-after-restart cache from earlier today is
# more accurate than a freshly-computed cache from a partially-loaded SQLite.
# Atomic writes via _atomic_write_json so concurrent writers race to be last
# without producing torn reads.
_SIGNAL_UTILITY_DISK_TTL = 3600.0
SIGNAL_UTILITY_CACHE_FILE = DATA_DIR / "signal_utility_cache.json"

# Separate from _signal_utility_cache_lock — that one only guards the in-memory
# dict swap. This one serializes the read-modify-write of the disk file so two
# threads computing DIFFERENT horizons don't lose each other's writes (the
# benign-race assumption only holds for same-horizon races; cross-horizon
# races would lose 3 of 4 horizons on a 4-thread cold-start cycle, defeating
# the L2 cache's purpose). Disk write is ~10-50ms — keeping this lock
# separate from the L1 lock means L1 reads aren't blocked behind disk IO.
_signal_utility_disk_lock = threading.Lock()


def _load_signal_utility_disk(horizon: str) -> dict | None:
    """Return cached utility dict for horizon if disk cache is fresh, else None.

    Single global "time" timestamp gates TTL for all horizons (matches
    regime_accuracy_cache pattern). Per-horizon data persists across writes,
    so a horizon that hasn't been recomputed since the last process can
    still be served from disk as long as the file's "time" is fresh.
    """
    cache = load_json(SIGNAL_UTILITY_CACHE_FILE)
    if not isinstance(cache, dict):
        return None
    if time.time() - cache.get("time", 0) >= _SIGNAL_UTILITY_DISK_TTL:
        return None
    cached = cache.get(horizon)
    return cached if isinstance(cached, dict) else None


def _write_signal_utility_disk(horizon: str, data: dict) -> None:
    """Persist cached utility dict for horizon. Merges with existing horizons.

    Holds _signal_utility_disk_lock through the read-modify-write so two
    threads computing DIFFERENT horizons don't lose each other's writes.
    The earlier lock-free version assumed all races would be same-horizon
    (idempotent), but a 4-thread cold-start cycle realistically has 4
    different horizons in flight; the lockless last-writer-wins kept only
    1 of 4 horizons on disk and forced the other 3 to recompute next cycle.

    Atomic write via _atomic_write_json prevents torn reads even outside
    the lock.

    Failures are swallowed — a stale L2 cache is harmless (TTL expires) and
    must never crash the live signal pipeline (would re-trigger BUG-178's
    silent-failure pattern).
    """
    try:
        with _signal_utility_disk_lock:
            cache = load_json(SIGNAL_UTILITY_CACHE_FILE, default={})
            if not isinstance(cache, dict):
                cache = {}
            cache[horizon] = data
            cache["time"] = time.time()
            _atomic_write_json(SIGNAL_UTILITY_CACHE_FILE, cache)
    except Exception:
        logger.debug("signal_utility disk cache write failed", exc_info=True)


def load_entries():
    """Load signal log entries. Prefers SQLite if available, falls back to JSONL."""
    try:
        from portfolio.signal_db import SignalDB
        db = SignalDB()
        try:
            count = db.snapshot_count()
            if count > 0:
                entries = db.load_entries()
                return entries
        finally:
            # BUG-137: Always close DB, even if load_entries() throws.
            db.close()
    except Exception as e:
        logger.debug("SQLite signal_db unavailable, falling back to JSONL: %s", e)
    # H2: Fallback to JSONL using atomic load_jsonl_tail (avoids raw open()).
    # 50000 entries covers full accuracy computation; reading all 68MB risks OOM.
    if not SIGNAL_LOG.exists():
        return []
    entries = load_jsonl_tail(SIGNAL_LOG, max_entries=50000)
    return entries if entries else []


_MIN_CHANGE_PCT = 0.05  # outcomes within ±0.05% are treated as neutral (skip)


def _vote_correct(vote, change_pct, min_change_pct=None):
    """Check if a signal vote matches the price outcome.

    Returns True (correct), False (incorrect), or None (neutral — skip this outcome).
    Outcomes within ±min_change_pct are considered noise and should not count
    for or against the signal's accuracy.
    """
    threshold = min_change_pct if min_change_pct is not None else _MIN_CHANGE_PCT
    # 2026-04-22: some outcome entries have change_pct=None (missing backfill
    # data for 4h+ horizons). Treat as neutral instead of TypeErroring — was
    # killing --accuracy report mid-horizon. Matches the None-guard pattern
    # at accuracy_stats.py:1617 and in ic_computation / train_signal_weights.
    if change_pct is None or abs(change_pct) < threshold:
        return None  # neutral — price didn't move enough to judge (or unknown)
    if vote == "BUY" and change_pct > 0:
        return True
    return bool(vote == "SELL" and change_pct < 0)


def signal_accuracy(horizon="1d", since=None, entries=None):
    """Compute per-signal accuracy, optionally filtered to entries after `since`.

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        since: Optional ISO-8601 string cutoff. Only entries with ts >= since
               are included. None means all entries (no time filter).
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {correct, total, accuracy, pct}} for each signal.
    """
    if entries is None:
        entries = load_entries()
    stats = {s: {"correct": 0, "total": 0,
                 "correct_buy": 0, "total_buy": 0,
                 "correct_sell": 0, "total_sell": 0} for s in SIGNAL_NAMES}
    # 2026-04-22 follow-up: count outcomes we skip because change_pct is None.
    # Previously these crashed the report; now they're silently dropped, which
    # would let a data-quality regression (e.g. outcome_tracker writing nulls)
    # go unnoticed. Surface the count so operators see drift.
    null_change_pct_skipped = 0
    total_outcomes_seen = 0

    for entry in entries:
        if since and entry.get("ts", "") < since:
            continue
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue
            total_outcomes_seen += 1

            change_pct = outcome.get("change_pct", 0)
            if change_pct is None:
                null_change_pct_skipped += 1
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue  # neutral outcome — don't count
                stats[sig_name]["total"] += 1
                if vote == "BUY":
                    stats[sig_name]["total_buy"] += 1
                    if result_val:
                        stats[sig_name]["correct_buy"] += 1
                else:
                    stats[sig_name]["total_sell"] += 1
                    if result_val:
                        stats[sig_name]["correct_sell"] += 1
                if result_val:
                    stats[sig_name]["correct"] += 1

    if null_change_pct_skipped > 0:
        pct = 100.0 * null_change_pct_skipped / total_outcomes_seen
        logger.info(
            "signal_accuracy[%s]: skipped %d/%d outcomes (%.2f%%) with "
            "change_pct=None — check outcome_tracker / signal_db backfill",
            horizon, null_change_pct_skipped, total_outcomes_seen, pct,
        )

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0

 succeeded in 816ms:
        buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
        sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
        result[sig_name] = {
            "correct": s["correct"],
            "total": s["total"],
            "samples": s["total"],
            "enabled": sig_name not in DISABLED_SIGNALS,
            "accuracy": acc,
            "pct": round(acc * 100, 1),
            "correct_buy": s["correct_buy"],
            "total_buy": s["total_buy"],
            "buy_accuracy": round(buy_acc, 4),
            "correct_sell": s["correct_sell"],
            "total_sell": s["total_sell"],
            "sell_accuracy": round(sell_acc, 4),
        }
    return result


def signal_accuracy_recent(horizon="1d", days=7):
    """Compute per-signal accuracy using only the last N days of data.

    Thin wrapper around signal_accuracy() with a time cutoff.
    """
    from datetime import datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    return signal_accuracy(horizon, since=cutoff)


def signal_accuracy_ewma(horizon="1d", halflife_days=5, entries=None):
    """Compute per-signal accuracy with exponential decay weighting.

    Recent observations are weighted higher than older ones. The weight for
    an observation of age ``age_days`` is ``exp(-ln(2) / halflife_days * age_days)``,
    meaning entries that are ``halflife_days`` old receive half the weight of
    today's entries.

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        halflife_days: Half-life of the decay in days (default 5). Smaller
            values weight recent data more aggressively.
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {accuracy, total_weight, effective_samples, total, correct, pct}}
        where ``total`` and ``correct`` are int(round(...)) of the weighted sums
        for compatibility with the existing accuracy pipeline.
    """
    import math
    from datetime import datetime

    if entries is None:
        entries = load_entries()
    now = datetime.now(UTC)
    decay_rate = math.log(2) / halflife_days  # λ = ln(2) / t½

    # Accumulate per-signal: weighted_total, weighted_correct, sum_of_sq_weights
    stats = {
        s: {"w_total": 0.0, "w_correct": 0.0, "sum_w2": 0.0}
        for s in SIGNAL_NAMES
    }

    for entry in entries:
        # Compute age in days from entry timestamp
        ts_str = entry.get("ts", "")
        try:
            entry_dt = datetime.fromisoformat(ts_str)
            age_days = (now - entry_dt).total_seconds() / 86400.0
            age_days = max(age_days, 0.0)  # clamp: never negative for future entries
        except (ValueError, TypeError):
            continue  # skip malformed timestamps

        weight = math.exp(-decay_rate * age_days)

        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue  # neutral outcome — skip

                stats[sig_name]["w_total"] += weight
                stats[sig_name]["sum_w2"] += weight * weight
                if result_val:
                    stats[sig_name]["w_correct"] += weight

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        w_total = s["w_total"]
        w_correct = s["w_correct"]
        sum_w2 = s["sum_w2"]

        if w_total > 0:
            accuracy = w_correct / w_total
            # Kish (1965) effective sample size: n_eff = (Σwᵢ)² / Σwᵢ²
            effective_samples = (w_total * w_total) / sum_w2
        else:
            accuracy = 0.0
            effective_samples = 0.0

        result[sig_name] = {
            "accuracy": accuracy,
            "total_weight": w_total,
            "effective_samples": effective_samples,
            "total": int(round(w_total)),
            "samples": int(round(w_total)),
            "enabled": sig_name not in DISABLED_SIGNALS,
            "correct": int(round(w_correct)),
            "pct": round(accuracy * 100, 1),
        }
    return result


def signal_accuracy_cost_adjusted(horizon="1d", cost_bps=10.0, entries=None):
    """Compute per-signal accuracy adjusted for transaction costs.

    A signal vote is only counted as correct if the price move exceeds
    the estimated round-trip cost (spread + slippage).  This reveals
    signals that are "technically correct" but unprofitable after execution.

    Args:
        horizon: Outcome horizon to evaluate.
        cost_bps: Estimated round-trip cost in basis points (default 10 bps).
                  Must exceed _MIN_CHANGE_PCT (5 bps) to filter beyond
                  the neutral outcome threshold. Metals warrants ~10 bps,
                  crypto ~5 bps.
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {correct, total, accuracy, pct, cost_bps}}
    """
    if entries is None:
        entries = load_entries()

    # Cost threshold: moves below this are unprofitable even if directionally correct
    cost_pct = cost_bps / 100.0  # convert bps to percentage

    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue

                # Skip neutral outcomes (below minimum move)
                if abs(change_pct) < _MIN_CHANGE_PCT:
                    continue

                stats[sig_name]["total"] += 1

                # Cost-adjusted: correct only if move exceeds cost
                if (vote == "BUY" and change_pct > cost_pct) or (vote == "SELL" and change_pct < -cost_pct):
                    stats[sig_name]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        result[sig_name] = {
            "correct": s["correct"],
            "total": s["total"],
            "samples": s["total"],
            "enabled": sig_name not in DISABLED_SIGNALS,
            "accuracy": acc,
            "pct": round(acc * 100, 1),
            "cost_bps": cost_bps,
        }
    return result


def consensus_accuracy(horizon="1d", entries=None, days=None):
    """Aggregate consensus decision accuracy across all tickers.

    For each signal-log entry that has an outcome at `horizon`, check if
    the per-ticker `tdata["consensus"]` BUY/SELL call matched actual
    direction. HOLD calls are skipped (no direction to score).

    BUG-178/W15-W16 follow-up (2026-04-16): added optional `days` kwarg
    for the recent-window variant the degradation tracker needs. The
    existing `entries` kwarg is preserved for callers that already pass
    a pre-loaded list. When both are passed, `entries` wins (caller has
    already filtered).

    Args:
        horizon: Outcome horizon ("3h", "4h", "12h", "1d", "3d", "5d", "10d").
        entries: Pre-loaded entries list (skips both load_entries() and
            the days filter — caller is assumed to have filtered already).
        days: Optional lookback window in days. Ignored if entries is
            provided. None = lifetime aggregate.
    """
    if entries is None:
        entries = load_entries()
        if days is not None:
            from datetime import datetime, timedelta
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            entries = [e for e in entries if e.get("ts", "") >= cutoff]
    correct = 0
    total = 0

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus", "HOLD")
            if consensus == "HOLD":
                continue

            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            result_val = _vote_correct(consensus, change_pct)
            if result_val is None:
                continue
            total += 1
            if result_val:
                correct += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "pct": round(acc * 100, 1),
    }


def per_ticker_accuracy(horizon="1d", entries=None):
    if entries is None:
        entries = load_entries()
    stats = defaultdict(lambda: {"correct": 0, "total": 0})


 succeeded in 854ms:
        if result:
            write_accuracy_cache(horizon, result)
        return result


def get_or_compute_recent_accuracy(horizon: str, days: int = 7):
    """Cached recent-window (default 7d) accuracy, computed at most once."""
    cache_key = f"{horizon}_recent"
    cached = load_cached_accuracy(cache_key)
    if cached:
        return cached
    with _accuracy_compute_lock:
        cached = load_cached_accuracy(cache_key)
        if cached:
            return cached
        result = signal_accuracy_recent(horizon, days=days)
        if result:
            write_accuracy_cache(cache_key, result)
        return result


def get_or_compute_per_ticker_accuracy(horizon: str):
    """Cached per-ticker consensus accuracy, computed at most once.

    Cache key matches the BUG-164 lazy-populate convention used by
    signal_engine.py:_ptc_key.
    """
    cache_key = f"per_ticker_consensus_{horizon}"
    cached = load_cached_accuracy(cache_key)
    if cached:
        return cached
    with _accuracy_compute_lock:
        cached = load_cached_accuracy(cache_key)
        if cached:
            return cached
        result = per_ticker_accuracy(horizon)
        if result:
            write_accuracy_cache(cache_key, result)
        return result


def get_or_compute_consensus_accuracy(horizon: str):
    """Cached aggregate consensus accuracy, computed at most once.

    2026-05-03: added to fix /api/accuracy timeout (>15s). The dashboard
    requests 4 horizons × 3 metrics = 12 lookups; previously consensus
    was the only one without a cache, forcing a full signal-log scan
    per horizon per request.

    Cache key: `consensus_{horizon}`. Mirrors get_or_compute_accuracy
    semantics — double-checked locking, written on first miss.
    """
    cache_key = f"consensus_{horizon}"
    cached = load_cached_accuracy(cache_key)
    if cached:
        return cached
    with _accuracy_compute_lock:
        cached = load_cached_accuracy(cache_key)
        if cached:
            return cached
        result = consensus_accuracy(horizon)
        # consensus_accuracy always returns a dict (even with total=0);
        # write all of them so a horizon with zero scored entries doesn't
        # re-compute on every request.
        if result is not None:
            write_accuracy_cache(cache_key, result)
        return result


# Dashboard /api/accuracy serves these four horizons; the loop's normal
# operation only warms 1d / 3h / their _recent variants. Without an
# explicit prewarm, 3d / 5d / 10d are cold and the first dashboard
# request after a restart spends seconds re-scanning the signal log.
_DASHBOARD_PREWARM_HORIZONS: tuple = ("1d", "3d", "5d", "10d")
_DASHBOARD_PREWARM_INTERVAL_SEC = 3600.0  # 1 hour
_DASHBOARD_PREWARM_STATE_FILE = DATA_DIR / "dashboard_prewarm_state.json"
_DASHBOARD_PREWARM_LOCK_FILE = DATA_DIR / "dashboard_prewarm.lock"
_dashboard_prewarm_lock = threading.Lock()
_last_dashboard_prewarm_ts: float = 0.0
_dashboard_prewarm_loaded: bool = False


def _load_prewarm_ts_from_disk() -> float:
    """Read the persisted prewarm ts, returning 0.0 if missing/corrupt."""
    state = load_json(_DASHBOARD_PREWARM_STATE_FILE, default={}) or {}
    ts = state.get("last_prewarm_ts")
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    return 0.0


def _save_prewarm_ts_to_disk(ts: float) -> None:
    """Persist the latest prewarm ts. Best-effort — never raises."""
    try:
        _atomic_write_json(
            _DASHBOARD_PREWARM_STATE_FILE,
            {"last_prewarm_ts": float(ts)},
        )
    except Exception:
        logger.debug("prewarm-ts persist failed", exc_info=True)


def maybe_prewarm_dashboard_accuracy(now: float | None = None) -> bool:
    """Periodically pre-warm accuracy_cache.json for the dashboard endpoint.

    Self-gating to once per hour so the loop's per-cycle call is cheap
    on the steady-state path. On miss, fans out to 12 cache reads + at
    most 12 underlying computes — the same fanout the dashboard endpoint
    would otherwise pay on its first request after a restart.

    Internally uses the existing get_or_compute_* helpers, so it
    respects the BUG-178 thundering-herd lock and won't fight with
    in-loop callers that hit the same cache from ticker threads.

    Concurrency layers (2026-05-04 codex P2-2 follow-up):
      1. Process-local `threading.Lock` — guards the in-memory ts and
         lazy-load flag against concurrent threads in the same process.
      2. Re-read the persisted ts from disk inside the file-lock window
         — catches the case where another process wrote between our
         lazy-load and our gate decision.
      3. Cross-process file lock around the gate decision + fanout so
         only one of N concurrent processes (main loop + a manual
         trigger, two main loops during a botched restart) actually
         performs the fanout. Lock is non-blocking — a second process
         that races just returns False and treats the call as gated,
         since whichever process wins is doing the same fanout.

    Persistence: `data/dashboard_prewarm_state.json` survives loop
    restarts. Stale-on-corruption-or-missing falls back to 0.0 (next
    call fires) — safe direction.

    Args:
        now: Override clock for tests. Defaults to time.time().

    Returns:
        True if prewarm fired this call, False if gated by the interval
        or another process holds the file lock.
    """
    global _last_dashboard_prewarm_ts, _dashboard_prewarm_loaded
    t = now if now is not None else time.time()
    with _dashboard_prewarm_lock:
        # Layer 1: lazy-load the persisted ts on first call per process.
        if not _dashboard_prewarm_loaded:
            _last_dashboard_prewarm_ts = _load_prewarm_ts_from_disk()
            _dashboard_prewarm_loaded = True
        # First gate check using in-memory (and possibly stale) ts.
        if t - _last_dashboard_prewarm_ts < _DASHBOARD_PREWARM_INTERVAL_SEC:
            return False

    # Layer 3: cross-process exclusion. If another process is already
    # in the fanout window, skip — the work is being done.
    try:
        from portfolio.process_lock import acquire_lock_file, release_lock_file
    except Exception:
        # If process_lock is somehow unavailable, fall back to the
        # process-local guarantee (better than crashing the loop).
        acquire_lock_file = None
        release_lock_file = None

    fh = acquire_lock_file(_DASHBOARD_PREWARM_LOCK_FILE,
                            owner="dashboard_prewarm") if acquire_lock_file else "noop"
    if fh is None:
        # Another process holds the lock — they're doing the fanout.
        # Treat as gated; next caller will see the persisted ts they write.
        return False

    try:
        # Layer 2: re-read disk under the file lock. A racer that won
        # the lock just before us would have written; honor their work.
        # Only honor a positive disk_ts — "file missing" returns 0 from
        # _load_prewarm_ts_from_disk, and treating that as authoritative
        # would clobber a deliberately-old in-memory seed (e.g., test
        # fixtures that pre-seed -10000 to force a fire).
        with _dashboard_prewarm_lock:
            disk_ts = _load_prewarm_ts_from_disk()
            if disk_ts > 0 and disk_ts > _last_dashboard_prewarm_ts:
                _last_dashboard_prewarm_ts = disk_ts
            if t - _last_dashboard_prewarm_ts < _DASHBOARD_PREWARM_INTERVAL_SEC:
                return False
            _last_dashboard_prewarm_ts = t

        for h in _DASHBOARD_PREWARM_HORIZONS:
            get_or_compute_accuracy(h)
            get_or_compute_consensus_accuracy(h)
            get_or_compute_per_ticker_accuracy(h)
        # Persist AFTER the fanout completes so a crash mid-fanout doesn't
        # pin the gate. The file lock means we're the only writer.
        _save_prewarm_ts_to_disk(t)
        return True
    except Exception:
        logger.debug("maybe_prewarm_dashboard_accuracy failed", exc_info=True)
        return False
    finally:
        if release_lock_file and fh != "noop":
            release_lock_file(fh)


def _count_entries_with_outcomes(entries, horizon):
    count = 0
    for entry in entries:
        outcomes = entry.get("outcomes", {})
        for _ticker, horizons in outcomes.items():
            if horizons.get(horizon):
                count += 1
                break
    return count


def print_accuracy_report():
    entries = load_entries()
    if not entries:
        print("No signal log data found.")
        return

    horizon_counts = {h: _count_entries_with_outcomes(entries, h) for h in HORIZONS}
    counts_str = ", ".join(f"{horizon_counts[h]} with {h} outcomes" for h in HORIZONS)

    print("=== Signal Accuracy Report ===")
    print()
    print(f"Entries: {len(entries)} total, {counts_str}")

    for h in HORIZONS:
        if horizon_counts[h] == 0:
            continue

        print()
        print(f"--- {h} Horizon ({horizon_counts[h]} entries with outcomes) ---")
        print()

        # ARCH-24: Pass pre-loaded entries to avoid re-reading 68MB file per call.
        sig_acc = signal_accuracy(h, entries=entries)
        sorted_sigs = sorted(
            SIGNAL_NAMES, key=lambda s: sig_acc[s]["accuracy"], reverse=True
        )

        print(f"{'Signal':<16}{'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
        print(f"{'------':<16}{'-------':>7}  {'-----':>5}  {'--------':>8}")

        for sig_name in sorted_sigs:
            s = sig_acc[sig_name]
            if s["total"] == 0:
                continue
            disabled_tag = " (OFF)" if sig_name in DISABLED_SIGNALS else ""
            print(
                f"{sig_name:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%{disabled_tag}"
            )

        cons = consensus_accuracy(h, entries=entries)
        print()
        if cons["total"] > 0:
            print(
                f"{'Consensus':<16}{cons['correct']:>7}  {cons['total']:>5}  {cons['accuracy']*100:>7.1f}%"
            )

        ticker_acc = per_ticker_accuracy(h, entries=entries)
        if ticker_acc:
            print()
            print("Per-Ticker:")
            sorted_tickers = sorted(
                ticker_acc.keys(), key=lambda t: ticker_acc[t]["accuracy"], reverse=True
            )
            for ticker in sorted_tickers:
                s = ticker_acc[ticker]
                print(
                    f"{ticker:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%"
                )


REGIME_ACCURACY_CACHE_FILE = DATA_DIR / "regime_accuracy_cache.json"


def signal_accuracy_by_regime(horizon="1d", since=None, entries=None):
    """Compute per-signal accuracy grouped by market regime.

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        since: Optional ISO-8601 string cutoff. Only entries with ts >= since
               are included. None means all entries (no time filter).
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {regime: {signal_name: {correct, total, accuracy, pct}}}
              Only includes signals with total > 0.
    """
    if entries is None:
        entries = load_entries()

    # {regime: {signal_name: {correct, total}}}
    regime_stats = defaultdict(lambda: {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES})

    for entry in entries:
        if since and entry.get("ts", "") < since:
            continue
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})
            regime = tdata.get("regime", "unknown")

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue  # neutral outcome — don't count
                regime_stats[regime][sig_name]["total"] += 1
                if result_val:
                    regime_stats[regime][sig_name]["correct"] += 1

    result = {}
    for regime, sig_map in regime_stats.items():
        regime_result = {}
        for sig_name, s in sig_map.items():
            if s["total"] == 0:
                continue
            acc = s["correct"] / s["total"]
            regime_result[sig_name] = {
                "correct": s["correct"],
                "total": s["total"],
                "samples": s["total"],
                "enabled": sig_name not in DISABLED_SIGNALS,
                "accuracy": acc,
                "pct": round(acc * 100, 1),
            }
        if regime_result:
            result[regime] = regime_result

    return result


def load_cached_regime_accuracy(horizon="1d"):
    """Load cached regime accuracy, returning None if missing or stale.

    Uses the same TTL as the main accuracy cache (ACCURACY_CACHE_TTL).
    """
    cache = load_json(REGIME_ACCURACY_CACHE_FILE)
    if cache is not None:
        try:
            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (KeyError, AttributeError):
            logger.debug("Regime accuracy cache corrupted or missing horizon %s", horizon)
    return None


def write_regime_accuracy_cache(horizon, data):
    """Persist regime accuracy data to the cache file.

    Merges with any existing horizons to avoid overwriting other cached data.
    """
    with _accuracy_write_lock:
        cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}
        cache[horizon] = data
        cache["time"] = time.time()
        _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)


# L1 in-memory cache for regime accuracy (added 2026-05-04).
# Mirrors the signal_utility L1+L2 pattern at line 69-95. The existing
# load_cached_regime_accuracy/write_regime_accuracy_cache pair only had L2
# (disk) — every ticker call paid a JSON parse, and on TTL miss all 5 ticker
# threads cold-computed signal_accuracy_by_regime() in parallel (50K-entry
# walks @ ~30s each). That's the [SLOW-PHASE] *_utility_overlay timings
# observed 2026-05-04 that pushed cycles to ~595s and tripped the dashboard
# stale flag. This L1 cache makes the second-through-Nth ticker thread per
# cycle return in <1ms.
#
# 300s TTL matches _SIGNAL_UTILITY_CACHE_TTL. Dogpile behavior is identical:
# lock guards ONLY the dict swap, never the compute. Two threads that both
# miss on a TTL-boundary race will each recompute once (acceptable — outcome
# backfill is daily so the cache is normally cold once per cycle window).
_REGIME_ACCURACY_CACHE_TTL = 300.0
_regime_accuracy_cache: dict[str, tuple[float, dict]] = {}
_regime_accuracy_cache_lock = threading.Lock()


def get_or_compute_regime_accuracy(horizon: str = "1d") -> dict:
    """L1+L2 cached wrapper around signal_accuracy_by_regime.

    Hot path on every ticker × horizon. Three-tier resolution:

    1. L1 (in-memory dict) — sub-µs, returns the same dict ref each call.
    2. L2 (regime_accuracy_cache.json on disk) — survives process restart.
       ~10-50ms JSON parse; populates L1 on hit.
    3. Cold compute via signal_accuracy_by_regime — ~30s walking 50K signal
       log entries. Runs OUTSIDE any lock so concurrent ticker threads aren't
       serialized through it. On success, populates BOTH caches.

    Returns an empty dict on compute failure rather than raising — matches the
    pre-existing `if not regime_acc:` guard at signal_engine.py:3416 so the
    accuracy gate falls through to global per-signal accuracy. A cache miss
    must never crash the live signal pipeline (BUG-178 silent-failure pattern).
    """
    now = time.time()
    with _regime_accuracy_cache_lock:
        cached = _regime_accuracy_cache.get(horizon)
        if cached and now - cached[0] < _REGIME_ACCURACY_CACHE_TTL:
            return cached[1]

    # L1 miss — try L2 disk before paying the cold compute. After a process
    # restart this is what saves us: thread-1 reads disk (~ms), populates L1,
    # threads 2-5 hit L1.
    disk_cached = load_cached_regime_accuracy(horizon)
    if disk_cached is not None:
        with _regime_accuracy_cache_lock:
            _regime_accuracy_cache[horizon] = (time.time(), disk_cached)
        return disk_cached

    # L1 + L2 miss — compute outside the lock so concurrent ticker threads
    # don't serialize. Same benign-race trade-off as signal_utility: one
    # wasted walk on a cycle-boundary race is cheaper than serializing 4
    # threads through a 30s compute.
    #
    # On compute exception → return {} WITHOUT caching, so a transient
    # failure (e.g. SQLite locked) doesn't poison the cache for 5 minutes.
    # Empty success ({} from sparse data) IS cached: outcomes update daily,
    # so 5-min staleness on a legitimately-empty result is harmless and
    # avoids the 30s/ticker re-walk that this whole change is meant to kill.
    try:
        result = signal_accuracy_by_regime(horizon)
    except Exception:
        logger.debug("regime accuracy compute failed", exc_info=True)
        return {}
    with _regime_accuracy_cache_lock:
        _regime_accuracy_cache[horizon] = (time.time(), result)
    try:
        write_regime_accuracy_cache(horizon, result)
    except Exception:
        logger.debug("regime accuracy disk write failed", exc_info=True)
    return result


def invalidate_regime_accuracy_cache(horizon: str | None = None) -> None:
    """Clear both layers of the regime accuracy cache.

    Pass a specific horizon to evict just that entry, or None to clear all.
    Clears the in-memory L1 AND expires the on-disk L2 by zeroing the
    cache file's "time" field — without that, the L2 would keep serving
    stale data for up to 1h (its TTL) even after L1 is dropped, defeating
    the point of an explicit invalidation.

    Intended for callers that just changed the underlying outcome data
    (e.g. outcome_tracker after a backfill) and want the next signal cycle
    to recompute. The function is best-effort on the disk side: a write
    failure is logged but doesn't raise — a stale L2 self-corrects on its
    natural TTL boundary.
    """
    with _regime_accuracy_cache_lock:
        if horizon is None:
            _regime_accuracy_cache.clear()
        else:
            _regime_accuracy_cache.pop(horizon, None)

    try:
        with _accuracy_write_lock:
            cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
            if not isinstance(cache, dict):
                return
            if horizon is None:
                # Drop everything: write empty so stale entries can't reappear
                # if a partial write left the time field intact.
                _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, {})
            else:
                cache.pop(horizon, None)
                # Zero the global time so the surviving horizons fall through
                # to recompute too — matches the L1 semantics (per-horizon
                # eviction). The disk file uses a single shared "time" gate.
                cache["time"] = 0
                _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)
    except Exception:
        logger.debug("regime accuracy L2 invalidation failed", exc_info=True)


ACCURACY_SNAPSHOTS_FILE = DATA_DIR / "accuracy_snapshots.jsonl"


def save_accuracy_snapshot(extras=None):
    """Save current per-signal accuracy as a timestamped snapshot.

    Appends one JSON line to accuracy_snapshots.jsonl with the current
    accuracy for each signal at the 1d horizon. Used by check_accuracy_changes()
    and accuracy_degradation.check_degradation() to detect significant shifts
    over time.

    Args:
        extras: Optional dict of extra blocks to merge into the snapshot.
            BUG-178/W15-W16 follow-up (2026-04-16): the degradation tracker
            needs more than the lifetime per-signal block. Callers can pass
            recent-window per-signal accuracy, per-ticker per-signal blocks,
            forecast model accuracy, and aggregate consensus accuracy by
            providing keys like "signals_recent", "per_ticker", "forecast",
            "consensus", etc. Unknown keys are accepted as-is so future
            scopes can be added without churning this function. Old single-
            block snapshots remain readable — the loader treats absent
            keys as missing.
    """
    from datetime import datetime

    acc = signal_accuracy("1d")
    snapshot = {
        "ts": datetime.now(UTC).isoformat(),
        "signals": {
            name: {"accuracy": data["accuracy"], "total": data["total"]}
            for name, data in acc.items()
        },
    }
    if extras:
        for key, value in extras.items():
            snapshot[key] = value

 succeeded in 869ms:
    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus", "HOLD")
            if consensus == "HOLD":
                continue

            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            result_val = _vote_correct(consensus, change_pct)
            if result_val is None:
                continue
            stats[ticker]["total"] += 1
            if result_val:
                stats[ticker]["correct"] += 1

    result = {}
    for ticker, s in stats.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        result[ticker] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "pct": round(acc * 100, 1),
        }
    return result


def accuracy_by_signal_ticker(signal_name, horizon="1d", days=None, entries=None):
    """Compute per-ticker accuracy for one signal.

    Args:
        signal_name: Signal name present in SIGNAL_NAMES.
        horizon: Outcome horizon to evaluate.
        days: Optional lookback window in days.
        entries: Pre-loaded entries list. BUG-178/W15-W16 follow-up
            (2026-04-16 review): callers that iterate over many signal
            names (e.g. accuracy_degradation._per_ticker_recent) must
            pass a single pre-loaded list instead of letting each call
            re-scan the 50,000-entry SQLite file. Skipping that knob
            blew cycle time by ~290s in the original implementation.

    Returns:
        dict: {ticker: {"accuracy": float, "samples": int, "correct": int}}
    """
    if signal_name not in SIGNAL_NAMES:
        return {}

    if entries is None:
        entries = load_entries()
    cutoff = None
    if days is not None:
        from datetime import datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in entries:
        if cutoff and entry.get("ts", "") < cutoff:
            continue

        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})
        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            vote = (tdata.get("signals") or {}).get(signal_name, "HOLD")
            if vote == "HOLD":
                continue

            change_pct = outcome.get("change_pct", 0)
            result_val = _vote_correct(vote, change_pct)
            if result_val is None:
                continue
            stats[ticker]["total"] += 1
            if result_val:
                stats[ticker]["correct"] += 1

    result = {}
    for ticker, data in stats.items():
        total = data["total"]
        if total == 0:
            continue
        result[ticker] = {
            "accuracy": data["correct"] / total,
            "samples": total,
            "correct": data["correct"],
        }
    return result


def signal_utility(horizon="1d", entries=None):
    """Compute per-signal return magnitude utility.

    For each non-HOLD signal vote with a non-neutral outcome, compute the
    directional return:
      - BUY  → +change_pct   (positive = correct direction)
      - SELL → -change_pct   (negative change = correct direction → positive return)

    Neutral outcomes (|change_pct| < _MIN_CHANGE_PCT) are skipped.

    Args:
        horizon: Outcome horizon to evaluate.
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {avg_return, total_return, samples, utility_score}}
        where utility_score = avg_return * sqrt(samples).
        Signals with no data get zeros.

    2026-04-15 (BUG-178 mitigation): when `entries` is None, the result is
    cached for _SIGNAL_UTILITY_CACHE_TTL seconds keyed by horizon. The cold
    walk costs ~3.6s on a 6K-snapshot log; with 5 ticker threads per cycle
    and the OS file cache occasionally cold, this was a legitimate
    per-cycle cost. Passing an explicit `entries` list bypasses the cache
    (preserves the old behavior for test fixtures that want a specific
    entries snapshot).
    """
    if entries is None:
        now = time.time()
        with _signal_utility_cache_lock:
            cached = _signal_utility_cache.get(horizon)
            if cached and now - cached[0] < _SIGNAL_UTILITY_CACHE_TTL:
                return cached[1]
        # L1 miss — try L2 disk cache before paying the cold compute. After
        # a process restart this is what saves us the ~49s parallel-cold
        # cost: thread-1 reads disk (~ms), populates L1, threads 2-4 hit L1.
        disk_cached = _load_signal_utility_disk(horizon)
        if disk_cached is not None:
            with _signal_utility_cache_lock:
                _signal_utility_cache[horizon] = (time.time(), disk_cached)
            return disk_cached
        # L1 + L2 miss — compute outside the lock to avoid serializing all
        # threads behind the slow path. Populate both caches on success.
        result = _compute_signal_utility(horizon, None)
        with _signal_utility_cache_lock:
            _signal_utility_cache[horizon] = (time.time(), result)
        _write_signal_utility_disk(horizon, result)
        return result
    # Explicit entries — bypass cache (caller controls the dataset).
    return _compute_signal_utility(horizon, entries)


def _compute_signal_utility(horizon, entries):
    """Actual utility computation. Extracted from signal_utility so the
    cache wrapper can call it without re-entering the cached function
    (and so test fixtures passing explicit entries can hit the raw path).
    """
    import math

    if entries is None:
        entries = load_entries()
    # {sig_name: {"total_return": float, "samples": int}}
    stats = {s: {"total_return": 0.0, "samples": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            # 2026-05-03: None-guard mirrors _vote_correct (line 112) and the
            # 2026-04-22 outcome-backfill regression. Without this, every
            # cold compute crashes on 4h+ horizons that haven't backfilled
            # yet — silently swallowed by signal_engine.py:3486's broad
            # except, so the in-memory cache never populates and every call
            # pays cold cost (~2.5s). That's the entire BUG-178 cache regression.
            if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:
                continue  # neutral / unknown outcome — skip

            signals = tdata.get("signals", {})
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                # Directional return: positive when signal was correct
                if vote == "BUY":
                    dir_return = change_pct
                else:  # SELL
                    dir_return = -change_pct

                stats[sig_name]["total_return"] += dir_return
                stats[sig_name]["samples"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        n = s["samples"]
        total_ret = s["total_return"]
        avg_ret = total_ret / n if n > 0 else 0.0
        utility = avg_ret * math.sqrt(n) if n > 0 else 0.0
        result[sig_name] = {
            "avg_return": avg_ret,
            "total_return": total_ret,
            "samples": n,
            "utility_score": utility,
        }
    return result


def invalidate_signal_utility_cache():
    """Clear both the in-memory L1 and the disk-backed L2 signal_utility cache.

    Cross-process scope:
      - L1 in-memory clear is process-local. Other processes (crypto_loop,
        oil_loop, metals_loop) keep their own L1 until their TTL expires.
      - L2 disk delete is shared. After this call, the next call from ANY
        process for any horizon misses L2 and recomputes.

    Intended caller: outcome_tracker (which runs as the PF-OutcomeCheck
    daily scheduled task) after backfilling new outcomes — at that point
    the cached utility values are stale by definition and forcing recompute
    across all processes is correct. The satellite loops do not call this
    function (verified 2026-05-03 grep) and should not start to: a delete
    from one would force the others to pay the cold-compute cost on their
    next cycle. If a satellite ever needs to invalidate its own L1 only,
    add a separate L1-only function rather than reusing this one.

    Disk file removal is best-effort and silent on failure: a stale L2 file
    expires by TTL anyway and must never crash the caller.
    """
    with _signal_utility_cache_lock:
        _signal_utility_cache.clear()
    try:
        if SIGNAL_UTILITY_CACHE_FILE.exists():
            SIGNAL_UTILITY_CACHE_FILE.unlink()
    except Exception:
        logger.debug("signal_utility disk cache delete failed", exc_info=True)


def best_worst_signals(horizon="1d", acc=None):
    if acc is None:
        acc = signal_accuracy(horizon)
    qualified = {k: v for k, v in acc.items() if v["total"] >= 5}
    if not qualified:
        return {"best": None, "worst": None}

    best_name = max(qualified, key=lambda k: qualified[k]["accuracy"])
    worst_name = min(qualified, key=lambda k: qualified[k]["accuracy"])
    return {
        "best": (best_name, qualified[best_name]["accuracy"]),
        "worst": (worst_name, qualified[worst_name]["accuracy"]),
    }


def signal_activation_rates(entries=None):
    """Compute per-signal activation rates (how often each signal votes non-HOLD).

    Args:
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns dict: {signal_name: {activation_rate, buy_rate, sell_rate, bias, samples}}
    - activation_rate: fraction of votes that are BUY or SELL (0.0 to 1.0)
    - bias: directional bias = abs(buy_rate - sell_rate) / activation_rate (0=balanced, 1=all one side)
    - rarity_weight: log(1 + 1/activation_rate) — rare signals get higher weight
    - bias_penalty: 1 - bias (minimum 0.1 floor) — directional signals get penalized
    - normalized_weight: rarity_weight * bias_penalty — the final multiplier
    """
    import math

    if entries is None:
        entries = load_entries()
    stats = {s: {"buy": 0, "sell": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        tickers = entry.get("tickers", {})
        for _ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name)
                if vote is None:
                    continue
                stats[sig_name]["total"] += 1
                if vote == "BUY":
                    stats[sig_name]["buy"] += 1
                elif vote == "SELL":
                    stats[sig_name]["sell"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        total = s["total"]
        if total == 0:
            result[sig_name] = {
                "activation_rate": 0.0, "buy_rate": 0.0, "sell_rate": 0.0,
                "bias": 1.0, "rarity_weight": 1.0, "bias_penalty": 0.1,
                "normalized_weight": 0.1, "samples": 0,
            }
            continue

        buy_rate = s["buy"] / total
        sell_rate = s["sell"] / total
        activation_rate = buy_rate + sell_rate

        # Rarity: IDF-style weight — rare signals get more weight when they vote
        if activation_rate > 0.01:
            rarity_weight = math.log(1 + 1 / activation_rate)
        else:
            rarity_weight = math.log(1 + 100)  # cap for near-zero activation

        # Bias: penalize signals that always vote one direction
        if activation_rate > 0:
            bias = abs(buy_rate - sell_rate) / activation_rate
        else:
            bias = 1.0
        bias_penalty = max(1.0 - bias, 0.1)  # floor at 0.1

        result[sig_name] = {
            "activation_rate": round(activation_rate, 4),
            "buy_rate": round(buy_rate, 4),
            "sell_rate": round(sell_rate, 4),
            "bias": round(bias, 4),
            "rarity_weight": round(rarity_weight, 4),
            "bias_penalty": round(bias_penalty, 4),
            "normalized_weight": round(rarity_weight * bias_penalty, 4),
            "samples": total,
        }

    return result


# P2-B (2026-04-17 adversarial review): default min_recent_samples was 50
# while production (signal_engine) passes 30. Default lowered so non-prod
# callers (backtester, replay script) match live behavior rather than
# silently dropping the blended value for signals with 30-49 recent samples.
_BLEND_DEFAULT_MIN_RECENT_SAMPLES = 30


def blend_accuracy_data(alltime, recent, divergence_threshold=0.15,
                        normal_weight=0.70, fast_weight=0.90,
                        min_recent_samples=_BLEND_DEFAULT_MIN_RECENT_SAMPLES):
    """Blend all-time and recent accuracy using adaptive recency weighting.

    When recent accuracy diverges sharply from all-time (> divergence_threshold),
    fast-track to higher recent weight for faster regime adaptation.

    Args:
        alltime: Dict of {signal_name: {accuracy, total, correct, pct}}.
        recent: Dict of {signal_name: {accuracy, total, correct, pct}}.
        divergence_threshold: Absolute accuracy difference that triggers fast blend.
        normal_weight: Recent weight when divergence is below threshold.
        fast_weight: Recent weight when divergence exceeds threshold.
        min_recent_samples: Minimum recent samples before blending (else use alltime).

    Returns:
        dict: Blended {signal_name: {accuracy, total, correct, pct}}.
    """
    if not alltime and not recent:
        return {}
    # Codex round-10 P2 (2026-04-17 follow-up): don't early-return recent
    # without going through the blend loop - the min_recent_samples floor
    # must apply to recent-only signals too (a 20-sample signal with recent
    # accuracy=0.80 should default to neutral 0.5, not vote at 0.80).
    # Treating empty alltime/recent as {} keeps the loop's per-signal
    # sample-threshold logic authoritative.
    if not alltime:
        alltime = {}
    if not recent:
        recent = {}

    # P1-D (2026-04-17 adversarial review): iterate over the UNION of signal
    # names, not just alltime. Previously a signal present only in `recent`
    # was silently dropped, and directional keys (buy_accuracy, sell_accuracy,
    # total_buy, total_sell) were copied only from `at` - so a signal with
    # sell_accuracy=0.28 over 400 samples in recent but no alltime entry
    # silently passed the directional gate because total_sell defaulted to 0.
    accuracy_data = {}
    all_signal_names = set(alltime) | set(recent)
    for sig_name in all_signal_names:
        at = alltime.get(sig_name) or {}
        rc = recent.get(sig_name) or {}
        at_acc = at.get("accuracy", 0.5) if at else 0.5
        rc_acc = rc.get("accuracy", 0.5) if rc else 0.5
        rc_samples = rc.get("total", 0) if rc else 0
        at_samples = at.get("total", 0) if at else 0

        # Blend only when recent has enough samples AND alltime exists;
        # otherwise fall back to whichever source has data.
        # Codex round-10 P2 (2026-04-17 follow-up): previously a recent-only
        # signal with <min_recent_samples samples fell through to rc_acc,
        # letting an immature signal's raw recent accuracy drive consensus.
        # Now we require min_recent_samples even for recent-only signals,
        # falling back to a neutral 0.5 otherwise (matches pre-patch
        # semantics for signals below the recent-sample floor).
        if rc_samples >= min_recent_samples and at_samples > 0:
            divergence = abs(rc_acc - at_acc)
            w = fast_weight if divergence > divergence_threshold else normal_weight
            blended = w * rc_acc + (1 - w) * at_acc
        elif at_samples > 0:
            blended = at_acc
        elif rc_samples >= min_recent_samples:
            blended = rc_acc  # recent-only signal with enough samples
        else:
            blended = 0.5  # immature signal: neutral default

        total = max(at_samples, rc_samples)
        result = {
            "accuracy": blended,
            "total": total,
            "samples": total,
            "enabled": sig_name not in DISABLED_SIGNALS,
            "correct": int(round(blended * total)),  # BUG-186
            "pct": round(blended * 100, 1),
        }
        # Codex round 11 P2 (2026-04-17 follow-up): directional stats must
        # follow the same sample-floor rule as `accuracy`. Without this,
        # a recent-only signal with 20-29 one-sided votes still influenced
        # _weighted_consensus's directional gate/weighting at its raw
        # recent directional accuracy, even though the overall `accuracy`
        # field had already been set back to neutral 0.5. Omit directional
        # keys entirely for immature signals so downstream callers see the
        # `.get('buy_accuracy', acc)` fallback.
        _directionals_trustworthy = (
            at_samples > 0 or rc_samples >= min_recent_samples
        )
        if _directionals_trustworthy:
            # Merge directional keys from the larger-sample source per key.
            # Prevents silent gate-bypass when a key exists only in `recent`.
            for key in ("buy_accuracy", "sell_accuracy"):
                if key in at and key in rc:
                    side_total = "total_buy" if key == "buy_accuracy" else "total_sell"
                    at_side = at.get(side_total, 0) or 0
                    rc_side = rc.get(side_total, 0) or 0
                    result[key] = at[key] if at_side >= rc_side else rc[key]
                elif key in at:
                    result[key] = at[key]
                elif key in rc:
                    result[key] = rc[key]
            for key in ("total_buy", "total_sell"):
                at_v = at.get(key, 0) or 0
                rc_v = rc.get(key, 0) or 0
                if at_v or rc_v:
                    result[key] = max(at_v, rc_v)
        accuracy_data[sig_name] = result
    return accuracy_data


ACTIVATION_CACHE_TTL = 3600  # recompute hourly


def load_cached_activation_rates():
    """Load cached activation rates, recomputing if stale."""
    cache_file = DATA_DIR / "activation_cache.json"
    cache = load_json(cache_file)
    if cache is not None:
        try:
            if time.time() - cache.get("time", 0) < ACTIVATION_CACHE_TTL:
                return cache.get("rates", {})
        except (KeyError, AttributeError):
            logger.debug("Activation rates cache corrupted, regenerating")
    rates = signal_activation_rates()
    try:
        with _accuracy_write_lock:
            _atomic_write_json(cache_file, {"rates": rates, "time": time.time()})
    except Exception:
        logger.warning("Failed to write activation rates cache", exc_info=True)
    return rates


def load_cached_accuracy(horizon="1d"):
    cache = load_json(ACCURACY_CACHE_FILE)
    if cache is not None:
        try:
            # BUG-133: Use per-horizon timestamps to avoid cross-horizon staleness.
            # Fall back to legacy shared "time" key for backwards compatibility.
            ts = cache.get(f"time_{horizon}", cache.get("time", 0))
            if time.time() - ts < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (KeyError, AttributeError):
            logger.debug("Accuracy cache corrupted or missing horizon %s", horizon)
    return None


def write_accuracy_cache(horizon, data):
    with _accuracy_write_lock:
        cache = load_json(ACCURACY_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}
        cache[horizon] = data
        # BUG-133: Write per-horizon timestamp so other horizons don't appear fresh.
        cache[f"time_{horizon}"] = time.time()
        # Keep legacy "time" key for backwards compat with older code paths.
        cache["time"] = time.time()
        _atomic_write_json(ACCURACY_CACHE_FILE, cache)


# BUG-178 (2026-04-16) cache-miss wrappers. See _accuracy_compute_lock comment
# at the top of this module for the rationale. Callers that previously did
# `cached = load_cached_accuracy(h); if not cached: cached = signal_accuracy(h);
# write_accuracy_cache(h, cached)` should call these instead so the compute
# is done at most once across all racing ticker threads.

def get_or_compute_accuracy(horizon: str):
    """Return cached all-time accuracy, computing it once if cache is cold.

    Thread-safe via double-checked locking — first miss-thread computes,
    others wait on _accuracy_compute_lock and then read the populated cache.
    """
    cached = load_cached_accuracy(horizon)
    if cached:
        return cached
    with _accuracy_compute_lock:
        cached = load_cached_accuracy(horizon)
        if cached:
            return cached
        result = signal_accuracy(horizon)

 succeeded in 902ms:
    from portfolio.file_utils import atomic_append_jsonl
    atomic_append_jsonl(ACCURACY_SNAPSHOTS_FILE, snapshot)
    return snapshot




def _load_accuracy_snapshots():
    """Load all accuracy snapshots from JSONL file.

    2026-04-28 (audit C1): switched from raw read_text().splitlines() to
    file_utils.load_jsonl per CLAUDE.md rule 4 (atomic I/O). The previous
    implementation read the whole file at once via read_text(), which
    races against the live atomic_append_jsonl writer in maybe_save_daily_snapshot
    -> save_full_accuracy_snapshot — a torn last line silently produced an
    empty/missing snapshot, sending check_degradation() to a no-alert quiet
    state instead of erroring. load_jsonl streams line-by-line and logs
    malformed lines at debug level, so torn writes leave a footprint.
    """
    return load_jsonl(ACCURACY_SNAPSHOTS_FILE)


def _find_snapshot_near(snapshots, target_ts, max_delta_hours=36):
    """Find the snapshot closest to target_ts within max_delta_hours.

    Args:
        snapshots: List of snapshot dicts with 'ts' field.
        target_ts: datetime to search near.
        max_delta_hours: Maximum allowed time difference in hours.

    Returns:
        dict or None: The closest snapshot, or None if none within range.
    """
    from datetime import datetime

    best = None
    best_delta = None
    for snap in snapshots:
        try:
            snap_ts = datetime.fromisoformat(snap["ts"])
            delta = abs((snap_ts - target_ts).total_seconds()) / 3600
            if delta <= max_delta_hours and (best_delta is None or delta < best_delta):
                    best = snap
                    best_delta = delta
        except (ValueError, TypeError, KeyError):
            continue
    return best


def check_accuracy_changes(threshold_drop=0.1, threshold_rise=0.1):
    """Check for significant accuracy changes vs 7 days ago.

    Compares current signal accuracy against the snapshot closest to 7 days ago.
    Returns alerts for signals whose accuracy changed by more than the thresholds.

    Args:
        threshold_drop: Minimum accuracy drop (as fraction, e.g. 0.1 = 10pp) to alert.
        threshold_rise: Minimum accuracy rise (as fraction, e.g. 0.1 = 10pp) to alert.

    Returns:
        list[dict]: List of alert dicts with keys:
            signal, old_accuracy, new_accuracy, change, direction ("dropped"/"rose"),
            old_samples, new_samples.
        Empty list if no significant changes or no historical snapshot available.
    """
    from datetime import datetime, timedelta

    snapshots = _load_accuracy_snapshots()
    if not snapshots:
        return []

    now = datetime.now(UTC)
    target = now - timedelta(days=7)
    old_snapshot = _find_snapshot_near(snapshots, target)

    if old_snapshot is None:
        return []

    # Compute current accuracy
    current_acc = signal_accuracy("1d")
    old_signals = old_snapshot.get("signals", {})

    alerts = []
    for sig_name in SIGNAL_NAMES:
        old_data = old_signals.get(sig_name)
        new_data = current_acc.get(sig_name)

        if not old_data or not new_data:
            continue

        # Require minimum samples for meaningful comparison
        if old_data.get("total", 0) < 10 or new_data.get("total", 0) < 10:
            continue

        old_acc = old_data.get("accuracy", 0.0)
        new_acc = new_data.get("accuracy", 0.0)
        change = new_acc - old_acc

        if change <= -threshold_drop:
            alerts.append({
                "signal": sig_name,
                "old_accuracy": round(old_acc * 100, 1),
                "new_accuracy": round(new_acc * 100, 1),
                "change": round(change * 100, 1),
                "direction": "dropped",
                "old_samples": old_data.get("total", 0),
                "new_samples": new_data.get("total", 0),
            })
        elif change >= threshold_rise:
            alerts.append({
                "signal": sig_name,
                "old_accuracy": round(old_acc * 100, 1),
                "new_accuracy": round(new_acc * 100, 1),
                "change": round(change * 100, 1),
                "direction": "rose",
                "old_samples": old_data.get("total", 0),
                "new_samples": new_data.get("total", 0),
            })

    # Sort by absolute change magnitude, largest first
    alerts.sort(key=lambda a: abs(a["change"]), reverse=True)
    return alerts


def format_accuracy_alerts(alerts):
    """Format accuracy change alerts as human-readable strings.

    Args:
        alerts: List of alert dicts from check_accuracy_changes().

    Returns:
        list[str]: Formatted alert strings.
    """
    lines = []
    for a in alerts:
        lines.append(
            f"{a['signal']} accuracy {a['direction']} from "
            f"{a['old_accuracy']}% to {a['new_accuracy']}% "
            f"({a['change']:+.1f}pp, {a['new_samples']} samples)"
        )
    return lines


def signal_best_horizon_accuracy(min_samples=50, entries=None):
    """Compute each signal's best accuracy across all horizons.

    For each signal, evaluates accuracy at every horizon in HORIZONS and returns
    the horizon with the highest accuracy (provided it meets the minimum sample
    threshold). This allows the signal weighting system to use the most
    predictive horizon for each signal rather than a fixed 1d window.

    Results are cached in BEST_HORIZON_CACHE_FILE with the same TTL as the
    main accuracy cache (ACCURACY_CACHE_TTL).

    Args:
        min_samples: Minimum number of datapoints required for a horizon to be
            considered. Horizons below this threshold are skipped. Default 50.

    Returns:
        dict: {signal_name: {accuracy, total, correct, pct, best_horizon}}
        Signals with no qualifying horizons are omitted from the result.
    """
    # --- Cache check ---
    cached = load_json(BEST_HORIZON_CACHE_FILE)
    if cached is not None and isinstance(cached, dict):
        try:
            if time.time() - cached.get("time", 0) < ACCURACY_CACHE_TTL:
                data = cached.get("data")
                if isinstance(data, dict):
                    return data
        except (KeyError, TypeError):
            pass

    if entries is None:
        entries = load_entries()
    # {sig_name: {horizon: {correct, total}}}
    stats: dict[str, dict[str, dict[str, int]]] = {
        s: {h: {"correct": 0, "total": 0} for h in HORIZONS}
        for s in SIGNAL_NAMES
    }

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            for horizon in HORIZONS:
                outcome = outcomes.get(ticker, {}).get(horizon)
                if not outcome:
                    continue
                change_pct = outcome.get("change_pct", 0)

                for sig_name in SIGNAL_NAMES:
                    vote = signals.get(sig_name, "HOLD")
                    if vote == "HOLD":
                        continue
                    result_val = _vote_correct(vote, change_pct)
                    if result_val is None:
                        continue
                    stats[sig_name][horizon]["total"] += 1
                    if result_val:
                        stats[sig_name][horizon]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        best_hz = None
        best_acc = -1.0
        best_total = 0
        best_correct = 0

        for horizon in HORIZONS:
            h_stats = stats[sig_name][horizon]
            total = h_stats["total"]
            if total < min_samples:
                continue
            acc = h_stats["correct"] / total
            if acc > best_acc:
                best_acc = acc
                best_hz = horizon
                best_total = total
                best_correct = h_stats["correct"]

        if best_hz is not None:
            result[sig_name] = {
                "accuracy": best_acc,
                "total": best_total,
                "samples": best_total,
                "enabled": sig_name not in DISABLED_SIGNALS,
                "correct": best_correct,
                "pct": round(best_acc * 100, 1),
                "best_horizon": best_hz,
            }

    # --- Write cache ---
    try:
        with _accuracy_write_lock:
            _atomic_write_json(BEST_HORIZON_CACHE_FILE, {"time": time.time(), "data": result})
    except Exception:
        logger.debug("Failed to write best_horizon cache", exc_info=True)

    return result


def accuracy_by_ticker_signal(horizon="1d", min_samples=0):
    """Compute per-ticker per-signal accuracy cross-tabulation.

    Returns nested dict: {ticker: {signal_name: {correct, total, accuracy, pct,
        correct_buy, total_buy, buy_accuracy, correct_sell, total_sell, sell_accuracy}}}
    Only includes signals that voted BUY or SELL (HOLD excluded).
    Directional fields (buy_accuracy, sell_accuracy) enable per-ticker directional
    gating in signal_engine._weighted_consensus().

    Args:
        horizon: Outcome horizon ("1d", "3d", "5d", "10d").
        min_samples: Minimum votes required to include a signal for a ticker.
    """
    entries = load_entries()
    # {ticker: {signal: {correct, total, correct_buy, total_buy, correct_sell, total_sell}}}
    def _empty():
        return {"correct": 0, "total": 0, "correct_buy": 0, "total_buy": 0,
                "correct_sell": 0, "total_sell": 0}
    stats = defaultdict(lambda: defaultdict(_empty))

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name, vote in signals.items():
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue
                s = stats[ticker][sig_name]
                s["total"] += 1
                if vote == "BUY":
                    s["total_buy"] += 1
                    if result_val:
                        s["correct"] += 1
                        s["correct_buy"] += 1
                else:
                    s["total_sell"] += 1
                    if result_val:
                        s["correct"] += 1
                        s["correct_sell"] += 1

    result = {}
    for ticker, sig_stats in stats.items():
        ticker_result = {}
        for sig_name, s in sig_stats.items():
            if s["total"] < min_samples:
                continue
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
            sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
            ticker_result[sig_name] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
                "correct_buy": s["correct_buy"],
                "total_buy": s["total_buy"],
                "buy_accuracy": round(buy_acc, 4),
                "correct_sell": s["correct_sell"],
                "total_sell": s["total_sell"],
                "sell_accuracy": round(sell_acc, 4),
            }
        if ticker_result:
            result[ticker] = ticker_result

    return result


def top_signals_for_ticker(ticker, horizon="1d", min_samples=5):
    """Return ranked list of signals for a specific ticker, sorted by accuracy.

    Args:
        ticker: Ticker symbol (e.g. "BTC-USD").
        horizon: Outcome horizon.
        min_samples: Minimum votes to qualify.

    Returns:
        list[dict]: Sorted by accuracy descending. Each dict has:
            signal, correct, total, accuracy, pct.
    """
    all_data = accuracy_by_ticker_signal(horizon, min_samples=min_samples)
    ticker_data = all_data.get(ticker, {})
    ranked = [
        {"signal": sig, **data}
        for sig, data in ticker_data.items()
    ]
    ranked.sort(key=lambda x: x["accuracy"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Per-ticker accuracy cache
# ---------------------------------------------------------------------------

TICKER_ACCURACY_CACHE_FILE = DATA_DIR / "ticker_signal_accuracy_cache.json"


def load_cached_ticker_accuracy(horizon="1d"):
    """Load cached per-ticker per-signal accuracy, returning None if stale.

    Uses the same TTL as the main accuracy cache (ACCURACY_CACHE_TTL).
    Cache structure: {horizon: {ticker: {signal: {correct, total, accuracy, pct}}}, "time": ...}
    """
    cache = load_json(TICKER_ACCURACY_CACHE_FILE)
    if cache is not None:
        try:
            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (KeyError, AttributeError):
            logger.debug("Ticker accuracy cache corrupted or missing horizon %s", horizon)
    return None


def write_ticker_accuracy_cache(horizon, data):
    """Persist per-ticker per-signal accuracy data to the cache file.

    Merges with existing horizons to avoid overwriting other cached data.
    """
    with _accuracy_write_lock:
        cache = load_json(TICKER_ACCURACY_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}
        cache[horizon] = data
        cache["time"] = time.time()
        _atomic_write_json(TICKER_ACCURACY_CACHE_FILE, cache)


def _filter_min_samples(data, min_samples):
    if min_samples <= 0:
        return data
    return {
        ticker: {
            sig: sdata for sig, sdata in sigs.items()
            if sdata.get("total", 0) >= min_samples
        }
        for ticker, sigs in data.items()
    }


def accuracy_by_ticker_signal_cached(horizon="1d", min_samples=0):
    """Cached version of accuracy_by_ticker_signal().

    Checks the ticker accuracy cache first; on miss, computes from the
    full signal log and writes the cache. BUG-178 (2026-04-16): the
    cache-miss compute path is now serialized via _accuracy_compute_lock
    so concurrent ticker threads can't all redundantly walk the 50,000-
    entry signal log when the 1h TTL expires.
    """
    cached = load_cached_ticker_accuracy(horizon)
    if cached:
        return _filter_min_samples(cached, min_samples)

    with _accuracy_compute_lock:
        # Re-check after acquiring the lock — another thread may have
        # populated the cache while we waited.
        cached = load_cached_ticker_accuracy(horizon)
        if cached:
            return _filter_min_samples(cached, min_samples)

        data = accuracy_by_ticker_signal(horizon, min_samples=0)
        if data:
            write_ticker_accuracy_cache(horizon, data)
        return _filter_min_samples(data, min_samples)


def probability_calibration(horizon="1d", buckets=None, since=None):
    """Compute calibration data for reliability diagrams.

    Groups consensus predictions by confidence bucket and computes actual
    accuracy per bucket. Confidence = max(buy, sell) / (buy + sell).

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        buckets: List of bucket boundaries. Defaults to [0.5, 0.6, 0.7, 0.8, 0.9, 1.01].
        since: Optional ISO-8601 string cutoff. Only entries with ts >= since
               are included. None means all entries.

    Returns:
        list[dict]: One dict per bucket with keys: bucket_low, bucket_high,
        predicted_confidence, actual_accuracy, sample_count, correct_count.
    """
    if buckets is None:
        buckets = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]  # 1.01 to include 1.0

    entries = load_entries()
    # Build list of (confidence, correct_bool) tuples
    samples = []
    for entry in entries:
        if since and entry.get("ts", "") < since:
            continue
        outcomes = entry.get("outcomes") or {}
        tickers = entry.get("tickers") or {}
        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus")
            if consensus not in ("BUY", "SELL"):
                continue
            buy_c = int(tdata.get("buy_count") or 0)
            sell_c = int(tdata.get("sell_count") or 0)
            total = buy_c + sell_c
            if total < 1:
                continue
            confidence = max(buy_c, sell_c) / total

            outcome = (outcomes.get(ticker) or {}).get(horizon)
            if outcome is None:
                continue
            change_pct = outcome.get("change_pct")
            if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:
                continue

            correct = (consensus == "BUY" and change_pct > 0) or \
                      (consensus == "SELL" and change_pct < 0)
            samples.append((confidence, correct))

    # Bucket the samples
    result = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        bucket_samples = [(c, correct) for c, correct in samples if lo <= c < hi]
        if not bucket_samples:
            result.append({
                "bucket_low": lo, "bucket_high": hi,
                "predicted_confidence": (lo + hi) / 2,
                "actual_accuracy": None, "sample_count": 0, "correct_count": 0,
            })
            continue
        correct_count = sum(1 for _, c in bucket_samples if c)
        n = len(bucket_samples)
        avg_conf = sum(c for c, _ in bucket_samples) / n
        result.append({
            "bucket_low": lo, "bucket_high": hi,
            "predicted_confidence": round(avg_conf, 4),
            "actual_accuracy": round(correct_count / n, 4),
            "sample_count": n, "correct_count": correct_count,
        })
    return result


if __name__ == "__main__":
    print_accuracy_report()

    # Also show accuracy changes if snapshots exist
    alerts = check_accuracy_changes()
    if alerts:
        print()
        print("=== Accuracy Changes (vs 7 days ago) ===")
        print()
        for line in format_accuracy_alerts(alerts):
            print(f"  {line}")
    else:
        print()
        print("No significant accuracy changes detected (or no 7-day snapshot available).")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'accuracy_by_ticker_signal_cached\\(|load_cached_ticker_accuracy\\(|load_cached_regime_accuracy\\(|get_or_compute_regime_accuracy\\('" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 626ms:

portfolio\accuracy_degradation.py:161:        per_ticker_lifetime = [7maccuracy_by_ticker_signal_cached([0m"1d")
portfolio\accuracy_stats.py:1378:def [7mload_cached_regime_accuracy([0mhorizon="1d"):
portfolio\accuracy_stats.py:1428:def [7mget_or_compute_regime_accuracy([0mhorizon: str = "1d") -> dict:
portfolio\accuracy_stats.py:1454:    disk_cached = [7mload_cached_regime_accuracy([0mhorizon)
portfolio\accuracy_stats.py:1912:def [7mload_cached_ticker_accuracy([0mhorizon="1d"):
portfolio\accuracy_stats.py:1956:def [7maccuracy_by_ticker_signal_cached([0mhorizon="1d", min_samples=0):
portfolio\accuracy_stats.py:1965:    cached = [7mload_cached_ticker_accuracy([0mhorizon)
portfolio\accuracy_stats.py:1972:        cached = [7mload_cached_ticker_accuracy([0mhorizon)
portfolio\signal_engine.py:432:    # Source: [7maccuracy_by_ticker_signal_cached([0m) cross-referenced with
portfolio\signal_engine.py:3236:        _ticker_acc_data = ([7maccuracy_by_ticker_signal_cached([0macc_horizon) [0m[7m[0mor [0m
[7m[0m{}).get(ticker, [0m[7m[0m{})[0m
portfolio\signal_engine.py:3428:            regime_acc = [7mget_or_compute_regime_accuracy([0macc_horizon)
portfolio\signal_postmortem.py:216:        regime_acc = [7mload_cached_regime_accuracy([0m"1d")


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path=Path('portfolio/signal_engine.py')
for i,line in enumerate(path.read_text().splitlines(), start=1):
    if 3210 <= i <= 3255 or 3410 <= i <= 3445:
        print(f'{path}:{i}: {line}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 710ms:
portfolio\signal_engine.py:3210:     # Merge shadow votes so outcome_tracker can track accuracy for disabled
portfolio\signal_engine.py:3211:     # signals that were shadow-computed (math-only, no network I/O).
portfolio\signal_engine.py:3212:     raw_votes.update(shadow_votes)
portfolio\signal_engine.py:3213:     if shadow_votes:
portfolio\signal_engine.py:3214:         extra_info["_shadow_votes"] = shadow_votes
portfolio\signal_engine.py:3215: 
portfolio\signal_engine.py:3216:     # 3h horizon: gate slow signals that are noise at short timeframes
portfolio\signal_engine.py:3217:     if horizon in ("3h", "4h"):
portfolio\signal_engine.py:3218:         from portfolio.short_horizon import is_slow_signal_3h
portfolio\signal_engine.py:3219:         for sig_name in list(votes.keys()):
portfolio\signal_engine.py:3220:             if is_slow_signal_3h(sig_name) and votes[sig_name] != "HOLD":
portfolio\signal_engine.py:3221:                 votes[sig_name] = "HOLD"
portfolio\signal_engine.py:3222: 
portfolio\signal_engine.py:3223:     # BUG-143: Apply regime gating BEFORE computing buy/sell counts so that
portfolio\signal_engine.py:3224:     # all downstream code (core gate, min_voters, unanimity penalty) sees
portfolio\signal_engine.py:3225:     # post-gated counts.  _weighted_consensus also applies this internally
portfolio\signal_engine.py:3226:     # (idempotent — gating HOLD→HOLD is a no-op).
portfolio\signal_engine.py:3227:     # BUG-149: now horizon-aware via _get_regime_gated()
portfolio\signal_engine.py:3228:     # BUG-158: Per-ticker exemption — if a signal has ≥60% accuracy with ≥50
portfolio\signal_engine.py:3229:     # samples on THIS ticker, exempt it from regime gating. fear_greed is 93.8%
portfolio\signal_engine.py:3230:     # on XAG-USD but globally gated in ranging — this recovers that alpha.
portfolio\signal_engine.py:3231:     regime_gated = _get_regime_gated(regime, horizon)
portfolio\signal_engine.py:3232:     _ticker_acc_data = {}
portfolio\signal_engine.py:3233:     try:
portfolio\signal_engine.py:3234:         from portfolio.accuracy_stats import accuracy_by_ticker_signal_cached
portfolio\signal_engine.py:3235:         acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
portfolio\signal_engine.py:3236:         _ticker_acc_data = (accuracy_by_ticker_signal_cached(acc_horizon) or {}).get(ticker, {})
portfolio\signal_engine.py:3237:     except Exception:
portfolio\signal_engine.py:3238:         logger.debug("Per-ticker accuracy unavailable for regime gating exemption", exc_info=True)
portfolio\signal_engine.py:3239:     _TICKER_EXEMPT_ACC = 0.60
portfolio\signal_engine.py:3240:     _TICKER_EXEMPT_MIN_SAMPLES = 50
portfolio\signal_engine.py:3241:     # RES-2026-04-21: Recent-accuracy override for regime gating. When a signal's
portfolio\signal_engine.py:3242:     # 7d recent accuracy is significantly above the gate threshold (>55%, 50+ samples),
portfolio\signal_engine.py:3243:     # exempt it from regime gating even if all-time data is bad. Prevents stale regime
portfolio\signal_engine.py:3244:     # gates from suppressing signals that have recovered in a new market regime.
portfolio\signal_engine.py:3245:     # Example: fibonacci went from 43% all-time to 68.2% recent — should not be gated.
portfolio\signal_engine.py:3246:     _RECENT_EXEMPT_ACC = 0.55
portfolio\signal_engine.py:3247:     _RECENT_EXEMPT_MIN_SAMPLES = 50
portfolio\signal_engine.py:3248:     _recent_acc_data = {}
portfolio\signal_engine.py:3249:     try:
portfolio\signal_engine.py:3250:         from portfolio.accuracy_stats import get_or_compute_recent_accuracy
portfolio\signal_engine.py:3251:         # get_or_compute_recent_accuracy expects the base horizon, not the cache key
portfolio\signal_engine.py:3252:         base_hz = "3h" if horizon in ("3h", "4h") else "1d"
portfolio\signal_engine.py:3253:         _recent_acc_data = get_or_compute_recent_accuracy(base_hz) or {}
portfolio\signal_engine.py:3254:     except Exception:
portfolio\signal_engine.py:3255:         logger.debug("Recent accuracy unavailable for regime gating override", exc_info=True)
portfolio\signal_engine.py:3410:         _record_phase(ticker, "acc_load", _phase_start)
portfolio\signal_engine.py:3411:         _phase_start = time.monotonic()
portfolio\signal_engine.py:3412: 
portfolio\signal_engine.py:3413:     # Overlay regime-specific accuracy when available.
portfolio\signal_engine.py:3414:     # H3: Skip all overlays when primary load failed — they would silently restore
portfolio\signal_engine.py:3415:     # real accuracy values for cached signals, negating the fail-closed gate.
portfolio\signal_engine.py:3416:     if not _accuracy_failed:
portfolio\signal_engine.py:3417:         try:
portfolio\signal_engine.py:3418:             from portfolio.accuracy_stats import get_or_compute_regime_accuracy
portfolio\signal_engine.py:3419:             # BUG-134: Use acc_horizon (not hardcoded "1d") so regime accuracy
portfolio\signal_engine.py:3420:             # matches the prediction horizon (3h/4h/12h/1d).
portfolio\signal_engine.py:3421:             # 2026-05-04: switched from manual L2-only dance to L1+L2 wrapper.
portfolio\signal_engine.py:3422:             # The previous code re-read disk on every ticker call (~10-50ms
portfolio\signal_engine.py:3423:             # JSON parse) and on TTL miss all 5 ticker threads cold-computed
portfolio\signal_engine.py:3424:             # in parallel (~30s × 5). The wrapper adds an in-memory L1 with
portfolio\signal_engine.py:3425:             # the same dogpile-resistant pattern as signal_utility, dropping
portfolio\signal_engine.py:3426:             # 2nd-through-Nth ticker calls per cycle to <1ms. Empty dict on
portfolio\signal_engine.py:3427:             # failure preserves the pre-existing fall-through behavior.
portfolio\signal_engine.py:3428:             regime_acc = get_or_compute_regime_accuracy(acc_horizon)
portfolio\signal_engine.py:3429:             current_regime_data = regime_acc.get(regime, {})
portfolio\signal_engine.py:3430:             for sig_name, rdata in current_regime_data.items():
portfolio\signal_engine.py:3431:                 if rdata.get("total", 0) >= 30:
portfolio\signal_engine.py:3432:                     accuracy_data[sig_name] = rdata
portfolio\signal_engine.py:3433:         except Exception:
portfolio\signal_engine.py:3434:             logger.debug("Regime-conditional accuracy unavailable", exc_info=True)
portfolio\signal_engine.py:3435: 
portfolio\signal_engine.py:3436:     # BUG-158: Override global accuracy with per-ticker accuracy for ALL signals.
portfolio\signal_engine.py:3437:     # Per-ticker variance is enormous: fear_greed is 93.8% on XAG-USD but 25.9%
portfolio\signal_engine.py:3438:     # globally. Using global accuracy throws away alpha on specific instruments.
portfolio\signal_engine.py:3439:     # H3: Skip when primary load failed to preserve fail-closed gate.
portfolio\signal_engine.py:3440:     _PER_TICKER_MIN_SAMPLES = 30
portfolio\signal_engine.py:3441:     if not _accuracy_failed and _ticker_acc_data:
portfolio\signal_engine.py:3442:         for sig_name, t_stats in _ticker_acc_data.items():
portfolio\signal_engine.py:3443:             if t_stats.get("total", 0) >= _PER_TICKER_MIN_SAMPLES:
portfolio\signal_engine.py:3444:                 override = {
portfolio\signal_engine.py:3445:                     "accuracy": t_stats["accuracy"],

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern '_fetch_current_price\\('" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 687ms:

portfolio\outcome_tracker.py:171:def [7m_fetch_current_price([0mticker):


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'signal_accuracy\\(|consensus_accuracy\\(|per_ticker_accuracy\\(|ticker_signal_accuracy\\('" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 770ms:

portfolio\accuracy_degradation.py:150:        recent = [7msignal_accuracy([0m"1d", entries=recent_entries)
portfolio\accuracy_degradation.py:190:        extras["consensus"] = [7mconsensus_accuracy([0m"1d", entries=all_entries)
portfolio\accuracy_degradation.py:191:        extras["consensus_recent"] = [7mconsensus_accuracy([0m
portfolio\accuracy_degradation.py:458:        new_signals = [7msignal_accuracy([0m"1d", entries=recent_entries)
portfolio\accuracy_degradation.py:536:        new_consensus = [7mconsensus_accuracy([0m"1d", entries=recent_entries)
portfolio\accuracy_stats.py:20:# threads race through load_cached_accuracy() → None → [7msignal_accuracy([0m) and
portfolio\accuracy_stats.py:189:def [7msignal_accuracy([0mhorizon="1d", since=None, entries=None):
portfolio\accuracy_stats.py:283:    Thin wrapper around [7msignal_accuracy([0m) with a time cutoff.
portfolio\accuracy_stats.py:288:    return [7msignal_accuracy([0mhorizon, since=cutoff)
portfolio\accuracy_stats.py:457:def [7mconsensus_accuracy([0mhorizon="1d", entries=None, days=None):
portfolio\accuracy_stats.py:516:def [7mper_ticker_accuracy([0mhorizon="1d", entries=None):
portfolio\accuracy_stats.py:765:        acc = [7msignal_accuracy([0mhorizon)
portfolio\accuracy_stats.py:1023:# `cached = load_cached_accuracy(h); if not cached: cached = [7msignal_accuracy([0mh);
portfolio\accuracy_stats.py:1040:        result = [7msignal_accuracy([0mhorizon)
portfolio\accuracy_stats.py:1062:def get_or_compute_[7mper_ticker_accuracy([0mhorizon: str):
portfolio\accuracy_stats.py:1076:        result = [7mper_ticker_accuracy([0mhorizon)
portfolio\accuracy_stats.py:1082:def get_or_compute_[7mconsensus_accuracy([0mhorizon: str):
portfolio\accuracy_stats.py:1101:        result = [7mconsensus_accuracy([0mhorizon)
portfolio\accuracy_stats.py:1224:            get_or_compute_[7mconsensus_accuracy([0mh)
portfolio\accuracy_stats.py:1225:            get_or_compute_[7mper_ticker_accuracy([0mh)
portfolio\accuracy_stats.py:1271:        sig_acc = [7msignal_accuracy([0mh, entries=entries)
portfolio\accuracy_stats.py:1288:        cons = [7mconsensus_accuracy([0mh, entries=entries)
portfolio\accuracy_stats.py:1295:        ticker_acc = [7mper_ticker_accuracy([0mh, entries=entries)
portfolio\accuracy_stats.py:1550:    acc = [7msignal_accuracy([0m"1d")
portfolio\accuracy_stats.py:1640:    current_acc = [7msignal_accuracy([0m"1d")
portfolio\signal_db.py:245:    def [7msignal_accuracy([0mself, horizon="1d"):
portfolio\signal_db.py:248:        Returns dict matching accuracy_stats.[7msignal_accuracy([0m) format.
portfolio\signal_db.py:286:    def [7mconsensus_accuracy([0mself, horizon="1d"):
portfolio\signal_db.py:314:    def [7mper_ticker_accuracy([0mself, horizon="1d"):
portfolio\signal_db.py:345:    def [7mticker_signal_accuracy([0mself, horizon="1d", min_samples=0):
portfolio\signal_engine.py:3379:        _ptc_data = get_or_compute_[7mper_ticker_accuracy([0macc_horizon)
portfolio\signal_postmortem.py:211:            acc = [7msignal_accuracy([0m"1d")
portfolio\ticker_accuracy.py:32:    # is None) matches [7msignal_accuracy([0m) / per_ticker_accuracy() / consensus_
portfolio\__pycache__\accuracy_stats.cpython-312.pyc:112:    Thin wrapper around [7msignal_accuracy([0m) with a time cutoff.
portfolio\__pycache__\signal_db.cpython-312.pyc:150:        Returns dict matching accuracy_stats.[7msignal_accuracy([0m) [0m
[7m[0mformat.[0m


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path=Path('portfolio/signal_engine.py')
for i,line in enumerate(path.read_text().splitlines(), start=1):
    if 3080 <= i <= 3125:
        print(f'{path}:{i}: {line}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 598ms:
portfolio\signal_engine.py:3080:                 _sig_t0 = time.monotonic()
portfolio\signal_engine.py:3081:                 compute_fn = load_signal_func(entry)
portfolio\signal_engine.py:3082:                 if compute_fn is None:
portfolio\signal_engine.py:3083:                     votes[sig_name] = "HOLD"
portfolio\signal_engine.py:3084:                     continue
portfolio\signal_engine.py:3085:                 # BUG-178 diagnostic: track which signal each ticker is currently
portfolio\signal_engine.py:3086:                 # running so main.py's pool-timeout handler can name the culprit.
portfolio\signal_engine.py:3087:                 # Ticker guard added 2026-04-10 in the phase-marker diag commit
portfolio\signal_engine.py:3088:                 # to prevent leaking a None-keyed entry when callers pass
portfolio\signal_engine.py:3089:                 # ticker=None (legacy test harnesses and backtester paths).
portfolio\signal_engine.py:3090:                 if ticker:
portfolio\signal_engine.py:3091:                     _set_last_signal(ticker, sig_name)
portfolio\signal_engine.py:3092:                 if entry.get("requires_context"):
portfolio\signal_engine.py:3093:                     result = compute_fn(df, context=context_data)
portfolio\signal_engine.py:3094:                 elif entry.get("requires_macro"):
portfolio\signal_engine.py:3095:                     result = compute_fn(df, macro=macro_data or None)
portfolio\signal_engine.py:3096:                 else:
portfolio\signal_engine.py:3097:                     result = compute_fn(df)
portfolio\signal_engine.py:3098:                 _sig_dt = time.monotonic() - _sig_t0
portfolio\signal_engine.py:3099:                 if _sig_dt > 1.0:
portfolio\signal_engine.py:3100:                     logger.info("[SLOW] %s/%s: %.1fs", ticker, sig_name, _sig_dt)
portfolio\signal_engine.py:3101:                 max_conf = entry.get("max_confidence", 1.0)
portfolio\signal_engine.py:3102:                 validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
portfolio\signal_engine.py:3103:                 extra_info[f"{sig_name}_action"] = validated["action"]
portfolio\signal_engine.py:3104:                 extra_info[f"{sig_name}_confidence"] = validated["confidence"]
portfolio\signal_engine.py:3105:                 extra_info[f"{sig_name}_sub_signals"] = validated["sub_signals"]
portfolio\signal_engine.py:3106:                 if validated["indicators"]:
portfolio\signal_engine.py:3107:                     extra_info[f"{sig_name}_indicators"] = validated["indicators"]
portfolio\signal_engine.py:3108:                 votes[sig_name] = validated["action"]
portfolio\signal_engine.py:3109:             except Exception as e:
portfolio\signal_engine.py:3110:                 logger.warning("Signal %s failed: %s", sig_name, e)
portfolio\signal_engine.py:3111:                 votes[sig_name] = "HOLD"
portfolio\signal_engine.py:3112:                 _signal_failures.append(sig_name)
portfolio\signal_engine.py:3113:         if _signal_failures:
portfolio\signal_engine.py:3114:             extra_info["_signal_failures"] = _signal_failures
portfolio\signal_engine.py:3115:             if len(_signal_failures) > 3:
portfolio\signal_engine.py:3116:                 logger.warning(
portfolio\signal_engine.py:3117:                     "%s: %d enhanced signals failed: %s",
portfolio\signal_engine.py:3118:                     ticker, len(_signal_failures), ", ".join(_signal_failures),
portfolio\signal_engine.py:3119:                 )
portfolio\signal_engine.py:3120: 
portfolio\signal_engine.py:3121:         # Persist signal health (single batch write for all enhanced signals)
portfolio\signal_engine.py:3122:         try:
portfolio\signal_engine.py:3123:             from portfolio.health import update_signal_health_batch
portfolio\signal_engine.py:3124:             health_results = {
portfolio\signal_engine.py:3125:                 sig_name: (sig_name not in _signal_failures)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path_str, start,end in [('portfolio/accuracy_stats.py',144,166), ('portfolio/outcome_tracker.py',142,155)]:
    path=Path(path_str)
    for i,line in enumerate(path.read_text().splitlines(), start=1):
        if start <= i <= end:
            print(f'{path}:{i}: {line}')
    print('---')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 577ms:
portfolio\accuracy_stats.py:144: def load_entries():
portfolio\accuracy_stats.py:145:     """Load signal log entries. Prefers SQLite if available, falls back to JSONL."""
portfolio\accuracy_stats.py:146:     try:
portfolio\accuracy_stats.py:147:         from portfolio.signal_db import SignalDB
portfolio\accuracy_stats.py:148:         db = SignalDB()
portfolio\accuracy_stats.py:149:         try:
portfolio\accuracy_stats.py:150:             count = db.snapshot_count()
portfolio\accuracy_stats.py:151:             if count > 0:
portfolio\accuracy_stats.py:152:                 entries = db.load_entries()
portfolio\accuracy_stats.py:153:                 return entries
portfolio\accuracy_stats.py:154:         finally:
portfolio\accuracy_stats.py:155:             # BUG-137: Always close DB, even if load_entries() throws.
portfolio\accuracy_stats.py:156:             db.close()
portfolio\accuracy_stats.py:157:     except Exception as e:
portfolio\accuracy_stats.py:158:         logger.debug("SQLite signal_db unavailable, falling back to JSONL: %s", e)
portfolio\accuracy_stats.py:159:     # H2: Fallback to JSONL using atomic load_jsonl_tail (avoids raw open()).
portfolio\accuracy_stats.py:160:     # 50000 entries covers full accuracy computation; reading all 68MB risks OOM.
portfolio\accuracy_stats.py:161:     if not SIGNAL_LOG.exists():
portfolio\accuracy_stats.py:162:         return []
portfolio\accuracy_stats.py:163:     entries = load_jsonl_tail(SIGNAL_LOG, max_entries=50000)
portfolio\accuracy_stats.py:164:     return entries if entries else []
portfolio\accuracy_stats.py:165: 
portfolio\accuracy_stats.py:166: 
---
portfolio\outcome_tracker.py:142:             "buy_count": buy_count,
portfolio\outcome_tracker.py:143:             "sell_count": sell_count,
portfolio\outcome_tracker.py:144:             "total_voters": total_voters,
portfolio\outcome_tracker.py:145:             "signals": signals,
portfolio\outcome_tracker.py:146:             "regime": regime,
portfolio\outcome_tracker.py:147:         }
portfolio\outcome_tracker.py:148: 
portfolio\outcome_tracker.py:149:     entry = {
portfolio\outcome_tracker.py:150:         "ts": ts,
portfolio\outcome_tracker.py:151:         "trigger_reasons": trigger_reasons,
portfolio\outcome_tracker.py:152:         "fx_rate": fx_rate,
portfolio\outcome_tracker.py:153:         "tickers": tickers,
portfolio\outcome_tracker.py:154:         "outcomes": {},
portfolio\outcome_tracker.py:155:     }
---

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path=Path('portfolio/outcome_tracker.py')
for i,line in enumerate(path.read_text().splitlines(), start=1):
    if 156 <= i <= 170:
        print(f'{path}:{i}: {line}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 572ms:
portfolio\outcome_tracker.py:156: 
portfolio\outcome_tracker.py:157:     atomic_append_jsonl(SIGNAL_LOG, entry)
portfolio\outcome_tracker.py:158: 
portfolio\outcome_tracker.py:159:     # Dual-write to SQLite
portfolio\outcome_tracker.py:160:     try:
portfolio\outcome_tracker.py:161:         from portfolio.signal_db import SignalDB
portfolio\outcome_tracker.py:162:         db = SignalDB()
portfolio\outcome_tracker.py:163:         db.insert_snapshot(entry)
portfolio\outcome_tracker.py:164:         db.close()
portfolio\outcome_tracker.py:165:     except Exception as e:
portfolio\outcome_tracker.py:166:         logger.warning("SQLite snapshot write failed (JSONL has data, SQLite may lag): %s", e)
portfolio\outcome_tracker.py:167: 
portfolio\outcome_tracker.py:168:     return entry
portfolio\outcome_tracker.py:169: 
portfolio\outcome_tracker.py:170: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path=Path('portfolio/outcome_tracker.py')
for i,line in enumerate(path.read_text().splitlines(), start=1):
    if 233 <= i <= 258:
        print(f'{path}:{i}: {line}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 570ms:
portfolio\outcome_tracker.py:233:     if ticker in BINANCE_SPOT_MAP:
portfolio\outcome_tracker.py:234:         symbol = BINANCE_SPOT_MAP[ticker]
portfolio\outcome_tracker.py:235:         start_ms = int(target_ts * 1000)
portfolio\outcome_tracker.py:236:         r = fetch_with_retry(
portfolio\outcome_tracker.py:237:             f"{BINANCE_BASE}/klines",
portfolio\outcome_tracker.py:238:             params={
portfolio\outcome_tracker.py:239:                 "symbol": symbol,
portfolio\outcome_tracker.py:240:                 "interval": "1h",
portfolio\outcome_tracker.py:241:                 "startTime": start_ms,
portfolio\outcome_tracker.py:242:                 "limit": 1,
portfolio\outcome_tracker.py:243:             },
portfolio\outcome_tracker.py:244:             timeout=10,
portfolio\outcome_tracker.py:245:         )
portfolio\outcome_tracker.py:246:         if r is None:
portfolio\outcome_tracker.py:247:             return None
portfolio\outcome_tracker.py:248:         r.raise_for_status()
portfolio\outcome_tracker.py:249:         data = r.json()
portfolio\outcome_tracker.py:250:         if not data:
portfolio\outcome_tracker.py:251:             return None
portfolio\outcome_tracker.py:252:         return float(data[0][4])
portfolio\outcome_tracker.py:253: 
portfolio\outcome_tracker.py:254:     if ticker in YF_MAP:
portfolio\outcome_tracker.py:255:         # 2026-04-14: historical price-at-timestamp needs a date window
portfolio\outcome_tracker.py:256:         # that price_source.fetch_klines's `limit`-based contract can't
portfolio\outcome_tracker.py:257:         # express precisely. Keep direct yfinance here — the call is
portfolio\outcome_tracker.py:258:         # only for outcome backfill (non-hot-path; 15-min delay is

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path=Path('portfolio/outcome_tracker.py')
for i,line in enumerate(path.read_text().splitlines(), start=1):
    if 258 <= i <= 279:
        print(f'{path}:{i}: {line}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 603ms:
portfolio\outcome_tracker.py:258:         # only for outcome backfill (non-hot-path; 15-min delay is
portfolio\outcome_tracker.py:259:         # acceptable for an outcome at target_ts already in the past).
portfolio\outcome_tracker.py:260:         # To migrate, add start/end support to price_source and update
portfolio\outcome_tracker.py:261:         # this block.
portfolio\outcome_tracker.py:262:         import yfinance as yf
portfolio\outcome_tracker.py:263: 
portfolio\outcome_tracker.py:264:         _yfinance_limiter.wait()
portfolio\outcome_tracker.py:265:         target_dt = datetime.fromtimestamp(target_ts, tz=UTC)
portfolio\outcome_tracker.py:266:         start_date = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
portfolio\outcome_tracker.py:267:         end_date = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
portfolio\outcome_tracker.py:268:         t = yf.Ticker(YF_MAP[ticker])
portfolio\outcome_tracker.py:269:         h = t.history(start=start_date, end=end_date)
portfolio\outcome_tracker.py:270:         if h.empty:
portfolio\outcome_tracker.py:271:             return None
portfolio\outcome_tracker.py:272:         target_date = target_dt.date()
portfolio\outcome_tracker.py:273:         candidates = h[h.index.date <= target_date]
portfolio\outcome_tracker.py:274:         if candidates.empty:
portfolio\outcome_tracker.py:275:             return float(h["Close"].iloc[0])
portfolio\outcome_tracker.py:276:         return float(candidates["Close"].iloc[-1])
portfolio\outcome_tracker.py:277: 
portfolio\outcome_tracker.py:278:     return None
portfolio\outcome_tracker.py:279: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, sys, types, json, tempfile
from pathlib import Path

# stub package
portfolio = types.ModuleType('portfolio')
portfolio.__path__ = []
sys.modules['portfolio'] = portfolio
file_utils = types.ModuleType('portfolio.file_utils')

def atomic_write_jsonl(path, entries):
    Path(path).write_text('\\n'.join(json.dumps(e) for e in entries) + ('\\n' if entries else ''), encoding='utf-8')
file_utils.atomic_write_jsonl = atomic_write_jsonl
sys.modules['portfolio.file_utils'] = file_utils

spec = importlib.util.spec_from_file_location('portfolio.forecast_accuracy', 'portfolio/forecast_accuracy.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['portfolio.forecast_accuracy'] = mod
spec.loader.exec_module(mod)

# monkeypatch lookup
mod._lookup_price_at_time = lambda ticker, target_time, snapshot_file=None: 110.0

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'predictions.jsonl'
    entries = [
        {'ts':'2026-05-01T00:00:00+00:00','ticker':'A','current_price':100,'sub_signals':{},'outcome':{}},
        {'ts':'2026-05-01T00:00:00+00:00','ticker':'B','current_price':100,'sub_signals':{},'outcome':{}},
        {'ts':'2026-05-01T00:00:00+00:00','ticker':'C','current_price':100,'sub_signals':{},'outcome':{}},
    ]
    p.write_text('\\n'.join(json.dumps(e) for e in entries)+'\\n', encoding='utf-8')
    updated = mod.backfill_forecast_outcomes(max_entries=2, predictions_file=p)
    print('updated', updated)
    lines = p.read_text(encoding='utf-8').splitlines()
    print('linecount', len(lines))
    print(lines)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-08T15:33:17.737033Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 30, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1047, in write_text
    with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpxf6tsqq0\\predictions.jsonl'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<frozen os>", line 366, in walk
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpxf6tsqq0'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 23, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 950, in __exit__
    self.cleanup()
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 954, in cleanup
    self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 934, in _rmtree
    _shutil.rmtree(name, onexc=onexc)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 781, in rmtree
    return _rmtree_unsafe(path, onexc)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 623, in _rmtree_unsafe
    for dirpath, dirnames, filenames in results:
                                        ^^^^^^^
  File "<frozen os>", line 369, in walk
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 621, in onerror
    onexc(os.scandir, err.filename, err)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 906, in onexc
    _resetperms(path)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 286, in _resetperms
    _dont_follow_symlinks(_os.chmod, path, 0o700)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 277, in _dont_follow_symlinks
    func(path, *args)
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpxf6tsqq0'

 exited 1 in 689ms:
Traceback (most recent call last):
  File "<stdin>", line 30, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1047, in write_text
    with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpxf6tsqq0\\predictions.jsonl'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<frozen os>", line 366, in walk
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpxf6tsqq0'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 23, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 950, in __exit__
    self.cleanup()
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 954, in cleanup
    self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 934, in _rmtree
    _shutil.rmtree(name, onexc=onexc)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 781, in rmtree
    return _rmtree_unsafe(path, onexc)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 623, in _rmtree_unsafe
    for dirpath, dirnames, filenames in results:
                                        ^^^^^^^
  File "<frozen os>", line 369, in walk
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 621, in onerror
    onexc(os.scandir, err.filename, err)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 906, in onexc
    _resetperms(path)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 286, in _resetperms
    _dont_follow_symlinks(_os.chmod, path, 0o700)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 277, in _dont_follow_symlinks
    func(path, *args)
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpxf6tsqq0'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, sys, types, json
from pathlib import Path

# stub package
portfolio = types.ModuleType('portfolio')
portfolio.__path__ = []
sys.modules['portfolio'] = portfolio
file_utils = types.ModuleType('portfolio.file_utils')

def atomic_write_jsonl(path, entries):
    Path(path).write_text('\\n'.join(json.dumps(e) for e in entries) + ('\\n' if entries else ''), encoding='utf-8')
file_utils.atomic_write_jsonl = atomic_write_jsonl
sys.modules['portfolio.file_utils'] = file_utils

spec = importlib.util.spec_from_file_location('portfolio.forecast_accuracy', 'portfolio/forecast_accuracy.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['portfolio.forecast_accuracy'] = mod
spec.loader.exec_module(mod)
mod._lookup_price_at_time = lambda ticker, target_time, snapshot_file=None: 110.0

p = Path('portfolio/test_predictions.jsonl')
entries = [
    {'ts':'2026-05-01T00:00:00+00:00','ticker':'A','current_price':100,'sub_signals':{},'outcome':{}},
    {'ts':'2026-05-01T00:00:00+00:00','ticker':'B','current_price':100,'sub_signals':{},'outcome':{}},
    {'ts':'2026-05-01T00:00:00+00:00','ticker':'C','current_price':100,'sub_signals':{},'outcome':{}},
]
p.write_text('\\n'.join(json.dumps(e) for e in entries)+'\\n', encoding='utf-8')
updated = mod.backfill_forecast_outcomes(max_entries=2, predictions_file=p)
print('updated', updated)
lines = p.read_text(encoding='utf-8').splitlines()
print('linecount', len(lines))
for line in lines:
    print(line)
p.unlink()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 634ms:
updated 2
linecount 1
{"ts": "2026-05-01T00:00:00+00:00", "ticker": "A", "current_price": 100, "sub_signals": {}, "outcome": {"1h": {"actual_price": 110.0, "change_pct": 10.0, "backfilled_at": "2026-05-08T15:33:33.732688+00:00"}, "24h": {"actual_price": 110.0, "change_pct": 10.0, "backfilled_at": "2026-05-08T15:33:33.732688+00:00"}}}

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse -File | Select-String -Pattern 'backfill_forecast_outcomes\\('" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 744ms:

portfolio\forecast_accuracy.py:275:def [7mbackfill_forecast_outcomes([0mmax_entries=500, predictions_file=None,


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib.util, sys, types, json, sqlite3
from pathlib import Path

# Stub package and dependencies
portfolio = types.ModuleType('portfolio'); portfolio.__path__ = []
sys.modules['portfolio'] = portfolio
file_utils = types.ModuleType('portfolio.file_utils')
file_utils.atomic_write_json = lambda path, data: None
file_utils.load_json = lambda path, default=None: default
file_utils.load_jsonl = lambda path: [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
file_utils.load_jsonl_tail = lambda path, max_entries=50000: [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()][-max_entries:]
file_utils.atomic_append_jsonl = lambda path, entry: None
sys.modules['portfolio.file_utils'] = file_utils

tickers = types.ModuleType('portfolio.tickers')
tickers.DISABLED_SIGNALS = set()
tickers.SIGNAL_NAMES = ['sig']
sys.modules['portfolio.tickers'] = tickers

# import signal_db
spec = importlib.util.spec_from_file_location('portfolio.signal_db', 'portfolio/signal_db.py')
signal_db = importlib.util.module_from_spec(spec); sys.modules['portfolio.signal_db'] = signal_db; spec.loader.exec_module(signal_db)

# import accuracy_stats
spec2 = importlib.util.spec_from_file_location('portfolio.accuracy_stats', 'portfolio/accuracy_stats.py')
acc = importlib.util.module_from_spec(spec2); sys.modules['portfolio.accuracy_stats'] = acc; spec2.loader.exec_module(acc)

# redirect paths
work = Path('portfolio')
db_path = work / 'tmp_signal.db'
jsonl_path = work / 'tmp_signal.jsonl'
if db_path.exists(): db_path.unlink()
if jsonl_path.exists(): jsonl_path.unlink()
signal_db.DEFAULT_DB_PATH = db_path
acc.SIGNAL_LOG = jsonl_path

# Write DB with one entry
entry1 = {'ts':'2026-05-01T00:00:00+00:00','trigger_reasons':[],'fx_rate':1,'tickers':{'A':{'price_usd':1,'consensus':'BUY','buy_count':1,'sell_count':0,'total_voters':1,'signals':{'sig':'BUY'},'regime':'unknown'}},'outcomes':{}}
db = signal_db.SignalDB(db_path)
db.insert_snapshot(entry1)
db.close()
# JSONL has a newer entry too
entry2 = {'ts':'2026-05-02T00:00:00+00:00','trigger_reasons':[],'fx_rate':1,'tickers':{'B':{'price_usd':1,'consensus':'BUY','buy_count':1,'sell_count':0,'total_voters':1,'signals':{'sig':'BUY'},'regime':'unknown'}},'outcomes':{}}
jsonl_path.write_text(json.dumps(entry1)+'\\n'+json.dumps(entry2)+'\\n')
print(acc.load_entries())
# cleanup
jsonl_path.unlink(); db_path.unlink()
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 602ms:
[{'ts': '2026-05-01T00:00:00+00:00', 'trigger_reasons': [], 'fx_rate': 1.0, 'tickers': {'A': {'price_usd': 1.0, 'consensus': 'BUY', 'buy_count': 1, 'sell_count': 0, 'total_voters': 1, 'signals': {'sig': 'BUY'}, 'regime': 'unknown'}}, 'outcomes': {}}]

  ✓ Inspect diff against merge base
  ✓ Analyze changed files for correctness issues
  → Return prioritized findings in JSON
codex
The patch introduces a confirmed data-loss bug in forecast backfilling and accuracy-data bugs that can leave downstream signal evaluation stale or wrong. Because these issues affect persisted history and core accuracy computations, the patch should not be considered correct.

Full review comments:

- [P1] Preserve the unprocessed tail when `max_entries` is reached — Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\forecast_accuracy.py:343-348
  When `updated >= max_entries`, the loop breaks and the function rewrites `predictions_file` with only `modified_entries`. Because that list contains just the prefix visited before the break, every later JSONL row is silently discarded. This is easy to hit once more than `max_entries` mature outcomes exist in the file (default 500), and it permanently deletes forecast history instead of stopping after N updates.

- [P1] Fall back when SQLite is behind the JSONL log — Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\accuracy_stats.py:150-153
  If any SQLite dual-write fails and `signal_log.jsonl` gets ahead of `signal_log.db` (which `log_signal_snapshot()` already anticipates with the "SQLite may lag" warning), this branch still returns `db.load_entries()` as soon as the DB has one row. That makes every caller of `load_entries()`—including `signal_accuracy()`, `accuracy_by_ticker_signal_cached()`, and the signal-engine accuracy gates—ignore newer snapshots/outcomes that exist only in JSONL, so decisions can stay stale indefinitely instead of using the fresher source.

- [P2] Fetch intraday stock data for sub-day outcome backfills — Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\outcome_tracker.py:269-276
  For tickers in `YF_MAP`, this yfinance call uses the default daily interval and then picks the last close on or before `target_date`. That means the 3h/4h/12h horizons in `HORIZONS` are backfilled with an end-of-day close rather than a price near `target_ts`, which mislabels short-horizon outcomes and poisons downstream accuracy stats for stock tickers.
The patch introduces a confirmed data-loss bug in forecast backfilling and accuracy-data bugs that can leave downstream signal evaluation stale or wrong. Because these issues affect persisted history and core accuracy computations, the patch should not be considered correct.

Full review comments:

- [P1] Preserve the unprocessed tail when `max_entries` is reached — Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\forecast_accuracy.py:343-348
  When `updated >= max_entries`, the loop breaks and the function rewrites `predictions_file` with only `modified_entries`. Because that list contains just the prefix visited before the break, every later JSONL row is silently discarded. This is easy to hit once more than `max_entries` mature outcomes exist in the file (default 500), and it permanently deletes forecast history instead of stopping after N updates.

- [P1] Fall back when SQLite is behind the JSONL log — Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\accuracy_stats.py:150-153
  If any SQLite dual-write fails and `signal_log.jsonl` gets ahead of `signal_log.db` (which `log_signal_snapshot()` already anticipates with the "SQLite may lag" warning), this branch still returns `db.load_entries()` as soon as the DB has one row. That makes every caller of `load_entries()`—including `signal_accuracy()`, `accuracy_by_ticker_signal_cached()`, and the signal-engine accuracy gates—ignore newer snapshots/outcomes that exist only in JSONL, so decisions can stay stale indefinitely instead of using the fresher source.

- [P2] Fetch intraday stock data for sub-day outcome backfills — Q:\finance-analyzer\.worktrees\adv-signals-core\portfolio\outcome_tracker.py:269-276
  For tickers in `YF_MAP`, this yfinance call uses the default daily interval and then picks the last close on or before `target_date`. That means the 3h/4h/12h horizons in `HORIZONS` are backfilled with an end-of-day close rather than a price near `target_ts`, which mislabels short-horizon outcomes and poisons downstream accuracy stats for stock tickers.
