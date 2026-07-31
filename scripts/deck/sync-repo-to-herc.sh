#!/usr/bin/env bash
# sync-repo-to-herc.sh — push the Deck's git history to herc2. No GitHub involved.
#
# Why a bundle rather than a git remote: herc2's checkout lives at a Windows
# drive path (Q:\finance-analyzer). Git's SSH URL parser chokes on the drive
# colon, and pushing into a non-bare checked-out repo needs
# receive.denyCurrentBranch=updateInstead configured on the far side. A bundle
# sidesteps both — it is a single file, verifiable before use, and needs no
# daemon or config on either end.
#
# Safety properties, deliberate:
#   - NEVER force-pushes and never resets herc2's tree. If herc2 has commits the
#     Deck lacks, this reports the divergence and stops rather than choosing.
#   - Fast-forward only. A merge that isn't a fast-forward is a human decision.
#   - Refuses to run with a dirty Deck tree, so what lands on herc2 is exactly a
#     committed state you can name.
#   - Carries only git-tracked content. data/swedbank_*.json (real positions),
#     config.json (symlink to secrets outside the repo) and every other
#     gitignored runtime file stay on the Deck by construction.
#
# Wrap in with-herc.sh so herc2 gets shut down again if we woke it:
#   scripts/deck/with-herc.sh scripts/deck/sync-repo-to-herc.sh
#
# Usage:
#   scripts/deck/sync-repo-to-herc.sh [branch]     # default: main
set -uo pipefail

BRANCH="${1:-main}"
REPO="${REPO:-$HOME/projects/finance-analyzer}"
HERC_REPO='Q:\finance-analyzer'
HERC_REPO_SH='/q/finance-analyzer'
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 herc2"
BUNDLE_LOCAL="/tmp/fa-sync-${BRANCH}.bundle"
BUNDLE_REMOTE='C:\Users\herc2\fa-sync.bundle'
BUNDLE_REMOTE_SH='fa-sync.bundle'   # home-relative: Windows scp rejects /c/Users/...

die() { echo "sync: $*" >&2; exit 1; }

cd "$REPO" || die "no repo at $REPO"

# 1. Refuse to sync an ambiguous state.
# Block on uncommitted CODE. Tolerate churn in data/, which live loops rewrite
# continuously (metals_swing_state.json et al are tracked but are runtime state)
# — blocking on those would make this script unrunnable whenever a loop is up.
DIRTY_CODE=$(git status --porcelain --untracked-files=no | awk '{print $2}' | grep -v '^data/' || true)
if [ -n "$DIRTY_CODE" ]; then
    echo "$DIRTY_CODE" | sed 's/^/  /' >&2
    die "uncommitted code above — commit or stash first."
fi
DIRTY_DATA=$(git status --porcelain --untracked-files=no | awk '{print $2}' | grep '^data/' || true)
[ -n "$DIRTY_DATA" ] && echo "sync: note — uncommitted data/ churn, not sent: $(echo "$DIRTY_DATA" | tr '\n' ' ')"
git rev-parse --verify "$BRANCH" >/dev/null 2>&1 || die "no such branch: $BRANCH"
LOCAL_SHA=$(git rev-parse "$BRANCH")
echo "sync: Deck $BRANCH @ ${LOCAL_SHA:0:8}"

# 2. Reachability.
# NOT `ssh herc2 true` — herc2 is Windows and has no `true` command, so that
# probe exits non-zero on a perfectly healthy connection. Also retry: with-herc
# declares herc2 awake off an RDP probe, which answers before sshd does.
SSH_UP=""
for _try in 1 2 3 4 5 6; do
    if [ "$($SSH 'echo ok' 2>/dev/null | tr -d '\r')" = "ok" ]; then SSH_UP=1; break; fi
    sleep 5
done
[ -n "$SSH_UP" ] || die "herc2 unreachable over SSH after 30s. Wake it first:
    scripts/deck/with-herc.sh scripts/deck/sync-repo-to-herc.sh $BRANCH"

# 3. What does herc2 already have? Compare before sending anything.
REMOTE_SHA=$($SSH "git -C $HERC_REPO rev-parse $BRANCH 2>nul" 2>/dev/null | tr -d '\r')
if [ -z "$REMOTE_SHA" ]; then
    die "could not read $BRANCH on herc2 at $HERC_REPO (is the repo there, is git on PATH?)"
fi
echo "sync: herc2 $BRANCH @ ${REMOTE_SHA:0:8}"

if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
    echo "sync: already identical — nothing to do."
    exit 0
fi

if ! git cat-file -e "$REMOTE_SHA^{commit}" 2>/dev/null; then
    die "herc2 is at $REMOTE_SHA which the Deck has never seen — herc2 has its own
commits. Refusing to overwrite. Inspect both sides and merge by hand."
fi
if ! git merge-base --is-ancestor "$REMOTE_SHA" "$LOCAL_SHA"; then
    die "herc2's $BRANCH is not an ancestor of the Deck's — the histories have
diverged. Refusing to force. Reconcile by hand."
fi

AHEAD=$(git rev-list --count "${REMOTE_SHA}..${LOCAL_SHA}")
echo "sync: fast-forwardable, $AHEAD commit(s) to send"

# 4. Bundle only what herc2 is missing, then verify it before shipping.
git bundle create "$BUNDLE_LOCAL" "${REMOTE_SHA}..${BRANCH}" "$BRANCH" >/dev/null 2>&1 \
    || die "git bundle failed"
git bundle verify "$BUNDLE_LOCAL" >/dev/null 2>&1 || die "bundle failed verification"
echo "sync: bundle $(du -h "$BUNDLE_LOCAL" | cut -f1)"

scp -q "$BUNDLE_LOCAL" "herc2:${BUNDLE_REMOTE_SH}" || die "scp failed"

# 5. Fetch + fast-forward on herc2. --ff-only is the guard: if anything moved on
#    herc2 between our check and now, this fails rather than clobbering.
OUT=$($SSH "git -C $HERC_REPO fetch \"$BUNDLE_REMOTE\" ${BRANCH}:refs/remotes/deck/${BRANCH} --force && git -C $HERC_REPO merge --ff-only refs/remotes/deck/${BRANCH}" 2>&1 | tr -d '\r')
RC=$?
echo "$OUT" | sed 's/^/  herc2: /'
[ $RC -eq 0 ] || die "herc2-side fetch/merge failed (see above)"

NEW_SHA=$($SSH "git -C $HERC_REPO rev-parse $BRANCH 2>nul" 2>/dev/null | tr -d '\r')
$SSH "del \"$BUNDLE_REMOTE\"" >/dev/null 2>&1
rm -f "$BUNDLE_LOCAL"

[ "$NEW_SHA" = "$LOCAL_SHA" ] \
    && echo "sync: herc2 now at ${NEW_SHA:0:8} — matches Deck ✓" \
    || die "herc2 ended at ${NEW_SHA:0:8}, expected ${LOCAL_SHA:0:8}"
