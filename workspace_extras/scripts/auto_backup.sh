#!/bin/bash
# Auto-backup script for RMDM code
# Commits changes to auto-backup branch and pushes to GitHub
# Called on shell login/logout via .bashrc / .bash_logout

REPO_DIR="/home/ubuntu/efs/RMDM"
BACKUP_BRANCH="auto-backup"
LOG_FILE="$REPO_DIR/logs/auto_backup.log"

mkdir -p "$(dirname "$LOG_FILE")"

backup() {
    cd "$REPO_DIR" || return 1

    local trigger="$1"  # "login" or "logout"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Save current branch
    local current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ -z "$current_branch" ]; then
        echo "[$timestamp] ERROR: not a git repo" >> "$LOG_FILE"
        return 1
    fi

    # Check if there are any changes to backup
    if git diff --quiet HEAD 2>/dev/null && [ -z "$(git ls-files --others --exclude-standard)" ]; then
        echo "[$timestamp] [$trigger] No changes to backup" >> "$LOG_FILE"
        return 0
    fi

    # Stash any staged changes to restore later
    local had_staged=false
    if ! git diff --cached --quiet 2>/dev/null; then
        had_staged=true
    fi

    # Create backup branch if it doesn't exist
    if ! git rev-parse --verify "$BACKUP_BRANCH" >/dev/null 2>&1; then
        git branch "$BACKUP_BRANCH" HEAD 2>/dev/null
    fi

    # Switch to backup branch, commit, switch back
    git stash push -q --include-untracked -m "auto-backup-stash" 2>/dev/null
    git checkout -q "$BACKUP_BRANCH" 2>/dev/null
    git merge -q --no-edit "$current_branch" 2>/dev/null
    git stash pop -q 2>/dev/null

    # Add all tracked + untracked (respects .gitignore)
    git add -A 2>/dev/null

    local changes=$(git diff --cached --stat 2>/dev/null | tail -1)
    if [ -n "$changes" ]; then
        git commit -q -m "auto-backup [$trigger] $timestamp

$changes" 2>/dev/null
        echo "[$timestamp] [$trigger] Backup committed: $changes" >> "$LOG_FILE"

        # Push to GitHub (both backup branch and current branch)
        git push -q origin "$BACKUP_BRANCH" 2>/dev/null && \
            echo "[$timestamp] [$trigger] Pushed $BACKUP_BRANCH to GitHub" >> "$LOG_FILE" || \
            echo "[$timestamp] [$trigger] Push failed (network?)" >> "$LOG_FILE"
    else
        echo "[$timestamp] [$trigger] No new changes on backup branch" >> "$LOG_FILE"
    fi

    # Switch back to original branch
    git checkout -q "$current_branch" 2>/dev/null

    # Also push current branch if it has a remote
    if git config "branch.$current_branch.remote" >/dev/null 2>&1; then
        git push -q origin "$current_branch" 2>/dev/null
    fi
}

backup "$@"
