#!/bin/bash
# git-smart.sh - FIX DIVERGENT BRANCHES 100%
set -e

REPO_NAME="4t-task"
GITHUB_USER="Chunn241529"
REMOTE="origin"
BRANCH_MAIN="main"

setup_config() {
    if ! git config user.name &>/dev/null; then
        echo "👤 Nhập tên:"
        read -r user_name
        git config --global user.name "$user_name"
        echo "📧 Nhập email:"
        read -r user_email
        git config --global user.email "$user_email"
        echo "✅ Config git hoàn tất"
    else
        echo "✅ Config git đã có"
    fi
}

# **FIX CHÍNH: SET PULL STRATEGY**
setup_pull_strategy() {
    git config pull.rebase false
    echo "✅ Pull strategy: MERGE (fixed)"
}

init_repo() {
    if [[ ! -d ".git" ]]; then
        git init
        echo "# $REPO_NAME" > README.md
        git add README.md
        git commit -m "Initial commit"
    fi

    if ! git remote | grep -q "$REMOTE"; then
        git remote add $REMOTE "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    fi

    setup_pull_strategy

    echo "🔄 Đồng bộ GitHub..."
    git pull $REMOTE $BRANCH_MAIN --allow-unrelated-histories
    git push -u $REMOTE $BRANCH_MAIN
    echo "✅ Repository đồng bộ OK"
}

do_commit() {
    local msg="${1:-Auto commit $(date '+%Y-%m-%d %H:%M')}"
    if ! git diff --quiet; then
        git add .
        git commit -m "$msg"
        echo "✅ Đã commit: $msg"
    else
        echo "⚠️  Không có thay đổi"
    fi
}

do_push() {
    git pull $REMOTE $(git branch --show-current) || true
    git push $REMOTE $(git branch --show-current)
    echo "✅ Đã push"
}

create_branch() {
    local branch_name="${1:-feature/$(date '+%Y%m%d-%H%M')}"
    git checkout -b "$branch_name"
    echo "✅ Tạo & checkout: $branch_name"
}

checkout_branch() {
    local branch_name="$1"
    if git branch | grep -q "$branch_name"; then
        git checkout "$branch_name"
        echo "✅ Checkout: $branch_name"
    else
        echo "❌ Branch không tồn tại"
        exit 1
    fi
}

do_merge() {
    local target_branch="${1:-$BRANCH_MAIN}"
    local current_branch=$(git branch --show-current)

    if [[ "$current_branch" != "$target_branch" ]]; then
        git checkout "$target_branch"
        git pull $REMOTE $target_branch || true
        git merge "$current_branch" --no-ff -m "Merge '$current_branch'"
        git push $REMOTE $target_branch
        echo "✅ Merge $current_branch → $target_branch"
    else
        echo "⚠️  Đã ở branch đích"
    fi
}

main() {
    setup_config
    init_repo
    do_commit "$1"
    do_push
    create_branch "$2"
    echo "🎉 Hoàn tất workflow!"
    echo "📂 https://github.com/$GITHUB_USER/$REPO_NAME"
}

main "$@"
