#!/bin/bash
# git-smart.sh - Git workflow thông minh HOÀN CHỈNH
# Sử dụng: ./git-smart.sh [commit_message]

set -e

REPO_NAME="4t-task"
GITHUB_USER="Chunn241529"
REMOTE="origin"
BRANCH_MAIN="main"

# 1. TẠO CONFIG GIT (TỰ ĐỘNG)
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

# 2. TẠO REPOSITORY NẾU CHƯA CÓ
init_repo() {
    if [[ ! -d ".git" ]]; then
        git init
        echo "# $REPO_NAME" > README.md
        git add README.md
        git commit -m "Initial commit"

        # TẠO GITHUB REPO TỰ ĐỘNG
        if ! curl -s "https://github.com/$GITHUB_USER/$REPO_NAME" | grep -q "404"; then
            echo "✅ Repo GitHub đã tồn tại"
        else
            echo "🚀 Tạo repo GitHub..."
            xdg-open "https://github.com/new?repo=$REPO_NAME"
            echo "   ⬆️ Tạo xong thì Enter để tiếp tục"
            read
        fi

        git remote add $REMOTE "https://github.com/$GITHUB_USER/$REPO_NAME.git"
        git push -u $REMOTE $BRANCH_MAIN
        echo "✅ Repository local + GitHub OK"
    fi
}

# 3. COMMIT
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

# 4. PUSH
do_push() {
    git push $REMOTE $(git branch --show-current)
    echo "✅ Đã push"
}

# 5. TẠO BRANCH
create_branch() {
    local branch_name="${1:-feature/$(date '+%Y%m%d-%H%M')}"
    git checkout -b "$branch_name"
    echo "✅ Tạo & checkout: $branch_name"
}

# 6. CHECKOUT BRANCH
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

# 7. MERGE
do_merge() {
    local target_branch="${1:-$BRANCH_MAIN}"
    local current_branch=$(git branch --show-current)

    if [[ "$current_branch" != "$target_branch" ]]; then
        git checkout "$target_branch"
        git merge "$current_branch" --no-ff -m "Merge '$current_branch'"
        git push $REMOTE $target_branch
        echo "✅ Merge $current_branch → $target_branch"
    else
        echo "⚠️  Đã ở branch đích"
    fi
}

# FULL WORKFLOW THEO THỨ TỰ
main() {
    setup_config
    init_repo
    do_commit "$1"
    do_push
    create_branch "$2"
    echo "🎉 Hoàn tất workflow!"
    echo "📂 https://github.com/$GITHUB_USER/$REPO_NAME"
}

# CHẠY
main "$@"
