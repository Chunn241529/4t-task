#!/bin/bash

# git-smart.sh - Git workflow thông minh (KHÔNG manual config)
# Sử dụng: ./git-smart.sh [commit_message]

set -e  # Dừng nếu có lỗi

REPO_NAME=$(basename "$PWD")
REMOTE="origin"
BRANCH_MAIN="main"

# 1. SKIP CONFIG - SỬ DỤNG GIT DEFAULT
setup_config() {
    echo "⏭️  Skip manual config - dùng Git default"
}

# 2. TẠO REPOSITORY NẾU CHƯA CÓ
init_repo() {
    if [[ ! -d ".git" ]]; then
        git init
        echo "# $REPO_NAME" > README.md
        git add README.md
        git commit -m "Initial commit"
        echo "✅ Repository đã tạo"
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
        echo "⚠️  Không có thay đổi để commit"
    fi
}

# 4. PUSH (TỰ ĐỘNG ADD REMOTE NẾU CHƯA CÓ)
do_push() {
    current_branch=$(git branch --show-current)

    if ! git remote | grep -q "$REMOTE"; then
        echo "❓ Chưa có remote. Nhập URL GitHub (Enter để skip):"
        read -r remote_url
        if [[ -n "$remote_url" ]]; then
            git remote add $REMOTE "$remote_url"
            echo "✅ Đã thêm remote: $remote_url"
        else
            echo "⚠️  Skip push (chưa có remote)"
            return
        fi
    fi

    git push -u $REMOTE $current_branch
    echo "✅ Đã push lên $REMOTE/$current_branch"
}

# 5. TẠO BRANCH
create_branch() {
    local branch_name="${1:-feature/$(date '+%Y%m%d-%H%M')}"
    git checkout -b "$branch_name"
    echo "✅ Đã tạo & checkout branch: $branch_name"
}

# 6. CHECKOUT BRANCH
checkout_branch() {
    local branch_name="$1"
    if git branch | grep -q "$branch_name"; then
        git checkout "$branch_name"
        echo "✅ Đã checkout: $branch_name"
    else
        echo "❌ Branch '$branch_name' không tồn tại"
        exit 1
    fi
}

# 7. MERGE
do_merge() {
    local target_branch="${1:-$BRANCH_MAIN}"
    local current_branch=$(git branch --show-current)

    if [[ "$current_branch" != "$target_branch" ]]; then
        git checkout "$target_branch"
        git merge "$current_branch" --no-ff -m "Merge branch '$current_branch'"
        echo "✅ Đã merge $current_branch vào $target_branch"
    else
        echo "⚠️  Đã ở branch đích"
    fi
}

# CHẠY THEO LUỒNG
main() {
    setup_config
    init_repo

    case "$1" in
        "commit")
            do_commit "$2"
            ;;
        "push")
            do_commit "$2"
            do_push
            ;;
        "branch")
            create_branch "$2"
            ;;
        "checkout")
            checkout_branch "$2"
            ;;
        "merge")
            do_merge "$2"
            ;;
        *)
            # DEFAULT: Full workflow
            do_commit "$1"
            do_push
            create_branch
            ;;
    esac
}

main "$@"
