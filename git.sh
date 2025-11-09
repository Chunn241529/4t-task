#!/bin/bash
# git.sh - Smart Git Helper Script
set -e

REPO_NAME="4T_task"
GITHUB_USER="Chunn241529"
REMOTE="origin"
BRANCH_MAIN="main"

# ====== COLORS ======
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
RED='\033[1;31m'
BLUE='\033[1;34m'
RESET='\033[0m'

# ====== CONFIG ======
setup_config() {
    if ! git config user.name &>/dev/null; then
        echo -e "${CYAN}👤 Nhập tên Git user:${RESET}"
        read -r user_name
        git config --global user.name "$user_name"
    fi
    if ! git config user.email &>/dev/null; then
        echo -e "${CYAN}📧 Nhập email:${RESET}"
        read -r user_email
        git config --global user.email "$user_email"
    fi
    git config pull.rebase false
    echo -e "${GREEN}✅ Git config OK${RESET}"
}

# ====== INIT ======
init_repo() {
    if [[ ! -d ".git" ]]; then
        git init
        echo "# $REPO_NAME" > README.md
        git add README.md
        git commit -m "Initial commit"
        echo -e "${GREEN}✅ Repo initialized${RESET}"
    fi

    if ! git remote | grep -q "$REMOTE"; then
        git remote add "$REMOTE" "https://github.com/$GITHUB_USER/$REPO_NAME.git"
        echo -e "${GREEN}✅ Remote added${RESET}"
    fi

    git pull "$REMOTE" "$BRANCH_MAIN" --allow-unrelated-histories || true
    git push -u "$REMOTE" "$BRANCH_MAIN"
    echo -e "${GREEN}✅ Repo synced with GitHub${RESET}"
}

# ====== COMMIT ======
do_commit() {
    local msg="${1:-Auto commit $(date '+%Y-%m-%d %H:%M')}"
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git add .
        git commit -m "$msg"
        echo -e "${GREEN}✅ Commit: $msg${RESET}"
    else
        echo -e "${YELLOW}⚠️  Không có thay đổi để commit${RESET}"
    fi
}

# ====== PUSH ======
do_push() {
    local branch
    branch=$(git branch --show-current)
    git pull "$REMOTE" "$branch" --no-edit || true
    git push "$REMOTE" "$branch"
    echo -e "${GREEN}✅ Push branch '$branch' OK${RESET}"
}

# ====== BRANCH ======
create_branch() {
    local branch_name="${1:-feature/$(date '+%Y%m%d-%H%M')}"
    git checkout -b "$branch_name"
    echo -e "${GREEN}✅ Tạo & chuyển sang branch: $branch_name${RESET}"
}

checkout_branch() {
    local branch_name="$1"
    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        git checkout "$branch_name"
        echo -e "${GREEN}✅ Checkout branch: $branch_name${RESET}"
    else
        echo -e "${RED}❌ Branch '$branch_name' không tồn tại${RESET}"
        exit 1
    fi
}

# ====== MERGE ======
do_merge() {
    local target_branch="${1:-$BRANCH_MAIN}"
    local current_branch
    current_branch=$(git branch --show-current)

    if [[ "$current_branch" == "$target_branch" ]]; then
        echo -e "${YELLOW}⚠️  Đang ở branch đích ($target_branch), không thể merge chính nó${RESET}"
        exit 0
    fi

    git checkout "$target_branch"
    git pull "$REMOTE" "$target_branch" || true
    git merge "$current_branch" --no-ff -m "Merge branch '$current_branch' into '$target_branch'"
    git push "$REMOTE" "$target_branch"
    echo -e "${GREEN}✅ Merge '$current_branch' → '$target_branch' thành công${RESET}"
}

# ====== REVERT/RESET ======
show_history() {
    echo -e "${CYAN}📜 Lịch sử commit (10 cái gần nhất):${RESET}"
    git log --oneline --graph -10 --color=always
}

revert_commit() {
    local commit_hash="$1"
    if [[ -z "$commit_hash" ]]; then
        echo -e "${CYAN}📜 Chọn commit để revert:${RESET}"
        show_history
        echo -e "${CYAN}📝 Nhập commit hash:${RESET}"
        read -r commit_hash
    fi
    
    if git show "$commit_hash" &>/dev/null; then
        git revert --no-edit "$commit_hash"
        echo -e "${GREEN}✅ Đã revert commit: $commit_hash${RESET}"
        echo -e "${YELLOW}📝 Revert tạo commit mới, cần push để áp dụng${RESET}"
    else
        echo -e "${RED}❌ Commit '$commit_hash' không tồn tại${RESET}"
        exit 1
    fi
}

reset_to_commit() {
    local commit_hash="$1"
    if [[ -z "$commit_hash" ]]; then
        echo -e "${CYAN}📜 Chọn commit để reset về:${RESET}"
        show_history
        echo -e "${CYAN}📝 Nhập commit hash:${RESET}"
        read -r commit_hash
    fi

    echo -e "${YELLOW}⚠️  CẢNH BÁO: Reset sẽ xóa các commit sau commit được chọn!${RESET}"
    echo -e "${CYAN}Chọn loại reset:${RESET}"
    echo -e "  ${GREEN}1. Soft${RESET} - Giữ thay đổi trong staging area"
    echo -e "  ${GREEN}2. Mixed${RESET} - Giữ thay đổi trong working directory (mặc định)"
    echo -e "  ${GREEN}3. Hard${RESET} - Xóa hết thay đổi"
    echo -e "${CYAN}Lựa chọn (1/2/3):${RESET}"
    read -r reset_type

    case "$reset_type" in
        1|"soft"|"Soft")
            git reset --soft "$commit_hash"
            echo -e "${GREEN}✅ Soft reset đến: $commit_hash${RESET}"
            ;;
        3|"hard"|"Hard")
            echo -e "${RED}🚨 HARD RESET - Tất cả thay đổi sau commit sẽ bị XÓA!${RESET}"
            echo -e "${CYAN}Bạn có chắc chắn? (y/N):${RESET}"
            read -r confirm
            if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
                git reset --hard "$commit_hash"
                echo -e "${GREEN}✅ Hard reset đến: $commit_hash${RESET}"
            else
                echo -e "${YELLOW}⚠️  Đã hủy reset${RESET}"
                exit 0
            fi
            ;;
        *)
            git reset --mixed "$commit_hash"
            echo -e "${GREEN}✅ Mixed reset đến: $commit_hash${RESET}"
            ;;
    esac
    
    echo -e "${YELLOW}📝 Cần push force để áp dụng reset (chỉ khi cần thiết)${RESET}"
}

# ====== STATUS ======
show_status() {
    echo -e "${CYAN}📂 Repo: $(basename "$(git rev-parse --show-toplevel 2>/dev/null)")${RESET}"
    echo -e "${CYAN}🌿 Branch hiện tại: $(git branch --show-current)${RESET}"
    echo
    git status -sb
}

# ====== HELP ======
show_help() {
    echo -e "${YELLOW}🚀 Cách dùng git.sh:${RESET}"
    echo -e "  ${GREEN}./git.sh init${RESET}                  - Khởi tạo repo & sync GitHub"
    echo -e "  ${GREEN}./git.sh commit 'msg'${RESET}          - Commit thay đổi"
    echo -e "  ${GREEN}./git.sh push${RESET}                  - Push branch hiện tại"
    echo -e "  ${GREEN}./git.sh branch new-branch${RESET}     - Tạo branch mới"
    echo -e "  ${GREEN}./git.sh checkout branch-name${RESET}  - Chuyển sang branch khác"
    echo -e "  ${GREEN}./git.sh merge main${RESET}            - Merge branch hiện tại → main"
    echo -e "  ${GREEN}./git.sh history${RESET}               - Xem lịch sử commit"
    echo -e "  ${GREEN}./git.sh revert [hash]${RESET}         - Revert commit cụ thể"
    echo -e "  ${GREEN}./git.sh reset [hash]${RESET}          - Reset về commit cũ"
    echo -e "  ${GREEN}./git.sh status${RESET}                - Xem trạng thái nhanh"
    echo -e "  ${GREEN}./git.sh help${RESET}                  - Hiển thị hướng dẫn này"
}

# ====== MAIN FLOW ======
case "$1" in
    init)
        setup_config
        init_repo
        ;;
    commit)
        setup_config
        do_commit "$2"
        ;;
    push)
        do_push
        ;;
    branch)
        create_branch "$2"
        ;;
    checkout)
        checkout_branch "$2"
        ;;
    merge)
        do_merge "$2"
        ;;
    history|log)
        show_history
        ;;
    revert)
        revert_commit "$2"
        ;;
    reset)
        reset_to_commit "$2"
        ;;
    status)
        show_status
        ;;
    ""|help|-h|--help)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Lệnh không hợp lệ:${RESET} '$1'"
        echo
        show_help
        ;;
esac
