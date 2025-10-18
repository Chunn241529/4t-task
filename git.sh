#!/bin/bash

set -e  # Thoát ngay nếu có lỗi

# Auto-detect project root
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$PROJECT_ROOT" || exit 1

# ❌ XÓA GIT REPO CŨ
echo "🗑️  XÓA GIT REPO CŨ..."
rm -rf .git
echo "✅ Đã xóa!"

# 1. TẠO REPO MỚI
echo "📦 Tạo repo mới..."
git init
git branch -M main
echo "✅ OK!"

# 2. CẤU HÌNH GIT
git config user.name "Chunn241529"
git config user.email "chunn241529@gmail.com"
echo "✅ $(git config user.name)"

# 3. SETUP REMOTE + TẠO REPO TRÊN GITHUB TỰ ĐỘNG
PROJECT_NAME=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
echo "🌐 Tạo repo: Chunn241529/$PROJECT_NAME"
REMOTE_URL="https://github.com/Chunn241529/$PROJECT_NAME.git"
git remote add origin "$REMOTE_URL"

# 4. COMMIT MESSAGE
echo "📝 Nhập message:"
read -r commit_message
commit_message=${commit_message:-"✨ Initial commit"}

# 5. INITIAL COMMIT
git add .
git commit -m "$commit_message"
echo "✅ Commit OK!"

# 6. PUSH MAIN - FIX TOKEN PASTE + TẠO REPO AUTO
echo "🔐 Setup token (PASTE ĐƯỢC):"
echo "Copy token từ github.com/settings/tokens → Paste vào dưới → ENTER"

# DÙNG HERE-DOC ĐỂ PASTE DỄ
cat > ~/.git-credentials << 'EOF'
protocol=https
host=github.com
username=Chunn241529
password=
EOF

echo -n "Paste TOKEN vào đây: "
read -r -e TOKEN  # -e cho phép PASTE + ARROW KEYS

# GHI TOKEN VÀO FILE
sed -i '' "s/password=/password=$TOKEN/" ~/.git-credentials
chmod 600 ~/.git-credentials
git config --global credential.helper store

echo "✅ Token saved! Push main..."
git push -u origin main
echo "✅ MAIN PUSHED!"

# 7. SMART BRANCH
SMART_BRANCH="${PROJECT_NAME}-$(date +%Y%m%d)-wip"
echo "🌿 Tạo branch: $SMART_BRANCH"
git checkout -b "$SMART_BRANCH"
git push -u origin "$SMART_BRANCH"
echo "✅ BRANCH OK!"

# 8. TAG
TAG_NAME="${PROJECT_NAME}-$(date +%Y%m%d)"
git tag -a "$TAG_NAME" -m "WIP"
git push origin "$TAG_NAME"

# 9. PR TEMPLATE
mkdir -p .github
cat > .github/PULL_REQUEST_TEMPLATE.md << EOF
# 🚀 $PROJECT_NAME
**Changes:** $commit_message
**Branch:** $SMART_BRANCH
EOF
git add .github/
git commit -m "🤖 PR template"
git push

echo "🎉 HOÀN THÀNH 100%!"
echo "📍 $SMART_BRANCH"
echo "🔗 https://github.com/Chunn241529/$PROJECT_NAME"
