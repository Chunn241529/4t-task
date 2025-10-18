#!/bin/bash

# Kiểm tra xem môi trường ảo đã được kích hoạt hay chưa
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Môi trường ảo đã được kích hoạt: $VIRTUAL_ENV"
else
    # Tìm môi trường ảo trong thư mục hiện tại (venv hoặc .venv)
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "Đã kích hoạt môi trường ảo: venv"
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "Đã kích hoạt môi trường ảo: .venv"
    else
        echo "Lỗi: Không tìm thấy môi trường ảo (venv hoặc .venv) trong thư mục hiện tại."
        exit 1
    fi
fi

# Kiểm tra xem uvicorn đã được cài đặt chưa
if ! command -v uvicorn >/dev/null 2>&1; then
    echo "Lỗi: uvicorn không được cài đặt trong môi trường ảo. Hãy cài đặt bằng 'pip install uvicorn'."
    exit 1
fi

# Dọn dẹp cache Python (__pycache__ và file .pyc)
echo "Đang dọn dẹp cache Python..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
if [ $? -eq 0 ]; then
    echo "Đã xóa cache Python thành công."
else
    echo "Lỗi khi dọn dẹp cache Python."
fi

# Kiểm tra sự tồn tại của file app/main.py
if [ -f "app/main.py" ]; then
    echo "Chạy ứng dụng FastAPI từ app/main.py..."
    # python3 app/main.py
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Lỗi: Không tìm thấy file app/main.py trong thư mục hiện tại."
    exit 1
fi
