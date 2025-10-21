#!/bin/bash

# Kiểm tra xem ngrok có được cài đặt không
if ! command -v ngrok &> /dev/null; then
    echo "Lỗi: ngrok không được cài đặt. Vui lòng cài đặt ngrok trước khi chạy script."
    exit 1
fi

# Chạy lệnh ngrok
ngrok http --url=living-tortoise-polite.ngrok-free.app 8000
