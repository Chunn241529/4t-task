import ollama

def test_ollama_think_low():
    """
    Test gpt-oss:20b với think="low" (tắt thinking gần hoàn toàn, stream nhanh).
    """
    messages = [
        {"role": "user", "content": "Tại sao bầu trời màu xanh?"}
    ]

    try:
        print("✅ **Bắt đầu stream với think=low...**")
        print("\n📝 **Response (nhanh, không thinking sâu):**\n")

        stream = ollama.chat(
            model="gpt-oss:20b",
            messages=messages,
            stream=True,
            think="low"  # ← Key fix: Dùng string "low" thay vì False
        )

        full_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            if content:  # Bỏ qua chunk rỗng
                print(content, end="", flush=True)
                full_response += content

        print(f"\n\n{'='*50}")
        print("✅ **Stream hoàn tất nhanh!**")
        print(f"🔢 Độ dài: {len(full_response)} ký tự")
        print(f"⏱️ Ước tính: Ít delay hơn so với medium")

    except Exception as e:
        print(f"\n❌ **Lỗi:** {e}")
        print("💡 Kiểm tra: ollama serve chạy? Model pull? Thử think='medium' để so sánh.")

# Chạy
if __name__ == "__main__":
    test_ollama_think_low()
