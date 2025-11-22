import ollama
from typing import List, Dict


def stream_chat(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Stream chat với Ollama và accumulate full content từ raw chunks.
    Trả về full content sau khi stream kết thúc.
    """
    full_content = ""
    print("Bắt đầu streaming... (Raw chunks sẽ được in từng phần)")

    stream = ollama.chat(model=model, messages=messages, stream=True)  # Bật streaming

    for chunk in stream:
        # Raw chunk (dict tương đương JSON)
        print(f"{len(full_content.split())}: {chunk}")

        delta = chunk["message"]["content"]
        if delta:  # Chỉ nếu có nội dung mới
            full_content += delta
            print(f"Delta (real-time): {delta}", end="", flush=True)  # Hiển thị dần
            print()  # Newline cho chunk tiếp theo

    print("\n--- Streaming kết thúc ---")
    return full_content


# Sử dụng
if __name__ == "__main__":
    MODEL = "4T"  # Đảm bảo model đã pull
    messages = [{"role": "user", "content": "xin chào"}]

    result = stream_chat(MODEL, messages)
    print(f"\nFull Accumulated Content:\n{result}")
