import httpx
import json
import base64
from typing import Optional
from textual.widgets import Static, Markdown
from textual.containers import ScrollableContainer
from textual.reactive import reactive

class AnimatedSpinner(Static):
    """A custom Static widget that animates a spinner using a sequence of characters."""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    current_index = reactive(0)

    def on_mount(self) -> None:
        """Start the animation when the widget is mounted."""
        self.set_interval(0.1, self.update_spinner)

    def update_spinner(self) -> None:
        """Cycle through spinner characters."""
        self.current_index = (self.current_index + 1) % len(self.spinner_chars)
        self.update(self.spinner_chars[self.current_index])

async def send_chat_request(
    http_client: httpx.AsyncClient,
    message: str,
    conversation_id: Optional[int],
    attached_file_path: Optional[str],
    chat_history: ScrollableContainer,
) -> Optional[int]:
    """Gửi yêu cầu chat đến API và hiển thị phản hồi."""
    json_payload = {"message": {"message": message}}
    if attached_file_path:
        try:
            with open(attached_file_path, "rb") as f:
                encoded_file = base64.b64encode(f.read()).decode("utf-8")
                json_payload["file"] = encoded_file
        except Exception as e:
            chat_history.mount(Static(f"[red]Lỗi khi đọc file: {e}[/red]"))
            return None
    params = {"conversation_id": conversation_id} if conversation_id else {}

    try:
        # Thêm spinner trước khi stream
        spinner = AnimatedSpinner("⠋", classes="spinner")
        spinner.styles.width = 1
        spinner.styles.height = 1
        spinner.styles.color = "white"
        spinner.styles.offset = (0, 0)  # Bottom-left relative to container
        chat_history.mount(spinner)
        chat_history.scroll_end()

        async with http_client.stream(
            "POST", "/chat", params=params, json=json_payload
        ) as response:
            response.raise_for_status()
            accumulated_content = ""
            ai_response_md = None

            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                    if not content:
                        continue
                    try:
                        data_chunk = json.loads(content)
                        if "conversation_id" in data_chunk:
                            conversation_id = data_chunk["conversation_id"]
                            continue
                        elif data_chunk.get("done"):
                            if accumulated_content and ai_response_md:
                                ai_response_md.update(accumulated_content)
                            break
                        elif data_chunk.get("error"):
                            chat_history.mount(
                                Static(
                                    f"[bold red]Lỗi Stream: {data_chunk['error']}[/bold red]"
                                )
                            )
                            break
                        elif data_chunk.get("content"):
                            decoded_content = (
                                data_chunk["content"]
                                .encode()
                                .decode("utf-8", errors="replace")
                            )
                            accumulated_content += decoded_content
                            if not ai_response_md:
                                chat_history.mount(Static(""))
                                ai_response_md = Markdown("")
                                chat_history.mount(ai_response_md)
                            ai_response_md.update(accumulated_content)
                            chat_history.scroll_end()
                    except json.JSONDecodeError:
                        decoded_content = content.encode().decode(
                            "utf-8", errors="replace"
                        )
                        accumulated_content += decoded_content
                        if not ai_response_md:
                            chat_history.mount(Static(""))
                            ai_response_md = Markdown("")
                            chat_history.mount(ai_response_md)
                        ai_response_md.update(accumulated_content)
                        chat_history.scroll_end()
            # Xóa spinner sau khi stream xong
            spinner.remove()
            return conversation_id

    except httpx.HTTPStatusError as e:
        # Xóa spinner nếu lỗi
        spinner.remove()
        chat_history.mount(
            Static(
                f"[bold red]Lỗi API {e.response.status_code}: {e.response.text}[/red]"
            )
        )
        if e.response.status_code in (401, 403):
            return "auth_error"
        return None
    except httpx.ConnectError:
        # Xóa spinner nếu lỗi kết nối
        spinner.remove()
        chat_history.mount(
            Static(f"[bold red]Lỗi kết nối tới {http_client.base_url}.[/bold red]")
        )
        return None


async def fetch_conversations(
    http_client: httpx.AsyncClient, chat_history: ScrollableContainer
) -> None:
    """Lấy danh sách các cuộc hội thoại từ API."""
    try:
        response = await http_client.get("/conversations")
        response.raise_for_status()
        conversations = response.json()
        if not conversations:
            chat_history.mount(Static("Chưa có cuộc hội thoại nào."))
            return
        history_text = "[bold]Danh sách cuộc hội thoại:[/bold]\n" + "\n".join(
            [
                f"- ID: {conv['id']} (Tạo lúc: {conv['created_at']})"
                for conv in conversations
            ]
        )
        chat_history.mount(Static(history_text))
        chat_history.scroll_end()
    except Exception as e:
        chat_history.mount(Static(f"[red]Lỗi khi tải lịch sử: {e}[/red]"))


async def load_conversation_history(
    http_client: httpx.AsyncClient, conv_id: int, chat_history: ScrollableContainer
) -> None:
    """Tải lịch sử cuộc hội thoại từ API."""
    chat_history.query("*").remove()
    chat_history.mount(
        Static(f"Đang tải lịch sử cho cuộc hội thoại [bold cyan]#{conv_id}[/]...")
    )
    try:
        response = await http_client.get(f"/conversations/{conv_id}/messages")
        response.raise_for_status()
        messages = response.json()
        for msg in messages:
            chat_history.mount(Static(""))
            if msg["role"] == "user":
                chat_history.mount(Static(f">>> {msg['content']}"))
            else:
                chat_history.mount(Markdown(msg['content']))
        chat_history.scroll_end()
        chat_history.mount(
            Static(f"Bạn đang ở trong cuộc hội thoại [bold cyan]#{conv_id}[/].")
        )
    except Exception as e:
        chat_history.mount(Static(f"[red]Lỗi tải lịch sử chat: {e}[/red]"))
