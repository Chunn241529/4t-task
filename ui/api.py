import httpx
import json
import base64
from typing import Optional
from textual.widgets import Static, Markdown
from textual.containers import ScrollableContainer
from textual.reactive import reactive
import asyncio


class AnimatedSpinner(Static):
    """A custom Static widget that animates a spinner using a sequence of characters."""

    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
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
    """Gửi yêu cầu chat đến API và hiển thị phản hồi mượt mà với spinner."""
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
        # Khởi tạo biến
        accumulated_content = ""
        ai_response_md = None
        initial_spinner = None
        initial_spinner_container = None
        response_spinner = None
        response_spinner_container = None

        # Hiển thị spinner ban đầu khi gửi yêu cầu
        initial_spinner = AnimatedSpinner("⠋", classes="spinner")
        initial_spinner.styles.width = 1
        initial_spinner.styles.height = 1
        initial_spinner.styles.color = "white"
        initial_spinner_container = Static("")
        initial_spinner_container.styles.display = "block"
        initial_spinner_container.styles.padding = (0, 0, 0, 2)
        chat_history.mount(initial_spinner_container)
        initial_spinner_container.mount(initial_spinner)

        async with http_client.stream(
            "POST", "/chat", params=params, json=json_payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    content = line[len("data:") :].strip()
                    if not content:
                        continue
                    try:
                        data_chunk = json.loads(content)
                        if "conversation_id" in data_chunk:
                            conversation_id = data_chunk["conversation_id"]
                            continue
                        elif data_chunk.get("done"):
                            # Đợi UI render nội dung cuối cùng
                            if ai_response_md and accumulated_content:
                                ai_response_md.update(accumulated_content)
                                chat_history.scroll_end()
                            await asyncio.sleep(0.1)  # Đợi UI ổn định
                            break
                        elif data_chunk.get("error"):
                            # Xóa cả hai spinner nếu có lỗi
                            if initial_spinner_container:
                                initial_spinner_container.remove()
                            if response_spinner_container:
                                response_spinner_container.remove()
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
                                # Xóa spinner ban đầu khi phản hồi bắt đầu
                                if initial_spinner_container:
                                    initial_spinner_container.remove()
                                    initial_spinner_container = None
                                # Hiển thị phản hồi
                                chat_history.mount(Static(""))
                                ai_response_md = Markdown("")
                                chat_history.mount(ai_response_md)
                                # Hiển thị spinner phản hồi
                                response_spinner = AnimatedSpinner("⠋", classes="spinner")
                                response_spinner.styles.width = 1
                                response_spinner.styles.height = 1
                                response_spinner.styles.color = "white"
                                response_spinner_container = Static("")
                                response_spinner_container.styles.display = "block"
                                response_spinner_container.styles.padding = (0, 0, 0, 2)
                                chat_history.mount(response_spinner_container)
                                response_spinner_container.mount(response_spinner)
                            # Cập nhật nội dung tích lũy
                            ai_response_md.update(accumulated_content)
                            chat_history.scroll_end()
                            await asyncio.sleep(0.05)  # Đợi UI render
                    except json.JSONDecodeError:
                        decoded_content = content.encode().decode(
                            "utf-8", errors="replace"
                        )
                        accumulated_content += decoded_content
                        if not ai_response_md:
                            # Xóa spinner ban đầu khi phản hồi bắt đầu
                            if initial_spinner_container:
                                initial_spinner_container.remove()
                                initial_spinner_container = None
                            # Hiển thị phản hồi
                            chat_history.mount(Static(""))
                            ai_response_md = Markdown("")
                            chat_history.mount(ai_response_md)
                            # Hiển thị spinner phản hồi
                            response_spinner = AnimatedSpinner("⠋", classes="spinner")
                            response_spinner.styles.width = 1
                            response_spinner.styles.height = 1
                            response_spinner.styles.color = "white"
                            response_spinner_container = Static("")
                            response_spinner_container.styles.display = "block"
                            response_spinner_container.styles.padding = (0, 0, 0, 2)
                            chat_history.mount(response_spinner_container)
                            response_spinner_container.mount(response_spinner)
                        # Cập nhật nội dung tích lũy
                        ai_response_md.update(accumulated_content)
                        chat_history.scroll_end()
                        await asyncio.sleep(0.05)  # Đợi UI render

        # Đảm bảo nội dung cuối cùng được hiển thị
        if ai_response_md and accumulated_content:
            ai_response_md.update(accumulated_content)
            chat_history.scroll_end()
            await asyncio.sleep(0.1)  # Đợi UI ổn định

        # Xóa spinner phản hồi sau khi render hoàn tất
        if response_spinner_container:
            response_spinner_container.remove()

        return conversation_id

    except httpx.HTTPStatusError as e:
        # Xóa cả hai spinner nếu có lỗi
        if initial_spinner_container:
            initial_spinner_container.remove()
        if response_spinner_container:
            response_spinner_container.remove()
        chat_history.mount(
            Static(
                f"[bold red]Lỗi API {e.response.status_code}: {e.response.text}[/red]"
            )
        )
        if e.response.status_code in (401, 403):
            return "auth_error"
        return None
    except httpx.ConnectError:
        # Xóa cả hai spinner nếu có lỗi kết nối
        if initial_spinner_container:
            initial_spinner_container.remove()
        if response_spinner_container:
            response_spinner_container.remove()
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
) -> bool:  # Thêm kiểu trả về bool để báo hiệu thành công/thất bại
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
                chat_history.mount(Markdown(msg["content"]))
        chat_history.scroll_end()
        chat_history.mount(
            Static(f"Bạn đang ở trong cuộc hội thoại [bold cyan]#{conv_id}[/].")
        )
        return True  # Tải thành côngset
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            chat_history.mount(
                Static(f"[red]Conversation #{conv_id} không tồn tại.[/red]")
            )
        else:
            chat_history.mount(
                Static(f"[red]Lỗi khi tải lịch sử: {e.response.status_code} - {e.response.text}[/red]")
            )
        return False  # Tải thất bại
    except Exception as e:
        chat_history.mount(Static(f"[red]Lỗi khi tải lịch sử: {e}[/red]"))
        return False  # Tải thất bại
