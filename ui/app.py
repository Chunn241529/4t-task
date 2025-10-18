import httpx
import os
from typing import Optional
from textual.app import App, ComposeResult
from textual.widgets import Header, Input, Static, Footer
from textual.containers import Vertical, ScrollableContainer
from textual.reactive import var
from textual.binding import Binding
from config import API_BASE_URL, TOKEN_FILE_PATH
from api import send_chat_request, fetch_conversations, load_conversation_history

class FourTAIApp(App):
    """Giao diện TUI tối giản cho 4T AI, có khả năng lưu token."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Thoát"),
        Binding("ctrl+l", "new_chat", "Chat Mới (Clean màn hình)"),
    ]

    CSS = """
    Screen { background: #0D1117; color: #C9D1D9; }
    #login-area { height: auto; padding: 1; }
    #chat-history { padding: 1; }
    #input-area { dock: bottom; height: auto; padding: 0 1; }
    #chat-input { margin: 1 0; border: round #30363D; }
    #chat-input:focus { border: round #58A6FF; }
    #file-status { height: 1; color: #888; padding-left: 1; }
    .help-box {
        margin: 1 2;
        padding: 2 3;
        background: #1C2526;
        color: #C9D1D9;
        border: double #58A6FF;
        max-width: 80;
        text-align: left;
    }
    .help-item {
        padding: 0 1;
        margin: 0 1;
    }
    """

    current_conversation_id = var(None, init=False)
    attached_file_path = var(None, init=False)

    def __init__(self):
        super().__init__()
        self.http_client: Optional[httpx.AsyncClient] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="login-area"):
            yield Static("[bold]Vui lòng nhập Access Token của bạn cho 4T AI và nhấn Enter:[/bold]")
            yield Input(placeholder="dán token...", password=True, id="token-input")
        yield ScrollableContainer(id="chat-history")
        with Vertical(id="input-area"):
            yield Static("", id="file-status")
            yield Input(
                placeholder="Nhập tin nhắn hoặc /help để xem lệnh...",
                id="chat-input",
                disabled=True,
            )
        yield Footer()

    async def on_mount(self) -> None:
        """Kiểm tra token đã lưu khi khởi động."""
        if os.path.exists(TOKEN_FILE_PATH):
            try:
                with open(TOKEN_FILE_PATH, "r") as f:
                    token = f.read().strip()
                if token:
                    await self.perform_login(token, is_saved_token=True)
                else:
                    self.query_one("#token-input").focus()
            except Exception as e:
                self.mount_info_log(f"[red]Lỗi khi đọc token: {e}[/red]")
                self.query_one("#token-input").focus()
        else:
            self.query_one("#token-input").focus()

    async def perform_login(self, token: str, is_saved_token: bool = False) -> None:
        """Thực hiện đăng nhập với token."""
        self.http_client = httpx.AsyncClient(
            base_url=API_BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=300.0,
        )
        self.query_one("#login-area").display = False
        chat_input = self.query_one("#chat-input")
        chat_input.disabled = False
        chat_input.focus()

        if is_saved_token:
            self.mount_info_log("[green]Đã tự động đăng nhập vào 4T AI thành công.[/green]")
        else:
            self.mount_info_log("[green]Đăng nhập vào 4T AI thành công! Token đã được lưu cho lần sau.[/green]")

        # Hiển thị thông báo chào mừng và lệnh /help
        self.mount_info_log("[bold cyan]Chào mừng bạn đến với 4T AI! Nhập tin nhắn hoặc xem các lệnh dưới đây:[/bold cyan]")
        await self.handle_client_command("/help", self.query_one("#chat-history"))

    def watch_attached_file_path(self, new_path: Optional[str]) -> None:
        """Cập nhật trạng thái file đính kèm."""
        status_widget = self.query_one("#file-status", Static)
        if new_path:
            filename = os.path.basename(new_path)
            status_widget.update(f"📎 Đính kèm: [bold cyan]{filename}[/]. Gõ /clearfile để gỡ.")
        else:
            status_widget.update("")

    async def handle_client_command(self, command: str, chat_history: ScrollableContainer) -> None:
        """Xử lý các lệnh client-side cho 4T AI."""
        parts = command.split(" ", 1)
        cmd = parts[0]
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            help_text = """[bold][#58A6FF]📚 HƯỚNG DẪN SỬ DỤNG 4T AI[/bold]

[#58A6FF]Các lệnh có sẵn:[/]
> [bold][#58A6FF]/new[/]: Bắt đầu một cuộc hội thoại mới 🆕
> [bold][#58A6FF]/history[/]: Xem danh sách các cuộc hội thoại đã có 📜
> [bold][#58A6FF]/load <id>[/]: Tải lại lịch sử của một cuộc hội thoại 📂
> [bold][#58A6FF]/file <path>[/]: Đính kèm một file vào tin nhắn tiếp theo 📎
> [bold][#58A6FF]/clearfile[/]: Gỡ file đã đính kèm 🗑️
> [bold][#58A6FF]/clear[/]: Xóa trắng màn hình chat hiện tại 🧹
> [bold][#58A6FF]/logout[/]: Xóa token đã lưu và thoát 🚪
"""
            chat_history.mount(Static(help_text, classes="help-box"))
            chat_history.scroll_end()
        elif cmd == "/logout":
            if os.path.exists(TOKEN_FILE_PATH):
                try:
                    os.remove(TOKEN_FILE_PATH)
                    self.exit("Token đã được xóa. Vui lòng khởi động lại ứng dụng 4T AI.")
                except Exception as e:
                    chat_history.mount(Static(f"[red]Lỗi khi xóa token: {e}[/red]"))
            else:
                self.exit("Không có token nào được lưu để xóa. Đang thoát khỏi 4T AI...")
        elif cmd == "/new":
            await self.action_new_chat()
        elif cmd == "/history":
            await self.fetch_conversations()
        elif cmd == "/load":
            if args.isdigit():
                await self.load_conversation_history(int(args))
            else:
                chat_history.mount(Static(f"[red]Lỗi: ID cuộc hội thoại không hợp lệ.[/red]"))
        elif cmd == "/file":
            if os.path.exists(args):
                self.attached_file_path = args
            else:
                chat_history.mount(Static(f"[red]Lỗi: File không tồn tại: {args}[/red]"))
        elif cmd == "/clearfile":
            self.attached_file_path = None
        elif cmd == "/clear":
            chat_history.query("*").remove()
        else:
            chat_history.mount(Static(f"[yellow]Lệnh không xác định: {cmd}.[/yellow]"))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
      """Xử lý sự kiện khi người dùng gửi input."""
      if event.input.id == "token-input":
          token = event.value.strip()
          if not token:
              self.mount_info_log("[red]Token không được để trống.[/red]")
              return
          try:
              with open(TOKEN_FILE_PATH, "w") as f:
                  f.write(token)
          except Exception as e:
              self.mount_info_log(f"[red]Không thể lưu token: {e}[/red]")
          await self.perform_login(token)
      elif event.input.id == "chat-input":
          user_message = event.value.strip()
          event.input.value = ""
          if not user_message:
              return
          if user_message.startswith("/"):
              await self.handle_client_command(user_message, self.query_one("#chat-history"))
          else:
              chat_history = self.query_one("#chat-history")
              chat_history.mount(Static(""))
              chat_history.mount(Static(f">>> {user_message}"))
              chat_history.scroll_end()
              result = await send_chat_request(
                  self.http_client,
                  user_message,
                  self.current_conversation_id,
                  self.attached_file_path,
                  self.query_one("#chat-history")
              )
              if result == "auth_error":
                  self.query_one("#chat-input").disabled = True
              elif result is not None:
                  self.current_conversation_id = result
                  chat_history.scroll_end()
          self.attached_file_path = None

    async def fetch_conversations(self) -> None:
        """Lấy danh sách các cuộc hội thoại."""
        if self.http_client:
            await fetch_conversations(self.http_client, self.query_one("#chat-history"))

    async def load_conversation_history(self, conv_id: int) -> None:
        """Tải lịch sử cuộc hội thoại."""
        if self.http_client:
            await load_conversation_history(self.http_client, conv_id, self.query_one("#chat-history"))
            self.current_conversation_id = conv_id

    def mount_info_log(self, text: str) -> None:
        """Hiển thị thông báo trong chat history."""
        log_widget = Static(text)
        self.query_one("#chat-history").mount(log_widget)
        self.query_one("#chat-history").scroll_end()

    async def action_new_chat(self) -> None:
        """Bắt đầu một cuộc hội thoại mới."""
        self.current_conversation_id = None
        self.attached_file_path = None
        self.query_one("#chat-history").query("*").remove()
        self.query_one("#chat-input").focus()
        self.mount_info_log("Đã bắt đầu cuộc hội thoại mới với 4T AI.")
