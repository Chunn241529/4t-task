import logging
import httpx
import os
from typing import Optional
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Input, Static
from textual.containers import Vertical, ScrollableContainer, Horizontal
from textual.reactive import var
from textual.binding import Binding
import webbrowser

from config import API_BASE_URL, TOKEN_FILE_PATH
from api import (
    delete_all_conversation,
    delete_current_conversation,
    send_chat_request,
    fetch_conversations,
    load_conversation_history,
)

# Cấu hình logging vào file trong thư mục logs/
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(
    log_dir, f"app_{os.path.basename(__file__).replace('.py', '')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
    ],
)


class FourTAIApp(App):
    """Giao diện TUI tối giản cho FourT AI với nút đăng nhập và chức năng chat."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Thoát"),
        Binding("ctrl+l", "new_chat", "Chat mới"),
    ]

    CSS = """
    Screen { 
        background: #0D1117; 
        color: #C9D1D9; 
    }
    
    /* Login Area Styling - Compact & Professional */
    #login-area { 
        height: 100%; 
        align: center middle; 
        background: #0D1117;
    }
    
    .login-container {
        width: 50;
        height: auto;
        padding: 1 2;
        background: #161B22;
        border: tall #30363D;
    }
    
    .login-title {
        text-align: center;
        margin-bottom: 1;
        color: #58A6FF;
    }
    
    .login-button {
        width: 100%;
        margin: 1 0;
        background: #238636;
        color: white;
    }
    
    .login-button:hover {
        background: #2EA043;
    }
    
    .token-input-row {
        height: auto;
        margin: 1 0;
        align: center middle;
    }
    
    #token-input {
        width: 1fr;
        border: round #30363D;
        background: #010409;
        color: #C9D1D9;
        height: 3;
    }
    
    #token-input:focus {
        border: round #58A6FF;
    }
    
    #get-token-button {
        min-width: 12;
        height: 3;
        margin-left: 1;
        background: #1F6FEB;
        color: white;
    }
    
    #get-token-button:hover {
        background: #388BFD;
    }
    
    .login-instruction {
        text-align: center;
        margin: 0 0 1 0;
        color: #8B949E;
        text-style: italic;
    }
    
    /* Chat Area Styling */
    #chat-history { padding: 1; }
    #input-area { dock: bottom; height: auto; padding: 0 0; }
    #chat-input { 
        margin: 1 1; 
        background: #0D1117; 
        color: #C9D1D9;
        border: round #30363D;
        height: 3; /* Giảm chiều cao input */
    }
    #chat-input:focus { 
        border: round #58A6FF; 
    }
    #file-status { height: 1; color: #888; padding-left: 1; }
    
    .hidden { display: none; }
    .help-box {
        margin: 1 2;
        padding: 1 2;
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
    token = var(None, init=False)

    def __init__(self):
        super().__init__()
        self.http_client: Optional[httpx.AsyncClient] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="login-area"):
            with Vertical(classes="login-container"):
                yield Static("🔐 FourT AI LOGIN", classes="login-title")
                yield Static("Nhập token để bắt đầu", classes="login-instruction")
                with Horizontal(classes="token-input-row"):
                    yield Input(
                        placeholder="Dán Access Token...",
                        password=True,
                        id="token-input",
                    )
                    yield Button("Lấy token", id="get-token-button")
                yield Button(
                    "🚀 Đăng nhập", id="login-submit-button", classes="login-button"
                )
        yield ScrollableContainer(id="chat-history")
        with Vertical(id="input-area", classes="hidden"):
            yield Static("", id="file-status")
            yield Input(
                placeholder="Nhập tin nhắn hoặc /help để xem lệnh...", id="chat-input"
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
            except Exception as e:
                self.mount_info_log(f"[red]Lỗi khi đọc token: {e}[/red]")
                self.query_one("#token-input").focus()
        else:
            self.query_one("#token-input").focus()

    # Các phương thức còn lại giữ nguyên...
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "get-token-button":
            url_open = os.getenv(
                "API_URL", "https://living-tortoise-polite.ngrok-free.app"
            )
            webbrowser.open(url_open)
        elif event.button.id == "login-submit-button":
            # Lấy token từ input và xử lý đăng nhập
            token_input = self.query_one("#token-input")
            token = token_input.value.strip()
            if not token:
                return
            await self.process_token_login(token)

    async def process_token_login(self, token: str) -> None:
        """Xử lý đăng nhập với token từ nút submit."""
        try:
            # Tạo thư mục chứa TOKEN_FILE_PATH nếu chưa tồn tại
            token_dir = os.path.dirname(TOKEN_FILE_PATH)
            if token_dir:
                os.makedirs(token_dir, exist_ok=True)
            with open(TOKEN_FILE_PATH, "w") as f:
                f.write(token)
        except PermissionError as e:
            self.mount_info_log(
                f"[red]Lỗi quyền truy cập: Không thể ghi file token tại {TOKEN_FILE_PATH}. Vui lòng kiểm tra quyền thư mục.[/red]"
            )
            return
        except OSError as e:
            self.mount_info_log(
                f"[red]Lỗi khi lưu token tại {TOKEN_FILE_PATH}: {e}[/red]"
            )
            return
        except Exception as e:
            self.mount_info_log(f"[red]Lỗi không xác định khi lưu token: {e}[/red]")
            return
        await self.perform_login(token)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Xử lý sự kiện khi người dùng gửi input."""
        if event.input.id == "token-input":
            token = event.value.strip()
            if not token:
                self.mount_info_log("[red]Token không được để trống.[/red]")
                return
            await self.process_token_login(token)
        elif event.input.id == "chat-input":
            user_message = event.value.strip()
            event.input.value = ""
            if not user_message:
                return
            if user_message.startswith("/"):
                await self.handle_client_command(
                    user_message, self.query_one("#chat-history")
                )
            else:
                chat_history = self.query_one("#chat-history")
                chat_history.mount(Static(f">>> {user_message}"))
                chat_history.scroll_end()
                if self.http_client:
                    result = await send_chat_request(
                        self.http_client,
                        user_message,
                        self.current_conversation_id,
                        self.attached_file_path,
                        chat_history,
                    )
                    if result == "auth_error":
                        self.query_one("#chat-input").disabled = True
                        self.mount_info_log(
                            "[red]Lỗi xác thực. Vui lòng đăng nhập lại.[/red]"
                        )
                    elif result is not None:
                        self.current_conversation_id = result
                        chat_history.scroll_end()
                else:
                    self.mount_info_log(
                        "[yellow]Chưa kết nối API. Vui lòng kiểm tra backend.[/yellow]"
                    )
            self.attached_file_path = None

    # Các phương thức còn lại giữ nguyên hoàn toàn...
    async def perform_login(self, token: str, is_saved_token: bool = False) -> None:
        """Thực hiện đăng nhập với token."""
        self.http_client = httpx.AsyncClient(
            base_url=API_BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=300.0,
        )
        self.token = token
        self.query_one("#login-area").display = False
        self.query_one("#input-area").remove_class("hidden")
        chat_input = self.query_one("#chat-input")
        chat_input.disabled = False
        chat_input.focus()

        if is_saved_token:
            print("[AUTO LOGIN SUCCESFULL]")
        else:
            self.mount_info_log(
                "[green]Đăng nhập vào FourT AI thành công! Token đã được lưu cho lần sau.[/green]"
            )

        await self.handle_client_command("/help", self.query_one("#chat-history"))

    def watch_attached_file_path(self, new_path: Optional[str]) -> None:
        """Cập nhật trạng thái file đính kèm."""
        status_widget = self.query_one("#file-status", Static)
        if new_path:
            filename = os.path.basename(new_path)
            status_widget.update(
                f"📎 Đính kèm: [bold cyan]{filename}[/]. Gõ /clearfile để gỡ."
            )
        else:
            status_widget.update("")

    async def handle_client_command(
        self, command: str, chat_history: ScrollableContainer
    ) -> None:
        """Xử lý các lệnh client-side cho FourT AI."""
        parts = command.split(" ", 1)
        cmd = parts[0]
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            help_text = """[bold][#58A6FF]📚 HƯỚNG DẪN SỬ DỤNG FourT AI[/bold]

  [bold][#58A6FF]/new[/]: Bắt đầu một cuộc hội thoại mới 🆕
  [bold][#58A6FF]/history[/]: Xem danh sách các cuộc hội thoại đã có 📜
  [bold][#58A6FF]/load <id>[/]: Tải lại lịch sử của một cuộc hội thoại 📂
  [bold][#58A6FF]/file <path>[/]: Đính kèm một file vào tin nhắn tiếp theo 📎
  [bold][#58A6FF]/clearfile[/]: Gỡ file đã đính kèm 🗑️
  [bold][#58A6FF]/clear[/]: Xóa trắng màn hình chat hiện tại 🧹
  [bold][#58A6FF]/delete[/]: Xóa cuộc hội thoại hiện tại 🗑️
  [bold][#58A6FF]/delete_all[/]: Xóa cuộc tất cả hội thoại 🗑️
  [bold][#58A6FF]/logout[/]: Xóa token đã lưu và thoát 🚪
  """
            chat_history.mount(Static(help_text, classes="help-box"))
            chat_history.scroll_end()
        elif cmd == "/logout":
            if os.path.exists(TOKEN_FILE_PATH):
                try:
                    os.remove(TOKEN_FILE_PATH)
                    self.exit(
                        "Token đã được xóa. Vui lòng khởi động lại ứng dụng FourT AI."
                    )
                except Exception as e:
                    chat_history.mount(Static(f"[red]Lỗi khi xóa token: {e}[/red]"))
            else:
                self.exit(
                    "Không có token nào được lưu để xóa. Đang thoát khỏi FourT AI..."
                )
        elif cmd == "/new":
            await self.action_new_chat()
        elif cmd == "/history":
            if self.http_client:
                await fetch_conversations(self.http_client, chat_history)
            else:
                chat_history.mount(
                    Static(
                        "[yellow]Chưa kết nối API. Vui lòng kiểm tra backend.[/yellow]"
                    )
                )
        elif cmd == "/load":
            if args.isdigit():
                if self.http_client:
                    success = await load_conversation_history(
                        self.http_client, int(args), chat_history
                    )
                    if success:
                        self.current_conversation_id = int(args)
                    else:
                        self.current_conversation_id = None  # Đặt lại nếu tải thất bại
                else:
                    chat_history.mount(
                        Static(
                            "[yellow]Chưa kết nối API. Vui lòng kiểm tra backend.[/yellow]"
                        )
                    )
            else:
                chat_history.mount(
                    Static(f"[red]Lỗi: ID cuộc hội thoại không hợp lệ.[/red]")
                )
        elif cmd == "/file":
            if os.path.exists(args):
                self.attached_file_path = args
            else:
                chat_history.mount(
                    Static(f"[red]Lỗi: File không tồn tại: {args}[/red]")
                )
        elif cmd == "/clearfile":
            self.attached_file_path = None
        elif cmd == "/clear":
            chat_history.query("*").remove()
        elif cmd == "/delete":
            if self.http_client:
                await delete_current_conversation(
                    self.http_client, self.current_conversation_id, chat_history
                )
                await self.action_new_chat()
            else:
                chat_history.mount(
                    Static(
                        "[yellow]Chưa kết nối API. Vui lòng kiểm tra backend.[/yellow]"
                    )
                )
        elif cmd == "/delete_all":
            if self.http_client:
                await delete_all_conversation(self.http_client, chat_history)
                await self.action_new_chat()
            else:
                chat_history.mount(
                    Static(
                        "[yellow]Chưa kết nối API. Vui lòng kiểm tra backend.[/yellow]"
                    )
                )
        else:
            chat_history.mount(Static(f"[yellow]Lệnh không xác định: {cmd}.[/yellow]"))

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
        self.mount_info_log("Đã bắt đầu cuộc hội thoại mới với FourT AI.")
