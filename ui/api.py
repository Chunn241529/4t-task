import httpx
import json
import base64
import os
from typing import Optional
from textual.widgets import Static, Markdown
from textual.containers import ScrollableContainer
from textual.reactive import reactive
import asyncio

from config import TOKEN_FILE_PATH


class AnimatedSpinner(Static):
    """A custom Static widget that animates a spinner using a sequence of characters."""

    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    current_index = reactive(0)

    def on_mount(self) -> None:
        """Start the animation when the widget is mounted."""
        # Tăng interval lên để giảm CPU usage
        self.set_interval(0.15, self.update_spinner)  # từ 0.1 lên 0.15

    def update_spinner(self) -> None:
        """Cycle through spinner characters."""
        self.current_index = (self.current_index + 1) % len(self.spinner_chars)
        self.update(self.spinner_chars[self.current_index])


# UI constants for spinner styles
THINKING_COLOR = "cyan"
TOOL_COLOR = "yellow"
RESPONSE_TOOL_COLOR = "green"
THINKING_PREFIX = "💭"
TOOL_PREFIX = "🔎"
THINKING_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
TOOL_SPINNER = ["◐", "◓", "◑", "◒"]


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
                filename = os.path.basename(attached_file_path)
                # Send file as an object so backend can know filename and detect images
                json_payload["file"] = {"content": encoded_file, "filename": filename}
        except Exception as e:
            chat_history.mount(Static(f"[red]Lỗi khi đọc file: {e}[/]"))
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
        is_using_tool = False
        pre_spinner_container = None
        
        # Biến để điều khiển tần suất cập nhật
        last_update_time = 0
        update_interval = 0.1  # Chỉ update mỗi 100ms
        last_scroll_time = 0
        scroll_interval = 0.3  # Chỉ scroll mỗi 300ms
        
        # Biến để lưu tool calls và search results
        current_tool_calls = []
        search_results_displayed = False
        search_notification_widget = None

        # show a simple, non-animated spinner immediately so the UI reacts before server sends anything
        

        # If no conversation_id provided, create a new conversation first and use its id
        if conversation_id is None:
            try:
                create_resp = await http_client.post("/conversations/")
                create_resp.raise_for_status()
                create_json = create_resp.json()
                # try to extract id from response
                new_id = (
                    create_json.get("id") if isinstance(create_json, dict) else None
                )
                if new_id is None:
                    # fallback: try common keys
                    new_id = (
                        create_json.get("conversation_id")
                        if isinstance(create_json, dict)
                        else None
                    )
                if new_id is None:
                    # couldn't determine id — show message and continue without id
                    chat_history.mount(
                        Static(
                            "[yellow]Tạo cuộc hội thoại mới nhưng không nhận được ID. Tiếp tục gửi mà không có conversation_id.[/]"
                        )
                    )
                else:
                    conversation_id = new_id
            except httpx.HTTPStatusError as e:
                # read body safely
                body = ""
                try:
                    body = (await e.response.aread()).decode("utf-8", errors="replace")
                except Exception:
                    body = str(e.response)

                if e.response.status_code == 401 and "Token has expired" in body:
                    error_message = "Token của bạn đã hết hạn. Vui lòng khởi động lại."
                    if os.path.exists(TOKEN_FILE_PATH):
                        try:
                            os.remove(TOKEN_FILE_PATH)
                        except Exception as e:
                            chat_history.mount(
                                Static(f"[red]Lỗi khi xóa token: {e}[/red]")
                            )

                chat_history.mount(Static(f"[red]{error_message}[/]"))
                if pre_spinner_container:
                    try:
                        pre_spinner_container.remove()
                    except Exception:
                        pass
                return None
            except Exception as e:
                chat_history.mount(Static(f"[red]Lỗi khi tạo cuộc hội thoại: {e}[/]"))
                if pre_spinner_container:
                    try:
                        pre_spinner_container.remove()
                    except Exception:
                        pass
                return None

        # attach conversation_id param if we have one
        params = {"conversation_id": conversation_id} if conversation_id else {}

        async with http_client.stream(
            "POST", "/send", params=params, json=json_payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                content = line[len("data:") :].strip()
                if not content:
                    continue
                try:
                    data_chunk = json.loads(content)
                except json.JSONDecodeError:
                    # non-json data — treat as raw content
                    data_chunk = {"content": content}

                print(f"DEBUG: Stream chunk: {data_chunk}")  # Log full chunk

                # Conversation id ack
                if "conversation_id" in data_chunk:
                    conversation_id = data_chunk["conversation_id"]
                    continue

                # Typing indicator from server
                if data_chunk.get("typing") is True:
                    # remove the pre-request spinner once the server signals typing
                    if pre_spinner_container:
                        try:
                            pre_spinner_container.remove()
                        except Exception:
                            pass
                        pre_spinner_container = None

                    if not initial_spinner_container:
                        initial_spinner = AnimatedSpinner("⠋", classes="spinner")
                        initial_spinner.spinner_chars = THINKING_SPINNER
                        initial_spinner.current_index = 0
                        initial_spinner.styles.width = 1
                        initial_spinner.styles.height = 1
                        initial_spinner.styles.color = THINKING_COLOR
                        initial_spinner_container = Static(
                            f"  [{THINKING_COLOR}]{THINKING_PREFIX} Nhi đang suy nghĩ...[/]"
                        )
                        initial_spinner_container.styles.display = "block"
                        initial_spinner_container.styles.padding = (0, 0, 0, 2)
                        chat_history.mount(initial_spinner_container)
                        initial_spinner_container.mount(initial_spinner)
                    continue

                if data_chunk.get("typing") is False:
                    if initial_spinner_container:
                        initial_spinner_container.remove()
                        initial_spinner_container = None
                    continue

                # Done / error / tool_calls / content handling
                if data_chunk.get("done"):
                    if ai_response_md and accumulated_content:
                        ai_response_md.update(accumulated_content)
                        chat_history.scroll_end()
                    await asyncio.sleep(0.1)
                    break

                if data_chunk.get("error"):
                    if initial_spinner_container:
                        initial_spinner_container.remove()
                    if response_spinner_container:
                        response_spinner_container.remove()
                    chat_history.mount(
                        Static(f"[bold red]Lỗi Stream: {data_chunk['error']}[/]")
                    )
                    break

                # Xử lý tool_calls - HIỂN THỊ SEARCH RESULTS
                if (
                    data_chunk.get("tool_calls")
                    and isinstance(data_chunk["tool_calls"], list)
                    and data_chunk["tool_calls"]
                ):
                    print(f"DEBUG: Tool calls detected: {data_chunk['tool_calls']}")
                    
                    # Lưu tool calls hiện tại
                    current_tool_calls = data_chunk["tool_calls"]
                    
                    # HIỂN THỊ THÔNG BÁO ĐANG SEARCH
                    if not search_results_displayed:
                        search_notification_widget = Static(
                            f"[{TOOL_COLOR}]{TOOL_PREFIX} Đang tìm kiếm thông tin...[/]"
                        )
                        search_notification_widget.styles.padding = (0, 0, 0, 2)
                        chat_history.mount(search_notification_widget)
                        chat_history.scroll_end()
                        search_results_displayed = True

                    if initial_spinner_container and not is_using_tool:
                        # change spinner visual to indicate a tool/search action
                        try:
                            initial_spinner.spinner_chars = TOOL_SPINNER
                            initial_spinner.current_index = 0
                            initial_spinner.styles.color = TOOL_COLOR
                        except Exception:
                            pass
                        initial_spinner_container.update(
                            f"  [{TOOL_COLOR}]{TOOL_PREFIX} Nhi đang tìm...[/]"
                        )
                        initial_spinner_container.refresh()
                        is_using_tool = True
                    
                    # Hiển thị search results nếu có trong tool calls
                    for tool_call in current_tool_calls:
                        if isinstance(tool_call, dict):
                            tool_type = tool_call.get("type", "")
                            tool_function = tool_call.get("function", {})
                            
                            # Hiển thị thông tin search
                            if tool_type == "web_search" or tool_function.get("name") == "web_search":
                                query = tool_function.get("arguments", {}).get("query", "")
                                if query:
                                    search_info = Static(
                                        f"[dim]{TOOL_PREFIX} Tìm kiếm: \"{query}\"[/dim]"
                                    )
                                    search_info.styles.padding = (0, 0, 0, 2)
                                    chat_history.mount(search_info)
                                    chat_history.scroll_end()
                            
                            # Hiển thị kết quả search nếu có
                            if "result" in tool_call or "content" in tool_call:
                                result = tool_call.get("result") or tool_call.get("content")
                                if result:
                                    result_display = Static(
                                        f"[dim]{TOOL_PREFIX} Kết quả tìm được: {result[:200]}...[/dim]"
                                    )
                                    result_display.styles.padding = (0, 0, 0, 2)
                                    chat_history.mount(result_display)
                                    chat_history.scroll_end()
                    
                    # Create a placeholder response area so any tool output/content is shown
                    if not ai_response_md:
                        # ensure typing indicator removed
                        if initial_spinner_container:
                            try:
                                initial_spinner_container.remove()
                            except Exception:
                                pass
                            initial_spinner_container = None
                        chat_history.mount(Static(""))
                        ai_response_md = Markdown("")
                        chat_history.mount(ai_response_md)
                        # show a response spinner to indicate tool work
                        response_spinner = AnimatedSpinner("⠋", classes="spinner")
                        response_spinner.spinner_chars = THINKING_SPINNER
                        response_spinner.current_index = 0
                        response_spinner.styles.width = 1
                        response_spinner.styles.height = 1
                        response_spinner.styles.color = RESPONSE_TOOL_COLOR
                        response_spinner_container = Static("")
                        response_spinner_container.styles.display = "block"
                        response_spinner_container.styles.padding = (0, 0, 0, 2)
                        chat_history.mount(response_spinner_container)
                        response_spinner_container.mount(response_spinner)
                    continue

                # Xử lý content - THÊM HIỂN THỊ SEARCH CONTEXT và GIẢM TẦN SUẤT UPDATE
                if data_chunk.get("content"):
                    decoded_content = (
                        data_chunk["content"].encode().decode("utf-8", errors="replace")
                    )
                    
                    # Nếu có search results trước đó, thêm context vào content
                    if current_tool_calls and not accumulated_content:
                        search_context = "\n\n_Dựa trên kết quả tìm kiếm..._\n\n"
                        decoded_content = search_context + decoded_content
                        current_tool_calls = []  # Reset sau khi đã sử dụng
                        
                        # Xóa thông báo search đang chờ
                        if search_notification_widget:
                            try:
                                search_notification_widget.remove()
                            except Exception:
                                pass
                    
                    accumulated_content += decoded_content
                    
                    if not ai_response_md:
                        # ensure typing indicator / pre-request spinner removed before showing content
                        if pre_spinner_container:
                            try:
                                pre_spinner_container.remove()
                            except Exception:
                                pass
                            pre_spinner_container = None
                        if initial_spinner_container:
                            try:
                                initial_spinner_container.remove()
                            except Exception:
                                pass
                            initial_spinner_container = None
                        chat_history.mount(Static(""))
                        ai_response_md = Markdown("")
                        chat_history.mount(ai_response_md)
                        response_spinner = AnimatedSpinner("⠋", classes="spinner")
                        # if we were using a tool, show a different color for the response spinner
                        response_spinner.spinner_chars = THINKING_SPINNER
                        response_spinner.current_index = 0
                        response_spinner.styles.width = 1
                        response_spinner.styles.height = 1
                        response_spinner.styles.color = (
                            RESPONSE_TOOL_COLOR if is_using_tool else "white"
                        )
                        response_spinner_container = Static("")
                        response_spinner_container.styles.display = "block"
                        response_spinner_container.styles.padding = (0, 0, 0, 2)
                        chat_history.mount(response_spinner_container)
                        response_spinner_container.mount(response_spinner)
                    
                    # CHỈ UPDATE KHI ĐỦ THỜI GIAN - giảm giật lag
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_update_time >= update_interval:
                        if ai_response_md:
                            ai_response_md.update(accumulated_content)
                            last_update_time = current_time
                            
                            # CHỈ SCROLL KHI ĐỦ THỜI GIAN
                            if current_time - last_scroll_time >= scroll_interval:
                                chat_history.scroll_end()
                                last_scroll_time = current_time
                    
                    await asyncio.sleep(0.05)

        # Final update sau khi kết thúc stream
        if ai_response_md and accumulated_content:
            ai_response_md.update(accumulated_content)
            chat_history.scroll_end()
            await asyncio.sleep(0.1)

        if response_spinner_container:
            response_spinner_container.remove()

        # make sure pre-request spinner is removed after finishing
        if pre_spinner_container:
            try:
                pre_spinner_container.remove()
            except Exception:
                pass

        return conversation_id

    except httpx.HTTPStatusError as e:
        if initial_spinner_container:
            initial_spinner_container.remove()
        if response_spinner_container:
            response_spinner_container.remove()
        # Safely read streaming response body (if any) to avoid ResponseNotRead
        body_text = ""
        try:
            # try to asynchronously read the response body if it's available
            body_bytes = await e.response.aread()
            body_text = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            # fallback: try to get a repr or leave empty
            try:
                body_text = str(e.response)
            except Exception:
                body_text = "<không thể đọc body>"

        chat_history.mount(
            Static(f"[bold red]Lỗi API {e.response.status_code}: {body_text}[/]")
        )
        if e.response.status_code in (401, 403):
            return "auth_error"
        return None
    except httpx.ConnectError:
        if initial_spinner_container:
            initial_spinner_container.remove()
        if response_spinner_container:
            response_spinner_container.remove()
        chat_history.mount(
            Static(f"[bold red]Lỗi kết nối tới {http_client.base_url}.[/]")
        )
        return None


async def fetch_conversations(
    http_client: httpx.AsyncClient, chat_history: ScrollableContainer
) -> None:
    """Lấy danh sách các cuộc hội thoại từ API."""
    try:
        response = await http_client.get("/conversations/")
        response.raise_for_status()
        conversations = response.json()
        if not conversations:
            chat_history.mount(Static("Chưa có cuộc hội thoại nào."))
            return
        history_text = "[bold]Danh sách cuộc hội thoại:[/]\n" + "\n".join(
            [
                f"- ID: {conv['id']} (Tạo lúc: {conv['created_at']})"
                for conv in conversations
            ]
        )
        chat_history.mount(Static(history_text))
        chat_history.scroll_end()
    except Exception as e:
        chat_history.mount(Static(f"[red]Lỗi khi tải lịch sử: {e}[/]"))


async def load_conversation_history(
    http_client: httpx.AsyncClient, conv_id: int, chat_history: ScrollableContainer
) -> bool:  # Thêm kiểu trả về bool để báo hiệu thành công/thất bại
    """Tải lịch sử cuộc hội thoại từ API."""
    chat_history.query("*").remove()
    chat_history.mount(
        Static(f"Đang tải lịch sử cho cuộc hội thoại [bold cyan]#{conv_id}[/]...")
    )
    try:
        response = await http_client.get(f"/messages/conversations/{conv_id}/messages")
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
                Static(f"[red]Conversation #{conv_id} không tồn tại.[/]")
            )
        else:
            chat_history.mount(
                Static(
                    f"[red]Lỗi khi tải lịch sử: {e.response.status_code} - {e.response.text}[/]"
                )
            )
        return False  # Tải thất bại
    except Exception as e:
        chat_history.mount(Static(f"[red]Lỗi khi tải lịch sử: {e}[/]"))
        return False  # Tải thất bại


async def delete_current_conversation(
    http_client: httpx.AsyncClient,
    conversation_id: Optional[int],
    chat_history: ScrollableContainer,
) -> Optional[int]:
    """Xóa cuộc hội thoại hiện tại đang được tải."""
    if conversation_id is None:
        chat_history.mount(
            Static("[yellow]Bạn đang ở ngoài cuộc trò chuyện, không thể xóa.[/]")
        )
        return None

    try:
        response = await http_client.delete(f"/conversations/{conversation_id}")
        response.raise_for_status()
        chat_history.query("*").remove()
        chat_history.scroll_end()
        return None

    except httpx.HTTPStatusError as e:
        chat_history.mount(
            Static(
                f"[bold red]Lỗi khi xóa cuộc hội thoại: {e.response.status_code} - {e.response.text}[/]"
            )
        )
        chat_history.scroll_end()
        if e.response.status_code in (401, 403):
            return "auth_error"
        return conversation_id
    except httpx.ConnectError:
        chat_history.mount(
            Static(f"[bold red]Lỗi kết nối tới {http_client.base_url}.[/]")
        )
        chat_history.scroll_end()
        return conversation_id


async def delete_all_conversation(
    http_client: httpx.AsyncClient,
    chat_history: ScrollableContainer,
) -> Optional[int]:
    """Xóa cuộc hội thoại hiện tại đang được tải."""
    try:
        response = await http_client.delete(f"/conversations/")
        response.raise_for_status()
        chat_history.query("*").remove()
        chat_history.scroll_end()
        return None

    except httpx.HTTPStatusError as e:
        chat_history.mount(
            Static(
                f"[bold red]Lỗi khi xóa cuộc hội thoại: {e.response.status_code} - {e.response.text}[/]"
            )
        )
        chat_history.scroll_end()
        if e.response.status_code in (401, 403):
            return "auth_error"

    except httpx.ConnectError:
        chat_history.mount(
            Static(f"[bold red]Lỗi kết nối tới {http_client.base_url}.[/]")
        )
        chat_history.scroll_end()
