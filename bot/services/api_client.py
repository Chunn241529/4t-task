import httpx
import json
import base64
import logging
from typing import Optional, AsyncGenerator
from config.settings import settings

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self):
        self.base_url = settings.FASTAPI_BACKEND_URL
        self.timeout = 60.0  # 60 seconds timeout for streaming
    
    async def create_conversation(self, auth_token: str) -> str:
        """Create a new conversation on the backend."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/conversations",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    json={}
                )
                response.raise_for_status()
                data = response.json()
                return data.get('id')
            except Exception as e:
                logger.error(f"Error creating conversation: {e}")
                raise
    
    async def chat_stream(
        self, 
        auth_token: str, 
        conversation_id: str, 
        message: str, 
        file_data: Optional[bytes] = None,
        file_name: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Stream chat response from backend - similar to TUI client."""
        json_payload = {"message": {"message": message}}
        
        if file_data:
            json_payload["file"] = base64.b64encode(file_data).decode("utf-8")
        
        params = {"conversation_id": conversation_id}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    params=params,
                    json=json_payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            content = line[len("data:"):].strip()
                            if not content:
                                continue
                            
                            try:
                                data_chunk = json.loads(content)
                                yield data_chunk
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error on chunk: {e}")
                                # Vẫn yield raw content như client TUI
                                yield {"content": content}
                                
            except httpx.ReadTimeout:
                logger.warning("Streaming timed out")
                yield {"error": "Request timeout"}
            except Exception as e:
                logger.error(f"Error in chat stream: {e}")
                yield {"error": str(e)}
    
    async def delete_conversation(self, auth_token: str, conversation_id: str) -> bool:
        """Delete a conversation from backend."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.delete(
                    f"{self.base_url}/conversations/{conversation_id}",
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
                response.raise_for_status()
                return True
            except Exception as e:
                logger.error(f"Error deleting conversation: {e}")
                return False

# Global API client instance
api_client = APIClient()
