from pydantic import BaseModel
from typing import Optional, List, Union
from datetime import datetime


class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    gender: Optional[str]  # Thêm trường gender, có thể là "male", "female", hoặc None


class UserLogin(BaseModel):
    username_or_email: str
    password: str
    device_id: Optional[str] = None


class VerifyCode(BaseModel):
    code: str


class ResetPassword(BaseModel):
    reset_token: str
    new_password: str


class TaskPrompt(BaseModel):
    prompt: str


class Task(BaseModel):
    id: int
    user_id: int
    task_name: str
    due_date: Optional[str]
    priority: str
    tags: str
    original_query: str
    created_at: datetime

    class Config:
        from_attributes = True


class TaskUpdate(BaseModel):
    task_name: Optional[str]
    due_date: Optional[str]
    priority: Optional[str]
    tags: Optional[str]
    original_query: Optional[str]


class ConversationCreate(BaseModel):
    pass


class Conversation(BaseModel):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationUpdate(BaseModel):
    pass


class ChatMessageIn(BaseModel):
    message: str


class ChatMessage(BaseModel):
    id: int
    user_id: int
    conversation_id: int
    content: str
    role: str
    timestamp: datetime
    embedding: Optional[Union[list, dict]]

    class Config:
        from_attributes = True


class ChatMessageUpdate(BaseModel):
    content: Optional[str]
