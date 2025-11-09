from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.db import Base, engine
from app.models import *  # Import tất cả model để đăng ký với Base
from app.routers.task import router as task_router
from app.routers.auth import router as auth_router
from app.routers.chat import router as chat_router
from app.routers.conversations import router as conversations_router
from app.routers.messages import router as messages_router
from app.routers.rag import router as rag_router
from app.utils import verify_jwt
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

logger = logging.getLogger(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)

# Tạo tất cả bảng trong database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="4T AI", version="1.0.0", description="4T AI - Your Personal Task and Chat Assistant",openapi_tags=[
    {"name": "auth", "description": "Authentication and User Management"},
    {"name": "task", "description": "Task Management"},
    {"name": "chat", "description": "Chat with AI"},
    {"name": "conversations", "description": "Conversation History Management"},
    {"name": "messages", "description": "Message Handling"},
    {"name": "rag", "description": "Retrieval-Augmented Generation"}])

# Cấu hình Jinja2Templates
templates = Jinja2Templates(directory="ui/web/pages")

# Phục vụ file tĩnh (script.js, style.css)
app.mount("/static", StaticFiles(directory="ui/web/static"), name="static")

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://living-tortoise-polite.ngrok-free.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Access-Control-Allow-Origin", "Access-Control-Allow-Methods", "Access-Control-Allow-Headers"],
)

# Route để render login.html tại '/'
@app.get("/", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/forgetpw", response_class=HTMLResponse)
async def get_forgetpw(request: Request):
    return templates.TemplateResponse("forget-password.html", {"request": request})

@app.get("/reset-password", response_class=HTMLResponse)
async def get_reset_password(request: Request):
    return templates.TemplateResponse("reset-password.html", {"request": request})
  


app.include_router(auth_router)
app.include_router(task_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(messages_router)
app.include_router(rag_router)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
