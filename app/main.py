from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from app.db import Base, engine
from app.models import *  # Import tất cả model để đăng ký với Base
from app.routers import auth, task, chat
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)

# Tạo tất cả bảng trong database
Base.metadata.create_all(bind=engine)

app = FastAPI()

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

# Middleware để log yêu cầu và thêm header CORS
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     logger.debug(f"Received request: {request.method} {request.url}, headers: {request.headers}")
#     response = await call_next(request)
#     logger.debug(f"Response status: {response.status_code} for {request.method} {request.url}, response headers: {response.headers}")
#     # Thêm header CORS vào mọi phản hồi
#     response.headers["Access-Control-Allow-Origin"] = "https://living-tortoise-polite.ngrok-free.app"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#     response.headers["Access-Control-Allow-Credentials"] = "true"
#     return response

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

app.include_router(auth.router, prefix="/auth")
app.include_router(task.router)
app.include_router(chat.router)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
