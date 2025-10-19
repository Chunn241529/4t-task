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
    allow_origins=["http://localhost:8000", "http://localhost:3001", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route để render login.html tại '/'
@app.get("/", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

app.include_router(auth.router, prefix="/auth")
app.include_router(task.router)
app.include_router(chat.router)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
