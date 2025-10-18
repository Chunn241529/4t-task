from fastapi import FastAPI
from app.db import Base, engine
from app.models import *  # Import tất cả model để đăng ký với Base
from app.routers import auth, task, chat
import uvicorn

# Tạo tất cả bảng trong database
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(auth.router, prefix="/auth")
app.include_router(task.router)
app.include_router(chat.router)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
