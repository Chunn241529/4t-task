from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.schemas import TaskPrompt, TaskUpdate, Task as TaskSchema
from app.utils import decode_jwt
from app.services.task_service import (
    create_task_service,
    get_tasks_service,
    get_task_service,
    update_task_service,
    delete_task_service
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import logging
from typing import List

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/task", tags=["task"])
bearer_scheme = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        token = credentials.credentials
        logger.debug(f"Received token in get_current_user: {token}")
        payload = decode_jwt(token)
        logger.debug(f"Current user_id: {payload['sub']}")
        return int(payload["sub"])
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        raise HTTPException(401, f"Invalid token: {str(e)}")


@router.post("/task", response_model=TaskSchema)
def create_task(prompt: TaskPrompt, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    return create_task_service(prompt, user_id, db)


@router.get("/tasks", response_model=List[TaskSchema])
def get_tasks(user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_tasks_service(user_id, db)


@router.get("/tasks/{id}", response_model=TaskSchema)
def get_task(id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_task_service(id, user_id, db)


@router.put("/tasks/{id}", response_model=TaskSchema)
def update_task(id: int, task_update: TaskUpdate, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    return update_task_service(id, task_update, user_id, db)


@router.delete("/tasks/{id}")
def delete_task(id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    return delete_task_service(id, user_id, db)
