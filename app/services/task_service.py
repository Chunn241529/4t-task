from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models import Task as TaskModel
from app.schemas import TaskPrompt, TaskUpdate
import ollama
import json
import logging
from datetime import datetime, timedelta
from typing import List

logger = logging.getLogger(__name__)


def create_task_service(prompt: TaskPrompt, user_id: int, db: Session):
    current_time = datetime.now()
    tomorrow = current_time.date() + timedelta(days=1)
    ollama_prompt = f"""
        Bạn là một trợ lý ảo chuyên phân tích và bóc tách thông tin công việc từ văn bản người dùng.
        Nhiệm vụ của bạn là đọc kỹ yêu cầu và trả về một đối tượng JSON duy nhất, không giải thích gì thêm.
        Các trường thông tin cần bóc tách:
        - "task_name": Tên chính của công việc (string).
        - "due_date": Thời gian cần hoàn thành. Nếu có, chuyển đổi về định dạng ISO 8601 (YYYY-MM-DDTHH:MM:SS). Nếu không có, dùng giá trị null.
        - "priority": Mức độ ưu tiên. Chỉ được dùng một trong ba giá trị: "low", "medium", "high".
        - "tags": Một chuỗi chứa các từ khóa quan trọng, cách nhau bởi dấu phẩy (string).
        - "original_query": Giữ nguyên câu lệnh gốc của người dùng.

        Hôm nay là: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
        Ví dụ về ngày: 'ngày mai' có nghĩa là '{tomorrow.strftime('%Y-%m-%d')}'.

        Yêu cầu của người dùng: "{prompt.prompt}"

        Trả về JSON thuần túy, không markdown, không text thừa.
    """
    try:
        response = ollama.chat(
            model="qwen3:4b-instruct-2507-q8_0",
            messages=[{"role": "user", "content": ollama_prompt}],
            options={"temperature": 0.0, "top_p": 0.9},
            format="json"
        )

        llm_output = response["message"]["content"].strip()
        logger.debug(f"Raw LLM output: {llm_output}")

        # Xử lý nếu LLM trả về dạng code block
        if llm_output.startswith("```json") and llm_output.endswith("```"):
            llm_output = llm_output[7:-3].strip()

        task_data = json.loads(llm_output)

        required_keys = {"task_name", "due_date", "priority", "tags", "original_query"}
        if not isinstance(task_data, dict) or not required_keys.issubset(task_data.keys()):
            raise ValueError("Invalid JSON format: missing required keys")

        if task_data["priority"] not in ["low", "medium", "high"]:
            raise ValueError("priority must be 'low', 'medium', or 'high'")

        logger.debug(f"Parsed task JSON: {task_data}")

        # Lưu vào DB
        task = TaskModel(
            user_id=user_id,
            task_name=task_data["task_name"],
            due_date=task_data["due_date"],
            priority=task_data["priority"],
            tags=task_data["tags"],
            original_query=task_data["original_query"],
        )

        db.add(task)
        db.commit()
        db.refresh(task)

        logger.debug(f"Created TaskModel: id={task.id}, created_at={task.created_at}")
        return task

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Invalid LLM output: {str(e)}, raw output: {response}")
        raise HTTPException(400, f"Invalid LLM output: {str(e)}")


def get_tasks_service(user_id: int, db: Session) -> List[TaskModel]:
    tasks = db.query(TaskModel).filter(TaskModel.user_id == user_id).all()
    logger.debug(f"Retrieved {len(tasks)} tasks for user_id {user_id}")
    return tasks


def get_task_service(id: int, user_id: int, db: Session) -> TaskModel:
    task = db.query(TaskModel).filter(TaskModel.id == id, TaskModel.user_id == user_id).first()
    if not task:
        logger.error(f"Task not found or unauthorized: id={id}, user_id={user_id}")
        raise HTTPException(404, "Task not found or not authorized")
    return task


def update_task_service(id: int, task_update: TaskUpdate, user_id: int, db: Session) -> TaskModel:
    task = db.query(TaskModel).filter(TaskModel.id == id, TaskModel.user_id == user_id).first()
    if not task:
        logger.error(f"Task not found or unauthorized for update: id={id}, user_id={user_id}")
        raise HTTPException(404, "Task not found or not authorized")

    for field, value in task_update.dict(exclude_unset=True).items():
        setattr(task, field, value)

    db.commit()
    db.refresh(task)
    logger.debug(f"Updated task id={id} for user_id={user_id}")
    return task


def delete_task_service(id: int, user_id: int, db: Session) -> dict:
    task = db.query(TaskModel).filter(TaskModel.id == id, TaskModel.user_id == user_id).first()
    if not task:
        logger.error(f"Task not found or unauthorized for delete: id={id}, user_id={user_id}")
        raise HTTPException(404, "Task not found or not authorized")

    db.delete(task)
    db.commit()
    logger.debug(f"Deleted task id={id} for user_id={user_id}")
    return {"message": "Task deleted"}
