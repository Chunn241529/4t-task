from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
import os
from datetime import datetime

from app.db import get_db
from app.models import Conversation as ModelConversation, User
from app.routers.task import get_current_user
from app.services.rag_service import RAGService
from app.services.file_service import FileService

router = APIRouter(prefix="/rag", tags=["rag"])
rag_service = RAGService()
file_service = FileService()

@router.post("/load-to-conversation")
async def load_rag_files_to_current_conversation(
    conversation_id: int,
    user_id: int = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Load tất cả file trong thư mục rag_files vào conversation hiện tại"""
    conversation = db.query(ModelConversation).filter(
        ModelConversation.id == conversation_id,
        ModelConversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise HTTPException(404, "Conversation not found")
    
    loaded_files = rag_service.load_rag_files_to_conversation(user_id, conversation_id)
    
    return {
        "message": f"Loaded {len(loaded_files)} RAG files into conversation",
        "loaded_files": loaded_files
    }

@router.get("/files")
async def list_rag_files():
    """Liệt kê tất cả file có trong thư mục RAG"""
    rag_files = []
    
    supported_extensions = ['.pdf', '.txt', '.docx', '.xlsx', '.xls', '.csv']
    
    for filename in os.listdir(rag_service.rag_files_dir):
        file_path = os.path.join(rag_service.rag_files_dir, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            file_stats = os.stat(file_path)
            rag_files.append({
                "filename": filename,
                "size": file_stats.st_size,
                "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
    
    return {"rag_files": rag_files}

@router.post("/analyze-file")
async def analyze_rag_file(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user)
):
    """Phân tích metadata của file RAG"""
    try:
        file_content = await file.read()
        filename = file.filename
        
        metadata = {}
        if filename.lower().endswith(('.xlsx', '.xls')):
            metadata = file_service.extract_excel_metadata(file_content)
        elif filename.lower().endswith('.docx'):
            metadata = file_service.extract_docx_metadata(file_content)
        
        return {
            "filename": filename,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(500, f"Error analyzing file: {str(e)}")
