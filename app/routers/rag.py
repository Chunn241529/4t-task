from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
import os
from datetime import datetime
from typing import List, Dict, Any
import logging

from app.db import get_db
from app.models import Conversation as ModelConversation
from app.routers.task import get_current_user
from app.services.rag_service import RAGService
from app.services.file_service import FileService

router = APIRouter(prefix="/rag", tags=["rag"])
logger = logging.getLogger(__name__)

@router.post("/conversation/{conversation_id}/load-files")
async def load_rag_files_to_current_conversation(
    conversation_id: int,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Load tất cả file trong thư mục rag_files vào conversation hiện tại"""
    try:
        # Verify conversation exists and belongs to user
        conversation = db.query(ModelConversation).filter(
            ModelConversation.id == conversation_id,
            ModelConversation.user_id == user_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Run in background to avoid timeout for large files
        def process_rag_files():
            try:
                loaded_files = RAGService.load_rag_files_to_conversation(user_id, conversation_id)
                logger.info(f"Loaded {len(loaded_files)} RAG files for user {user_id}, conversation {conversation_id}")
                return loaded_files
            except Exception as e:
                logger.error(f"Error processing RAG files: {e}")
                return []
        
        background_tasks.add_task(process_rag_files)
        
        return {
            "message": "Started loading RAG files into conversation. Processing in background.",
            "conversation_id": conversation_id,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_rag_files_to_conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/files")
async def list_rag_files():
    """Liệt kê tất cả file có trong thư mục RAG"""
    try:
        rag_files = []
        rag_dir = RAGService.rag_files_dir
        
        if not os.path.exists(rag_dir):
            return {"rag_files": []}
        
        supported_extensions = ['.pdf', '.txt', '.docx', '.xlsx', '.xls', '.csv']
        
        for filename in os.listdir(rag_dir):
            file_path = os.path.join(rag_dir, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
                try:
                    file_stats = os.stat(file_path)
                    rag_files.append({
                        "filename": filename,
                        "size": file_stats.st_size,
                        "size_readable": f"{file_stats.st_size / 1024:.1f} KB",
                        "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        "extension": os.path.splitext(filename)[1].lower()
                    })
                except Exception as e:
                    logger.warning(f"Could not get stats for file {filename}: {e}")
                    continue
        
        # Sort by modified time (newest first)
        rag_files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        return {
            "rag_files": rag_files,
            "total_files": len(rag_files),
            "directory": rag_dir
        }
        
    except Exception as e:
        logger.error(f"Error listing RAG files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing RAG files: {str(e)}")

@router.post("/analyze-file")
async def analyze_rag_file(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user)
):
    """Phân tích metadata của file RAG"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_content = await file.read()
        filename = file.filename
        
        # Validate file size (max 50MB)
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large. Maximum 50MB allowed.")
        
        metadata = {}
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension in ['xlsx', 'xls']:
            metadata = FileService.extract_excel_metadata(file_content)
        elif file_extension == 'docx':
            metadata = FileService.extract_docx_metadata(file_content)
        elif file_extension == 'pdf':
            # Extract basic PDF info
            text_content = FileService.extract_text_from_file(file_content)
            metadata = {
                "file_type": "pdf",
                "text_length": len(text_content),
                "has_content": len(text_content.strip()) > 0
            }
        else:
            # For other file types, extract basic info
            text_content = FileService.extract_text_from_file(file_content)
            metadata = {
                "file_type": file_extension,
                "text_length": len(text_content),
                "has_content": len(text_content.strip()) > 0
            }
        
        return {
            "filename": filename,
            "file_size": len(file_content),
            "file_type": file_extension,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing file {file.filename if file else 'unknown'}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@router.post("/upload-file")
async def upload_rag_file(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user)
):
    """Upload file vào thư mục RAG"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Validate file type
        supported_extensions = ['.pdf', '.txt', '.docx', '.xlsx', '.xls', '.csv']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Supported types: {', '.join(supported_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (max 50MB)
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large. Maximum 50MB allowed.")
        
        # Save file to RAG directory
        rag_dir = RAGService.rag_files_dir
        file_path = os.path.join(rag_dir, file.filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File with same name already exists")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Uploaded RAG file: {file.filename} for user {user_id}")
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_path": file_path,
            "size": len(file_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading RAG file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.delete("/file/{filename}")
async def delete_rag_file(
    filename: str,
    user_id: int = Depends(get_current_user)
):
    """Xóa file từ thư mục RAG"""
    try:
        # Security: prevent path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join(RAGService.rag_files_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        logger.info(f"Deleted RAG file: {filename} by user {user_id}")
        
        return {
            "message": "File deleted successfully",
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting RAG file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@router.get("/conversation/{conversation_id}/index-stats")
async def get_rag_index_stats(
    conversation_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lấy thống kê về FAISS index của conversation"""
    try:
        # Verify conversation exists and belongs to user
        conversation = db.query(ModelConversation).filter(
            ModelConversation.id == conversation_id,
            ModelConversation.user_id == user_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Sửa: Thay vì gọi RAGService.get_index_stats, chúng ta sẽ tự lấy stats
        index, exists = RAGService.load_faiss(user_id, conversation_id)
        
        stats = {
            "exists": exists,
            "vector_count": index.ntotal,
            "dimension": index.d,
            "index_type": "FlatIP (Cosine Similarity)"
        }
        
        return {
            "conversation_id": conversation_id,
            "index_stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RAG index stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting index stats: {str(e)}")

@router.delete("/conversation/{conversation_id}/index")
async def cleanup_rag_index(
    conversation_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dọn dẹp FAISS index của conversation"""
    try:
        # Verify conversation exists and belongs to user
        conversation = db.query(ModelConversation).filter(
            ModelConversation.id == conversation_id,
            ModelConversation.user_id == user_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        RAGService.cleanup_faiss_index(user_id, conversation_id)
        
        return {
            "message": "RAG index cleaned up successfully",
            "conversation_id": conversation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up RAG index: {e}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up index: {str(e)}")
