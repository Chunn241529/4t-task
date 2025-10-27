import os
import faiss
import numpy as np
from typing import List, Tuple, Any
import logging
import glob

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, rag_files_dir: str = "rag_files", faiss_indices_dir: str = "faiss_indices", dim: int = 768):
        self.rag_files_dir = rag_files_dir
        self.faiss_indices_dir = faiss_indices_dir
        self.dim = dim
        
        # Tạo thư mục nếu chưa tồn tại (tuyệt đối path)
        os.makedirs(rag_files_dir, exist_ok=True)
        os.makedirs(faiss_indices_dir, exist_ok=True)
        logger.info(f"RAG service initialized. FAISS indices dir: {os.path.abspath(faiss_indices_dir)}")

    def get_faiss_path(self, user_id: int, conversation_id: int) -> str:
        """Tạo đường dẫn cho FAISS index trong thư mục faiss_indices"""
        filename = f"faiss_{user_id}_{conversation_id}.index"
        path = os.path.join(self.faiss_indices_dir, filename)
        logger.debug(f"FAISS path for user {user_id}, conv {conversation_id}: {path}")
        return path

    def cleanup_faiss_index(self, user_id: int, conversation_id: int):
        """Dọn dẹp FAISS index nếu tồn tại trong faiss_indices"""
        path = self.get_faiss_path(user_id, conversation_id)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"Đã xóa FAISS index: {path}")
            except Exception as e:
                logger.error(f"Lỗi khi xóa FAISS index {path}: {e}")
        else:
            logger.debug(f"FAISS index không tồn tại: {path}")

    def load_faiss(self, user_id: int, conversation_id: int) -> Tuple[Any, bool]:
        """Tải hoặc tạo mới FAISS index từ faiss_indices"""
        path = self.get_faiss_path(user_id, conversation_id)
        logger.debug(f"Loading FAISS from: {path}")
        
        if os.path.exists(path):
            try:
                index = faiss.read_index(path)
                logger.info(f"Đã tải FAISS index từ: {path}")
                return index, True
            except Exception as e:
                logger.error(f"Error loading FAISS index {path}, creating new: {e}")
        
        # Tạo index mới
        index = faiss.IndexFlatL2(self.dim)
        logger.info(f"Tạo FAISS index mới cho user {user_id}, conversation {conversation_id}")
        return index, False

    def save_faiss_index(self, index: Any, user_id: int, conversation_id: int):
        """Lưu FAISS index vào thư mục faiss_indices"""
        try:
            path = self.get_faiss_path(user_id, conversation_id)
            faiss.write_index(index, path)
            logger.info(f"Đã lưu FAISS index: {path}")
            logger.info(f"FAISS index saved at: {os.path.abspath(path)}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu FAISS index: {e}")

    def get_user_faiss_files(self, user_id: int) -> List[str]:
        """Lấy danh sách tất cả FAISS files của user"""
        pattern = os.path.join(self.faiss_indices_dir, f"faiss_{user_id}_*.index")
        files = glob.glob(pattern)
        logger.debug(f"Found {len(files)} FAISS files for user {user_id}")
        return files

    def cleanup_all_user_faiss(self, user_id: int):
        """Xóa tất cả FAISS indices của user"""
        try:
            faiss_files = self.get_user_faiss_files(user_id)
            deleted_count = 0
            
            for file_path in faiss_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"Đã xóa FAISS index: {file_path}")
                except Exception as e:
                    logger.error(f"Lỗi khi xóa FAISS index {file_path}: {e}")
            
            logger.info(f"Đã xóa {deleted_count} FAISS indices cho user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Lỗi khi xóa tất cả FAISS indices của user {user_id}: {e}")
            return 0

    def list_all_faiss_files(self):
        """Liệt kê tất cả FAISS files (for debugging)"""
        pattern = os.path.join(self.faiss_indices_dir, "*.index")
        files = glob.glob(pattern)
        logger.info(f"Tất cả FAISS files trong {self.faiss_indices_dir}:")
        for file in files:
            logger.info(f"  - {file}")
        return files
