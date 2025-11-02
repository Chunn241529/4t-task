import os
import faiss
import numpy as np
from typing import List, Tuple, Any
import logging
import glob
import json

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
      
    def validate_and_normalize_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        """Validate và chuẩn hóa vectors trước khi thêm vào FAISS"""
        if not vectors:
            raise ValueError("Danh sách vectors không được rỗng")
        
        logger.info(f"Validating {len(vectors)} vectors, target dimension: {self.dim}")
        
        # Kiểm tra và chuẩn hóa từng vector
        normalized_vectors = []
        for i, vec in enumerate(vectors):
            if not isinstance(vec, (list, np.ndarray)):
                logger.warning(f"Vector {i} không phải list/array: {type(vec)}")
                continue
                
            # Chuyển đổi sang list nếu là numpy array
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            
            # Kiểm tra kích thước
            if len(vec) != self.dim:
                logger.warning(f"Vector {i} có kích thước {len(vec)}, expected {self.dim}. Đang chuẩn hóa...")
                vec = self._normalize_vector_size(vec, self.dim)
            
            normalized_vectors.append(vec)
        
        if not normalized_vectors:
            raise ValueError("Không có vector hợp lệ sau khi chuẩn hóa")
        
        # Chuyển đổi sang numpy array
        try:
            vectors_array = np.array(normalized_vectors, dtype=np.float32)
            logger.info(f"Đã tạo numpy array với shape: {vectors_array.shape}")
            return vectors_array
        except Exception as e:
            logger.error(f"Lỗi chuyển đổi sang numpy array: {e}")
            raise

    def _normalize_vector_size(self, vector: List[float], target_dim: int) -> List[float]:
        """Chuẩn hóa kích thước vector về target_dim"""
        current_len = len(vector)
        
        if current_len < target_dim:
            # Padding với zeros
            padding = [0.0] * (target_dim - current_len)
            return vector + padding
        else:
            # Cắt bớt
            return vector[:target_dim]

    def safe_add_to_faiss(self, index: Any, vectors: List[List[float]], user_id: int, conversation_id: int):
        """Thêm vectors vào FAISS index một cách an toàn"""
        try:
            if not vectors:
                logger.warning("Không có vectors để thêm vào FAISS")
                return index
            
            # Chuẩn hóa vectors
            normalized_vectors = self.validate_and_normalize_vectors(vectors)
            
            # Thêm vào index
            index.add(normalized_vectors)
            
            # Lưu index
            self.save_faiss_index(index, user_id, conversation_id)
            
            logger.info(f"Đã thêm {len(vectors)} vectors vào FAISS index. Tổng: {index.ntotal}")
            return index
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm vectors vào FAISS: {e}")
            raise

    def search_similar_context(self, query_vector: List[float], user_id: int, conversation_id: int,
                               messages: List[Any], top_k: int = 5) -> str:
        """Search FAISS for messages similar to query_vector and return joined message contents.

        - If FAISS index doesn't match the provided messages (by count), try to (re)build it from message embeddings.
        - messages is expected to be a list of objects with an `embedding` attribute (JSON string) and `content`.
        """
        try:
            # Load or create index
            index, _ = self.load_faiss(user_id, conversation_id)

            # If index is out-of-sync with provided messages, attempt to rebuild/add
            valid_embs = []
            for m in messages:
                if getattr(m, 'embedding', None):
                    try:
                        emb = json.loads(m.embedding)
                        valid_embs.append(emb)
                    except Exception:
                        continue

            if not valid_embs:
                return ""

            if index.ntotal != len(valid_embs):
                # Rebuild by adding normalized embeddings
                try:
                    index = self.safe_add_to_faiss(index, valid_embs, user_id, conversation_id)
                except Exception as e:
                    logger.error(f"Failed to sync FAISS with message embeddings: {e}")

            # Prepare query vector
            qv = np.array(query_vector, dtype=np.float32)
            if qv.ndim == 1:
                qv = qv.reshape(1, -1)

            if index.ntotal == 0:
                return ""

            k = min(top_k, index.ntotal)
            D, I = index.search(qv, k)

            contents = []
            for idx in I[0]:
                if idx is None or idx < 0:
                    continue
                if idx < len(messages):
                    contents.append(messages[idx].content)

            return "\n".join(contents)

        except Exception as e:
            logger.error(f"Error searching similar context: {e}")
            return ""
