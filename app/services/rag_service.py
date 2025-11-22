import faiss
import numpy as np
import json
import os
import re
import glob
from typing import List, Dict, Any, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.services.embedding_service import EmbeddingService
from app.services.file_service import FileService
from app.models import ChatMessage as ModelChatMessage

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)

RAG_FILES_DIR = "rag_files"
os.makedirs(RAG_FILES_DIR, exist_ok=True)


class RAGService:
    rag_files_dir = RAG_FILES_DIR

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
        """Chunk text thành các đoạn nhỏ với overlap - improved version"""
        if len(text) <= chunk_size:
            return [text]

        text = re.sub(r"\n+", "\n", text.strip())
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            if end < text_length:
                # Tìm điểm cắt tốt hơn
                break_points = [
                    text.rfind("\n\n", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end),
                ]

                best_break = -1
                for bp in break_points:
                    if bp != -1 and bp > start + (chunk_size // 3):
                        best_break = bp
                        break

                if best_break != -1:
                    end = best_break + 1

            chunk = text[start:end].strip()
            if chunk and len(chunk) >= 25:
                chunks.append(chunk)

            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

    @staticmethod
    def process_file_for_rag(
        file_content: bytes, user_id: int, conversation_id: int, filename: str = ""
    ) -> str:
        """Xử lý file để tạo RAG context - với debug chi tiết"""
        try:
            logger.info(f"Starting to process file for RAG: {filename}")

            file_text = FileService.extract_text_from_file(file_content)
            if not file_text or not file_text.strip():
                logger.warning(f"No text extracted from file {filename}")
                return ""

            logger.info(f"Extracted {len(file_text)} characters from {filename}")

            # Chunk text
            chunks = RAGService.chunk_text(file_text, chunk_size=600, overlap=80)
            if not chunks:
                logger.warning(f"No chunks created from file {filename}")
                return ""

            logger.info(f"Created {len(chunks)} chunks from {filename}")

            # Load FAISS index
            index, exists = RAGService.load_faiss(user_id, conversation_id)

            embeddings = []
            valid_chunks = []

            # Tạo embeddings cho từng chunk
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 25:
                    continue

                chunk_with_info = (
                    f"[File: {filename}] [Chunk {i+1}/{len(chunks)}] {chunk}"
                )
                emb = EmbeddingService.get_embedding(chunk_with_info, max_length=512)

                if np.any(emb) and not np.all(emb == 0):
                    embeddings.append(emb)
                    valid_chunks.append(chunk_with_info)
                    logger.debug(f"Created embedding for chunk {i+1}")
                else:
                    logger.warning(f"Failed to create embedding for chunk {i+1}")

            logger.info(
                f"Created {len(embeddings)} valid embeddings from {len(chunks)} chunks"
            )

            if embeddings:
                emb_array = np.array(embeddings).astype("float32")
                faiss.normalize_L2(emb_array)

                # Kiểm tra index trước khi thêm
                initial_count = index.ntotal
                index.add(emb_array)
                new_count = index.ntotal

                logger.info(f"Added {new_count - initial_count} vectors to FAISS index")

                # Lưu index
                faiss_path = RAGService.get_faiss_path(user_id, conversation_id)
                faiss.write_index(index, faiss_path)

                result_context = f"[File: {filename}] [Loaded: {len(valid_chunks)} chunks, {sum(len(chunk) for chunk in valid_chunks)} characters]"
                logger.info(f"Successfully processed file {filename}: {result_context}")
                return result_context
            else:
                logger.warning(f"No valid embeddings created from file {filename}")
                return ""

        except Exception as e:
            logger.error(f"Error processing file {filename} for RAG: {e}")
            return ""

    @staticmethod
    def _ensure_rag_loaded(user_id: int, conversation_id: int) -> bool:
        """Đảm bảo RAG files được load - trả về True nếu đã load hoặc load thành công"""
        try:
            index_path = RAGService.get_faiss_path(user_id, conversation_id)

            # Kiểm tra xem đã có index chưa
            if os.path.exists(index_path) and os.path.getsize(index_path) > 100:
                index, exists = RAGService.load_faiss(user_id, conversation_id)
                if exists and index.ntotal > 0:
                    logger.info(f"RAG already loaded with {index.ntotal} vectors")
                    return True

            # Nếu chưa, load RAG files
            logger.info(
                f"Loading RAG files for user {user_id}, conversation {conversation_id}"
            )
            loaded_files = RAGService.load_rag_files_to_conversation(
                user_id, conversation_id
            )

            if loaded_files:
                logger.info(f"Successfully loaded {len(loaded_files)} RAG files")
                return True
            else:
                logger.warning("No RAG files were loaded")
                return False

        except Exception as e:
            logger.error(f"Error ensuring RAG loaded: {e}")
            return False

    @staticmethod
    def load_rag_files_to_conversation(
        user_id: int, conversation_id: int
    ) -> List[Dict[str, Any]]:
        """Tự động load tất cả file trong thư mục rag_files vào conversation"""
        rag_files = []

        supported_patterns = [
            os.path.join(RAG_FILES_DIR, "*.pdf"),
            os.path.join(RAG_FILES_DIR, "*.txt"),
            os.path.join(RAG_FILES_DIR, "*.docx"),
            os.path.join(RAG_FILES_DIR, "*.xlsx"),
            os.path.join(RAG_FILES_DIR, "*.xls"),
            os.path.join(RAG_FILES_DIR, "*.csv"),
            os.path.join(RAG_FILES_DIR, "*.parquet"),
        ]

        for pattern in supported_patterns:
            rag_files.extend(glob.glob(pattern))

        if not rag_files:
            logger.info(f"No RAG files found in {RAG_FILES_DIR}")
            return []

        logger.info(
            f"Found {len(rag_files)} RAG files to load: {[os.path.basename(f) for f in rag_files]}"
        )

        loaded_files = []

        for file_path in rag_files:
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()

                filename = os.path.basename(file_path)
                logger.info(f"Processing RAG file: {filename}")

                rag_context = RAGService.process_file_for_rag(
                    file_content, user_id, conversation_id, filename
                )

                if rag_context:
                    loaded_files.append(
                        {
                            "filename": filename,
                            "path": file_path,
                            "chunks_loaded": rag_context,
                        }
                    )
                    logger.info(f"Successfully loaded RAG file: {filename}")
                else:
                    logger.warning(f"Failed to load RAG file: {filename}")

            except Exception as e:
                logger.error(f"Error loading RAG file {file_path}: {e}")

        logger.info(
            f"Loaded {len(loaded_files)} RAG files for user {user_id}, conversation {conversation_id}"
        )
        return loaded_files

    @staticmethod
    def get_faiss_path(user_id: int, conversation_id: int) -> str:
        """Tạo đường dẫn cho FAISS index"""
        index_dir = "faiss_indices"
        os.makedirs(index_dir, exist_ok=True)
        return os.path.join(index_dir, f"faiss_{user_id}_{conversation_id}.index")

    @staticmethod
    def load_faiss(user_id: int, conversation_id: int) -> Tuple[Any, bool]:
        """Tải hoặc tạo mới FAISS index"""
        path = RAGService.get_faiss_path(user_id, conversation_id)

        if os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                index = faiss.read_index(path)
                logger.info(
                    f"Loaded FAISS index with {index.ntotal} vectors from {path}"
                )
                return index, True
            except Exception as e:
                logger.error(f"Error loading FAISS index {path}: {e}")
                try:
                    os.remove(path)
                    logger.info(f"Removed corrupted FAISS index: {path}")
                except:
                    pass

        logger.info(
            f"Creating new FAISS index for user {user_id}, conversation {conversation_id}"
        )
        index = faiss.IndexFlatIP(EmbeddingService.DIM)
        return index, False

    @staticmethod
    def get_rag_context(
        effective_query: str,
        user_id: int,
        conversation_id: int,
        db: Session,
        top_k: int = 10,
    ) -> str:
        """Lấy RAG context từ FAISS index và history - với debug chi tiết"""
        try:
            logger.info(f"Getting RAG context for query: {effective_query[:100]}...")

            # Load FAISS index directly
            index, exists = RAGService.load_faiss(user_id, conversation_id)

            # If index doesn't exist or is empty, try to load RAG files
            if not exists or index.ntotal == 0:
                logger.info("Index missing or empty, attempting to load RAG files...")
                RAGService.load_rag_files_to_conversation(user_id, conversation_id)
                index, exists = RAGService.load_faiss(user_id, conversation_id)

            if not exists or index.ntotal == 0:
                logger.warning("FAISS index is empty after loading attempt")
                return ""

            query_emb = EmbeddingService.get_embedding(effective_query, max_length=512)
            if np.all(query_emb == 0):
                logger.warning("Failed to generate query embedding")
                return ""

            logger.info(f"FAISS index has {index.ntotal} vectors")

            # Lấy history messages
            history = (
                db.query(ModelChatMessage)
                .filter(
                    ModelChatMessage.user_id == user_id,
                    ModelChatMessage.conversation_id == conversation_id,
                )
                .order_by(ModelChatMessage.timestamp.asc())
                .limit(50)
                .all()
            )

            valid_history = [
                h
                for h in history
                if h.embedding and json.loads(h.embedding) is not None
            ]
            logger.info(f"Found {len(valid_history)} valid history messages")

            # Normalize query vector
            query_emb = query_emb.astype("float32").reshape(1, -1)
            faiss.normalize_L2(query_emb)

            # Tìm kiếm
            context_from_search = RAGService._hybrid_search(
                effective_query, query_emb, index, valid_history, top_k=top_k
            )

            if context_from_search:
                logger.info(
                    f"Retrieved {len(context_from_search.split('|||'))} context chunks"
                )
                return context_from_search
            else:
                logger.warning("No relevant context found in RAG search")
                return ""

        except Exception as e:
            logger.error(f"Error in get_rag_context: {e}")
            return ""

    @staticmethod
    def _hybrid_search(
        query: str,
        query_emb: np.ndarray,
        index: Any,
        history: List[ModelChatMessage],
        top_k: int = 10,
    ) -> str:
        """Thực hiện hybrid search với FAISS và BM25"""
        if not history or index.ntotal == 0:
            return ""

        try:
            index_contents = [h.content for h in history]

            # BM25 search
            tokenized_contents = [
                re.findall(r"\w+", content.lower()) for content in index_contents
            ]
            if not any(tokenized_contents):
                return ""

            bm25 = BM25Okapi(tokenized_contents)
            query_tokens = re.findall(r"\w+", query.lower())

            if not query_tokens:
                bm25_scores = np.zeros(len(history))
            else:
                bm25_scores = bm25.get_scores(query_tokens)

            # FAISS search
            search_k = min(20, index.ntotal)
            if index.ntotal > 0:
                D, I_faiss = index.search(query_emb, k=search_k)
                faiss_scores = D[0]
                faiss_indices = I_faiss[0]
            else:
                faiss_scores = np.array([])
                faiss_indices = np.array([])

            # Kết hợp scores
            hybrid_scores = {}
            for i, idx in enumerate(faiss_indices):
                if idx < len(bm25_scores):
                    faiss_score = (faiss_scores[i] + 1) / 2
                    bm25_score = min(bm25_scores[idx] / 5, 1.0)
                    hybrid_score = 0.6 * faiss_score + 0.4 * bm25_score
                    hybrid_scores[idx] = hybrid_score

            # Nếu không có kết quả từ FAISS, sử dụng BM25
            if not hybrid_scores and len(bm25_scores) > 0:
                for idx, bm25_score in enumerate(bm25_scores):
                    if bm25_score > 0:
                        hybrid_scores[idx] = bm25_score / 5

            # Chọn top k results
            reranked_indices = sorted(
                hybrid_scores, key=hybrid_scores.get, reverse=True
            )[:top_k]

            context_messages = []
            for idx in reranked_indices:
                if idx < len(history):
                    msg = history[idx]
                    sim_score = hybrid_scores[idx]
                    if sim_score > 0.15:  # Threshold vừa phải
                        context_messages.append(msg.content)
                        logger.debug(f"Retrieved context with score {sim_score:.3f}")

            if context_messages:
                # Sử dụng separator rõ ràng
                return "|||".join(context_messages)
            else:
                return ""

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return ""

    @staticmethod
    def update_faiss_index(user_id: int, conversation_id: int, db: Session):
        """Cập nhật FAISS index với tất cả messages"""
        try:
            index = faiss.IndexFlatIP(EmbeddingService.DIM)
            all_msgs = (
                db.query(ModelChatMessage)
                .filter(ModelChatMessage.conversation_id == conversation_id)
                .all()
            )

            valid_embs = []
            for m in all_msgs:
                if m.embedding:
                    try:
                        emb = json.loads(m.embedding)
                        if isinstance(emb, list) and len(emb) == EmbeddingService.DIM:
                            valid_embs.append(emb)
                    except:
                        continue

            if valid_embs:
                emb_array = np.array(valid_embs).astype("float32")
                faiss.normalize_L2(emb_array)
                index.add(emb_array)

            faiss.write_index(
                index, RAGService.get_faiss_path(user_id, conversation_id)
            )
            logger.info(f"Updated FAISS index with {len(valid_embs)} vectors")

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật FAISS index: {e}")

    @staticmethod
    def cleanup_faiss_index(user_id: int, conversation_id: int):
        """Xóa FAISS index của một conversation"""
        try:
            path = RAGService.get_faiss_path(user_id, conversation_id)
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted FAISS index: {path}")
        except Exception as e:
            logger.error(f"Error deleting FAISS index {path}: {e}")

    @staticmethod
    def cleanup_all_user_faiss(user_id: int) -> int:
        """Xóa tất cả FAISS indices của user"""
        count = 0
        try:
            index_dir = "faiss_indices"
            if not os.path.exists(index_dir):
                return 0

            pattern = f"faiss_{user_id}_*.index"
            for f in glob.glob(os.path.join(index_dir, pattern)):
                try:
                    os.remove(f)
                    count += 1
                    logger.info(f"Deleted FAISS index: {f}")
                except Exception as e:
                    logger.error(f"Error deleting FAISS index {f}: {e}")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up user FAISS indices: {e}")
            return 0
