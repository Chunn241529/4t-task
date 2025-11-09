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

# Định nghĩa thư mục RAG
RAG_FILES_DIR = "rag_files"
os.makedirs(RAG_FILES_DIR, exist_ok=True)

class RAGService:
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text thành các đoạn nhỏ với overlap"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Tìm điểm cắt tốt
            if end < len(text):
                section_break = text.rfind('\n===', start, end)
                if section_break != -1 and section_break > start + chunk_size // 2:
                    end = section_break
                else:
                    line_break = text.rfind('\n', start, end)
                    if line_break != -1 and line_break > start + chunk_size // 2:
                        end = line_break + 1
                    else:
                        for punctuation in ['. ', '! ', '? ', '。', '！', '？', '; ']:
                            punctuation_pos = text.rfind(punctuation, start, end)
                            if punctuation_pos != -1 and punctuation_pos > start + chunk_size // 2:
                                end = punctuation_pos + len(punctuation)
                                break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    @staticmethod
    def process_file_for_rag(file_content: bytes, user_id: int, conversation_id: int, filename: str = "") -> str:
        """Xử lý file để tạo RAG context"""
        try:
            # Extract metadata
            metadata = {}
            file_extension = filename.lower().split('.')[-1] if filename else ""
            
            if file_extension in ['xlsx', 'xls']:
                metadata = FileService.extract_excel_metadata(file_content)
            elif file_extension == 'docx':
                metadata = FileService.extract_docx_metadata(file_content)
            
            file_text = FileService.extract_text_from_file(file_content)
            if not file_text.strip():
                logger.warning("No text extracted from file for RAG")
                return ""
            
            # Thêm metadata vào context
            metadata_context = RAGService._build_metadata_context(metadata, filename)
            
            # Chunk text
            chunks = RAGService.chunk_text(file_text, chunk_size=600, overlap=75)
            if not chunks:
                logger.warning("No chunks created from file text")
                return ""
            
            # Load FAISS index
            index, exists = RAGService.load_faiss(user_id, conversation_id)
            
            embeddings = []
            valid_chunks = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue
                    
                chunk_with_info = f"[Chunk {i+1}/{len(chunks)}] {chunk}"
                emb = EmbeddingService.get_embedding(chunk_with_info, max_length=512)
                
                if np.any(emb) and not np.all(emb == 0):
                    embeddings.append(emb)
                    valid_chunks.append(chunk_with_info)
            
            if embeddings:
                emb_array = np.array(embeddings)
                index.add(emb_array)
                
                faiss_path = RAGService.get_faiss_path(user_id, conversation_id)
                executor.submit(faiss.write_index, index, faiss_path).result()
                
                logger.info(f"Added {len(embeddings)} chunks from file to RAG index")
                
                result_context = f"{metadata_context}\n[File content loaded: {len(valid_chunks)} chunks, {sum(len(chunk) for chunk in valid_chunks)} characters]"
                return result_context
            else:
                logger.warning("No valid embeddings created from file chunks")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing file for RAG: {e}")
            return ""
    
    @staticmethod
    def _build_metadata_context(metadata: Dict[str, Any], filename: str) -> str:
        """Xây dựng metadata context string"""
        if not metadata:
            return f"[File: {filename}]" if filename else ""
        
        metadata_context = ""
        if metadata.get("file_type") == "excel":
            metadata_context = f"\n[Excel File: {metadata['sheet_count']} sheets"
            for sheet in metadata["sheets"]:
                metadata_context += f", '{sheet['name']}' ({sheet['rows']} rows, {sheet['columns']} columns)"
            metadata_context += "]"
        elif metadata.get("file_type") == "docx":
            metadata_context = f"\n[DOCX File: {metadata['paragraph_count']} paragraphs, {metadata['table_count']} tables]"
        
        if filename:
            metadata_context = f"[File: {filename}]" + metadata_context
            
        return metadata_context
    
    @staticmethod
    def load_rag_files_to_conversation(user_id: int, conversation_id: int):
        """Tự động load tất cả file trong thư mục rag_files vào conversation"""
        rag_files = []
        
        supported_patterns = [
            os.path.join(RAG_FILES_DIR, "*.pdf"),
            os.path.join(RAG_FILES_DIR, "*.txt"), 
            os.path.join(RAG_FILES_DIR, "*.docx"),
            os.path.join(RAG_FILES_DIR, "*.xlsx"),
            os.path.join(RAG_FILES_DIR, "*.xls"),
            os.path.join(RAG_FILES_DIR, "*.csv")
        ]
        
        for pattern in supported_patterns:
            rag_files.extend(glob.glob(pattern))
        
        if not rag_files:
            logger.info(f"No RAG files found in {RAG_FILES_DIR}")
            return []
        
        loaded_files = []
        for file_path in rag_files:
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                
                filename = os.path.basename(file_path)
                logger.info(f"Processing RAG file: {filename}")
                
                rag_context = RAGService.process_file_for_rag(file_content, user_id, conversation_id, filename)
                
                if rag_context:
                    loaded_files.append({
                        "filename": filename,
                        "path": file_path,
                        "chunks_loaded": rag_context
                    })
                    logger.info(f"Successfully loaded RAG file: {filename}")
                else:
                    logger.warning(f"Failed to load RAG file: {filename}")
                    
            except Exception as e:
                logger.error(f"Error loading RAG file {file_path}: {e}")
        
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
        if os.path.exists(path):
            try:
                index = faiss.read_index(path)
                return index, True
            except Exception as e:
                logger.error(f"Error loading FAISS index, creating new: {e}")
        index = faiss.IndexFlatL2(EmbeddingService.DIM)
        return index, False
    
    @staticmethod
    def cleanup_faiss_index(user_id: int, conversation_id: int):
        """Dọn dẹp FAISS index"""
        path = RAGService.get_faiss_path(user_id, conversation_id)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"Đã xóa FAISS index: {path}")
            except Exception as e:
                logger.error(f"Lỗi khi xóa FAISS index {path}: {e}")
    
    @staticmethod
    def get_rag_context(effective_query: str, user_id: int, conversation_id: int, db: Session) -> str:
        """Lấy RAG context từ FAISS index và history"""
        query_emb = EmbeddingService.get_embedding(effective_query, max_length=1024)
        index, exists = RAGService.load_faiss(user_id, conversation_id)
        
        # Lấy history messages
        history = db.query(ModelChatMessage).filter(
            ModelChatMessage.user_id == user_id,
            ModelChatMessage.conversation_id == conversation_id
        ).order_by(ModelChatMessage.timestamp.asc()).limit(50).all()
        
        valid_history = [h for h in history if h.embedding and json.loads(h.embedding) is not None]
        
        all_context_parts = []
        
        if not valid_history:
            context_from_history = ""
        else:
            # Rebuild index nếu cần
            if not exists or index.ntotal != len(valid_history):
                index = faiss.IndexFlatL2(EmbeddingService.DIM)
                embs = np.array([json.loads(h.embedding) for h in valid_history])
                if len(embs) > 0:
                    index.add(embs)
                    executor.submit(faiss.write_index, index, RAGService.get_faiss_path(user_id, conversation_id)).result()
            
            # Hybrid search (FAISS + BM25)
            context_from_history = RAGService._hybrid_search(
                effective_query, index, valid_history
            )
            if context_from_history:
                all_context_parts.append(f"History Context:\n{context_from_history}")
        
        final_context = "\n\n".join(all_context_parts)
        if not final_context:
            logger.warning("No relevant context retrieved, using recent history fallback")
            final_context = "\n".join([h.content for h in valid_history[-10:]])
        
        return final_context
    
    @staticmethod
    def _hybrid_search(query: str, index: Any, history: List[ModelChatMessage]) -> str:
        """Thực hiện hybrid search với FAISS và BM25"""
        index_contents = [h.content for h in history]
        tokenized_contents = [re.findall(r'\w+', content.lower()) for content in index_contents]
        bm25 = BM25Okapi(tokenized_contents)
        
        query_tokens = re.findall(r'\w+', query.lower())
        bm25_scores = bm25.get_scores(query_tokens)
        
        query_emb = EmbeddingService.get_embedding(query, max_length=1024)
        
        if index.ntotal > 0:
            D, I_faiss = index.search(query_emb.reshape(1, -1), k=min(10, index.ntotal))
            faiss_indices = I_faiss[0]
        else:
            faiss_indices = []
        
        hybrid_scores = {}
        for i, idx in enumerate(faiss_indices):
            if idx < len(bm25_scores):
                hybrid_score = 0.7 * (1 - D[0][i]) + 0.3 * bm25_scores[idx]
                hybrid_scores[idx] = hybrid_score
        
        reranked_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:5]
        
        context_messages = []
        for idx in reranked_indices:
            if idx < len(history):
                msg = history[idx]
                sim_score = hybrid_scores[idx]
                if sim_score > 0.3:
                    context_messages.append(msg.content)
                    logger.debug(f"Retrieved msg {idx}: score {sim_score:.3f}, content: {msg.content[:50]}...")
        
        return "\n".join(context_messages)
    
    @staticmethod
    def update_faiss_index(user_id: int, conversation_id: int, db: Session):
        """Cập nhật FAISS index với tất cả messages"""
        try:
            index = faiss.IndexFlatL2(EmbeddingService.DIM)
            all_msgs = db.query(ModelChatMessage).filter(
                ModelChatMessage.conversation_id == conversation_id
            ).all()
            
            valid_embs = [json.loads(m.embedding) for m in all_msgs if m.embedding]
            if valid_embs:
                index.add(np.array(valid_embs))
            
            executor.submit(
                faiss.write_index, 
                index, 
                RAGService.get_faiss_path(user_id, conversation_id)
            ).result()
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật FAISS index: {e}")
