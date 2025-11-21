from concurrent.futures import as_completed
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
    rag_files_dir = RAG_FILES_DIR
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text thành các đoạn nhỏ với overlap - improved version"""
        if len(text) <= chunk_size:
            return [text]
        
        # Preprocess: loại bỏ khoảng trắng thừa và normalize newlines
        text = re.sub(r'\n+', '\n', text.strip())
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Nếu không phải chunk cuối, tìm điểm cắt tốt
            if end < text_length:
                # Ưu tiên tìm các điểm ngắt tự nhiên
                break_points = [
                    text.rfind('\n\n', start, end),  # Double newline
                    text.rfind('\n===', start, end),  # Section break
                    text.rfind('\n', start, end),     # Single newline
                    text.rfind('. ', start, end),     # Sentence end
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('; ', start, end),
                    text.rfind(', ', start, end),
                ]
                
                # Chọn điểm break tốt nhất
                best_break = -1
                for bp in break_points:
                    if bp != -1 and bp > start + (chunk_size // 3):
                        best_break = bp
                        break
                
                if best_break != -1:
                    end = best_break + 1  # +1 để bao gồm ký tự break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= 20:  # Chỉ thêm chunks có ý nghĩa
                chunks.append(chunk)
            
            # Di chuyển start với overlap, đảm bảo không quay lại
            next_start = end - overlap
            if next_start <= start:  # Tránh infinite loop
                next_start = end
            start = next_start
        
        return chunks
    
    @staticmethod
    def process_file_for_rag(file_content: bytes, user_id: int, conversation_id: int, filename: str = "") -> str:
        """Xử lý file để tạo RAG context - improved with batch processing"""
        try:
            # Extract metadata
            metadata = {}
            file_extension = filename.lower().split('.')[-1] if filename else ""
            
            if file_extension in ['xlsx', 'xls']:
                metadata = FileService.extract_excel_metadata(file_content)
            elif file_extension == 'docx':
                metadata = FileService.extract_docx_metadata(file_content)
            elif file_extension == 'txt':
                metadata = RAGService._extract_txt_metadata(file_content)
            elif file_extension == 'parquet':
                metadata = RAGService._extract_parquet_metadata(file_content)
            
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
            
            # Batch processing embeddings
            embeddings = []
            valid_chunks = []
            
            # Chuẩn bị chunks với metadata
            chunk_texts = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue
                chunk_with_info = f"[File: {filename}] [Chunk {i+1}/{len(chunks)}] {chunk}"
                chunk_texts.append(chunk_with_info)
                valid_chunks.append(chunk_with_info)
            
            # Tạo embeddings batch
            if chunk_texts:
                embeddings_batch = EmbeddingService.get_embeddings_batch(chunk_texts, max_length=512)
                
                for emb in embeddings_batch:
                    if np.any(emb) and not np.all(emb == 0):
                        embeddings.append(emb)
            
            if embeddings:
                emb_array = np.array(embeddings).astype('float32')
                
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(emb_array)
                index.add(emb_array)
                
                faiss_path = RAGService.get_faiss_path(user_id, conversation_id)
                executor.submit(faiss.write_index, index, faiss_path).result()
                
                logger.info(f"Added {len(embeddings)} chunks from file '{filename}' to RAG index")
                
                result_context = f"{metadata_context}\n[File content loaded: {len(valid_chunks)} chunks, {sum(len(chunk) for chunk in valid_chunks)} characters]"
                return result_context
            else:
                logger.warning(f"No valid embeddings created from file '{filename}' chunks")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing file for RAG: {e}")
            return ""
    
    @staticmethod
    def _extract_txt_metadata(file_content: bytes) -> Dict[str, Any]:
        """Trích xuất metadata từ TXT file"""
        try:
            text = file_content.decode('utf-8', errors='replace')
            lines = text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            metadata = {
                "file_type": "txt",
                "line_count": len(lines),
                "non_empty_line_count": len(non_empty_lines),
                "character_count": len(text),
                "word_count": len(text.split()),
                "has_content": len(text.strip()) > 0
            }
            
            # Thêm thông tin về encoding
            try:
                file_content.decode('utf-8')
                metadata["encoding"] = "utf-8"
            except:
                metadata["encoding"] = "other"
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting TXT metadata: {e}")
            return {"file_type": "txt", "error": str(e)}
    
    @staticmethod
    def _extract_parquet_metadata(file_content: bytes) -> Dict[str, Any]:
        """Trích xuất metadata từ Parquet file"""
        try:
            import pandas as pd
            import io
            
            # Đọc Parquet file từ bytes
            parquet_file = io.BytesIO(file_content)
            df = pd.read_parquet(parquet_file)
            
            metadata = {
                "file_type": "parquet",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": df.memory_usage(deep=True).sum(),
                "has_content": not df.empty
            }
            
            # Thêm thông tin về missing values
            missing_values = df.isnull().sum().to_dict()
            metadata["missing_values"] = missing_values
            
            # Thêm thông tin về dtypes chi tiết
            dtype_counts = df.dtypes.value_counts().to_dict()
            metadata["dtype_distribution"] = {str(k): int(v) for k, v in dtype_counts.items()}
            
            # Thêm sample data (3 dòng đầu)
            if not df.empty:
                metadata["sample_data"] = df.head(3).to_dict('records')
            
            logger.info(f"Extracted Parquet metadata: {len(df)} rows, {len(df.columns)} columns")
            return metadata
            
        except ImportError:
            logger.error("pandas is required for Parquet file processing")
            return {"file_type": "parquet", "error": "pandas library not available"}
        except Exception as e:
            logger.error(f"Error extracting Parquet metadata: {e}")
            return {"file_type": "parquet", "error": str(e)}
    
    @staticmethod
    def get_index_stats(user_id: int, conversation_id: int) -> Dict[str, Any]:
        """Lấy thống kê về FAISS index"""
        try:
            index, exists = RAGService.load_faiss(user_id, conversation_id)
            return {
                "exists": exists,
                "vector_count": index.ntotal,
                "dimension": index.d,
                "index_type": "FlatIP (Cosine Similarity)"
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"exists": False, "vector_count": 0, "error": str(e)}
    
    @staticmethod
    def _build_metadata_context(metadata: Dict[str, Any], filename: str) -> str:
        """Xây dựng metadata context string - improved với TXT và Parquet"""
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
        elif metadata.get("file_type") == "txt":
            metadata_context = f"\n[TXT File: {metadata.get('line_count', 0)} lines, {metadata.get('word_count', 0)} words, {metadata.get('character_count', 0)} characters]"
        elif metadata.get("file_type") == "parquet":
            metadata_context = f"\n[Parquet File: {metadata.get('row_count', 0)} rows, {metadata.get('column_count', 0)} columns"
            if 'columns' in metadata:
                metadata_context += f", columns: {', '.join(metadata['columns'][:5])}"
                if len(metadata['columns']) > 5:
                    metadata_context += f" ... (+{len(metadata['columns']) - 5} more)"
            metadata_context += "]"
        
        if filename:
            metadata_context = f"[File: {filename}]" + metadata_context
            
        return metadata_context
    
    # Trong rag_service.py - sửa method load_rag_files_to_conversation

    @staticmethod
    def load_rag_files_to_conversation(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
        """Tự động load tất cả file trong thư mục rag_files vào conversation - improved with parallel processing"""
        rag_files = []
        
        supported_patterns = [
            os.path.join(RAG_FILES_DIR, "*.pdf"),
            os.path.join(RAG_FILES_DIR, "*.txt"), 
            os.path.join(RAG_FILES_DIR, "*.docx"),
            os.path.join(RAG_FILES_DIR, "*.xlsx"),
            os.path.join(RAG_FILES_DIR, "*.xls"),
            os.path.join(RAG_FILES_DIR, "*.csv"),
            os.path.join(RAG_FILES_DIR, "*.parquet")
        ]
        
        for pattern in supported_patterns:
            rag_files.extend(glob.glob(pattern))
        
        if not rag_files:
            logger.info(f"No RAG files found in {RAG_FILES_DIR}")
            return []
        
        loaded_files = []
        
        # SỬA: Sử dụng ThreadPoolExecutor thay vì as_completed trong async context
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file_path in rag_files:
                future = executor.submit(RAGService._process_single_rag_file, file_path, user_id, conversation_id)
                futures.append(future)
            
            # Chờ tất cả futures hoàn thành
            for future in futures:
                try:
                    result = future.result(timeout=300)  # Timeout 5 phút
                    if result:
                        loaded_files.append(result)
                        logger.info(f"Successfully loaded RAG file: {result['filename']}")
                except Exception as e:
                    logger.error(f"Error processing RAG file: {e}")
        
        logger.info(f"Loaded {len(loaded_files)} RAG files for user {user_id}, conversation {conversation_id}")
        return loaded_files

    @staticmethod
    def _process_single_rag_file(file_path: str, user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Xử lý một file RAG - tách riêng để dùng với ThreadPoolExecutor"""
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            filename = os.path.basename(file_path)
            logger.info(f"Processing RAG file: {filename}")
            
            rag_context = RAGService.process_file_for_rag(file_content, user_id, conversation_id, filename)
            
            if rag_context:
                return {
                    "filename": filename,
                    "path": file_path,
                    "chunks_loaded": rag_context
                }
            else:
                logger.warning(f"Failed to load RAG file: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading RAG file {file_path}: {e}")
            return None
    
    @staticmethod
    def get_faiss_path(user_id: int, conversation_id: int) -> str:
        """Tạo đường dẫn cho FAISS index"""
        index_dir = "faiss_indices"
        os.makedirs(index_dir, exist_ok=True)
        return os.path.join(index_dir, f"faiss_{user_id}_{conversation_id}.index")
    
    @staticmethod
    def load_faiss(user_id: int, conversation_id: int) -> Tuple[Any, bool]:
        """Tải hoặc tạo mới FAISS index - improved with error handling"""
        path = RAGService.get_faiss_path(user_id, conversation_id)
        
        # Kiểm tra file size để tránh corrupted index
        if os.path.exists(path) and os.path.getsize(path) > 100:  # Ít nhất 100 bytes
            try:
                index = faiss.read_index(path)
                if index.ntotal >= 0:  # Kiểm tra index hợp lệ
                    logger.info(f"Loaded FAISS index with {index.ntotal} vectors from {path}")
                    return index, True
            except Exception as e:
                logger.error(f"Error loading FAISS index {path}: {e}")
                # Xóa file corrupted
                try:
                    os.remove(path)
                    logger.info(f"Removed corrupted FAISS index: {path}")
                except:
                    pass
        
        # Tạo index mới với cosine similarity
        logger.info(f"Creating new FAISS index for user {user_id}, conversation {conversation_id}")
        index = faiss.IndexFlatIP(EmbeddingService.DIM)  # Inner product for cosine similarity
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
    
    # Trong rag_service.py - sửa method get_rag_context

    @staticmethod
    def get_rag_context(effective_query: str, user_id: int, conversation_id: int, db: Session, top_k: int = 5) -> str:
        """Lấy RAG context từ FAISS index và history - với auto-load RAG files"""
        try:
            query_emb = EmbeddingService.get_embedding(effective_query, max_length=512)
            if np.all(query_emb == 0):
                logger.warning("Failed to generate query embedding")
                return RAGService._get_fallback_context(db, user_id, conversation_id)
            
            index, exists = RAGService.load_faiss(user_id, conversation_id)
            
            # THÊM: Tự động load RAG files nếu index trống
            if index.ntotal == 0:
                logger.info(f"Auto-loading RAG files for conversation {conversation_id} (empty index)")
                RAGService.load_rag_files_to_conversation(user_id, conversation_id)
                # Load lại index sau khi đã thêm RAG files
                index, exists = RAGService.load_faiss(user_id, conversation_id)
            
            # Lấy history messages
            history = db.query(ModelChatMessage).filter(
                ModelChatMessage.user_id == user_id,
                ModelChatMessage.conversation_id == conversation_id
            ).order_by(ModelChatMessage.timestamp.asc()).limit(100).all()
            
            valid_history = [h for h in history if h.embedding and json.loads(h.embedding) is not None]
            
            if not valid_history and index.ntotal == 0:
                logger.info("No history or FAISS data available for RAG")
                return ""
            
            # Normalize query vector for cosine similarity
            query_emb = query_emb.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_emb)
            
            # Hybrid search (FAISS + BM25)
            context_from_search = RAGService._hybrid_search(
                effective_query, query_emb, index, valid_history, top_k=top_k
            )
            
            if context_from_search:
                return f"Relevant Context:\n{context_from_search}"
            else:
                return RAGService._get_fallback_context(db, user_id, conversation_id)
                
        except Exception as e:
            logger.error(f"Error in get_rag_context: {e}")
            return RAGService._get_fallback_context(db, user_id, conversation_id)
    
    @staticmethod
    def _get_fallback_context(db: Session, user_id: int, conversation_id: int, limit: int = 10) -> str:
        """Fallback context khi search không có kết quả"""
        try:
            recent_messages = db.query(ModelChatMessage).filter(
                ModelChatMessage.user_id == user_id,
                ModelChatMessage.conversation_id == conversation_id
            ).order_by(ModelChatMessage.timestamp.desc()).limit(limit).all()
            
            if recent_messages:
                context = "\n".join([msg.content for msg in reversed(recent_messages)])
                return f"Recent Conversation:\n{context}"
            else:
                return ""
        except Exception as e:
            logger.error(f"Error getting fallback context: {e}")
            return ""
    
    @staticmethod
    def _hybrid_search(query: str, query_emb: np.ndarray, index: Any, history: List[ModelChatMessage], top_k: int = 5) -> str:
        """Thực hiện hybrid search với FAISS và BM25 - improved with better scoring"""
        if not history or index.ntotal == 0:
            return ""
        
        try:
            index_contents = [h.content for h in history]
            
            # BM25 search
            tokenized_contents = [re.findall(r'\w+', content.lower()) for content in index_contents]
            if not any(tokenized_contents):
                return ""
                
            bm25 = BM25Okapi(tokenized_contents)
            query_tokens = re.findall(r'\w+', query.lower())
            
            if not query_tokens:
                bm25_scores = np.zeros(len(history))
            else:
                bm25_scores = bm25.get_scores(query_tokens)
            
            # FAISS search (cosine similarity)
            if index.ntotal > 0:
                D, I_faiss = index.search(query_emb, k=min(20, index.ntotal))
                faiss_scores = D[0]  # Cosine similarity scores
                faiss_indices = I_faiss[0]
            else:
                faiss_scores = np.array([])
                faiss_indices = np.array([])
            
            # Kết hợp scores
            hybrid_scores = {}
            for i, idx in enumerate(faiss_indices):
                if idx < len(bm25_scores):
                    # Normalize scores và kết hợp
                    faiss_score = (faiss_scores[i] + 1) / 2  # Convert từ [-1,1] sang [0,1]
                    bm25_score = min(bm25_scores[idx] / 10, 1.0)  # Normalize BM25 score
                    
                    hybrid_score = 0.6 * faiss_score + 0.4 * bm25_score
                    hybrid_scores[idx] = hybrid_score
            
            # Nếu không có kết quả từ FAISS, sử dụng BM25
            if not hybrid_scores and len(bm25_scores) > 0:
                for idx, bm25_score in enumerate(bm25_scores):
                    if bm25_score > 0:
                        hybrid_scores[idx] = bm25_score / 10
            
            # Chọn top k results
            reranked_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_k]
            
            context_messages = []
            for idx in reranked_indices:
                if idx < len(history):
                    msg = history[idx]
                    sim_score = hybrid_scores[idx]
                    if sim_score > 0.2:  # Threshold thấp hơn để lấy nhiều context hơn
                        context_messages.append(f"[Relevance: {sim_score:.2f}] {msg.content}")
                        logger.debug(f"Retrieved msg {idx}: score {sim_score:.3f}")
            
            return "\n".join(context_messages) if context_messages else ""
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return ""
    
    @staticmethod
    def update_faiss_index(user_id: int, conversation_id: int, db: Session):
        """Cập nhật FAISS index với tất cả messages - improved with normalization"""
        try:
            index = faiss.IndexFlatIP(EmbeddingService.DIM)
            all_msgs = db.query(ModelChatMessage).filter(
                ModelChatMessage.conversation_id == conversation_id
            ).all()
            
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
                emb_array = np.array(valid_embs).astype('float32')
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(emb_array)
                index.add(emb_array)
            
            executor.submit(
                faiss.write_index, 
                index, 
                RAGService.get_faiss_path(user_id, conversation_id)
            ).result()
            
            logger.info(f"Updated FAISS index with {len(valid_embs)} vectors for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật FAISS index: {e}")
