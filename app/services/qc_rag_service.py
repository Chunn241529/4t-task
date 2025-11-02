import os
import json
from typing import List, Dict, Any
import faiss
import numpy as np
from datetime import datetime

class QCRAGService:
    def __init__(self, base_rag_service):
        self.rag_service = base_rag_service
        self.qc_index_prefix = "qc_"
    
    def process_project_document(self, file_content: bytes, filename: str, user_id: int) -> str:
        """Xử lý document dự án cho QC knowledge base"""
        try:
            # Extract text (dùng hàm có sẵn từ chat.py)
            from app.routers.chat import extract_text_from_file, chunk_text
            
            full_text = extract_text_from_file(file_content)
            if not full_text.strip():
                return f"❌ Không thể extract text từ {filename}"
            
            # Chunk text
            chunks = chunk_text(full_text, chunk_size=1000, overlap=100)
            
            # Tạo embeddings và add to FAISS
            embeddings = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk)
                embeddings.append(embedding)
                
                chunk_metadata.append({
                    "content": chunk,
                    "source": filename,
                    "chunk_id": i,
                    "type": "project_requirement",
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": user_id
                })
            
            # Save to FAISS
            if embeddings:
                self._save_to_qc_faiss(np.array(embeddings), chunk_metadata, user_id)
            
            return f"✅ Đã import {filename}: {len(chunks)} chunks, {len(full_text)} ký tự"
            
        except Exception as e:
            return f"❌ Lỗi khi xử lý {filename}: {str(e)}"
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding cho text - dùng hàm có sẵn"""
        from app.routers.chat import get_embedding
        return get_embedding(text)
    
    def _save_to_qc_faiss(self, embeddings: np.ndarray, metadata: List[Dict], user_id: int):
        """Lưu embeddings vào QC FAISS index"""
        index_path = f"faiss_indices/{user_id}_{self.qc_index_prefix}project_docs.index"
        metadata_path = f"faiss_indices/{user_id}_{self.qc_index_prefix}project_docs_metadata.json"
        
        # Tạo hoặc load index hiện có
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            existing_metadata = []
        
        # Add new embeddings
        index.add(embeddings)
        existing_metadata.extend(metadata)
        
        # Lưu lại
        os.makedirs("faiss_indices", exist_ok=True)
        faiss.write_index(index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, ensure_ascii=False, indent=2)
    
    def get_qc_context(self, query: str, user_id: int, top_k: int = 8) -> str:
        """Lấy context liên quan cho QC workflow"""
        try:
            index_path = f"faiss_indices/{user_id}_{self.qc_index_prefix}project_docs.index"
            metadata_path = f"faiss_indices/{user_id}_{self.qc_index_prefix}project_docs_metadata.json"
            
            if not os.path.exists(index_path):
                return "📝 Chưa có project documents nào được import. Dùng /qc_import để thêm documents."
            
            # Load index và metadata
            index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if index.ntotal == 0:
                return "📝 QC knowledge base đang trống."
            
            # Search
            query_embedding = self._get_embedding(query)
            D, I = index.search(query_embedding.reshape(1, -1), k=min(top_k, index.ntotal))
            
            # Lấy relevant chunks
            relevant_chunks = []
            for idx in I[0]:
                if idx < len(metadata):
                    chunk_data = metadata[idx]
                    source = chunk_data.get('source', 'unknown')
                    content = chunk_data.get('content', '')[:500] + "..."  # Limit length
                    relevant_chunks.append(f"📄 {source}:\n{content}")
            
            return "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else "Không tìm thấy context phù hợp."
            
        except Exception as e:
            return f"❌ Lỗi khi retrieve context: {str(e)}"
