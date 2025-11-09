import ollama
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingService:
    DIM = 768
    
    @staticmethod
    def get_embedding(text: str, max_length: int = 1024) -> np.ndarray:
        """Tạo embedding cho text"""
        try:
            if len(text) > max_length:
                text = text[:max_length]
            resp = ollama.embeddings(model="embeddinggemma:latest", prompt=text)
            return np.array(resp["embedding"])
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding từ Ollama: {e}")
            return np.zeros(EmbeddingService.DIM)
    
    @staticmethod
    def get_embeddings_batch(texts: List[str], max_length: int = 1024) -> List[np.ndarray]:
        """Tạo embeddings cho batch texts (có thể optimize sau)"""
        embeddings = []
        for text in texts:
            emb = EmbeddingService.get_embedding(text, max_length)
            embeddings.append(emb)
        return embeddings
