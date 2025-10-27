import numpy as np
import ollama
import logging
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model: str = "embeddinggemma:latest", dim: int = 768):
        self.model = model
        self.dim = dim

    def get_embedding(self, text: str, max_length: int = 1024) -> np.ndarray:
        """Tạo embedding cho text"""
        try:
            if len(text) > max_length:
                text = text[:max_length]
            resp = ollama.embeddings(model=self.model, prompt=text)
            return np.array(resp["embedding"])
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding từ Ollama: {e}")
            return np.zeros(self.dim)

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Tạo embedding cho nhiều text"""
        return [self.get_embedding(text) for text in texts]
