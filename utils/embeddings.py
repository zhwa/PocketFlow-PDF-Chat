"""
Text embedding utilities
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

# Global embedding model instance
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Get the embedding model (lazy loading)"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    return _embedding_model

def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Get embeddings for a list of texts"""
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [np.array(emb, dtype=np.float32) for emb in embeddings]

if __name__ == "__main__":
    # Test function
    test_texts = ["This is a test", "这是一个测试"]
    embeddings = get_embeddings(test_texts)
    print(f"Created {len(embeddings)} embeddings with shape {embeddings[0].shape}")