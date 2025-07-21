"""
BM25 indexing and search utilities
"""
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BM25IndexManager:
    """Manager for BM25 index creation and searching"""
    
    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}  # Maps BM25 index position to document ID
        self.tokenized_corpus = []
        self.raw_corpus = []
        
    def build_index(self, documents: List[str], doc_ids: List[str]) -> bool:
        """Build BM25 index from documents"""
        try:
            self.raw_corpus = documents
            self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
            
            # Tokenize documents using jieba for Chinese text
            self.tokenized_corpus = []
            for doc in documents:
                tokens = list(jieba.cut(doc))
                self.tokenized_corpus.append(tokens)
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            logger.info(f"BM25 index built with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if not self.bm25_index:
            return []
        
        try:
            # Tokenize query
            tokenized_query = list(jieba.cut(query))
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top documents
            top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
            
            # Return results
            results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # Only return relevant results
                    results.append({
                        'id': self.doc_mapping[idx],
                        'score': float(bm25_scores[idx]),
                        'content': self.raw_corpus[idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []
    
    def clear(self):
        """Clear the index"""
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

if __name__ == "__main__":
    # Test function
    manager = BM25IndexManager()
    docs = ["这是第一个文档", "这是第二个文档关于机器学习", "第三个文档讨论深度学习"]
    ids = ["doc1", "doc2", "doc3"]
    
    if manager.build_index(docs, ids):
        results = manager.search("机器学习", top_k=2)
        print(f"Found {len(results)} results for query '机器学习'")
