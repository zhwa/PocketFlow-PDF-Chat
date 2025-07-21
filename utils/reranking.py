"""
Result reranking utilities using cross-encoder and LLM
"""
import threading
import logging
from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Global cross-encoder instance with thread lock
_cross_encoder = None
_cross_encoder_lock = threading.Lock()

def get_cross_encoder():
    """Get cross-encoder model (lazy loading with thread safety)"""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    _cross_encoder = CrossEncoder('sentence-transformers/distiluse-base-multilingual-cased-v2')
                    logger.info("Cross-encoder loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load cross-encoder: {str(e)}")
                    _cross_encoder = None
    return _cross_encoder

def rerank_with_cross_encoder(query: str, docs: List[str], doc_ids: List[str], 
                            metadata_list: List[Dict], top_k: int = 5) -> List[Tuple]:
    """Rerank results using cross-encoder"""
    if not docs:
        return []
    
    encoder = get_cross_encoder()
    if encoder is None:
        logger.warning("Cross-encoder not available, skipping reranking")
        # Return original order with dummy scores
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
    
    try:
        # Prepare cross-encoder inputs
        cross_inputs = [[query, doc] for doc in docs]
        
        # Calculate relevance scores
        scores = encoder.predict(cross_inputs)
        
        # Combine results
        results = [
            (doc_id, {
                'content': doc, 
                'metadata': meta,
                'score': float(score)
            }) 
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        
        # Sort by score (descending)
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Cross-encoder reranking failed: {str(e)}")
        # Fallback to original order
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

def rerank_with_llm(query: str, docs: List[str], doc_ids: List[str], 
                   metadata_list: List[Dict], top_k: int = 5) -> List[Tuple]:
    """Rerank results using LLM scoring"""
    if not docs:
        return []
    
    try:
        from .llm_client import call_llm
        
        results = []
        
        # Score each document
        for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
            prompt = f"""给定以下查询和文档片段，评估它们的相关性。
评分标准：0分表示完全不相关，10分表示高度相关。
只需返回一个0-10之间的整数分数，不要有任何其他解释。

查询: {query}

文档片段: {doc}

相关性分数(0-10):"""
            
            try:
                result = call_llm(prompt, model_choice="ollama")
                
                # Extract score
                import re
                match = re.search(r'\b([0-9]|10)\b', result)
                if match:
                    score = float(match.group(1)) / 10.0  # Normalize to 0-1
                else:
                    score = 0.5  # Default score
                    
            except Exception as e:
                logger.error(f"LLM scoring failed for doc {doc_id}: {str(e)}")
                score = 0.5  # Default score
            
            results.append((doc_id, {
                'content': doc, 
                'metadata': meta,
                'score': score
            }))
        
        # Sort by score (descending)
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"LLM reranking failed: {str(e)}")
        # Fallback to original order
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

if __name__ == "__main__":
    # Test function
    query = "machine learning"
    docs = ["This is about machine learning", "This is about cooking", "Deep learning is a subset of ML"]
    doc_ids = ["doc1", "doc2", "doc3"]
    metadata = [{"source": "test"} for _ in docs]
    
    reranked = rerank_with_cross_encoder(query, docs, doc_ids, metadata, top_k=2)
    print(f"Reranked {len(reranked)} results")
