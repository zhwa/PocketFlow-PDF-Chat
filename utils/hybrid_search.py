"""
Hybrid search utilities for merging semantic and BM25 results
"""
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def hybrid_merge(semantic_results: Dict, bm25_results: List[Dict], alpha: float = 0.7) -> List[Tuple]:
    """
    Merge semantic search and BM25 search results
    
    Args:
        semantic_results: Dictionary with 'ids', 'documents', 'metadatas' keys
        bm25_results: List of dicts with 'id', 'score', 'content' keys
        alpha: Weight for semantic search (0-1)
        
    Returns:
        List of tuples: [(doc_id, {'score': score, 'content': content, 'metadata': metadata}), ...]
    """
    merged_dict = {}
    
    # Process semantic search results
    if (semantic_results and 
        semantic_results.get('documents') and len(semantic_results['documents']) > 0 and
        semantic_results.get('metadatas') and len(semantic_results['metadatas']) > 0 and
        semantic_results.get('ids') and len(semantic_results['ids']) > 0):
        
        # Handle nested list structure from FAISS/ChromaDB
        docs = semantic_results['documents']
        metas = semantic_results['metadatas'] 
        ids = semantic_results['ids']
        
        # Flatten if nested (common with batch queries)
        if isinstance(docs[0], list):
            docs = docs[0]
            metas = metas[0] 
            ids = ids[0]
        
        # Ensure all lists have the same length
        if len(ids) == len(docs) == len(metas):
            num_results = len(ids)
            for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
                score = 1.0 - (i / max(1, num_results))  # Higher rank gets higher score
                merged_dict[doc_id] = {
                    'score': alpha * score, 
                    'content': doc,
                    'metadata': meta
                }
        else:
            # Let the calling Node handle this via retry mechanism
            raise ValueError("Semantic results have mismatched lengths")
    else:
        logger.info("No valid semantic results to merge")
    
    # Process BM25 results
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Get max BM25 score for normalization
    valid_bm25_scores = [r['score'] for r in bm25_results if 'score' in r]
    max_bm25_score = max(valid_bm25_scores) if valid_bm25_scores else 1.0
    
    for result in bm25_results:
        if not ('id' in result and 'score' in result and 'content' in result):
            # Let the calling Node handle this via retry mechanism
            raise ValueError(f"Invalid BM25 result format: {result}")
        
        doc_id = result['id']
        normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
        
        if doc_id in merged_dict:
            # Combine scores
            merged_dict[doc_id]['score'] += (1 - alpha) * normalized_score
        else:
            # Add new result
            merged_dict[doc_id] = {
                'score': (1 - alpha) * normalized_score,
                'content': result['content'],
                'metadata': {}  # BM25 results may not have metadata
            }
    
    # Sort by combined score (descending)
    merged_results = sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    
    logger.info(f"Hybrid merge completed: {len(merged_results)} total results")
    return merged_results

if __name__ == "__main__":
    # Test function
    semantic_results = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Document 1 content", "Document 2 content"]],
        "metadatas": [[{"source": "file1"}, {"source": "file2"}]]
    }
    
    bm25_results = [
        {"id": "doc1", "score": 2.5, "content": "Document 1 content"},
        {"id": "doc3", "score": 1.8, "content": "Document 3 content"}
    ]
    
    merged = hybrid_merge(semantic_results, bm25_results, alpha=0.7)
    print(f"Merged {len(merged)} results")
