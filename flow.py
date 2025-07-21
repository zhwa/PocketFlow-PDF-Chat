"""
Flow definitions for PocketFlow-based PDF Chat RAG system
Defines the offline (document processing) and online (query answering) flows
"""
from pocketflow import Flow
from nodes import (
    # Offline flow nodes
    ExtractTextNode, ChunkDocumentsNode, EmbedDocumentsNode, 
    CreateFAISSIndexNode, CreateBM25IndexNode,
    
    # Online flow nodes  
    EmbedQueryNode, SemanticRetrievalNode, BM25RetrievalNode,
    HybridMergeNode, WebSearchNode, RerankResultsNode,
    RecursiveRetrievalNode, GenerateAnswerNode
)

def create_offline_flow():
    """
    Create the offline flow for document processing and indexing
    
    Flow: Extract Text -> Chunk Documents -> Embed Documents -> Create FAISS Index -> Create BM25 Index
    """
    # Create nodes
    extract_text = ExtractTextNode()
    chunk_docs = ChunkDocumentsNode()
    embed_docs = EmbedDocumentsNode()
    create_faiss = CreateFAISSIndexNode()
    create_bm25 = CreateBM25IndexNode()
    
    # Connect nodes in sequence
    extract_text >> chunk_docs >> embed_docs >> create_faiss >> create_bm25
    
    # Create and return flow
    offline_flow = Flow(start=extract_text)
    return offline_flow

def create_online_flow():
    """
    Create the online flow for query processing and answer generation
    
    Simple flow: Embed Query -> Semantic + BM25 + Web Search -> Hybrid Merge -> Rerank -> Generate Answer
    """
    # Create nodes
    embed_query = EmbedQueryNode()
    semantic_retrieval = SemanticRetrievalNode()
    bm25_retrieval = BM25RetrievalNode()
    web_search = WebSearchNode()
    hybrid_merge = HybridMergeNode()
    rerank_results = RerankResultsNode()
    generate_answer = GenerateAnswerNode()
    
    # Connect main flow (simplified, non-recursive)
    embed_query >> semantic_retrieval >> bm25_retrieval >> web_search >> hybrid_merge >> rerank_results >> generate_answer
    
    # Create and return flow
    online_flow = Flow(start=embed_query)
    return online_flow

def create_retrieval_subflow():
    """
    Create the retrieval subflow used within recursive retrieval
    
    Flow: Semantic Retrieval + BM25 Retrieval + Web Search -> Hybrid Merge -> Rerank
    """
    # Create nodes
    semantic_retrieval = SemanticRetrievalNode()
    bm25_retrieval = BM25RetrievalNode()
    web_search = WebSearchNode()
    hybrid_merge = HybridMergeNode()
    rerank_results = RerankResultsNode()
    
    # This is a more complex flow with parallel retrieval
    # For now, we'll handle the parallel execution within the RecursiveRetrievalNode
    # In a more advanced implementation, we could use AsyncParallelFlow
    
    # Connect in sequence (actual parallel execution handled in RecursiveRetrievalNode)
    semantic_retrieval >> bm25_retrieval >> web_search >> hybrid_merge >> rerank_results
    
    # Create and return flow
    retrieval_flow = Flow(start=semantic_retrieval)
    return retrieval_flow
