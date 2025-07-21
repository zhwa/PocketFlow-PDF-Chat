"""
Node definitions for PocketFlow-based PDF Chat RAG system
Contains all nodes for document processing, indexing, retrieval, and answer generation
"""
import os
import numpy as np
import faiss
import time
import logging
from io import StringIO
from datetime import datetime

from pocketflow import Node, BatchNode
from utils.pdf_processing import extract_text_from_pdf
from utils.chunking import create_text_chunks  
from utils.embeddings import get_embedding_model, get_embeddings
from utils.bm25_manager import BM25IndexManager
from utils.web_search import search_web
from utils.llm_client import call_llm
from utils.reranking import rerank_with_cross_encoder, rerank_with_llm
from utils.hybrid_search import hybrid_merge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################
# OFFLINE FLOW NODES (Document Processing)
#########################################

class ExtractTextNode(BatchNode):
    """Extract text from uploaded PDF files"""
    
    def __init__(self, max_retries=2, wait=1):
        super().__init__(max_retries=max_retries, wait=wait)
    
    def prep(self, shared):
        """Get uploaded files from shared store"""
        files = shared.get("files", [])
        if not files:
            return []
        
        logger.info(f"Processing {len(files)} PDF files")
        return files
    
    def exec(self, file):
        """Extract text from a single PDF file"""
        if hasattr(file, 'name'):
            file_path = file.name
        else:
            file_path = str(file)
        
        logger.info(f"Extracting text from: {os.path.basename(file_path)}")
        text = extract_text_from_pdf(file_path)
        
        return {
            "filename": os.path.basename(file_path),
            "text": text,
            "file_path": file_path
        }
    
    def exec_fallback(self, file, exc):
        """Fallback when extraction fails"""
        logger.warning(f"Failed to extract text from {file} after retries: {str(exc)}")
        return {
            "filename": str(file),
            "text": "",
            "error": str(exc)
        }
    
    def post(self, shared, prep_res, exec_res_list):
        """Store extracted texts in shared store"""
        extracted_texts = []
        for result in exec_res_list:
            if result.get("text"):
                extracted_texts.append(result)
            else:
                logger.warning(f"No text extracted from {result.get('filename', 'unknown file')}")
        
        shared["extracted_texts"] = extracted_texts
        logger.info(f"✅ Successfully extracted text from {len(extracted_texts)} files")
        return "default"

class ChunkDocumentsNode(Node):
    """Split documents into smaller chunks for processing"""
    
    def prep(self, shared):
        """Get extracted texts from shared store"""
        return shared.get("extracted_texts", [])
    
    def exec(self, extracted_texts):
        """Split all texts into chunks"""
        if not extracted_texts:
            return [], [], []
        
        # Note: shared is not available in exec(), so we use default values
        chunk_size = 400
        chunk_overlap = 40
        
        all_chunks = []
        all_metadatas = []
        all_chunk_ids = []
        
        for idx, text_data in enumerate(extracted_texts):
            filename = text_data["filename"]
            text = text_data["text"]
            
            if not text.strip():
                continue
            
            # Create chunks
            chunks = create_text_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Create metadata and IDs for each chunk
            doc_id = f"doc_{int(time.time())}_{idx}"
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                metadata = {
                    "source": filename,
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                }
                
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_chunk_ids.append(chunk_id)
        
        logger.info(f"✅ Created {len(all_chunks)} chunks from {len(extracted_texts)} documents")
        return all_chunks, all_metadatas, all_chunk_ids
    
    def post(self, shared, prep_res, exec_res):
        """Store chunks in shared store"""
        chunks, metadatas, chunk_ids = exec_res
        shared["chunks"] = chunks
        shared["chunk_metadatas"] = metadatas  
        shared["chunk_ids"] = chunk_ids
        return "default"

class EmbedDocumentsNode(BatchNode):
    """Create embeddings for document chunks"""
    
    def __init__(self, max_retries=3, wait=1):
        super().__init__(max_retries=max_retries, wait=wait)
    
    def prep(self, shared):
        """Get chunks from shared store"""
        return shared.get("chunks", [])
    
    def exec(self, chunk):
        """Create embedding for a single chunk"""
        embedding = get_embeddings([chunk])[0]
        return embedding
    
    def exec_fallback(self, chunk, exc):
        """Fallback when embedding fails"""
        logger.warning(f"Failed to create embedding for chunk after retries: {str(exc)}")
        # Return zero vector as fallback
        return np.zeros(384, dtype=np.float32)  # all-MiniLM-L6-v2 dimension
    
    def post(self, shared, prep_res, exec_res_list):
        """Store embeddings in shared store"""
        embeddings = np.array(exec_res_list, dtype=np.float32)
        shared["embeddings"] = embeddings
        logger.info(f"✅ Created {len(embeddings)} embeddings")
        return "default"

class CreateFAISSIndexNode(Node):
    """Create FAISS index from embeddings"""
    
    def prep(self, shared):
        """Get embeddings and metadata from shared store"""
        embeddings = shared.get("embeddings")
        chunk_ids = shared.get("chunk_ids", [])
        chunks = shared.get("chunks", [])
        metadatas = shared.get("chunk_metadatas", [])
        
        return embeddings, chunk_ids, chunks, metadatas
    
    def exec(self, inputs):
        """Create FAISS index"""
        embeddings, chunk_ids, chunks, metadatas = inputs
        
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings to index")
            return None, {}, {}, []
        
        try:
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            # Create mapping dictionaries
            contents_map = {chunk_id: chunk for chunk_id, chunk in zip(chunk_ids, chunks)}
            metadatas_map = {chunk_id: meta for chunk_id, meta in zip(chunk_ids, metadatas)}
            
            logger.info(f"✅ FAISS index created with {index.ntotal} vectors")
            return index, contents_map, metadatas_map, chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
            return None, {}, {}, []
    
    def post(self, shared, prep_res, exec_res):
        """Store FAISS index in shared store"""
        index, contents_map, metadatas_map, id_order = exec_res
        
        shared["faiss_index"] = index
        shared["faiss_contents_map"] = contents_map
        shared["faiss_metadatas_map"] = metadatas_map
        shared["faiss_id_order"] = id_order
        
        return "default"

class CreateBM25IndexNode(Node):
    """Create BM25 index for keyword-based search"""
    
    def prep(self, shared):
        """Get chunks and IDs from shared store"""
        chunks = shared.get("chunks", [])
        chunk_ids = shared.get("chunk_ids", [])
        return chunks, chunk_ids
    
    def exec(self, inputs):
        """Create BM25 index"""
        chunks, chunk_ids = inputs
        
        if not chunks:
            logger.warning("No chunks to index for BM25")
            return None
        
        try:
            bm25_manager = BM25IndexManager()
            bm25_manager.build_index(chunks, chunk_ids)
            logger.info(f"✅ BM25 index created with {len(chunks)} documents")
            return bm25_manager
        except Exception as e:
            logger.error(f"Failed to create BM25 index: {str(e)}")
            return None
    
    def post(self, shared, prep_res, exec_res):
        """Store BM25 manager in shared store"""
        shared["bm25_manager"] = exec_res
        return "default"

#########################################
# ONLINE FLOW NODES (Query Processing)
#########################################

class EmbedQueryNode(Node):
    """Create embedding for user query"""
    
    def prep(self, shared):
        """Get query from shared store"""
        return shared.get("query", "")
    
    def exec(self, query):
        """Create query embedding"""
        if not query.strip():
            return None
        
        try:
            query_embedding = get_embeddings([query])[0]
            logger.info(f"✅ Query embedded: {query[:50]}...")
            return query_embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            return None
    
    def post(self, shared, prep_res, exec_res):
        """Store query embedding in shared store"""
        shared["query_embedding"] = exec_res
        return "default"

class SemanticRetrievalNode(Node):
    """Perform semantic retrieval using FAISS"""
    
    def prep(self, shared):
        """Get query embedding and FAISS components"""
        query_embedding = shared.get("query_embedding")
        faiss_index = shared.get("faiss_index")
        contents_map = shared.get("faiss_contents_map", {})
        metadatas_map = shared.get("faiss_metadatas_map", {})
        id_order = shared.get("faiss_id_order", [])
        
        config = shared.get("config", {})
        top_k = config.get("top_k", 10)
        
        return query_embedding, faiss_index, contents_map, metadatas_map, id_order, top_k
    
    def exec(self, inputs):
        """Perform semantic search"""
        query_embedding, faiss_index, contents_map, metadatas_map, id_order, top_k = inputs
        
        if query_embedding is None or faiss_index is None:
            logger.warning("Missing query embedding or FAISS index for semantic retrieval")
            return []
        
        try:
            # Search FAISS index
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = faiss_index.search(query_vector, top_k)
            
            # Convert FAISS indices to document results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(id_order):
                    chunk_id = id_order[idx]
                    content = contents_map.get(chunk_id, "")
                    metadata = metadatas_map.get(chunk_id, {})
                    
                    results.append({
                        "id": chunk_id,
                        "content": content,
                        "metadata": metadata,
                        "score": float(1.0 / (1.0 + distance))  # Convert distance to similarity
                    })
            
            logger.info(f"✅ Semantic retrieval found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {str(e)}")
            return []
    
    def post(self, shared, prep_res, exec_res):
        """Store semantic results"""
        shared["semantic_results"] = exec_res
        return "default"

class BM25RetrievalNode(Node):
    """Perform keyword-based retrieval using BM25"""
    
    def prep(self, shared):
        """Get query and BM25 manager"""
        query = shared.get("query", "")
        bm25_manager = shared.get("bm25_manager")
        
        config = shared.get("config", {})
        top_k = config.get("top_k", 10)
        
        return query, bm25_manager, top_k
    
    def exec(self, inputs):
        """Perform BM25 search"""
        query, bm25_manager, top_k = inputs
        
        if not query.strip() or bm25_manager is None:
            logger.warning("Missing query or BM25 manager for keyword retrieval")
            return []
        
        try:
            results = bm25_manager.search(query, top_k=top_k)
            logger.info(f"✅ BM25 retrieval found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {str(e)}")
            return []
    
    def post(self, shared, prep_res, exec_res):
        """Store BM25 results"""
        shared["bm25_results"] = exec_res
        return "default"

class WebSearchNode(Node):
    """Perform web search if enabled"""
    
    def prep(self, shared):
        """Get query and web search settings"""
        query = shared.get("query", "")
        enable_web_search = shared.get("enable_web_search", False)
        return query, enable_web_search
    
    def exec(self, inputs):
        """Perform web search"""
        query, enable_web_search = inputs
        
        if not enable_web_search or not query.strip():
            return []
        
        try:
            results = search_web(query, num_results=5)
            logger.info(f"✅ Web search found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []
    
    def post(self, shared, prep_res, exec_res):
        """Store web search results"""
        shared["web_results"] = exec_res
        return "default"

class HybridMergeNode(Node):
    """Merge semantic and BM25 search results"""
    
    def prep(self, shared):
        """Get semantic and BM25 results"""
        semantic_results = shared.get("semantic_results", [])
        bm25_results = shared.get("bm25_results", [])
        
        config = shared.get("config", {})
        alpha = config.get("hybrid_alpha", 0.7)
        
        return semantic_results, bm25_results, alpha
    
    def exec(self, inputs):
        """Merge results using hybrid scoring"""
        semantic_results, bm25_results, alpha = inputs
        
        try:
            # Convert semantic results to expected format
            semantic_formatted = {
                "ids": [[r["id"] for r in semantic_results]],
                "documents": [[r["content"] for r in semantic_results]],
                "metadatas": [[r["metadata"] for r in semantic_results]]
            }
            
            merged_results = hybrid_merge(semantic_formatted, bm25_results, alpha=alpha)
            logger.info(f"✅ Hybrid merge created {len(merged_results)} results")
            return merged_results
            
        except Exception as e:
            logger.error(f"Hybrid merge failed: {str(e)}")
            # Fallback to semantic results
            return [(r["id"], {"content": r["content"], "metadata": r["metadata"], "score": r["score"]}) 
                    for r in semantic_results]
    
    def post(self, shared, prep_res, exec_res):
        """Store hybrid results"""
        shared["hybrid_results"] = exec_res
        return "default"

class RerankResultsNode(Node):
    """Rerank results using cross-encoder or LLM"""
    
    def prep(self, shared):
        """Get query and hybrid results"""
        query = shared.get("query", "")
        hybrid_results = shared.get("hybrid_results", [])
        
        config = shared.get("config", {})
        rerank_method = config.get("rerank_method", "cross_encoder")
        final_k = config.get("final_k", 5)
        
        return query, hybrid_results, rerank_method, final_k
    
    def exec(self, inputs):
        """Rerank results"""
        query, hybrid_results, rerank_method, final_k = inputs
        
        if not hybrid_results:
            return []
        
        try:
            # Extract data for reranking
            doc_ids = [item[0] for item in hybrid_results]
            docs = [item[1]["content"] for item in hybrid_results]
            metadatas = [item[1]["metadata"] for item in hybrid_results]
            
            # Perform reranking
            if rerank_method == "cross_encoder":
                reranked = rerank_with_cross_encoder(query, docs, doc_ids, metadatas, top_k=final_k)
            elif rerank_method == "llm":
                reranked = rerank_with_llm(query, docs, doc_ids, metadatas, top_k=final_k)
            else:
                # No reranking, just take top results
                reranked = hybrid_results[:final_k]
            
            logger.info(f"✅ Reranking completed, {len(reranked)} final results")
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Fallback to original results
            return hybrid_results[:final_k]
    
    def post(self, shared, prep_res, exec_res):
        """Store reranked results"""
        shared["reranked_results"] = exec_res
        return "default"

class RecursiveRetrievalNode(Node):
    """Perform recursive retrieval with multiple iterations"""
    
    def prep(self, shared):
        """Get all necessary data for recursive retrieval"""
        query = shared.get("query", "")
        config = shared.get("config", {})
        max_iterations = config.get("max_iterations", 3)
        enable_web_search = shared.get("enable_web_search", False)
        model_choice = shared.get("model_choice", "ollama")
        
        return query, max_iterations, enable_web_search, model_choice, shared
    
    def exec(self, inputs):
        """Perform recursive retrieval"""
        initial_query, max_iterations, enable_web_search, model_choice, shared = inputs
        
        query = initial_query
        all_contexts = []
        all_doc_ids = []
        all_metadata = []
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Recursive retrieval iteration {iteration + 1}/{max_iterations}")
            logger.info(f"Current query: {query}")
            
            # Update query in shared for sub-retrievals
            shared["query"] = query
            
            # Perform retrieval steps
            try:
                # Semantic retrieval
                semantic_node = SemanticRetrievalNode()
                semantic_node.run(shared)
                
                # BM25 retrieval  
                bm25_node = BM25RetrievalNode()
                bm25_node.run(shared)
                
                # Web search (if enabled)
                if enable_web_search:
                    web_node = WebSearchNode()
                    web_node.run(shared)
                
                # Hybrid merge
                hybrid_node = HybridMergeNode()
                hybrid_node.run(shared)
                
                # Reranking
                rerank_node = RerankResultsNode()
                rerank_node.run(shared)
                
                # Get results
                reranked_results = shared.get("reranked_results", [])
                
                # Add to cumulative results
                for doc_id, result_data in reranked_results:
                    if doc_id not in all_doc_ids:
                        all_doc_ids.append(doc_id)
                        all_contexts.append(result_data["content"])
                        all_metadata.append(result_data["metadata"])
                
                # Check if we should continue (simplified logic)
                if iteration == max_iterations - 1:
                    break
                
                # Generate next query using LLM (simplified)
                if len(all_contexts) > 0:
                    current_summary = "\n".join(all_contexts[:3])
                    
                    next_query_prompt = f"""根据原始问题和已检索信息，决定是否需要更具体的查询。
                    
原始问题: {initial_query}
已检索信息: {current_summary}

如果信息足够，回复"不需要进一步查询"。否则，提供一个简短的新查询词。
输出:"""
                    
                    try:
                        next_query_response = call_llm(next_query_prompt, model_choice=model_choice)
                        
                        if "不需要" in next_query_response:
                            logger.info("LLM判断不需要进一步查询")
                            break
                        
                        if len(next_query_response.strip()) > 100:
                            logger.warning("LLM响应过长，停止迭代")
                            break
                        
                        query = next_query_response.strip()
                        logger.info(f"生成新查询: {query}")
                        
                    except Exception as e:
                        logger.error(f"生成新查询失败: {str(e)}")
                        break
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Iteration {iteration + 1} failed: {str(e)}")
                break
        
        logger.info(f"✅ Recursive retrieval completed with {len(all_contexts)} total contexts")
        return all_contexts, all_doc_ids, all_metadata
    
    def post(self, shared, prep_res, exec_res):
        """Store final retrieval results"""
        all_contexts, all_doc_ids, all_metadata = exec_res
        
        shared["final_contexts"] = all_contexts
        shared["final_doc_ids"] = all_doc_ids 
        shared["final_metadata"] = all_metadata
        
        # Also restore original query
        original_query = prep_res[0]
        shared["query"] = original_query
        
        return "default"

class GenerateAnswerNode(Node):
    """Generate final answer using LLM"""
    
    def prep(self, shared):
        """Get query, contexts, and settings"""
        query = shared.get("query", "")
        
        # Get reranked results instead of final_contexts
        reranked_results = shared.get("reranked_results", [])
        contexts = []
        doc_ids = []
        metadata = []
        
        # Extract data from reranked results
        for doc_id, result_data in reranked_results:
            doc_ids.append(doc_id)
            contexts.append(result_data["content"])
            metadata.append(result_data["metadata"])
        
        web_results = shared.get("web_results", [])
        enable_web_search = shared.get("enable_web_search", False)
        model_choice = shared.get("model_choice", "ollama")
        
        return query, contexts, doc_ids, metadata, web_results, enable_web_search, model_choice
    
    def exec(self, inputs):
        """Generate answer using LLM"""
        query, contexts, doc_ids, metadata, web_results, enable_web_search, model_choice = inputs
        
        if not query.strip():
            return "请输入有效问题", []
        
        try:
            # Build context with sources
            context_with_sources = []
            sources = []
            
            # Add document contexts
            for ctx, doc_id, meta in zip(contexts, doc_ids, metadata):
                source = meta.get("source", "未知来源")
                context_with_sources.append(f"[本地文档: {source}]\n{ctx}")
                sources.append({
                    "type": "document",
                    "source": source,
                    "content": ctx[:100] + "..." if len(ctx) > 100 else ctx
                })
            
            # Add web results if enabled
            if enable_web_search and web_results:
                for result in web_results[:3]:  # Limit web results
                    title = result.get("title", "")
                    url = result.get("url", "")
                    snippet = result.get("snippet", "")
                    
                    if snippet:
                        context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{snippet}")
                        sources.append({
                            "type": "web",
                            "title": title,
                            "url": url,
                            "content": snippet
                        })
            
            # Combine contexts
            context = "\n\n".join(context_with_sources)
            
            # Create prompt
            context_type = "本地文档和网络搜索结果" if enable_web_search and web_results else "本地文档"
            
            prompt = f"""作为专业的问答助手，请基于以下{context_type}回答用户问题。

参考内容：
{context}

用户问题：{query}

回答原则：
1. 仅基于提供的参考内容回答，不要使用自己的知识
2. 如果信息不足，请明确说明
3. 回答要准确、完整、有条理
4. 请用中文回答
5. 在回答末尾标注信息来源

请现在开始回答："""
            
            # Generate answer
            answer = call_llm(prompt, model_choice=model_choice)
            
            logger.info(f"✅ Answer generated for query: {query[:50]}...")
            return answer, sources
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return f"生成回答时出错: {str(e)}", []
    
    def post(self, shared, prep_res, exec_res):
        """Store final answer and sources"""
        answer, sources = exec_res
        shared["answer"] = answer
        shared["sources"] = sources
        
        logger.info("✅ RAG pipeline completed successfully")
        return "default"
