"""
Simple test script for core components without external dependencies
Tests individual utility functions and basic functionality
"""
import os
import sys
import numpy as np

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pdf_file_access():
    """Test if we can access the PDF file"""
    print("=== Testing PDF File Access ===")
    
    # Check PDF in tests directory
    test_dir = os.path.dirname(__file__)
    pdf_paths = [
        os.path.join(test_dir, "Progressive_web_app.pdf"),  # In tests directory
        "Progressive_web_app.pdf",  # In current directory
        "../Progressive_web_app.pdf",  # In case it's still in parent
    ]
    
    pdf_path = None
    for path in pdf_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            pdf_path = full_path
            print(f"[PASS] Found PDF at: {pdf_path}")
            print(f"   File size: {os.path.getsize(pdf_path)} bytes")
            break
    
    if not pdf_path:
        print("[FAIL] PDF file not found in expected locations")
        for path in pdf_paths:
            print(f"   Checked: {os.path.abspath(path)}")
        return None
    
    return pdf_path

def test_pdf_extraction():
    """Test PDF text extraction"""
    print("\n=== Testing PDF Text Extraction ===")
    
    try:
        from utils.pdf_processing import extract_text_from_pdf
        
        pdf_path = test_pdf_file_access()
        if not pdf_path:
            return False
        
        # Test extraction
        text = extract_text_from_pdf(pdf_path)
        
        if text and len(text.strip()) > 0:
            print(f"[PASS] Successfully extracted text")
            print(f"   Text length: {len(text)} characters")
            print(f"   First 100 chars: {text[:100]}...")
            return True
        else:
            print("[FAIL] No text extracted from PDF")
            return False
            
    except Exception as e:
        print(f"[FAIL] PDF extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_text_chunking():
    """Test text chunking functionality"""
    print("\n=== Testing Text Chunking ===")
    
    try:
        from utils.chunking import create_text_chunks
        
        # Create sample text
        sample_text = "This is a test document. " * 50  # Create a longer text
        
        chunks = create_text_chunks(
            text=sample_text,
            chunk_size=100,
            chunk_overlap=20
        )
        
        if chunks and len(chunks) > 0:
            print(f"[PASS] Successfully created chunks")
            print(f"   Number of chunks: {len(chunks)}")
            print(f"   First chunk length: {len(chunks[0])}")
            print(f"   First chunk: {chunks[0][:50]}...")
            return True
        else:
            print("[FAIL] No chunks created")
            return False
            
    except Exception as e:
        print(f"[FAIL] Text chunking failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_embeddings():
    """Test embedding generation"""
    print("\n=== Testing Embedding Generation ===")
    
    try:
        from utils.embeddings import get_embeddings
        
        # Test with simple text
        test_texts = ["This is a test sentence.", "Another test sentence for embedding."]
        
        print("   Loading embedding model...")
        embeddings = get_embeddings(test_texts)
        
        if embeddings and len(embeddings) == len(test_texts):
            print(f"[PASS] Successfully created embeddings")
            print(f"   Number of embeddings: {len(embeddings)}")
            print(f"   Embedding dimension: {embeddings[0].shape}")
            print(f"   Embedding type: {type(embeddings[0])}")
            return True
        else:
            print("[FAIL] Failed to create embeddings")
            return False
            
    except Exception as e:
        print(f"[FAIL] Embedding generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_indexing():
    """Test FAISS index creation"""
    print("\n=== Testing FAISS Indexing ===")
    
    try:
        import faiss
        from utils.embeddings import get_embeddings
        
        # Create sample embeddings
        test_texts = ["Sample text one.", "Sample text two.", "Sample text three."]
        print("   Creating embeddings for FAISS test...")
        embeddings = get_embeddings(test_texts)
        
        if not embeddings:
            print("[FAIL] Cannot test FAISS without embeddings")
            return False
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        if index.ntotal == len(embeddings):
            print(f"[PASS] Successfully created FAISS index")
            print(f"   Index dimension: {dimension}")
            print(f"   Number of vectors: {index.ntotal}")
            
            # Test search
            query_embedding = embeddings_array[0:1]  # Use first embedding as query
            distances, indices = index.search(query_embedding, k=2)
            print(f"   Search test - distances: {distances[0]}")
            print(f"   Search test - indices: {indices[0]}")
            return True
        else:
            print("[FAIL] FAISS index creation failed")
            return False
            
    except Exception as e:
        print(f"[FAIL] FAISS indexing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_bm25_indexing():
    """Test BM25 index creation"""
    print("\n=== Testing BM25 Indexing ===")
    
    try:
        from utils.bm25_manager import BM25IndexManager
        
        # Create sample documents
        test_docs = [
            "This is a test document about machine learning.",
            "Another document discussing artificial intelligence.",
            "A third document about natural language processing."
        ]
        
        doc_ids = ["doc1", "doc2", "doc3"]
        
        # Create BM25 manager
        bm25_manager = BM25IndexManager()
        bm25_manager.build_index(test_docs, doc_ids)
        
        # Test search
        query = "machine learning"
        results = bm25_manager.search(query, top_k=2)
        
        if results and len(results) > 0:
            print(f"[PASS] Successfully created BM25 index")
            print(f"   Number of documents: {len(test_docs)}")
            print(f"   Search results: {len(results)}")
            print(f"   Top result score: {results[0]['score']:.4f}")
            print(f"   Top result content: {results[0]['content'][:50]}...")
            return True
        else:
            print("[FAIL] BM25 search returned no results")
            return False
            
    except Exception as e:
        print(f"[FAIL] BM25 indexing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_search():
    """Test hybrid search functionality"""
    print("\n=== Testing Hybrid Search ===")
    
    try:
        from utils.hybrid_search import hybrid_merge
        
        # Create mock semantic results
        semantic_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[{"source": "file1"}, {"source": "file2"}]]
        }
        
        # Create mock BM25 results
        bm25_results = [
            {"id": "doc1", "score": 2.5, "content": "Document 1 content"},
            {"id": "doc3", "score": 1.8, "content": "Document 3 content"}
        ]
        
        merged = hybrid_merge(semantic_results, bm25_results, alpha=0.7)
        
        if merged and len(merged) > 0:
            print(f"[PASS] Successfully merged search results")
            print(f"   Number of merged results: {len(merged)}")
            print(f"   Top result ID: {merged[0][0]}")
            print(f"   Top result score: {merged[0][1]['score']:.4f}")
            return True
        else:
            print("[FAIL] No merged results")
            return False
            
    except Exception as e:
        print(f"[FAIL] Hybrid search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """Test a complete pipeline from PDF to search"""
    print("\n=== Testing Complete Pipeline ===")
    
    try:
        pdf_path = test_pdf_file_access()
        if not pdf_path:
            return False
        
        # 1. Extract text from PDF
        from utils.pdf_processing import extract_text_from_pdf
        text = extract_text_from_pdf(pdf_path)
        print(f"   ✓ Extracted {len(text)} characters from PDF")
        
        # 2. Chunk the text
        from utils.chunking import create_text_chunks
        chunks = create_text_chunks(
            text=text,
            chunk_size=200,
            chunk_overlap=20
        )
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Create simple metadata for each chunk
        metadatas = [{"filename": "Progressive_web_app.pdf", "chunk_id": i} for i in range(len(chunks))]
        doc_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # 3. Create embeddings
        from utils.embeddings import get_embeddings
        embeddings = get_embeddings(chunks[:5])  # Test with first 5 chunks
        print(f"   ✓ Created {len(embeddings)} embeddings")
        
        # 4. Create FAISS index
        import faiss
        embeddings_array = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        print(f"   ✓ Created FAISS index with {index.ntotal} vectors")
        
        # 5. Create BM25 index
        from utils.bm25_manager import BM25IndexManager
        bm25_manager = BM25IndexManager()
        bm25_manager.build_index(chunks[:5], doc_ids[:5])
        print(f"   ✓ Created BM25 index")
        
        # 6. Test search
        query = "web application"
        
        # Semantic search
        query_embeddings = get_embeddings([query])
        query_array = np.array(query_embeddings, dtype=np.float32)
        distances, indices = index.search(query_array, k=3)
        
        semantic_results = {
            "ids": [indices[0].tolist()],
            "documents": [[chunks[i] for i in indices[0]]],
            "metadatas": [[metadatas[i] for i in indices[0]]]
        }
        
        # BM25 search
        bm25_results = bm25_manager.search(query, top_k=3)
        
        # Hybrid search
        from utils.hybrid_search import hybrid_merge
        merged_results = hybrid_merge(semantic_results, bm25_results)
        
        print(f"   ✓ Found {len(merged_results)} hybrid search results")
        print(f"   ✓ Top result score: {merged_results[0][1]['score']:.4f}")
        
        print(f"[PASS] Complete pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Complete pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("[TEST] Testing PocketFlow PDF Chat - Core Components")
    print("=" * 60)
    
    tests = [
        ("PDF Text Extraction", test_pdf_extraction),
        ("Text Chunking", test_text_chunking),
        ("Embedding Generation", test_embeddings),
        ("FAISS Indexing", test_faiss_indexing),
        ("BM25 Indexing", test_bm25_indexing),
        ("Hybrid Search", test_hybrid_search),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"[FAIL] {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! The core components are working correctly.")
        print("\n[INFO] Next steps:")
        print("   - Configure API keys (OLLAMA for LLM, SERPAPI_KEY for web search)")
        print("   - Run: python main.py")
        print("   - Upload a PDF and ask questions!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
