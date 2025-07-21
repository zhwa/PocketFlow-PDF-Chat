"""
Test script for LLM functionality with Ollama
Tests the complete RAG pipeline including LLM answer generation
"""
import os
import sys

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("=== Testing Ollama Connection ===")
    
    try:
        from utils.api_config import check_environment
        
        is_running = check_environment()
        if is_running:
            print("[PASS] Ollama service is running")
            return True
        else:
            print("[FAIL] Ollama service is not running")
            return False
    except Exception as e:
        print(f"[FAIL] Failed to check Ollama: {e}")
        return False

def test_basic_llm_call():
    """Test basic LLM call with Ollama"""
    print("\n=== Testing Basic LLM Call ===")
    
    try:
        from utils.llm_client import call_llm
        
        # Simple test prompt
        prompt = "请用中文回答：什么是人工智能？请简短回答。"
        
        print("   Sending request to Ollama...")
        response = call_llm(prompt, model_choice="ollama", timeout=60)
        
        if response and len(response.strip()) > 0:
            print("[PASS] LLM call successful!")
            print(f"   Response length: {len(response)} characters")
            print(f"   Response preview: {response[:150]}...")
            return True
        else:
            print("[FAIL] Empty response from LLM")
            return False
            
    except Exception as e:
        print(f"[FAIL] LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_with_llm():
    """Test complete RAG pipeline with LLM"""
    print("\n=== Testing Complete RAG Pipeline with LLM ===")
    
    try:
        # Import required modules
        from flow import create_offline_flow, create_online_flow
        from main import create_shared_store
        import tempfile
        import shutil
        
        # Check if PDF exists in tests directory
        pdf_path = os.path.join(os.path.dirname(__file__), "Progressive_web_app.pdf")
        if not os.path.exists(pdf_path):
            print("[FAIL] Test PDF not found")
            return False
        
        print("   Setting up test data...")
        
        # Create a mock file object for testing
        class MockFile:
            def __init__(self, path):
                self.name = path
        
        # Create shared store
        shared = create_shared_store()
        shared["files"] = [MockFile(pdf_path)]
        shared["query"] = "什么是Progressive Web App？请简短介绍。"
        shared["enable_web_search"] = False
        shared["model_choice"] = "ollama"
        
        print("   Running offline flow (document processing)...")
        offline_flow = create_offline_flow()
        offline_flow.run(shared)
        
        # Check if processing was successful
        chunks = shared.get("chunks", [])
        embeddings = shared.get("embeddings")
        faiss_index = shared.get("faiss_index")
        bm25_manager = shared.get("bm25_manager")
        
        print(f"   ✓ Processed {len(chunks)} chunks")
        print(f"   ✓ Created {len(embeddings) if embeddings is not None else 0} embeddings")
        print(f"   ✓ FAISS index: {'Created' if faiss_index is not None else 'Failed'}")
        print(f"   ✓ BM25 index: {'Created' if bm25_manager is not None else 'Failed'}")
        
        print("   Running online flow (query processing and answer generation)...")
        online_flow = create_online_flow()
        online_flow.run(shared)
        
        # Check results
        answer = shared.get("answer", "")
        sources = shared.get("sources", [])
        
        if answer and len(answer.strip()) > 0:
            print("[PASS] Complete RAG pipeline successful!")
            print(f"   Query: {shared['query']}")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Number of sources: {len(sources)}")
            print(f"   Answer preview: {answer[:200]}...")
            return True
        else:
            print("[FAIL] No answer generated")
            return False
            
    except Exception as e:
        print(f"[FAIL] RAG pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all LLM tests"""
    print("[TEST] Testing PocketFlow PDF Chat - LLM Integration")
    print("=" * 60)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Basic LLM Call", test_basic_llm_call),
        ("Complete RAG Pipeline", test_rag_with_llm)
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
    print("🏁 LLM Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} LLM tests passed")
    
    if passed == total:
        print("[SUCCESS] All LLM tests passed! The system is ready for use.")
        print("\n[INFO] The web interface is available at:")
        print("   http://localhost:17995")
        print("\n[NOTE] How to use:")
        print("   1. Upload one or more PDF files")
        print("   2. Ask questions about the content")
        print("   3. Get AI-powered answers with source citations")
    else:
        print("⚠️  Some LLM tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
