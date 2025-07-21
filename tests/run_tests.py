"""
Test runner for PocketFlow PDF Chat
Runs all tests in the correct order
"""
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all tests"""
    print("Running PocketFlow PDF Chat Test Suite")
    print("=" * 60)
    
    # Change to tests directory
    original_dir = os.getcwd()
    tests_dir = os.path.dirname(__file__)
    os.chdir(tests_dir)
    
    try:
        # Run basic component tests
        print("\nRunning Basic Component Tests...")
        from test_simple import main as run_simple_tests
        run_simple_tests()
        
        # Run LLM integration tests
        print("\nRunning LLM Integration Tests...")
        from test_llm import main as run_llm_tests
        run_llm_tests()
        
        print("\nAll test suites completed!")
        
    except Exception as e:
        print(f"\nTest runner failed: {e}")
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    run_all_tests()
