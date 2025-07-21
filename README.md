# PocketFlow PDF Chat

A production-ready RAG (Retrieval-Augmented Generation) system built with PocketFlow for PDF document question answering.

## Features

- **PDF Processing**: Extract and chunk text from multiple PDF documents  
- **Hybrid Search**: Combines semantic (FAISS) and keyword (BM25) search
- **Multiple LLM Support**: Works with local Ollama and cloud SiliconFlow models
- **Web Search Integration**: Optional web search for additional context
- **Modern UI**: Clean Gradio web interface
- **Chinese Support**: Full Chinese text processing with jieba tokenization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Ollama (Recommended)

```bash
# Install Ollama from https://ollama.ai/
ollama serve
ollama pull deepseek-r1:1.5b
```

### 3. Run the Application

```bash
python main.py
```

Open http://localhost:17995 in your browser.

### 4. Use the System

1. Upload one or more PDF files
2. Ask questions about the content  
3. Get AI-powered answers with source citations

## Configuration

### Environment Variables

- `SILICONFLOW_API_KEY`: For cloud LLM service (optional)
- `SERPAPI_KEY`: For web search functionality (optional)

### Model Settings

- **Local**: Uses Ollama with deepseek-r1:1.5b model
- **Cloud**: Uses SiliconFlow API
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local)

## Architecture

Built using PocketFlow's modular design:

- **Offline Flow**: PDF processing → Text chunking → Embedding generation → Index creation
- **Online Flow**: Query embedding → Hybrid retrieval → Reranking → Answer generation
- **Shared Store**: Efficient data communication between nodes
- **Fault Tolerance**: Built-in retry mechanisms and graceful fallbacks

## Project Structure

```
├── main.py              # Gradio web interface
├── flow.py              # PocketFlow workflow definitions  
├── nodes.py             # Processing nodes (13 total)
├── pocketflow.py        # PocketFlow framework (100 lines)
├── start.bat            # Windows startup script
├── requirements.txt     # Python dependencies
├── utils/               # Utility modules
│   ├── __init__.py          # Package initialization
│   ├── api_config.py        # Environment configuration
│   ├── pdf_processing.py    # PDF text extraction
│   ├── chunking.py          # Text chunking
│   ├── embeddings.py        # Embedding generation
│   ├── hybrid_search.py     # Hybrid search logic
│   ├── llm_client.py        # LLM API clients
│   ├── bm25_manager.py      # BM25 keyword search
│   ├── reranking.py         # Result reranking
│   └── web_search.py        # Web search integration
└── tests/               # Test suite
    ├── __init__.py          # Test package initialization
    ├── run_tests.py         # Test runner script
    ├── test_simple.py       # Basic functionality tests
    ├── test_llm.py          # LLM integration tests
    └── Progressive_web_app.pdf  # Sample PDF for testing
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python tests/run_tests.py

# Or run individual test suites
python tests/test_simple.py    # Basic functionality tests
python tests/test_llm.py       # LLM integration tests
```

### Test Coverage

**Basic Functionality Tests (`test_simple.py`)**:
- PDF text extraction
- Text chunking  
- Embedding generation
- FAISS indexing
- BM25 search
- Hybrid search
- Complete pipeline

**LLM Integration Tests (`test_llm.py`)**:
- Ollama connection verification
- Basic LLM call functionality
- Complete RAG pipeline with answer generation

## Performance

- **PDF Processing**: ~1 second for 20KB+ documents
- **Embedding Generation**: 384-dimensional vectors using sentence-transformers
- **Search**: Sub-second retrieval from indexed content
- **Answer Generation**: 2-3 second response time

## Development

### Project Organization

The project follows a clean, modular structure:

- **Core Application**: `main.py`, `flow.py`, `nodes.py`, `pocketflow.py`
- **Utilities**: Organized in `utils/` package with specific functionality
- **Tests**: Comprehensive test suite in `tests/` directory
- **Configuration**: Environment settings and startup scripts

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test suites
python tests/test_simple.py    # Basic functionality
python tests/test_llm.py       # LLM integration
```

### Code Quality

- **Modular Design**: Each component has a single responsibility
- **Error Handling**: Comprehensive error handling with fallbacks
- **Logging**: Detailed logging for debugging and monitoring
- **Type Safety**: Clear input/output contracts for all functions

## Requirements

- Python 3.8+
- 4GB+ RAM (for embedding models)
- Ollama service (for local LLM)
- Optional: SERPAPI key for web search

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Install required model
   ollama pull deepseek-r1:1.5b
   ```

2. **Port Already in Use**
   - The application automatically tries ports 17995-17999
   - Check `main.py` for port configuration

3. **Memory Issues**
   - Ensure at least 4GB RAM available
   - Close other memory-intensive applications

## License

MIT License - See original PocketFlow project for details.

## Credits

- Built with [PocketFlow](https://github.com/the-pocket/PocketFlow) - 100-line LLM framework
- Majority of the code came from [Local_Pdf_Chat_RAG](https://github.com/weiwill88/Local_Pdf_Chat_RAG)
- Converted from a 2,597-line monolithic system to modular architecture  
- Demonstrates PocketFlow's RAG design pattern in production