"""
Text chunking utilities
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def create_text_chunks(text: str, chunk_size: int = 400, chunk_overlap: int = 40) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    if not text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

if __name__ == "__main__":
    # Test function
    test_text = "这是一个测试文本。" * 100
    chunks = create_text_chunks(test_text, chunk_size=50, chunk_overlap=10)
    print(f"Created {len(chunks)} chunks from test text")
