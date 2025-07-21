"""
API configuration and environment checking utilities
"""
import os
import requests
import socket
from typing import Optional

def check_environment() -> bool:
    """Check if Ollama service is running and accessible"""
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Ollama connection failed: {str(e)}")
        return False

def check_serpapi_key() -> bool:
    """Check if SERPAPI key is configured"""
    serpapi_key = os.getenv("SERPAPI_KEY")
    return serpapi_key is not None and serpapi_key.strip() != ""

def is_port_available(port: int) -> bool:
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0

def get_available_port(start_port: int = 17995, max_attempts: int = 5) -> Optional[int]:
    """Get an available port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    return None
