"""
Web search utilities using SerpAPI
"""
import os
import requests
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web using SerpAPI"""
    serpapi_key = os.getenv("SERPAPI_KEY")
    
    if not serpapi_key or not serpapi_key.strip():
        logger.info("SERPAPI_KEY not configured, skipping web search")
        return []
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key,
        "num": num_results,
        "hl": "zh-CN",
        "gl": "cn"
    }
    
    response = requests.get("https://serpapi.com/search", params=params, timeout=15)
    response.raise_for_status()
    search_data = response.json()
    
    return _parse_serpapi_results(search_data)

def _parse_serpapi_results(data: Dict) -> List[Dict[str, Any]]:
    """Parse SerpAPI response data"""
    results = []
    
    if "organic_results" in data:
        for item in data["organic_results"]:
            result = {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "timestamp": item.get("date", "")
            }
            results.append(result)
    
    # Add knowledge graph results if available
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title", ""),
            "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description", ""),
            "source": "knowledge_graph"
        })
    
    return results

if __name__ == "__main__":
    # Test function
    results = search_web("Python programming", num_results=3)
    print(f"Found {len(results)} web search results")
