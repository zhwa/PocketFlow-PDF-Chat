"""
LLM client utilities for both Ollama and SiliconFlow
"""
import os
import json
import requests
import logging

logger = logging.getLogger(__name__)

def call_llm(prompt: str, model_choice: str = "ollama", **kwargs) -> str:
    """Call LLM with the given prompt"""
    if model_choice == "siliconflow":
        return call_siliconflow_api(prompt, **kwargs)
    return call_ollama(prompt, **kwargs)

def call_ollama(prompt: str, model: str = "deepseek-r1:1.5b", **kwargs) -> str:
    """Call local Ollama model"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=kwargs.get("timeout", 180)
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")

def call_siliconflow_api(prompt: str, temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> str:
    """Call SiliconFlow API"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY not configured")

    payload = {
        "model": "Pro/deepseek-ai/DeepSeek-R1",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # Encode payload as UTF-8 JSON
    json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')

    response = requests.post(
        "https://api.siliconflow.cn/v1/chat/completions",
        data=json_payload,
        headers=headers,
        timeout=180
    )

    response.raise_for_status()
    result = response.json()

    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0]["message"]
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

        # Handle thinking chain if present
        if reasoning:
            return f"{content}<think>{reasoning}</think>"
        return content
    raise ValueError("Invalid API response format")

def process_thinking_content(text: str) -> str:
    """Process text with <think> tags for display"""
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    processed_text = text

    # Process thinking chain tags
    try:
        while "<think>" in processed_text and "</think>" in processed_text:
            start_idx = processed_text.find("<think>")
            end_idx = processed_text.find("</think>")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                thinking_content = processed_text[start_idx + 7:end_idx]
                before_think = processed_text[:start_idx]
                after_think = processed_text[end_idx + 8:]

                # Use collapsible details for thinking process
                processed_text = (before_think + 
                                "\n\n<details>\n<summary>思考过程（点击展开）</summary>\n\n" + 
                                thinking_content + 
                                "\n\n</details>\n\n" + 
                                after_think)
    except Exception as e:
        logger.error(f"Error processing thinking content: {str(e)}")
        # Fallback to safe HTML escaping
        return text.replace("<", "&lt;").replace(">", "&gt;")

    return processed_text

if __name__ == "__main__":
    try:
        response = call_llm("Hello, how are you?", model_choice="ollama")
        print(f"LLM response: {response[:100]}...")
    except Exception as e:
        print(f"LLM test failed: {e}")