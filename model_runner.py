import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def call_gemini(question, retry=False):
    """Using Groq Gemma as third model (Gemini API quota exhausted on free tier)"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[ERROR: GROQ_API_KEY not found in .env]"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama-3.3-70b-versatile",   # Google's Gemma 2, free on Groq
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 512
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        data = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return f"[Gemma error: {data.get('error', {}).get('message', str(data))}]"

    except Exception as e:
        return f"[Gemma exception: {str(e)}]"


def call_groq(question):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[ERROR: GROQ_API_KEY not found in .env]"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Updated to current active Groq models (llama3-8b-8192 was decommissioned)
    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 512
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        data = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return f"[Groq error: {data.get('error', {}).get('message', str(data))}]"

    except Exception as e:
        return f"[Groq exception: {str(e)}]"


def call_ollama(question):
    # First check if Ollama server is reachable
    try:
        requests.get("http://localhost:11434", timeout=3)
    except requests.exceptions.ConnectionError:
        return "[Ollama error: server not running — open a new terminal and run: ollama serve]"

    # Get list of available models
    try:
        models_resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in models_resp.json().get("models", [])]
    except:
        available = []

    # Pick whichever model you have downloaded
    preferred = ["qwen2.5:1.5b", "qwen2.5", "llama3.2", "phi3", "mistral"]
    model_to_use = None
    for p in preferred:
        if p in available:
            model_to_use = p
            break

    if not model_to_use:
        if available:
            model_to_use = available[0]   # use whatever is downloaded
        else:
            return f"[Ollama error: no models downloaded. Run: ollama pull phi3]"

    url = "http://localhost:11434/api/generate"
    body = {
        "model": model_to_use,
        "prompt": question,
        "stream": False            # critical — ensures we get a single JSON response
    }

    try:
        response = requests.post(url, json=body, timeout=120)
        data = response.json()

        if "response" in data:
            return data["response"]

        return f"[Ollama error: unexpected response format: {list(data.keys())}]"

    except Exception as e:
        return f"[Ollama exception: {str(e)}]"


def run_all(dataset):
    results = []
    for item in dataset:
        q = item["question"]
        print(f"\nRunning: {q[:55]}...")

        llama70b_resp = call_gemini(q)
        status = "OK" if not llama70b_resp.startswith("[") else llama70b_resp[:80]
        print(f"  Llama70B: {status}")

        groq_resp = call_groq(q)
        status = "OK" if not groq_resp.startswith("[") else groq_resp[:80]
        print(f"  Groq:   {status}")

        ollama_resp = call_ollama(q)
        status = "OK" if not ollama_resp.startswith("[") else ollama_resp[:80]
        print(f"  Ollama: {status}")

        print("  Waiting 2s...")
        time.sleep(2)

        results.append({
            "id": item["id"],
            "question": q,
            "context": item["context"],
            "reference": item["reference"],
            "llama70b": llama70b_resp,
            "groq": groq_resp,
            "ollama": ollama_resp,
        })
    return results