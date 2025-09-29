import requests

def test_ollama():
    # Test connection
    try:
        response = requests.get("http://localhost:11434/api/tags")
        print("Available models:")
        for model in response.json()['models']:
            print(f"  - {model['name']}")
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    
    model_name = "gemma3:4b"  
    
    payload = {
        "model": model_name,
        "prompt": "Say hello",
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        print(f"\nStatus: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()['response']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    test_ollama()