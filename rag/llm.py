import json
import requests
from abc import ABC

class BaseLLM(ABC):
    def generate(self, prompt: str):
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    def __init__(self, model_name, base_url, timeout=30):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        print(f"[LLM] Initialized OllamaLLM with model={model_name}, base_url={base_url}, timeout={timeout}", flush=True)

    def generate(self, prompt: str):
        print(f"[LLM] Generating with model: {self.model_name}", flush=True)
        print(f"[LLM] Prompt length: {len(prompt)}", flush=True)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "num_predict": 200  # Limit output length for faster generation
        }

        print(f"[LLM] Sending request to {self.base_url}/api/generate", flush=True)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            print(f"[LLM] Response status: {response.status_code}", flush=True)
            
            response.raise_for_status()
            response_data = response.json()
            
            answer = response_data.get("response", "")
            print(f"[LLM] Response length: {len(answer)}", flush=True)
            print(f"[LLM] Response preview: {answer[:100]}", flush=True)
            
            if not answer or not answer.strip():
                print("[LLM] WARNING: Empty response from Ollama!", flush=True)
                print(f"[LLM] Full response data: {response_data}", flush=True)
            
            return answer
        except requests.exceptions.Timeout as e:
            print(f"[LLM] ERROR: Timeout after {self.timeout}s: {e}", flush=True)
            raise
        except requests.exceptions.ConnectionError as e:
            print(f"[LLM] ERROR: Connection error to {self.base_url}: {e}", flush=True)
            raise
        except Exception as e:
            print(f"[LLM] ERROR: {e}", flush=True)
            raise