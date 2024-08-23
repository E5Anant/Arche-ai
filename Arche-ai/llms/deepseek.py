from dotenv import load_dotenv
import os
from typing import List, Dict, Optional

# Mocking DeepSeekClient for demonstration purposes.
# Replace this with the actual DeepSeekClient import from deepseek.
class DeepSeekClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chat(self, model_name: str, temperature: float, max_tokens: int, messages: List[Dict[str, List[str]]], stream: bool):
        # Mocking a chat response. Replace this with actual API call.
        response = {"choices": [{"delta": {"content": "This is a mock response based on the provided messages."}}]}
        return [response]

load_dotenv()

class DeepSeek:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    
    def __init__(self,
                 messages: List[Dict[str, List[str]]] = [],
                 model: str = "deepseek-model",
                 temperature: float = 0.7,
                 system_prompt: Optional[str] = None,
                 max_tokens: int = 2048,
                 verbose: Optional[bool] = False,
                 api_key: str | None = None):
        self.api_key = api_key if api_key else os.getenv("DEEPSEEK_API_KEY")
        self.client = DeepSeekClient(api_key=self.api_key)
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.verbose = verbose
        if self.system_prompt:
            self.add_message(self.SYSTEM, self.system_prompt)
    
    def run(self, prompt: str) -> str:
        self.add_message(self.USER, prompt)
        stream = self.client.chat(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=self.messages,
            stream=True
        )
        self.messages.pop()
        response_text = ""
        for chunk in stream:
            if chunk["choices"][0]["delta"]["content"]:
                response_text += chunk["choices"][0]["delta"]["content"]
            if self.verbose:
                print(chunk["choices"][0]["delta"]["content"] or "", end="")
        self.add_message(self.ASSISTANT, response_text)
        return response_text
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "parts": [content]})
    
    def __getitem__(self, index) -> Dict[str, List[str]] | List[Dict[str, List[str]]]:
        if isinstance(index, slice):
            return self.messages[index]
        elif isinstance(index, int):
            return self.messages[index]
        else:
            raise TypeError("Invalid argument type")
    
    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            self.messages[index] = value
        elif isinstance(index, int):
            self.messages[index] = value
        else:
            raise TypeError("Invalid argument type")

if __name__ == "__main__":
    llm = DeepSeek(verbose=True)
    llm.add_message(DeepSeek.USER, "hello how are you?")
    response = llm.run("hello how are you?")
    print(response)
