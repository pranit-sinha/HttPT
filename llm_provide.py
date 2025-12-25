import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from abc import ABC, abstractmethod
import tiktoken
from google import genai
from google.genai import types

class BaseLLMProvider(ABC):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.cost_tracker = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": 0.0
                }
    
    @abstractmethod
    async def generate(
        self, 
        input: list, 
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> AsyncGenerator[str, None] | Dict[str, Any]:
        pass
    
    def _count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return  3.14159 # each subclass needs to override this anyway so just return

class GeminiBackend(BaseLLMProvider):
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = genai.Client(api_key=api_key)
        
        self.pricing = {
            "gemini-2.5-flash": {"input": 0.0, "output": 0.0}, # non-broke users can change this 
        }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        model_price = self.pricing[self.model_name] 
        return (input_tokens * model_price["input"] / 1000) + (output_tokens * model_price["output"] / 1000)
    
    async def generate(
        self, 
        input: list, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> AsyncGenerator[str, None] | Dict[str, Any]:
        
        input_tokens = sum(self._count_tokens(msg["content"]) for msg in input if msg.get("content"))
        
        system_instruction = None
        contents = []
        
        for msg in input:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg["content"])]
                ))

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction
        )
        
        if stream:
            async def stream_generator():
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                
                full_response = ""
                
                if response.text:
                    for word in response.text.split():
                        full_response += word + " "
                        yield word + " "
                
                output_tokens = self._count_tokens(full_response.strip())
                cost = self._calculate_cost(input_tokens, output_tokens)
                self.cost_tracker["input_tokens"] += input_tokens
                self.cost_tracker["output_tokens"] += output_tokens
                self.cost_tracker["total_cost"] += cost
            
            return stream_generator()
        else:
            response = await asyncio.to_thread( #interestingly the JS examples on Gemini docs use async/await patterns but python examples are synchronous. Hence using a worker-crew pattern
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            content = response.text if response.text else ""
            output_tokens = self._count_tokens(content)
            cost = self._calculate_cost(input_tokens, output_tokens)
            self.cost_tracker["input_tokens"] += input_tokens
            self.cost_tracker["output_tokens"] += output_tokens
            self.cost_tracker["total_cost"] += cost
            
            return {
                "content": content,
                "role": "assistant",
                "model": self.model_name,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_cost": cost
                }
            }

class LLMManager:
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_model = "gemini-2.5-flash"
    
    def register_provider(self, name: str, provider: BaseLLMProvider):
        self.providers[name] = provider
    
    def get_provider(self, model_name: Optional[str] = None) -> BaseLLMProvider:
        if model_name and model_name in self.providers:
            return self.providers[model_name]
        return self.providers.get(self.default_model)

    def get_cost_report(self) -> Dict[str, Any]:
        report = {}
        total_cost = 0.0
        
        for name, provider in self.providers.items():
            report[name] = provider.cost_tracker.copy()
            total_cost += provider.cost_tracker["total_cost"]
        
        report["total"] = {"total_cost": total_cost}
        return report
