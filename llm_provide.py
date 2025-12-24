import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from abc import ABC, abstractmethod
import tiktoken

class BaseLLMProvider(ABC):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    async def generate(
        self, 
        messages: list, 
        temperature: float = 0.7,
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
        # TODO
        return 

class LLMManager:
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
    
    def register_provider(self, name: str, Provider: BaseLLMProvider):
        self.providers[name] = provider
    
    def get_provider(self, model_name: Optional[str] = None) -> BaseLLMProvider:
        if model_name and model_name in self.providers:
            return self.providers[model_name]
        return self.providers.get(self.default_model)
