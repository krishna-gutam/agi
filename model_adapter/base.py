from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseAdapter(ABC):
    """
    Provider-agnostic base adapter interface adhering to Section 3.1 & Section 5.
    """
    @abstractmethod
    def infer(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes model inference and returns standardized response structure:
        {
            "text": str,
            "tool_calls": List[Dict[str, Any]],
            "usage": Dict[str, int],
            "stop_reason": str
        }
        """
        pass

def get_adapter(provider: str, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any) -> BaseAdapter:
    """
    Factory function to retrieve the appropriate model adapter implementation.
    """
    from .providers.openrouter import OpenRouterAdapter
    from .providers.openai_provider import OpenAIAdapter
    from .providers.anthropic_provider import AnthropicAdapter

    provider_lower = (provider or "openrouter").lower()
    if provider_lower == "openrouter":
        return OpenRouterAdapter(model_id=model_id, api_key=api_key, **kwargs)
    elif provider_lower == "openai":
        return OpenAIAdapter(model_id=model_id, api_key=api_key, **kwargs)
    elif provider_lower == "anthropic":
        return AnthropicAdapter(model_id=model_id, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
