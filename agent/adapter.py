import os
import time
import random
import requests
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ModelResponse(BaseModel):
    content: Optional[str] = None # alias or primary text field
    text: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    stop_reason: Optional[str] = "stop"
    raw_response: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        if self.text and not self.content:
            self.content = self.text
        elif self.content and not self.text:
            self.text = self.content

@runtime_checkable
class ModelAdapter(Protocol):
    def infer(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> ModelResponse:
        """Execute model inference given messages and available tools."""
        ...

class BaseAdapter:
    """Base class for concrete model adapters."""
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any):
        self.model_id = model_id
        self.api_key = api_key
        self.kwargs = kwargs

    def infer(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

class OpenRouterAdapter(BaseAdapter):
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model_id=model_id, api_key=api_key, **kwargs)
        self.model_id = self.model_id or os.getenv("MODEL_ID", "deepseek/deepseek-chat")
        if self.model_id.startswith("openrouter/"):
            self.model_id = self.model_id.split("/", 1)[1]
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def infer(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ModelResponse:
        config = dict(config or {})
        config.update(self.kwargs)
        config.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": config.get("site_url", "https://github.com/agi-agent"),
            "X-Title": config.get("site_name", "AGI Agent"),
            "Content-Type": "application/json"
        }

        # Convert Messages to dicts if needed
        serialized_messages = []
        for m in messages:
            if isinstance(m, Message):
                m_dict = {"role": m.role}
                if m.content is not None:
                    m_dict["content"] = m.content
                if m.tool_calls is not None:
                    m_dict["tool_calls"] = m.tool_calls
                if m.tool_call_id is not None:
                    m_dict["tool_call_id"] = m.tool_call_id
                serialized_messages.append(m_dict)
            else:
                serialized_messages.append(m)

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": serialized_messages,
        }

        if tools:
            payload["tools"] = tools
            if "tool_choice" in config:
                payload["tool_choice"] = config["tool_choice"]

        for param in ["temperature", "max_tokens", "top_p", "stop"]:
            if param in config:
                payload[param] = config[param]

        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error ({response.status_code}): {response.text}")

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        text = message.get("content") or ""
        tool_calls_raw = message.get("tool_calls") or []
        
        tool_calls = []
        for tc in tool_calls_raw:
            func = tc.get("function", {})
            import json
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tool_calls.append({
                "id": tc.get("id"),
                "name": func.get("name"),
                "arguments": args
            })

        usage_raw = data.get("usage")
        if not isinstance(usage_raw, dict):
            usage_raw = {}
        
        prompt_tokens = usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0
        completion_tokens = usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0
        total_tokens = usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

        stop_reason = choice.get("finish_reason", "stop")

        return ModelResponse(
            content=text,
            text=text,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            stop_reason=stop_reason,
            raw_response=data
        )

def get_adapter(provider: str, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any) -> ModelAdapter:
    """Factory function to retrieve the appropriate model adapter implementation."""
    provider_lower = (provider or "openrouter").lower()
    if provider_lower == "openrouter":
        return OpenRouterAdapter(model_id=model_id, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def infer(
    messages: List[Any],
    tools: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> ModelResponse:
    """
    Provider-agnostic router function implementing the infer interface with retry logic and tier routing.
    """
    config = dict(config or {})
    config.update(kwargs)
    
    model_cfg = config.get("model")
    tier_model_id = None
    if isinstance(model_cfg, dict):
        tier = config.get("tier") or config.get("role") or "planner"
        tier_model_id = model_cfg.get(tier) or model_cfg.get("default")
    
    model_id = tier_model_id or config.get("model_id") or os.getenv("MODEL_ID", "openrouter/free")
    
    provider = config.get("provider")
    if not provider:
        if "/" in model_id:
            parts = model_id.split("/", 1)
            provider = parts[0]
            model_id = parts[1]
        else:
            provider = "openrouter"
    elif isinstance(model_id, str):
        if "/" in model_id:
            parts = model_id.split("/", 1)
            provider = parts[0]
            model_id = parts[1]

    api_key = config.get("api_key")
    
    config_clean = dict(config)
    config_clean.pop("provider", None)
    config_clean.pop("model_id", None)
    config_clean.pop("api_key", None)
    config_clean.pop("model", None)

    adapter = get_adapter(provider=provider, model_id=model_id, api_key=api_key, **config_clean)

    max_attempts = 3
    base_delay = 1.0
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return adapter.infer(messages, tools=tools, config=config)
        except (requests.exceptions.RequestException, TimeoutError, RuntimeError) as e:
            err_str = str(e)
            if "error (4" in err_str and not "429" in err_str:
                raise
            
            last_exception = e
            if attempt == max_attempts:
                break
            
            sleep_time = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            time.sleep(sleep_time)

    raise RuntimeError(f"Model inference failed after {max_attempts} attempts. Last error: {last_exception}")
