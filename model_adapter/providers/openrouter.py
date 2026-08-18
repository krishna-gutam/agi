import os
import requests
from typing import Dict, Any, List, Optional
from ..base import BaseAdapter

class OpenRouterAdapter(BaseAdapter):
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any):
        self.model_id = model_id or os.getenv("MODEL_ID", "deepseek/deepseek-chat")
        if self.model_id.startswith("openrouter/"):
            self.model_id = self.model_id.split("/", 1)[1]
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def infer(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config = config or {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": config.get("site_url", "https://github.com/agi-agent"),
            "X-Title": config.get("site_name", "AGI Agent"),
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
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

        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": usage,
            "stop_reason": stop_reason
        }
