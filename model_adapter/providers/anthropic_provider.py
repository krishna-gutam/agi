import os
import requests
from typing import Dict, Any, List, Optional
from ..base import BaseAdapter

class AnthropicAdapter(BaseAdapter):
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any):
        self.model_id = model_id or "claude-3-5-sonnet-20241022"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"

    def infer(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config = config or {}
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        # Simplified conversion for skeleton/stub
        system_msg = ""
        filtered_msgs = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
            else:
                filtered_msgs.append(m)

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": filtered_msgs,
            "max_tokens": config.get("max_tokens", 4096)
        }
        if system_msg:
            payload["system"] = system_msg

        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Anthropic API error ({response.status_code}): {response.text}")
        data = response.json()
        content = data.get("content", [{}])
        text = content[0].get("text", "") if content else ""
        usage = data.get("usage", {"input_tokens": 0, "output_tokens": 0})
        return {
            "text": text,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            },
            "stop_reason": data.get("stop_reason", "end_turn")
        }
