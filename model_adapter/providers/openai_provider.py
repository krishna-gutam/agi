import os
import requests
from typing import Dict, Any, List, Optional
from ..base import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any):
        self.model_id = model_id or "gpt-4o"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def infer(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config = config or {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error ({response.status_code}): {response.text}")
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return {
            "text": message.get("content") or "",
            "tool_calls": [],
            "usage": data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "stop_reason": choice.get("finish_reason", "stop")
        }
