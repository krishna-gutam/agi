import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class ToolCall:
    """Represents a standardized tool call requested by a model."""
    def __init__(self, id: str, name: str, arguments: Dict[str, Any]):
        self.id = id
        self.name = name
        self.arguments = arguments

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, name={self.name!r}, arguments={self.arguments!r})"


class ModelResponse:
    """Standardized model response containing content and optional tool calls."""
    def __init__(
        self,
        raw_data: Any,
        content: str = "",
        tool_calls: Optional[List[ToolCall]] = None
    ):
        self.raw_data = raw_data
        self.content = content
        self.tool_calls = tool_calls or []

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseModelAdapter(ABC):
    """Abstract base adapter for LLM REST API integrations."""

    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None):
        self.api_key = api_key
        self.model_id = model_id

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai', 'gemini')."""
        pass

    @abstractmethod
    def create_user_message(self, text: str) -> Dict[str, Any]:
        """Formats a user input message for the provider's history structure."""
        pass

    @abstractmethod
    def append_raw_response_to_history(self, history: List[Any], raw_response: Any) -> None:
        """Appends the model's raw response to the conversation history."""
        pass

    @abstractmethod
    def create_tool_response_message(self, tool_call_id: str, tool_name: str, tool_result_str: str) -> Dict[str, Any]:
        """Formats a tool execution result into the provider's message format."""
        pass

    @abstractmethod
    def generate(
        self,
        history: List[Any],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Sends chat history and tools to the LLM API and returns a ModelResponse."""
        pass


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI Chat Completions REST API."""

    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None):
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model_id=model_id or os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        )
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    @property
    def provider_name(self) -> str:
        return "openai"

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
        } for t in tools]

    def create_user_message(self, text: str) -> Dict[str, Any]:
        return {"role": "user", "content": text}

    def append_raw_response_to_history(self, history: List[Any], raw_response: Any) -> None:
        history.append(raw_response)

    def create_tool_response_message(self, tool_call_id: str, tool_name: str, tool_result_str: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result_str
        }

    def generate(
        self,
        history: List[Any],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(history)

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": full_messages
        }
        if tools:
            payload["tools"] = self._format_tools(tools)
            payload["tool_choice"] = "auto"

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"OpenAI API Error ({response.status_code}): {response.text}")

        message = response.json()["choices"][0]["message"]
        content = message.get("content") or ""

        tool_calls: List[ToolCall] = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args
                    )
                )

        return ModelResponse(raw_data=message, content=content, tool_calls=tool_calls)


class GeminiAdapter(BaseModelAdapter):
    """Adapter for Google Gemini generateContent REST API."""

    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None):
        super().__init__(
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            model_id=model_id or os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash")
        )
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        declarations = [{
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        } for t in tools]
        return [{"functionDeclarations": declarations}]

    def create_user_message(self, text: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "parts": [{"text": text}]
        }

    def append_raw_response_to_history(self, history: List[Any], raw_response: Any) -> None:
        history.append(raw_response)

    def create_tool_response_message(self, tool_call_id: str, tool_name: str, tool_result_str: str) -> Dict[str, Any]:
        try:
            tool_response_json = json.loads(tool_result_str)
        except Exception:
            tool_response_json = {"result": tool_result_str}

        return {
            "role": "function",
            "parts": [{
                "functionResponse": {
                    "name": tool_name,
                    "response": tool_response_json
                }
            }]
        }

    def generate(
        self,
        history: List[Any],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "contents": history
        }
        if tools:
            payload["tools"] = self._format_tools(tools)
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Gemini API Error ({response.status_code}): {response.text}")

        data = response.json()
        candidate = data["candidates"][0]
        content_obj = candidate["content"]

        parts = content_obj.get("parts", [])
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        content = "".join(text_parts)

        tool_calls: List[ToolCall] = []
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=fc.get("name", ""),
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {})
                    )
                )

        return ModelResponse(raw_data=content_obj, content=content, tool_calls=tool_calls)


def get_adapter(provider: str, api_key: Optional[str] = None, model_id: Optional[str] = None) -> BaseModelAdapter:
    """Factory to retrieve a model adapter by provider name."""
    provider_lower = provider.strip().lower()
    if provider_lower == "openai":
        return OpenAIAdapter(api_key=api_key, model_id=model_id)
    elif provider_lower == "gemini":
        return GeminiAdapter(api_key=api_key, model_id=model_id)
    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Supported: 'openai', 'gemini'.")
