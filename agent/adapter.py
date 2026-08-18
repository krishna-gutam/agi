from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ModelResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Any] = None

@runtime_checkable
class ModelAdapter(Protocol):
    def infer(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> ModelResponse:
        """Execute model inference given messages and available tools."""
        ...
