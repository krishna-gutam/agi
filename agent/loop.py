import json
import logging
from typing import Any, Callable, Dict, List, Optional
from agent.adapter import Message, ModelAdapter, ModelResponse
from agent.tools import TOOL_REGISTRY, ToolResult

logger = logging.getLogger(__name__)

class AgentLoop:
    def __init__(
        self,
        adapter: ModelAdapter,
        tools: Optional[Dict[str, Callable[..., ToolResult]]] = None,
        max_iterations: int = 10,
    ):
        self.adapter = adapter
        self.tools = tools if tools is not None else TOOL_REGISTRY
        self.max_iterations = max_iterations

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Execute the agent loop given a user prompt."""
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        # Format tool definitions for the adapter if needed
        tool_definitions = [
            {
                "name": name,
                "description": func.__doc__ or "",
            }
            for name, func in self.tools.items()
        ]

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            response: ModelResponse = self.adapter.infer(messages=messages, tools=tool_definitions)

            # If the model produced content, add assistant message
            if response.content or response.tool_calls:
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

            # If no tool calls were requested, we return the content
            if not response.tool_calls:
                return response.content or ""

            # Execute tool calls
            for call in response.tool_calls:
                call_id = call.get("id", "call_1")
                func_name = call.get("name") or call.get("function", {}).get("name")
                arguments = call.get("arguments") or call.get("function", {}).get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                tool_func = self.tools.get(func_name)
                if not tool_func:
                    result_msg = f"Error: Tool '{func_name}' not found."
                    tool_result = ToolResult(output="", error=result_msg, exit_code=1)
                else:
                    try:
                        tool_result = tool_func(**arguments)
                    except Exception as e:
                        tool_result = ToolResult(output="", error=str(e), exit_code=1)

                # Format tool output message according to contract
                tool_output_str = json.dumps({
                    "output": tool_result.output,
                    "error": tool_result.error,
                    "exit_code": tool_result.exit_code,
                    "metadata": tool_result.metadata,
                })

                messages.append(
                    Message(
                        role="tool",
                        content=tool_output_str,
                        tool_call_id=call_id,
                    )
                )

        return "Max iterations reached without final response."
