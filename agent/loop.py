import json
import logging
from typing import Any, Callable, Dict, List, Optional
from agent.adapter import Message, ModelAdapter, ModelResponse
from agent.tools import TOOL_REGISTRY, ToolResult
from agent.policy import PolicyEngine, PolicyDecision

logger = logging.getLogger(__name__)

class AgentLoop:
    def __init__(
        self,
        adapter: ModelAdapter,
        tools: Optional[Dict[str, Callable[..., ToolResult]]] = None,
        max_iterations: int = 10,
        policy_engine: Optional[PolicyEngine] = None,
    ):
        self.adapter = adapter
        self.tools = tools if tools is not None else TOOL_REGISTRY
        self.max_iterations = max_iterations
        self.policy_engine = policy_engine if policy_engine is not None else PolicyEngine()

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

            content = response.content or response.text

            # If the model produced content, add assistant message
            if content or response.tool_calls:
                messages.append(
                    Message(
                        role="assistant",
                        content=content,
                        tool_calls=response.tool_calls,
                    )
                )

            # If no tool calls were requested, we return the content
            if not response.tool_calls:
                return content or ""

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

                # Route tool call through PolicyEngine before execution
                decision = self.policy_engine.evaluate(func_name, arguments)
                if decision == PolicyDecision.DENY:
                    tool_result = ToolResult(
                        output="",
                        error=f"Tool execution denied by policy for '{func_name}'.",
                        exit_code=1
                    )
                elif decision == PolicyDecision.APPROVE and not self.policy_engine.approval_callback:
                    # Fail closed if APPROVE is required but no approval callback is provided
                    tool_result = ToolResult(
                        output="",
                        error=f"Tool execution requires approval (APPROVE policy) but no approval callback was provided for '{func_name}'.",
                        exit_code=1
                    )
                else:
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
                        tool_call_id=tool_id if 'tool_id' in locals() else call_id,
                    )
                )

        return "Max iterations reached without final response."
