from typing import Any, Callable, Dict, List, Optional
from model_adapter import BaseModelAdapter, ModelResponse, ToolCall
from tools import TOOLS, execute_tool


class AgentOrchestrator:
    """
    Coordinates conversation state, model adapter queries, and tool execution loops.
    """

    def __init__(
        self,
        adapter: BaseModelAdapter,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_tool_iterations: int = 10
    ):
        self.adapter = adapter
        self.tools = tools if tools is not None else TOOLS
        self.system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations
        self.history: List[Any] = []

    def reset(self) -> None:
        """Clears conversation history."""
        self.history = []

    def run_turn(
        self,
        user_input: str,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None
    ) -> str:
        """
        Processes a single user turn:
        1. Formats and adds user input to history.
        2. Calls model adapter.
        3. Executes tool calls if returned and feeds results back to model.
        4. Continues until model produces final text or iteration limit is reached.
        """
        user_msg = self.adapter.create_user_message(user_input)
        self.history.append(user_msg)

        final_response_text = ""

        for iteration in range(self.max_tool_iterations):
            response: ModelResponse = self.adapter.generate(
                history=self.history,
                system_prompt=self.system_prompt,
                tools=self.tools
            )

            # Record model raw response in conversation history
            self.adapter.append_raw_response_to_history(self.history, response.raw_data)

            if response.has_tool_calls():
                for tool_call in response.tool_calls:
                    if on_tool_call:
                        on_tool_call(tool_call.name, tool_call.arguments)

                    tool_result_str = execute_tool(tool_call.name, tool_call.arguments)

                    if on_tool_result:
                        on_tool_result(tool_call.name, tool_result_str)

                    tool_msg = self.adapter.create_tool_response_message(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_result_str=tool_result_str
                    )
                    self.history.append(tool_msg)
            else:
                final_response_text = response.content
                break
        else:
            final_response_text = response.content or "[Reached maximum tool iterations]"

        return final_response_text
