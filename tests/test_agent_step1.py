import os
import tempfile
import json
import pytest
from agent.adapter import Message, ModelAdapter, ModelResponse
from agent.loop import AgentLoop
from agent.tools import fs_read, fs_write, shell_exec, TOOL_REGISTRY

class MockAdapter(ModelAdapter):
    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.call_count = 0
        self.history: list[list[Message]] = []

    def infer(self, messages: list[Message], tools: list[dict] = None, **kwargs) -> ModelResponse:
        self.history.append(list(messages))
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return ModelResponse(content="Done")

def test_fs_tools():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        
        # Write test
        write_res = fs_write(file_path, "Hello, World!\nLine 2\n")
        assert write_res.exit_code == 0
        assert write_res.error is None
        
        # Read test
        read_res = fs_read(file_path)
        assert read_res.exit_code == 0
        assert "Hello, World!" in read_res.output
        assert read_res.metadata["total_lines"] == 2

        # Read with offset and limit
        read_part = fs_read(file_path, offset=1, limit=1)
        assert read_part.exit_code == 0
        assert "Line 2" in read_part.output

def test_shell_exec():
    res = shell_exec("echo 'hello shell'")
    assert res.exit_code == 0
    assert "hello shell" in res.output

def test_agent_loop_basic():
    # Mock model response: first requests a tool call, then returns final answer
    tool_call_resp = ModelResponse(
        content=None,
        tool_calls=[{
            "id": "call_1",
            "name": "shell_exec",
            "arguments": {"command": "echo 'loop test'"}
        }]
    )
    final_resp = ModelResponse(content="The command executed successfully.")

    adapter = MockAdapter([tool_call_resp, final_resp])
    loop = AgentLoop(adapter=adapter)

    result = loop.run("Run echo test")
    assert result == "The command executed successfully."
    assert adapter.call_count == 2

def test_tool_error_handling():
    # Test direct tool errors
    read_res = fs_read("non_existent_file_12345.txt")
    assert read_res.exit_code != 0
    assert read_res.error is not None

    shell_res = shell_exec("exit 1")
    assert shell_res.exit_code == 1

    # Test agent loop error feedback handling
    tool_call_resp = ModelResponse(
        content=None,
        tool_calls=[{
            "id": "call_err",
            "name": "fs_read",
            "arguments": {"path": "non_existent_file_12345.txt"}
        }]
    )
    final_resp = ModelResponse(content="I noticed the file was not found, handling gracefully.")

    adapter = MockAdapter([tool_call_resp, final_resp])
    loop = AgentLoop(adapter=adapter)

    result = loop.run("Read missing file")
    assert result == "I noticed the file was not found, handling gracefully."
    assert adapter.call_count == 2

    # Verify tool error result was fed back to the model in the second call
    second_call_messages = adapter.history[1]
    tool_msg = [m for m in second_call_messages if m.role == "tool"]
    assert len(tool_msg) == 1
    tool_data = json.loads(tool_msg[0].content or "{}")
    assert tool_data["exit_code"] != 0
    assert "File not found" in tool_data["error"]

def test_max_iterations_limit():
    # Adapter that continuously returns tool calls without stopping
    class InfiniteToolAdapter(ModelAdapter):
        def infer(self, messages: list[Message], tools: list[dict] = None, **kwargs) -> ModelResponse:
            return ModelResponse(
                content=None,
                tool_calls=[{
                    "id": "call_inf",
                    "name": "shell_exec",
                    "arguments": {"command": "echo 'infinite'"}
                }]
            )

    adapter = InfiniteToolAdapter()
    loop = AgentLoop(adapter=adapter, max_iterations=2)

    result = loop.run("Run forever")
    assert result == "Max iterations reached without final response."

def test_empty_response_handling():
    # Adapter returning a ModelResponse with neither content nor tool calls
    empty_resp = ModelResponse(content=None, tool_calls=None)
    adapter = MockAdapter([empty_resp])
    loop = AgentLoop(adapter=adapter)

    result = loop.run("Hello")
    assert result == ""

def test_adapter_exception_handling():
    class FaultyAdapter(ModelAdapter):
        def infer(self, messages: list[Message], tools: list[dict] = None, **kwargs) -> ModelResponse:
            raise RuntimeError("Model inference failed")

    adapter = FaultyAdapter()
    loop = AgentLoop(adapter=adapter)

    with pytest.raises(RuntimeError, match="Model inference failed"):
        loop.run("Test failure")

def test_invalid_tool_arguments_handling():
    # Model returns a tool call with malformed JSON string arguments
    tool_call_resp = ModelResponse(
        content=None,
        tool_calls=[{
            "id": "call_bad_json",
            "name": "shell_exec",
            "arguments": "{invalid json}"
        }]
    )
    final_resp = ModelResponse(content="Handled invalid argument syntax.")

    adapter = MockAdapter([tool_call_resp, final_resp])
    loop = AgentLoop(adapter=adapter)

    result = loop.run("Run command with bad json args")
    assert result == "Handled invalid argument syntax."
    assert adapter.call_count == 2

    # Verify that when arguments fail JSON parsing, tool_func receives empty dict `{}` or handles it,
    # or if tool fails due to missing arguments, it returns a tool error result to the model.
    second_call_messages = adapter.history[1]
    tool_msg = [m for m in second_call_messages if m.role == "tool"]
    assert len(tool_msg) == 1
    tool_data = json.loads(tool_msg[0].content or "{}")
    # shell_exec expects 'command' keyword argument. Since arguments parsed to {} due to JSONDecodeError,
    # shell_exec() raises TypeError or handles missing argument, resulting in exit_code != 0 and error message.
    assert tool_data["exit_code"] != 0
    assert tool_data["error"] is not None





from agent.policy import PolicyEngine, PolicyDecision

def test_agent_loop_policy_denial():
    tool_call_resp = ModelResponse(
        content=None,
        tool_calls=[{
            "id": "call_denied",
            "name": "fs_read",
            "arguments": {"path": "../outside.txt"}
        }]
    )
    final_resp = ModelResponse(content="Action was denied by policy.")
    adapter = MockAdapter([tool_call_resp, final_resp])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_engine = PolicyEngine(workspace_dir=tmpdir)
        loop = AgentLoop(adapter=adapter, policy_engine=policy_engine)
        result = loop.run("Read outside file")
        assert result == "Action was denied by policy."
        
        # Check second call messages for tool error feedback
        second_call_messages = adapter.history[1]
        tool_msg = [m for m in second_call_messages if m.role == "tool"]
        assert len(tool_msg) == 1
        tool_data = json.loads(tool_msg[0].content or "{}")
        assert tool_data["exit_code"] != 0
        assert "denied by policy" in tool_data["error"]

def test_agent_loop_approval_required_fail_closed():
    tool_call_resp = ModelResponse(
        content=None,
        tool_calls=[{
            "id": "call_approve",
            "name": "fs_write",
            "arguments": {"path": "test.txt", "content": "data"}
        }]
    )
    final_resp = ModelResponse(content="Handled approval requirement.")
    adapter = MockAdapter([tool_call_resp, final_resp])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # PolicyEngine with default rule fs.write -> APPROVE, but NO approval_callback provided
        policy_engine = PolicyEngine(workspace_dir=tmpdir)
        loop = AgentLoop(adapter=adapter, policy_engine=policy_engine)
        result = loop.run("Write file without approval callback")
        assert result == "Handled approval requirement."

        second_call_messages = adapter.history[1]
        tool_msg = [m for m in second_call_messages if m.role == "tool"]
        assert len(tool_msg) == 1
        tool_data = json.loads(tool_msg[0].content or "{}")
        assert tool_data["exit_code"] != 0
        assert "denied by policy" in tool_data["error"] or "requires approval" in tool_data["error"]

