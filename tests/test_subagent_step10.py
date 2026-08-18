import tempfile
import pytest
from agent.subagent import SubagentManager, SubagentConfig
from agent.tools import TOOL_REGISTRY, ToolResult

def test_subagent_spawn_success():
    manager = SubagentManager(workspace_dir=".")
    config = SubagentConfig(
        goal="Search logs",
        parent_run_id="parent_001",
        depth=1,
        max_depth=2,
        permissions=["fs.read"],
        budget_tokens=500
    )

    def mock_task(cfg):
        return f"Completed goal: {cfg.goal}"

    result = manager.spawn_subagent(config, mock_task)
    assert result.success
    assert "Completed goal: Search logs" in result.output

def test_subagent_depth_limit_exceeded():
    manager = SubagentManager(workspace_dir=".")
    # Depth 3 exceeds max_depth 2
    config = SubagentConfig(
        goal="Deep task",
        parent_run_id="parent_001",
        depth=3,
        max_depth=2
    )

    result = manager.spawn_subagent(config, lambda c: "should not run")
    assert not result.success
    assert "Max subagent depth exceeded" in result.error

def test_parallel_read_only_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SubagentManager(workspace_dir=tmpdir)
        
        # Create test files
        f1 = f"{tmpdir}/file1.txt"
        f2 = f"{tmpdir}/file2.txt"
        TOOL_REGISTRY["fs.write"](path=f1, content="content 1")
        TOOL_REGISTRY["fs.write"](path=f2, content="content 2")

        calls = [
            {"name": "fs.read", "arguments": {"path": f1}},
            {"name": "fs.read", "arguments": {"path": f2}}
        ]

        results = manager.execute_parallel_read_only(calls, TOOL_REGISTRY)
        assert len(results) == 2
        assert results[0].output == "content 1" or results[1].output == "content 1"
        assert results[0].output == "content 2" or results[1].output == "content 2"
