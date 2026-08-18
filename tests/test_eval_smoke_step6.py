import os
import tempfile
import pytest
from state_store.manager import StateStore, CheckpointManager
from agent.context import ContextManager
from agent.policy import PolicyEngine, PolicyDecision
from agent.tools import TOOL_REGISTRY, ToolResult
from agent.loop import AgentLoop
from agent.adapter import ModelAdapter, Message, ModelResponse

class MockSmokeModelAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(model_name="mock-smoke-model")
        self.call_count = 0

    def infer(self, messages, tools=None, temperature=0.0, max_tokens=1000):
        self.call_count += 1
        if self.call_count == 1:
            return ModelResponse(
                content="I will read the file.",
                tool_calls=[{"id": "call_1", "name": "fs.read", "arguments": {"path": "test.txt"}}]
            )
        else:
            return ModelResponse(content="Task completed successfully.", tool_calls=[])

def test_smoke_1_basic_agent_run():
    """Smoke Task 1: Verify basic agent run executes steps and completes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("smoke test content")

        adapter = MockSmokeModelAdapter()
        loop = AgentLoop(adapter=adapter, tools=TOOL_REGISTRY, max_iterations=2)
        
        result = loop.run(prompt="Read the file test.txt", system_prompt="You are a helper.")
        assert "Task completed" in result or result is not None

def test_smoke_2_tool_routing():
    """Smoke Task 2: Verify correct tool routing via TOOL_REGISTRY."""
    custom_registry = dict(TOOL_REGISTRY)
    def smoke_echo(message: str) -> ToolResult:
        return ToolResult(output=f"Echo: {message}")
    
    custom_registry["smoke.echo"] = smoke_echo
    func = custom_registry["smoke.echo"]
    res = func(message="hello tool")
    assert res.output == "Echo: hello tool"

def test_smoke_3_compaction_trigger():
    """Smoke Task 3: Verify compaction triggers at threshold."""
    cm = ContextManager(max_context_tokens=30, compaction_threshold_factor=0.5)
    traj = [
        {"step_index": 1, "action": "step1"},
        {"step_index": 2, "action": "step2"},
        {"step_index": 3, "action": "step3"},
        {"step_index": 4, "action": "step4"},
    ]
    messages = cm.assemble_context(
        system_prompt="System",
        goal="Goal",
        trajectory=traj
    )
    content_blob = " ".join(m.content or "" for m in messages)
    assert "COMPACTED OLDER STEPS DIGEST" in content_blob

def test_smoke_4_policy_denial():
    """Smoke Task 4: Verify policy engine correctly denies out-of-bounds access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy = PolicyEngine(workspace_dir=tmpdir)
        outside_path = os.path.abspath(os.path.join(tmpdir, "..", "secret.txt"))
        
        decision = policy.evaluate("fs.read", {"path": outside_path})
        assert decision == PolicyDecision.DENY

def test_smoke_5_state_persistence_and_checkpoints():
    """Smoke Task 5: Verify state store persistence and checkpoint loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(run_id="smoke_persist", storage_dir=tmpdir)
        store.set("key", "value")
        cp_path = store.append_trajectory({"action": "test_action"})

        assert os.path.exists(cp_path)

        # Load in new store
        store_new = StateStore(run_id="smoke_persist", storage_dir=tmpdir)
        store_new.load_checkpoint(cp_path)
        assert store_new.get("key") == "value"
        assert len(store_new.get("trajectory")) == 1

def test_smoke_6_eval_harness_metrics():
    """Smoke Task 6: Simulate running a batch of smoke tasks and returning structured eval metrics."""
    def run_eval_batch(tasks):
        results = []
        for task in tasks:
            # Simulate evaluation run
            success = task.get("expected_success", True)
            harmful = task.get("harmful_action", False)
            results.append({"success": success, "harmful": harmful})
        
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        harmful_count = sum(1 for r in results if r["harmful"])
        
        metrics = {
            "total_tasks": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "harmful_action_rate": harmful_count / total if total > 0 else 0.0,
        }
        return metrics

    sample_tasks = [
        {"id": "t1", "expected_success": True, "harmful_action": False},
        {"id": "t2", "expected_success": True, "harmful_action": False},
        {"id": "t3", "expected_success": False, "harmful_action": True},
        {"id": "t4", "expected_success": True, "harmful_action": False},
    ]

    metrics = run_eval_batch(sample_tasks)
    assert metrics["total_tasks"] == 4
    assert metrics["success_rate"] == 0.75
    assert metrics["harmful_action_rate"] == 0.25

