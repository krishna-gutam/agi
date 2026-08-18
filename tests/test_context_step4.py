import pytest
from agent.context import ContextManager

def test_context_assembly_basic():
    cm = ContextManager(max_context_tokens=1000)
    messages = cm.assemble_context(
        system_prompt="You are a helper.",
        goal="Build a website",
        constraints=["No downtime"],
        current_plan="1. Setup HTML",
        working_state={"step": "init"},
        trajectory=[{"step_index": 1, "action": "fs_write"}],
        retrieved_memories=["Memory A"],
        tool_schemas=[{"name": "fs_write"}]
    )

    assert len(messages) >= 3
    assert messages[0].role == "system"
    assert "Build a website" in (messages[1].content or "")
    assert "No downtime" in (messages[1].content or "")
    assert "CURRENT PLAN" in (messages[2].content or "")
    assert "WORKING STATE" in (messages[2].content or "")
    assert "RETRIEVED MEMORIES" in (messages[2].content or "")

def test_context_compaction():
    # Set low max_context_tokens so compaction triggers easily
    cm = ContextManager(max_context_tokens=50, compaction_threshold_factor=0.5) # threshold = 25 tokens (~100 chars)
    
    # Create a long trajectory with 5 steps
    long_trajectory = [
        {"step_index": i, "action": f"action_{i}", "output": "A" * 50, "exit_code": 0}
        for i in range(1, 6)
    ]

    messages = cm.assemble_context(
        system_prompt="System",
        goal="Goal",
        constraints=["Constraint 1"],
        current_plan="Plan A",
        trajectory=long_trajectory
    )

    # Check that compaction occurred (contains compacted digest)
    content_blob = " ".join(m.content or "" for m in messages)
    assert "COMPACTED OLDER STEPS DIGEST" in content_blob
    # Verify last 3 steps (3, 4, 5) are preserved verbatim or present in recent trajectory
    assert "action_5" in content_blob
    assert "action_4" in content_blob
    assert "action_3" in content_blob
