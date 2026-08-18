import time
import pytest
from agent.memory import MemoryStore

def test_working_memory():
    store = MemoryStore()
    store.working_set("plan", "Step 1: Build")
    assert store.working_get("plan") == "Step 1: Build"
    assert store.working_get("missing", "default") == "default"

def test_episodic_memory():
    store = MemoryStore()
    store.record_episode("run_001", 1, "Executed ls", metadata={"status": "success"})
    store.record_episode("run_001", 2, "Executed cat", metadata={"status": "success"})
    store.record_episode("run_002", 1, "Executed pytest", metadata={"status": "failed"})

    run1_episodes = store.retrieve_episodes(run_id="run_001")
    assert len(run1_episodes) == 2
    assert run1_episodes[0].step_index == 1
    assert run1_episodes[1].content == "Executed cat"

    all_episodes = store.retrieve_episodes(limit=10)
    assert len(all_episodes) == 3

def test_semantic_memory_gating_and_ttl():
    store = MemoryStore()
    
    # Below confidence gate (0.5) -> should fail to write
    success_low = store.write_semantic(
        key="fact_1",
        content="Low confidence fact",
        source_run_id="run_1",
        confidence=0.3,
        min_confidence_gate=0.5
    )
    assert not success_low
    assert len(store.retrieve_semantic("fact_1")) == 0

    # Above confidence gate -> should succeed
    success_high = store.write_semantic(
        key="fact_2",
        content="High confidence python best practice",
        source_run_id="run_1",
        confidence=0.9,
        min_confidence_gate=0.5
    )
    assert success_high
    results = store.retrieve_semantic("python")
    assert len(results) == 1
    assert results[0].content == "High confidence python best practice"

    # Test TTL expiration
    store.write_semantic(
        key="temp_fact",
        content="Ephemeral information",
        source_run_id="run_1",
        confidence=1.0,
        ttl_seconds=-1 # already expired
    )
    assert len(store.retrieve_semantic("Ephemeral")) == 0
