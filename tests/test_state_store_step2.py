import os
import tempfile
import pytest
from state_store.manager import StateStore, CheckpointManager

def test_state_store_initialization():
    store = StateStore(run_id="test_run_001")
    assert store.run_id == "test_run_001"
    assert store.get("step_count") == 0
    assert store.get("status") == "initialized"
    assert store.get("total_tokens") == 0
    assert store.get("total_cost_usd") == 0.0

def test_state_store_append_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(run_id="test_run_002", storage_dir=tmpdir)
        
        # Append trajectory step
        step_data = {
            "action": "shell_exec",
            "output": "hello",
            "usage": {"total_tokens": 150},
            "cost_usd": 0.002
        }
        cp_path = store.append_trajectory(step_data)
        
        assert store.get("step_count") == 1
        assert store.get("total_tokens") == 150
        assert store.get("total_cost_usd") == 0.002
        assert os.path.exists(cp_path)

        # Test reloading state from checkpoint
        new_store = StateStore(run_id="test_run_002", storage_dir=tmpdir)
        new_store.load_checkpoint(cp_path)
        assert new_store.get("step_count") == 1
        assert new_store.get("total_tokens") == 150
        assert new_store.get("total_cost_usd") == 0.002
        assert len(new_store.get("trajectory")) == 1

def test_checkpoint_manager_listing_and_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(run_id="test_run_003", storage_dir=tmpdir)
        store.append_trajectory({"action": "fs_read", "output": "content 1", "usage": {"total_tokens": 10}, "cost_usd": 0.0001})
        store.append_trajectory({"action": "fs_write", "output": "content 2", "usage": {"total_tokens": 20}, "cost_usd": 0.0002})

        manager = CheckpointManager(storage_dir=tmpdir)
        checkpoints = manager.list_checkpoints(run_id="test_run_003")
        assert len(checkpoints) == 2

        viewer_data = manager.export_trajectory_viewer_data(store)
        assert viewer_data["run_id"] == "test_run_003"
        assert viewer_data["step_count"] == 2
        assert viewer_data["total_tokens"] == 30
        assert len(viewer_data["trajectory"]) == 2

def test_budget_exceeded_and_graceful_halt():
    store = StateStore(
        run_id="test_run_budget",
        budget={"max_steps": 2, "max_tokens": 100, "headroom_factor": 0.9}
    )
    # Step 1: 50 tokens (below 90% of 100 tokens, but steps=1 vs 2*0.9=1.8 -> steps >= 1.8? no, 1 < 1.8)
    store.append_trajectory({"action": "test", "usage": {"total_tokens": 50}, "cost_usd": 0.01})
    assert not store.check_budget_exceeded()

    # Step 2: takes step count to 2, which is >= 2 * 0.9 (1.8)
    store.append_trajectory({"action": "test", "usage": {"total_tokens": 10}, "cost_usd": 0.01})
    assert store.check_budget_exceeded()
    assert store.should_graceful_halt()

    store.mark_graceful_halt()
    assert store.get("status") == "graceful_halt"
    assert not store.should_graceful_halt() # Already triggered

def test_checkpoint_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(run_id="test_run_nf", storage_dir=tmpdir)
        missing_path = os.path.join(tmpdir, "non_existent_cp.json")
        
        with pytest.raises(FileNotFoundError):
            store.load_checkpoint(missing_path)

        manager = CheckpointManager(storage_dir=tmpdir)
        with pytest.raises(FileNotFoundError):
            manager.delete_checkpoint(missing_path)

def test_checkpoint_manager_list_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_a = StateStore(run_id="run_alpha", storage_dir=tmpdir)
        store_a.append_trajectory({"action": "alpha_step"})

        store_b = StateStore(run_id="run_beta", storage_dir=tmpdir)
        store_b.append_trajectory({"action": "beta_step_1"})
        store_b.append_trajectory({"action": "beta_step_2"})

        manager = CheckpointManager(storage_dir=tmpdir)
        
        # List all checkpoints
        all_cps = manager.list_checkpoints()
        assert len(all_cps) == 3

        # Filter by run_alpha
        alpha_cps = manager.list_checkpoints(run_id="run_alpha")
        assert len(alpha_cps) == 1
        assert "run_alpha" in alpha_cps[0]

        # Filter by run_beta
        beta_cps = manager.list_checkpoints(run_id="run_beta")
        assert len(beta_cps) == 2
        for cp in beta_cps:
            assert "run_beta" in cp

def test_checkpoint_manager_list_filtering_malformed_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid store and checkpoint
        store = StateStore(run_id="run_valid", storage_dir=tmpdir)
        store.append_trajectory({"action": "step_1"})

        # Create a malformed JSON checkpoint file in the same storage directory
        malformed_path = os.path.join(tmpdir, "run_valid_malformed.json")
        with open(malformed_path, "w", encoding="utf-8") as f:
            f.write("{malformed json content...")

        manager = CheckpointManager(storage_dir=tmpdir)
        # Should not crash when encountering malformed JSON file during filtering
        cps = manager.list_checkpoints(run_id="run_valid")
        assert len(cps) == 1
        assert "run_valid" in cps[0]
        assert "malformed" not in cps[0]

def test_export_trajectory_viewer_data_sanitization():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(run_id="run_sanitization", storage_dir=tmpdir)
        # Add a very long memory value (> 100 chars)
        long_value = "A" * 200
        store.set("memory", {"key1": long_value})

        manager = CheckpointManager(storage_dir=tmpdir)
        viewer_data = manager.export_trajectory_viewer_data(store)

        memory_summary = viewer_data["memory_summary"]
        assert "key1" in memory_summary
        assert len(memory_summary["key1"]) <= 100

def test_checkpoint_manager_delete_non_existent_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(storage_dir=tmpdir)
        fake_path = os.path.join(tmpdir, "does_not_exist.json")
        with pytest.raises(FileNotFoundError):
            manager.delete_checkpoint(fake_path)





