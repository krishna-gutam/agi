import os
import pytest
from state_store import StateStore, CheckpointManager

def test_state_store_initialization():
    store = StateStore(run_id="test_run_001")
    assert store.get("run_id") == "test_run_001"
    assert store.get("step_count") == 0
    assert store.get("trajectory") == []
    assert store.get("status") == "initialized"

def test_state_store_mutations_and_trajectory():
    store = StateStore(run_id="test_run_002")
    store.set("status", "running")
    store.set("memory", {"key1": "value1"})

    store.append_trajectory({
        "role": "planner",
        "action": "plan",
        "thought": "Planning next steps"
    })

    store.append_trajectory({
        "role": "executor",
        "action": "execute",
        "output": "Done"
    })

    assert store.get("status") == "running"
    assert store.get("step_count") == 2
    traj = store.get("trajectory")
    assert len(traj) == 2
    assert traj[0]["step_index"] == 1
    assert traj[0]["role"] == "planner"
    assert traj[1]["step_index"] == 2
    assert traj[1]["role"] == "executor"

def test_checkpoint_save_and_load(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    store = StateStore(run_id="run_cp_test", storage_dir=storage_dir)
    store.set("status", "in_progress")
    store.append_trajectory({"action": "test_step"})

    cp_path = store.save_checkpoint(tag="checkpoint_1")
    assert os.path.exists(cp_path)

    store.set("status", "altered")
    store.set("step_count", 99)

    store.load_checkpoint(cp_path)
    assert store.get("run_id") == "run_cp_test"
    assert store.get("status") == "in_progress"
    assert store.get("step_count") == 1
    assert len(store.get("trajectory")) == 1

def test_checkpoint_manager_and_viewer_export(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(storage_dir=storage_dir)

    store = StateStore(run_id="run_viewer_test", storage_dir=storage_dir)
    store.set("memory", {"user_goal": "build an AGI agent"})
    store.append_trajectory({"step": 1, "thought": "starting"})
    cp_path = store.save_checkpoint(tag="step_1")

    checkpoints = manager.list_checkpoints(run_id="run_viewer_test")
    assert len(checkpoints) == 1
    assert checkpoints[0] == cp_path

    viewer_data = manager.export_trajectory_viewer_data(store)
    assert viewer_data["run_id"] == "run_viewer_test"
    assert viewer_data["step_count"] == 1
    assert len(viewer_data["trajectory"]) == 1
    assert "user_goal" in viewer_data["memory_summary"]

def test_budget_enforcement_and_headroom():
    budget = {
        "max_steps": 2,
        "max_tokens": 1000,
        "max_cost_usd": 0.10,
        "headroom_factor": 0.9
    }
    store = StateStore(run_id="run_budget_test", budget=budget)

    assert store.should_graceful_halt() is False

    store.append_trajectory({"action": "step 1", "usage": {"total_tokens": 100}, "cost_usd": 0.01})
    assert store.should_graceful_halt() is False

    store.append_trajectory({"action": "step 2", "usage": {"total_tokens": 100}, "cost_usd": 0.01})
    assert store.should_graceful_halt() is True

    store.mark_graceful_halt()
    assert store.get("graceful_halt_triggered") is True
    assert store.get("status") == "graceful_halt"
    assert store.should_graceful_halt() is False

def test_token_and_cost_budget_headroom():
    budget_tokens = {
        "max_steps": 10,
        "max_tokens": 1000,
        "max_cost_usd": 1.0,
        "headroom_factor": 0.9
    }
    store_tokens = StateStore(run_id="run_token_budget", budget=budget_tokens)
    assert store_tokens.should_graceful_halt() is False

    store_tokens.append_trajectory({"action": "big step", "usage": {"total_tokens": 900}, "cost_usd": 0.01})
    assert store_tokens.should_graceful_halt() is True

    budget_cost = {
        "max_steps": 10,
        "max_tokens": 10000,
        "max_cost_usd": 0.50,
        "headroom_factor": 0.9
    }
    store_cost = StateStore(run_id="run_cost_budget", budget=budget_cost)
    assert store_cost.should_graceful_halt() is False

    store_cost.append_trajectory({"action": "expensive step", "usage": {"total_tokens": 100}, "cost_usd": 0.45})
    assert store_cost.should_graceful_halt() is True

def test_multi_step_trajectory_persistence(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    store = StateStore(run_id="run_multi_step", storage_dir=storage_dir)

    for i in range(1, 4):
        store.append_trajectory({
            "role": "planner" if i % 2 != 0 else "executor",
            "action": f"action_{i}",
            "thought": f"thought_{i}",
            "usage": {"total_tokens": 50 * i},
            "cost_usd": 0.01 * i
        })

    assert store.get("step_count") == 3
    assert store.get("total_tokens") == 300
    assert abs(store.get("total_cost_usd") - 0.06) < 1e-6

    cp_path = store.save_checkpoint(tag="multi_step_cp")
    assert os.path.exists(cp_path)

    restored_store = StateStore(run_id="run_multi_step", storage_dir=storage_dir)
    restored_store.load_checkpoint(cp_path)

    assert restored_store.get("step_count") == 3
    assert restored_store.get("total_tokens") == 300
    assert abs(restored_store.get("total_cost_usd") - 0.06) < 1e-6
    
    trajectory = restored_store.get("trajectory")
    assert len(trajectory) == 3
    for i, step in enumerate(trajectory, start=1):
        assert step["step_index"] == i
        assert step["action"] == f"action_{i}"

def test_checkpoint_manager_filtering_by_run_id(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(storage_dir=storage_dir)

    store_a = StateStore(run_id="run_alpha", storage_dir=storage_dir)
    store_a.append_trajectory({"action": "alpha_step"})
    cp_a = store_a.save_checkpoint(tag="step_1")

    store_b = StateStore(run_id="run_beta", storage_dir=storage_dir)
    store_b.append_trajectory({"action": "beta_step"})
    cp_b1 = store_b.save_checkpoint(tag="step_1")
    cp_b2 = store_b.save_checkpoint(tag="step_2")

    all_cps = manager.list_checkpoints()
    assert len(all_cps) == 3

    alpha_cps = manager.list_checkpoints(run_id="run_alpha")
    assert len(alpha_cps) == 1
    assert alpha_cps[0] == cp_a

    beta_cps = manager.list_checkpoints(run_id="run_beta")
    assert len(beta_cps) == 2
    assert cp_b1 in beta_cps
    assert cp_b2 in beta_cps

def test_auto_checkpoint_on_append_trajectory(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(storage_dir=storage_dir)

    store = StateStore(run_id="run_auto_cp", storage_dir=storage_dir)
    assert len(manager.list_checkpoints(run_id="run_auto_cp")) == 0

    store.append_trajectory({"action": "auto_saved_step_1"})

    checkpoints = manager.list_checkpoints(run_id="run_auto_cp")
    assert len(checkpoints) == 1
    assert "step_1" in checkpoints[0]

    store.append_trajectory({"action": "auto_saved_step_2"})
    checkpoints = manager.list_checkpoints(run_id="run_auto_cp")
    assert len(checkpoints) == 2
    assert "step_2" in checkpoints[1]

def test_disabled_auto_checkpoint(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(storage_dir=storage_dir)

    store = StateStore(run_id="run_no_auto_cp", storage_dir=storage_dir, auto_checkpoint=False)
    assert len(manager.list_checkpoints(run_id="run_no_auto_cp")) == 0

    store.append_trajectory({"action": "step_one"})
    store.append_trajectory({"action": "step_two"})

    assert len(manager.list_checkpoints(run_id="run_no_auto_cp")) == 0

    manual_cp = store.save_checkpoint(tag="manual")
    assert os.path.exists(manual_cp)
    assert len(manager.list_checkpoints(run_id="run_no_auto_cp")) == 1

def test_custom_budget_headroom_dimensions():
    budget = {
        "max_steps": 10,
        "max_tokens": 2000,
        "max_cost_usd": 2.0,
        "headroom_factor": 0.9
    }
    store = StateStore(run_id="run_custom_budgets", budget=budget)

    for i in range(8):
        store.append_trajectory({"action": f"step_{i+1}", "usage": {"total_tokens": 10}, "cost_usd": 0.01})
        assert store.should_graceful_halt() is False

    store.append_trajectory({"action": "step_9", "usage": {"total_tokens": 10}, "cost_usd": 0.01})
    assert store.should_graceful_halt() is True

    store2 = StateStore(run_id="run_custom_budgets_2", budget=budget)
    assert store2.should_graceful_halt() is False

    store2.append_trajectory({"action": "bulk_tokens", "usage": {"total_tokens": 1750}, "cost_usd": 0.10})
    assert store2.should_graceful_halt() is False

    store2.append_trajectory({"action": "push_tokens", "usage": {"total_tokens": 60}, "cost_usd": 0.01})
    assert store2.should_graceful_halt() is True

    store3 = StateStore(run_id="run_custom_budgets_3", budget=budget)
    assert store3.should_graceful_halt() is False

    store3.append_trajectory({"action": "expensive_step", "usage": {"total_tokens": 10}, "cost_usd": 1.75})
    assert store3.should_graceful_halt() is False

    store3.append_trajectory({"action": "final_expensive_step", "usage": {"total_tokens": 10}, "cost_usd": 0.06})
    assert store3.should_graceful_halt() is True

def test_checkpoint_manager_export_viewer_data_structure(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(storage_dir=storage_dir)

    store = StateStore(run_id="run_viewer_export", storage_dir=storage_dir)
    store.set("status", "running")
    store.set("memory", {"project_goal": "build autonomous AGI agent"})
    store.append_trajectory({
        "role": "planner",
        "action": "analyze",
        "thought": "Analyzing requirements",
        "usage": {"total_tokens": 120},
        "cost_usd": 0.002
    })
    store.mark_graceful_halt()

    viewer_data = manager.export_trajectory_viewer_data(store)

    assert viewer_data["run_id"] == "run_viewer_export"
    assert viewer_data["status"] == "graceful_halt"
    assert viewer_data["step_count"] == 1
    assert viewer_data["total_tokens"] == 120
    assert abs(viewer_data["total_cost_usd"] - 0.002) < 1e-6
    assert viewer_data["graceful_halt_triggered"] is True
    assert isinstance(viewer_data["trajectory"], list)
    assert len(viewer_data["trajectory"]) == 1
    assert viewer_data["trajectory"][0]["action"] == "analyze"
    assert isinstance(viewer_data["memory_summary"], dict)
    assert "project_goal" in viewer_data["memory_summary"]
    assert viewer_data["memory_summary"]["project_goal"] == "build autonomous AGI agent"

def test_load_checkpoint_file_not_found(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    store = StateStore(run_id="run_missing_cp", storage_dir=storage_dir)

    nonexistent_path = os.path.join(storage_dir, "nonexistent_checkpoint.json")

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        store.load_checkpoint(nonexistent_path)

def test_checkpoint_manager_delete_checkpoint(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(storage_dir=storage_dir)

    store = StateStore(run_id="run_del_cp", storage_dir=storage_dir)
    store.append_trajectory({"action": "step_to_delete"})

    checkpoints = manager.list_checkpoints(run_id="run_del_cp")
    assert len(checkpoints) == 1
    cp_path = checkpoints[0]
    assert os.path.exists(cp_path)

    manager.delete_checkpoint(cp_path)
    assert not os.path.exists(cp_path)
    assert len(manager.list_checkpoints(run_id="run_del_cp")) == 0

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        manager.delete_checkpoint(cp_path)

def test_checkpoint_manager_strict_run_id_filtering(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(storage_dir=storage_dir)

    store1 = StateStore(run_id="run_alpha", storage_dir=storage_dir)
    store1.append_trajectory({"action": "action_1"})

    store_similar = StateStore(run_id="run_alpha_extended", storage_dir=storage_dir)
    store_similar.append_trajectory({"action": "action_similar"})

    # Check filtering for "run_alpha"
    cps_run_alpha = manager.list_checkpoints(run_id="run_alpha")
    # Should match run_alpha but NOT run_alpha_extended
    assert len(cps_run_alpha) == 1
    assert "run_alpha_step_1.json" in cps_run_alpha[0]
    assert "run_alpha_extended" not in cps_run_alpha[0]
