import os
import json
import time
from typing import Dict, Any, List, Optional

class StateStore:
    """
    State Store adhering to Sections 3.1, 4.2, and 12 for agent state, memory, 
    trajectory persistence, checkpointing, auto-saving per step, and budget enforcement.
    """
    def __init__(
        self,
        run_id: Optional[str] = None,
        storage_dir: str = ".checkpoints",
        budget: Optional[Dict[str, Any]] = None,
        auto_checkpoint: bool = True
    ):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.storage_dir = storage_dir
        self.auto_checkpoint = auto_checkpoint
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Budget configuration per Section 12 (steps, tokens, cost) with ~10% headroom buffer
        default_budget = {
            "max_steps": 20,
            "max_tokens": 100000,
            "max_cost_usd": 1.0,
            "headroom_factor": 0.9  # Triggers graceful halt at 90% of budget
        }
        self.budget = {**default_budget, **(budget or {})}
        
        self.state: Dict[str, Any] = {
            "run_id": self.run_id,
            "status": "initialized",
            "step_count": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "messages": [],
            "trajectory": [],
            "memory": {},
            "metadata": {},
            "graceful_halt_triggered": False
        }

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def update(self, data: Dict[str, Any]) -> None:
        self.state.update(data)

    def append_trajectory(self, step_data: Dict[str, Any]) -> str:
        """Appends step data to trajectory and auto-saves checkpoint per Section 12."""
        step_data.setdefault("timestamp", time.time())
        step_index = len(self.state["trajectory"]) + 1
        step_data.setdefault("step_index", step_index)
        
        # Accumulate usage if present in step_data
        usage = step_data.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        cost = step_data.get("cost_usd", 0.0)
        
        self.state["total_tokens"] += tokens
        self.state["total_cost_usd"] += cost
        
        self.state["trajectory"].append(step_data)
        self.state["step_count"] = len(self.state["trajectory"])

        cp_path = ""
        if self.auto_checkpoint:
            cp_path = self.save_checkpoint(tag=f"step_{self.state['step_count']}")
        return cp_path

    def check_budget_exceeded(self) -> bool:
        """
        Checks if any budget threshold (steps, tokens, or cost) has been reached or exceeded,
        incorporating a ~10% headroom buffer as specified in Section 12.
        """
        headroom = self.budget.get("headroom_factor", 0.9)
        
        max_steps = self.budget.get("max_steps")
        if max_steps and self.state["step_count"] >= (max_steps * headroom):
            return True
            
        max_tokens = self.budget.get("max_tokens")
        if max_tokens and self.state["total_tokens"] >= (max_tokens * headroom):
            return True
            
        max_cost = self.budget.get("max_cost_usd")
        if max_cost and self.state["total_cost_usd"] >= (max_cost * headroom):
            return True
            
        return False

    def should_graceful_halt(self) -> bool:
        """Returns True if graceful halt should be triggered."""
        if self.state.get("graceful_halt_triggered"):
            return False  # Already triggered/handled
        return self.check_budget_exceeded()

    def mark_graceful_halt(self) -> None:
        """Marks graceful halt as triggered and updates status."""
        self.state["graceful_halt_triggered"] = True
        self.state["status"] = "graceful_halt"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.state)

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """Saves current state as a checkpoint JSON file."""
        tag = tag or f"step_{self.state['step_count']}"
        filename = f"{self.run_id}_{tag}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        checkpoint_data = {
            "version": "1.0",
            "timestamp": time.time(),
            "state": self.state
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)
            
        return filepath

    def load_checkpoint(self, filepath: str) -> None:
        """Loads state from a checkpoint file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "state" in data:
            self.state = data["state"]
        else:
            self.state = data


class CheckpointManager:
    """Manages multiple checkpoints and trajectory persistence."""
    def __init__(self, storage_dir: str = ".checkpoints"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def list_checkpoints(self, run_id: Optional[str] = None) -> List[str]:
        files = os.listdir(self.storage_dir)
        checkpoints = []
        for f in files:
            if f.endswith(".json"):
                filepath = os.path.join(self.storage_dir, f)
                if run_id:
                    try:
                        with open(filepath, "r", encoding="utf-8") as jf:
                            jdata = json.load(jf)
                            stored_run_id = jdata.get("state", {}).get("run_id") or jdata.get("run_id")
                            if stored_run_id != run_id:
                                continue
                    except Exception:
                        # Skip malformed JSON files when filtering by run_id
                        continue
                checkpoints.append(filepath)
        return sorted(checkpoints)

    def delete_checkpoint(self, filepath: str) -> None:
        """Deletes a specific checkpoint file, raising FileNotFoundError if it does not exist."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        os.remove(filepath)

    def export_trajectory_viewer_data(self, store: StateStore) -> Dict[str, Any]:
        """Exports data structured specifically for trajectory viewer UI / inspection."""
        state = store.to_dict()
        return {
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "step_count": state.get("step_count"),
            "total_tokens": state.get("total_tokens"),
            "total_cost_usd": state.get("total_cost_usd"),
            "graceful_halt_triggered": state.get("graceful_halt_triggered"),
            "trajectory": state.get("trajectory", []),
            "memory_summary": {k: str(v)[:100] for k, v in state.get("memory", {}).items()}
        }
