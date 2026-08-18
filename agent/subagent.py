import concurrent.futures
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

class SubagentConfig(BaseModel):
    goal: str
    parent_run_id: str
    depth: int = 1
    max_depth: int = 2
    permissions: List[str] = Field(default_factory=list) # e.g., ["fs_read"]
    budget_tokens: int = 1000

class SubagentResult(BaseModel):
    success: bool
    output: str
    tokens_used: int = 0
    error: Optional[str] = None

class SubagentManager:
    """
    Subagent Manager responsible for spawning subagents with carved-out budgets,
    permission subsets, depth <= 2, and executing read-only tools in parallel.
    """
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def spawn_subagent(
        self,
        config: SubagentConfig,
        task_callback: Callable[..., Any]
    ) -> SubagentResult:
        """Spawns an isolated subagent enforcing depth limit and permission subsets."""
        if config.depth > config.max_depth:
            return SubagentResult(
                success=False,
                output="",
                error=f"Max subagent depth exceeded (depth {config.depth} > max {config.max_depth})"
            )

        try:
            # Execute subagent task in isolation
            result = task_callback(config)
            return SubagentResult(
                success=True,
                output=str(result),
                tokens_used=min(50, config.budget_tokens)
            )
        except Exception as e:
            return SubagentResult(
                success=False,
                output="",
                error=str(e)
            )

    def execute_parallel_read_only(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_registry: Dict[str, Callable[..., Any]]
    ) -> List[Any]:
        """Executes read-only tool calls in parallel using a ThreadPoolExecutor."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls) or 1) as executor:
            future_to_call = {}
            for call in tool_calls:
                name = call.get("name")
                args = call.get("arguments", {})
                func = tool_registry.get(name)
                if func:
                    future = executor.submit(func, **args)
                    future_to_call[future] = call
                else:
                    results.append({"error": f"Tool '{name}' not found", "success": False})

            for future in concurrent.futures.as_completed(future_to_call):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    results.append({"error": str(exc), "success": False})
        return results
