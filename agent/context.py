import json
from typing import Any, Dict, List, Optional
from agent.adapter import Message

class ContextManager:
    """
    Context Manager responsible for assembling the model payload each step 
    (system prompt, goal, tool schemas, working state, recent trajectory, retrieved memories)
    and implementing compaction at 70% utilization.
    """
    def __init__(self, max_context_tokens: int = 8000, compaction_threshold_factor: float = 0.7):
        self.max_context_tokens = max_context_tokens
        self.compaction_threshold_factor = compaction_threshold_factor

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (approx 4 chars per token)."""
        if not text:
            return 0
        return len(text) // 4

    def assemble_context(
        self,
        system_prompt: str,
        goal: str,
        constraints: Optional[List[str]] = None,
        current_plan: Optional[str] = None,
        working_state: Optional[Dict[str, Any]] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
        retrieved_memories: Optional[List[str]] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Message]:
        """Assembles messages list for the model payload."""
        messages: List[Message] = []

        # 1. System Prompt & Instructions
        sys_content = system_prompt or "You are an autonomous AI agent."
        messages.append(Message(role="system", content=sys_content))

        # 2. Goal and Constraints
        goal_content = f"GOAL:\n{goal}"
        if constraints:
            goal_content += "\n\nCONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraints)
        messages.append(Message(role="user", content=goal_content))

        # 3. Current Plan & Working State
        state_parts = []
        if current_plan:
            state_parts.append(f"CURRENT PLAN:\n{current_plan}")
        if working_state:
            state_parts.append(f"WORKING STATE:\n{json.dumps(working_state, indent=2)}")
        if retrieved_memories:
            state_parts.append(f"RETRIEVED MEMORIES:\n" + "\n".join(f"- {m}" for m in retrieved_memories))

        if state_parts:
            messages.append(Message(role="user", content="\n\n".join(state_parts)))

        # 4. Trajectory (with compaction if needed)
        traj = trajectory or []
        threshold_tokens = int(self.max_context_tokens * self.compaction_threshold_factor)
        
        # Estimate current size
        current_text = sys_content + goal_content + "\n".join(state_parts)
        for step in traj:
            current_text += json.dumps(step)

        if self.estimate_tokens(current_text) >= threshold_tokens and len(traj) > 3:
            # Compaction: preserve verbatim goal, constraints, current plan, and last 3 steps
            older_steps = traj[:-3]
            recent_steps = traj[-3:]

            summary_digest = "COMPACTED OLDER STEPS DIGEST:\n" + "; ".join(
                f"Step {s.get('step_index', i+1)}: {s.get('action', 'action')} -> exit {s.get('exit_code', 0)}"
                for i, s in enumerate(older_steps)
            )
            messages.append(Message(role="assistant", content=summary_digest))

            for step in recent_steps:
                messages.append(Message(role="assistant", content=json.dumps(step)))
        else:
            for step in traj:
                messages.append(Message(role="assistant", content=json.dumps(step)))

        return messages
