import os
from enum import Enum
from typing import Any, Dict, List, Optional

class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    APPROVE = "APPROVE"
    DENY = "DENY"

class PolicyEngine:
    """
    Policy Engine that evaluates proposed tool calls against the run's permission 
    envelope and tool effects (ALLOW | APPROVE | DENY), handling approval gates 
    and workspace boundary checks.
    """
    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        default_policy: PolicyDecision = PolicyDecision.ALLOW,
        custom_rules: Optional[Dict[str, PolicyDecision]] = None,
        approval_callback: Optional[Any] = None
    ):
        self.workspace_dir = os.path.abspath(workspace_dir or os.getcwd())
        self.default_policy = default_policy
        self.custom_rules = custom_rules or {
            "fs.read": PolicyDecision.ALLOW,
            "fs.write": PolicyDecision.APPROVE,
            "shell.exec": PolicyDecision.APPROVE,
        }
        self.approval_callback = approval_callback

    def check_workspace_boundary(self, path: str) -> bool:
        """Verifies that the requested path is within the workspace directory."""
        try:
            abs_path = os.path.abspath(path)
            common = os.path.commonpath([self.workspace_dir, abs_path])
            return common == self.workspace_dir
        except Exception:
            return False

    def evaluate(self, tool_name: str, arguments: Dict[str, Any]) -> PolicyDecision:
        """Evaluate a tool call and return its policy decision."""
        # 1. Workspace boundary check for file operations
        if tool_name in ["fs.read", "fs.write"]:
            path = arguments.get("path")
            if path and not self.check_workspace_boundary(path):
                return PolicyDecision.DENY

        # 2. Check custom rules or tool defaults
        if tool_name in self.custom_rules:
            decision = self.custom_rules[tool_name]
        else:
            decision = self.default_policy

        # 3. Handle approval gates
        if decision == PolicyDecision.APPROVE and self.approval_callback:
            approved = self.approval_callback(tool_name, arguments)
            if not approved:
                return PolicyDecision.DENY
            return PolicyDecision.APPROVE

        return decision
