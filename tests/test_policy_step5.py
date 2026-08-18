import os
import tempfile
import pytest
from agent.policy import PolicyEngine, PolicyDecision

def test_policy_allow_and_deny():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PolicyEngine(workspace_dir=tmpdir)

        # fs.read inside workspace -> ALLOW
        in_path = os.path.join(tmpdir, "test.txt")
        assert engine.evaluate("fs_read", {"path": in_path}) == PolicyDecision.ALLOW

        # fs.write inside workspace -> APPROVE (by default rule)
        assert engine.evaluate("fs_write", {"path": in_path, "content": "hello"}) == PolicyDecision.APPROVE

        # fs.read outside workspace -> DENY
        out_path = os.path.abspath(os.path.join(tmpdir, "..", "outside.txt"))
        assert engine.evaluate("fs_read", {"path": out_path}) == PolicyDecision.DENY

def test_policy_approval_callback():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Callback that approves if content starts with "safe" or equals "approved", denies otherwise
        def mock_approval(tool_name, args):
            return args.get("content") == "approved content"

        engine = PolicyEngine(workspace_dir=tmpdir, approval_callback=mock_approval)
        in_path = os.path.join(tmpdir, "test.txt")

        # APPROVE rule with callback returning False -> DENY
        decision_denied = engine.evaluate("fs_write", {"path": in_path, "content": "unapproved content"})
        assert decision_denied == PolicyDecision.DENY

        # APPROVE rule with callback returning True -> APPROVE
        decision_approved = engine.evaluate("fs_write", {"path": in_path, "content": "approved content"})
        assert decision_approved == PolicyDecision.APPROVE

def test_shell_exec_approval():
    engine = PolicyEngine()
    assert engine.evaluate("shell_exec", {"command": "ls"}) == PolicyDecision.APPROVE
