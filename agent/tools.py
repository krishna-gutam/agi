import os
import subprocess
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    output: str
    error: Optional[str] = None
    exit_code: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

def fs_read(path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> ToolResult:
    """Read contents of a file."""
    try:
        if not os.path.exists(path):
            return ToolResult(output="", error=f"File not found: {path}", exit_code=1)
        if not os.path.isfile(path):
            return ToolResult(output="", error=f"Path is not a file: {path}", exit_code=1)
        
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        start = offset or 0
        end = start + limit if limit is not None else len(lines)
        selected_lines = lines[start:end]
        content = "".join(selected_lines)
        
        return ToolResult(
            output=content,
            error=None,
            exit_code=0,
            metadata={"total_lines": len(lines), "offset": start, "limit": limit}
        )
    except Exception as e:
        return ToolResult(output="", error=str(e), exit_code=1)

def fs_write(path: str, content: str, append: bool = False) -> ToolResult:
    """Write content to a file."""
    try:
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
            
        return ToolResult(
            output=f"Successfully wrote to {path}",
            error=None,
            exit_code=0,
            metadata={"path": path, "bytes_written": len(content.encode("utf-8"))}
        )
    except Exception as e:
        return ToolResult(output="", error=str(e), exit_code=1)

def shell_exec(command: str, timeout: Optional[int] = 60, cwd: Optional[str] = None) -> ToolResult:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        exit_code = result.returncode
        output = result.stdout
        error = result.stderr if result.stderr else None
        
        return ToolResult(
            output=output,
            error=error,
            exit_code=exit_code,
            metadata={"command": command, "cwd": cwd or os.getcwd()}
        )
    except subprocess.TimeoutExpired:
        return ToolResult(output="", error=f"Command timed out after {timeout} seconds", exit_code=-1)
    except Exception as e:
        return ToolResult(output="", error=str(e), exit_code=1)

# Registry of available tools
TOOL_REGISTRY = {
    "fs_read": fs_read,
    "fs_write": fs_write,
    "shell_exec": shell_exec,
}

def get_tool_schemas() -> List[Dict[str, Any]]:
    """Generate JSON-compatible tool schemas with real JSON Schema parameters."""
    from pydantic import TypeAdapter
    schemas = []
    for name, func in TOOL_REGISTRY.items():
        parameters = TypeAdapter(func).json_schema()
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": func.__doc__ or "",
                "parameters": parameters
            }
        })
    return schemas

