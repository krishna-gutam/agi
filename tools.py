import os
import json
from typing import Any, Dict, List

# --- TOOL SCHEMA DEFINITIONS ---

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List files and directories in the given path (relative to current directory).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to current directory '.'",
                    "default": "."
                }
            }
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a specified file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read."
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a specified file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write into the file."
                }
            },
            "required": ["file_path", "content"]
        }
    }
]

# --- TOOL IMPLEMENTATIONS ---

def list_files(path: str = ".") -> Dict[str, Any]:
    """Lists files and directories at the specified path."""
    return {"files": os.listdir(path)}


def read_file(file_path: str) -> Dict[str, Any]:
    """Reads content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content}


def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """Writes content to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "success", "file_path": file_path}


# --- TOOL DISPATCHER ---

TOOL_HANDLERS = {
    "list_files": lambda args: list_files(args.get("path", ".")),
    "read_file": lambda args: read_file(args.get("file_path")),
    "write_file": lambda args: write_file(args.get("file_path"), args.get("content")),
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Executes the specified tool by name with arguments and returns a JSON string result.
    """
    try:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {name}"})
        result = handler(arguments)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
