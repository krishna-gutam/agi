import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash")

# --- TOOL DEFINITIONS ---

TOOLS = [
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

def execute_tool(name, arguments):
    try:
        if name == "list_files":
            path = arguments.get("path", ".")
            files = os.listdir(path)
            return json.dumps({"files": files})
        elif name == "read_file":
            file_path = arguments.get("file_path")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return json.dumps({"content": content})
        elif name == "write_file":
            file_path = arguments.get("file_path")
            content = arguments.get("content")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return json.dumps({"status": "success", "file_path": file_path})
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# --- OPENAI REST API WITH TOOLS ---

def openai_tools_format():
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        }
    } for t in TOOLS]

def call_openai_with_tools(messages, system_prompt=None):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "model": OPENAI_MODEL_ID,
        "messages": full_messages,
        "tools": openai_tools_format(),
        "tool_choice": "auto"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenAI API Error ({response.status_code}): {response.text}")
    return response.json()["choices"][0]["message"]

# --- GEMINI REST API WITH TOOLS ---

def gemini_tools_format():
    # Gemini tool declaration format for REST API
    declarations = []
    for t in TOOLS:
        declarations.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        })
    return [{"functionDeclarations": declarations}]

def call_gemini_with_tools(contents, system_prompt=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_ID}:generateContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": contents,
        "tools": gemini_tools_format()
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Gemini API Error ({response.status_code}): {response.text}")
    return response.json()

# --- AGENT RUNNER ---

def main():
    print("=== Multi-Provider CLI Agent with Tools (REST API) ===")
    provider = input("Choose provider [openai / gemini]: ").strip().lower()
    if provider not in ["openai", "gemini"]:
        print("Invalid provider. Defaulting to openai.")
        provider = "openai"

    system_prompt = input("Enter system prompt (optional): ").strip() or None

    print(f"\nAgent initialized using {provider.upper()}. Available tools: list_files, read_file, write_file.")
    print("Type 'exit' or 'quit' to end.\n")

    if provider == "openai":
        messages = []
        while True:
            try:
                user_input = input(f"[openai]> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                messages.append({"role": "user", "content": user_input})
                
                response_message = call_openai_with_tools(messages, system_prompt)
                messages.append(response_message)

                if response_message.get("tool_calls"):
                    for tool_call in response_message["tool_calls"]:
                        func_name = tool_call["function"]["name"]
                        func_args = json.loads(tool_call["function"]["arguments"])
                        call_id = tool_call["id"]

                        print(f"\n[Tool Execution] Calling {func_name} with {func_args}...")
                        tool_result = execute_tool(func_name, func_args)
                        print(f"[Tool Result] {tool_result}\n")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result
                        })

                    # Follow-up call after tool execution
                    follow_up = call_openai_with_tools(messages, system_prompt)
                    messages.append(follow_up)
                    print(f"\n{follow_up.get('content')}\n")
                else:
                    print(f"\n{response_message.get('content')}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    else: # gemini
        contents = []
        while True:
            try:
                user_input = input(f"[gemini]> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                contents.append({
                    "role": "user",
                    "parts": [{"text": user_input}]
                })

                data = call_gemini_with_tools(contents, system_prompt)
                candidate = data["candidates"][0]
                content_part = candidate["content"]
                contents.append(content_part)

                parts = content_part.get("parts", [])
                
                # Check for function call
                has_function_call = False
                for part in parts:
                    if "functionCall" in part:
                        has_function_call = True
                        fc = part["functionCall"]
                        func_name = fc["name"]
                        func_args = fc.get("args", {})

                        print(f"\n[Tool Execution] Calling {func_name} with {func_args}...")
                        tool_result_str = execute_tool(func_name, func_args)
                        print(f"[Tool Result] {tool_result_str}\n")

                        # Parse result for Gemini response
                        try:
                            tool_response_json = json.loads(tool_result_str)
                        except:
                            tool_response_json = {"result": tool_result_str}

                        contents.append({
                            "role": "function",
                            "parts": [{
                                "functionResponse": {
                                    "name": func_name,
                                    "response": tool_response_json
                                }
                            }]
                        })

                if has_function_call:
                    follow_up_data = call_gemini_with_tools(contents, system_prompt)
                    follow_up_candidate = follow_up_data["candidates"][0]
                    follow_up_content = follow_up_candidate["content"]
                    contents.append(follow_up_content)
                    
                    final_text = "".join([p.get("text", "") for p in follow_up_content.get("parts", [])])
                    print(f"\n{final_text}\n")
                else:
                    final_text = "".join([p.get("text", "") for p in parts])
                    print(f"\n{final_text}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
