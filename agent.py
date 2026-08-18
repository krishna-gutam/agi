import sys
from dotenv import load_dotenv
from model_adapter import get_adapter
from orchestrator import AgentOrchestrator
from tools import TOOLS

load_dotenv()


def main():
    print("=== Multi-Provider CLI Agent with Tools ===")
    provider = input("Choose provider [openai / gemini]: ").strip().lower()
    if provider not in ["openai", "gemini"]:
        print("Invalid provider. Defaulting to openai.")
        provider = "openai"

    try:
        adapter = get_adapter(provider)
    except Exception as e:
        print(f"Failed to initialize adapter for {provider}: {e}")
        sys.exit(1)

    system_prompt = input("Enter system prompt (optional): ").strip() or None

    orchestrator = AgentOrchestrator(
        adapter=adapter,
        tools=TOOLS,
        system_prompt=system_prompt
    )

    print(f"\nAgent initialized using {provider.upper()}. Available tools: list_files, read_file, write_file.")
    print("Type 'exit' or 'quit' to end.\n")

    def on_tool_call(name, args):
        print(f"\n[Tool Execution] Calling {name} with {args}...")

    def on_tool_result(name, result_str):
        print(f"[Tool Result] {result_str}\n")

    while True:
        try:
            user_input = input(f"[{provider}]> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            response = orchestrator.run_turn(
                user_input=user_input,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result
            )
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
