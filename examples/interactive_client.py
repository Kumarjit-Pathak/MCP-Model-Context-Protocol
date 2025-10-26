"""
Interactive MCP Client Demo - Chat with MCP servers using LLMs
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.mcp_client import MCPClient
from client.config import get_client_settings


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("MCP CLIENT - Interactive Demo")
    print("=" * 70)
    settings = get_client_settings()
    print(f"LLM Provider: {settings.llm_provider.upper()}")
    if settings.llm_provider == "openai":
        print(f"Model: {settings.openai_model}")
    print("=" * 70)
    print()


async def run_interactive_mode(client: MCPClient, server_path: str):
    """Run interactive chat mode."""
    print(f"\n[INIT] Connecting to MCP server: {server_path}")

    try:
        await client.connect_to_server(server_path)
    except Exception as e:
        print(f"[ERROR] Failed to connect to server: {e}")
        return

    print("\n[INTERACTIVE] Mode")
    print("=" * 70)
    print("Type your questions below. Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'reset' - Clear conversation history")
    print("  'tools' - Show available tools")
    print("=" * 70)
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("\n[EXIT] Goodbye!")
                    break

                if user_input.lower() == "reset":
                    client.reset_conversation()
                    print("[RESET] Conversation reset\n")
                    continue

                if user_input.lower() == "tools":
                    print("\n[TOOLS] Available Tools:")
                    for tool_name in client.get_available_tools():
                        tool_info = client.get_tool_info(tool_name)
                        print(f"  - {tool_name}: {tool_info.get('description', 'No description')}")
                    print()
                    continue

                # Process user query
                response = await client.chat(user_input)

            except KeyboardInterrupt:
                print("\n\n[EXIT] Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] Error: {e}\n")

    finally:
        await client.disconnect()


async def main():
    """Main entry point."""
    print_banner()

    # Check command line arguments for server selection
    if len(sys.argv) > 1:
        server_choice = sys.argv[1]
    else:
        print("Select MCP Server:")
        print("  1. Weather Server (weather data)")
        print("  2. News Server (news headlines)")
        print("  3. Finance Server (currency exchange)")
        print()
        server_choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"

    # Map choice to server script
    server_map = {
        "1": "server/weather_server.py",
        "weather": "server/weather_server.py",
        "2": "server/news_server.py",
        "news": "server/news_server.py",
        "3": "server/finance_server.py",
        "finance": "server/finance_server.py",
    }

    server_path = server_map.get(server_choice.lower(), "server/weather_server.py")

    try:
        # Initialize MCP Client
        print("[INIT] Initializing MCP Client...")
        client = MCPClient()
        print("[OK] MCP Client initialized successfully\n")

        # Run interactive mode
        await run_interactive_mode(client, server_path)

    except KeyboardInterrupt:
        print("\n\n[EXIT] Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
