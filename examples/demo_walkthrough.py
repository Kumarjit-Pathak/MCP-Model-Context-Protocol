"""
Interactive Demo Walkthrough - Shows the MCP client in action
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.mcp_client import MCPClient


async def demo_weather_server():
    """Demo the weather server capabilities."""
    print("=" * 70)
    print("DEMO 1: WEATHER SERVER")
    print("=" * 70)
    print()

    client = MCPClient()

    try:
        print("[STEP 1] Connecting to Weather Server...")
        await client.connect_to_server("server/weather_server.py")
        print()

        print("[STEP 2] Asking: 'What's the weather in London?'")
        print()
        response = await client.chat("What's the weather in London?")
        print()

        print("[STEP 3] Asking for forecast: 'What's the 3-day forecast for Tokyo?'")
        print()
        client.reset_conversation()
        response = await client.chat("What's the 3-day forecast for Tokyo?")
        print()

    finally:
        await client.disconnect()


async def demo_news_server():
    """Demo the news server capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 2: NEWS SERVER")
    print("=" * 70)
    print()

    client = MCPClient()

    try:
        print("[STEP 1] Connecting to News Server...")
        await client.connect_to_server("server/news_server.py")
        print()

        print("[STEP 2] Asking: 'Show me the top technology headlines'")
        print()
        response = await client.chat("Show me the top technology headlines")
        print()

        print("[STEP 3] Searching news: 'Find articles about artificial intelligence'")
        print()
        client.reset_conversation()
        response = await client.chat("Find articles about artificial intelligence")
        print()

    finally:
        await client.disconnect()


async def demo_finance_server():
    """Demo the finance server capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 3: FINANCE SERVER")
    print("=" * 70)
    print()

    client = MCPClient()

    try:
        print("[STEP 1] Connecting to Finance Server...")
        await client.connect_to_server("server/finance_server.py")
        print()

        print("[STEP 2] Asking: 'Convert 100 USD to EUR'")
        print()
        response = await client.chat("Convert 100 USD to EUR")
        print()

        print("[STEP 3] Asking: 'What are the current exchange rates for USD?'")
        print()
        client.reset_conversation()
        response = await client.chat("What are the current exchange rates for USD?")
        print()

    finally:
        await client.disconnect()


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("MCP INTERACTIVE DEMO WALKTHROUGH")
    print("=" * 70)
    print()
    print("This demo shows how the MCP client:")
    print("  1. Connects to MCP servers")
    print("  2. Discovers available tools")
    print("  3. Uses LLM to select and call tools")
    print("  4. Returns formatted responses")
    print()
    input("Press Enter to start the demo...")

    # Run all demos
    await demo_weather_server()
    await asyncio.sleep(2)

    await demo_news_server()
    await asyncio.sleep(2)

    await demo_finance_server()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Key Observations:")
    print("  - Each server starts independently")
    print("  - Tools are discovered automatically")
    print("  - LLM intelligently selects the right tool")
    print("  - Results are formatted naturally")
    print()
    print("To try it yourself:")
    print("  uv run python examples/interactive_client.py")
    print()


if __name__ == "__main__":
    asyncio.run(main())
