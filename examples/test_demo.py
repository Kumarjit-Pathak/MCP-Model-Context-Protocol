"""
Test Demo - Automated testing of MCP client with all servers
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
    print("MCP CLIENT - Automated Test Demo")
    print("=" * 70)
    settings = get_client_settings()
    print(f"LLM Provider: {settings.llm_provider.upper()}")
    if settings.llm_provider == "openai":
        print(f"Model: {settings.openai_model}")
    print("=" * 70)
    print()


async def test_server(server_name: str, server_path: str, test_query: str):
    """Test a single server with a query."""
    print(f"\n{'='*70}")
    print(f"Testing {server_name}")
    print(f"Server: {server_path}")
    print(f"{'='*70}\n")

    client = MCPClient()

    try:
        # Connect to server
        print(f"[CONNECT] Connecting to {server_name}...")
        await client.connect_to_server(server_path)
        print(f"[OK] Connected successfully")
        print(f"[TOOLS] Available: {', '.join(client.get_available_tools())}\n")

        # Run test query
        print(f"[TEST] Query: {test_query}")
        response = await client.chat(test_query)

        print(f"\n[SUCCESS] {server_name} test completed!\n")
        print("-" * 70)

        return True

    except Exception as e:
        print(f"\n[ERROR] {server_name} test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await client.disconnect()


async def main():
    """Main entry point."""
    print_banner()

    # Define test cases
    tests = [
        {
            "name": "Weather Server",
            "path": "server/weather_server.py",
            "query": "What's the weather like in London?",
        },
        {
            "name": "News Server",
            "path": "server/news_server.py",
            "query": "Show me the top technology headlines",
        },
        {
            "name": "Finance Server",
            "path": "server/finance_server.py",
            "query": "Convert 100 USD to EUR",
        },
    ]

    results = []

    # Run all tests
    for test in tests:
        result = await test_server(test["name"], test["path"], test["query"])
        results.append((test["name"], result))

        # Wait a bit between tests
        await asyncio.sleep(2)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    total = len(results)
    passed = sum(1 for _, result in results if result)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
