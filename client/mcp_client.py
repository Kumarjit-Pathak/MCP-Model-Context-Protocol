"""
MCP Client - Connects to MCP servers and uses LLM for tool calling
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from client.config import get_client_settings
from client.llm_provider import create_llm_provider, LLMProvider


class MCPClient:
    """MCP Client that connects to servers and uses LLM to invoke tools."""

    def __init__(self):
        """Initialize MCP Client."""
        self.settings = get_client_settings()
        self.tools: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.session: Optional[ClientSession] = None
        self.server_process = None

        # Initialize LLM provider
        if self.settings.llm_provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.llm = create_llm_provider(
                provider_name="anthropic",
                api_key=self.settings.anthropic_api_key,
            )
        elif self.settings.llm_provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.llm = create_llm_provider(
                provider_name="openai",
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
                base_url=self.settings.openai_base_url,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")

    async def connect_to_server(self, server_script: str) -> None:
        """Connect to an MCP server via stdio."""
        server = StdioServerParameters(
            command="python",
            args=[server_script],
            env=None,
        )

        # Create stdio client using context manager
        stdio = stdio_client(server)
        read_stream, write_stream = await stdio.__aenter__()

        # Store the context manager for cleanup
        self._stdio_context = stdio

        # Create session
        self.session = ClientSession(read_stream, write_stream)

        # Initialize session
        await self.session.__aenter__()
        await self.session.initialize()

        # Discover tools
        tools_result = await self.session.list_tools()

        # Convert tools to format expected by LLM
        for tool in tools_result.tools:
            self.tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema,
            })

        if self.settings.debug:
            print(f"[OK] Connected to server and discovered {len(self.tools)} tools")
            print(f"   Available tools: {', '.join([t['name'] for t in self.tools])}")

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, '_stdio_context') and self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)

    async def _invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool on the MCP server."""
        if not self.session:
            return {"success": False, "error": "Not connected to server"}

        try:
            if self.settings.debug:
                print(f"[TOOL] Invoking tool: {tool_name}")
                print(f"   Arguments: {json.dumps(arguments, indent=2)}")

            # Call tool via MCP
            result = await self.session.call_tool(tool_name, arguments)

            if self.settings.debug:
                result_str = str(result.content)[:200] if result.content else "No content"
                print(f"[OK] Tool result: {result_str}...")

            # Extract text content
            content = ""
            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        content += item.text

            return {"success": True, "data": content}

        except Exception as e:
            error_msg = str(e)
            if self.settings.debug:
                print(f"[ERROR] Tool invocation error: {error_msg}")
            return {"success": False, "error": error_msg}

    async def chat(self, user_message: str) -> str:
        """Process a user message using LLM and tool calling."""
        # Add user message to conversation
        self.conversation_history.append({"role": "user", "content": user_message})

        if self.settings.debug:
            print(f"\n[USER] {user_message}\n")

        # Iterative tool calling loop
        for iteration in range(self.settings.max_iterations):
            if self.settings.debug:
                print(f"[ITER] Iteration {iteration + 1}/{self.settings.max_iterations}")

            # Get LLM response
            response = self.llm.create_chat_completion(
                messages=self.conversation_history,
                tools=self.tools if self.tools else None,
            )

            # Check if LLM wants to call tools
            if response["tool_calls"]:
                if self.settings.debug:
                    print(f"[LLM] LLM wants to call {len(response['tool_calls'])} tool(s)")

                # Execute each tool call
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]

                    # Invoke tool on MCP server
                    result = await self._invoke_tool(tool_name, tool_args)

                    # Add tool result to conversation
                    if self.settings.llm_provider == "anthropic":
                        # Anthropic format: assistant message with tool_use, then user message with tool_result
                        if not any(
                            msg.get("role") == "assistant"
                            and any(
                                isinstance(c, dict) and c.get("type") == "tool_use"
                                for c in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                            )
                            for msg in self.conversation_history[-1:]
                        ):
                            # Add assistant message with tool use
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": [{
                                    "type": "tool_use",
                                    "id": tool_call["id"],
                                    "name": tool_name,
                                    "input": tool_args,
                                }]
                            })

                        # Add tool result as user message
                        self.conversation_history.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_call["id"],
                                "content": json.dumps(result),
                            }]
                        })

                    elif self.settings.llm_provider == "openai":
                        # OpenAI format: assistant message with tool_calls, then tool message
                        if not any(
                            msg.get("role") == "assistant" and msg.get("tool_calls")
                            for msg in self.conversation_history[-1:]
                        ):
                            # Add assistant message with tool calls
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_args),
                                    }
                                }]
                            })

                        # Add tool result as tool message
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result),
                        })

                # Continue loop to get LLM's next response
                continue

            else:
                # No more tool calls - LLM has final response
                final_response = response["content"]

                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response,
                })

                if self.settings.debug:
                    print(f"\n[ASSISTANT] {final_response}\n")

                return final_response

        # Max iterations reached
        return "I apologize, but I've reached the maximum number of tool calls. Please try rephrasing your question."

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        if self.settings.debug:
            print("[RESET] Conversation reset")

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool["name"] for tool in self.tools]

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        for tool in self.tools:
            if tool["name"] == tool_name:
                return tool
        return None
