"""
LLM Provider abstraction supporting both Anthropic Claude and OpenAI
"""

import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a chat completion with optional tool calling."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the Messages API."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Anthropic provider."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: uv add anthropic"
            )

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _convert_tools_to_anthropic_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tool schemas to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {}),
            })
        return anthropic_tools

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create chat completion using Anthropic API."""

        # Prepare tool definitions if provided
        anthropic_tools = None
        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)

        # Call Anthropic API
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = self.client.messages.create(**kwargs)

        # Parse response
        result = {
            "content": "",
            "tool_calls": [],
            "stop_reason": response.stop_reason,
        }

        # Extract text content and tool calls
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return result


class OpenAIProvider(LLMProvider):
    """OpenAI provider using the Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        """Initialize OpenAI provider."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: uv add openai"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _convert_tools_to_openai_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tool schemas to OpenAI format."""
        openai_tools = []
        for tool in tools:
            # Get input schema and ensure 'required' field is present
            input_schema = tool.get("inputSchema", {})
            if isinstance(input_schema, dict):
                input_schema = input_schema.copy()
                # OpenAI requires 'required' to be an array (even if empty)
                if "required" not in input_schema:
                    input_schema["required"] = []

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": input_schema,
                },
            })
        return openai_tools

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create chat completion using OpenAI API."""

        # Prepare tool definitions if provided
        openai_tools = None
        if tools:
            openai_tools = self._convert_tools_to_openai_format(tools)

        # Call OpenAI API
        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        # Parse response
        message = response.choices[0].message
        result = {
            "content": message.content or "",
            "tool_calls": [],
            "stop_reason": response.choices[0].finish_reason,
        }

        # Extract tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                })

        return result


def create_llm_provider(
    provider_name: str,
    api_key: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """Factory function to create appropriate LLM provider."""
    if provider_name == "anthropic":
        model = model or "claude-3-5-sonnet-20241022"
        return AnthropicProvider(api_key=api_key, model=model)
    elif provider_name == "openai":
        model = model or "openai/gpt-4o-mini"
        base_url = base_url or "https://api.openai.com/v1"
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
