# Complete Guide: Building Production MCP Servers with FastMCP

> **A comprehensive, step-by-step guide from zero to production-ready MCP implementation**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Part 1: Initial Setup](#part-1-initial-setup)
4. [Part 2: Building MCP Servers](#part-2-building-mcp-servers)
5. [Part 3: Building MCP Client](#part-3-building-mcp-client)
6. [Part 4: Testing & Demo](#part-4-testing--demo)
7. [Part 5: Extending for Production](#part-5-extending-for-production)
8. [Part 6: LangSmith Monitoring](#part-6-langsmith-monitoring)
9. [Part 7: Deployment](#part-7-deployment)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### What You'll Build

By the end of this guide, you'll have:
- ✅ **3 Production MCP Servers** (Weather, News, Finance)
- ✅ **Intelligent MCP Client** with LLM integration
- ✅ **Interactive Chat Interface**
- ✅ **Monitoring with LangSmith**
- ✅ **Ready-to-extend architecture** for custom tools

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
│              "What's the weather in London?"                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Receives query                                   │  │
│  │  2. Sends to LLM (OpenAI/Anthropic)                 │  │
│  │  3. LLM selects tool: get_current_weather           │  │
│  │  4. Invokes tool via MCP protocol                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ stdio
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (Weather)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  @mcp.tool()                                         │  │
│  │  async def get_current_weather(location, units):    │  │
│  │      # Call OpenWeatherMap API                      │  │
│  │      return formatted_weather_data                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    External API                             │
│              (OpenWeatherMap, NewsAPI, etc.)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Software

- **Python 3.13+** - Latest Python release
- **UV** - Fast Python package manager
- **Git** - Version control
- **Code Editor** - VS Code recommended

### API Keys (Get Free Accounts)

1. **LLM Providers** (Required - choose one):
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/

2. **External APIs** (Optional - for real data):
   - OpenWeatherMap: https://openweathermap.org/api
   - NewsAPI: https://newsapi.org
   - ExchangeRate-API: https://www.exchangerate-api.com

---

## Part 1: Initial Setup

### Step 1.1: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/fastmcp-implementation.git
cd fastmcp-implementation

# Or start fresh
mkdir fastmcp-implementation
cd fastmcp-implementation
git init
```

### Step 1.2: Install UV (if not already installed)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Step 1.3: Initialize Project with Python 3.13

```bash
# Initialize UV project
uv init --python 3.13

# Create directory structure
mkdir -p server client examples tests
touch server/__init__.py client/__init__.py examples/__init__.py tests/__init__.py
```

### Step 1.4: Configure Dependencies

Create `pyproject.toml`:

```toml
[project]
name = "fastmcp-implementation"
version = "0.1.0"
description = "Production-ready MCP server and client using FastMCP"
requires-python = ">=3.13"
dependencies = [
    "mcp>=1.0.0",
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "langsmith>=0.1.0",  # For monitoring
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "black>=24.10.0",
    "mypy>=1.13.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["server", "client"]
```

### Step 1.5: Install Dependencies

```bash
# Install all dependencies
uv sync

# Verify installation
uv run python --version
# Should show: Python 3.13.x
```

### Step 1.6: Configure Environment Variables

Create `.env` file:

```bash
# LLM Provider Selection
LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Anthropic Configuration (alternative)
ANTHROPIC_API_KEY=your_anthropic_key_here

# External API Keys
OPENWEATHER_API_KEY=your_weather_key_here
NEWS_API_KEY=your_news_key_here
EXCHANGERATE_API_KEY=your_exchange_key_here

# LangSmith Configuration (for monitoring)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=mcp-production

# Client Settings
DEBUG=true
MAX_ITERATIONS=10
TIMEOUT=30
```

Create `.gitignore`:

```
# Python
__pycache__/
*.py[oc]
*.egg-info
.venv/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
```

---

## Part 2: Building MCP Servers

### Understanding FastMCP Pattern

FastMCP uses simple decorators to expose Python functions as MCP tools:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="my_server")

@mcp.tool()
async def my_tool(param: str) -> str:
    """Tool description for LLM."""
    return result
```

**Key Benefits:**
- ✅ Zero boilerplate - decorators handle everything
- ✅ Auto-generated schemas from docstrings
- ✅ Type hints become parameter validation
- ✅ Async/await for performance

---

### Step 2.1: Build Weather Server

Create `server/weather_server.py`:

```python
"""
Weather MCP Server - Provides weather data using OpenWeatherMap API
"""

import os
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(name="weather")

# Constants
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"


async def make_weather_request(url: str) -> dict[str, Any] | None:
    """Make a request to OpenWeatherMap API with error handling."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Weather API error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


@mcp.tool()
async def get_current_weather(location: str, units: str = "metric") -> str:
    """Get current weather conditions for any location.

    Args:
        location: City name (e.g., "London", "New York", "Tokyo")
        units: Temperature units - "metric" (Celsius), "imperial" (Fahrenheit), or "standard" (Kelvin)

    Returns:
        Current weather information including temperature, humidity, and conditions
    """
    if not OPENWEATHER_API_KEY:
        return "Error: OPENWEATHER_API_KEY not configured"

    url = f"{OPENWEATHER_BASE_URL}/weather?q={location}&appid={OPENWEATHER_API_KEY}&units={units}"
    data = await make_weather_request(url)

    if not data:
        return f"Unable to fetch weather data for {location}. Please check the city name."

    # Format response
    temp_unit = "°C" if units == "metric" else "°F" if units == "imperial" else "K"

    weather_info = f"""
Weather in {data['name']}, {data['sys']['country']}:
Temperature: {data['main']['temp']}{temp_unit}
Feels like: {data['main']['feels_like']}{temp_unit}
Conditions: {data['weather'][0]['description'].capitalize()}
Humidity: {data['main']['humidity']}%
Wind Speed: {data['wind']['speed']} {"m/s" if units == "metric" else "mph"}
"""
    return weather_info.strip()


@mcp.tool()
async def get_forecast(location: str, days: int = 5) -> str:
    """Get weather forecast for upcoming days.

    Args:
        location: City name (e.g., "London", "Paris")
        days: Number of days to forecast (1-5)

    Returns:
        Weather forecast for the specified number of days
    """
    if not OPENWEATHER_API_KEY:
        return "Error: OPENWEATHER_API_KEY not configured"

    # Limit days to 1-5
    days = max(1, min(5, days))

    url = f"{OPENWEATHER_BASE_URL}/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric&cnt={days * 8}"
    data = await make_weather_request(url)

    if not data:
        return f"Unable to fetch forecast for {location}. Please check the city name."

    # Parse forecast data (API returns 3-hour intervals)
    forecasts = []
    for i in range(0, min(len(data['list']), days * 8), 8):  # One forecast per day
        forecast_data = data['list'][i]
        date = forecast_data['dt_txt'].split()[0]
        temp = forecast_data['main']['temp']
        description = forecast_data['weather'][0]['description']

        forecasts.append(f"{date}: {temp}°C, {description.capitalize()}")

    forecast_text = f"Weather forecast for {data['city']['name']}, {data['city']['country']}:\n"
    forecast_text += "\n".join(forecasts)

    return forecast_text


# Run the server
if __name__ == "__main__":
    print("Starting Weather MCP Server...")
    print(f"API Key configured: {'Yes' if OPENWEATHER_API_KEY else 'No'}")
    mcp.run()
```

**Test the server:**

```bash
# Start the weather server
uv run python server/weather_server.py
```

You should see:
```
Starting Weather MCP Server...
API Key configured: Yes
```

---

### Step 2.2: Build News Server

Create `server/news_server.py`:

```python
"""
News MCP Server - Provides news headlines using NewsAPI
"""

import os
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name="news")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_BASE_URL = "https://newsapi.org/v2"


async def make_news_request(url: str) -> dict[str, Any] | None:
    """Make a request to NewsAPI with error handling."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"News API error: {e}")
            return None


@mcp.tool()
async def get_headlines(category: str = "general", country: str = "us") -> str:
    """Get top news headlines by category and country.

    Args:
        category: News category - "business", "technology", "sports", "entertainment", "health", "science", or "general"
        country: Two-letter country code (e.g., "us", "gb", "ca", "au")

    Returns:
        Top news headlines for the specified category and country
    """
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not configured"

    url = f"{NEWS_API_BASE_URL}/top-headlines?apiKey={NEWS_API_KEY}&country={country}&category={category}&pageSize=10"
    data = await make_news_request(url)

    if not data or data.get('status') != 'ok':
        return f"Unable to fetch news headlines."

    articles = data.get('articles', [])
    if not articles:
        return f"No headlines found for {category} news in {country.upper()}"

    headlines = [f"Top {category.capitalize()} Headlines ({country.upper()}):\n"]

    for i, article in enumerate(articles[:10], 1):
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown')
        headlines.append(f"{i}. {title}\n   Source: {source}")

    return "\n\n".join(headlines)


@mcp.tool()
async def search_news(query: str, language: str = "en", page_size: int = 10) -> str:
    """Search news articles by keyword.

    Args:
        query: Search keyword or phrase
        language: Language code (e.g., "en", "es", "fr")
        page_size: Number of results (1-100)

    Returns:
        News articles matching the search query
    """
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not configured"

    page_size = max(1, min(100, page_size))

    url = f"{NEWS_API_BASE_URL}/everything?apiKey={NEWS_API_KEY}&q={query}&language={language}&pageSize={page_size}&sortBy=publishedAt"
    data = await make_news_request(url)

    if not data or data.get('status') != 'ok':
        return f"Unable to search news."

    articles = data.get('articles', [])
    if not articles:
        return f"No articles found for query: '{query}'"

    results = [f"Search results for '{query}':\n"]

    for i, article in enumerate(articles[:page_size], 1):
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown')
        published = article.get('publishedAt', 'Unknown date')[:10]
        description = article.get('description', 'No description')[:150]

        results.append(f"{i}. {title}\n   Source: {source} | Date: {published}\n   {description}...")

    return "\n\n".join(results)


if __name__ == "__main__":
    print("Starting News MCP Server...")
    print(f"API Key configured: {'Yes' if NEWS_API_KEY else 'No'}")
    mcp.run()
```

---

### Step 2.3: Build Finance Server

Create `server/finance_server.py`:

```python
"""
Finance MCP Server - Provides currency exchange data using ExchangeRate-API
"""

import os
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name="finance")

EXCHANGERATE_API_KEY = os.getenv("EXCHANGERATE_API_KEY", "")
EXCHANGERATE_BASE_URL = "https://v6.exchangerate-api.com/v6"


async def make_exchange_request(url: str) -> dict[str, Any] | None:
    """Make a request to ExchangeRate-API with error handling."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Exchange API error: {e}")
            return None


@mcp.tool()
async def get_exchange_rates(base_currency: str = "USD") -> str:
    """Get current exchange rates for a base currency.

    Args:
        base_currency: Three-letter currency code (e.g., "USD", "EUR", "GBP")

    Returns:
        Current exchange rates for major currencies
    """
    if not EXCHANGERATE_API_KEY:
        return "Error: EXCHANGERATE_API_KEY not configured"

    url = f"{EXCHANGERATE_BASE_URL}/{EXCHANGERATE_API_KEY}/latest/{base_currency.upper()}"
    data = await make_exchange_request(url)

    if not data or data.get('result') != 'success':
        return f"Unable to fetch exchange rates for {base_currency}."

    rates = data.get('conversion_rates', {})
    last_update = data.get('time_last_update_utc', 'Unknown')

    major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'INR', 'MXN']
    rate_lines = [f"Exchange rates for 1 {base_currency.upper()}:"]
    rate_lines.append(f"Last updated: {last_update}\n")

    for currency in major_currencies:
        if currency in rates and currency != base_currency.upper():
            rate_lines.append(f"{currency}: {rates[currency]:.4f}")

    return "\n".join(rate_lines)


@mcp.tool()
async def convert_currency(from_currency: str, to_currency: str, amount: float) -> str:
    """Convert amount between two currencies.

    Args:
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR")
        amount: Amount to convert (must be positive)

    Returns:
        Converted amount with exchange rate information
    """
    if not EXCHANGERATE_API_KEY:
        return "Error: EXCHANGERATE_API_KEY not configured"

    if amount <= 0:
        return "Error: Amount must be positive"

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    url = f"{EXCHANGERATE_BASE_URL}/{EXCHANGERATE_API_KEY}/pair/{from_currency}/{to_currency}/{amount}"
    data = await make_exchange_request(url)

    if not data or data.get('result') != 'success':
        return f"Unable to convert {from_currency} to {to_currency}."

    conversion_rate = data.get('conversion_rate', 0)
    conversion_result = data.get('conversion_result', 0)
    last_update = data.get('time_last_update_utc', 'Unknown')

    result = f"""
Currency Conversion:
{amount:,.2f} {from_currency} = {conversion_result:,.2f} {to_currency}

Exchange Rate: 1 {from_currency} = {conversion_rate:.6f} {to_currency}
Last updated: {last_update}
"""
    return result.strip()


if __name__ == "__main__":
    print("Starting Finance MCP Server...")
    print(f"API Key configured: {'Yes' if EXCHANGERATE_API_KEY else 'No'}")
    mcp.run()
```

---

## Part 3: Building MCP Client

### Step 3.1: Client Configuration

Create `client/config.py`:

```python
"""
Client Configuration - Manages settings and environment variables
"""

from pydantic_settings import BaseSettings
from typing import Literal


class ClientSettings(BaseSettings):
    """Client configuration loaded from environment variables."""

    # LLM Provider Selection
    llm_provider: Literal["anthropic", "openai"] = "openai"

    # Anthropic Configuration
    anthropic_api_key: str = ""

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # LangSmith Configuration
    langsmith_api_key: str = ""
    langsmith_project: str = "mcp-production"

    # Client Settings
    debug: bool = True
    max_iterations: int = 10
    timeout: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


def get_client_settings() -> ClientSettings:
    """Get client settings instance."""
    return ClientSettings()
```

---

### Step 3.2: LLM Provider Abstraction

Create `client/llm_provider.py`:

```python
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
        from anthropic import Anthropic
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
        anthropic_tools = None
        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = self.client.messages.create(**kwargs)

        result = {
            "content": "",
            "tool_calls": [],
            "stop_reason": response.stop_reason,
        }

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
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        """Initialize OpenAI provider."""
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _convert_tools_to_openai_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tool schemas to OpenAI format."""
        openai_tools = []
        for tool in tools:
            input_schema = tool.get("inputSchema", {})
            if isinstance(input_schema, dict):
                input_schema = input_schema.copy()
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
        openai_tools = None
        if tools:
            openai_tools = self._convert_tools_to_openai_format(tools)

        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        result = {
            "content": message.content or "",
            "tool_calls": [],
            "stop_reason": response.choices[0].finish_reason,
        }

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
        model = model or "gpt-4o-mini"
        base_url = base_url or "https://api.openai.com/v1"
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
```

---

### Step 3.3: MCP Client Implementation

Create `client/mcp_client.py`:

```python
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
                        # Anthropic format
                        if not any(
                            msg.get("role") == "assistant"
                            and any(
                                isinstance(c, dict) and c.get("type") == "tool_use"
                                for c in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                            )
                            for msg in self.conversation_history[-1:]
                        ):
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": [{
                                    "type": "tool_use",
                                    "id": tool_call["id"],
                                    "name": tool_name,
                                    "input": tool_args,
                                }]
                            })

                        self.conversation_history.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_call["id"],
                                "content": json.dumps(result),
                            }]
                        })

                    elif self.settings.llm_provider == "openai":
                        # OpenAI format
                        if not any(
                            msg.get("role") == "assistant" and msg.get("tool_calls")
                            for msg in self.conversation_history[-1:]
                        ):
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
```

---

## Part 4: Testing & Demo

### Step 4.1: Create Interactive Demo

Create `examples/interactive_client.py`:

```python
"""
Interactive MCP Client Demo - Chat with MCP servers using LLMs
"""

import sys
import asyncio
from pathlib import Path

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
```

### Step 4.2: Create Automated Test

Create `examples/test_demo.py`:

```python
"""
Test Demo - Automated testing of MCP client with all servers
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.mcp_client import MCPClient
from client.config import get_client_settings


async def test_server(server_name: str, server_path: str, test_query: str):
    """Test a single server with a query."""
    print(f"\n{'='*70}")
    print(f"Testing {server_name}")
    print(f"{'='*70}\n")

    client = MCPClient()

    try:
        print(f"[CONNECT] Connecting to {server_name}...")
        await client.connect_to_server(server_path)
        print(f"[OK] Connected successfully")
        print(f"[TOOLS] Available: {', '.join(client.get_available_tools())}\n")

        print(f"[TEST] Query: {test_query}")
        response = await client.chat(test_query)

        print(f"\n[SUCCESS] {server_name} test completed!\n")
        print("-" * 70)

        return True

    except Exception as e:
        print(f"\n[ERROR] {server_name} test failed: {e}\n")
        return False

    finally:
        await client.disconnect()


async def main():
    """Main entry point."""
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

    for test in tests:
        result = await test_server(test["name"], test["path"], test["query"])
        results.append((test["name"], result))
        await asyncio.sleep(2)

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
```

### Step 4.3: Run Tests

```bash
# Run automated tests
uv run python examples/test_demo.py

# Run interactive mode - Weather
uv run python examples/interactive_client.py 1

# Run interactive mode - News
uv run python examples/interactive_client.py 2

# Run interactive mode - Finance
uv run python examples/interactive_client.py 3
```

---

## Part 5: Extending for Production

### Step 5.1: Adding Custom Tools

**Example: Adding a Database Query Tool**

Create `server/database_server.py`:

```python
"""
Database MCP Server - Custom tool for your business needs
"""

import os
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP(name="database")


@mcp.tool()
async def query_customer_data(customer_id: str) -> str:
    """Query customer information from database.

    Args:
        customer_id: Unique customer identifier

    Returns:
        Customer information including name, email, and purchase history
    """
    # TODO: Connect to your actual database
    # import asyncpg
    # conn = await asyncpg.connect('postgresql://...')
    # result = await conn.fetchrow('SELECT * FROM customers WHERE id = $1', customer_id)

    # For demo purposes
    return f"""
Customer ID: {customer_id}
Name: John Doe
Email: john@example.com
Total Orders: 15
Lifetime Value: $5,234.56
Status: Active
"""


@mcp.tool()
async def get_product_inventory(product_sku: str) -> str:
    """Get current inventory levels for a product.

    Args:
        product_sku: Product SKU code

    Returns:
        Inventory information including stock levels and warehouse locations
    """
    # TODO: Connect to your inventory system
    return f"""
Product SKU: {product_sku}
Name: Premium Widget
Stock: 1,234 units
Warehouses:
  - NY: 456 units
  - CA: 778 units
Status: In Stock
"""


if __name__ == "__main__":
    print("Starting Database MCP Server...")
    mcp.run()
```

**To use this custom tool:**
1. Add it to your server directory
2. Update your client to connect to this server
3. The LLM will automatically discover and use it!

---

### Step 5.2: Which Code Components to Modify

#### For Adding New Tools:

**1. Server Side (`server/your_server.py`):**
```python
# ✏️ MODIFY: Add new @mcp.tool() decorated functions
@mcp.tool()
async def your_new_tool(param: str) -> str:
    """Clear description for LLM."""
    # Your logic here
    return result
```

**2. Client Side:**
- ❌ NO CHANGES NEEDED - Tools are auto-discovered!

**3. Environment Variables (`.env`):**
```bash
# ✏️ ADD: Any new API keys or configuration
YOUR_NEW_API_KEY=your_key_here
```

#### For Changing LLM Providers:

**1. Configuration (`.env`):**
```bash
# ✏️ CHANGE: Switch between providers
LLM_PROVIDER=anthropic  # or "openai"

# ✏️ UPDATE: Corresponding API key
ANTHROPIC_API_KEY=your_key_here
```

**2. Client (`client/config.py`):**
```python
# ✏️ ADD: New provider configuration if needed
class ClientSettings(BaseSettings):
    # Add new provider settings here
    azure_api_key: str = ""
    azure_endpoint: str = ""
```

**3. LLM Provider (`client/llm_provider.py`):**
```python
# ✏️ ADD: New provider class
class AzureProvider(LLMProvider):
    def __init__(self, api_key: str, endpoint: str):
        # Implementation
        pass
```

---

### Step 5.3: Production Checklist

#### Before Going Live:

**Security:**
- [ ] Replace all test API keys with production keys
- [ ] Move secrets to environment variables or secret manager
- [ ] Implement rate limiting on tools
- [ ] Add authentication/authorization to servers
- [ ] Enable HTTPS for any HTTP endpoints

**Error Handling:**
- [ ] Add comprehensive error logging
- [ ] Implement retry logic for API calls
- [ ] Set up error alerting (email, Slack, etc.)
- [ ] Add fallback mechanisms for critical tools

**Monitoring:**
- [ ] Set up LangSmith (see Part 6)
- [ ] Add application performance monitoring (APM)
- [ ] Create dashboards for key metrics
- [ ] Set up uptime monitoring

**Testing:**
- [ ] Write unit tests for all tools
- [ ] Create integration tests
- [ ] Load test the system
- [ ] Test error scenarios

**Documentation:**
- [ ] Document all custom tools
- [ ] Create runbook for operations
- [ ] Write API documentation
- [ ] Create user guides

---

## Part 6: LangSmith Monitoring

### Step 6.1: Setup LangSmith Account

1. Go to https://smith.langchain.com/
2. Create account and get API key
3. Create a new project (e.g., "mcp-production")

### Step 6.2: Add LangSmith to Configuration

Update `.env`:
```bash
# LangSmith Configuration
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=mcp-production
LANGSMITH_TRACING=true
```

### Step 6.3: Integrate LangSmith Monitoring

Create `client/monitoring.py`:

```python
"""
LangSmith monitoring integration
"""

import os
from typing import Optional, Dict, Any
from functools import wraps
import asyncio


class LangSmithMonitor:
    """Monitor MCP operations with LangSmith."""

    def __init__(self):
        self.enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.project = os.getenv("LANGSMITH_PROJECT", "mcp-production")

        if self.enabled:
            try:
                from langsmith import Client
                self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
                print(f"[MONITOR] LangSmith enabled for project: {self.project}")
            except ImportError:
                print("[WARNING] langsmith package not installed. Run: uv add langsmith")
                self.enabled = False
            except Exception as e:
                print(f"[WARNING] Could not initialize LangSmith: {e}")
                self.enabled = False

    def trace_tool_call(self, tool_name: str):
        """Decorator to trace tool calls."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)

                # Create trace
                run_id = None
                try:
                    from langsmith import traceable

                    @traceable(
                        run_type="tool",
                        name=tool_name,
                        project_name=self.project,
                    )
                    async def traced_func():
                        return await func(*args, **kwargs)

                    result = await traced_func()
                    return result

                except Exception as e:
                    print(f"[MONITOR] Trace error: {e}")
                    return await func(*args, **kwargs)

            return wrapper
        return decorator

    def trace_llm_call(self, provider: str, model: str):
        """Decorator to trace LLM calls."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                try:
                    from langsmith import traceable

                    @traceable(
                        run_type="llm",
                        name=f"{provider}-{model}",
                        project_name=self.project,
                    )
                    def traced_func():
                        return func(*args, **kwargs)

                    return traced_func()

                except Exception as e:
                    print(f"[MONITOR] Trace error: {e}")
                    return func(*args, **kwargs)

            return wrapper
        return decorator

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log custom events."""
        if not self.enabled:
            return

        try:
            self.client.create_feedback(
                run_id=None,
                key=event_type,
                score=1.0,
                comment=str(data),
            )
        except Exception as e:
            print(f"[MONITOR] Log error: {e}")


# Global monitor instance
monitor = LangSmithMonitor()
```

### Step 6.4: Update Client to Use Monitoring

Update `client/mcp_client.py` (add these imports and changes):

```python
# Add at the top
from client.monitoring import monitor

# In the _invoke_tool method, wrap the call:
async def _invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a tool on the MCP server."""

    # Log to LangSmith
    monitor.log_event("tool_invocation", {
        "tool": tool_name,
        "arguments": arguments,
    })

    # ... rest of implementation
```

### Step 6.5: View Monitoring Data

1. Go to https://smith.langchain.com/
2. Select your project
3. View traces, latency, errors, and costs
4. Create dashboards and alerts

**What You'll See:**
- Every LLM call with tokens used
- Every tool invocation with parameters
- Latency for each step
- Error rates and types
- Cost tracking
- User feedback

---

## Part 7: Deployment

### Option 1: Local Deployment

**Simple script (`deploy/start.sh`):**

```bash
#!/bin/bash

# Load environment
source .venv/bin/activate

# Start servers in background
python server/weather_server.py &
python server/news_server.py &
python server/finance_server.py &

# Start client
python examples/interactive_client.py
```

### Option 2: Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install UV
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies
RUN uv sync

# Expose ports (if needed)
EXPOSE 8000

# Start application
CMD ["uv", "run", "python", "examples/interactive_client.py"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  weather-server:
    build: .
    command: uv run python server/weather_server.py
    env_file: .env

  news-server:
    build: .
    command: uv run python server/news_server.py
    env_file: .env

  finance-server:
    build: .
    command: uv run python server/finance_server.py
    env_file: .env

  client:
    build: .
    command: uv run python examples/interactive_client.py
    env_file: .env
    depends_on:
      - weather-server
      - news-server
      - finance-server
```

### Option 3: Cloud Deployment (AWS Lambda)

**handler.py:**

```python
import json
import asyncio
from client.mcp_client import MCPClient


def lambda_handler(event, context):
    """AWS Lambda handler."""
    query = event.get('query', '')
    server = event.get('server', 'server/weather_server.py')

    async def process():
        client = MCPClient()
        await client.connect_to_server(server)
        response = await client.chat(query)
        await client.disconnect()
        return response

    result = asyncio.run(process())

    return {
        'statusCode': 200,
        'body': json.dumps({'response': result})
    }
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Module not found"

**Solution:**
```bash
# Reinstall dependencies
rm -rf .venv
uv sync
```

#### Issue 2: "API key not configured"

**Solution:**
```bash
# Check .env file exists
cat .env

# Verify keys are set
uv run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

#### Issue 3: "Connection refused"

**Solution:**
```bash
# Check server is running
ps aux | grep python

# Check ports
netstat -an | grep 8000
```

#### Issue 4: "Tool calls failing"

**Solution:**
1. Check external API keys are valid
2. Test API directly with curl
3. Check rate limits
4. Enable debug mode: `DEBUG=true`

---

## Next Steps

### Congratulations! 🎉

You now have a complete, production-ready MCP implementation!

**What you've built:**
- ✅ 3 working MCP servers (Weather, News, Finance)
- ✅ Intelligent MCP client with dual LLM support
- ✅ Interactive demo and automated tests
- ✅ LangSmith monitoring integration
- ✅ Extensible architecture for custom tools

**Where to go from here:**

1. **Add More Tools** - Extend with domain-specific tools
2. **Improve Error Handling** - Add retries, fallbacks
3. **Scale Up** - Deploy to cloud, add load balancing
4. **Add UI** - Build web interface with Streamlit/Gradio
5. **Fine-tune** - Optimize prompts and parameters

---

## Resources

### Documentation
- MCP Protocol: https://modelcontextprotocol.io
- FastMCP: https://gofastmcp.com
- Python MCP SDK: https://github.com/modelcontextprotocol/python-sdk

### APIs
- OpenWeatherMap: https://openweathermap.org/api
- NewsAPI: https://newsapi.org
- ExchangeRate-API: https://www.exchangerate-api.com

### Monitoring
- LangSmith: https://smith.langchain.com
- LangChain Docs: https://python.langchain.com/docs/langsmith

### Community
- GitHub Issues: [Your Repo URL]
- Discord: [Your Discord]
- Twitter: [Your Twitter]

---

**Happy Building! 🚀**

For questions or contributions, please open an issue on GitHub.
