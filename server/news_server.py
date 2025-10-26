"""
News MCP Server - Provides news headlines using NewsAPI
"""

import os
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(name="news")

# Constants
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
        except Exception as e:
            print(f"Unexpected error: {e}")
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
        return f"Unable to fetch news headlines. Please check your API key and parameters."

    articles = data.get('articles', [])
    if not articles:
        return f"No headlines found for {category} news in {country.upper()}"

    # Format headlines
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
        query: Search keyword or phrase (e.g., "artificial intelligence", "climate change")
        language: Language code (e.g., "en", "es", "fr")
        page_size: Number of results to return (1-100)

    Returns:
        News articles matching the search query
    """
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not configured"

    # Limit page size
    page_size = max(1, min(100, page_size))

    url = f"{NEWS_API_BASE_URL}/everything?apiKey={NEWS_API_KEY}&q={query}&language={language}&pageSize={page_size}&sortBy=publishedAt"
    data = await make_news_request(url)

    if not data or data.get('status') != 'ok':
        return f"Unable to search news. Please check your query and try again."

    articles = data.get('articles', [])
    if not articles:
        return f"No articles found for query: '{query}'"

    # Format search results
    results = [f"Search results for '{query}':\n"]

    for i, article in enumerate(articles[:page_size], 1):
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown')
        published = article.get('publishedAt', 'Unknown date')[:10]  # Get date only
        description = article.get('description', 'No description')[:150]

        results.append(f"{i}. {title}\n   Source: {source} | Date: {published}\n   {description}...")

    return "\n\n".join(results)


# Run the server
if __name__ == "__main__":
    print("Starting News MCP Server...")
    print(f"API Key configured: {'Yes' if NEWS_API_KEY else 'No'}")
    mcp.run()
