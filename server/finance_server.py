"""
Finance MCP Server - Provides currency exchange data using ExchangeRate-API
"""

import os
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(name="finance")

# Constants
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
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


@mcp.tool()
async def get_exchange_rates(base_currency: str = "USD") -> str:
    """Get current exchange rates for a base currency.

    Args:
        base_currency: Three-letter currency code (e.g., "USD", "EUR", "GBP", "JPY")

    Returns:
        Current exchange rates for the base currency against other major currencies
    """
    if not EXCHANGERATE_API_KEY:
        return "Error: EXCHANGERATE_API_KEY not configured"

    url = f"{EXCHANGERATE_BASE_URL}/{EXCHANGERATE_API_KEY}/latest/{base_currency.upper()}"
    data = await make_exchange_request(url)

    if not data or data.get('result') != 'success':
        return f"Unable to fetch exchange rates for {base_currency}. Please check the currency code."

    rates = data.get('conversion_rates', {})
    last_update = data.get('time_last_update_utc', 'Unknown')

    # Format major currencies
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
        from_currency: Source currency code (e.g., "USD", "EUR")
        to_currency: Target currency code (e.g., "GBP", "JPY")
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
        return f"Unable to convert {from_currency} to {to_currency}. Please check the currency codes."

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


@mcp.tool()
async def get_supported_currencies() -> str:
    """Get list of supported currency codes.

    Returns:
        List of all supported currency codes and names
    """
    if not EXCHANGERATE_API_KEY:
        return "Error: EXCHANGERATE_API_KEY not configured"

    url = f"{EXCHANGERATE_BASE_URL}/{EXCHANGERATE_API_KEY}/codes"
    data = await make_exchange_request(url)

    if not data or data.get('result') != 'success':
        return "Unable to fetch supported currencies"

    supported_codes = data.get('supported_codes', [])

    # Format currency list (show first 50 as example)
    currency_lines = ["Supported Currencies (showing first 50):\n"]

    for code, name in supported_codes[:50]:
        currency_lines.append(f"{code}: {name}")

    currency_lines.append(f"\nTotal: {len(supported_codes)} currencies supported")

    return "\n".join(currency_lines)


# Run the server
if __name__ == "__main__":
    print("Starting Finance MCP Server...")
    print(f"API Key configured: {'Yes' if EXCHANGERATE_API_KEY else 'No'}")
    mcp.run()
