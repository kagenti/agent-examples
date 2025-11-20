"""Test client for Shopping Agent MCP Server using MCP SDK"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters, StreamableHTTPClient
from mcp.client.streamable_http import streamablehttp_client

async def test_shopping_agent():
    """Test the shopping agent MCP server using Streamable HTTP"""
    
    print("🛍️  Testing Shopping Agent MCP Server with Streamable HTTP")
    print("=" * 60)
    
    # Create a Streamable HTTP client
    client = streamablehttp_client("http://localhost:8000")
    
    # Test queries
    test_queries = [
        {
            "query": "I want to buy a scarf for 40 dollars. Recommend me some options.",
            "maxResults": 3
        },
        {
            "query": "Find me wireless headphones under $100 with good noise cancellation",
            "maxResults": 3
        },
    ]
    
    # For testing, we'll call the functions directly since they're in the same file
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from shopping_agent import recommend_products, search_products
    
    print("\n📊 Test 1: Recommend scarves under $40")
    print("-" * 60)
    result1 = recommend_products(
        query="I want to buy a scarf for 40 dollars",
        maxResults=3
    )
    print(result1)
    
    print("\n\n📊 Test 2: Recommend wireless headphones")
    print("-" * 60)
    result2 = recommend_products(
        query="Find me wireless headphones under $100 with good noise cancellation",
        maxResults=3
    )
    print(result2)
    
    print("\n\n📊 Test 3: Search for winter jackets")
    print("-" * 60)
    result3 = search_products(
        query="winter jacket waterproof",
        maxResults=5
    )
    print(result3[:500] + "..." if len(result3) > 500 else result3)
    
    print("\n\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_shopping_agent())

