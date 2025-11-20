"""Test client for the Shopping Agent MCP server"""

import requests
import json
import sys

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    # Accept both JSON (for immediate responses) and SSE (for streamed events).
    "Accept": "application/json, text/event-stream",
}


def test_recommend_products(base_url: str, query: str, max_results: int = 5):
    """Test the recommend_products tool"""
    print(f"\n{'='*60}")
    print(f"Testing recommend_products with query: '{query}'")
    print(f"{'='*60}\n")
    
    url = f"{base_url}/tools/recommend_products"
    payload = {
        "query": query,
        "maxResults": max_results
    }
    
    try:
        response = requests.post(url, json=payload, headers=DEFAULT_HEADERS, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


def test_search_products(base_url: str, query: str, max_results: int = 10):
    """Test the search_products tool"""
    print(f"\n{'='*60}")
    print(f"Testing search_products with query: '{query}'")
    print(f"{'='*60}\n")
    
    url = f"{base_url}/tools/search_products"
    payload = {
        "query": query,
        "maxResults": max_results
    }
    
    try:
        response = requests.post(url, json=payload, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


def main():
    """Main test function"""
    base_url = "http://localhost:8000/mcp"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip("/")
    
    print(f"Testing Shopping Agent MCP Server at: {base_url}")
    
    # Test 1: Recommend scarves under $40
    test_recommend_products(
        base_url,
        "I want to buy a scarf for 40 dollars. Recommend me some options.",
        max_results=5
    )
    
    # Test 2: Recommend headphones
    test_recommend_products(
        base_url,
        "Find me wireless headphones under $100 with good noise cancellation",
        max_results=5
    )
    
    # Test 3: Search products (lower-level API)
    test_search_products(
        base_url,
        "winter jacket waterproof",
        max_results=10
    )
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

