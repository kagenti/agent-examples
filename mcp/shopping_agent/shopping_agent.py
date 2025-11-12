"""Shopping Agent MCP Tool - Uses LangChain, LangGraph, OpenAI, and SerpAPI"""

import argparse
import os
import sys
import json
import logging
from typing import TypedDict, Annotated, List, Dict, Any
from fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout, format='%(levelname)s: %(message)s')


def _env_flag(name: str, default: str = "false") -> bool:
    """Parse environment flag strings like 1/true/on into booleans."""
    value = os.getenv(name)
    if value is None:
        value = default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize FastMCP
mcp = FastMCP("Shopping Agent")


class AgentState(TypedDict):
    """State for the shopping agent graph"""
    messages: Annotated[List, add_messages]
    search_results: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


def create_shopping_agent():
    """Create a LangGraph agent for shopping recommendations"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Initialize SerpAPI search
    search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    
    def parse_query_node(state: AgentState) -> AgentState:
        """Parse the user query to extract product type and budget"""
        logger.debug("Parsing user query...")
        messages = state["messages"]
        user_query = messages[-1].content if messages else ""
        
        # Use LLM to extract structured information
        system_prompt = """You are a shopping assistant. Extract the product type and budget from the user's query.
Return a JSON object with 'product' and 'budget' fields. If budget is not specified, use 'unknown'.
Example: {"product": "scarf", "budget": "40"}"""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ])
        
        try:
            parsed_data = json.loads(response.content)
            product = parsed_data.get("product", "product")
            budget = parsed_data.get("budget", "unknown")
        except:
            # Fallback parsing
            product = "product"
            budget = "unknown"
            for word in user_query.split():
                if word.startswith("$") or word.isdigit():
                    budget = word.replace("$", "")
                    break
        
        state["messages"].append(AIMessage(content=f"Searching for {product} within budget ${budget}..."))
        return state
    
    def search_products_node(state: AgentState) -> AgentState:
        """Search for products using SerpAPI"""
        logger.debug("Searching for products...")
        messages = state["messages"]
        
        # Extract search query from conversation
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        if not user_query:
            state["search_results"] = []
            return state
        
        # Construct search query optimized for shopping
        search_query = f"{user_query} buy online shop"
        
        try:
            # Perform search
            search_results_raw = search.run(search_query)
            
            # SerpAPI returns a string, we need to parse it
            state["search_results"] = [{"raw_results": search_results_raw}]
            logger.debug(f"Search completed with results")
        except Exception as e:
            logger.error(f"Search error: {e}")
            state["search_results"] = []
        
        return state
    
    def generate_recommendations_node(state: AgentState) -> AgentState:
        """Generate product recommendations from search results"""
        logger.debug("Generating recommendations...")
        
        search_results = state.get("search_results", [])
        
        if not search_results:
            state["recommendations"] = []
            state["messages"].append(AIMessage(content="Sorry, I couldn't find any products matching your criteria."))
            return state
        
        # Use LLM to parse search results and generate recommendations
        system_prompt = """You are a shopping assistant. Based on the search results provided, 
create a list of exactly 5 product recommendations. For each product, provide:
- name: Product name
- price: Estimated price (extract from results if available)
- description: Brief description
- url: Purchase link if available
- reason: Why this product is recommended

Return the recommendations as a JSON array. If you can't find 5 products, provide as many as you can find."""
        
        search_content = json.dumps(search_results)
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Search results:\n{search_content}\n\nGenerate 5 product recommendations.")
        ])
        
        try:
            # Try to parse JSON response
            content = response.content
            # Find JSON array in response
            if "[" in content and "]" in content:
                start_idx = content.index("[")
                end_idx = content.rindex("]") + 1
                json_str = content[start_idx:end_idx]
                recommendations = json.loads(json_str)
            else:
                # Fallback: create structured response from text
                recommendations = [{
                    "name": "Product recommendations",
                    "description": content,
                    "note": "Please refine your search for more specific results"
                }]
        except Exception as e:
            logger.error(f"Error parsing recommendations: {e}")
            recommendations = [{
                "error": "Could not parse recommendations",
                "raw_response": response.content[:500]
            }]
        
        state["recommendations"] = recommendations[:5]  # Limit to 5
        
        # Format recommendations as a message
        formatted_recs = "\n\n".join([
            f"**{i+1}. {rec.get('name', 'Product')}**\n"
            f"Price: {rec.get('price', 'Price not available')}\n"
            f"Description: {rec.get('description', 'N/A')}\n"
            f"Reason: {rec.get('reason', 'Good match for your needs')}\n"
            f"URL: {rec.get('url', 'Search online for this product')}"
            for i, rec in enumerate(state["recommendations"])
        ])
        
        state["messages"].append(AIMessage(content=f"Here are my top recommendations:\n\n{formatted_recs}"))
        
        return state
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("search_products", search_products_node)
    workflow.add_node("generate_recommendations", generate_recommendations_node)
    
    # Define edges
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "search_products")
    workflow.add_edge("search_products", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def recommend_products(query: str, maxResults: int = 5) -> str:
    """
    Recommend products based on natural language query (e.g., "good curtains under $40")
    
    Args:
        query: Natural language product request with price range and preferences
        maxResults: Maximum number of product recommendations to return (default 5, max 20)
    
    Returns:
        JSON string containing product recommendations with names, prices, descriptions, and links
    """
    logger.info(f"Recommending products for query: '{query}'")
    
    if not OPENAI_API_KEY:
        return json.dumps({"error": "OPENAI_API_KEY not configured"})
    
    if not SERPAPI_API_KEY:
        return json.dumps({"error": "SERPAPI_API_KEY not configured"})
    
    # Limit maxResults
    maxResults = min(maxResults, 20)
    
    try:
        # Create the agent
        agent = create_shopping_agent()
        
        # Run the agent
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "search_results": [],
            "recommendations": []
        }
        
        result = agent.invoke(initial_state)
        
        # Extract recommendations
        recommendations = result.get("recommendations", [])[:maxResults]
        
        return json.dumps({
            "query": query,
            "recommendations": recommendations,
            "count": len(recommendations)
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in recommend_products: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def search_products(query: str, maxResults: int = 10) -> str:
    """
    Search for products across retailers (internal tool)
    
    Args:
        query: Product search query
        maxResults: Maximum number of results to return (default 10, max 100)
    
    Returns:
        JSON string containing search results
    """
    logger.info(f"Searching products for query: '{query}'")
    
    if not SERPAPI_API_KEY:
        return json.dumps({"error": "SERPAPI_API_KEY not configured"})
    
    # Limit maxResults
    maxResults = min(maxResults, 100)
    
    try:
        search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
        results = search.run(f"{query} buy online shopping")
        
        return json.dumps({
            "query": query,
            "results": results,
            "note": "Raw search results - use recommend_products for curated recommendations"
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_products: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def run_server(
    transport: str | None = None,
    host: str | None = None,
    port: int | str | None = None,
    json_response: bool | None = None,
    stateless_http: bool | None = None,
) -> None:
    """Run the MCP server with optional overrides from CLI or environment."""
    if transport is None:
        transport = os.getenv("MCP_TRANSPORT", "http")
    if host is None:
        host = os.getenv("HOST", "0.0.0.0")
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    else:
        port = int(port)
    if json_response is None:
        json_response = _env_flag("MCP_JSON_RESPONSE", "true")
    if stateless_http is None:
        stateless_http = _env_flag("MCP_STATELESS_HTTP", "false")

    logger.info(
        "Starting MCP server transport=%s host=%s port=%s json_response=%s stateless_http=%s",
        transport,
        host,
        port,
        json_response,
        stateless_http,
    )
    mcp.run(
        transport=transport,
        host=host,
        port=port,
        json_response=json_response,
        stateless_http=stateless_http,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shopping Agent MCP Server")
    parser.add_argument(
        "--transport",
        dest="transport",
        default=None,
        help="Transport to use for FastMCP (default: env MCP_TRANSPORT or http)",
    )
    parser.add_argument(
        "--host",
        dest="host",
        default=None,
        help="Host interface to bind (default: env HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=None,
        help="Port to bind (default: env PORT or 8000)",
    )
    parser.add_argument(
        "--json-response",
        dest="json_response",
        action="store_true",
        help="Force JSON responses (overrides env MCP_JSON_RESPONSE)",
    )
    parser.add_argument(
        "--no-json-response",
        dest="json_response",
        action="store_false",
        help="Disable JSON responses (overrides env MCP_JSON_RESPONSE)",
    )
    parser.add_argument(
        "--stateless-http",
        dest="stateless_http",
        action="store_true",
        help="Enable stateless HTTP transport mode",
    )
    parser.add_argument(
        "--stateful-http",
        dest="stateless_http",
        action="store_false",
        help="Disable stateless HTTP transport mode",
    )
    parser.set_defaults(json_response=None, stateless_http=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if OPENAI_API_KEY is None:
        logger.warning("Please configure the OPENAI_API_KEY environment variable before running the server")
    if SERPAPI_API_KEY is None:
        logger.warning("Please configure the SERPAPI_API_KEY environment variable before running the server")
    
    if OPENAI_API_KEY and SERPAPI_API_KEY:
        logger.info("Starting Shopping Agent MCP Server with LangChain and LangGraph")
        run_server(
            transport=args.transport,
            host=args.host,
            port=args.port,
            json_response=args.json_response,
            stateless_http=args.stateless_http,
        )
        return 0
    else:
        logger.error("Cannot start server without required API keys")
        return 1


if __name__ == "__main__":
    sys.exit(main())

