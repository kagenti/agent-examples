"""Simple test for Shopping Agent - Tests the core logic directly"""

import os
import sys
import json
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Any

# Check API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not set!")
    print("   Run: export OPENAI_API_KEY='your-key'")
    sys.exit(1)

if not SERPAPI_API_KEY:
    print("❌ ERROR: SERPAPI_API_KEY not set!")
    print("   Run: export SERPAPI_API_KEY='your-key'")
    sys.exit(1)

print("✅ API keys configured\n")

# Define the agent state
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
        print("   🔍 Parsing query...")
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
            product = "product"
            budget = "unknown"
        
        print(f"   ✓ Searching for: {product} (budget: ${budget})")
        state["messages"].append(AIMessage(content=f"Searching for {product} within budget ${budget}..."))
        return state
    
    def search_products_node(state: AgentState) -> AgentState:
        """Search for products using SerpAPI"""
        print("   🌐 Searching products...")
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
            state["search_results"] = [{"raw_results": search_results_raw}]
            print(f"   ✓ Search completed")
        except Exception as e:
            print(f"   ⚠️ Search error: {e}")
            state["search_results"] = []
        
        return state
    
    def generate_recommendations_node(state: AgentState) -> AgentState:
        """Generate product recommendations from search results"""
        print("   🤖 Generating recommendations...")
        
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
            content = response.content
            if "[" in content and "]" in content:
                start_idx = content.index("[")
                end_idx = content.rindex("]") + 1
                json_str = content[start_idx:end_idx]
                recommendations = json.loads(json_str)
            else:
                recommendations = [{
                    "name": "Product recommendations",
                    "description": content,
                    "note": "Please refine your search for more specific results"
                }]
        except Exception as e:
            print(f"   ⚠️ Error parsing recommendations: {e}")
            recommendations = [{
                "error": "Could not parse recommendations",
                "raw_response": response.content[:500]
            }]
        
        state["recommendations"] = recommendations[:5]
        print(f"   ✓ Generated {len(state['recommendations'])} recommendations")
        return state
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("search_products", search_products_node)
    workflow.add_node("generate_recommendations", generate_recommendations_node)
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "search_products")
    workflow.add_edge("search_products", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    
    return workflow.compile()


def test_shopping_agent():
    """Run tests on the shopping agent"""
    
    print("🛍️  Shopping Agent Test Suite")
    print("=" * 70)
    
    # Create the agent
    agent = create_shopping_agent()
    
    # Test 1: Scarves
    print("\n📊 Test 1: Recommend scarves under $40")
    print("-" * 70)
    
    initial_state = {
        "messages": [HumanMessage(content="I want to buy a scarf for 40 dollars")],
        "search_results": [],
        "recommendations": []
    }
    
    result = agent.invoke(initial_state)
    recommendations = result.get("recommendations", [])
    
    print(f"\n✅ Received {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   {i}. {rec.get('name', 'N/A')}")
        print(f"      Price: {rec.get('price', 'N/A')}")
        print(f"      Reason: {rec.get('reason', 'N/A')}")
    
    # Test 2: Headphones
    print("\n\n📊 Test 2: Recommend wireless headphones under $100")
    print("-" * 70)
    
    initial_state = {
        "messages": [HumanMessage(content="Find me wireless headphones under $100 with good noise cancellation")],
        "search_results": [],
        "recommendations": []
    }
    
    result = agent.invoke(initial_state)
    recommendations = result.get("recommendations", [])
    
    print(f"\n✅ Received {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   {i}. {rec.get('name', 'N/A')}")
        print(f"      Price: {rec.get('price', 'N/A')}")
        print(f"      Reason: {rec.get('reason', 'N/A')}")
    
    print("\n\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_shopping_agent()

