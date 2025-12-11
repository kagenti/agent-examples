from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from functools import lru_cache
from typing import List

from generic_agent.config import Configuration

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
    final_answer: str = ""

def _get_mcp_urls() -> List[str]:
    """Helper function to parse MCP URLs from environment variable."""
    urls_str = config.MCP_URLS
    return [url.strip() for url in urls_str.split(',') if url.strip()]

@lru_cache(maxsize=1)
def get_mcpclient() -> MultiServerMCPClient:
    urls = _get_mcp_urls()
    
    client_configs = {}
    transport = config.MCP_TRANSPORT
    
    for i, url in enumerate(urls, 1):
        client_configs[f"mcp{i}"] = {
            "url": url,
            "transport": transport,
        }
    return MultiServerMCPClient(client_configs)

def get_mcp_server_names() -> List[str]:
    """
    Extract MCP server names from URLs.
    
    Strips protocol (http/https), port, and path to get just the host names.
    Example: "http://weather-tool:8000/mcp" -> "weather-tool"

    Returns:
        List of MCP server host names
    """
    urls = _get_mcp_urls()

    mcp_names = []
    for url in urls:
        # Remove protocol
        name = url.replace('http://', '').replace('https://', '')
        # Remove port and path (everything after first :)
        name = name.split(':')[0]
        # Remove /mcp or any path
        name = name.split('/')[0]
        if name:
            mcp_names.append(name)
    
    return mcp_names
    

async def get_graph(client: MultiServerMCPClient) -> StateGraph:
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0,
    )

    # Get tools asynchronously
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # System message
    sys_msg = SystemMessage(
    content="You are the **Generic Assistant**, a multi-purpose, tool-based expert. Your primary directive is to fulfill user requests by effectively utilizing the available **MCP tools**. You will select the most appropriate tool(s) based on the user's need (e.g., weather, calculations, data retrieval) and strictly adhere to their output to generate your final answer. Be precise and concise."
)

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        
        updated_state = {
            "messages": state["messages"] + [result]
        }
        
        # Set final_answer when LLM returns a text response (not a tool call)
        # This indicates the assistant has completed its reasoning and tool usage
        if isinstance(result, AIMessage) and not result.tool_calls:
            updated_state["final_answer"] = result.content
        
        return updated_state

    # Build graph
    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()
    return graph