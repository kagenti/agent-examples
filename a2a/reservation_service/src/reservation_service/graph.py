from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
import os
from reservation_service.configuration import Configuration

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
    final_answer: str = ""

def get_mcpclient():
    return MultiServerMCPClient({
        "reservations": {
            "url": os.getenv("MCP_URL", "http://reservation-tool:8000/mcp"),
            "transport": os.getenv("MCP_TRANSPORT", "streamable_http"),
        }
    })

async def get_graph(client) -> StateGraph:
    llm = ChatOpenAI(
        model=config.llm_model,
        openai_api_key=config.llm_api_key,
        openai_api_base=config.llm_api_base,
        temperature=0,
    )

    # Get tools asynchronously
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # System message
    sys_msg = SystemMessage(content="""You are a helpful restaurant reservation assistant. You have access to tools for:
- Searching restaurants by city, cuisine, price tier
- Checking availability at restaurants
- Making reservations
- Canceling reservations
- Listing user reservations

When helping users:
1. Always search for restaurants first if they haven't specified one
2. Check availability before attempting to make a reservation
3. For reservations, collect: date/time, party size, guest name, phone, and email
4. Provide confirmation codes when reservations are successful
5. Be conversational and helpful

Use the provided tools to complete your tasks.""")

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        state["messages"].append(result)
        # Set the final answer only if the result is an AIMessage without tool calls
        if isinstance(result, AIMessage) and not result.tool_calls:
            state["final_answer"] = result.content
        return state

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
