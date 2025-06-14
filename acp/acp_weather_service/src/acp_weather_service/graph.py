from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage,  AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
import os
from acp_weather_service.configuration import Configuration

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
     final_answer: str = ""

def get_mcpclient():
    return MultiServerMCPClient({
        "math": {
            "url": os.getenv("MCP_URL", "http://localhost:8000/sse"),
            "transport": "sse",
        }
    })

async def get_graph(client) -> StateGraph:
    llm = ChatOpenAI(
        model=config.llm_model,
        openai_api_key=config.llm_api_key,
        openai_api_base=config.llm_api_base,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(client.get_tools())

    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with providing weather information. You must use the provided tools to complete your task.")

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        state["messages"].append(result)
        # Set the final answer only if the result is an AIMessage (i.e., not a tool call)
        # and it's meant to be the final response to the user.
        # This logic might need refinement based on when you truly consider the answer "final".
        if isinstance(result, AIMessage) and not result.tool_calls:
            state["final_answer"] = result.content
        return state

    # Build graph
    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(client.get_tools()))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()
    return graph

# async def main():
#     from langchain_core.messages import HumanMessage
#     async with get_mcpclient() as client:
#         graph = await get_graph(client)
#         messages = [HumanMessage(content="how is the weather in NY today?")]
#         async for event in graph.astream_events({"messages": messages}, stream_mode="updates"):
#             print(event)
#             output = event
#         output = output.get("data",{}).get("output",{}).get("final_answer")
#         print(f">>> {output}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())