import os
from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage,  AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI

from file_organizer.configuration import Configuration

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
     final_answer: str = ""

def get_mcpclient():
    
    return MultiServerMCPClient({
        "cloud_storage": {
            "url": os.getenv("MCP_URL", "http://localhost:8000/mcp"),
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
    bucket_uri = os.getenv("BUCKET_URI")

    bucket_info = f"Target bucket: {bucket_uri}" if bucket_uri else "No bucket URI configured. Ask the user to specify which bucket to organize."

    sys_msg = SystemMessage(content=f"""You are a file organization assistant for cloud storage buckets.

{bucket_info}

Your workflow:
1. Discover what tools are available to you by examining your tool list
2. Use the appropriate tool to LIST or discover files in the bucket
3. Analyze each file and decide how to organize it based on:
   - File extension and type (e.g., .pdf, .jpg, .txt)
   - Filename patterns or naming conventions
   - Logical grouping (similar file types together)
4. Use the appropriate tool to MOVE or COPY each file to its organized location
5. Provide a summary of what you actually did (not what you would do)

IMPORTANT:
- You have access to cloud storage tools - USE THEM
- Every file operation must be done through a tool call
- Do not hallucinate or pretend to organize files - actually call the tools
- If you don't know what tools are available, examine them first
- Always verify the bucket URI before performing operations
- Provide concrete evidence of what actions were taken (via tool results)
""")

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
