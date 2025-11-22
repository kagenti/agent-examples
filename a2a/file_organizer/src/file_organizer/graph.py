import os
from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage
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
            "url": os.getenv("MCP_URL", "http://cloud-storage-tool:8000/mcp"),
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

Tool output schema:
- get_objects -> {{"provider": str, "bucket": str, "objects": [{{"name": str, "path": str, "file_uri": str, "size_bytes": int|None}}]}}
- perform_action -> {{"status": "success", "action": "move|copy", "source_uri": str, "target_uri": str, "target_folder_uri": str, "filename": str, "message": str}}

Guidelines:
1. Always call get_objects (or ask the user for the bucket) before assuming any files exist.
2. Only reference files that appear in tool results or explicit user input.
3. Derive organization rules from extensions, folders, or user instructions; ask clarifying questions if uncertain.
4. When moving/copying, explicitly describe which files are affected and why.
5. Provide a concise final summary listing the actions performed and remaining follow-ups.
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
