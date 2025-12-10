from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
import os
import json
from image_service.configuration import Configuration
from typing import Optional

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
    final_answer: Optional[dict] = None

def get_mcpclient():
    return MultiServerMCPClient({
        "image": {
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

    # System message
    sys_msg = SystemMessage(content=
    """
    You are a helpful assistant. Only call the get_image tool when the user EXPLICITLY asks for an image with specific dimensions (e.g., 'show me an image', 'generate an image 400x400', 'image 200 300'). 
    For any conversation that does NOT explicitly request an image, respond directly with text. DO NOT call any tools for these cases.
    When you do call get_image, you MUST provide valid positive integers for both height and width parameters.
    """)

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        state["messages"].append(result)

        # Find the most recent ToolMessage and set its content as final_answer.
        # NOTE: Only the most recent ToolMessage is processed intentionally.
        # If multiple tools are called in sequence, earlier tool results are 
        # intermediate steps, while the final ToolMessage represents the complete
        # answer to return to the user. The graph ends once final_answer is set.
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                continue
            try:
                content = msg.content
                if isinstance(content, str):
                    try:
                        state["final_answer"] = json.loads(content)
                    except json.JSONDecodeError:
                        # JSON parsing failed, use raw content
                        state["final_answer"] = {"raw": content}
                else:
                    state["final_answer"] = content
            except Exception as e:
                logger.error("Assistant node: error processing ToolMessage: %s", e)
                state["final_answer"] = {
                    "error": "Failed to process tool result",
                    "details": str(e)
                }
            break

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
    
    # After tools run, check if we have final_answer, if so END, otherwise go back to assistant
    def should_continue(state: ExtendedMessagesState):
        # End the graph once a final_answer (tool result) is captured
        return END if state.get("final_answer") is not None else "assistant"
    
    builder.add_conditional_edges(
        "tools",
        should_continue,
    )

    # Compile and return graph
    graph = builder.compile()
    return graph