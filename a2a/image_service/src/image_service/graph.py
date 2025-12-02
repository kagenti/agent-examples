from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
import os
import json
from image_service.configuration import Configuration

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
     final_answer: dict = None  # Changed to dict to hold tool output directly

def get_mcpclient():
    return MultiServerMCPClient({
        "math": {
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
    sys_msg = SystemMessage(content="You are a helpful assistant that provides random images. When a user requests an image, call the get_image tool with the specified height and width dimensions.")

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        # Invoke LLM and append its response to the conversation
        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        state["messages"].append(result)

        # Find the most recent ToolMessage and set its content as final_answer
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                continue
            try:
                content = msg.content
                if isinstance(content, str):
                    try:
                        state["final_answer"] = json.loads(content)
                    except Exception:
                        state["final_answer"] = {"raw": content}
                else:
                    state["final_answer"] = content
            except Exception as e:
                logger.error("Assistant node: error processing ToolMessage: %s", e)
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

    # Compile graph
    graph = builder.compile()
    return graph

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
