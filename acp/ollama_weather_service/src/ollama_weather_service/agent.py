import os
from textwrap import dedent
from typing import AsyncIterator

from acp_sdk import Metadata, Message, Link, LinkType, MessagePart
from acp_sdk.server import Server
from openinference.instrumentation.langchain import LangChainInstrumentor
from pydantic import AnyUrl
from langchain_core.messages import HumanMessage

from ollama_weather_service.graph import get_graph, get_mcpclient


LangChainInstrumentor().instrument()

server = Server()


@server.agent(
    metadata=Metadata(
        programming_language="Python",
        license="Apache 2.0",
        framework="LangGraph",
        links=[
            Link(
                type=LinkType.SOURCE_CODE,
                url=AnyUrl(
                    f"https://github.com/i-am-bee/beeai-platform/blob/{os.getenv('RELEASE_VERSION', 'main')}"
                    "/agents/community/ollama-weather-service"
                ),
            )
        ],
        documentation=dedent(
            """\
            This agent provides a simple weather infor assistant.

            ## Input Parameters
            - **prompt** (string) – the city for which you want to know weather info.

            ## Key Features
            - **MCP Tool Calling** – uses a MCP tool to get weather info.
            """,
        ),
        use_cases=[
            "**Weather Assistant** – Personalized assistant for weather info.",
        ],
        env=[
            {"name": "LLM_MODEL", "description": "Model to use from the specified OpenAI-compatible API."},
            {"name": "LLM_API_BASE", "description": "Base URL for OpenAI-compatible API endpoint"},
            {"name": "LLM_API_KEY", "description": "API key for OpenAI-compatible API endpoint"},
            {"name": "MCP_URL", "description": "MCP Server URL for the weather tool"},
        ],
        ui={"type": "hands-off", "user_greeting": "Ask me about the weather"},
        examples={
            "cli": [
                {
                    "command": 'beeai run ollama_weather_service "what is the weather in NY?"',
                    "description": "Running a Weather Query",
                    "processing_steps": [
                        "Calls the weather MCP tool to get the weather info"
                        "Parses results and return it",
                    ],
                }
            ]
        },
    )
)
async def acp_weather_service(input: list[Message]) -> AsyncIterator:
    """
    The agent allows to retrieve weather info through a natural language conversatinal interface
    """
    messages = [HumanMessage(content=input[-1].parts[-1].content)]
    input = {"messages": messages}
    print(f"{input}")
    try:
        output = None
        async with get_mcpclient() as mcpclient:
            graph = await get_graph(mcpclient)
            async for event in graph.astream(input, stream_mode="updates"):
                yield {
                    "message": "\n".join(
                        f"🚶‍♂️{key}: {str(value)[:100] + '...' if len(str(value)) > 100 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                }
                output = event
                print(event)
            output = output.get("assistant", {}).get("messages")
            yield MessagePart(content=str(output))
    except Exception as e:
        raise Exception(f"An error occurred while running the graph: {e}")


def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
