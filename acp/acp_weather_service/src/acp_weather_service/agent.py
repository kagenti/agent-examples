import logging
import os
from textwrap import dedent
from typing import AsyncIterator

from acp_sdk import Metadata, Message, Link, LinkType, MessagePart
from acp_sdk.models.errors import ACPError, Error, ErrorCode
from acp_sdk.server import Server
from openinference.instrumentation.langchain import LangChainInstrumentor
from pydantic import AnyUrl
from langchain_core.messages import HumanMessage

from acp_weather_service.graph import get_graph, get_mcpclient
from keycloak import KeycloakOpenID

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()

server = Server()

def get_token() -> str:
    keycloak_url = os.getenv("KEYCLOAK_URL", "http://keycloak.localtest.me:8080")
    client_id = os.getenv("CLIENT_NAME", "NOTSET")
    realm_name = "master"
    client_secret = os.getenv("CLIENT_SECRET")

    user_username = "test-user"
    user_password = "test-password"

    # print(f"client_id: {client_id}")
    logger.info(
          f"Using client_id='{client_id}' with realm={realm_name}"
    )

    try:
        keycloak_openid = KeycloakOpenID(server_url=keycloak_url,
                                        client_id=client_id,
                                        realm_name=realm_name,
                                        client_secret_key=client_secret)
    
        access_token = keycloak_openid.token(
                username=user_username,
                password=user_password)
    except Exception as e:
        raise Exception(f"Authorization error getting the access token: {e}")    

    logger.info(
          f"Received access token: {access_token}"
    )
    return access_token


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
            This agent provides a simple weather information assistance.

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
            {"name": "ACP_MCP_TRANSPORT", "description": "MCP transport type: sse, stdio, streamable_http, websocket (defaults to 'streamable_http')"},
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
    The agent allows to retrieve weather info through a natural language conversational interface
    """
    messages = [HumanMessage(content=input[-1].parts[-1].content)]
    input = {"messages": messages}
    logger.info(f'Processing messages: {input}')

    # demo - if keycloak is enabled, try to acquire token
    try:
        if os.getenv("KEYCLOAK_URL"):
            token = get_token()
            logger.info(f'received token: {token}')
    except Exception as e:
        yield {"message": {'type': 'run.failed', 'error': str(e)}}
        raise ACPError(Error(code=ErrorCode.SERVER_ERROR, message=str(e))) 

    try:
        output = None
        # Test MCP connection first
        logger.info(f'Attempting to connect to MCP server at: {os.getenv("MCP_URL", "http://localhost:8000/sse")}')
        
        mcpclient = get_mcpclient()
        
        # Try to get tools to verify connection
        try:
            tools = await mcpclient.get_tools()
            logger.info(f'Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}')
        except Exception as tool_error:
            logger.error(f'Failed to connect to MCP server: {tool_error}')
            yield MessagePart(content=f"Error: Cannot connect to MCP weather service at {os.getenv('MCP_URL', 'http://localhost:8000/sse')}. Please ensure the weather MCP server is running. Error: {tool_error}")
            return
            
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
            logger.info(f'event: {event}')
        output =  output.get("assistant", {}).get("final_answer")
        yield MessagePart(content=str(output))
    except Exception as e:
        logger.error(f'Graph execution error: {e}')
        yield MessagePart(content=f"Error: Failed to process weather request. {str(e)}")
        raise ACPError(Error(code=ErrorCode.SERVER_ERROR, message=str(e)))


def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
