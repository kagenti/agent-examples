import logging
import os
import sys

import click
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from app.agent import CurrencyAgent
from app.agent_executor import CurrencyAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10000)
def main(host, port):
    """Starts the Currency Agent server."""
    try:
        # We don't check OPENAI_API_KEY here.  We want the agent pod to run even if OPENAI_API_KEY isn't defined.

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skill = AgentSkill(
            id="convert_currency",
            name="Currency Exchange Rates Tool",
            description="Helps with exchange values between various currencies",
            tags=["currency conversion", "currency exchange"],
            examples=["What is exchange rate between USD and GBP?"],
        )
        agent_card = AgentCard(
            name="Currency Agent",
            description="Helps with exchange rates for currencies",
            version="1.0.0",
            default_input_modes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
            supported_interfaces=[
                AgentInterface(
                    # Allow env var AGENT_ENDPOINT to override the URL in the agent card
                    url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
                    protocol_binding="JSONRPC",
                )
            ],
        )

        # --8<-- [start:DefaultRequestHandler]
        request_handler = DefaultRequestHandler(
            agent_executor=CurrencyAgentExecutor(),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )

        routes = []
        routes.extend(create_agent_card_routes(agent_card))
        # enable_v0_3_compat is needed because Kagenti uses A2A 0.3 client libraries
        routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))
        app = Starlette(routes=routes)

        uvicorn.run(app, host=host, port=port)
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
