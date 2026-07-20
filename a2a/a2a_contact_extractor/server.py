"""
This is a sample agent that uses the Marvin framework to extract structured contact information from text.
It is integrated with the Agent2Agent (A2A) protocol.

The server logic lives here (rather than in ``__main__.py``) so it can be exposed
as the ``server`` console-script entry point: a ``__main__`` module cannot be
referenced from ``[project.scripts]`` because the generated wrapper script is
itself imported as ``__main__``.
"""

import logging
import os

import click
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from starlette.applications import Starlette

from agent import ExtractorAgent
from agent_executor import ExtractorAgentExecutor  # type: ignore[import-untyped]

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContactInfo(BaseModel):
    """Structured contact information extracted from text."""

    name: str = Field(description="Person's first and last name")
    email: EmailStr = Field(description="Email address")
    phone: str = Field(description="Phone number if present")
    organization: str | None = Field(
        None, description="Organization or company if mentioned"
    )
    role: str | None = Field(None, description="Job title or role if mentioned")


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10030)
@click.option("--result-type", "result_type", default="ContactInfo")
@click.option(
    "--instructions",
    "instructions",
    default="Politely interrogate the user for their contact information. The schema of the result type implies what things you _need_ to get from the user.",
)
def main(host, port, result_type, instructions):
    """Starts the Marvin Contact Extractor Agent server."""
    try:
        result_type = eval(result_type)
    except Exception as e:
        logger.error(f"Invalid result type: {e}")
        exit(1)
    agent = ExtractorAgent(instructions=instructions, result_type=result_type)
    agent_card = get_agent_card(host, port)

    request_handler = DefaultRequestHandler(
        agent_executor=ExtractorAgentExecutor(agent=agent),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    import uvicorn

    routes = []
    # create_agent_card_routes serves /.well-known/agent-card.json natively
    routes.extend(create_agent_card_routes(agent_card))
    # enable_v0_3_compat is needed because Rossoctl uses A2A 0.3 client libraries
    routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))
    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the ExtractorAgent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="extract_contacts",
        name="Contact Information Extraction",
        description="Extracts structured contact information from text",
        tags=["contact info", "structured extraction", "information extraction"],
        examples=[
            "My name is John Doe, email: john@example.com, phone: (555) 123-4567"
        ],
    )
    return AgentCard(
        name="Marvin Contact Extractor",
        description="Extracts structured contact information from text using Marvin's extraction capabilities",
        version="1.0.0",
        default_input_modes=ExtractorAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=ExtractorAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
        supported_interfaces=[
            AgentInterface(
                # Allow env var AGENT_ENDPOINT to override the URL in the agent card
                url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/")
                + "/",
                protocol_binding="JSONRPC",
            )
        ],
    )


if __name__ == "__main__":
    main()
