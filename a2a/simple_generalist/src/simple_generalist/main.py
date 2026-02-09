"""Main entry point for the Simple Generalist Agent."""

import logging
import sys
import uvicorn

from simple_generalist.config import Settings
from simple_generalist.a2a_server import create_app

logger = logging.getLogger(__name__)


def setup_logging(settings: Settings):
    """
    Setup logging configuration.
    
    Args:
        settings: Application settings
    """
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )


def run():
    """Run the Simple Generalist server."""
    # Load settings
    settings = Settings()  # type: ignore[call-arg]
    
    # Setup logging
    setup_logging(settings)
    
    logger.info("Starting Simple Generalist Agent")
    logger.info(f"LLM Model: {settings.LLM_MODEL}")
    logger.info(f"Max Iterations: {settings.MAX_ITERATIONS}")
    
    if settings.MCP_SERVER_URL.strip():
        logger.info(f"MCP Server URL: {settings.MCP_SERVER_URL.strip()}")
    else:
        logger.warning("No MCP server configured - agent will run without tools")
    
    # Create A2A app (MCP connection will be established per-request)
    try:
        app = create_app(settings)
    except Exception as exc:
        logger.error(f"Failed to create A2A app: {exc}", exc_info=True)
        sys.exit(1)
    
    # Run server
    logger.info(f"Starting A2A server on {settings.A2A_HOST}:{settings.A2A_PORT}")
    uvicorn.run(
        app,
        host=settings.A2A_HOST,
        port=settings.A2A_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run()

# Made with Bob
