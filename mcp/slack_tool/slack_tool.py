import os
import sys
import logging
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from mcp.server.auth.middleware.auth_context import get_access_token
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from auth import get_token_verifier, get_auth

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"), stream=sys.stdout, format='%(levelname)s: %(message)s')

# setup slack client
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
ADMIN_SLACK_BOT_TOKEN = os.getenv("ADMIN_SLACK_BOT_TOKEN")

def slack_client_from_bot_token(bot_token):
    try: 
        slack_client = WebClient(token=bot_token)
        auth_test = slack_client.auth_test()
        logger.info(f"Successfully authenticated as bot '{auth_test['user']}' in workspace '{auth_test['team']}'.")
        return slack_client
    except SlackApiError as e:
        # Handle authentication errors, such as an invalid token
        logger.error(f"Error authenticating with Slack: {e.response['error']}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during Slack client initialization: {e}")
        return None

def get_slack_client(access_token = None):
    if ADMIN_SLACK_BOT_TOKEN is None:
        logger.debug(f"No ADMIN_SLACK_BOT_TOKEN configured - automatically configuring based on SLACK_BOT_TOKEN. ")
        return slack_client_from_bot_token(SLACK_BOT_TOKEN)
    # we do fine-grained authz
    if access_token is None:
        logger.error(f"ADMIN_SLACK_BOT_TOKEN configured, but no access token passed. ")
        return None
    access_token_scopes = access_token.scopes
    logger.debug(f"Received scopes: {access_token_scopes}")
    admin_scope = os.getenv("ADMIN_SCOPE_NAME")
    is_admin = admin_scope in access_token_scopes
    logger.debug(f"Is admin: {is_admin}")
    if is_admin:
        return slack_client_from_bot_token(ADMIN_SLACK_BOT_TOKEN)
    return slack_client_from_bot_token(SLACK_BOT_TOKEN)


mcp = FastMCP("Slack", host="0.0.0.0", port=8000,
              token_verifier=get_token_verifier(),
              auth=get_auth(),
        )

@mcp.tool()
def get_channels() -> List[Dict[str, Any]]:
    """
    Lists all public and private channels the bot has access to.
    The docstring is crucial as it becomes the tool's description for the LLM.
    """
    logger.debug(f"Called get_channels tool")


    slack_client = get_slack_client(access_token=get_access_token())
    if slack_client is None:
        return [{"error": f"Could not start slack client. Check the configured bot token"}]

    try:
        # Call the conversations_list method to get public channels
        result = slack_client.conversations_list(types="public_channel")
        channels = result.get("channels", [])
        # We'll just return some key information for each channel
        logger.debug(f"Successful get_channels call: {channels}")
        return [
            {"id": c["id"], "name": c["name"], "purpose": c.get("purpose", {}).get("value", "")}
            for c in channels
        ]
    except SlackApiError as e:
        # Handle API errors and return a descriptive message
        logger.error(f"Slack API Error: {e.response['error']}")
        return [{"error": f"Slack API Error: {e.response['error']}"}]
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        return [{"error": f"An unexpected error occurred: {e}"}]

@mcp.tool()
def get_channel_history(channel_id: str, limit: int = 20) -> List:
    """
    Fetches the most recent messages from a specific Slack channel ID.

    Args:
        channel_id: The ID of the channel (e.g., 'C024BE91L').
        limit: The maximum number of messages to return (default is 20).
    """
    logger.debug(f"Called get_channel_history tool: {channel_id}")

    slack_client = get_slack_client(access_token=get_access_token())
    if slack_client is None:
        return [{"error": f"Could not start slack client. Check the configured bot token"}]

    try:
        # Call the Slack API to list conversations the bot is part of.
        response = slack_client.conversations_history(
            channel=channel_id
        )
        logger.debug(f"Successful get_channel_history call: {response}")
        return response.get("messages",)
    except SlackApiError as e:
        # Handle API errors and return a descriptive message
        return [{"error": f"Slack API Error: {e.response['error']}"}]
    except Exception as e:
        return [{"error": f"An unexpected error occurred: {e}"}]

# host can be specified with HOST env variable
# transport can be specified with MCP_TRANSPORT env variable (defaults to streamable-http)
def run_server():
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    mcp.run(transport=transport)

if __name__ == "__main__":
    if SLACK_BOT_TOKEN is None: # default slack token
        logger.warning("Please configure the SLACK_BOT_TOKEN environment variable before running the server")
    elif ADMIN_SLACK_BOT_TOKEN is None: # one token set -> we just validate the JWT
        logger.info("Configured SLACK_BOT_TOKEN environment variable but not ADMIN_SLACK_BOT_TOKEN; will validate token signature")
        logger.info("Starting Slack MCP Server")
        run_server()
    else: # two tokens set -> we validate the JWT and connect to slack based on access token scope
        # check if other required Auth variables are set
        introspection_endpoint = os.getenv("INTROSPECTION_ENDPOINT")
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        expected_audience = os.getenv("AUDIENCE")
        admin_scope = os.getenv("ADMIN_SCOPE_NAME")
        issuer = os.getenv("ISSUER")
        if None in [introspection_endpoint, client_id, client_secret, expected_audience, issuer, admin_scope]:
            logger.error("Configured ADMIN_SLACK_BOT_TOKEN but not one or more of INTROSPECTION_ENDPOINT, CLIENT_ID, CLIENT_SECRET, AUDIENCE, ISSUER. ")
        else: 
            logger.info("Configured two slack tokens; finer-grained authz enabled")
            logger.info("Starting Slack MCP Server")
            run_server()
