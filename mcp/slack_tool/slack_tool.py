import os
import sys
import logging
from typing import List, Dict, Any
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token, AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def get_client_id() -> str:
    """
    Read the SVID JWT from file and extract the client ID from the "sub" claim.
    """
    # Read SVID JWT from file to get client ID
    jwt_file_path = "/opt/jwt_svid.token"
    try:
        with open(jwt_file_path, "r") as file:
            content = file.read()

    except FileNotFoundError:
        print(f"Error: The file {jwt_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    if content is None or content.strip() == "":
        raise Exception(f"No content read from SVID JWT.")

    decoded = jwt.decode(content, options={"verify_signature": False})
    if "sub" not in decoded:
        raise Exception('SVID JWT does not contain a "sub" claim.')
    return decoded["sub"]

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

def get_slack_client(access_token=None):
    if ADMIN_SLACK_BOT_TOKEN is None:
        logger.debug("No ADMIN_SLACK_BOT_TOKEN configured - automatically configuring based on SLACK_BOT_TOKEN. ")
        return slack_client_from_bot_token(SLACK_BOT_TOKEN)
    # we do fine-grained authz
    if access_token is None:
        logger.error("ADMIN_SLACK_BOT_TOKEN configured, but no access token passed. ")
        return None
    
    # Access token is now claims dict from FastMCP AccessToken
    access_token_scopes = access_token.get("scope", "").split() if access_token.get("scope") else []
    logger.debug(f"Received scopes: {access_token_scopes}")
    admin_scope = os.getenv("ADMIN_SCOPE_NAME")
    is_admin = admin_scope in access_token_scopes
    logger.debug(f"Is admin: {is_admin}")
    if is_admin:
        return slack_client_from_bot_token(ADMIN_SLACK_BOT_TOKEN)
    return slack_client_from_bot_token(SLACK_BOT_TOKEN)


# Create FastMCP app
# Temporary environment variables to manually create verifier
verifier = None
JWKS_URI = os.getenv("JWKS_URI")
ISSUER = os.getenv("ISSUER")
CLIENT_ID = get_client_id()
if not JWKS_URI is None:
    verifier = JWTVerifier(
        jwks_uri = JWKS_URI,
        issuer = ISSUER,
        audience = CLIENT_ID # the CLIENT_ID will be the same as the AUDIENCE
    )
mcp = FastMCP("Slack", auth=verifier)

@mcp.tool()
def get_channels() -> List[Dict[str, Any]]:
    """
    Lists all public and private slack channels you have access to.
    """
    logger.debug(f"Called get_channels tool")

    # Get the current authenticated user's token using FastMCP 2.0 dependency
    access_token: AccessToken | None = get_access_token()
    slack_client = get_slack_client(access_token=access_token.claims if access_token else None)
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

    access_token: AccessToken | None = get_access_token()
    slack_client = get_slack_client(access_token=access_token.claims if access_token else None)
    if slack_client is None:
        return [{"error": f"Could not start slack client. Check the configured bot token"}]

    try:
        # Call the Slack API to list conversations the bot is part of.
        response = slack_client.conversations_history(
            channel=channel_id,
            limit=limit
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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)

if __name__ == "__main__":
    if SLACK_BOT_TOKEN is None: # default slack token
        logger.warning("Please configure the SLACK_BOT_TOKEN environment variable before running the server")
    elif ADMIN_SLACK_BOT_TOKEN is None: # one token set -> we just validate the JWT
        logger.info("Configured SLACK_BOT_TOKEN environment variable but not ADMIN_SLACK_BOT_TOKEN; all requests will use the `SLACK_BOT_TOKEN` to reach the Slack API")
        logger.info("Starting Slack MCP Server")
        run_server()
    else: # two tokens set -> we validate the JWT and connect to slack based on access token scope
        # check if other required Auth variables are set
        auth = os.getenv("JWKS_URI")
        if auth is None:
            logger.error("Configured ADMIN_SLACK_BOT_TOKEN but auth is not configured - fine grained authz requires token validation")
        else: 
            logger.info("Configured two slack tokens; finer-grained authz enabled")
            logger.info("Starting Slack MCP Server")
            run_server()
