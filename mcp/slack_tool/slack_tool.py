import os
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# set configurations from environment variables
#ISSUER_URL = os.getenv("ISSUER_URL")
#JWKS_URL = f"{ISSUER_URL}/protocol/openid-connect/certs"
#CLIENT_ID = os.getenv("CLIENT_ID")

# slack bot token to access slack api
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "YOUR_SLACK_BOT_TOKEN")
try: 
    slack_client = WebClient(token=SLACK_BOT_TOKEN)
    auth_test = slack_client.auth_test()
    print(f"Successfully authenticated as bot '{auth_test['user']}' in workspace '{auth_test['team']}'.")
except SlackApiError as e:
    # Handle authentication errors, such as an invalid token
    print(f"Error authenticating with Slack: {e.response['error']}")
    slack_client = None
except Exception as e:
    print(f"An unexpected error occurred during Slack client initialization: {e}")
    slack_client = None

mcp = FastMCP("Slack", stateless_http=True, port=8000)

@mcp.tool()
def get_channels() -> List[Dict[str, Any]]:
    """
    Lists all public and private channels the bot has access to.
    The docstring is crucial as it becomes the tool's description for the LLM.
    """
    if slack_client is None: 
        return [{"error": "Slack client not initialized"}]
    
    try:
        # Call the conversations_list method to get public channels
        result = slack_client.conversations_list(types="public_channel")
        channels = result.get("channels", [])
        print(channels)
        # We'll just return some key information for each channel
        return [
            {"id": c["id"], "name": c["name"], "purpose": c.get("purpose", {}).get("value", "")}
            for c in channels
        ]
    except SlackApiError as e:
        # Handle API errors and return a descriptive message
        return [{"error": f"Slack API Error: {e.response['error']}"}]
    except Exception as e:
        return [{"error": f"An unexpected error occurred: {e}"}]

@mcp.tool()
def get_channel_history(channel_id: str, limit: int = 20) -> List:
    """
    Fetches the most recent messages from a specific Slack channel ID.

    Args:
        channel_id: The ID of the channel (e.g., 'C024BE91L').
        limit: The maximum number of messages to return (default is 20).
    """
    print("Tool executed: get_slack_channels")
    try:
        # Call the Slack API to list conversations the bot is part of.
        response = slack_client.conversations_history(
            channel=channel_id
        )
        return response.get("messages",)
    except SlackApiError as e:
        # Handle API errors and return a descriptive message
        return [{"error": f"Slack API Error: {e.response['error']}"}]
    except Exception as e:
        return [{"error": f"An unexpected error occurred: {e}"}]


# host can be specified with HOST env variable
def run_server():
    mcp.run(transport="streamable-http") 

if __name__ == "__main__":
    if not slack_client or SLACK_BOT_TOKEN == "YOUR_SLACK_BOT_TOKEN":
        print("Please configure the SLACK_BOT_TOKEN environment variable before running the server")
    else:
        print("Starting Slack MCP Server")
        run_server()
