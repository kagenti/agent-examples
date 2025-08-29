# Slack MCP Server

This is a simple Slack MCP Server with two tools:

- `get_channels` to list all public and private channels the bot has access to
- `get_channel_history` to list messages from a specific channel by `channel_id`

You can configure the server with the following environment variables:

| Variable name            | Required? | Default                | Description |
| ------------------------ | --------- | ---------------------- | ----------------------------- |
| `SLACK_BOT_TOKEN`        | Yes       | - | Bot token for the Slack server. Required for any functionality |
| `LOG_LEVEL`              | No        | `DEBUG`                | Application log level |
| `MCP_TRANSPORT`          | No        | `streamable-http`      | Passed into mcp.run to determine mcp transport |
| `ISSUER`                 | No        | - | If populated, will publish that it is OAuth-secured by this issuer (but no actual verification). Must be URI format |
| `INTROSPECTION_ENDPOINT` | No        | - | If populated with `CLIENT_ID` and `CLIENT_SECRET`, will introspect access tokens at this endpoint |
| `CLIENT_ID`              | No        | - | If populated with `INTROSPECTION_ENDPOINT` and `CLIENT_SECRET`, will introspect access tokens using this as the client id to authenticate |
| `CLIENT_SECRET`          | No        | - | If populated with `INTROSPECTION_ENDPOINT` and `CLIENT_ID`, will introspect access tokens using this as the client secret to authenticate |
| `AUDIENCE `              | No        | - | If populated with `INTROSPECTION_ENDPOINT` will perform audience validation |
| `ADMIN_SLACK_BOT_TOKEN`  | No        | - | Bot token for Slack server with Admin privileges. Required for fine grained authz |
| `ADMIN_SCOPE_NAME`       | No        | - | Scope that triggers `ADMIN_SLACK_BOT_TOKEN` to be used |

Note: `ISSUER` only affects the published authorization endpoint. All three of `INTROSPECTION_ENDPOINT`, `CLIENT_ID`, and `CLIENT_SECRET` are required for token validation to occur. `AUDIENCE` enables the additional audience check. 

Note: Fine-grained authz is enabled with `ADMIN_SLACK_BOT_TOKEN` and `ADMIN_SCOPE_NAME`. If a received access token includes the `ADMIN_SCOPE_NAME` as a scope, it will use the `ADMIN_SLACK_BOT_TOKEN`

You can run this locally with `uv run slack_tool.py` so long as the `SLACK_BOT_TOKEN` is set. 