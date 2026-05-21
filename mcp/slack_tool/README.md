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
| `JWKS_URI`               | No        | - | If populated, will perform token validation using the JWKS endpoint |
| `ISSUER`                 | No        | - | If populated with `JWKS_URI`, will additionally check the `iss` claim during token validation |
| `ADMIN_SLACK_BOT_TOKEN`  | No        | - | Bot token for Slack server with Admin privileges. Required for fine grained authz |
| `ADMIN_SCOPE_NAME`       | No        | - | Scope that triggers `ADMIN_SLACK_BOT_TOKEN` to be used |

Note: `JWKS_URI` triggers token validation at runtime. `ISSUER` will not affect behavior if `JWKS_URI` is not implemented. 

Note: Fine-grained authz is enabled with `ADMIN_SLACK_BOT_TOKEN` and `ADMIN_SCOPE_NAME`. If a received access token includes the `ADMIN_SCOPE_NAME` as a scope, it will use the `ADMIN_SLACK_BOT_TOKEN`

You can run this locally with `uv run slack_tool.py` so long as the `SLACK_BOT_TOKEN` is set. 