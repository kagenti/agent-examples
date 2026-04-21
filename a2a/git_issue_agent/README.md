# Github Issue Agent

## Introduction

The Github Issue Agent is designed to perform tasks dealing with Github issues, leveraging large language models (LLMs) to generate plans, execute steps, and summarize findings. This agent integrates with [the upstream Github MCP Server](https://github.com/github/github-mcp-server) as a tool via MCP and utilizes AI models for data extraction, summarization, and more. This agent also integrates with [this MCP Man-in-the-Middle](https://github.com/kagenti/agent-examples/tree/main/mcp/github_tool) that will verify OAuth tokens and can be configured with Github PATs to communicate with the upstream Github MPC Server. 

## Architecture

<TODO>

### Configuration Variables

| Variable Name | Description | Required? | Default Value |
|---------------|-------------|-----------|---------------|
| LLM_API_BASE | The URL for OpenAI compatible API | Yes | `http://localhost:11434/v1` |
| LLM_API_KEY | The API key for OpenAI compatible API | No |  `my_api_key` |
| TASK_MODEL_ID | The ID of the LLM | Yes | `granite3.3:8b` |
| EXTRA_HEADERS | Extra headers for the OpenAI API, e.g. {"MY_HEADER": "my_value"} | No | `{}` |
| MODEL_TEMPERATURE | The temperature for the model | Yes | `0` |
| MCP_URL | Endpoint where the Slack MCP server can be found | No |  "" |
| PORT | Port on which the service will run | Yes | `8000` |
| LOG_LEVEL | Application log level | No | DEBUG |
| GITHUB_TOKEN | If set, will send requests to Github MCP Server with `Authorization: Bearer <GITHUB_TOKEN>` header. | No | - |
| JWKS_URI | Endpoint to obtain JWKS for token validation. Enables token validation. | No | - |
| ISSUER | Expected `iss` value of incoming bearer tokens | No | - |
| TOKEN_URL | Endpoint to perform token exchange. Required for token exchange. | No | - |
| CLIENT_ID | Client ID to authenticate to auth server with. Required for token exchange. Expected `aud` value of incoming bearer tokens | No | - |
| CLIENT_SECRET | Client secret to authenticate to auth server with. Required for token exchange. | No | - |
| TARGET_SCOPES | Requested scopes of token exchanged token | No | - |
| TARGET_AUDIENCE | Requested audience of token exchanged token | No | - |

> **Note on Authorization configuration**
> By default, no token validation is performed. To enable token validation, set `JWKS_URI`.
> If `ISSUER` is additionally set, the `iss` claim will be checked to equal this value.
> If all of `TOKEN_URL`, `CLIENT_ID`, and `CLIENT_SECRET` are set in addition, token exchange will be performed using Bearer tokens from incoming requests, to send to the MCP endpoint.
> In addition to `TOKEN_URL`, `CLIENT_ID`, `CLIENT_SECRET`, which trigger token exchange, `TARGET_SCOPES` can be optionally configured to be the `scope` in the token exchange request.