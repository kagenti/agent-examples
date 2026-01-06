
# MCP-MitM

This is a man-in-the-middle that facades to MCP tools of any MCP HTTPStreamable server, and does some cool OIDC to APIKey stuff.

# Develop and debug locally

To use it, `go run cli/main/main.go`.  Most secured MCP servers expect authorization to get a list of tools, and this
is provided with `$INIT_AUTH_HEADER`.  

The code can be tested locally.  For example, to facade the public GitHub MCP server,

```bash
UPSTREAM_MCP="https://api.githubcopilot.com/mcp/" \
INIT_AUTH_HEADER="Bearer ${GITHUB_TOKEN}" \
REQUIRED_SCOPE=profile \
UPSTREAM_HEADER_TO_USE_IF_IN_AUDIENCE="Bearer $GITHUB_TOKEN" \
UPSTREAM_HEADER_TO_USE_IF_NOT_IN_AUDIENCE="Bearer rutabaga" \
go run cli/main/main.go
```

First, get a `GITHUB_TOKEN` by creating one at https://github.com/settings/tokens

At this point, the server localhost:9090 will be offering the same tools as the upstream.

To use it, we must first follow the standard MCP HTTPSTreamable init dance:

```bash
MCP=http://localhost:9090/mcp
curl --include -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream"  ${MCP} --data '
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "roots": {
        "listChanged": true
      },
      "sampling": {}
    },
    "clientInfo": {
      "name": "ExampleClient",
      "version": "1.0.0"
    }
  }
}
' | tee /tmp/init-response.txt

SESSION_ID=$(cat /tmp/init-response.txt | grep -i mcp-session-id: | sed 's/mcp-session-id: //I' | sed 's/\r//g')
echo SESSION_ID is "${SESSION_ID}"

curl -v -X POST -H "mcp-session-id: ${SESSION_ID}" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" ${MCP} --data '
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
'
```

At this point, optionally, you can ask for tools to verify everything worked:

```bash
curl -v ${MCP} -X POST -H "mcp-session-id: ${SESSION_ID}" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" --data '
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list"
}
' | jq
```

Now we want to invoke a tool, to prove it works.  This facade expects an OIDC token.  We'll need one.  To get one from Keycloak in a Kagenti environment:

```bash
KEYCLOAK_USER=admin
KEYCLOAK_PASSWORD=... # use yours
KAGENTI_SECRET=$(oc -n kagenti-system get secret kagenti-ui-oauth-secret -o jsonpath="{.data.CLIENT_SECRET}" | base64 -d)
MITM_TOKEN=$(curl -sX POST -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_secret=$KAGENTI_SECRET" -d "client_id=kagenti" \
  -d "grant_type=password" \
  -d "username=${KEYCLOAK_USER}" -d "password=${KEYCLOAK_PASSWORD}" \
    "http://keycloak.localtest.me:8080/realms/master/protocol/openid-connect/token" | jq --raw-output ".access_token")
echo '$MITM_TOKEN' is ${MITM_TOKEN}
MITM_AUTH="Bearer ${MITM_TOKEN}"
```

REQUIRED_SCOPE tells the man-in-the-middle the expected scope to find.

```bash
export REQUIRED_SCOPE=profile
export UPSTREAM_HEADER_TO_USE_IF_IN_AUDIENCE="Bearer $GITHUB_TOKEN"
export UPSTREAM_HEADER_TO_USE_IF_NOT_IN_AUDIENCE="Bearer rutabaga"
```

To test, we can invoke the man-in-the-middle MCP server with the token from Keycloak simulating how an OIDC-secured Agent
would invoke it:

```bash
curl -v ${MCP} -H "Authorization: ${MITM_AUTH}" -H "mcp-session-id: ${SESSION_ID}" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" --data '
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "list_branches",
    "arguments": {
      "owner": "kagenti",
      "repo": "kagenti"
    }
  }
}
' | jq
```

If this works, you should see the list of tools, as JSON.

# Deploying in Kagenti

First, you'll need to give Kagenti credentials.  For the GitHub OAuth API key demo, you'll need to supply API keys

oc -n team1 get configmap environments -o yaml > /tmp/environments.yaml
(edit /tmp/environments.yaml) creating the following, and replacing `${GITHUB_TOKEN}` with your GitHub personal access token (PAT):

```YAML
data:
  UPSTREAM_MCP: https://api.githubcopilot.com/mcp/
  REQUIRED_SCOPE: profile
  INIT_AUTH_HEADER: Bearer ${GITHUB_TOKEN}
  UPSTREAM_HEADER_TO_USE_IF_IN_AUDIENCE: Bearer $GITHUB_TOKEN
  UPSTREAM_HEADER_TO_USE_IF_NOT_IN_AUDIENCE: Bearer rutabaga
```

After adding the new env vars, apply to Kagenti using `kubectl apply -n team1 -f /tmp/environments.yaml`.

Now that the environment variables are available, start an instance of the tool

- Browse to http://kagenti-ui.localtest.me:8080/Import_New_Tool
- Select namespace 
- Set the Target Port to 9090
- Specify Subfolder `mcp/github_tool`
- Click "Build & Deploy New Tool" to deploy.

Once the tool is deployed, if you wish to test it from outside Kagenti you'll need to create an HTTPRoute for it.

```bash
cat <<EOF | > /tmp/expose-github-tool.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  labels:
    mcp-server: "true"
  name: github-route-ext
  namespace: team1
spec:
  hostnames:
  - github-tool.127-0-0-1.sslip.io
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: mcp-gateway
    namespace: gateway-system
  rules:
  - backendRefs:
    - group: ""
      kind: Service
      name: github-tool
      port: 8000
      weight: 1
    matches:
    - path:
        type: PathPrefix
        value: /
EOF
oc --context kind-agent-platform apply -f /tmp/expose-github-tool.yaml
```

At this point, you can do the MCP curls above if you define `MCP=http://github-tool.127-0-0-1.sslip.io:8888/mcp`

