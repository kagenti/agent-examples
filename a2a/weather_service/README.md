# Agents

The Weather Service Agent is an example of an [A2A](https://a2a-protocol.org/latest/) agent.

This agent depends on the Rossoctl [Weather Tool](https://github.com/rossoctl/examples/tree/main/mcp/weather_tool).  The weather tool should be running before chatting with the weather service agent.

## Run the agent on Kubernetes with Rossoctl

You may deploy using Rossoctl's UI or through a Kubernetes manifest.

### Deploy using Rossoctl's UI

Rossoctl's UI is aware of this example agent.  To deploy through the UI

- Browse to http://rossoctl-ui.localtest.me:8080/agents/
- Build from source
- Weather service agent
- Expand Environment Variables
  - Import from File/URL, URL, https://raw.githubusercontent.com/rossoctl/examples/refs/heads/main/a2a/weather_service/.env.openai
    - If using [Ollama](https://ollama.com/), instead of the default use https://raw.githubusercontent.com/rossoctl/examples/refs/heads/main/a2a/weather_service/.env.ollama
  - Fetch and parse
  - Import
- Build and deploy agent
- Chat
  - `What is the weather in New York?`

### Deploy using a Kubernetes deployment manifest

Deploy the sample manifest:

```bash
kubectl apply -f deployment/k8s.yaml
```
