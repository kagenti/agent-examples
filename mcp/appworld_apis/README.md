# AppWorld APIs MCP Tool


This MCP tool packages and serves [AppWorld](https://github.com/StonyBrookNLP/appworld), a benchmark and execution environment for function-calling and interactive coding agents.  
AppWorld provides a simulated ecosystem of apps and users, with rich API surfaces and tasks designed to evaluate tool use, multi-step reasoning, and end-to-end agent behavior.  
This deployment exposes both the AppWorld REST API surface (`8000`) and the AppWorld MCP interface (`8001`) from a single service.

## Deploy In Kagenti

Deploy this as an MCP tool in the Kagenti UI.

During setup, configure the tool so both ports are available:

- `8000` for REST APIs
- `8001` for MCP

## Kagenti UI Bug Workaround (Required)

There is a current Kagenti UI bug where additional ports are ignored during deployment:
`https://github.com/kagenti/kagenti/issues/646`

After deploying, manually add the missing `8001` port in both the Deployment and Service.

### 1) Edit Deployment

```bash
kubectl edit deployment <appworld-api-deployment-name> -n <namespace>
```

Find:

```yaml
        - containerPort: 8000
          name: http
          protocol: TCP
```

Update to:

```yaml
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 8001
          name: mcp
          protocol: TCP
```

### 2) Edit Service

```bash
kubectl edit service <appworld-api-service> -n <namespace>
```

Find:

```yaml
  - name: http
    port: 8000
    protocol: TCP
    targetPort: 8000
```

Update to:

```yaml
  - name: http
    port: 8000
    protocol: TCP
    targetPort: 8000
  - name: mcp
    port: 8001
    protocol: TCP
    targetPort: 8001
```
