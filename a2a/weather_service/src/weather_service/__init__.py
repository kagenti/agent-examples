"""Weather Service - A2A weather agent with zero observability code.

All tracing and observability is handled externally by the AuthBridge
ext_proc sidecar which creates root spans and nested child spans from
the A2A SSE event stream. No OTEL dependencies needed in the agent.
"""
