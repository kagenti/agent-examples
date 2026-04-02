"""
OpenTelemetry observability setup for Sandbox Agent.

Key Features:
- Tracing middleware for root span with MLflow attributes
- Auto-instrumentation of LangChain with OpenInference
- Resource attributes for static agent metadata
- W3C Trace Context propagation for distributed tracing

Phase 1: Root span + auto-instrumentation only.
Node-level manual spans will be added in a later phase.
"""

import json
import logging
import os
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# Agent metadata (static, used in Resource and spans)
AGENT_NAME = os.getenv("SANDBOX_AGENT_NAME", "sandbox-legion")
AGENT_VERSION = "1.0.0"
AGENT_FRAMEWORK = "langgraph"

# ContextVar to pass root span from middleware to agent code.
# This allows execute() to access the middleware-created root span
# even though trace.get_current_span() would return a child span.
_root_span_var: ContextVar = ContextVar("root_span", default=None)


def get_root_span():
    """Get the root span created by tracing middleware.

    Use this instead of trace.get_current_span() when you need to set
    attributes on the root span (e.g., mlflow.spanOutputs for streaming).

    Returns:
        The root span, or None if not in a traced request context.
    """
    return _root_span_var.get()


# OpenInference semantic conventions
try:
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False
    logger.warning("openinference-semantic-conventions not available")


def _get_otlp_exporter(endpoint: str):
    """Get HTTP OTLP exporter."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    if not endpoint.endswith("/v1/traces"):
        endpoint = endpoint.rstrip("/") + "/v1/traces"
    return OTLPSpanExporter(endpoint=endpoint)


def setup_observability() -> bool:
    """
    Set up OpenTelemetry tracing with OpenInference instrumentation.

    Call this ONCE at agent startup, before importing agent code.
    NEVER raises — all exceptions are caught and logged. OTel issues
    must never break the agent's main processing loop.

    Returns:
        True if tracing was set up successfully, False otherwise.
    """
    service_name = os.getenv("OTEL_SERVICE_NAME", "sandbox-agent")
    namespace = os.getenv("K8S_NAMESPACE_NAME", "team1")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    if not otlp_endpoint:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled. Set this env var to enable OpenTelemetry tracing."
        )
        return False

    try:
        return _setup_observability_inner(service_name, namespace, otlp_endpoint)
    except Exception:
        logger.exception("OTel setup failed — tracing disabled (agent continues without tracing)")
        return False


def _setup_observability_inner(service_name: str, namespace: str, otlp_endpoint: str) -> bool:
    """Internal setup — may raise. Called by setup_observability() which catches all errors."""
    from opentelemetry import trace
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    logger.info("=" * 60)
    logger.info("Setting up OpenTelemetry observability")
    logger.info("  Service: %s", service_name)
    logger.info("  Agent: %s", AGENT_NAME)
    logger.info("  Framework: %s", AGENT_FRAMEWORK)
    logger.info("  Namespace: %s", namespace)
    logger.info("  OTLP Endpoint: %s", otlp_endpoint)
    logger.info("=" * 60)

    # Create resource with service and MLflow attributes.
    # Resource attributes are STATIC and apply to ALL spans/traces.
    # See: https://mlflow.org/docs/latest/genai/tracing/opentelemetry/
    resource = Resource(
        attributes={
            # Standard OTEL service attributes
            SERVICE_NAME: service_name,
            SERVICE_VERSION: AGENT_VERSION,
            "service.namespace": namespace,
            "k8s.namespace.name": namespace,
            # MLflow static metadata (applies to all traces)
            "mlflow.traceName": AGENT_NAME,
            "mlflow.source": service_name,
            # GenAI static attributes
            "gen_ai.agent.name": AGENT_NAME,
            "gen_ai.agent.version": AGENT_VERSION,
            "gen_ai.system": AGENT_FRAMEWORK,
        }
    )

    # Create and configure tracer provider
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(_get_otlp_exporter(otlp_endpoint)))
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument LangChain with OpenInference
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
        logger.info("LangChain instrumented with OpenInference")
    except ImportError:
        logger.warning("openinference-instrumentation-langchain not available")

    # Configure W3C Trace Context propagation
    set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
    )

    # Instrument OpenAI for GenAI semantic conventions
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
        logger.info("OpenAI instrumented with GenAI semantic conventions")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-openai not available")

    return True


# Tracer for manual spans — use OpenInference-compatible name
_tracer = None
TRACER_NAME = "openinference.instrumentation.agent"


def get_tracer():
    """Get tracer for creating manual spans."""
    from opentelemetry import trace

    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(TRACER_NAME)
    return _tracer


def enrich_current_span(**kwargs: Any) -> None:
    """Add attributes to the currently active span.

    Convenience helper so agent code can annotate spans without importing
    opentelemetry directly.

    Args:
        **kwargs: Attribute key-value pairs to set on the current span.
    """
    from opentelemetry import trace

    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in kwargs.items():
            span.set_attribute(key, value)


def create_tracing_middleware():
    """
    Create Starlette middleware that wraps all requests in a root tracing span.

    This middleware:
    1. Creates a root span BEFORE A2A handlers run
    2. Sets MLflow/GenAI attributes on the root span
    3. Parses A2A JSON-RPC request to extract user input
    4. Captures response to set output attributes
    5. For streaming (SSE) responses, sets status without capturing body

    Usage in agent.py:
        from sandbox_agent.observability import create_tracing_middleware
        app = server.build()
        app.add_middleware(BaseHTTPMiddleware, dispatch=create_tracing_middleware())
    """
    from opentelemetry import context
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse

    async def tracing_middleware(request: Request, call_next):
        # Skip non-API paths (health checks, agent card, etc.)
        if request.url.path in [
            "/health",
            "/ready",
            "/.well-known/agent-card.json",
            "/.well-known/agent-graph-card.json",
        ]:
            return await call_next(request)

        tracer = get_tracer()

        # Parse request body to extract user input and context
        user_input = None
        context_id = None
        message_id = None

        try:
            body = await request.body()
            if body:
                data = json.loads(body)
                # A2A JSON-RPC format: params.message.parts[0].text
                params = data.get("params", {})
                message = params.get("message", {})
                parts = message.get("parts", [])
                if parts and isinstance(parts, list):
                    user_input = parts[0].get("text", "")
                context_id = params.get("contextId") or message.get("contextId")
                message_id = message.get("messageId")
        except Exception as e:
            logger.debug("Could not parse request body: %s", e)

        # Break parent chain to make this a true root span.
        # Without this, the span would inherit parent from W3C Trace Context headers.
        empty_ctx = context.Context()
        detach_token = context.attach(empty_ctx)

        try:
            # Create root span with correct GenAI naming convention.
            # Per https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
            # Span name: "invoke_agent {gen_ai.agent.name}"
            span_name = f"invoke_agent {AGENT_NAME}"

            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,  # In-process agent (not remote service)
            ) as span:
                # Store span in ContextVar so agent code can access it.
                # trace.get_current_span() in execute() returns the innermost
                # span (A2A span), not our root span.
                span_token = _root_span_var.set(span)

                # === GenAI Semantic Conventions (Required) ===
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
                span.set_attribute("gen_ai.provider.name", AGENT_FRAMEWORK)
                span.set_attribute("gen_ai.agent.name", AGENT_NAME)
                span.set_attribute("gen_ai.agent.version", AGENT_VERSION)

                # Set input attributes (Prompt column in MLflow)
                if user_input:
                    span.set_attribute("gen_ai.prompt", user_input[:1000])
                    span.set_attribute("input.value", user_input[:1000])
                    span.set_attribute("mlflow.spanInputs", user_input[:1000])

                # Session tracking — use context_id or message_id as fallback
                session_id = context_id or message_id

                if session_id:
                    span.set_attribute("gen_ai.conversation.id", session_id)
                    span.set_attribute("mlflow.trace.session", session_id)
                    span.set_attribute("session.id", session_id)

                # MLflow trace metadata (appears in trace list columns)
                span.set_attribute("mlflow.spanType", "AGENT")
                span.set_attribute("mlflow.traceName", AGENT_NAME)
                span.set_attribute("mlflow.runName", f"{AGENT_NAME}-invoke")
                span.set_attribute("mlflow.source", os.getenv("OTEL_SERVICE_NAME", "sandbox-agent"))
                span.set_attribute("mlflow.version", AGENT_VERSION)

                # User tracking — extract from auth header if available
                auth_header = request.headers.get("authorization", "")
                if auth_header:
                    span.set_attribute("mlflow.user", "authenticated")
                    span.set_attribute("enduser.id", "authenticated")
                else:
                    span.set_attribute("mlflow.user", "anonymous")
                    span.set_attribute("enduser.id", "anonymous")

                # OpenInference span kind (for Phoenix)
                if OPENINFERENCE_AVAILABLE:
                    span.set_attribute(
                        SpanAttributes.OPENINFERENCE_SPAN_KIND,
                        OpenInferenceSpanKindValues.AGENT.value,
                    )

                try:
                    # Call the next handler (A2A)
                    response = await call_next(request)

                    # Try to capture response for output attributes.
                    # This only works for non-streaming responses.
                    if isinstance(response, Response) and not isinstance(response, StreamingResponse):
                        # Read response body — we MUST recreate response after
                        _chunks: list[bytes] = []
                        async for chunk in response.body_iterator:
                            _chunks.append(chunk)
                        response_body = b"".join(_chunks)

                        # Try to parse and extract output for MLflow
                        try:
                            if response_body:
                                resp_data = json.loads(response_body)
                                result = resp_data.get("result", {})
                                artifacts = result.get("artifacts", [])
                                if artifacts:
                                    parts = artifacts[0].get("parts", [])
                                    if parts:
                                        output_text = parts[0].get("text", "")
                                        if output_text:
                                            span.set_attribute("gen_ai.completion", output_text[:1000])
                                            span.set_attribute("output.value", output_text[:1000])
                                            span.set_attribute("mlflow.spanOutputs", output_text[:1000])
                        except Exception as e:
                            logger.debug("Could not parse response body: %s", e)

                        # Always recreate response since we consumed the iterator
                        span.set_status(Status(StatusCode.OK))
                        return Response(
                            content=response_body,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type,
                        )

                    # For streaming responses (SSE), just set status and return.
                    # Don't try to capture the full stream body.
                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    # Reset the ContextVar to avoid leaking span reference
                    _root_span_var.reset(span_token)
        finally:
            # Always detach the context to restore parent chain for other requests
            context.detach(detach_token)

    return tracing_middleware
