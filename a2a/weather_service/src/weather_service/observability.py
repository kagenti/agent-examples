"""
OpenTelemetry observability setup for Weather Agent.

Key Features:
- Tracing middleware for root span with MLflow attributes
- Auto-instrumentation of LangChain with OpenInference
- Resource attributes for static agent metadata
- W3C Trace Context propagation for distributed tracing
"""

import json
import logging
import os
from contextvars import ContextVar
from typing import Dict, Any, Optional
from contextlib import contextmanager
from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.propagate import set_global_textmap, extract
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

logger = logging.getLogger(__name__)

# Agent metadata (static, used in Resource and spans)
AGENT_NAME = "weather-assistant"
AGENT_VERSION = "1.0.0"
AGENT_FRAMEWORK = "langchain"

# ContextVar to pass root span from middleware to agent code
# This allows execute() to access the middleware-created root span
# even though trace.get_current_span() would return a child span
_root_span_var: ContextVar = ContextVar('root_span', default=None)


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
    from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
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


def setup_observability() -> None:
    """
    Set up OpenTelemetry tracing with OpenInference instrumentation.

    Call this ONCE at agent startup, before importing agent code.
    """
    service_name = os.getenv("OTEL_SERVICE_NAME", "weather-service")
    namespace = os.getenv("K8S_NAMESPACE_NAME", "team1")
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.kagenti-system.svc.cluster.local:8335"
    )

    logger.info("=" * 60)
    logger.info("Setting up OpenTelemetry observability")
    logger.info(f"  Service: {service_name}")
    logger.info(f"  Namespace: {namespace}")
    logger.info(f"  OTLP Endpoint: {otlp_endpoint}")
    logger.info("=" * 60)

    # Create resource with service and MLflow attributes
    # Resource attributes are STATIC and apply to ALL spans/traces
    # See: https://mlflow.org/docs/latest/genai/tracing/opentelemetry/
    resource = Resource(attributes={
        # Standard OTEL service attributes
        SERVICE_NAME: service_name,
        SERVICE_VERSION: AGENT_VERSION,
        "service.namespace": namespace,
        "k8s.namespace.name": namespace,
        # MLflow static metadata (applies to all traces)
        # These appear in MLflow trace list columns
        "mlflow.traceName": AGENT_NAME,
        "mlflow.source": service_name,
        # GenAI static attributes
        "gen_ai.agent.name": AGENT_NAME,
        "gen_ai.agent.version": AGENT_VERSION,
        "gen_ai.system": AGENT_FRAMEWORK,
    })

    # Create and configure tracer provider
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(_get_otlp_exporter(otlp_endpoint))
    )
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument LangChain with OpenInference
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        logger.info("LangChain instrumented with OpenInference")
    except ImportError:
        logger.warning("openinference-instrumentation-langchain not available")

    # Configure W3C Trace Context propagation
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    # Instrument httpx for automatic traceparent propagation on outgoing requests.
    # This is critical for distributed tracing: langchain-mcp-adapters uses httpx
    # for streamable_http transport, so each MCP tool call will automatically carry
    # the current span's traceparent header to the MCP gateway (Envoy).
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("httpx instrumented for automatic trace context propagation")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-httpx not available - "
                       "MCP tool calls will not propagate trace context")

    # Instrument OpenAI for GenAI semantic conventions
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        logger.info("OpenAI instrumented with GenAI semantic conventions")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-openai not available")


# Tracer for manual spans - use OpenInference-compatible name
_tracer: Optional[trace.Tracer] = None
TRACER_NAME = "openinference.instrumentation.agent"


def get_tracer() -> trace.Tracer:
    """Get tracer for creating manual spans."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(TRACER_NAME)
    return _tracer


def _set_genai_mlflow_attributes(
    span,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input_text: Optional[str] = None,
):
    """Set GenAI and MLflow attributes on a span."""
    # === GenAI Semantic Conventions ===
    if context_id:
        span.set_attribute("gen_ai.conversation.id", context_id)
    if input_text:
        span.set_attribute("gen_ai.prompt", input_text[:1000])
        span.set_attribute("input.value", input_text[:1000])
    span.set_attribute("gen_ai.agent.name", "weather-assistant")
    span.set_attribute("gen_ai.system", "langchain")

    # OpenInference span kind
    if OPENINFERENCE_AVAILABLE:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.AGENT.value
        )

    # === MLflow-specific Attributes ===
    # TODO: Could be handled by OTEL Collector transform/genai_to_mlflow
    if input_text:
        span.set_attribute("mlflow.spanInputs", input_text[:1000])
    span.set_attribute("mlflow.spanType", "AGENT")
    span.set_attribute("mlflow.traceName", "weather-assistant")
    span.set_attribute("mlflow.source", "weather-service")
    if context_id:
        span.set_attribute("mlflow.trace.session", context_id)
    if user_id:
        span.set_attribute("mlflow.user", user_id)
        span.set_attribute("enduser.id", user_id)

    # Custom attributes
    if task_id:
        span.set_attribute("a2a.task_id", task_id)
    if user_id:
        span.set_attribute("user.id", user_id)


@contextmanager
def enrich_current_span(
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input_text: Optional[str] = None,
):
    """
    Enrich the current span (e.g., A2A root span) with GenAI and MLflow attributes.

    If there's no recording span in the current context, creates a new one named
    'gen_ai.agent.invoke' to ensure traces are captured.

    Args:
        context_id: A2A context_id (becomes gen_ai.conversation.id)
        task_id: A2A task_id
        user_id: User identifier
        input_text: User input message

    Yields:
        The span (either enriched existing or newly created)

    Note:
        TODO: The following could be handled by OTEL Collector transform instead:
        - mlflow.spanInputs/spanOutputs (currently set explicitly for reliability)
        - mlflow.spanType classification (could use span name pattern matching)
        - mlflow.user/source (could be derived from resource attributes)
    """
    current_span = trace.get_current_span()

    # Check if we have a recording span to enrich
    # get_current_span() returns INVALID_SPAN if none exists
    if current_span.is_recording():
        # Enrich the existing span
        _set_genai_mlflow_attributes(current_span, context_id, task_id, user_id, input_text)
        try:
            yield current_span
        except Exception as e:
            current_span.set_status(Status(StatusCode.ERROR, str(e)))
            current_span.record_exception(e)
            raise
    else:
        # No recording span - create one
        # This ensures our GenAI attributes are captured even if A2A doesn't trace
        logger.info("No current recording span - creating gen_ai.agent.invoke span")
        tracer = get_tracer()
        with tracer.start_as_current_span("gen_ai.agent.invoke") as new_span:
            _set_genai_mlflow_attributes(new_span, context_id, task_id, user_id, input_text)
            try:
                yield new_span
                new_span.set_status(Status(StatusCode.OK))
            except Exception as e:
                new_span.set_status(Status(StatusCode.ERROR, str(e)))
                new_span.record_exception(e)
                raise


def set_span_output(span, output: str):
    """
    Set output attributes on a span after work completes.

    Call this after getting a response to populate:
    - gen_ai.completion (standard GenAI attribute)
    - output.value (OpenInference attribute)
    - mlflow.spanOutputs (MLflow Response column)

    Args:
        span: The span to update
        output: The output/response text
    """
    if output:
        truncated = str(output)[:1000]
        span.set_attribute("gen_ai.completion", truncated)
        span.set_attribute("output.value", truncated)
        span.set_attribute("mlflow.spanOutputs", truncated)


def set_token_usage(span, input_tokens: int = 0, output_tokens: int = 0):
    """
    Set token usage attributes on a span.

    Args:
        span: The span to update
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Note:
        TODO: This could be handled by OTEL Collector transform copying
        gen_ai.usage.* to mlflow.span.chat_usage.* attributes.
    """
    if input_tokens:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("mlflow.span.chat_usage.input_tokens", input_tokens)
    if output_tokens:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("mlflow.span.chat_usage.output_tokens", output_tokens)


@contextmanager
def create_agent_span(
    name: str = "gen_ai.agent.invoke",
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input_text: Optional[str] = None,
    break_parent_chain: bool = False,
):
    """
    Create a NEW AGENT span with GenAI semantic conventions.

    Use `enrich_current_span` instead if you want to add attributes to
    the existing A2A span rather than creating a new child span.

    Args:
        name: Span name (use gen_ai.agent.* for MLflow AGENT type detection)
        context_id: A2A context_id (becomes gen_ai.conversation.id)
        task_id: A2A task_id
        user_id: User identifier
        input_text: User input message
        break_parent_chain: If True, breaks the trace chain (creates isolated root).
                           Default False to preserve distributed trace visibility.

    Yields:
        The span object (set output.value on it before exiting)
    """
    tracer = get_tracer()

    # Build attributes
    attributes = {}

    # GenAI semantic conventions (for OTEL Collector transforms)
    if context_id:
        attributes["gen_ai.conversation.id"] = context_id
    if input_text:
        attributes["gen_ai.prompt"] = input_text[:1000]
        attributes["input.value"] = input_text[:1000]
    attributes["gen_ai.agent.name"] = "weather-assistant"
    attributes["gen_ai.system"] = "langchain"

    # OpenInference span kind - marks this as an AGENT span
    if OPENINFERENCE_AVAILABLE:
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT.value

    # Custom attributes for debugging
    if task_id:
        attributes["a2a.task_id"] = task_id
    if user_id:
        attributes["user.id"] = user_id

    # Optional: break the parent chain for isolated traces
    detach_token = None
    if break_parent_chain:
        empty_ctx = context.Context()
        detach_token = context.attach(empty_ctx)

    # Start the span - becomes child of current context (A2A span) by default
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            if detach_token:
                context.detach(detach_token)


@contextmanager
def trace_context_from_headers(headers: Dict[str, str]):
    """
    Activate trace context from HTTP headers.

    Use this to connect to incoming distributed trace.
    """
    ctx = extract(headers)
    token = context.attach(ctx)
    try:
        yield ctx
    finally:
        context.detach(token)


def create_tracing_middleware():
    """
    Create Starlette middleware that wraps all requests in a root tracing span.

    This middleware:
    1. Creates a root span BEFORE A2A handlers run
    2. Sets MLflow/GenAI attributes on the root span
    3. Parses A2A JSON-RPC request to extract user input
    4. Captures response to set output attributes

    Usage in agent.py:
        from weather_service.observability import create_tracing_middleware
        app = server.build()
        app.add_middleware(BaseHTTPMiddleware, dispatch=create_tracing_middleware())
    """
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    import io

    async def tracing_middleware(request: Request, call_next):
        # Skip non-API paths (health checks, agent card, etc.)
        if request.url.path in ["/health", "/ready", "/.well-known/agent-card.json"]:
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
            logger.debug(f"Could not parse request body: {e}")

        # Extract incoming W3C Trace Context from request headers.
        # If the request carries a traceparent (e.g., from Envoy/MCP gateway or
        # an upstream orchestrator), the root span becomes a child of that trace.
        # This is what connects the agent's spans to the MCP gateway's spans.
        incoming_ctx = extract(dict(request.headers))
        detach_token = context.attach(incoming_ctx)

        try:
            # Create root span with correct GenAI naming convention
            # Per https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
            # Span name: "invoke_agent {gen_ai.agent.name}" when name is available
            span_name = f"invoke_agent {AGENT_NAME}"

            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,  # In-process agent (not remote service)
            ) as span:
                # Store span in ContextVar so agent code can access it
                # This is needed because trace.get_current_span() in execute()
                # returns the innermost span (A2A span), not our root span
                span_token = _root_span_var.set(span)

                # === GenAI Semantic Conventions (Required) ===
                # Per https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
                span.set_attribute("gen_ai.provider.name", AGENT_FRAMEWORK)
                span.set_attribute("gen_ai.agent.name", AGENT_NAME)
                span.set_attribute("gen_ai.agent.version", AGENT_VERSION)

                # Set input attributes (Prompt column in MLflow)
                if user_input:
                    span.set_attribute("gen_ai.prompt", user_input[:1000])
                    span.set_attribute("input.value", user_input[:1000])
                    span.set_attribute("mlflow.spanInputs", user_input[:1000])

                # Session tracking - use context_id or message_id as fallback
                session_id = context_id or message_id

                if session_id:
                    span.set_attribute("gen_ai.conversation.id", session_id)
                    span.set_attribute("mlflow.trace.session", session_id)
                    span.set_attribute("session.id", session_id)

                # MLflow trace metadata (appears in trace list columns)
                span.set_attribute("mlflow.spanType", "AGENT")
                span.set_attribute("mlflow.traceName", AGENT_NAME)
                span.set_attribute("mlflow.runName", f"{AGENT_NAME}-invoke")
                span.set_attribute("mlflow.source", "weather-service")
                span.set_attribute("mlflow.version", AGENT_VERSION)

                # User tracking - extract from auth header if available
                auth_header = request.headers.get("authorization", "")
                if auth_header:
                    # For Bearer tokens, we could decode JWT to get user
                    # For now, just indicate authenticated request
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

                    # Try to capture response for output attributes
                    # Note: This only works for non-streaming responses
                    if isinstance(response, Response) and not isinstance(
                        response, StreamingResponse
                    ):
                        # Read response body - we MUST recreate response after this
                        response_body = b""
                        async for chunk in response.body_iterator:
                            response_body += chunk

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
                                            span.set_attribute(
                                                "gen_ai.completion", output_text[:1000]
                                            )
                                            span.set_attribute(
                                                "output.value", output_text[:1000]
                                            )
                                            span.set_attribute(
                                                "mlflow.spanOutputs", output_text[:1000]
                                            )
                        except Exception as e:
                            logger.debug(f"Could not parse response body: {e}")

                        # Always recreate response since we consumed the iterator
                        span.set_status(Status(StatusCode.OK))
                        return Response(
                            content=response_body,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type,
                        )

                    # For streaming responses, just return as-is
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
