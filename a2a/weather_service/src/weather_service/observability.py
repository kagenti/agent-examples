"""
OpenTelemetry and OpenInference observability setup for Weather Agent.

This module provides:
- Auto-instrumentation with OpenInference semantic conventions
- OpenInference context managers for session/user tracking
- Phoenix project routing via resource attributes
- GenAI semantic conventions compliance

Key Features:
- `using_attributes`: Add session_id, user_id, metadata to all spans in scope
- `create_agent_span`: Create a root AGENT span for the conversation
- Auto-instrumentation of LangChain/LangGraph via OpenInference

Usage:
    from weather_service.observability import (
        setup_observability,
        create_agent_span,
    )
    from openinference.instrumentation import using_attributes

    # At agent startup
    setup_observability()

    # In request handler - wrap execution with context
    with using_attributes(
        session_id="context-123",
        user_id="alice",
        metadata={"task_id": "task-456"},
    ):
        with create_agent_span("agent_task", task_id="task-456") as span:
            # All LangChain spans inside will have session.id and user.id
            result = await graph.astream(input)
"""

import logging
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager
from opentelemetry import trace, baggage, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import set_global_textmap, extract, inject
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

logger = logging.getLogger(__name__)


def _get_otlp_exporter(endpoint: str, protocol: str):
    """
    Get the appropriate OTLP exporter based on protocol.

    Args:
        endpoint: OTLP endpoint URL
        protocol: Protocol to use ('grpc' or 'http/protobuf')

    Returns:
        Configured OTLP span exporter
    """
    if protocol.lower() == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcExporter,
        )
        # For gRPC, endpoint should not have http:// prefix
        grpc_endpoint = endpoint.replace("http://", "").replace("https://", "")
        return GrpcExporter(endpoint=grpc_endpoint, insecure=True)
    else:
        # Default to HTTP/protobuf
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpExporter,
        )
        # Ensure endpoint has /v1/traces path for HTTP
        if not endpoint.endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"
        return HttpExporter(endpoint=endpoint)


class ObservabilityConfig:
    """
    Configuration for observability setup.

    Reads from environment variables with sensible defaults.
    """

    def __init__(self):
        # Service identification
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "weather-service")
        self.namespace = os.getenv("K8S_NAMESPACE_NAME", "team1")
        self.deployment_env = os.getenv("DEPLOYMENT_ENVIRONMENT", "kind-local")

        # Phoenix project routing
        self.phoenix_project = os.getenv(
            "PHOENIX_PROJECT_NAME",
            f"{self.namespace}-agents"
        )

        # OTLP endpoint and protocol
        self.otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://otel-collector.kagenti-system.svc.cluster.local:8335"
        )
        self.otlp_protocol = os.getenv(
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            "http/protobuf"  # Default to HTTP for wider compatibility
        )

        # Additional resource attributes
        self.extra_resource_attrs = self._parse_resource_attrs()

    def _parse_resource_attrs(self) -> Dict[str, str]:
        """
        Parse OTEL_RESOURCE_ATTRIBUTES environment variable.

        Format: key1=value1,key2=value2
        """
        attrs = {}
        resource_attrs_str = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")

        if resource_attrs_str:
            for pair in resource_attrs_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    attrs[key.strip()] = value.strip()

        return attrs

    def get_resource_attributes(self) -> Dict[str, str]:
        """
        Get complete set of resource attributes for OTEL tracer.

        Returns:
            Dict with service.name, namespace, Phoenix project, etc.
        """
        attrs = {
            # Service identification
            "service.name": self.service_name,
            "service.namespace": self.namespace,

            # Kubernetes metadata
            "k8s.namespace.name": self.namespace,

            # Phoenix project routing
            "phoenix.project.name": self.phoenix_project,

            # Deployment environment
            "deployment.environment": self.deployment_env,
        }

        # Merge extra attributes from env var
        attrs.update(self.extra_resource_attrs)

        return attrs


def setup_observability(config: Optional[ObservabilityConfig] = None) -> None:
    """
    Set up OpenTelemetry tracing with OpenInference instrumentation.

    This function:
    1. Creates OTEL tracer provider with proper resource attributes
    2. Configures OTLP gRPC exporter
    3. Instruments LangChain with OpenInference

    Args:
        config: Optional ObservabilityConfig. If not provided, creates default.

    Example:
        >>> setup_observability()
        >>> # All LangChain operations now automatically traced to Phoenix
    """
    if config is None:
        config = ObservabilityConfig()

    logger.info("=" * 70)
    logger.info("Setting up OpenTelemetry observability")
    logger.info("-" * 70)
    logger.info(f"Service Name:      {config.service_name}")
    logger.info(f"Namespace:         {config.namespace}")
    logger.info(f"Phoenix Project:   {config.phoenix_project}")
    logger.info(f"OTLP Endpoint:     {config.otlp_endpoint}")
    logger.info(f"OTLP Protocol:     {config.otlp_protocol}")
    logger.info(f"Deployment Env:    {config.deployment_env}")
    logger.info("=" * 70)

    # Create resource with all attributes
    resource_attrs = config.get_resource_attributes()
    resource = Resource(attributes=resource_attrs)

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Create OTLP exporter based on configured protocol
    otlp_exporter = _get_otlp_exporter(config.otlp_endpoint, config.otlp_protocol)

    # Add batch span processor for efficiency
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument LangChain with OpenInference
    # This adds openinference.span.kind attributes to LangChain spans
    LangChainInstrumentor().instrument()

    # Configure W3C Trace Context and Baggage propagators for distributed tracing
    # This enables trace context propagation across HTTP calls:
    # - TraceContextTextMapPropagator: handles 'traceparent' header (trace_id + parent_span_id)
    # - W3CBaggagePropagator: handles 'baggage' header (user metadata like user_id, tenant_id)
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    logger.info("✅ OpenTelemetry observability configured successfully")
    logger.info("✅ LangChain auto-instrumented with OpenInference")
    logger.info("✅ W3C Trace Context and Baggage propagators configured")
    logger.info("✅ Traces will route to Phoenix project: %s", config.phoenix_project)


def set_baggage_context(context_data: Dict[str, Any]) -> context.Context:
    """
    Set OTEL baggage for context propagation across services.

    Baggage allows passing context (user_id, request_id, etc.) across
    all spans in a trace, even across service boundaries.

    Args:
        context_data: Dict with keys like:
            - user_id: User identifier
            - request_id: Request identifier
            - conversation_id: Conversation identifier
            - tenant_id: Tenant identifier (for multi-tenancy)
            - Any other context you want to track

    Returns:
        Updated context with baggage

    Example:
        >>> ctx = set_baggage_context({
        ...     "user_id": "alice",
        ...     "request_id": "req-123",
        ...     "conversation_id": "conv-456"
        ... })
        >>> # All subsequent spans will have these attributes
    """
    # Start with the current context (not baggage.get_current which doesn't exist)
    ctx = context.get_current()

    for key, value in context_data.items():
        if value is not None:  # Only set non-None values
            ctx = baggage.set_baggage(key, str(value), context=ctx)
            logger.debug(f"Set baggage: {key}={value}")

    # Attach the updated context with baggage
    context.attach(ctx)

    return ctx


def get_baggage_context() -> Dict[str, str]:
    """
    Get current OTEL baggage context.

    Returns:
        Dict of baggage key-value pairs

    Example:
        >>> baggage_data = get_baggage_context()
        >>> print(baggage_data)
        {'user_id': 'alice', 'request_id': 'req-123'}
    """
    # Get current context and extract all baggage from it
    ctx = context.get_current()
    return dict(baggage.get_all(ctx))


def extract_baggage_from_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Extract baggage context from HTTP headers.

    Common headers to extract:
    - user-id, x-user-id
    - request-id, x-request-id
    - conversation-id, x-conversation-id
    - tenant-id, x-tenant-id

    Args:
        headers: Dict of HTTP headers (lowercase keys)

    Returns:
        Dict of extracted baggage context

    Example:
        >>> headers = {"user-id": "alice", "request-id": "req-123"}
        >>> context = extract_baggage_from_headers(headers)
        >>> set_baggage_context(context)
    """
    baggage_data = {}

    # Normalize headers to lowercase for case-insensitive matching
    headers_lower = {k.lower(): v for k, v in headers.items()}

    # Map header names to baggage keys
    header_mappings = {
        "user-id": "user_id",
        "x-user-id": "user_id",
        "request-id": "request_id",
        "x-request-id": "request_id",
        "conversation-id": "conversation_id",
        "x-conversation-id": "conversation_id",
        "tenant-id": "tenant_id",
        "x-tenant-id": "tenant_id",
        "trace-id": "trace_id",
        "x-trace-id": "trace_id",
    }

    for header_name, baggage_key in header_mappings.items():
        if header_name in headers_lower:
            baggage_data[baggage_key] = headers_lower[header_name]

    logger.debug(f"Extracted baggage from headers: {baggage_data}")
    return baggage_data


def log_trace_info() -> None:
    """
    Log current trace and baggage information for debugging.

    Useful for verifying trace context is properly set.
    """
    span = trace.get_current_span()
    span_context = span.get_span_context()

    if span_context.is_valid:
        logger.info("=" * 70)
        logger.info("Current Trace Context")
        logger.info("-" * 70)
        logger.info(f"Trace ID:  {format(span_context.trace_id, '032x')}")
        logger.info(f"Span ID:   {format(span_context.span_id, '016x')}")

        baggage_data = get_baggage_context()
        if baggage_data:
            logger.info("Baggage:")
            for key, value in baggage_data.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("Baggage: (empty)")
        logger.info("=" * 70)
    else:
        logger.warning("No active trace context")


# Global tracer for creating manual spans
# IMPORTANT: Use OpenInference-compatible tracer name so spans pass through
# the OTEL Collector filter which only allows "openinference.instrumentation.*"
_tracer: Optional[trace.Tracer] = None
TRACER_NAME = "openinference.instrumentation.agent"


def get_tracer() -> trace.Tracer:
    """Get the global tracer for creating manual spans.

    Uses OpenInference-compatible tracer name to ensure spans are routed
    to Phoenix by the OTEL Collector's filter/phoenix processor.
    """
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(TRACER_NAME)
    return _tracer


@contextmanager
def create_agent_span(
    name: str = "agent_task",
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input_text: Optional[str] = None,
):
    """
    Create a root AGENT span for the conversation with OpenInference attributes.

    This span serves as the root for all LangChain/LangGraph auto-instrumented
    spans, providing a clear entry point for each agent interaction.

    Args:
        name: Span name (default: "agent_task")
        task_id: A2A task ID for filtering conversations
        context_id: A2A context ID (conversation session)
        user_id: User identifier
        input_text: The user's input message

    Yields:
        The created span

    Example:
        with create_agent_span(
            task_id="task-123",
            context_id="ctx-456",
            user_id="alice",
            input_text="What's the weather in Tokyo?"
        ) as span:
            result = await graph.astream(input)
            span.set_attribute("output.value", str(result))
    """
    tracer = get_tracer()

    # Build attributes following OpenInference semantic conventions
    attributes = {
        # OpenInference span kind for AI observability
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
    }

    # Add A2A task/context IDs as custom attributes for filtering
    if task_id:
        attributes["a2a.task_id"] = task_id
    if context_id:
        attributes["a2a.context_id"] = context_id
    if user_id:
        attributes["user.id"] = user_id
    if input_text:
        attributes[SpanAttributes.INPUT_VALUE] = input_text

    with tracer.start_as_current_span(name, attributes=attributes) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def extract_trace_context(headers: Dict[str, str]) -> context.Context:
    """
    Extract trace context from HTTP headers.

    This extracts both W3C Trace Context (traceparent) and Baggage headers,
    enabling proper parent-child span relationships across service boundaries.

    Args:
        headers: HTTP headers dict (can be any mapping type)

    Returns:
        Context with extracted trace information

    Example:
        >>> # In request handler
        >>> ctx = extract_trace_context(request.headers)
        >>> with tracer.start_as_current_span("handle_request", context=ctx):
        ...     # Spans here are children of the incoming trace
    """
    return extract(headers)


def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject current trace context into HTTP headers.

    This injects both W3C Trace Context (traceparent) and Baggage headers,
    enabling proper parent-child span relationships when calling other services.

    Args:
        headers: Dict to inject headers into (modified in place)

    Returns:
        The headers dict with trace context added

    Example:
        >>> # Before making HTTP call
        >>> headers = {"Content-Type": "application/json"}
        >>> inject_trace_context(headers)
        >>> response = await client.post(url, headers=headers)
    """
    inject(headers)
    return headers


@contextmanager
def trace_context_from_headers(headers: Dict[str, str]):
    """
    Context manager that activates trace context from HTTP headers.

    Use this to wrap request handling code so that all spans created
    within the context become children of the incoming trace.

    Args:
        headers: HTTP headers containing traceparent/baggage

    Yields:
        The extracted context

    Example:
        >>> async def handle_request(request):
        ...     with trace_context_from_headers(request.headers) as ctx:
        ...         # All spans here are connected to incoming trace
        ...         result = await process_message(request.message)
    """
    ctx = extract(headers)
    token = context.attach(ctx)
    try:
        yield ctx
    finally:
        context.detach(token)
