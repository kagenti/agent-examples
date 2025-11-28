"""
OpenTelemetry and OpenInference observability setup for Weather Agent.

This module provides:
- Auto-instrumentation with OpenInference semantic conventions
- OTEL baggage propagation for context tracking (user_id, request_id, etc.)
- Phoenix project routing via resource attributes
- GenAI semantic conventions compliance

Usage:
    from weather_service.observability import setup_observability, set_baggage_context

    # At agent startup
    setup_observability()

    # In request handler
    set_baggage_context({
        "user_id": "alice",
        "request_id": "req-123",
    })
"""

import logging
import os
from typing import Dict, Any, Optional
from opentelemetry import trace, baggage, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.langchain import LangChainInstrumentor

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
            "http://otel-collector.kagenti-system.svc.cluster.local:4318"
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

    logger.info("✅ OpenTelemetry observability configured successfully")
    logger.info("✅ LangChain auto-instrumented with OpenInference")
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
