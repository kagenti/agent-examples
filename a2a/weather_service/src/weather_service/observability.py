"""
Optional OTEL auto-instrumentation for Weather Agent.

Controlled by OTEL_INSTRUMENT env var:
  - "none"           — no agent-side instrumentation (ext_proc sidecar only)
  - "openinference"  — LangChainInstrumentor (OpenInference conventions)
  - "openai"         — OpenAIInstrumentor (gen_ai.* conventions)

The ext_proc sidecar creates the root span and injects traceparent.
ASGI middleware here extracts that traceparent so auto-instrumented
spans become children of the ext_proc root span.
"""

import logging
import os

logger = logging.getLogger(__name__)

OTEL_INSTRUMENT = os.getenv("OTEL_INSTRUMENT", "none").lower()

# Stash the provider so wrap_asgi_app can use it
_tracer_provider = None


def setup_observability():
    """Initialize OTEL tracing and auto-instrumentation based on OTEL_INSTRUMENT env var."""
    global _tracer_provider

    if OTEL_INSTRUMENT == "none":
        logger.info("[OTEL] Instrumentation disabled (OTEL_INSTRUMENT=none)")
        return

    # Set up TracerProvider with OTLP exporter
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator

    service_name = os.getenv("OTEL_SERVICE_NAME", "weather-service")
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.kagenti-system.svc.cluster.local:8335",
    )
    if not otlp_endpoint.endswith("/v1/traces"):
        otlp_endpoint = otlp_endpoint.rstrip("/") + "/v1/traces"

    resource = Resource({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: "1.0.0",
    })
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(_tracer_provider)

    # W3C Trace Context propagation so auto-instrumented spans
    # become children of the ext_proc root span (via traceparent header)
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    logger.info(f"[OTEL] TracerProvider initialized: service={service_name} endpoint={otlp_endpoint}")

    # Enable auto-instrumentation based on mode
    if OTEL_INSTRUMENT == "openinference":
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            LangChainInstrumentor().instrument()
            logger.info("[OTEL] LangChain instrumented with OpenInference")
        except ImportError:
            logger.warning("[OTEL] openinference-instrumentation-langchain not installed")

    elif OTEL_INSTRUMENT == "openai":
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
            logger.info("[OTEL] OpenAI instrumented with GenAI semantic conventions")
        except ImportError:
            logger.warning("[OTEL] opentelemetry-instrumentation-openai not installed")

    else:
        logger.warning(f"[OTEL] Unknown OTEL_INSTRUMENT value: {OTEL_INSTRUMENT}")


def wrap_asgi_app(app):
    """Wrap a Starlette/ASGI app with OpenTelemetry ASGI middleware.

    This extracts the traceparent header from incoming requests (injected
    by the ext_proc sidecar) and sets it as the active trace context.
    All auto-instrumented spans (LangChain, OpenAI) then become children
    of the ext_proc root span.

    Call this after app = server.build() and before uvicorn.run(app).
    """
    if OTEL_INSTRUMENT == "none" or _tracer_provider is None:
        return app

    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
        wrapped = OpenTelemetryMiddleware(app, tracer_provider=_tracer_provider)
        logger.info("[OTEL] ASGI middleware installed (traceparent extraction enabled)")
        return wrapped
    except ImportError:
        logger.warning("[OTEL] opentelemetry-instrumentation-asgi not installed, traceparent will not be extracted")
        return app
