"""Weather Service - Minimal OTEL setup for Approach A (AuthBridge root span).

The agent only needs:
1. TracerProvider + OTLP exporter (standard OTEL boilerplate)
2. Auto-instrumentation (LangChain + OpenAI)
3. W3C Trace Context propagation (default in OTEL SDK)

The AuthBridge ext_proc creates the root span with all MLflow/OpenInference/GenAI
attributes. Agent auto-instrumented spans become children via traceparent header.
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

logger = logging.getLogger(__name__)

def setup_tracing():
    """Initialize OTEL tracing with auto-instrumentation. Call once at startup."""
    service_name = os.getenv("OTEL_SERVICE_NAME", "weather-service")

    resource = Resource.create(attributes={
        SERVICE_NAME: service_name,
        SERVICE_VERSION: "1.0.0",
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    # W3C Trace Context propagation - ensures agent spans inherit
    # the trace context from AuthBridge's traceparent header
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    # Auto-instrument LangChain
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        logger.info("LangChain auto-instrumented")
    except ImportError:
        logger.warning("openinference-instrumentation-langchain not available")

    # Auto-instrument OpenAI (for GenAI token metrics)
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        logger.info("OpenAI auto-instrumented")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-openai not available")

    logger.info(f"OTEL tracing initialized: service={service_name}")

setup_tracing()
