"Weather MCP tool example"

import asyncio
import functools
import json
import logging
import os
import sys

import requests
import uvicorn
from fastmcp import FastMCP
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from requests.adapters import HTTPAdapter
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from urllib3.util.retry import Retry

mcp = FastMCP("Weather")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
)

# Configurable timeouts and retries for external API resilience
_REQUEST_TIMEOUT = int(os.getenv("WEATHER_REQUEST_TIMEOUT", "30"))
_MAX_RETRIES = int(os.getenv("WEATHER_MAX_RETRIES", "3"))
_BACKOFF_FACTOR = float(os.getenv("WEATHER_BACKOFF_FACTOR", "1.0"))


def _build_resilient_session() -> requests.Session:
    """Create an HTTP session with automatic retry and exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Module-level session — reused across tool calls for connection pooling
_session = _build_resilient_session()


def setup_tracing() -> None:
    """Initialize OpenTelemetry tracing with W3C trace context propagation."""
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector.kagenti-system.svc.cluster.local:8335"
    )
    service_name = os.getenv("OTEL_SERVICE_NAME", "weather-mcp-tool")

    if not otlp_endpoint.endswith("/v1/traces"):
        otlp_endpoint = otlp_endpoint.rstrip("/") + "/v1/traces"

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)

    set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
    )

    logger.info("Tracing initialized: service=%s otlp=%s", service_name, otlp_endpoint)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def get_weather(city: str) -> str:
    """Get weather info for a city"""
    # Enrich FastMCP's span with gen_ai attributes rather than creating a child span.
    span = trace.get_current_span()
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", "get_weather")
    span.set_attribute("gen_ai.tool.call.arguments", json.dumps({"city": city}))

    logger.debug(f"Getting weather info for city '{city}'.")

    loop = asyncio.get_running_loop()

    try:
        base_url = "https://geocoding-api.open-meteo.com/v1/search"
        response = await loop.run_in_executor(
            None,
            functools.partial(_session.get, base_url, params={"name": city, "count": 1}, timeout=_REQUEST_TIMEOUT),
        )
        response.raise_for_status()
        data = response.json()

        if not data or "results" not in data:
            result = f"City {city} not found"
            span.set_attribute("gen_ai.tool.call.result", result)
            span.set_status(Status(StatusCode.OK))
            return result

        latitude = data["results"][0]["latitude"]
        longitude = data["results"][0]["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature_unit": "fahrenheit",
            "current_weather": True,
        }
        weather_response = await loop.run_in_executor(
            None,
            functools.partial(_session.get, weather_url, params=weather_params, timeout=_REQUEST_TIMEOUT),
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        result = json.dumps(weather_data["current_weather"])
        span.set_attribute("gen_ai.tool.call.result", result)
        span.set_status(Status(StatusCode.OK))
        return result

    except requests.RequestException as e:
        logger.warning("Weather API error for '%s': %s", city, e)
        span.set_attribute("error.type", type(e).__name__)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        return f"Weather service temporarily unavailable for {city}"
    except Exception as e:
        span.set_attribute("error.type", type(e).__name__)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise


async def _trace_propagation_middleware(request, call_next):
    """Extract W3C traceparent from HTTP headers before FastMCP creates its span.

    FastMCP creates its own root span per request. Without this middleware,
    that span has no parent, producing a disconnected trace in Phoenix/Jaeger.
    Attaching the incoming context here makes FastMCP's span a child of the
    caller's span automatically, since OTEL picks up the ambient context.
    """
    incoming_ctx = extract(dict(request.headers))
    token = otel_context.attach(incoming_ctx)
    try:
        return await call_next(request)
    finally:
        otel_context.detach(token)


# Environment variables: host can be specified with HOST, port with PORT
def run_server():
    "Run the MCP server"
    setup_tracing()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    app = mcp.http_app(middleware=[Middleware(BaseHTTPMiddleware, dispatch=_trace_propagation_middleware)])
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
