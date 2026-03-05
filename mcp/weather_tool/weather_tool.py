"Weather MCP tool example"

import json
import logging
import os
import sys

import httpx
from fastmcp import FastMCP
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastmcp.server.dependencies import get_http_headers

from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.trace import Status, StatusCode

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


_tracer: trace.Tracer | None = None


def setup_tracing() -> None:
    """Initialize OpenTelemetry tracing with W3C trace context propagation."""
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.kagenti-system.svc.cluster.local:8335"
    )
    service_name = os.getenv("OTEL_SERVICE_NAME", "weather-mcp-tool")

    if not otlp_endpoint.endswith("/v1/traces"):
        otlp_endpoint = otlp_endpoint.rstrip("/") + "/v1/traces"

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    )
    trace.set_tracer_provider(provider)

    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    logger.info(f"Tracing initialized: service={service_name} otlp={otlp_endpoint}")


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("weather-mcp-tool")
    return _tracer


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def get_weather(city: str) -> str:
    """Get weather info for a city"""
    # Extract W3C traceparent from the incoming MCP HTTP request so this tool's
    # span becomes a child of the MCP gateway span (which is itself a child of
    # the agent span), giving a full agent → gateway → tool trace chain.
    headers = get_http_headers()
    incoming_ctx = extract(headers)
    token = otel_context.attach(incoming_ctx)

    try:
        with get_tracer().start_as_current_span("get_weather") as span:
            span.set_attribute("mcp.tool.name", "get_weather")
            span.set_attribute("mcp.tool.input.city", city)

            logger.debug(f"Getting weather info for city '{city}'.")

            base_url = "https://geocoding-api.open-meteo.com/v1/search"
            async with httpx.AsyncClient() as client:
                response = await client.get(base_url, params={"name": city, "count": 1}, timeout=10)
            data = response.json()

            if not data or "results" not in data:
                result = f"City {city} not found"
                span.set_attribute("mcp.tool.output", result)
                span.set_status(Status(StatusCode.OK))
                return result

            latitude = data["results"][0]["latitude"]
            longitude = data["results"][0]["longitude"]
            span.set_attribute("geo.latitude", latitude)
            span.set_attribute("geo.longitude", longitude)

            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": latitude,
                "longitude": longitude,
                "temperature_unit": "fahrenheit",
                "current_weather": True,
            }
            async with httpx.AsyncClient() as client:
                weather_response = await client.get(weather_url, params=weather_params, timeout=10)
            weather_data = weather_response.json()

            result = json.dumps(weather_data["current_weather"])
            span.set_attribute("mcp.tool.output", result[:500])
            span.set_status(Status(StatusCode.OK))
            return result

    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        otel_context.detach(token)



# host can be specified with HOST env variable
# transport can be specified with MCP_TRANSPORT env variable (defaults to streamable-http)
def run_server():
    "Run the MCP server"
    setup_tracing()
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
