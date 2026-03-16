"""Weather Service - OpenTelemetry Observability Setup"""

from weather_service.observability import setup_observability

# Initialize observability before importing agent
setup_observability()
