"""Weather Service - OpenTelemetry Observability Setup"""

import logging

from weather_service.observability import setup_observability

logging.basicConfig(level=logging.DEBUG)

# Initialize observability before importing agent
setup_observability()
