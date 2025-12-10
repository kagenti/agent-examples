"""
Weather Service - OpenTelemetry Observability Setup

This module initializes OpenTelemetry tracing with OpenInference instrumentation
for automatic LLM observability in Phoenix.
"""

from weather_service.observability import setup_observability

# Set up OpenTelemetry tracing with OpenInference
# This must run before importing agent code to ensure instrumentation is active
setup_observability()
