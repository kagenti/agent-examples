"""Shared configuration for agent-examples tests.

Adds agent and tool source directories to sys.path at import time
so that test modules can import from them.
"""

import sys
from pathlib import Path

root = Path(__file__).parent.parent

_paths = [
    str(root / "a2a" / "a2a_currency_converter"),
    str(root / "a2a" / "weather_service" / "src"),
    str(root / "a2a" / "simple_generalist" / "src"),
    str(root / "a2a" / "a2a_contact_extractor"),
    str(root / "mcp" / "flight_tool"),
    str(root / "mcp" / "reservation_tool"),
]

for p in _paths:
    if p not in sys.path:
        sys.path.insert(0, p)
