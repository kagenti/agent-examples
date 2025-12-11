"""Restaurant Reservation MCP Server.

This MCP server provides tools for searching restaurants, checking availability,
and managing reservations through a provider abstraction layer.
"""

import os
import sys
import logging
import json
from typing import Optional
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier

from providers import MockProvider, ReservationProvider

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize provider
# Future: Could be configurable via env var to support different providers
provider: ReservationProvider = MockProvider()
logger.info("Initialized MockProvider for reservations")

# Setup JWT authentication if configured
verifier = None
JWKS_URI = os.getenv("JWKS_URI")
ISSUER = os.getenv("ISSUER")
if JWKS_URI:
    # Note: In production, CLIENT_ID should be read from SVID or config
    CLIENT_ID = os.getenv("CLIENT_ID", "reservation-tool")
    verifier = JWTVerifier(
        jwks_uri=JWKS_URI,
        issuer=ISSUER,
        audience=CLIENT_ID
    )
    logger.info("JWT authentication enabled")

# Create FastMCP app
mcp = FastMCP("Restaurant Reservations", auth=verifier)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def search_restaurants(
    city: str,
    cuisine: Optional[str] = None,
    date_time: Optional[str] = None,
    party_size: Optional[int] = None,
    price_tier: Optional[int] = None,
    distance_km: Optional[float] = None,
) -> str:
    """
    Search for restaurants matching the given criteria.

    Args:
        city: City name to search in (e.g., "Boston", "New York")
        cuisine: Optional cuisine type filter (e.g., "Italian", "Japanese", "Mexican")
        date_time: Optional ISO 8601 datetime for availability-aware search
        party_size: Optional number of guests for filtering
        price_tier: Optional price tier 1-4 (1=$, 2=$$, 3=$$$, 4=$$$$)
        distance_km: Optional maximum distance from city center in kilometers

    Returns:
        JSON string containing list of matching restaurants
    """
    logger.info(f"search_restaurants called: city={city}, cuisine={cuisine}, price_tier={price_tier}")

    try:
        restaurants = provider.search_restaurants(
            city=city,
            cuisine=cuisine,
            date_time=date_time,
            party_size=party_size,
            price_tier=price_tier,
            distance_km=distance_km,
        )

        # Convert to dict for JSON serialization
        results = [r.model_dump() for r in restaurants]
        logger.debug(f"Returning {len(results)} restaurants")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.exception(f"Error in search_restaurants: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def check_availability(
    restaurant_id: str,
    date_time: str,
    party_size: int,
) -> str:
    """
    Check availability for a specific restaurant on a given date.

    Args:
        restaurant_id: Unique restaurant identifier (e.g., "rest_001")
        date_time: ISO 8601 datetime for the date to check (e.g., "2025-03-15T18:00:00")
        party_size: Number of guests in the party

    Returns:
        JSON string containing list of available time slots
    """
    logger.info(f"check_availability called: restaurant={restaurant_id}, date_time={date_time}, party_size={party_size}")

    try:
        slots = provider.check_availability(
            restaurant_id=restaurant_id,
            date_time=date_time,
            party_size=party_size,
        )

        # Convert to dict for JSON serialization
        results = [s.model_dump() for s in slots]
        logger.debug(f"Returning {len(results)} time slots")
        return json.dumps(results, indent=2)

    except ValueError as e:
        logger.warning(f"Validation error in check_availability: {e}")
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception(f"Error in check_availability: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
def place_reservation(
    restaurant_id: str,
    date_time: str,
    party_size: int,
    name: str,
    phone: str,
    email: str,
    notes: Optional[str] = None,
) -> str:
    """
    Place a reservation at a restaurant.

    This operation is idempotent - submitting the same reservation details multiple times
    will return the same confirmation without creating duplicates.

    Args:
        restaurant_id: Unique restaurant identifier (e.g., "rest_001")
        date_time: ISO 8601 datetime for the reservation (e.g., "2025-03-15T19:00:00")
        party_size: Number of guests
        name: Guest's full name
        phone: Guest's contact phone number (e.g., "+1-555-123-4567")
        email: Guest's email address
        notes: Optional special requests or dietary restrictions

    Returns:
        JSON string containing reservation confirmation details
    """
    logger.info(f"place_reservation called: restaurant={restaurant_id}, name={name}, party_size={party_size}")

    try:
        reservation = provider.place_reservation(
            restaurant_id=restaurant_id,
            date_time=date_time,
            party_size=party_size,
            name=name,
            phone=phone,
            email=email,
            notes=notes,
        )

        result = reservation.model_dump()
        logger.info(f"Reservation placed successfully: {reservation.confirmation_code}")
        return json.dumps(result, indent=2)

    except ValueError as e:
        logger.warning(f"Validation error in place_reservation: {e}")
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception(f"Error in place_reservation: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True})
def cancel_reservation(
    reservation_id: str,
    reason: Optional[str] = None,
) -> str:
    """
    Cancel an existing reservation.

    This is a destructive operation that removes the reservation from the system.
    However, it is idempotent - cancelling the same reservation multiple times
    will return an appropriate error message.

    Args:
        reservation_id: Unique reservation identifier (e.g., "reservation_abc123")
        reason: Optional reason for cancellation

    Returns:
        JSON string containing cancellation confirmation
    """
    logger.info(f"cancel_reservation called: reservation_id={reservation_id}")

    try:
        receipt = provider.cancel_reservation(
            reservation_id=reservation_id,
            reason=reason,
        )

        result = receipt.model_dump()
        logger.info(f"Reservation cancelled successfully: {reservation_id}")
        return json.dumps(result, indent=2)

    except ValueError as e:
        logger.warning(f"Validation error in cancel_reservation: {e}")
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception(f"Error in cancel_reservation: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def list_reservations(
    user_id: str,
) -> str:
    """
    List all reservations for a user.

    Args:
        user_id: User identifier - can be email address or phone number

    Returns:
        JSON string containing list of user's reservations
    """
    logger.info(f"list_reservations called: user_id={user_id}")

    try:
        reservations = provider.list_reservations(user_id=user_id)

        # Convert to dict for JSON serialization
        results = [r.model_dump() for r in reservations]
        logger.debug(f"Returning {len(results)} reservations for user {user_id}")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.exception(f"Error in list_reservations: {e}")
        return json.dumps({"error": str(e)})


def run_server():
    """Run the MCP server with configured transport."""
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Starting Restaurant Reservation MCP Server on {host}:{port} with transport={transport}")
    logger.info(f"Registered tools: search_restaurants, check_availability, place_reservation, cancel_reservation, list_reservations")

    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
