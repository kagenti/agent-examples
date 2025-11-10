# Fast Flights MCP tool

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from datetime import date, datetime

from fastmcp import FastMCP
from fast_flights import (
    FlightData,
    Passengers,
    Result,
    get_flights,
    search_airport as ff_search_airport)


mcp = FastMCP("Flights")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout, format='%(levelname)s: %(message)s')


def _result_to_dict(r: Result) -> List[Dict[str, Any]]:
    flights = getattr(r, 'flights', [])
    if not flights:
        return [{
            "id": None,
            "airline": None,
            "price": "N/A",
            "price_value": None,
            "duration_minutes": None,
            "stops": None,
            "departure": None,
            "arrival": None,
        }]
    
    flight_results = []
    for flight in flights:
        flight_results.append({
            "id": getattr(flight, 'name', None),
            "airline": getattr(flight, 'name', None),
            "price_value": getattr(r, 'current_price', None),
            "duration_minutes": getattr(flight, 'duration', None),
            "stops": getattr(flight, 'stops', None),
            "departure": getattr(flight, 'departure', None),
            "arrival": getattr(flight, 'arrival', None),
            "is_best": getattr(flight, 'is_best', False),
            "delay": getattr(flight, 'delay', None),
        })
    
    return flight_results

def _parse_iso_date(d: str) -> Optional[date]:
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None


def _date_in_past(d: date) -> bool:
    try:
        return d < date.today()
    except Exception:
        return False


def _coerce_int(val: Any, name: str, default: int) -> tuple[int, Optional[str]]:
    """Coerce a string or int to a non-negative int.

    This simpler variant accepts:
    - ints (returned as-is)
    - strings that parse directly with int(), e.g. "1" or " 2 "

    Returns (int, None) on success or (default, error_message) on failure.
    """
    if isinstance(val, int):
        i = val
    elif isinstance(val, str):
        s = val.strip()
        try:
            i = int(s)
        except Exception:
            return default, f"Invalid integer value for '{name}': {val!r}"
    else:
        return default, f"Invalid type for '{name}': expected int or str, got {type(val).__name__}"

    if i < 0:
        return default, f"'{name}' must be >= 0"

    return i, None

@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def search_airports(query: str, limit: int = 10) -> str:
    """Search for airports by name or code.

    This wrapper calls the underlying fast-flights airport search and returns a
    simple JSON array of results. Each element will be one of:
      - the enum `.value` (common for IATA codes), or
      - the enum `.name`, or
      - the string representation of the item.

    Parameters:
    - query: search string (city name, airport name, or IATA code)
    - limit: max number of results to return
    """
    try:
        results = ff_search_airport(query)
    except Exception as e:
        return json.dumps({"error": str(e)})

    airports = []
    for a in (results or [])[:limit]:
        airports.append(getattr(a, "value"))

    return json.dumps(airports)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def search_flights(
    from_airport: str,
    to_airport: str,
    departure_date: str,
    return_date: Optional[str] = None,
    cabin: Optional[str] = None,
    adults: Optional[int] = 1,
    children: Optional[int] = 0,
    infants_in_seat: Optional[int] = 0,
    infants_on_lap: Optional[int] = 0,
    airlines: Optional[str] = None,
    max_stops: Optional[int] = None,
) -> str:
    """Search flights between two cities.

    Required parameters:
    - from_airport, to_airport: IATA codes or city names
    - departure_date: YYYY-MM-DD

    Optional parameters:
    - return_date: YYYY-MM-DD (for round-trip flights)
    - cabin: economy|premium-economy|business|first (defaults to economy)
    - adults: an integer number of adult passengers (defaults to 1)
    - children: an integer number of child passengers (defaults to 0)
    - infants_in_seat: an integer number of infants with seats (defaults to 0)
    - infants_on_lap: an integer number of infants on lap (defaults to 0)
    - airlines: comma-separated IATA codes or alliances (SKYTEAM, STAR_ALLIANCE, ONEWORLD)
    - max_stops: an integer, maximum number of stops (defaults to no limit)
    """
    # Coerce passenger counts to integers, necessary due to a fastMCP issue
    adults, err = _coerce_int(adults, "adults", 1)
    if err:
        return json.dumps({"error": err, "adults": adults})
    children, err = _coerce_int(children, "children", 0)
    if err:
        return json.dumps({"error": err, "children": children})
    infants_in_seat, err = _coerce_int(infants_in_seat, "infants_in_seat", 0)
    if err:
        return json.dumps({"error": err, "infants_in_seat": infants_in_seat})
    infants_on_lap, err = _coerce_int(infants_on_lap, "infants_on_lap", 0)
    if err:
        return json.dumps({"error": err, "infants_on_lap": infants_on_lap})

    # Validate dates are not in the past
    dep_date_obj = _parse_iso_date(departure_date)
    if dep_date_obj is None:
        return json.dumps({"error": "Invalid departure_date format. Use YYYY-MM-DD", "departure_date": departure_date})
    if _date_in_past(dep_date_obj):
        return json.dumps({"error": "departure_date cannot be in the past", "departure_date": departure_date})

    ret_date_obj = None
    if return_date:
        ret_date_obj = _parse_iso_date(return_date)
        if ret_date_obj is None:
            return json.dumps({"error": "Invalid return_date format. Use YYYY-MM-DD", "return_date": return_date})
        if _date_in_past(ret_date_obj):
            return json.dumps({"error": "return_date cannot be in the past", "return_date": return_date})
        # Ensure return date is not before departure
        if ret_date_obj < dep_date_obj:
            return json.dumps({"error": "return_date cannot be before departure_date", "departure_date": departure_date, "return_date": return_date})
        
    flight_data_kwargs = {
        "date": departure_date,
        "from_airport": from_airport,
        "to_airport": to_airport,
    }
    
    if airlines:
        airline_list = [airline.strip().upper() for airline in airlines.split(",")]
        flight_data_kwargs["airlines"] = airline_list
    
    if max_stops is not None:
        flight_data_kwargs["max_stops"] = max_stops
    
    flight_data_list = [FlightData(**flight_data_kwargs)]
    
    # Add return flight for round-trip
    if return_date:
        return_flight_kwargs = flight_data_kwargs.copy()
        return_flight_kwargs.update({
            "date": return_date,
            "from_airport": to_airport,
            "to_airport": from_airport,
        })
        flight_data_list.append(FlightData(**return_flight_kwargs))
        trip_type = "round-trip"
    else:
        trip_type = "one-way"
    
    seat_mapping = {
        "economy": "economy",
        "premium_economy": "premium_economy", 
        "business": "business",
        "first": "first"
    }
    seat_type = seat_mapping.get(cabin, "economy")
    
    total_passengers = adults + children + infants_in_seat + infants_on_lap
    if total_passengers > 9:
        return json.dumps({
            "error": "Total passengers cannot exceed 9",
            "request": {
                "from_airport": from_airport,
                "to_airport": to_airport,
                "departure_date": departure_date,
                "return_date": return_date,
                "cabin": cabin,
                "adults": adults,
                "children": children,
                "infants_in_seat": infants_in_seat,
                "infants_on_lap": infants_on_lap,
                "airlines": airlines,
                "max_stops": max_stops,
            }
        })
    
    if infants_on_lap > adults:
        return json.dumps({
            "error": "Must have at least one adult per infant on lap",
            "request": {
                "from_airport": from_airport,
                "to_airport": to_airport,
                "departure_date": departure_date,
                "return_date": return_date,
                "cabin": cabin,
                "adults": adults,
                "children": children,
                "infants_in_seat": infants_in_seat,
                "infants_on_lap": infants_on_lap,
                "airlines": airlines,
                "max_stops": max_stops,
            }
        })
    
    passengers = Passengers(
        adults=adults,
        children=children,
        infants_in_seat=infants_in_seat,
        infants_on_lap=infants_on_lap
    )
    
    logger.debug(f"Searching flights: {flight_data_list}")
    try:
        result: Result = get_flights(
        flight_data=flight_data_list,
        trip=trip_type,
        seat=seat_type,
        passengers=passengers,
        fetch_mode="fallback"
    )
    except Exception:
        return json.dumps({
            "error": "An error occurred while fetching flight data, there may be no available flights for the given parameters.",
            "request": {
                "from_airport": from_airport,
                "to_airport": to_airport,
                "departure_date": departure_date,
                "return_date": return_date,
                "cabin": cabin,
                "adults": adults,
                "children": children,
                "infants_in_seat": infants_in_seat,
                "infants_on_lap": infants_on_lap,
                "airlines": airlines,
                "max_stops": max_stops,
            }
        })

    summary: List[Dict[str, Any]] = _result_to_dict(result)
    return json.dumps({
        "request": {
            "from_airport": from_airport,
            "to_airport": to_airport,
            "departure_date": departure_date,
            "return_date": return_date,
            "cabin": cabin,
            "adults": adults,
            "children": children,
            "infants_in_seat": infants_in_seat,
            "infants_on_lap": infants_on_lap,
            "airlines": airlines,
            "max_stops": max_stops,
        },
        "count": len(summary),
        "summary": summary
    })


def run_server():
    "Run the MCP server"
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()


