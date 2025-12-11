"""Mock provider with deterministic data for demonstration purposes."""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
from providers.base import ReservationProvider
from schemas import Restaurant, Location, AvailabilitySlot, Reservation, CancellationReceipt

logger = logging.getLogger(__name__)


class MockProvider(ReservationProvider):
    """
    Mock reservation provider with deterministic data.

    This provider uses in-memory storage and generates realistic but fake data
    for demonstration purposes. No external API calls are made.
    """

    def __init__(self):
        """Initialize the mock provider with sample restaurants and reservations."""
        self._restaurants: List[Restaurant] = self._initialize_restaurants()
        self._reservations: Dict[str, Reservation] = {}
        self._reservation_counter = 1000

    def _initialize_restaurants(self) -> List[Restaurant]:
        """Create a fixed set of sample restaurants."""
        return [
            Restaurant(
                id="rest_001",
                name="Trattoria di Mare",
                cuisine="Italian",
                price_tier=3,
                rating=4.5,
                phone="(617) 555-0101",
                description="Authentic Italian seafood in the North End",
                location=Location(
                    latitude=42.3656,
                    longitude=-71.0534,
                    address="123 Hanover St",
                    city="Boston",
                    state="MA",
                    postal_code="02113",
                ),
            ),
            Restaurant(
                id="rest_002",
                name="The Steakhouse",
                cuisine="American",
                price_tier=4,
                rating=4.7,
                phone="(617) 555-0102",
                description="Premium steaks and fine dining",
                location=Location(
                    latitude=42.3601,
                    longitude=-71.0589,
                    address="456 Newbury St",
                    city="Boston",
                    state="MA",
                    postal_code="02115",
                ),
            ),
            Restaurant(
                id="rest_003",
                name="Sakura Sushi",
                cuisine="Japanese",
                price_tier=2,
                rating=4.3,
                phone="(617) 555-0103",
                description="Fresh sushi and traditional Japanese cuisine",
                location=Location(
                    latitude=42.3505,
                    longitude=-71.0759,
                    address="789 Commonwealth Ave",
                    city="Boston",
                    state="MA",
                    postal_code="02215",
                ),
            ),
            Restaurant(
                id="rest_004",
                name="Le Petit Bistro",
                cuisine="French",
                price_tier=3,
                rating=4.6,
                phone="(617) 555-0104",
                description="Classic French bistro fare",
                location=Location(
                    latitude=42.3581,
                    longitude=-71.0636,
                    address="321 Beacon St",
                    city="Boston",
                    state="MA",
                    postal_code="02116",
                ),
            ),
            Restaurant(
                id="rest_005",
                name="Taqueria Azteca",
                cuisine="Mexican",
                price_tier=1,
                rating=4.1,
                phone="(617) 555-0105",
                description="Authentic Mexican street food",
                location=Location(
                    latitude=42.3736,
                    longitude=-71.1097,
                    address="555 Cambridge St",
                    city="Boston",
                    state="MA",
                    postal_code="02134",
                ),
            ),
            Restaurant(
                id="rest_006",
                name="Golden Dragon",
                cuisine="Chinese",
                price_tier=2,
                rating=4.4,
                phone="(212) 555-0201",
                description="Szechuan and Cantonese specialties",
                location=Location(
                    latitude=40.7589,
                    longitude=-73.9851,
                    address="100 Mott St",
                    city="New York",
                    state="NY",
                    postal_code="10013",
                ),
            ),
            Restaurant(
                id="rest_007",
                name="Bella Napoli",
                cuisine="Italian",
                price_tier=2,
                rating=4.2,
                phone="(212) 555-0202",
                description="Wood-fired Neapolitan pizza",
                location=Location(
                    latitude=40.7614,
                    longitude=-73.9776,
                    address="250 Mulberry St",
                    city="New York",
                    state="NY",
                    postal_code="10012",
                ),
            ),
            Restaurant(
                id="rest_008",
                name="Spice Route",
                cuisine="Indian",
                price_tier=2,
                rating=4.5,
                phone="(415) 555-0301",
                description="Modern Indian cuisine with California flair",
                location=Location(
                    latitude=37.7749,
                    longitude=-122.4194,
                    address="88 Mission St",
                    city="San Francisco",
                    state="CA",
                    postal_code="94105",
                ),
            ),
            Restaurant(
                id="rest_009",
                name="The Garden Café",
                cuisine="Vegetarian",
                price_tier=2,
                rating=4.4,
                phone="(415) 555-0302",
                description="Farm-to-table vegetarian dining",
                location=Location(
                    latitude=37.7833,
                    longitude=-122.4167,
                    address="456 Valencia St",
                    city="San Francisco",
                    state="CA",
                    postal_code="94110",
                ),
            ),
            Restaurant(
                id="rest_010",
                name="BBQ Pit Masters",
                cuisine="BBQ",
                price_tier=2,
                rating=4.6,
                phone="(512) 555-0401",
                description="Texas-style smoked meats",
                location=Location(
                    latitude=30.2672,
                    longitude=-97.7431,
                    address="123 Sixth St",
                    city="Austin",
                    state="TX",
                    postal_code="78701",
                ),
            ),
        ]

    def search_restaurants(
        self,
        city: str,
        cuisine: Optional[str] = None,
        date_time: Optional[str] = None,
        party_size: Optional[int] = None,
        price_tier: Optional[int] = None,
        distance_km: Optional[float] = None,
    ) -> List[Restaurant]:
        """Search restaurants with filters."""
        logger.debug(f"Searching restaurants in {city} with filters: cuisine={cuisine}, price_tier={price_tier}")

        results = []
        for restaurant in self._restaurants:
            # City filter (case-insensitive)
            if restaurant.location.city.lower() != city.lower():
                continue

            # Cuisine filter
            if cuisine and restaurant.cuisine.lower() != cuisine.lower():
                continue

            # Price tier filter
            if price_tier and restaurant.price_tier != price_tier:
                continue

            # Party size filter (simplified: assume all restaurants can handle up to 12)
            if party_size and party_size > 12:
                continue

            results.append(restaurant)

        logger.info(f"Found {len(results)} restaurants matching criteria")
        return results

    def check_availability(
        self,
        restaurant_id: str,
        date_time: str,
        party_size: int,
    ) -> List[AvailabilitySlot]:
        """Generate availability slots based on deterministic rules."""
        logger.debug(f"Checking availability for restaurant {restaurant_id} on {date_time} for {party_size} guests")

        # Validate restaurant exists
        restaurant = next((r for r in self._restaurants if r.id == restaurant_id), None)
        if not restaurant:
            raise ValueError(f"Restaurant {restaurant_id} not found")

        # Parse the date
        try:
            base_date = datetime.fromisoformat(date_time.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid date_time format: {date_time}")

        # Generate slots for lunch (11:30-14:00) and dinner (17:00-21:00)
        slots = []
        lunch_times = ["11:30", "12:00", "12:30", "13:00", "13:30"]
        dinner_times = ["17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30"]

        all_times = lunch_times + dinner_times

        for time_str in all_times:
            hour, minute = map(int, time_str.split(":"))
            slot_time = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Deterministic availability based on hash of restaurant_id + time + party_size
            # This ensures consistent results for the same inputs
            seed = f"{restaurant_id}_{slot_time.isoformat()}_{party_size}"
            hash_val = int(hashlib.sha256(seed.encode()).hexdigest(), 16)

            # 70% of slots are available
            available = (hash_val % 10) < 7

            # Max party size varies by restaurant (simulate table sizes)
            max_party_base = 8 if restaurant.price_tier >= 3 else 6
            max_party_size = max_party_base + (hash_val % 3)

            slots.append(
                AvailabilitySlot(
                    time=slot_time.isoformat(),
                    max_party_size=max_party_size,
                    available=available and party_size <= max_party_size,
                )
            )

        available_count = sum(1 for s in slots if s.available)
        logger.info(f"Found {available_count} available slots out of {len(slots)}")
        return slots

    def place_reservation(
        self,
        restaurant_id: str,
        date_time: str,
        party_size: int,
        name: str,
        phone: str,
        email: str,
        notes: Optional[str] = None,
    ) -> Reservation:
        """Create a mock reservation."""
        logger.debug(f"Placing reservation for {name} at {restaurant_id} on {date_time}")

        # Validate restaurant exists
        restaurant = next((r for r in self._restaurants if r.id == restaurant_id), None)
        if not restaurant:
            raise ValueError(f"Restaurant {restaurant_id} not found")

        # Check if duplicate (idempotency)
        # For simplicity, check if same email+datetime+restaurant already has a reservation
        duplicate_key = f"{email}_{date_time}_{restaurant_id}"
        for existing_res in self._reservations.values():
            existing_key = f"{existing_res.guest_email}_{existing_res.date_time}_{existing_res.restaurant_id}"
            if existing_key == duplicate_key:
                logger.info(f"Returning existing reservation (idempotent): {existing_res.id}")
                return existing_res

        # Generate confirmation code (only for new reservations)
        confirmation_code = f"RES{self._reservation_counter:06d}"
        self._reservation_counter += 1

        # Create new reservation
        reservation_id = f"reservation_{hashlib.sha256(confirmation_code.encode()).hexdigest()[:12]}"
        reservation = Reservation(
            id=reservation_id,
            restaurant_id=restaurant_id,
            restaurant_name=restaurant.name,
            date_time=date_time,
            party_size=party_size,
            guest_name=name,
            guest_phone=phone,
            guest_email=email,
            notes=notes,
            status="confirmed",
            confirmation_code=confirmation_code,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._reservations[reservation_id] = reservation
        logger.info(f"Created reservation {reservation_id} with confirmation {confirmation_code}")
        return reservation

    def cancel_reservation(
        self,
        reservation_id: str,
        reason: Optional[str] = None,
    ) -> CancellationReceipt:
        """Cancel a reservation."""
        logger.debug(f"Cancelling reservation {reservation_id}")

        reservation = self._reservations.get(reservation_id)
        if not reservation:
            raise ValueError(f"Reservation {reservation_id} not found")

        # Create cancellation receipt
        receipt = CancellationReceipt(
            reservation_id=reservation_id,
            restaurant_name=reservation.restaurant_name,
            original_date_time=reservation.date_time,
            cancelled_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            refund_policy="No charge for cancellations made more than 24 hours in advance",
        )

        # Remove from active reservations
        del self._reservations[reservation_id]
        logger.info(f"Cancelled reservation {reservation_id}")
        return receipt

    def list_reservations(
        self,
        user_id: str,
    ) -> List[Reservation]:
        """List all reservations for a user (by email or phone)."""
        logger.debug(f"Listing reservations for user {user_id}")

        results = []
        for reservation in self._reservations.values():
            if reservation.guest_email == user_id or reservation.guest_phone == user_id:
                results.append(reservation)

        logger.info(f"Found {len(results)} reservations for user {user_id}")
        return results
