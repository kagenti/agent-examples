"""Unit and integration tests for the reservation MCP server."""

import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.mock import MockProvider
from schemas import Restaurant, AvailabilitySlot, Reservation


class TestMockProvider:
    """Test the MockProvider implementation."""

    @pytest.fixture
    def provider(self):
        """Create a fresh MockProvider for each test."""
        return MockProvider()

    def test_search_restaurants_by_city(self, provider):
        """Test searching restaurants by city."""
        results = provider.search_restaurants(city="Boston")
        assert len(results) > 0
        assert all(isinstance(r, Restaurant) for r in results)
        assert all(r.location.city == "Boston" for r in results)

    def test_search_restaurants_by_cuisine(self, provider):
        """Test searching restaurants by cuisine."""
        results = provider.search_restaurants(city="Boston", cuisine="Italian")
        assert len(results) > 0
        assert all(r.cuisine == "Italian" for r in results)

    def test_search_restaurants_by_price_tier(self, provider):
        """Test searching restaurants by price tier."""
        results = provider.search_restaurants(city="Boston", price_tier=2)
        assert len(results) > 0
        assert all(r.price_tier == 2 for r in results)

    def test_search_restaurants_not_found(self, provider):
        """Test searching for restaurants in non-existent city."""
        results = provider.search_restaurants(city="NonExistentCity")
        assert len(results) == 0

    def test_check_availability(self, provider):
        """Test checking availability for a restaurant."""
        slots = provider.check_availability(
            restaurant_id="rest_001",
            date_time="2025-03-15T12:00:00",
            party_size=4
        )
        assert len(slots) > 0
        assert all(isinstance(s, AvailabilitySlot) for s in slots)
        # Slots should include both lunch and dinner times
        assert len(slots) >= 10

    def test_check_availability_invalid_restaurant(self, provider):
        """Test checking availability for non-existent restaurant."""
        with pytest.raises(ValueError, match="not found"):
            provider.check_availability(
                restaurant_id="invalid_id",
                date_time="2025-03-15T12:00:00",
                party_size=4
            )

    def test_place_reservation(self, provider):
        """Test placing a reservation."""
        reservation = provider.place_reservation(
            restaurant_id="rest_001",
            date_time="2025-03-15T19:00:00",
            party_size=4,
            name="John Doe",
            phone="+1-555-123-4567",
            email="john@example.com",
            notes="Window seat preferred"
        )
        assert isinstance(reservation, Reservation)
        assert reservation.restaurant_id == "rest_001"
        assert reservation.party_size == 4
        assert reservation.guest_name == "John Doe"
        assert reservation.status == "confirmed"
        assert len(reservation.confirmation_code) > 0

    def test_place_reservation_idempotent(self, provider):
        """Test that duplicate reservations are idempotent."""
        # Place first reservation
        res1 = provider.place_reservation(
            restaurant_id="rest_001",
            date_time="2025-03-15T19:00:00",
            party_size=4,
            name="John Doe",
            phone="+1-555-123-4567",
            email="john@example.com"
        )

        # Place duplicate reservation
        res2 = provider.place_reservation(
            restaurant_id="rest_001",
            date_time="2025-03-15T19:00:00",
            party_size=4,
            name="John Doe",
            phone="+1-555-123-4567",
            email="john@example.com"
        )

        # Should return same reservation
        assert res1.id == res2.id
        assert res1.confirmation_code == res2.confirmation_code

    def test_place_reservation_invalid_restaurant(self, provider):
        """Test placing reservation at non-existent restaurant."""
        with pytest.raises(ValueError, match="not found"):
            provider.place_reservation(
                restaurant_id="invalid_id",
                date_time="2025-03-15T19:00:00",
                party_size=4,
                name="John Doe",
                phone="+1-555-123-4567",
                email="john@example.com"
            )

    def test_list_reservations(self, provider):
        """Test listing reservations for a user."""
        # Place a reservation
        provider.place_reservation(
            restaurant_id="rest_001",
            date_time="2025-03-15T19:00:00",
            party_size=4,
            name="John Doe",
            phone="+1-555-123-4567",
            email="john@example.com"
        )

        # List by email
        reservations = provider.list_reservations(user_id="john@example.com")
        assert len(reservations) > 0
        assert all(r.guest_email == "john@example.com" for r in reservations)

        # List by phone
        reservations = provider.list_reservations(user_id="+1-555-123-4567")
        assert len(reservations) > 0
        assert all(r.guest_phone == "+1-555-123-4567" for r in reservations)

    def test_cancel_reservation(self, provider):
        """Test cancelling a reservation."""
        # Place a reservation
        reservation = provider.place_reservation(
            restaurant_id="rest_001",
            date_time="2025-03-15T19:00:00",
            party_size=4,
            name="John Doe",
            phone="+1-555-123-4567",
            email="john@example.com"
        )

        # Cancel it
        receipt = provider.cancel_reservation(
            reservation_id=reservation.id,
            reason="Change of plans"
        )

        assert receipt.reservation_id == reservation.id
        assert receipt.reason == "Change of plans"

        # Verify it's removed from list
        reservations = provider.list_reservations(user_id="john@example.com")
        assert not any(r.id == reservation.id for r in reservations)

    def test_cancel_reservation_not_found(self, provider):
        """Test cancelling non-existent reservation."""
        with pytest.raises(ValueError, match="not found"):
            provider.cancel_reservation(reservation_id="invalid_id")


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_restaurant_validation(self):
        """Test Restaurant schema validation."""
        from schemas import Location

        # Valid restaurant
        restaurant = Restaurant(
            id="test_001",
            name="Test Restaurant",
            cuisine="Italian",
            price_tier=3,
            rating=4.5,
            phone="555-1234",
            location=Location(
                latitude=42.36,
                longitude=-71.05,
                address="123 Main St",
                city="Boston",
                state="MA",
                postal_code="02101"
            )
        )
        assert restaurant.id == "test_001"

        # Invalid price tier
        with pytest.raises(ValueError):
            Restaurant(
                id="test_002",
                name="Test",
                cuisine="Italian",
                price_tier=5,  # Invalid: must be 1-4
                rating=4.5,
                phone="555-1234",
                location=Location(
                    latitude=42.36,
                    longitude=-71.05,
                    address="123 Main St",
                    city="Boston",
                    state="MA",
                    postal_code="02101"
                )
            )

    def test_availability_slot_validation(self):
        """Test AvailabilitySlot schema validation."""
        slot = AvailabilitySlot(
            time="2025-03-15T19:00:00",
            max_party_size=8,
            available=True
        )
        assert slot.available is True

        # Invalid party size
        with pytest.raises(ValueError):
            AvailabilitySlot(
                time="2025-03-15T19:00:00",
                max_party_size=0,  # Invalid: must be >= 1
                available=True
            )
