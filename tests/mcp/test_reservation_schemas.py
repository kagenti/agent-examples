"""Tests for reservation_tool schemas — Pydantic model validation."""

import pytest

from schemas import Location, Restaurant, AvailabilitySlot, Reservation, CancellationReceipt


class TestLocation:
    """Test Location model."""

    def test_valid_location(self):
        loc = Location(
            latitude=42.36,
            longitude=-71.05,
            address="123 Main St",
            city="Boston",
            state="MA",
            postal_code="02101",
        )
        assert loc.country == "USA"

    def test_custom_country(self):
        loc = Location(
            latitude=51.5,
            longitude=-0.12,
            address="10 Downing St",
            city="London",
            state="England",
            postal_code="SW1A 2AA",
            country="UK",
        )
        assert loc.country == "UK"


class TestRestaurant:
    """Test Restaurant model validation."""

    def _make_location(self):
        return Location(
            latitude=42.36,
            longitude=-71.05,
            address="123 Main St",
            city="Boston",
            state="MA",
            postal_code="02101",
        )

    def test_valid_restaurant(self):
        r = Restaurant(
            id="r1",
            name="Test",
            cuisine="Italian",
            price_tier=2,
            rating=4.5,
            phone="555-1234",
            location=self._make_location(),
        )
        assert r.accepts_reservations is True

    def test_price_tier_too_low(self):
        with pytest.raises(ValueError):
            Restaurant(
                id="r1",
                name="Test",
                cuisine="Italian",
                price_tier=0,
                rating=4.5,
                phone="555-1234",
                location=self._make_location(),
            )

    def test_price_tier_too_high(self):
        with pytest.raises(ValueError):
            Restaurant(
                id="r1",
                name="Test",
                cuisine="Italian",
                price_tier=5,
                rating=4.5,
                phone="555-1234",
                location=self._make_location(),
            )

    def test_rating_bounds(self):
        with pytest.raises(ValueError):
            Restaurant(
                id="r1",
                name="Test",
                cuisine="Italian",
                price_tier=2,
                rating=5.1,
                phone="555-1234",
                location=self._make_location(),
            )


class TestCancellationReceipt:
    """Test CancellationReceipt model."""

    def test_default_refund_policy(self):
        receipt = CancellationReceipt(
            reservation_id="res1",
            restaurant_name="Test",
            original_date_time="2025-03-15T19:00:00",
            cancelled_at="2025-03-14T10:00:00",
        )
        assert "No charge" in receipt.refund_policy
        assert receipt.reason is None
