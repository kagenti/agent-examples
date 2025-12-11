"""Data models for the Restaurant Reservation MCP server."""

from typing import Optional
from pydantic import BaseModel, Field


class Location(BaseModel):
    """Geographic location with coordinates and address."""

    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    address: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State or province")
    postal_code: str = Field(..., description="Postal/ZIP code")
    country: str = Field(default="USA", description="Country")


class Restaurant(BaseModel):
    """Restaurant information."""

    id: str = Field(..., description="Unique restaurant identifier")
    name: str = Field(..., description="Restaurant name")
    cuisine: str = Field(..., description="Cuisine type")
    price_tier: int = Field(..., ge=1, le=4, description="Price tier (1=$, 2=$$, 3=$$$, 4=$$$$)")
    rating: float = Field(..., ge=0, le=5, description="Average rating (0-5)")
    location: Location = Field(..., description="Restaurant location")
    phone: str = Field(..., description="Contact phone number")
    description: Optional[str] = Field(None, description="Restaurant description")
    accepts_reservations: bool = Field(default=True, description="Whether reservations are accepted")


class AvailabilitySlot(BaseModel):
    """Available time slot for reservations."""

    time: str = Field(..., description="ISO 8601 datetime of the slot")
    max_party_size: int = Field(..., ge=1, description="Maximum party size for this slot")
    available: bool = Field(..., description="Whether the slot is available")


class Reservation(BaseModel):
    """Restaurant reservation details."""

    id: str = Field(..., description="Unique reservation identifier")
    restaurant_id: str = Field(..., description="Restaurant identifier")
    restaurant_name: str = Field(..., description="Restaurant name")
    date_time: str = Field(..., description="ISO 8601 datetime of reservation")
    party_size: int = Field(..., ge=1, description="Number of guests")
    guest_name: str = Field(..., description="Guest name")
    guest_phone: str = Field(..., description="Guest phone number")
    guest_email: str = Field(..., description="Guest email address")
    notes: Optional[str] = Field(None, description="Special requests or notes")
    status: str = Field(default="confirmed", description="Reservation status")
    confirmation_code: str = Field(..., description="Confirmation code")
    created_at: str = Field(..., description="ISO 8601 datetime when reservation was created")


class CancellationReceipt(BaseModel):
    """Reservation cancellation confirmation."""

    reservation_id: str = Field(..., description="Cancelled reservation identifier")
    restaurant_name: str = Field(..., description="Restaurant name")
    original_date_time: str = Field(..., description="ISO 8601 datetime of original reservation")
    cancelled_at: str = Field(..., description="ISO 8601 datetime when cancelled")
    reason: Optional[str] = Field(None, description="Cancellation reason")
    refund_policy: str = Field(default="No charge for cancellations", description="Refund policy message")
