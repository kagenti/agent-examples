"""Abstract base class for reservation providers."""

from abc import ABC, abstractmethod
from typing import List, Optional
from schemas import Restaurant, AvailabilitySlot, Reservation, CancellationReceipt


class ReservationProvider(ABC):
    """
    Abstract interface for restaurant reservation systems.

    This interface defines the contract that all reservation providers must implement,
    enabling the MCP server to work with different backends (OpenTable, SevenRooms, etc.)
    """

    @abstractmethod
    def search_restaurants(
        self,
        city: str,
        cuisine: Optional[str] = None,
        date_time: Optional[str] = None,
        party_size: Optional[int] = None,
        price_tier: Optional[int] = None,
        distance_km: Optional[float] = None,
    ) -> List[Restaurant]:
        """
        Search for restaurants matching the given criteria.

        Args:
            city: City name to search in
            cuisine: Optional cuisine type filter
            date_time: Optional ISO 8601 datetime for availability check
            party_size: Optional party size for filtering
            price_tier: Optional price tier (1-4)
            distance_km: Optional max distance from city center

        Returns:
            List of matching restaurants
        """
        pass

    @abstractmethod
    def check_availability(
        self,
        restaurant_id: str,
        date_time: str,
        party_size: int,
    ) -> List[AvailabilitySlot]:
        """
        Check availability for a specific restaurant.

        Args:
            restaurant_id: Restaurant identifier
            date_time: ISO 8601 datetime (day to check)
            party_size: Number of guests

        Returns:
            List of available time slots
        """
        pass

    @abstractmethod
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
        """
        Place a reservation at a restaurant.

        Args:
            restaurant_id: Restaurant identifier
            date_time: ISO 8601 datetime for the reservation
            party_size: Number of guests
            name: Guest name
            phone: Guest phone number
            email: Guest email address
            notes: Optional special requests

        Returns:
            Reservation confirmation

        Raises:
            ValueError: If restaurant not found or slot unavailable
        """
        pass

    @abstractmethod
    def cancel_reservation(
        self,
        reservation_id: str,
        reason: Optional[str] = None,
    ) -> CancellationReceipt:
        """
        Cancel an existing reservation.

        Args:
            reservation_id: Reservation identifier
            reason: Optional cancellation reason

        Returns:
            Cancellation confirmation

        Raises:
            ValueError: If reservation not found
        """
        pass

    @abstractmethod
    def list_reservations(
        self,
        user_id: str,
    ) -> List[Reservation]:
        """
        List all reservations for a user.

        Args:
            user_id: User identifier (email or phone)

        Returns:
            List of user's reservations
        """
        pass
