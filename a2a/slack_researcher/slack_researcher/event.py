from abc import ABC, abstractmethod


class Event(ABC):
    @abstractmethod
    async def emit_event(self, message: str, event_type: str) -> None:
        """Emit Event"""
        pass
