"""Training event system for pub/sub notifications."""

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from largeforge.utils import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Types of training events."""

    # Job lifecycle events
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    JOB_PAUSED = "job_paused"
    JOB_RESUMED = "job_resumed"

    # Training progress events
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    STEP_COMPLETE = "step_complete"

    # Checkpoint events
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"

    # Evaluation events
    EVALUATION_START = "evaluation_start"
    EVALUATION_COMPLETE = "evaluation_complete"

    # Metrics events
    METRICS_LOGGED = "metrics_logged"

    # Resource events
    GPU_MEMORY_WARNING = "gpu_memory_warning"
    GPU_ERROR = "gpu_error"


@dataclass
class TrainingEvent:
    """A training event with associated data."""

    event_type: EventType
    job_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingEvent":
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            job_id=data["job_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
        )


# Type alias for event callbacks
EventCallback = Callable[[TrainingEvent], None]
AsyncEventCallback = Callable[[TrainingEvent], Any]


class EventEmitter:
    """
    Singleton event emitter for training events.

    Provides pub/sub functionality to decouple training from web layer.
    """

    _instance: Optional["EventEmitter"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "EventEmitter":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the event emitter."""
        if self._initialized:
            return

        self._subscribers: Dict[EventType, List[EventCallback]] = {}
        self._async_subscribers: Dict[EventType, List[AsyncEventCallback]] = {}
        self._global_subscribers: List[EventCallback] = []
        self._global_async_subscribers: List[AsyncEventCallback] = []
        self._event_history: List[TrainingEvent] = []
        self._max_history: int = 1000
        self._lock = threading.Lock()
        self._initialized = True

        logger.debug("EventEmitter initialized")

    @classmethod
    def get_instance(cls) -> "EventEmitter":
        """Get the singleton instance."""
        return cls()

    def subscribe(
        self,
        event_type: EventType,
        callback: EventCallback,
    ) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Subscribed to {event_type.value}")

    def subscribe_async(
        self,
        event_type: EventType,
        callback: AsyncEventCallback,
    ) -> None:
        """
        Subscribe to a specific event type with async callback.

        Args:
            event_type: Type of event to subscribe to
            callback: Async function to call when event occurs
        """
        with self._lock:
            if event_type not in self._async_subscribers:
                self._async_subscribers[event_type] = []
            if callback not in self._async_subscribers[event_type]:
                self._async_subscribers[event_type].append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """
        Subscribe to all event types.

        Args:
            callback: Function to call for all events
        """
        with self._lock:
            if callback not in self._global_subscribers:
                self._global_subscribers.append(callback)

    def subscribe_all_async(self, callback: AsyncEventCallback) -> None:
        """
        Subscribe to all event types with async callback.

        Args:
            callback: Async function to call for all events
        """
        with self._lock:
            if callback not in self._global_async_subscribers:
                self._global_async_subscribers.append(callback)

    def unsubscribe(
        self,
        event_type: EventType,
        callback: EventCallback,
    ) -> bool:
        """
        Unsubscribe from a specific event type.

        Args:
            event_type: Type of event
            callback: Callback to remove

        Returns:
            True if callback was removed, False otherwise
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    return True
                except ValueError:
                    pass
        return False

    def unsubscribe_all(self, callback: EventCallback) -> bool:
        """
        Unsubscribe from global events.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was removed
        """
        with self._lock:
            try:
                self._global_subscribers.remove(callback)
                return True
            except ValueError:
                return False

    def emit(self, event: TrainingEvent) -> int:
        """
        Emit an event to all subscribers.

        Args:
            event: Event to emit

        Returns:
            Number of callbacks notified
        """
        with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            callbacks = list(self._subscribers.get(event.event_type, []))
            global_callbacks = list(self._global_subscribers)

        notified = 0

        # Call type-specific callbacks
        for callback in callbacks:
            try:
                callback(event)
                notified += 1
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        # Call global callbacks
        for callback in global_callbacks:
            try:
                callback(event)
                notified += 1
            except Exception as e:
                logger.error(f"Global event callback error: {e}")

        logger.debug(f"Emitted {event.event_type.value} to {notified} subscribers")
        return notified

    async def emit_async(self, event: TrainingEvent) -> int:
        """
        Emit an event asynchronously.

        Args:
            event: Event to emit

        Returns:
            Number of callbacks notified
        """
        # First emit to sync subscribers
        notified = self.emit(event)

        # Then emit to async subscribers
        with self._lock:
            async_callbacks = list(self._async_subscribers.get(event.event_type, []))
            global_async_callbacks = list(self._global_async_subscribers)

        for callback in async_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                notified += 1
            except Exception as e:
                logger.error(f"Async event callback error: {e}")

        for callback in global_async_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                notified += 1
            except Exception as e:
                logger.error(f"Global async event callback error: {e}")

        return notified

    def emit_nonblocking(self, event: TrainingEvent) -> None:
        """
        Emit an event in a background thread.

        Args:
            event: Event to emit
        """
        thread = threading.Thread(target=self.emit, args=(event,), daemon=True)
        thread.start()

    def get_history(
        self,
        job_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[TrainingEvent]:
        """
        Get event history.

        Args:
            job_id: Filter by job ID
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        with self._lock:
            events = list(self._event_history)

        # Apply filters
        if job_id:
            events = [e for e in events if e.job_id == job_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Return most recent events
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._event_history.clear()

    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """Get number of subscribers for an event type."""
        with self._lock:
            if event_type:
                sync_count = len(self._subscribers.get(event_type, []))
                async_count = len(self._async_subscribers.get(event_type, []))
                return sync_count + async_count
            else:
                total = len(self._global_subscribers) + len(self._global_async_subscribers)
                for subs in self._subscribers.values():
                    total += len(subs)
                for subs in self._async_subscribers.values():
                    total += len(subs)
                return total


# Convenience functions
def emit_event(
    event_type: EventType,
    job_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Emit a training event.

    Args:
        event_type: Type of event
        job_id: Associated job ID
        data: Optional event data

    Returns:
        Number of callbacks notified
    """
    event = TrainingEvent(
        event_type=event_type,
        job_id=job_id,
        data=data or {},
    )
    return EventEmitter.get_instance().emit(event)


async def emit_event_async(
    event_type: EventType,
    job_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Emit a training event asynchronously.

    Args:
        event_type: Type of event
        job_id: Associated job ID
        data: Optional event data

    Returns:
        Number of callbacks notified
    """
    event = TrainingEvent(
        event_type=event_type,
        job_id=job_id,
        data=data or {},
    )
    return await EventEmitter.get_instance().emit_async(event)
