"""WebSocket handler for real-time training progress updates."""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from largeforge.utils import get_logger
from largeforge.web.schemas import TrainingProgress, WebSocketMessage
from largeforge.web.state import JobStateManager, JobStatus

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for training job updates."""

    def __init__(self):
        """Initialize connection manager."""
        # job_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # All connections for broadcast
        self.all_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, job_id: Optional[str] = None) -> None:
        """
        Accept and register a WebSocket connection.

        Args:
            websocket: WebSocket connection
            job_id: Optional job ID to subscribe to
        """
        await websocket.accept()
        self.all_connections.add(websocket)

        if job_id:
            if job_id not in self.active_connections:
                self.active_connections[job_id] = set()
            self.active_connections[job_id].add(websocket)
            logger.debug(f"WebSocket connected for job {job_id}")
        else:
            logger.debug("WebSocket connected (global)")

    async def disconnect(self, websocket: WebSocket, job_id: Optional[str] = None) -> None:
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection
            job_id: Optional job ID the connection was subscribed to
        """
        self.all_connections.discard(websocket)

        if job_id and job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
            logger.debug(f"WebSocket disconnected from job {job_id}")
        else:
            logger.debug("WebSocket disconnected")

    async def send_message(self, websocket: WebSocket, message: WebSocketMessage) -> bool:
        """
        Send a message to a specific WebSocket.

        Args:
            websocket: Target WebSocket
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_json(message.model_dump(mode="json"))
            return True
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            return False

    async def broadcast_to_job(self, job_id: str, message: WebSocketMessage) -> int:
        """
        Broadcast a message to all connections subscribed to a job.

        Args:
            job_id: Job ID
            message: Message to broadcast

        Returns:
            Number of successful sends
        """
        if job_id not in self.active_connections:
            return 0

        connections = list(self.active_connections[job_id])
        sent = 0
        failed = []

        for websocket in connections:
            if await self.send_message(websocket, message):
                sent += 1
            else:
                failed.append(websocket)

        # Clean up failed connections
        for websocket in failed:
            await self.disconnect(websocket, job_id)

        return sent

    async def broadcast_progress(self, job_id: str, progress: TrainingProgress) -> int:
        """
        Broadcast training progress to subscribers.

        Args:
            job_id: Job ID
            progress: Progress data

        Returns:
            Number of successful sends
        """
        message = WebSocketMessage(
            type="progress",
            job_id=job_id,
            data=progress.model_dump(),
            timestamp=datetime.utcnow(),
        )
        return await self.broadcast_to_job(job_id, message)

    async def broadcast_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> int:
        """
        Broadcast job status change to subscribers.

        Args:
            job_id: Job ID
            status: New status
            error: Optional error message

        Returns:
            Number of successful sends
        """
        message = WebSocketMessage(
            type="status",
            job_id=job_id,
            data={"status": status, "error": error},
            timestamp=datetime.utcnow(),
        )
        return await self.broadcast_to_job(job_id, message)

    async def broadcast_log(self, job_id: str, log_line: str, level: str = "info") -> int:
        """
        Broadcast a log line to subscribers.

        Args:
            job_id: Job ID
            log_line: Log message
            level: Log level

        Returns:
            Number of successful sends
        """
        message = WebSocketMessage(
            type="log",
            job_id=job_id,
            data={"line": log_line, "level": level},
            timestamp=datetime.utcnow(),
        )
        return await self.broadcast_to_job(job_id, message)

    async def broadcast_all(self, message: WebSocketMessage) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast

        Returns:
            Number of successful sends
        """
        connections = list(self.all_connections)
        sent = 0
        failed = []

        for websocket in connections:
            if await self.send_message(websocket, message):
                sent += 1
            else:
                failed.append(websocket)

        # Clean up failed connections
        for websocket in failed:
            await self.disconnect(websocket)

        return sent

    def get_connection_count(self, job_id: Optional[str] = None) -> int:
        """Get number of active connections."""
        if job_id:
            return len(self.active_connections.get(job_id, set()))
        return len(self.all_connections)


# Global connection manager
connection_manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    job_id: str,
    state_manager: JobStateManager,
) -> None:
    """
    WebSocket endpoint handler for job updates.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to subscribe to
        state_manager: Job state manager
    """
    await connection_manager.connect(websocket, job_id)

    try:
        # Send current job state on connect
        job = state_manager.get_job(job_id)
        if job:
            initial_message = WebSocketMessage(
                type="status",
                job_id=job_id,
                data={
                    "status": job.status.value,
                    "progress": job.progress.model_dump() if job.progress else None,
                    "error": job.error,
                },
                timestamp=datetime.utcnow(),
            )
            await connection_manager.send_message(websocket, initial_message)
        else:
            error_message = WebSocketMessage(
                type="error",
                job_id=job_id,
                data={"error": f"Job not found: {job_id}"},
                timestamp=datetime.utcnow(),
            )
            await connection_manager.send_message(websocket, error_message)

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages with timeout for ping/pong
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Parse message
                try:
                    msg = json.loads(data)
                    msg_type = msg.get("type", "")

                    if msg_type == "ping":
                        pong = WebSocketMessage(
                            type="pong",
                            job_id=job_id,
                            timestamp=datetime.utcnow(),
                        )
                        await connection_manager.send_message(websocket, pong)

                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                ping = WebSocketMessage(
                    type="ping",
                    job_id=job_id,
                    timestamp=datetime.utcnow(),
                )
                await connection_manager.send_message(websocket, ping)

    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await connection_manager.disconnect(websocket, job_id)


async def global_websocket_endpoint(websocket: WebSocket) -> None:
    """
    Global WebSocket endpoint for all job updates.

    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        pong = WebSocketMessage(
                            type="pong",
                            timestamp=datetime.utcnow(),
                        )
                        await connection_manager.send_message(websocket, pong)
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                ping = WebSocketMessage(
                    type="ping",
                    timestamp=datetime.utcnow(),
                )
                await connection_manager.send_message(websocket, ping)

    except WebSocketDisconnect:
        logger.debug("Global WebSocket disconnected")
    except Exception as e:
        logger.error(f"Global WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)
