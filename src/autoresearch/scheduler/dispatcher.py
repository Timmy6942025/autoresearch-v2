"""Async task dispatcher for parallel agent execution."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("autoresearch.scheduler")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A unit of work for the dispatcher."""

    id: str
    name: str
    coro: Optional[Coroutine] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskDispatcher:
    """Async task dispatcher with priority queue and concurrency control."""

    def __init__(self, max_workers: int = 8, memory_limit_gb: float = 0.8):
        self.max_workers = max_workers
        self.memory_limit_gb = memory_limit_gb
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._tasks: Dict[str, Task] = {}
        self._semaphore = asyncio.Semaphore(max_workers)
        self._running = False

    async def submit(
        self,
        task_id: str,
        name: str,
        coro: Coroutine,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Submit a task for execution."""
        task = Task(
            id=task_id,
            name=name,
            coro=coro,
            priority=priority,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        await self._queue.put((-priority, task_id))
        logger.debug(f"Task submitted: {task_id} ({name}) priority={priority}")
        return task

    async def start(self) -> None:
        """Start the dispatcher loop."""
        self._running = True
        workers = [
            asyncio.create_task(self._worker(i)) for i in range(self.max_workers)
        ]
        await asyncio.gather(*workers, return_exceptions=True)

    async def _worker(self, worker_id: int) -> None:
        """Worker that processes tasks from the queue."""
        while self._running:
            try:
                neg_priority, task_id = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                continue

            async with self._semaphore:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                logger.debug(f"Worker {worker_id} executing: {task_id}")

                try:
                    result = await task.coro
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    logger.debug(f"Task completed: {task_id}")
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    task.completed_at = time.time()
                    logger.error(f"Task failed: {task_id} — {e}")
                finally:
                    self._queue.task_done()

    async def stop(self) -> None:
        """Stop the dispatcher."""
        self._running = False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_results(self) -> Dict[str, Any]:
        """Get results of all completed tasks."""
        return {
            tid: t.result
            for tid, t in self._tasks.items()
            if t.status == TaskStatus.COMPLETED
        }

    def get_failed(self) -> Dict[str, str]:
        """Get errors of all failed tasks."""
        return {
            tid: t.error
            for tid, t in self._tasks.items()
            if t.status == TaskStatus.FAILED
        }

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
