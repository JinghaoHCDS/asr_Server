from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from asr_server.config import Settings
from asr_server.jobs import JobManager
from asr_server.errors import AsrError


class ManagerStub:
    async def abort_all_workers(self) -> None:
        pass


async def test_external_operations_share_fifo_and_cancel_only_waiting() -> None:
    queue = JobManager(ManagerStub(), Settings())  # type: ignore[arg-type]
    entered, release = asyncio.Event(), asyncio.Event()
    order: list[str] = []
    second_done = asyncio.Event()

    async def first() -> None:
        order.append("a")
        entered.set()
        await release.wait()

    async def second() -> None:
        order.append("b")
        second_done.set()

    async def cancelled() -> None:
        order.append("c")

    await queue.enqueue_external("wb_a", first)
    await entered.wait()
    await queue.enqueue_external("wb_b", second)
    await queue.enqueue_external("wb_c", cancelled)
    assert queue.external_queue_position("wb_b") == 1
    assert not await queue.cancel_external("wb_a")
    assert await queue.cancel_external("wb_c")
    release.set()
    await asyncio.wait_for(second_done.wait(), 1)
    await queue.shutdown()
    assert order == ["a", "b"]


async def test_external_failure_does_not_stop_worker_or_escape_capacity() -> None:
    queue = JobManager(ManagerStub(), Settings(max_queued_jobs=1))  # type: ignore[arg-type]
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail() -> None:
        entered.set()
        await release.wait()
        raise ValueError("expected test exception")

    await queue.enqueue_external("wb_fail", fail)
    await entered.wait()
    with pytest.raises(AsrError) as error:
        await queue.enqueue_external("wb_blocked", fail)
    assert error.value.code == "job_queue_full"
    release.set()
    await asyncio.sleep(.01)
    done = asyncio.Event()
    await queue.enqueue_external("wb_next", _set_event(done))
    await asyncio.wait_for(done.wait(), 1)
    await queue.shutdown()


def _set_event(event: asyncio.Event) -> Callable[[], Awaitable[None]]:
    async def operation() -> None:
        event.set()
    return operation


async def test_failed_workbench_startup_releases_its_scheduler(tmp_path: Path) -> None:
    from asr_server.main import create_app
    first = create_app(Settings(workbench_data_dir=tmp_path, min_free_disk_mb=0))
    second = create_app(Settings(workbench_data_dir=tmp_path, min_free_disk_mb=0))
    async with first.router.lifespan_context(first):
        with pytest.raises(AsrError) as error:
            async with second.router.lifespan_context(second):
                raise AssertionError("a second service must not own the same data directory")
        assert error.value.code == "data_directory_in_use"
        assert second.state.job_manager._worker is None
        assert first.state.workbench.process_lock is not None
