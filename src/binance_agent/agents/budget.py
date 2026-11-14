"""单次 run 的保守请求与费用预算。

外部请求发送前先登记预留;网络结果或计费不明的尝试不会返还额度。
预算可写入 checkpoint,恢复时延续累计消耗与期限,避免重启获得新额度。
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import math
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class BudgetExceeded(RuntimeError):
    """在外部请求前报告请求数、费用或截止时间预算已经耗尽。"""


class BudgetPersistenceError(RuntimeError):
    """预算持久化失效,后续请求必须停止,以免恢复后重复消费。"""


@dataclass(slots=True)
class AttemptReservation:
    id: str
    endpoint: str
    logical_task: str
    kind: str
    reserved_at: float
    semaphore: asyncio.Semaphore | None = None
    acquired: bool = False
    record: dict[str, Any] = field(default_factory=dict)


class EndpointPool:
    """可注入多个 run 的共享端点并发槽及冷却状态;不合并各 run 的费用额度。"""

    def __init__(
        self,
        *,
        endpoint_concurrency: int = 2,
        per_endpoint_concurrency: dict[str, int] | None = None,
    ) -> None:
        if endpoint_concurrency < 1:
            raise ValueError("endpoint_concurrency must be at least 1")
        self.per_endpoint_concurrency = dict(per_endpoint_concurrency or {})
        if any(value < 1 for value in self.per_endpoint_concurrency.values()):
            raise ValueError("per_endpoint_concurrency values must be at least 1")
        self.endpoint_concurrency = endpoint_concurrency
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        self.cooldown_until: dict[str, float] = {}

    def concurrency_for(self, endpoint: str) -> int:
        return self.per_endpoint_concurrency.get(endpoint, self.endpoint_concurrency)


_current_budget: contextvars.ContextVar[RunBudget | None] = contextvars.ContextVar(
    "binance_agent_run_budget", default=None
)


def current_run_budget() -> RunBudget | None:
    return _current_budget.get()


@contextlib.contextmanager
def run_budget_scope(budget: RunBudget | None) -> Iterator[RunBudget | None]:
    """在当前异步任务上下文内提供预算,退出时恢复此前上下文值。"""
    token = _current_budget.set(budget)
    try:
        yield budget
    finally:
        _current_budget.reset(token)


class RunBudget:
    """维护单次 run 的请求、美元费用与截止时间,并借助端点池限制并发。"""

    def __init__(
        self,
        *,
        max_requests: int | None = None,
        max_cost_usd: float | None = None,
        deadline_s: float | None = None,
        deadline_at: float | datetime | None = None,
        endpoint_concurrency: int = 2,
        endpoint_prices: dict[str, dict[str, float]] | None = None,
        endpoint_pool: EndpointPool | None = None,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_requests is not None and max_requests < 0:
            raise ValueError("max_requests must be non-negative")
        if endpoint_concurrency < 1:
            raise ValueError("endpoint_concurrency must be at least 1")
        if max_cost_usd is not None and (not math.isfinite(max_cost_usd) or max_cost_usd <= 0):
            raise ValueError("max_cost_usd must be positive")
        self.max_cost_usd = max_cost_usd
        self.cost_committed_usd = 0.0
        self.max_requests = max_requests
        self.endpoint_concurrency = endpoint_concurrency
        self.endpoint_prices = dict(endpoint_prices or {})
        if any(
            not math.isfinite(v) or v < 0
            for prices in self.endpoint_prices.values()
            for v in prices.values()
        ):
            raise ValueError("prices must be finite and nonnegative")
        self.deadline_at: float | None
        if deadline_at is not None:
            self.deadline_at = (
                deadline_at.timestamp() if isinstance(deadline_at, datetime) else float(deadline_at)
            )
            self._deadline_clock = "wall"
        elif deadline_s is not None:
            # Wall time survives checkpoint/restart; monotonic time does not.
            self.deadline_at = time.time() + max(0.0, deadline_s)
            self._deadline_clock = "wall"
        else:
            self.deadline_at = None
            self._deadline_clock = "monotonic"
        self.endpoint_pool = endpoint_pool or EndpointPool(
            endpoint_concurrency=endpoint_concurrency
        )
        self._semaphores = self.endpoint_pool.semaphores
        self._cooldown_until = self.endpoint_pool.cooldown_until
        self._on_change = on_change
        self._lock = asyncio.Lock()
        self.attempts: list[dict[str, Any]] = []
        self.requests_reserved = 0
        self.requests_settled = 0
        self.unknown_cost_requests = 0
        self.cost_usd: float | None = 0.0
        self._persistence_failed = False

    def _now(self) -> float:
        return time.time() if self._deadline_clock == "wall" else time.monotonic()

    def _notify(self) -> None:
        if self._on_change is None:
            return
        # A missing checkpoint makes the remaining budget unknowable after a
        # process crash.  The caller must fail closed before sending a request.
        try:
            self._on_change(self.snapshot())
        except Exception as exc:
            self._persistence_failed = True
            raise BudgetPersistenceError("budget checkpoint failed; execution stopped") from exc

    def remaining_s(self) -> float | None:
        """返回距离截止时间的剩余秒数;无期限返回 None,已过期返回 0。"""
        if self.deadline_at is None:
            return None
        return max(0.0, self.deadline_at - self._now())

    def _check_deadline(self) -> None:
        if self._persistence_failed:
            raise BudgetPersistenceError("budget checkpoint failed; execution stopped")
        if self.deadline_at is not None and self._now() >= self.deadline_at:
            raise BudgetExceeded("run deadline exceeded")

    async def reserve(
        self,
        endpoint: str,
        logical_task: str,
        *,
        kind: str = "invoke",
        input_tokens_upper: int = 0,
        output_tokens_limit: int = 0,
        fixed_cost_usd: float | None = None,
    ) -> AttemptReservation:
        """先登记请求及费用预留,再等待端点并发槽和冷却期。

        费用上限启用时要求已知价格及有限 token 上界;超额或超时抛 BudgetExceeded。
        取消和准入失败会释放已持有的槽位并记录 not_sent,但不返还已预留额度。
        成功返回的 reservation 必须由调用方在请求结束后交给 settle。
        """
        if fixed_cost_usd is not None and (not math.isfinite(fixed_cost_usd) or fixed_cost_usd < 0):
            raise ValueError("tool cost must be finite and nonnegative")
        async with self._lock:
            self._check_deadline()
            if self.max_requests is not None and self.requests_reserved >= self.max_requests:
                raise BudgetExceeded("run request budget exhausted")
            estimate = self._cost_for(
                endpoint, {"input_tokens": input_tokens_upper, "output_tokens": output_tokens_limit}
            )
            if kind == "tool":
                estimate = fixed_cost_usd
            if self.max_cost_usd is not None:
                if estimate is None or (
                    kind != "tool" and (input_tokens_upper <= 0 or output_tokens_limit <= 0)
                ):
                    raise BudgetExceeded("priced finite input/output bounds required")
                if self.cost_committed_usd + estimate > self.max_cost_usd:
                    raise BudgetExceeded("monetary budget exhausted")
                self.cost_committed_usd += estimate
            attempt_id = uuid.uuid4().hex
            record = {
                "id": attempt_id,
                "endpoint": endpoint,
                "logical_task": logical_task,
                "kind": kind,
                "reserved_at": time.time(),
                "status": "pending",
                "usage": None,
                "cost_usd": None,
                "reserved_cost_usd": estimate if self.max_cost_usd is not None else None,
                "committed_cost_usd": estimate if self.max_cost_usd is not None else None,
                "fixed_cost_usd": fixed_cost_usd,
                "wait_ms": 0.0,
            }
            self.attempts.append(record)
            self.requests_reserved += 1
            semaphore = self._semaphores.setdefault(
                endpoint, asyncio.Semaphore(self.endpoint_pool.concurrency_for(endpoint))
            )
            self._notify()
        started = time.monotonic()
        acquired = False
        try:
            remaining = self.remaining_s()
            if remaining is not None and remaining <= 0:
                raise BudgetExceeded("run deadline exceeded")
            if remaining is None:
                await semaphore.acquire()
            else:
                await asyncio.wait_for(semaphore.acquire(), timeout=remaining)
            acquired = True
            # Keep the acquired permit while cooling down. Re-acquiring it here
            # would leak a permit and deadlock the endpoint after a 429.
            while delay := max(0.0, self._cooldown_until.get(endpoint, 0.0) - time.monotonic()):
                remaining = self.remaining_s()
                if remaining is not None and delay >= remaining:
                    raise BudgetExceeded(
                        "run deadline exceeded while waiting for endpoint cooldown"
                    )
                await asyncio.sleep(delay)
            self._check_deadline()
            if self.max_cost_usd is not None and self.cost_committed_usd > self.max_cost_usd:
                raise BudgetExceeded("monetary budget exhausted during admission")
            record["wait_ms"] = round((time.monotonic() - started) * 1000, 2)
            self._notify()
            return AttemptReservation(
                id=attempt_id,
                endpoint=endpoint,
                logical_task=logical_task,
                kind=kind,
                reserved_at=time.monotonic(),
                semaphore=semaphore,
                acquired=True,
                record=record,
            )
        except asyncio.CancelledError:
            if acquired:
                semaphore.release()
            with contextlib.suppress(Exception):
                await self._mark_not_sent(record, "cancelled")
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            if acquired:
                semaphore.release()
            await self._mark_not_sent(record, "deadline")
            raise BudgetExceeded("run deadline exceeded") from exc
        except Exception:
            if acquired:
                semaphore.release()
            await self._mark_not_sent(record, "admission_error")
            raise

    async def _mark_not_sent(self, record: dict[str, Any], reason: str) -> None:
        async with self._lock:
            if record["status"] == "pending":
                record["status"] = "not_sent"
                record["reason"] = reason
                self.requests_settled += 1
                self._notify()

    async def settle(
        self,
        reservation: AttemptReservation,
        *,
        usage: dict[str, int] | None = None,
        outcome: str = "ok",
        possibly_billed: bool = False,
        error_category: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        """结算一次预留并释放并发槽;重复结算不会重复计费或释放。

        usage 使用 token 数,retry_after_s 使用秒,费用使用美元。可能已计费或缺失
        usage 的尝试保留未知成本;冷却状态由端点池共享。持久化失败继续抛出,
        但 finally 仍释放本次槽位,后续消费由失效预算阻止。
        """
        try:
            async with self._lock:
                record = reservation.record
                if record.get("status") != "pending":
                    return
                record["status"] = "unknown" if possibly_billed else outcome
                record["usage"] = usage or None
                record["error_category"] = error_category
                record["finished_at"] = time.time()
                self.requests_settled += 1
                if outcome == "not_sent":
                    record["cost_usd"] = 0.0
                elif (
                    record.get("kind") == "tool"
                    and not possibly_billed
                    and record.get("fixed_cost_usd") is not None
                ):
                    cost = float(record["fixed_cost_usd"])
                    record["cost_usd"] = cost
                    if self.cost_usd is not None:
                        self.cost_usd = round(self.cost_usd + cost, 8)
                elif possibly_billed or not usage:
                    record["cost_usd"] = None
                    self.unknown_cost_requests += 1
                    self.cost_usd = None
                else:
                    actual_cost: float | None = self._cost_for(reservation.endpoint, usage)
                    record["cost_usd"] = actual_cost
                    if actual_cost is None:
                        self.unknown_cost_requests += 1
                        self.cost_usd = None
                    elif self.cost_usd is not None:
                        self.cost_usd = round(self.cost_usd + actual_cost, 8)
                self._commit_actual_cost(record)
                if (
                    retry_after_s is not None
                    and not isinstance(retry_after_s, bool)
                    and math.isfinite(retry_after_s)
                    and retry_after_s > 0
                ):
                    self._cooldown_until[reservation.endpoint] = max(
                        self._cooldown_until.get(reservation.endpoint, 0.0),
                        time.monotonic() + retry_after_s,
                    )
                    record["retry_after_s"] = retry_after_s
                self._notify()
        finally:
            if reservation.acquired and reservation.semaphore is not None:
                reservation.semaphore.release()
                reservation.acquired = False

    def _commit_actual_cost(self, record: dict[str, Any]) -> None:
        """累计预留与已知实际费用中的较大值;重复结算及旧快照不会降低已占额度。"""
        reserved = record.get("reserved_cost_usd")
        if not isinstance(reserved, (int, float)) or isinstance(reserved, bool):
            return
        previous = float(record.get("committed_cost_usd", reserved))
        actual = record.get("cost_usd")
        committed = max(previous, float(reserved))
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            committed = max(committed, float(actual))
        self.cost_committed_usd = round(self.cost_committed_usd + committed - previous, 8)
        record["committed_cost_usd"] = committed

    def _cost_for(self, endpoint: str, usage: dict[str, int]) -> float | None:
        """按端点每百万 token 的美元单价计算成本;缺少输入或输出价格时返回 None。"""
        price = self.endpoint_prices.get(endpoint)
        if not price:
            return None
        input_price = price.get("input_per_million")
        output_price = price.get("output_per_million")
        if input_price is None or output_price is None:
            return None
        return (
            usage.get("input_tokens", 0) * input_price
            + usage.get("output_tokens", 0) * output_price
        ) / 1_000_000

    def snapshot(self) -> dict[str, Any]:
        """生成可写入 checkpoint 的预算状态,不改变当前尝试的状态。

        端点冷却保存剩余秒数以跨进程恢复;尚为 pending 的尝试在 from_snapshot
        中转成 unknown,不能据旧快照认定请求未发送或费用为零。
        """
        return {
            "version": 1,
            "max_requests": self.max_requests,
            "max_cost_usd": self.max_cost_usd,
            "cost_committed_usd": self.cost_committed_usd,
            "deadline_at": self.deadline_at,
            "deadline_clock": self._deadline_clock,
            "endpoint_concurrency": self.endpoint_concurrency,
            "endpoint_prices": self.endpoint_prices,
            "requests_reserved": self.requests_reserved,
            "requests_settled": self.requests_settled,
            "unknown_cost_requests": self.unknown_cost_requests,
            "cost_usd": self.cost_usd,
            # ``EndpointPool`` uses a monotonic clock while the process is
            # alive.  Persist only the remaining duration so a restored run
            # can rebuild the cooldown against the new process' clock.
            "endpoint_cooldown_remaining_s": {
                endpoint: max(0.0, deadline - time.monotonic())
                for endpoint, deadline in self._cooldown_until.items()
                if deadline > time.monotonic()
            },
            "attempts": [dict(item) for item in self.attempts],
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any],
        *,
        endpoint_pool: EndpointPool | None = None,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunBudget:
        """恢复累计预算、截止时间和端点冷却,不重新发放请求或费用额度。

        pending 尝试统一转成 unknown 并计入已结算数;费用无法确认时保持 None。
        恢复只重建状态,不重发外部请求,也不重新获取旧请求的并发槽。
        """
        budget = cls(
            endpoint_pool=endpoint_pool,
            on_change=on_change,
            max_requests=snapshot.get("max_requests"),
            max_cost_usd=snapshot.get("max_cost_usd"),
            deadline_at=snapshot.get("deadline_at"),
            endpoint_concurrency=int(snapshot.get("endpoint_concurrency", 2)),
            endpoint_prices=snapshot.get("endpoint_prices") or {},
        )
        budget.cost_committed_usd = float(snapshot.get("cost_committed_usd", 0.0))
        budget._deadline_clock = str(snapshot.get("deadline_clock", "wall"))
        budget.attempts = [
            dict(item) for item in snapshot.get("attempts", []) if isinstance(item, dict)
        ]
        budget.requests_reserved = int(snapshot.get("requests_reserved", len(budget.attempts)))
        budget.requests_settled = int(snapshot.get("requests_settled", 0))
        budget.unknown_cost_requests = int(snapshot.get("unknown_cost_requests", 0))
        raw_cost = snapshot.get("cost_usd")
        budget.cost_usd = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
        raw_cooldowns = snapshot.get("endpoint_cooldown_remaining_s")
        if isinstance(raw_cooldowns, dict):
            now = time.monotonic()
            for endpoint, remaining in raw_cooldowns.items():
                try:
                    duration = max(0.0, float(remaining))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(duration) and duration:
                    budget._cooldown_until[str(endpoint)] = max(
                        budget._cooldown_until.get(str(endpoint), 0.0), now + duration
                    )
        # A checkpoint cannot know whether an interrupted pending request was billed.
        for item in budget.attempts:
            budget._commit_actual_cost(item)
            if item.get("status") == "pending":
                item["status"] = "unknown"
                item["recovered_from_pending"] = True
                item["cost_usd"] = None
                budget.unknown_cost_requests += 1
                budget.requests_settled += 1
                budget.cost_usd = None
        return budget
