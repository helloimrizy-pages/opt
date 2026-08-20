from __future__ import annotations

from collections import deque
from itertools import combinations, product
from typing import Iterable, Sequence

import numpy as np

from residency_headroom.exact_solver import solve_exact
from residency_headroom.policies import CacheTransition


class LimitedLookaheadOracle:
    """Exact receding finite-horizon retention for equal-size, unit-cost misses."""

    def __init__(
        self,
        requests: Sequence[Iterable[int]],
        capacity: int,
        num_experts: int,
        horizon: int | None,
    ) -> None:
        if capacity < 0 or capacity > num_experts:
            raise ValueError("Invalid cache capacity")
        if horizon is not None and horizon < 1:
            raise ValueError("Lookahead horizon must be positive or None")
        self.requests = tuple(frozenset(map(int, value)) for value in requests)
        self.capacity = int(capacity)
        self.num_experts = int(num_experts)
        self.horizon = None if horizon is None else int(horizon)
        self.resident: frozenset[int] = frozenset()
        self.position = 0
        self.last_used = np.full(num_experts, -1, dtype=np.int64)
        self.future = [deque() for _ in range(num_experts)]
        for position, request in enumerate(self.requests):
            if not request or min(request) < 0 or max(request) >= num_experts:
                raise ValueError("Invalid atomic request")
            if capacity > 0 and len(request) > capacity:
                raise ValueError("Atomic request exceeds cache capacity")
            for expert in request:
                self.future[expert].append(position)

    @property
    def name(self) -> str:
        return "perfect_score_simple_policy" if self.horizon is None else f"lookahead_oracle_h{self.horizon}"

    def process(self, request: Iterable[int]) -> CacheTransition:
        if self.position >= len(self.requests):
            raise RuntimeError("Lookahead policy consumed too many events")
        requested = frozenset(map(int, request))
        if requested != self.requests[self.position]:
            raise RuntimeError("Lookahead policy request differs from frozen future")
        before = self.resident
        hits = requested & before
        misses = requested - before
        for expert in requested:
            if not self.future[expert] or self.future[expert][0] != self.position:
                raise RuntimeError("Lookahead future-use accounting changed")
            self.future[expert].popleft()
            self.last_used[expert] = self.position
        if self.capacity == 0:
            after = frozenset()
        else:
            spare = self.capacity - len(requested)
            old = before - requested
            infinity = len(self.requests) + 1
            cutoff = infinity if self.horizon is None else self.position + self.horizon

            def rank(expert: int) -> tuple[int, int, int]:
                following = self.future[expert][0] if self.future[expert] else infinity
                visible = following if following <= cutoff else infinity
                return visible, -int(self.last_used[expert]), expert

            after = requested | frozenset(sorted(old, key=rank)[:spare])
        admissions = after - before
        evictions = before - after
        if self.capacity > 0 and admissions != misses:
            raise RuntimeError("Lookahead policy violated mandatory admission")
        self.resident = after
        self.position += 1
        return CacheTransition(
            request=requested,
            before=before,
            after=after,
            hits=hits,
            misses=misses,
            admissions=admissions,
            evictions=evictions,
        )

    def finish(self) -> None:
        if self.position != len(self.requests):
            raise RuntimeError("Lookahead policy did not consume its complete future")


def validate_limited_lookahead(
    *, random_cases: int = 300, seed: int = 20260820
) -> dict[str, object]:
    """Validate local finite-horizon actions and full-horizon cost against exact DP."""

    cases: list[tuple[tuple[frozenset[int], ...], int, int]] = []
    for num_experts in range(1, 5):
        universe = tuple(range(num_experts))
        for capacity in range(1, num_experts + 1):
            choices = tuple(
                frozenset(items)
                for size in range(1, min(2, capacity) + 1)
                for items in combinations(universe, size)
            )
            for length in range(1, 5):
                for requests in product(choices, repeat=length):
                    cases.append((tuple(requests), capacity, num_experts))
    exhaustive_cases = len(cases)
    rng = np.random.default_rng(seed)
    for _ in range(random_cases):
        num_experts = int(rng.integers(2, 9))
        capacity = int(rng.integers(1, min(4, num_experts) + 1))
        length = int(rng.integers(1, 13))
        requests = tuple(
            frozenset(
                map(
                    int,
                    rng.choice(
                        num_experts,
                        size=int(rng.integers(1, min(3, capacity) + 1)),
                        replace=False,
                    ),
                )
            )
            for _position in range(length)
        )
        cases.append((requests, capacity, num_experts))

    action_checks = 0
    maximum_difference = 0.0
    for requests, capacity, num_experts in cases:
        exact = solve_exact(requests, capacity, num_experts)
        full = LimitedLookaheadOracle(requests, capacity, num_experts, None)
        full_misses = sum(len(full.process(request).misses) for request in requests)
        full.finish()
        difference = abs(float(full_misses) - exact.miss_cost)
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-10:
            raise AssertionError("Full perfect-score retention differs from Stage 0 exact DP")
        for horizon in (1, 2, 4, 8, 16, 32):
            policy = LimitedLookaheadOracle(requests, capacity, num_experts, horizon)
            for position, request in enumerate(requests):
                before = policy.resident
                transition = policy.process(request)
                visible = requests[position : min(len(requests), position + horizon + 1)]
                optimal = _optimal_cost_from_initial(visible, before, capacity)
                forced = len(request - before) + _optimal_cost_from_initial(
                    visible[1:], transition.after, capacity
                )
                action_checks += 1
                difference = abs(float(forced) - float(optimal))
                maximum_difference = max(maximum_difference, difference)
                if difference > 1e-10:
                    raise AssertionError(
                        f"H={horizon} first action is not finite-horizon optimal: "
                        f"requests={visible}, before={before}, after={transition.after}"
                    )
                if position == 0:
                    stage0_exact = solve_exact(visible, capacity, num_experts)
                    if abs(float(optimal) - stage0_exact.miss_cost) > 1e-10:
                        raise AssertionError("Initial-state validator differs from Stage 0 exact DP")
            policy.finish()
    return {
        "schema_version": "race_stage1_lookahead_validation_v1",
        "passed": True,
        "exhaustive_cases": exhaustive_cases,
        "random_cases": random_cases,
        "action_checks": action_checks,
        "maximum_cost_difference": maximum_difference,
        "seed": seed,
        "horizons": [1, 2, 4, 8, 16, 32],
        "stage0_exact_solver_reference": True,
    }


def _optimal_cost_from_initial(
    requests: Sequence[frozenset[int]], initial: frozenset[int], capacity: int
) -> int:
    current: dict[frozenset[int], int] = {frozenset(initial): 0}
    for request in requests:
        following: dict[frozenset[int], int] = {}
        for before, cost in current.items():
            event_cost = len(request - before)
            if capacity == 0:
                candidates = (frozenset(),)
            else:
                old = sorted(before - request)
                spare = capacity - len(request)
                candidates = (
                    request | frozenset(extra)
                    for size in range(min(spare, len(old)) + 1)
                    for extra in combinations(old, size)
                )
            for after in candidates:
                value = cost + event_cost
                previous = following.get(after)
                if previous is None or value < previous:
                    following[after] = value
        current = following
    return min(current.values(), default=0)
