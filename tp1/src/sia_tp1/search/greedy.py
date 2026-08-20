"""Greedy best-first graph search for Sokoban levels."""

from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Callable, List, Optional, Set, Tuple

from ..domain import apply_move, is_goal
from ..model import Direction, Level, State
from .model import (
    CutoffReason,
    Node,
    SearchLimits,
    SearchResult,
    SearchStatus,
)


Clock = Callable[[], float]
Heuristic = Callable[[Level, State], float]
PriorityEntry = Tuple[float, int, Node]


def greedy_search(
    level: Level,
    initial_state: State,
    heuristic: Heuristic,
    limits: Optional[SearchLimits] = None,
    *,
    clock: Clock = perf_counter,
) -> SearchResult:
    """Run deterministic Greedy search ordered only by heuristic value."""

    if limits is None:
        limits = SearchLimits()

    started_at = clock()
    root = Node(initial_state, None, None, 0, 0)
    insertion_order = count()
    frontier: List[PriorityEntry] = []
    heappush(
        frontier,
        (heuristic(level, initial_state), next(insertion_order), root),
    )
    discovered_states: Set[State] = {initial_state}
    expanded_nodes = 0
    max_frontier_size = 1

    while frontier:
        elapsed_seconds = clock() - started_at
        if (
            limits.timeout_seconds is not None
            and elapsed_seconds >= limits.timeout_seconds
        ):
            return _result(
                status=SearchStatus.CUTOFF,
                goal_node=None,
                expanded_nodes=expanded_nodes,
                frontier=frontier,
                max_frontier_size=max_frontier_size,
                started_at=started_at,
                clock=clock,
                cutoff_reason=CutoffReason.TIMEOUT,
            )

        if (
            limits.max_expanded_nodes is not None
            and expanded_nodes >= limits.max_expanded_nodes
        ):
            return _result(
                status=SearchStatus.CUTOFF,
                goal_node=None,
                expanded_nodes=expanded_nodes,
                frontier=frontier,
                max_frontier_size=max_frontier_size,
                started_at=started_at,
                clock=clock,
                cutoff_reason=CutoffReason.MAX_EXPANDED_NODES,
            )

        _, _, node = heappop(frontier)
        if is_goal(level, node.state):
            return _result(
                status=SearchStatus.SUCCESS,
                goal_node=node,
                expanded_nodes=expanded_nodes,
                frontier=frontier,
                max_frontier_size=max_frontier_size,
                started_at=started_at,
                clock=clock,
            )

        expanded_nodes += 1
        for direction in Direction:
            transition = apply_move(level, node.state, direction)
            if transition is None:
                continue

            child_state = transition.state
            if child_state in discovered_states:
                continue

            discovered_states.add(child_state)
            child = Node(
                state=child_state,
                parent=node,
                transition=transition,
                depth=node.depth + 1,
                path_cost=node.path_cost + 1,
            )
            heappush(
                frontier,
                (
                    heuristic(level, child_state),
                    next(insertion_order),
                    child,
                ),
            )

        max_frontier_size = max(max_frontier_size, len(frontier))

    return _result(
        status=SearchStatus.FAILURE,
        goal_node=None,
        expanded_nodes=expanded_nodes,
        frontier=frontier,
        max_frontier_size=max_frontier_size,
        started_at=started_at,
        clock=clock,
    )


def _result(
    *,
    status: SearchStatus,
    goal_node: Optional[Node],
    expanded_nodes: int,
    frontier: List[PriorityEntry],
    max_frontier_size: int,
    started_at: float,
    clock: Clock,
    cutoff_reason: Optional[CutoffReason] = None,
) -> SearchResult:
    return SearchResult(
        status=status,
        goal_node=goal_node,
        expanded_nodes=expanded_nodes,
        frontier_size_at_end=len(frontier),
        max_frontier_size=max_frontier_size,
        elapsed_seconds=clock() - started_at,
        cutoff_reason=cutoff_reason,
    )
