"""A* graph search for unit-cost Sokoban levels."""

from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

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


def a_star_search(
    level: Level,
    initial_state: State,
    heuristic: Heuristic,
    limits: Optional[SearchLimits] = None,
    *,
    clock: Clock = perf_counter,
) -> SearchResult:
    """Run deterministic A* with best-cost tracking and state reopening."""

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
    best_g: Dict[State, float] = {initial_state: 0}
    active_frontier_g: Dict[State, float] = {initial_state: 0}
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
                active_frontier_g=active_frontier_g,
                max_frontier_size=max_frontier_size,
                started_at=started_at,
                clock=clock,
                cutoff_reason=CutoffReason.TIMEOUT,
            )

        _, _, node = frontier[0]
        if active_frontier_g.get(node.state) != node.path_cost:
            heappop(frontier)
            continue

        if (
            limits.max_expanded_nodes is not None
            and expanded_nodes >= limits.max_expanded_nodes
        ):
            return _result(
                status=SearchStatus.CUTOFF,
                goal_node=None,
                expanded_nodes=expanded_nodes,
                active_frontier_g=active_frontier_g,
                max_frontier_size=max_frontier_size,
                started_at=started_at,
                clock=clock,
                cutoff_reason=CutoffReason.MAX_EXPANDED_NODES,
            )

        _, _, node = heappop(frontier)
        del active_frontier_g[node.state]

        if is_goal(level, node.state):
            return _result(
                status=SearchStatus.SUCCESS,
                goal_node=node,
                expanded_nodes=expanded_nodes,
                active_frontier_g=active_frontier_g,
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
            child_g = node.path_cost + 1
            known_g = best_g.get(child_state)
            if known_g is not None and child_g >= known_g:
                continue

            best_g[child_state] = child_g
            active_frontier_g[child_state] = child_g
            child = Node(
                state=child_state,
                parent=node,
                transition=transition,
                depth=node.depth + 1,
                path_cost=child_g,
            )
            heappush(
                frontier,
                (
                    child_g + heuristic(level, child_state),
                    next(insertion_order),
                    child,
                ),
            )

        max_frontier_size = max(max_frontier_size, len(frontier))

    return _result(
        status=SearchStatus.FAILURE,
        goal_node=None,
        expanded_nodes=expanded_nodes,
        active_frontier_g=active_frontier_g,
        max_frontier_size=max_frontier_size,
        started_at=started_at,
        clock=clock,
    )


def _result(
    *,
    status: SearchStatus,
    goal_node: Optional[Node],
    expanded_nodes: int,
    active_frontier_g: Dict[State, float],
    max_frontier_size: int,
    started_at: float,
    clock: Clock,
    cutoff_reason: Optional[CutoffReason] = None,
) -> SearchResult:
    return SearchResult(
        status=status,
        goal_node=goal_node,
        expanded_nodes=expanded_nodes,
        frontier_size_at_end=len(active_frontier_g),
        max_frontier_size=max_frontier_size,
        elapsed_seconds=clock() - started_at,
        cutoff_reason=cutoff_reason,
    )
