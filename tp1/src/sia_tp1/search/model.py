"""Algorithm-independent models for search nodes, limits, and results."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..model import State, Transition


@dataclass(frozen=True, eq=False)
class Node:
    """One occurrence of a state in a search tree."""

    state: State
    parent: Optional["Node"]
    transition: Optional[Transition]
    depth: int
    path_cost: float

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("Node depth cannot be negative")
        if self.path_cost < 0:
            raise ValueError("Node path cost cannot be negative")

        is_root = self.parent is None
        if is_root != (self.transition is None):
            raise ValueError(
                "Root nodes need both parent and transition set to None; "
                "child nodes need both values"
            )

        if is_root:
            if self.depth != 0 or self.path_cost != 0:
                raise ValueError("The root node must have depth and cost zero")
            return

        if self.depth != self.parent.depth + 1:
            raise ValueError("Child depth must be parent depth plus one")
        if self.state != self.transition.state:
            raise ValueError("Node state must match its incoming transition")


@dataclass(frozen=True)
class SearchLimits:
    max_expanded_nodes: Optional[int] = None
    timeout_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_expanded_nodes is not None and (
            type(self.max_expanded_nodes) is not int
            or self.max_expanded_nodes <= 0
        ):
            raise ValueError(
                "max_expanded_nodes must be a positive integer or None"
            )
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number or None")


class SearchStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CUTOFF = "cutoff"


class CutoffReason(Enum):
    TIMEOUT = "timeout"
    MAX_EXPANDED_NODES = "max_expanded_nodes"


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    goal_node: Optional[Node]
    expanded_nodes: int
    frontier_size_at_end: int
    max_frontier_size: int
    elapsed_seconds: float
    cutoff_reason: Optional[CutoffReason] = None

    def __post_init__(self) -> None:
        if self.expanded_nodes < 0:
            raise ValueError("expanded_nodes cannot be negative")
        if self.frontier_size_at_end < 0:
            raise ValueError("frontier_size_at_end cannot be negative")
        if self.max_frontier_size < 0:
            raise ValueError("max_frontier_size cannot be negative")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")

        if self.status is SearchStatus.SUCCESS:
            if self.goal_node is None:
                raise ValueError("SUCCESS requires a goal node")
            if self.cutoff_reason is not None:
                raise ValueError("SUCCESS cannot have a cutoff reason")
        elif self.goal_node is not None:
            raise ValueError("Only SUCCESS can have a goal node")

        if self.status is SearchStatus.CUTOFF:
            if self.cutoff_reason is None:
                raise ValueError("CUTOFF requires a cutoff reason")
        elif self.cutoff_reason is not None:
            raise ValueError("Only CUTOFF can have a cutoff reason")

    @property
    def solution_cost(self) -> Optional[float]:
        return self.goal_node.path_cost if self.goal_node is not None else None

    @property
    def solution_nodes(self) -> Optional[Tuple[Node, ...]]:
        if self.goal_node is None:
            return None
        return reconstruct_nodes(self.goal_node)

    @property
    def solution_transitions(self) -> Optional[Tuple[Transition, ...]]:
        nodes = self.solution_nodes
        if nodes is None:
            return None
        return tuple(
            node.transition
            for node in nodes[1:]
            if node.transition is not None
        )

    @property
    def solution_moves(self) -> Optional[int]:
        transitions = self.solution_transitions
        return len(transitions) if transitions is not None else None

    @property
    def solution_pushes(self) -> Optional[int]:
        transitions = self.solution_transitions
        if transitions is None:
            return None
        return sum(transition.pushed for transition in transitions)


def reconstruct_nodes(goal_node: Node) -> Tuple[Node, ...]:
    """Follow parent references and return the path from root to goal."""

    reversed_path = []
    current: Optional[Node] = goal_node
    while current is not None:
        reversed_path.append(current)
        current = current.parent

    reversed_path.reverse()
    return tuple(reversed_path)

