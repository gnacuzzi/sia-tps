"""Student-designed admissible heuristics for Sokoban."""

from collections import deque
from itertools import permutations
from math import inf
from typing import Callable, Deque, FrozenSet, Mapping, Set, Tuple

from .domain import is_goal
from .model import Direction, Level, Position, State


Heuristic = Callable[[Level, State], float]


def minimum_matching_manhattan_distance(
    level: Level,
    state: State,
) -> int:
    """Return the minimum Manhattan sum over one-to-one box-goal assignments."""

    if len(state.boxes) != len(level.goals):
        raise ValueError("The number of boxes and goals must match")

    return min(
        sum(
            _manhattan_distance(box, goal)
            for box, goal in zip(state.boxes, assigned_goals)
        )
        for assigned_goals in permutations(level.goals)
    )


def _manhattan_distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def shortest_push_access_distance(level: Level, state: State) -> float:
    """Return the shortest push-free player path to a valid push position."""

    if is_goal(level, state):
        return 0

    push_positions = _valid_push_positions(level, state)
    frontier: Deque[Tuple[Position, int]] = deque([(state.player, 0)])
    discovered_positions: Set[Position] = {state.player}

    while frontier:
        position, distance = frontier.popleft()
        if position in push_positions:
            return distance

        for direction in Direction:
            next_position = _add(position, direction.delta)
            if next_position in discovered_positions:
                continue
            if not level.is_floor(next_position):
                continue
            if next_position in state.boxes:
                continue

            discovered_positions.add(next_position)
            frontier.append((next_position, distance + 1))

    return inf


def _valid_push_positions(level: Level, state: State) -> Set[Position]:
    positions: Set[Position] = set()

    for box in state.boxes:
        for direction in Direction:
            player_position = _subtract(box, direction.delta)
            box_destination = _add(box, direction.delta)

            if not level.is_floor(player_position):
                continue
            if not level.is_floor(box_destination):
                continue
            if player_position in state.boxes:
                continue
            if box_destination in state.boxes:
                continue

            positions.add(player_position)

    return positions


def _add(first: Position, second: Position) -> Position:
    return first[0] + second[0], first[1] + second[1]


def _subtract(first: Position, second: Position) -> Position:
    return first[0] - second[0], first[1] - second[1]


_HEURISTICS: Mapping[str, Heuristic] = {
    "minimum_matching_manhattan": minimum_matching_manhattan_distance,
    "shortest_push_access": shortest_push_access_distance,
}
HEURISTIC_NAMES: FrozenSet[str] = frozenset(_HEURISTICS)


def get_heuristic(identifier: str) -> Heuristic:
    """Return the heuristic registered under a configuration identifier."""

    try:
        return _HEURISTICS[identifier]
    except KeyError as error:
        raise ValueError(f"Unknown heuristic: {identifier!r}") from error
