"""Sokoban transition rules and goal condition."""

from typing import Optional

from .model import Direction, Level, Position, State, Transition


def apply_move(
    level: Level, state: State, direction: Direction
) -> Optional[Transition]:
    """Apply one direction without mutating the level or parent state.

    Return ``None`` when the movement is blocked by a wall, the exterior,
    or a box that cannot be pushed.
    """

    next_position = _add(state.player, direction.delta)

    if not level.is_floor(next_position):
        return None

    if next_position not in state.boxes:
        return Transition(
            state=State(player=next_position, boxes=state.boxes),
            direction=direction,
            pushed=False,
        )

    box_destination = _add(next_position, direction.delta)
    if not level.is_floor(box_destination):
        return None
    if box_destination in state.boxes:
        return None

    new_boxes = frozenset(
        (state.boxes - {next_position}) | {box_destination}
    )
    return Transition(
        state=State(player=next_position, boxes=new_boxes),
        direction=direction,
        pushed=True,
    )


def is_goal(level: Level, state: State) -> bool:
    """Return whether every box occupies an objective position."""

    return state.boxes == level.goals


def _add(first: Position, second: Position) -> Position:
    """Add two row-column coordinate pairs."""

    return first[0] + second[0], first[1] + second[1]

