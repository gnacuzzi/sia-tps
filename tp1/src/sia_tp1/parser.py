"""Parser and validation for text-based Sokoban levels."""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from .model import Direction, Level, Position, State


class LevelFormatError(ValueError):
    """Raised when a level file violates the agreed format."""


_SYMBOLS: Dict[str, Tuple[bool, bool, bool, bool]] = {
    # symbol: (is_wall, is_floor, is_goal, has_box)
    "#": (True, False, False, False),
    "_": (False, True, False, False),
    ".": (False, True, True, False),
    "$": (False, True, False, True),
    "@": (False, True, False, False),
    "*": (False, True, True, True),
    "+": (False, True, True, False),
}


def parse_level(path: Union[str, Path]) -> Tuple[Level, State]:
    """Read, validate, and split one level into static and dynamic data."""

    level_path = Path(path)
    lines = level_path.read_text(encoding="utf-8").splitlines()
    while lines and lines[-1] == "":
        lines.pop()

    if not lines or not any(lines):
        raise LevelFormatError("The level file is empty")

    walls: Set[Position] = set()
    floors: Set[Position] = set()
    goals: Set[Position] = set()
    boxes: Set[Position] = set()
    players: List[Position] = []

    for row, line in enumerate(lines):
        for column, symbol in enumerate(line):
            if symbol == " ":
                continue
            if symbol not in _SYMBOLS:
                raise LevelFormatError(
                    f"Unknown symbol {symbol!r} at row {row}, column {column}"
                )

            position = (row, column)
            is_wall, is_floor, is_goal, has_box = _SYMBOLS[symbol]

            if is_wall:
                walls.add(position)
            if is_floor:
                floors.add(position)
            if is_goal:
                goals.add(position)
            if has_box:
                boxes.add(position)
            if symbol in {"@", "+"}:
                players.append(position)

    _validate_counts(players, boxes, goals)
    _validate_closed_floor(floors, walls)

    level = Level(
        walls=frozenset(walls),
        floors=frozenset(floors),
        goals=frozenset(goals),
        height=len(lines),
        width=max(len(line) for line in lines),
    )
    initial_state = State(player=players[0], boxes=frozenset(boxes))
    return level, initial_state


def _validate_counts(
    players: List[Position], boxes: Set[Position], goals: Set[Position]
) -> None:
    if len(players) != 1:
        raise LevelFormatError(
            f"A level must contain exactly one player; found {len(players)}"
        )
    if not boxes:
        raise LevelFormatError("A level must contain at least one box")
    if not goals:
        raise LevelFormatError("A level must contain at least one goal")
    if len(boxes) != len(goals):
        raise LevelFormatError(
            "A level must contain the same number of boxes and goals; "
            f"found {len(boxes)} boxes and {len(goals)} goals"
        )


def _validate_closed_floor(
    floors: Set[Position], walls: Set[Position]
) -> None:
    existing_positions = floors | walls
    for row, column in floors:
        for direction in Direction:
            delta_row, delta_column = direction.delta
            neighbor = (row + delta_row, column + delta_column)
            if neighbor not in existing_positions:
                raise LevelFormatError(
                    "Traversable position is open to the exterior at "
                    f"row {row}, column {column}"
                )
