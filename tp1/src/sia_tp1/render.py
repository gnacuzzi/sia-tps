"""Text rendering for Sokoban levels and states."""

from typing import List

from .model import Level, Position, State


def render_state(level: Level, state: State) -> str:
    """Combine static and dynamic layers into the agreed text format."""

    rendered_rows: List[str] = []
    for row in range(level.height):
        symbols: List[str] = []
        for column in range(level.width):
            position = (row, column)
            symbols.append(_symbol_at(level, state, position))
        rendered_rows.append("".join(symbols).rstrip())

    return "\n".join(rendered_rows)


def _symbol_at(level: Level, state: State, position: Position) -> str:
    if position == state.player:
        return "+" if position in level.goals else "@"
    if position in state.boxes:
        return "*" if position in level.goals else "$"
    if position in level.walls:
        return "#"
    if position in level.goals:
        return "."
    if position in level.floors:
        return "_"
    return " "

