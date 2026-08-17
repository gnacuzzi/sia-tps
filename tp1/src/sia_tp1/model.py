"""Immutable domain models shared by the Sokoban engine."""

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Tuple


Position = Tuple[int, int]


class Direction(Enum):
    """A player action and its change in row and column."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def delta(self) -> Position:
        return self.value


@dataclass(frozen=True)
class Level:
    """Static information shared by every state of one level."""

    walls: FrozenSet[Position]
    floors: FrozenSet[Position]
    goals: FrozenSet[Position]
    height: int
    width: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "walls", frozenset(self.walls))
        object.__setattr__(self, "floors", frozenset(self.floors))
        object.__setattr__(self, "goals", frozenset(self.goals))

        if self.height <= 0 or self.width <= 0:
            raise ValueError("Level dimensions must be positive")
        if self.walls & self.floors:
            raise ValueError("A position cannot be both wall and floor")
        if not self.goals <= self.floors:
            raise ValueError("Every goal must also be a traversable floor")

    def is_wall(self, position: Position) -> bool:
        return position in self.walls

    def is_floor(self, position: Position) -> bool:
        return position in self.floors

    def is_void(self, position: Position) -> bool:
        return position not in self.walls and position not in self.floors


@dataclass(frozen=True)
class State:
    """Dynamic Sokoban information that changes between moves."""

    player: Position
    boxes: FrozenSet[Position]

    def __post_init__(self) -> None:
        object.__setattr__(self, "boxes", frozenset(self.boxes))


@dataclass(frozen=True)
class Transition:
    """The result of applying one valid direction to a state."""

    state: State
    direction: Direction
    pushed: bool

