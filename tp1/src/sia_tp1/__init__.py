"""Sokoban search engine for SIA TP1."""

from .config import AppConfig, ConfigError, load_config
from .domain import apply_move, is_goal
from .heuristics import (
    HEURISTIC_NAMES,
    get_heuristic,
    minimum_matching_manhattan_distance,
    shortest_push_access_distance,
)
from .model import Direction, Level, Position, State, Transition
from .parser import LevelFormatError, parse_level
from .render import render_state
from .search import (
    CutoffReason,
    Node,
    SearchLimits,
    SearchResult,
    SearchStatus,
    a_star_search,
    breadth_first_search,
    depth_first_search,
    greedy_search,
    reconstruct_nodes,
)
from .visualization import save_solution_gif

__all__ = [
    "AppConfig",
    "ConfigError",
    "CutoffReason",
    "Direction",
    "HEURISTIC_NAMES",
    "Level",
    "LevelFormatError",
    "Position",
    "Node",
    "SearchLimits",
    "SearchResult",
    "SearchStatus",
    "State",
    "Transition",
    "a_star_search",
    "apply_move",
    "breadth_first_search",
    "depth_first_search",
    "greedy_search",
    "get_heuristic",
    "is_goal",
    "load_config",
    "minimum_matching_manhattan_distance",
    "parse_level",
    "reconstruct_nodes",
    "render_state",
    "save_solution_gif",
    "shortest_push_access_distance",
]
