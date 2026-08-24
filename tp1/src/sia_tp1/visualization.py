"""Raster rendering and animation export for Sokoban solution paths."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import List, Optional, Union

from PIL import Image, ImageDraw, ImageFont

from .domain import is_goal
from .model import Level, Position, State, Transition
from .search import SearchResult, SearchStatus


_TILE_SIZE = 64
_HEADER_HEIGHT = 48
_VOID_COLOR = (25, 30, 38)
_WALL_COLOR = (55, 63, 74)
_WALL_BORDER_COLOR = (32, 38, 47)
_FLOOR_COLOR = (224, 216, 198)
_GOAL_COLOR = (238, 188, 91)
_GRID_COLOR = (191, 181, 162)
_BOX_COLOR = (166, 108, 58)
_BOX_BORDER_COLOR = (101, 61, 32)
_PLAYER_COLOR = (67, 142, 219)
_PLAYER_BORDER_COLOR = (29, 75, 120)
_TEXT_COLOR = (245, 247, 250)


def save_solution_gif(
    level: Level,
    result: SearchResult,
    algorithm: str,
    output_path: Union[str, Path],
    *,
    frame_duration_ms: int = 500,
) -> int:
    """Save one GIF frame per solution state and return the frame count."""

    frames = _solution_frames(level, result, algorithm, frame_duration_ms)

    gif_path = Path(output_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return len(frames)


def save_solution_video(
    level: Level,
    result: SearchResult,
    algorithm: str,
    output_path: Union[str, Path],
    *,
    frame_duration_ms: int = 500,
) -> int:
    """Save one MP4 frame per solution state and return the frame count.

    Videos use H.264 with the broadly compatible yuv420p pixel format. FFmpeg
    must be installed and available on PATH.
    """

    video_path = Path(output_path)
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("The video output path must use the .mp4 extension")

    frames = _solution_frames(level, result, algorithm, frame_duration_ms)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise OSError(
            "FFmpeg is required to generate MP4 videos but was not found on PATH"
        )

    video_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="sokoban-video-",
        dir=video_path.parent,
    ) as directory:
        temporary_directory = Path(directory)
        for index, frame in enumerate(frames):
            frame.save(temporary_directory / f"frame_{index:06d}.png")

        encoded_path = temporary_directory / "solution.mp4"
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                f"1000/{frame_duration_ms}",
                "-i",
                str(temporary_directory / "frame_%06d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(encoded_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown FFmpeg error"
            raise OSError(f"Could not generate MP4 video: {detail}")
        encoded_path.replace(video_path)

    return len(frames)


def _solution_frames(
    level: Level,
    result: SearchResult,
    algorithm: str,
    frame_duration_ms: int,
) -> List[Image.Image]:
    if result.status is not SearchStatus.SUCCESS:
        raise ValueError("A solution animation requires a successful search result")
    if type(frame_duration_ms) is not int or frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be a positive integer")

    solution_nodes = result.solution_nodes
    if solution_nodes is None:
        raise ValueError("A successful result must contain solution nodes")
    if not is_goal(level, solution_nodes[-1].state):
        raise ValueError("The final solution state must satisfy the goal")

    frames: List[Image.Image] = []
    total_moves = len(solution_nodes) - 1
    for step, node in enumerate(solution_nodes):
        frames.append(
            _render_frame(
                level=level,
                state=node.state,
                algorithm=algorithm,
                step=step,
                total_moves=total_moves,
                transition=node.transition,
            )
        )
    return frames


def _render_frame(
    *,
    level: Level,
    state: State,
    algorithm: str,
    step: int,
    total_moves: int,
    transition: Optional[Transition],
) -> Image.Image:
    width = level.width * _TILE_SIZE
    height = _HEADER_HEIGHT + level.height * _TILE_SIZE
    image = Image.new("RGB", (width, height), _VOID_COLOR)
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, _HEADER_HEIGHT), fill=_VOID_COLOR)
    draw.text(
        (12, 16),
        _frame_label(algorithm, step, total_moves, transition),
        fill=_TEXT_COLOR,
        font=_font(),
    )

    for row in range(level.height):
        for column in range(level.width):
            position = (row, column)
            _draw_static_tile(draw, level, position)

    for box in state.boxes:
        _draw_box(draw, box)
    _draw_player(draw, state.player)
    return image


def _frame_label(
    algorithm: str,
    step: int,
    total_moves: int,
    transition: Optional[Transition],
) -> str:
    prefix = f"{algorithm.upper()} | Step {step}/{total_moves}"
    if transition is None:
        return f"{prefix} | Initial state"
    pushed = "yes" if transition.pushed else "no"
    return f"{prefix} | {transition.direction.name} | Push: {pushed}"


def _draw_static_tile(
    draw: ImageDraw.ImageDraw,
    level: Level,
    position: Position,
) -> None:
    bounds = _tile_bounds(position)
    if position in level.walls:
        draw.rectangle(
            bounds,
            fill=_WALL_COLOR,
            outline=_WALL_BORDER_COLOR,
            width=3,
        )
        _draw_wall_pattern(draw, bounds)
        return
    if position in level.floors:
        color = _GOAL_COLOR if position in level.goals else _FLOOR_COLOR
        draw.rectangle(bounds, fill=color, outline=_GRID_COLOR, width=1)


def _draw_wall_pattern(
    draw: ImageDraw.ImageDraw,
    bounds: tuple,
) -> None:
    left, top, right, bottom = bounds
    middle = top + _TILE_SIZE // 2
    draw.line((left, middle, right, middle), fill=_WALL_BORDER_COLOR, width=2)
    draw.line(
        (left + _TILE_SIZE // 2, top, left + _TILE_SIZE // 2, middle),
        fill=_WALL_BORDER_COLOR,
        width=2,
    )
    draw.line(
        (left + _TILE_SIZE // 4, middle, left + _TILE_SIZE // 4, bottom),
        fill=_WALL_BORDER_COLOR,
        width=2,
    )


def _draw_box(draw: ImageDraw.ImageDraw, position: Position) -> None:
    left, top, right, bottom = _inset_bounds(position, 8)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=7,
        fill=_BOX_COLOR,
        outline=_BOX_BORDER_COLOR,
        width=4,
    )
    draw.line(
        (left + 8, top + 8, right - 8, bottom - 8),
        fill=_BOX_BORDER_COLOR,
        width=3,
    )
    draw.line(
        (right - 8, top + 8, left + 8, bottom - 8),
        fill=_BOX_BORDER_COLOR,
        width=3,
    )


def _draw_player(draw: ImageDraw.ImageDraw, position: Position) -> None:
    left, top, right, bottom = _inset_bounds(position, 10)
    draw.ellipse(
        (left, top, right, bottom),
        fill=_PLAYER_COLOR,
        outline=_PLAYER_BORDER_COLOR,
        width=4,
    )


def _tile_bounds(position: Position) -> tuple:
    row, column = position
    left = column * _TILE_SIZE
    top = _HEADER_HEIGHT + row * _TILE_SIZE
    return left, top, left + _TILE_SIZE, top + _TILE_SIZE


def _inset_bounds(position: Position, inset: int) -> tuple:
    left, top, right, bottom = _tile_bounds(position)
    return left + inset, top + inset, right - inset, bottom - inset


def _font():
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        return ImageFont.load_default()
