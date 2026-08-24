"""Combine the six core solution videos into one synchronized 3x2 grid."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence


VIDEO_LAYOUT = (
    "bfs.mp4",
    "greedy_mmm.mp4",
    "astar_mmm.mp4",
    "dfs.mp4",
    "greedy_spa.mp4",
    "astar_spa.mp4",
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the six core Sokoban videos. Shorter videos restart "
            "until the longest video finishes."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the six conventionally named MP4 files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 (default: <input_dir>/all_methods.mp4)",
    )
    arguments = parser.parse_args(argv)

    input_dir = arguments.input_dir.resolve()
    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else input_dir / "all_methods.mp4"
    )

    try:
        combine_solution_videos(input_dir, output_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


def combine_solution_videos(input_dir: Path, output_path: Path) -> float:
    """Create the looping comparison and return its duration in seconds."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise OSError("FFmpeg and FFprobe are required to combine videos")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("The output path must use the .mp4 extension")

    input_paths = [input_dir / filename for filename in VIDEO_LAYOUT]
    missing = [path.name for path in input_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Missing input videos: {', '.join(missing)}")

    durations = [_video_duration(ffprobe, path) for path in input_paths]
    total_duration = max(durations)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_arguments = []
    for path in input_paths:
        input_arguments.extend(["-stream_loop", "-1", "-i", str(path)])

    prepared_streams = ";".join(
        (
            f"[{index}:v]"
            "scale=640:540:force_original_aspect_ratio=decrease,"
            "pad=640:540:(ow-iw)/2:(oh-ih)/2:color=#191e26,"
            f"setsar=1,setpts=PTS-STARTPTS[v{index}]"
        )
        for index in range(len(input_paths))
    )
    layout = "0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0"
    stacked_streams = "".join(
        f"[v{index}]" for index in range(len(input_paths))
    )
    filter_graph = (
        f"{prepared_streams};{stacked_streams}"
        f"xstack=inputs=6:layout={layout}:fill=black[outv]"
    )

    with tempfile.TemporaryDirectory(
        prefix="sokoban-comparison-",
        dir=output_path.parent,
    ) as directory:
        encoded_path = Path(directory) / "combined.mp4"
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                *input_arguments,
                "-filter_complex",
                filter_graph,
                "-map",
                "[outv]",
                "-an",
                "-t",
                f"{total_duration:.6f}",
                "-c:v",
                "libx264",
                "-crf",
                "18",
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
            raise OSError(f"Could not combine videos: {detail}")
        encoded_path.replace(output_path)

    print(
        f"Combined video saved to: {output_path} "
        f"({total_duration:.1f} seconds)"
    )
    return total_duration


def _video_duration(ffprobe: str, path: Path) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown FFprobe error"
        raise OSError(f"Could not inspect {path.name}: {detail}")
    try:
        duration = float(completed.stdout.strip())
    except ValueError as error:
        raise OSError(f"Invalid duration reported for {path.name}") from error
    if duration <= 0:
        raise OSError(f"Invalid duration reported for {path.name}")
    return duration


if __name__ == "__main__":
    raise SystemExit(main())
