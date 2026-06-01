"""FFmpeg CLI checks and shared literals for raw pipe I/O and H.264 export."""

import shutil
import subprocess
from typing import Optional


class RawGray:
    """Grayscale ``rawvideo`` piped to or from ffmpeg."""

    PIPE = "pipe:"
    FORMAT = "rawvideo"
    PIX_FMT = "gray"


class H264:
    """H.264 / MP4 output used by video export helpers."""

    OUTPUT_PIX_FMT = "yuv420p"
    VCODEC = "libx264"
    FRAME_RATE = 30
    PAD_FILTER = "pad"
    OUTPUT_OPTIONS: dict[str, str] = {"crf": "18", "preset": "ultrafast"}


class Uint8:
    """Single-channel byte range (matches gray rawvideo)."""

    MAX = 255
    MIN = 0


class VideoExport:
    """Chunk sizing for video export helpers."""

    STATS_REDUCE_FRAME_CHUNK_CAP = 32
    CONCAT_LIST_CHUNK = 256


class FFmpegUnavailableError(RuntimeError):
    """Raised when FFmpeg CLI tools are missing or fail a basic ``-version`` check."""


_success: bool = False
_failure: Optional[str] = None


def ensure_ffmpeg() -> None:
    """Require working ``ffmpeg`` and ``ffprobe`` on ``PATH``.

    Cached after the first successful check (or first failure message for stable
    error reporting). Safe to call from hot paths (e.g. delayed workers): after
    the first call it only branches on booleans.
    """
    global _success, _failure
    if _success:
        return
    if _failure is not None:
        raise FFmpegUnavailableError(_failure)

    for name in ("ffmpeg", "ffprobe"):
        exe = shutil.which(name)
        if exe is None:
            _failure = (
                f"{name!r} not found on PATH. MiniAn needs FFmpeg binaries for "
                "AVI/MKV ingest and MP4 export. Install FFmpeg and ensure it is on PATH "
                "(https://ffmpeg.org/download.html)."
            )
            raise FFmpegUnavailableError(_failure)
        try:
            subprocess.run(
                [exe, "-version"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            _failure = f"{name!r} at {exe!r} failed to run (-version): {e}"
            raise FFmpegUnavailableError(_failure) from e

    _success = True
