"""FFmpeg-related literals for raw pipe I/O, H.264 export, and video helpers."""


class RawGray:
    """Grayscale ``rawvideo`` piped to or from ffmpeg."""

    PIPE = "pipe:"
    FORMAT = "rawvideo"
    PIX_FMT = "gray"


class H264:
    """H.264 / MP4 output used by :func:`~minian.visualization.export.write_video`."""

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
    """Chunk sizing for :mod:`minian.visualization.export` helpers."""

    STATS_REDUCE_FRAME_CHUNK_CAP = 32
    CONCAT_LIST_CHUNK = 256
