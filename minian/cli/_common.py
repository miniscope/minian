"""Small shared helpers for the ``minian`` CLI subcommands."""


def human_size(num_bytes: int) -> str:
    """Human-readable byte count, e.g. ``688.0 MB``."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def print_table(rows: list[tuple]) -> None:
    """Print rows as left-aligned columns; the last column is free-width.

    Shared by ``minian data list`` and ``minian notebooks list`` so the two
    near-identical listing commands don't each reimplement column padding.
    """
    rows = [tuple(str(c) for c in row) for row in rows]
    if not rows:
        return
    ncol = len(rows[0])
    widths = [max(len(r[i]) for r in rows) for i in range(ncol - 1)]
    for row in rows:
        head = "  ".join(f"{row[i]:<{widths[i]}}" for i in range(ncol - 1))
        print(f"{head}  {row[-1]}" if head else row[-1])
