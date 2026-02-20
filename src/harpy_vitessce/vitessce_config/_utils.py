from pathlib import Path
from urllib.parse import urlparse

from loguru import logger


def _is_remote_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_path_or_url(path: str | Path, name: str) -> tuple[str, bool]:
    path_str = str(path)
    if not path_str:
        raise ValueError(f"{name} must be a non-empty path or URL.")
    parsed = urlparse(path_str)
    if parsed.scheme and not _is_remote_url(path_str):
        raise ValueError(
            f"{name} URL must start with http:// or https:// and include a host."
        )
    return path_str, _is_remote_url(path_str)


def _validate_camera(*, center: tuple[float, float] | None, zoom: float | None) -> None:
    if center is not None and len(center) != 2:
        raise ValueError("center must be a tuple of two floats: (x, y).")
    if zoom is not None and center is None:
        logger.warning(
            "zoom was provided without center. Vitessce ignores zoom unless "
            "center is also set."
        )
    if center is not None and zoom is None:
        logger.warning(
            "center was provided without zoom. Vitessce ignores center unless "
            "zoom is also set."
        )
