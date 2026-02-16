from pathlib import Path
from urllib.parse import urlparse


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
