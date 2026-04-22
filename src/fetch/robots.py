from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from src.fetch.http import USER_AGENT

_cache: dict[str, RobotFileParser] = {}


def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def can_fetch(url: str) -> bool:
    """Return True if robots.txt permits fetching *url*."""
    origin = _origin(url)
    if origin not in _cache:
        rp = RobotFileParser()
        rp.set_url(f"{origin}/robots.txt")
        try:
            rp.read()
        except Exception:
            # Unreachable robots.txt → assume allowed
            pass
        _cache[origin] = rp
    return _cache[origin].can_fetch(USER_AGENT, url)
