from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from src.services.config import get_fetcher_settings

_cache: dict[str, RobotFileParser] = {}


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def can_fetch(url: str) -> bool:
    origin = _origin(url)
    if origin not in _cache:
        parser = RobotFileParser()
        parser.set_url(f"{origin}/robots.txt")
        try:
            parser.read()
        except Exception:
            pass
        _cache[origin] = parser
    return _cache[origin].can_fetch(get_fetcher_settings().user_agent, url)
