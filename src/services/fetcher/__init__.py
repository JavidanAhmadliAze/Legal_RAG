from src.services.config import get_fetcher_settings

_SETTINGS = get_fetcher_settings()

DEFAULT_YEARS = list(_SETTINGS.default_years)
DEFAULT_JOURNALS = list(_SETTINGS.default_journals)
DEFAULT_RATE_LIMIT_S = _SETTINGS.rate_limit_s


def fetch(
    url: str,
    *,
    retries: int | None = None,
    backoff: float | None = None,
):
    from .factory import get_fetcher_client

    return get_fetcher_client().fetch(url, retries=retries, backoff=backoff)


def can_fetch(url: str) -> bool:
    from .factory import get_fetcher_client

    return get_fetcher_client().can_fetch(url)


def list_new_acts(
    *,
    years: list[int] | None = None,
    journals: list[str] | None = None,
) -> list[dict]:
    from .factory import get_fetcher_client

    return get_fetcher_client().list_new_acts(years=years, journals=journals)


def download_and_parse_acts(
    acts: list[dict],
    *,
    rate_limit_s: float | None = None,
) -> list[str]:
    from .factory import get_fetcher_client

    return get_fetcher_client().download_and_parse_acts(
        acts,
        rate_limit_s=rate_limit_s,
    )
