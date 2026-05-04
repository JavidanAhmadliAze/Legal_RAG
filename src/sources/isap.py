"""
Accessor for isap.sejm.gov.pl via the public ELI REST API.

Scope:
  - Dziennik Ustaw (DU)
  - Monitor Polski (MP)
"""

from dataclasses import dataclass

from src.services.fetcher.http import fetch

_API_BASE = "https://api.sejm.gov.pl/eli/acts"


@dataclass
class ActMeta:
    address: str       # e.g. "WDU20250001746"
    journal: str       # "DU" or "MP"
    title: str
    act_type: str      # "Ustawa", "Rozporządzenie", ...
    year: int
    pos: int
    status: str
    announcement_date: str
    has_pdf: bool

    @property
    def pdf_url(self) -> str:
        return f"{_API_BASE}/{self.journal}/{self.year}/{self.pos}/text.pdf"


def list_acts(year: int, journal: str = "DU") -> list[ActMeta]:
    """Return all acts published in *journal* for *year*."""
    journal = journal.upper()
    resp = fetch(f"{_API_BASE}/{journal}/{year}")
    payload = resp.json()
    # API wraps results: {"count": N, "items": [...]}
    acts = payload["items"] if isinstance(payload, dict) else payload
    result: list[ActMeta] = []
    for a in acts:
        result.append(
            ActMeta(
                address=a.get("address", ""),
                journal=journal,
                title=a.get("title", ""),
                act_type=a.get("type", ""),
                year=a.get("year", year),
                pos=a.get("pos", 0),
                status=a.get("status", ""),
                announcement_date=a.get("announcementDate", ""),
                has_pdf=bool(a.get("textPDF", False)),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Foreign / immigration law
# ---------------------------------------------------------------------------

FOREIGN_LAW_KEYWORDS: list[str] = [
    # Core acts — foreigners
    "cudzoziemcy",          # cudzoziemcy / cudzoziemców / cudzoziemiec ...
    "cudzoziemca",
    "cudzoziemcem",
    "udzielani",            # udzielanie ochrony cudzoziemcom
    "ochrony na terytorium",
    # Residence / permits
    "zezwolenie na pobyt",
    "zezwolenia na pobyt",
    "pobyt czasowy",
    "pobyt stały",
    "legalizacja pobytu",
    "karta pobytu",
    # Entry / exit / border
    "wjazd",                # wjazd / wjazdu / wjazdem ...
    "przekraczani",         # przekraczanie granicy
    "straż graniczna",
    "straży granicznej",
    "ochrona granicy",
    "granica państwow",
    # Visas
    "wiza",                 # wiza / wizy / wiz ...
    "zaproszeni",           # zaproszenie cudzoziemca
    # Asylum / protection
    "azyl",
    "uchodźc",              # uchodźca / uchodźców ...
    "ochrona uzupełniająca",
    "ochrona czasowa",
    # Removal / expulsion
    "wydaleni",             # wydalenie / wydalenia ...
    "deportacj",
    "zobowiązanie do powrotu",
    # Citizenship / identity
    "obywatelstwo polskie",
    "nabycie obywatelstwa",
    "karta polaka",
    "karcie polaka",
    "repatriacj",          # repatriacja / repatriacji / repatriacyjna ...
    "repatriant",          # repatriant / repatrianta / repatriantowi ...
    # EU / EEA citizens
    "obywateli unii europejskiej",
    "obywatel unii",
    "swoboda przepływu",
    # Consular / documents
    "prawo konsularne",
    "konsularn",
    "dokument podróży",
    "dokument tożsamości",
    # Work authorisation for foreigners
    "zezwolenie na pracę",
    "zezwolenia na pracę",
    "oświadczenie o powierzeniu",
    # Ukrainian temporary protection (post-2022)
    "ochrona tymczasowa",
    "obywatel ukraińsk",
    "obywatele ukrainy",
    "pomocy obywatelom ukrainy",  # genitive — e.g. "o wygaszeniu rozwiązań ... z ustawy o pomocy"
    "pomoc obywatelom ukrainy",
    "pomoc cudzoziemcom",
    "wygaszeni",                  # wygaszenie rozwiązań / wygaśnięcie — phase-out acts
    "pesel",
    "specjalna ochrona",
]


def filter_foreign(acts: list[ActMeta]) -> list[ActMeta]:
    """Keep only acts related to foreigners / immigration law."""
    return _filter_by(acts, FOREIGN_LAW_KEYWORDS)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

# Only this status value means the act is currently in force.
VALID_STATUS = "obowiązujący"


def filter_valid(acts: list[ActMeta]) -> list[ActMeta]:
    """Hard filter: drop every act that is not currently in force.

    Blocked statuses include: uchylony, uznany za uchylony, wygaśnięcie aktu,
    akt objęty tekstem jednolitym, akt posiada tekst jednolity, akt jednorazowy.
    This is the first gate in every ingestion pipeline — nothing expired may pass.
    """
    valid: list[ActMeta] = []
    blocked_statuses = {
        "uchylony",
        "uznany za uchylony",
        "wygaśnięcie aktu",
    }
    for act in acts:
        if act.journal == "MP":
            # MP contains rollout communications and notices that are still useful
            # even when they do not carry the DU-style "obowiązujący" status.
            if act.status.lower() not in blocked_statuses:
                valid.append(act)
            continue
        if act.status == VALID_STATUS:
            valid.append(act)
    return valid


def _filter_by(acts: list[ActMeta], keywords: list[str]) -> list[ActMeta]:
    matches: list[ActMeta] = []
    for act in acts:
        title_lower = act.title.lower()
        if any(kw.lower() in title_lower for kw in keywords):
            matches.append(act)
    return matches
