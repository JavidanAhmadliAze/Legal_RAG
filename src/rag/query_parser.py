"""
Pre-translation query parser.

Extracts terms that appear verbatim in Polish legal texts and must not be
translated (system names, article references, Dz.U. citations). These are
injected back into the retrieval query after translation so BM25/kNN can
match them even when the translator drops or corrupts them.

Also extracts metadata filters (year, act_type) to narrow OpenSearch results
after BM25/kNN search but before RRF combining.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Ordered from most specific to least so longer matches win.
_PRESERVE_RE = re.compile(
    r"Dz\.U\.\s*\d{4}\s*poz\.\s*\d+"        # Dz.U. 2025 poz. 1794
    r"|[Aa]rt\.\s*\d+[a-z]*"                 # Art. 225a / art. 106
    r"|ust\.\s*\d+"                           # ust. 1
    r"|pkt\s*\d+"                             # pkt 3
    r"|§\s*\d+"                               # § 2
    r"|\bPESEL\b"                             # PESEL
    r"|\bUKR\b"                               # status UKR / PESEL UKR
    r"|\bCUKR\b"                              # karta pobytu CUKR
    r"|\bMOS\b"                               # MOS (Moduł Obsługi Spraw)
    r"|\bUDSC\b"                              # Urząd do Spraw Cudzoziemców
    r"|\bPIP\b"                               # Państwowa Inspekcja Pracy
    r"|\bB2B\b"                               # business-to-business contract label
    r"|\bStraż\s+Graniczna\b"                 # Border Guard
    r"|\bpoz\.\s*\d+"                         # poz. 1079
    r"|\b\d{4}\b(?=\s+poz\.)",               # year before poz.
)

# Polish legal terms that users often embed verbatim in English questions.
# When found, we inject them directly into the retrieval query so BM25 matches
# the exact Polish phrasing used in the legal texts.
_POLISH_TERMS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\btemporary\s+residence\s+permit\b", re.IGNORECASE),
        "zezwolenie na pobyt czasowy",
    ),
    (
        re.compile(r"\btemporary\s+protection\b", re.IGNORECASE),
        "ochrona czasowa",
    ),
    (
        re.compile(
            r"\b(trip|travel|away from Poland|outside Poland|go back|back home|return to Ukraine|visit.*family)\b",
            re.IGNORECASE,
        ),
        "wyjazd z terytorium Rzeczypospolitej Polskiej",
    ),
    (
        re.compile(
            r"\b(lose|losing|loss|expire|expir|end|terminat)\b.{0,40}\b(protection|status|right to stay|permit)\b"
            r"|\b(protection|status|right to stay|permit)\b.{0,40}\b(expire|expir|end|los|terminat)\b",
            re.IGNORECASE,
        ),
        "wygaśnięcie ochrony czasowej",
    ),
    (
        re.compile(r"\bpermanent\s+residence\s+permit\b", re.IGNORECASE),
        "zezwolenie na pobyt stały",
    ),
    (
        re.compile(
            r"\b5[- ]year[s]?\s+rule\b"
            r"|\bfive[- ]year\s+rule\b"
            r"|\b5\s+years?\s+(?:continuous|of\s+(?:legal\s+)?(?:stay|residence|living))\b"
            r"|\bdon['’]?t\s+(?:want|have)\s+to\s+(?:keep\s+)?renew\b"
            r"|\bstop\s+renewing\b"
            r"|\bkeep\s+renewing\b",
            re.IGNORECASE,
        ),
        "zezwolenie na pobyt rezydenta długoterminowego UE 5 lat nieprzerwanego pobytu",
    ),
    (
        re.compile(r"\bphoto(?:graph)?s?\b|\bpicture\b|\bportrait\b|\bheadshot\b", re.IGNORECASE),
        "fotografia",
    ),
    (
        re.compile(r"\bmarr(?:ied|iage)\b|\bwife\b|\bhusband\b|\bspouse\b|\bwedding\b", re.IGNORECASE),
        "związek małżeński",
    ),
    (
        re.compile(r"\bPolish\s+(?:citizen|national|wife|husband|spouse|partner)\b|\bPolish\s+(?:man|woman)\b", re.IGNORECASE),
        "obywatel polski",
    ),
    (
        re.compile(
            r"\b(\d+)\s+month[s]?\s+ago\b|\bphoto.*(?:old|age|valid|recent|taken)\b"
            r"|\b(?:old|age|valid|recent)\s+photo\b|\bexpir\w*\s+photo\b",
            re.IGNORECASE,
        ),
        "fotografia wymogi termin 6 miesięcy",
    ),
    (
        re.compile(
            r"\blong[\s-]?term\s+eu\s+residence\s+permit\b"
            r"|\beu\s+long[\s-]?term\s+residence\s+permit\b"
            r"|\blong[\s-]?term\s+resident\s+permit\b"
            r"|\beu\s+resident\s+permit\b",
            re.IGNORECASE,
        ),
        "zezwolenie na pobyt rezydenta długoterminowego UE",
    ),
    (
        re.compile(
            r"\b(?:Polish\s+)?language\s+(?:certificate|test|exam|requirement|proficiency|level)\b"
            r"|\blanguage\s+(?:skills?|knowledge)\b"
            r"|\b[AB][12]\s+(?:level|proficiency|certificate)?\b"
            r"|\bPolish\s+(?:level|proficiency|fluency|skills?)\b",
            re.IGNORECASE,
        ),
        "znajomość języka polskiego poświadczenie B1",
    ),
    (
        re.compile(r"\brepatriation\s+visa\b", re.IGNORECASE),
        "wiza w celu repatriacji",
    ),
    (re.compile(r"\bKarta\s+Pobytu\b", re.IGNORECASE),         "karta pobytu"),
    (re.compile(r"\bresidence\s+card\b", re.IGNORECASE),       "karta pobytu"),
    (re.compile(r"\bpobyt\s+czasowy\b", re.IGNORECASE),        "pobyt czasowy"),
    (re.compile(r"\bpobyt\s+sta[łl]y\b", re.IGNORECASE),       "pobyt stały"),
    (re.compile(r"\bzezwolenie\s+na\s+pobyt\b", re.IGNORECASE),"zezwolenie na pobyt"),
    (re.compile(r"\bwojewoda\b", re.IGNORECASE),               "wojewoda"),
    (re.compile(r"\bcudzoziemiec\b", re.IGNORECASE),           "cudzoziemiec"),
    (re.compile(r"\bwniosek\b", re.IGNORECASE),                "wniosek"),
    (re.compile(r"\bmeldunek\b|\bzameldowanie\b", re.IGNORECASE), "zameldowanie"),
    (re.compile(r"\brepatriacja\b|\brepatriant\b", re.IGNORECASE), "repatriacja"),
    (re.compile(r"\bKarta\s+Polaka\b", re.IGNORECASE),         "Karta Polaka"),
    (
        re.compile(r"\bPESEL\s+UKR\b|\bstatus\s+UKR\b", re.IGNORECASE),
        "PESEL UKR",
    ),
    (
        re.compile(r"\bUkraine\b|\bUkrainian\b", re.IGNORECASE),
        "Ukraina",
    ),
    (
        re.compile(r"\bpaper\s+(application|version)\b|\bmail(ed|ing)?\b", re.IGNORECASE),
        "wniosek papierowy",
    ),
    (
        re.compile(r"\bportal\b|\bonline\b|\belectronic(?:ally)?\b", re.IGNORECASE),
        "elektronicznie",
    ),
    (
        re.compile(
            r"\b(report|reporting|deadline|notify|notification|reporting rule)s?\b",
            re.IGNORECASE,
        ),
        "powiadomienie o powierzeniu pracy",
    ),
    (
        re.compile(r"\bdigital\s+reporting\b|\bsystem\b", re.IGNORECASE),
        "system teleinformatyczny",
    ),
    (
        re.compile(
            r"\bforeign(er| worker| employee)?\b|\bhiring\s+a\s+foreigner\b",
            re.IGNORECASE,
        ),
        "powierzenie pracy cudzoziemcowi",
    ),
    (
        re.compile(r"\bemployer\b|\bemployee\b|\bemployment\b", re.IGNORECASE),
        "powierzenie pracy cudzoziemcowi",
    ),
    (
        re.compile(
            r"\bPIP\b|\bPa[nń]stwowa\s+Inspekcja\s+Pracy\b|\bLabou?r\s+Inspectorate\b",
            re.IGNORECASE,
        ),
        "Państwowa Inspekcja Pracy",
    ),
    (
        re.compile(r"\bB2B\b|\bself-employ(?:ed|ment)\b", re.IGNORECASE),
        "umowa cywilnoprawna",
    ),
    (
        re.compile(
            r"\bB2B\b|\bemployment\s+contract\b|\bcontract\s+change\b|\breclassif(y|ication)\b",
            re.IGNORECASE,
        ),
        "stosunek pracy",
    ),
    (
        re.compile(r"\bemployment\s+contract\b|\bcontract\s+of\s+employment\b", re.IGNORECASE),
        "umowa o pracę",
    ),
]


# ---------------------------------------------------------------------------
# Metadata filter extraction
# ---------------------------------------------------------------------------

# Exact year: "Dz.U. 2024", "from/of/in 2024", "2024 ustawa/rozporządzenie"
_YEAR_EXACT_RE = re.compile(
    r"Dz\.U\.\s*((?:19|20)\d{2})"
    r"|\b(?:from|of|in)\s+((?:19|20)\d{2})\b"
    r"|\b((?:19|20)\d{2})\s+(?:ustawa|rozporządzenie|law|act|regulation)\b"
    r"|\bustawa\s+z\s+(?:\S+\s+){0,3}((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
# Year lower/upper bounds
_YEAR_GTE_RE = re.compile(r"\b(?:since|after)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_YEAR_LTE_RE = re.compile(r"\b(?:before|until)\s+((?:19|20)\d{2})\b", re.IGNORECASE)

# Act type: only explicit Polish legal-document terms + "international treaty".
# "Ustawa" expands to [Ustawa, Obwieszczenie] because the consolidated text of
# a ustawa is published as an Obwieszczenie (jednolity tekst) — filtering to
# Ustawa alone would drop the authoritative consolidated texts.
_ACT_TYPE_MAP: list[tuple[re.Pattern[str], list[str] | str]] = [
    (re.compile(r"\bustawa\b", re.IGNORECASE), ["Ustawa", "Obwieszczenie"]),
    (re.compile(r"\brozporz[aą]dzeni\w*\b", re.IGNORECASE), "Rozporządzenie"),
    (re.compile(r"\bobwieszczeni\w*\b", re.IGNORECASE), "Obwieszczenie"),
    (re.compile(r"\bumowa\s+mi[eę]dzynarodow\w*\b", re.IGNORECASE), "Umowa międzynarodowa"),
]


def _extract_filters(text: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    gte_m = _YEAR_GTE_RE.search(text)
    lte_m = _YEAR_LTE_RE.search(text)
    if gte_m or lte_m:
        year_range: dict[str, int] = {}
        if gte_m:
            y = int(next(g for g in gte_m.groups() if g))
            year_range["gte"] = y + 1 if "after" in gte_m.group(0).lower() else y
        if lte_m:
            y = int(next(g for g in lte_m.groups() if g))
            year_range["lte"] = y - 1 if "before" in lte_m.group(0).lower() else y
        filters["year"] = year_range
    else:
        m = _YEAR_EXACT_RE.search(text)
        if m:
            filters["year"] = int(next(g for g in m.groups() if g))

    for pattern, act_type in _ACT_TYPE_MAP:
        if pattern.search(text):
            filters["act_type"] = act_type
            break

    return filters


@dataclass
class ParsedQuery:
    original: str
    preserved: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)

    def preserved_hint(self) -> str:
        """Comma-separated list for inclusion in the translation prompt."""
        return ", ".join(self.preserved) if self.preserved else ""

    def augment(self, translated: str) -> str:
        """Append any preserved terms not already present in *translated*."""
        extras = [t for t in self.preserved if t.lower() not in translated.lower()]
        if not extras:
            return translated
        return translated + " " + " ".join(extras)


def parse_query(text: str) -> ParsedQuery:
    """Extract verbatim-preserve terms and metadata filters from a query."""
    seen: dict[str, None] = {}
    for m in _PRESERVE_RE.finditer(text):
        seen[m.group()] = None
    for pattern, canonical in _POLISH_TERMS:
        if pattern.search(text):
            seen[canonical] = None
    return ParsedQuery(original=text, preserved=list(seen), filters=_extract_filters(text))
