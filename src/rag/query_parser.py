"""
Pre-translation query parser.

Extracts terms that appear verbatim in Polish legal texts and must not be
translated (system names, article references, Dz.U. citations). These are
injected back into the retrieval query after translation so BM25/kNN can
match them even when the translator drops or corrupts them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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
            r"\blong[\s-]?term\s+eu\s+residence\s+permit\b",
            re.IGNORECASE,
        ),
        "zezwolenie na pobyt rezydenta długoterminowego UE",
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


@dataclass
class ParsedQuery:
    original: str
    preserved: list[str] = field(default_factory=list)

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
    """Extract verbatim-preserve terms from an English or mixed query."""
    seen: dict[str, None] = {}
    for m in _PRESERVE_RE.finditer(text):
        seen[m.group()] = None
    for pattern, canonical in _POLISH_TERMS:
        if pattern.search(text):
            seen[canonical] = None
    return ParsedQuery(original=text, preserved=list(seen))
