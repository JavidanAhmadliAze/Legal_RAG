"""
Rule-based law domain classifier for Polish legal acts.

Document classification: title + act_type → single domain label stored in
chunk metadata and used as an OpenSearch keyword filter.

Query classification: English/Polish question text → domain filter injected
into retrieval only when the signal is unambiguous (exactly one domain
detected). Ambiguous queries (e.g. foreign-worker employment which spans
immigration AND labor) receive no filter so no relevant chunks are excluded.

Domain taxonomy
---------------
immigration    cudzoziemcy, pobyt, wiza, azyl, repatriacja, ochrona
labor          Kodeks pracy, zatrudnienie, wynagrodzenie, czas pracy
criminal       Kodeks karny, przestępstwo, wykroczenie, kara
civil          Kodeks cywilny, rękojmia, odszkodowanie, najem, sprzedaż
administrative KPA, postępowanie administracyjne
tax            VAT, PIT, CIT, Ordynacja podatkowa
social         ZUS, emerytury, renty, zasiłki, ubezpieczenia społeczne
commercial     spółki, Kodeks spółek handlowych, działalność gospodarcza
environmental  ochrona środowiska, gospodarka odpadami
constitutional Konstytucja RP, Trybunał Konstytucyjny
other          fallback when no rule matches
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Document-level classification  (title + act_type → domain)
# ---------------------------------------------------------------------------

# Rules are ordered most-specific-first so that, e.g., "o zatrudnieniu
# cudzoziemców" hits immigration before the generic labor rules.
_DOC_RULES: list[tuple[str, list[str]]] = [
    ("immigration", [
        r"cudzoziem",                       # all inflected forms: cudzoziemca, cudzoziemcach, …
        r"\bwiz[ay]\b",
        r"pobytu?\s+cudzoziem",
        r"\bazyl(?!u?\s+dla\s+[Zz]wierz)",  # azyl=asylum, not "Azyl dla Zwierząt"
        r"uchodźc",
        r"ochron\w+\s+czasow",
        r"Straży?\s+Graniczn",
        r"repatriacja",
        r"repatrianc",
        r"przekraczani.*granicy",
        r"ochron\w+\s+uzupełniaj",
        r"wjazdu?.*terytorium",
        r"udzielania.*ochrony",
        r"status.*uchodźc",
    ]),
    ("labor", [
        r"zatrudni\w+\s+cudzoziem",
        r"rynku?\s+pracy",
        r"Państwowej?\s+Inspekcji?\s+Pracy",
        r"Kodeks\s+pracy",
        r"stosunk.*pracy",
        r"czas\s+pracy",
        r"wynagrodzeni",
        r"związk.*zawodow",
        r"bezrobocie",
        r"pracę\s+zarobkow",
        r"zatrudnieniu?",
        r"umowy?\s+o\s+pracę",
        r"zwolnien.*grupow",
    ]),
    ("criminal", [
        r"Kodeks\s+karny",
        r"postępowania?\s+karnego?",
        r"przestępst",
        r"wykroczeni",
        r"\bkarny\b",
        r"prokuratura",
        r"aresztu?",
        r"karny\w*\s+skarbowy",
    ]),
    ("civil", [
        r"Kodeks\s+cywilny",
        r"zobowiązani\w+\s+cywilno",
        r"własno[śs]ci?\s+intelektualn",
        r"dziedziczen",
        r"\bspadk",
        r"rękojmi",
        r"umów?\s+cywilnoprawnych?",
        r"\bnajmu?\b",
        r"dzierżaw",
        r"odszkodowani.*cywil",
        r"sprzedaży?\s+nieruchom",
    ]),
    ("administrative", [
        r"postępowani.*administracyjn",
        r"Kodeks\s+postępowania\s+administracyjnego",
        r"\bKPA\b",
        r"administracji?\s+publicznej?",
        r"sąd.*administracyjn",
        r"postępowania?\s+egzekucyjn.*administrac",
    ]),
    ("tax", [
        r"podatk",                          # podatku, podatkiem, podatkowy, …
        r"\bVAT\b",
        r"\bPIT\b",
        r"\bCIT\b",
        r"\bakcyz",
        r"Ordynacja\s+podatkow",
        r"opodatkowania?",
        r"cen\s+transferowych?",
    ]),
    ("social", [
        r"ubezpieczen.*społeczn",
        r"\bZUS\b",
        r"świadczen.*społeczn",
        r"emerytur",
        r"\brenty?\b",
        r"\bzasiłk",
        r"rent.*socjaln",
        r"pomocy?\s+społecznej?",
        r"fundusz.*socjaln",
        r"świadczen.*rodzinnych?",
        r"zasiłk.*chorobow",
    ]),
    ("commercial", [
        r"spółk",
        r"handlow",
        r"działalnoś[śc].*gospodarcz",
        r"Kodeks\s+spółek\s+handlowych",
        r"rejestru?\s+przedsiębiorców?",
        r"koncesj",
        r"upadłoś[śc]",
        r"restrukturyzacj",
    ]),
    ("environmental", [
        r"ochron\w+\s+środowiska",          # ochrony, ochronie, ochrona środowiska
        r"gospodark.*odpadami",
        r"emisj.*gazów?",
        r"prawa?\s+wodnego?",
        r"geologiczn.*górniczego?",
        r"leśnictw",
    ]),
    ("constitutional", [
        r"Konstytucja\s+Rzeczypospolitej?",
        r"konstytucyjn",
        r"Trybunał\s+Konstytucyjny",
    ]),
]

_DOC_COMPILED: list[tuple[str, list[re.Pattern[str]]]] = [
    (domain, [re.compile(pat, re.IGNORECASE) for pat in pats])
    for domain, pats in _DOC_RULES
]


_VALID_DOMAINS: frozenset[str] = frozenset([
    "immigration", "labor", "criminal", "civil", "administrative",
    "tax", "social", "commercial", "environmental", "constitutional",
])

_LLM_CLASSIFY_PROMPT = """\
SYSTEM: You are a strict classifier. Output ONLY one word from the allowed list — no explanation, no punctuation, no extra text.

Polish legal act:
Title: {title}
Act type: {act_type}

Classify into EXACTLY one of these labels:
immigration, labor, criminal, civil, administrative, tax, social, commercial, environmental, constitutional

Output (one word only):"""


def _classify_llm(title: str, act_type: str) -> str:
    """Call DeepSeek to classify a document when rule-based patterns fail."""
    import json as _json
    import os
    import urllib.request

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "other"
    payload = _json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": _LLM_CLASSIFY_PROMPT.format(
            title=title, act_type=act_type,
        )}],
        "temperature": 0,
        "max_tokens": 5,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = _json.loads(resp.read())
        label = body["choices"][0]["message"]["content"].strip().lower().split()[0]
        return label if label in _VALID_DOMAINS else "other"
    except Exception:
        return "other"


def classify_law_domain(title: str, act_type: str = "") -> str:
    """Return the law domain for a Polish legal act given its title and act_type.

    Tries each domain in priority order (immigration first). Falls back to the
    LLM when no rule matches, so the result is never a generic 'other'.
    """
    text = f"{title} {act_type}"
    for domain, patterns in _DOC_COMPILED:
        if any(p.search(text) for p in patterns):
            return domain
    return _classify_llm(title, act_type)


# ---------------------------------------------------------------------------
# Query-level domain detection  (user question → retrieval filter)
# ---------------------------------------------------------------------------

# Conservative patterns — each must be strong enough that a single match
# justifies narrowing retrieval. Shared terms (e.g. "employment") that appear
# in multiple domains are intentionally left out; they would cause ambiguity
# and suppress the filter anyway.
_QUERY_DOMAIN_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:residence\s+permit|temporary\s+residence|permanent\s+residence"
            r"|karta\s+pobytu|zezwolenie\s+na\s+pobyt|pobyt\s+czasowy|pobyt\s+stały"
            r"|visa\s+application|wiza\b|cudzoziemiec\b"
            r"|PESEL\s+UKR|UKR\b|CUKR\b|MOS\b"
            r"|temporary\s+protection|ochrona\s+czasowa"
            r"|asylum\s+seeker|refugee\s+status|azyl\b"
            r"|repatriation|repatriacja"
            r"|Straż\s+Graniczna|border\s+crossing)\b",
            re.IGNORECASE,
        ),
        "immigration",
    ),
    (
        re.compile(
            r"\b(?:Kodeks\s+pracy|labour\s+law|labor\s+law"
            r"|umowa\s+o\s+pracę|czas\s+pracy|wynagrodzenie\b"
            r"|Inspekcja\s+Pracy|PIP\b|wypowiedzenie\s+umowy"
            r"|urlop\s+(?:macierzyński|wychowawczy|wypoczynkowy)"
            r"|zwolnienie\s+(?:z\s+pracy|grupowe))\b",
            re.IGNORECASE,
        ),
        "labor",
    ),
    (
        re.compile(
            r"\b(?:criminal\s+(?:law|charge|record|case)"
            r"|Kodeks\s+karny|postępowanie\s+karne"
            r"|przestępstwo\b|wykroczenie\b"
            r"|arrest|imprisonment|sentence|conviction)\b",
            re.IGNORECASE,
        ),
        "criminal",
    ),
    (
        re.compile(
            r"\b(?:Kodeks\s+cywilny|civil\s+(?:law|code|case|liability)"
            r"|rękojmia\b|wada\s+fizyczna|wada\s+prawna"
            r"|private\s+(?:sale|seller|buyer)"
            r"|used\s+car|defect.*(?:car|vehicle|goods?)"
            r"|(?:car|vehicle|goods?)\s+(?:defect|fault|problem)"
            r"|consumer\s+complaint\b|warranty\s+claim)\b",
            re.IGNORECASE,
        ),
        "civil",
    ),
    (
        re.compile(
            r"\b(?:income\s+tax|VAT\b|PIT\b|CIT\b"
            r"|Ordynacja\s+podatkowa|podatek\s+dochodowy"
            r"|tax\s+(?:return|relief|deduction|refund)"
            r"|podatkow\w+\s+(?:formularz|rozliczenie))\b",
            re.IGNORECASE,
        ),
        "tax",
    ),
    (
        re.compile(
            r"\b(?:ZUS\b|emerytura\b|renta\b|zasiłek\b"
            r"|ubezpieczenie\s+społeczne|social\s+security\s+contribution"
            r"|pension\s+(?:payment|calculation|eligibility)"
            r"|retirement\s+(?:age|benefit|fund))\b",
            re.IGNORECASE,
        ),
        "social",
    ),
    (
        re.compile(
            r"\b(?:Kodeks\s+spółek\s+handlowych|spółka\s+(?:z\s+o\.?o\.?|akcyjna|jawna)"
            r"|rejestr\s+przedsiębiorców|upadłość\b|restrukturyzacja\b"
            r"|company\s+(?:registration|liquidation|bankruptcy))\b",
            re.IGNORECASE,
        ),
        "commercial",
    ),
    (
        re.compile(
            r"\b(?:ochrona\s+środowiska|gospodarka\s+odpadami"
            r"|environmental\s+(?:law|permit|impact)"
            r"|emission\s+(?:standard|limit|permit))\b",
            re.IGNORECASE,
        ),
        "environmental",
    ),
]


# Polish BM25 boost terms injected into the retrieval query when a domain is
# detected — helps kNN and BM25 focus on the right corner of the corpus.
DOMAIN_QUERY_TERMS: dict[str, str] = {
    "immigration":    "cudzoziemiec pobyt zezwolenie wiza",
    "labor":          "zatrudnienie pracownik umowa praca",
    "criminal":       "przestępstwo kara postępowanie karne",
    "civil":          "zobowiązanie umowa cywilna odszkodowanie",
    "administrative": "postępowanie administracyjne organ decyzja",
    "tax":            "podatek opodatkowanie urząd skarbowy",
    "social":         "ubezpieczenie społeczne emerytura zasiłek",
    "commercial":     "spółka działalność gospodarcza rejestr",
    "environmental":  "ochrona środowiska emisja odpad",
    "constitutional": "konstytucja prawo podstawowe",
}


def detect_query_domain(text: str) -> str | None:
    """Detect the law domain of a user query for use as a retrieval filter.

    Returns the domain string when exactly ONE domain pattern matches —
    meaning the signal is unambiguous enough to narrow the corpus.
    Returns None when zero or multiple domains match so retrieval stays
    unrestricted and no relevant chunks are accidentally excluded.
    """
    matched: list[str] = []
    for pattern, domain in _QUERY_DOMAIN_RULES:
        if pattern.search(text) and domain not in matched:
            matched.append(domain)
    return matched[0] if len(matched) == 1 else None
