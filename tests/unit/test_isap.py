from src.sources.isap import ActMeta, filter_foreign


def _make(title: str) -> ActMeta:
    return ActMeta(
        address="WDU20250000001",
        journal="DU",
        title=title,
        act_type="Ustawa",
        year=2025,
        pos=1,
        status="obowiązujący",
        announcement_date="2025-01-01",
        has_pdf=True,
    )


def test_filter_foreign_matches_repatriation_act_title():
    acts = [_make("Obwieszczenie Marszałka Sejmu Rzeczypospolitej Polskiej w sprawie ogłoszenia jednolitego tekstu ustawy o repatriacji")]
    assert len(filter_foreign(acts)) == 1


def test_filter_foreign_matches_repatriation_amendment_title():
    acts = [_make("Ustawa z dnia 25 czerwca 2025 r. o zmianie ustawy o repatriacji oraz niektórych innych ustaw")]
    assert len(filter_foreign(acts)) == 1


def test_filter_foreign_matches_karta_polaka_case_insensitively():
    acts = [_make("USTAWA O KARCIE POLAKA")]
    assert len(filter_foreign(acts)) == 1


def test_pdf_url_format():
    act = _make("irrelevant")
    act.pos = 1746
    act.year = 2025
    assert act.pdf_url == "https://api.sejm.gov.pl/eli/acts/DU/2025/1746/text.pdf"


def test_pdf_url_format_for_monitor_polski():
    act = _make("irrelevant")
    act.journal = "MP"
    act.pos = 370
    act.year = 2026
    assert act.pdf_url == "https://api.sejm.gov.pl/eli/acts/MP/2026/370/text.pdf"
