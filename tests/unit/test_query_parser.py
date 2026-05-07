from src.services.agents.nodes.query_parser import parse_query


def test_parse_query_preserves_english_temporary_residence_phrase():
    parsed = parse_query("Can I mail a temporary residence permit application?")
    assert "zezwolenie na pobyt czasowy" in parsed.preserved


def test_parse_query_preserves_pesel_ukr_and_ukraine():
    parsed = parse_query("If I arrived from Ukraine last month, how long do I have to register for a PESEL UKR?")
    assert "PESEL" in parsed.preserved
    assert "UKR" in parsed.preserved
    assert "PESEL UKR" in parsed.preserved
    assert "Ukraina" in parsed.preserved


def test_parse_query_augment_adds_missing_preserved_terms():
    parsed = parse_query("What are the rules for a repatriation visa?")
    augmented = parsed.augment("warunki wizy krajowej")
    assert "wiza w celu repatriacji" in augmented


def test_parse_query_preserves_employer_reporting_terms():
    parsed = parse_query(
        "Is there a reporting deadline for hiring a foreigner with a residence card, "
        "and can PIP reclassify a B2B contract?"
    )
    assert "karta pobytu" in parsed.preserved
    assert "powiadomienie o powierzeniu pracy" in parsed.preserved
    assert "Państwowa Inspekcja Pracy" in parsed.preserved
    assert "umowa cywilnoprawna" in parsed.preserved
    assert "stosunek pracy" in parsed.preserved
