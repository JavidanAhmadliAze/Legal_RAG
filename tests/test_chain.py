from src.rag.chain import _split_into_subquestions


def test_split_into_subquestions_prefers_substantive_legal_sentences():
    question = (
        "Hi! I just started a small consulting business in Poland. "
        "I'm hiring my first employee next week - a foreigner who already has a residence card. "
        "My accountant mentioned that as of 2026, there's a new mandatory reporting deadline before they even start work. "
        "Is that true? "
        "Also, I heard that if I make a mistake in the digital reporting or if the PIP inspector decides it's not a real "
        "B2B contract, there's an automatic fine of 3,000 to 5,000 PLN and they can force a contract change immediately. "
        "Can you clarify these 2026 employer rules?"
    )

    parts = _split_into_subquestions(question)

    assert len(parts) == 3
    assert parts[0].startswith("I'm hiring my first employee next week")
    assert "mandatory reporting deadline" in parts[1]
    assert "Is that true?" in parts[1]
    assert "PIP inspector" in parts[2]
    assert "B2B contract" in parts[2]
