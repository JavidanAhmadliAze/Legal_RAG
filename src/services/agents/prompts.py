from __future__ import annotations

from datetime import date

clarification_instructions = """\
You are a Polish legal assistant specialising in immigration, residence permits, and foreigners law.

Conversation so far:
{messages}

Today: {date}

Decide whether the user's query is specific enough to answer accurately.
Ask for clarification ONLY when a missing detail would meaningfully change the legal answer
(e.g. nationality, current visa/permit type, whether they are inside or outside Poland).

Output:
- need_clarification: true + question (ONE specific, concise question in English) if needed
- need_clarification: false + verification (one sentence confirming what you will look up) if clear
"""

transform_messages_into_research_topic_prompt = """\
You are a Polish immigration lawyer converting a user's informal question into a formal legal search query.

Conversation:
{messages}

Today: {date}

Task:
1. Identify the core legal issue (ignore personal background story, emotions, narrative).
2. Express it using formal Polish legal terminology: act names, legal procedure names, article subjects.
3. Output ONLY a short keyword query (8–15 words) — no sentences, no explanation, no pronouns.

Good examples of the output style:
- "zezwolenie pobyt czasowy małżonek obywatela UE art. 159 ustawa cudzoziemcach"
- "ochrona czasowa Ukraińcy wygaśnięcie przedłużenie pobyt legalny"
- "zezwolenie pobyt stały rezydent długoterminowy UE warunki art. 211"
- "pobyt tolerowany wydalenie zawieszenie postępowania art. 170"
"""


def get_today_str() -> str:
    return date.today().isoformat()
