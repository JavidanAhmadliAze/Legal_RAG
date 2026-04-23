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
Based on the conversation below, write a short keyword search query for finding relevant Polish legal texts.

Conversation:
{messages}

Today: {date}

Output ONLY a short query (max 20 words). Use Polish legal terms where possible (e.g. ochrona czasowa,
pobyt czasowy, PESEL UKR, specustawa, zezwolenie na pobyt). No full sentences, no explanation.
"""


def get_today_str() -> str:
    return date.today().isoformat()
