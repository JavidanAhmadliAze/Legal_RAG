"""Shared DeepSeek client for LLM-judge evaluators."""

from __future__ import annotations

import os
import re

from openai import OpenAI

_client: OpenAI | None = None

PASS_THRESHOLD = 3  # scores >= 3 on 1–5 scale are Pass


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
    return _client


def judge(system_prompt: str, user_prompt: str) -> tuple[int, str]:
    """Call DeepSeek with a judge prompt; return (score 1-5, reasoning)."""
    response = _get_client().chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    # Extract the first integer 1-5 from the response
    match = re.search(r"\b([1-5])\b", content)
    score = int(match.group(1)) if match else 1
    return score, content
