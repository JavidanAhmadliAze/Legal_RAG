"""
Shared helpers used by multiple nodes:
  - _get_llm()                -> per-call non-streaming chat model
  - _get_translate_chain()    -> cached prompt | chat | parser pipe
  - _ainvoke_structured()     -> async structured-output wrapper that uses
                                 function calling (DeepSeek doesn't support
                                 the json_schema response_format)
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.services.agents._common import _TRANSLATE_PROMPT
from src.services.llm import get_llm_client


_translate_chain: object = None


def _get_translate_chain():
    global _translate_chain
    if _translate_chain is None:
        _translate_chain = (
            _TRANSLATE_PROMPT
            | get_llm_client().build_chat_model(streaming=False, request_timeout=30)
            | StrOutputParser()
        )
    return _translate_chain


def _get_llm() -> BaseChatModel:
    return get_llm_client().build_chat_model(streaming=False, request_timeout=90)


async def _ainvoke_structured(model: BaseChatModel, schema, messages: list):
    return await model.with_structured_output(
        schema, method="function_calling"
    ).ainvoke(messages)
