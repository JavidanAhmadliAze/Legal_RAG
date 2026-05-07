from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from src.services.config import LLMSettings, get_llm_settings


class LLMClient:
    def __init__(
        self,
        *,
        settings: LLMSettings | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._settings = settings or get_llm_settings()
        self._model = model or self._settings.model
        self._base_url = base_url or self._settings.base_url
        self._llm: ChatOpenAI | None = None
        self._translator: ChatOpenAI | None = None

    def _api_key(self) -> str:
        return os.environ[self._settings.api_key_env_var]

    def get_chat_model(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self._model,
                api_key=self._api_key(),
                base_url=self._base_url,
                streaming=self._settings.streaming,
                temperature=self._settings.temperature,
                request_timeout=self._settings.request_timeout_s,
            )
        return self._llm

    def get_translator_model(self) -> ChatOpenAI:
        if self._translator is None:
            self._translator = ChatOpenAI(
                model=self._model,
                api_key=self._api_key(),
                base_url=self._base_url,
                streaming=self._settings.translator_streaming,
                temperature=self._settings.temperature,
                request_timeout=self._settings.request_timeout_s,
            )
        return self._translator

    def build_chat_model(
        self,
        *,
        streaming: bool,
        request_timeout: float,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._model,
            api_key=self._api_key(),
            base_url=self._base_url,
            streaming=streaming,
            temperature=self._settings.temperature,
            request_timeout=request_timeout,
        )
