from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    message: str
    # UUID string (thread_id). Omit to start a new conversation.
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    message: str
    conversation_id: str   # UUID thread_id — pass this back to continue the conversation
    needs_clarification: bool


class ConversationOut(BaseModel):
    conversation_id: str   # UUID thread_id
    title: str | None
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]
