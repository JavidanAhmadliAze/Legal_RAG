from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationOut,
)
from src.db.models import Conversation, User
from src.db.session import get_session

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatResponse:
    graph = request.app.state.graph

    # Ensure user row exists (auto-create on first message)
    user = await session.get(User, body.user_id)
    if user is None:
        user = User(user_id=body.user_id)
        session.add(user)

    # Resolve or create conversation
    if body.conversation_id:
        conv = await session.scalar(
            select(Conversation).where(
                Conversation.thread_id == body.conversation_id,
                Conversation.user_id == body.user_id,
            )
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        thread_id = str(uuid.uuid4())
        conv = Conversation(
            user_id=body.user_id,
            thread_id=thread_id,
            title=body.message[:80],
        )
        session.add(conv)

    await session.commit()

    config = {"configurable": {"thread_id": conv.thread_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=body.message)]},
        config=config,
    )

    # Bump updated_at on every turn
    conv.updated_at = datetime.now(timezone.utc)
    await session.commit()

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    answer = ai_messages[-1].content if ai_messages else ""

    return ChatResponse(
        message=answer,
        conversation_id=conv.thread_id,
        needs_clarification=result.get("needs_clarification", False),
    )


@router.get("/conversations/{user_id}/{conversation_id}/messages")
async def get_messages(
    user_id: str,
    conversation_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    conv = await session.scalar(
        select(Conversation).where(
            Conversation.thread_id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    graph = request.app.state.graph
    state = await graph.aget_state({"configurable": {"thread_id": conversation_id}})
    raw = state.values.get("messages", []) if state and state.values else []

    return {
        "messages": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in raw
            if isinstance(m, (HumanMessage, AIMessage))
        ]
    }


@router.get("/conversations/{user_id}", response_model=ConversationListResponse)
async def list_conversations(
    user_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConversationListResponse:
    rows = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return ConversationListResponse(
        conversations=[
            ConversationOut(
                conversation_id=c.thread_id,
                title=c.title,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
            )
            for c in rows.all()
        ]
    )
