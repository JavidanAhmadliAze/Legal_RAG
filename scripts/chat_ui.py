#!/usr/bin/env python3
"""
Simple Gradio chat UI for the Polish legal RAG assistant.

Usage:
    DEEPSEEK_API_KEY=sk-... python scripts/chat_ui.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import gradio as gr
from src.monitoring.tracing import init as init_tracing
from src.rag.chain import build_chain
from src.rag.guardrail import check as guardrail_check

init_tracing(project_name="legal-rag", port=6006)
chain = build_chain()


def respond(message: str, history: list[dict]):
    result = guardrail_check(message)
    if not result.allowed:
        yield result.rejection_message
        return

    partial = ""
    for chunk in chain.stream(message):
        partial += chunk
        yield partial


demo = gr.ChatInterface(
    fn=respond,
    title="Polish Legal Assistant",
    description="Ask questions about Polish immigration and foreigners law (cudzoziemcy, repatriacja, obywatelstwo).",
    examples=[
        "What are the conditions for obtaining a repatriation visa?",
        "Who can apply for a Karta Polaka?",
        "What rights does a repatriate have after arriving in Poland?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
