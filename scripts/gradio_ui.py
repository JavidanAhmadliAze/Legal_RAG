"""
Gradio chat UI — left sidebar with conversation history, right panel with chat.

Run:  .venv/bin/python scripts/gradio_ui.py
Open: http://localhost:7860
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
_CLIENT = httpx.Client(timeout=300)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _fetch_conversations(user_id: str) -> list[dict]:
    try:
        r = _CLIENT.get(f"{API_BASE}/v1/conversations/{user_id}")
        r.raise_for_status()
        return r.json()["conversations"]
    except Exception:
        return []


def _fetch_messages(user_id: str, conversation_id: str) -> list[dict]:
    try:
        r = _CLIENT.get(f"{API_BASE}/v1/conversations/{user_id}/{conversation_id}/messages")
        r.raise_for_status()
        return r.json()["messages"]
    except Exception:
        return []


def _post_chat(user_id: str, message: str, conversation_id: str | None) -> dict:
    payload = {"user_id": user_id, "message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    try:
        r = _CLIENT.post(f"{API_BASE}/v1/chat", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"message": f"❌ Server error {e.response.status_code}: {e.response.text}", "conversation_id": conversation_id}
    except httpx.RequestError as e:
        return {"message": f"❌ Cannot reach API: {e}", "conversation_id": conversation_id}


# ---------------------------------------------------------------------------
# UI event handlers
# ---------------------------------------------------------------------------

def _build_choices(convs: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Return (radio_choices, label→thread_id mapping)."""
    mapping: dict[str, str] = {}
    choices: list[str] = []
    for c in convs:
        date = (c.get("updated_at") or "")[:10]
        title = (c.get("title") or "Untitled")[:50]
        label = f"{date}  {title}"
        # deduplicate
        base, i = label, 2
        while label in mapping:
            label = f"{base} ({i})"
            i += 1
        mapping[label] = c["conversation_id"]
        choices.append(label)
    return choices, mapping


def load_history(user_id: str):
    """Load conversation list for a user."""
    if not user_id.strip():
        return gr.update(choices=[], value=None), {}
    convs = _fetch_conversations(user_id.strip())
    choices, mapping = _build_choices(convs)
    return gr.update(choices=choices, value=None), mapping


def select_conversation(label: str, user_id: str, mapping: dict):
    """Load messages for the selected conversation into the chatbot."""
    if not label or label not in mapping:
        return [], None
    thread_id = mapping[label]
    msgs = _fetch_messages(user_id.strip(), thread_id)
    return msgs, thread_id


def send_message(
    message: str,
    history: list[dict],
    user_id: str,
    conversation_id: str | None,
    mapping: dict,
):
    if not message.strip():
        return history, conversation_id, "", mapping

    if not user_id.strip():
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please enter a User ID first."},
        ]
        return history, conversation_id, "", mapping

    data = _post_chat(user_id.strip(), message.strip(), conversation_id)
    new_conv_id = data.get("conversation_id", conversation_id)

    history = history + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": data["message"]},
    ]

    # Refresh sidebar so new conversation appears immediately
    convs = _fetch_conversations(user_id.strip())
    choices, new_mapping = _build_choices(convs)

    return history, new_conv_id, "", gr.update(choices=choices, value=None), new_mapping


def new_conversation():
    return [], None, ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Polish Legal Assistant") as demo:
    conv_map_state = gr.State({})

    with gr.Row():
        # ── Left sidebar ──────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### Polish Legal Assistant")
            user_id_box = gr.Textbox(
                label="User ID",
                placeholder="Enter your ID, e.g. alice",
            )
            load_btn = gr.Button("Load history", variant="secondary", size="sm")

            conv_radio = gr.Radio(
                label="Conversations",
                choices=[],
                interactive=True,
            )
            new_btn = gr.Button("＋ New conversation", variant="primary", size="sm")

        # ── Chat area ─────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                height=560,
                show_label=False,
                buttons=["copy"],
            )
            conversation_id_state = gr.State(None)

            with gr.Row():
                msg_box = gr.Textbox(
                    show_label=False,
                    placeholder="Ask about Polish immigration law…",
                    scale=9,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    # ── Wiring ────────────────────────────────────────────────────────────
    load_btn.click(
        load_history,
        inputs=[user_id_box],
        outputs=[conv_radio, conv_map_state],
    )
    # also load on Enter in the user_id field
    user_id_box.submit(
        load_history,
        inputs=[user_id_box],
        outputs=[conv_radio, conv_map_state],
    )

    conv_radio.change(
        select_conversation,
        inputs=[conv_radio, user_id_box, conv_map_state],
        outputs=[chatbot, conversation_id_state],
    )

    send_btn.click(
        send_message,
        inputs=[msg_box, chatbot, user_id_box, conversation_id_state, conv_map_state],
        outputs=[chatbot, conversation_id_state, msg_box, conv_radio, conv_map_state],
    )
    msg_box.submit(
        send_message,
        inputs=[msg_box, chatbot, user_id_box, conversation_id_state, conv_map_state],
        outputs=[chatbot, conversation_id_state, msg_box, conv_radio, conv_map_state],
    )

    new_btn.click(
        new_conversation,
        outputs=[chatbot, conversation_id_state, msg_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
