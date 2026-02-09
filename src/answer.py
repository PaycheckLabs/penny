# src/answer.py
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def _extract_output_text(resp: Any) -> str:
    """Best-effort extraction across OpenAI SDK response shapes."""
    # New Responses API shape
    if hasattr(resp, "output_text") and isinstance(getattr(resp, "output_text"), str):
        return resp.output_text.strip()

    # Chat Completions shape
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    # Fallback: try output[0].content[0].text
    try:
        out0 = resp.output[0]
        c0 = out0.content[0]
        if hasattr(c0, "text"):
            return str(c0.text).strip()
    except Exception:
        pass

    return ""


def openai_reply(
    chat_id: int,
    user_id: int,
    user_text: str,
    *,
    client: Any,
    openai_model: str,
    build_messages: Callable[[int, int, str], List[Dict[str, str]]],
    max_retries: int = 2,
) -> Tuple[str, Optional[str]]:
    """Call OpenAI and return (assistant_text, error_string_or_None)."""

    if not user_text:
        return "", None

    messages = build_messages(chat_id, user_id, user_text)

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            # Prefer the Responses API when available
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                resp = client.responses.create(
                    model=openai_model,
                    input=messages,
                )
                return _extract_output_text(resp), None

            # Fallback: Chat Completions
            resp = client.chat.completions.create(
                model=openai_model,
                messages=messages,
            )
            return _extract_output_text(resp), None

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return "", last_err

    return "", last_err
