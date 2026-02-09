import os
from typing import Optional, List, Dict

from openai import OpenAI

from src.checks_rag import retrieve_checks_context, format_context_for_prompt

client = OpenAI()

PENNY_MODEL = os.getenv("PENNY_MODEL", "gpt-4o-mini")
CHECKS_TOP_K = int(os.getenv("CHECKS_TOP_K", "6"))


def _system_prompt() -> str:
    return (
        "You are Penny, the official AI advisor for Paycheck Labs.\n\n"
        "If the question is about Checks, the Checks Platform, NFT Checks, Check Token ($CHECK), "
        "tokenomics, MVP features, or anything that sounds like the Checks whitepaper, use the "
        "WHITEPAPER CONTEXT provided.\n\n"
        "Rules:\n"
        "- Answer clearly and directly.\n"
        "- If you use facts from the context, cite them like [WP1], [WP2], etc.\n"
        "- If the context does not contain the answer, say you're not sure and ask 1 short follow-up question.\n"
        "- Do not paste long excerpts from the whitepaper.\n"
    )


def generate_penny_reply(user_text: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    chunks = retrieve_checks_context(user_text, top_k=CHECKS_TOP_K)
    whitepaper_context = format_context_for_prompt(chunks)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt()},
    ]

    if chat_history:
        for m in chat_history[-6:]:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": (
                f"WHITEPAPER CONTEXT:\n{whitepaper_context}\n\n"
                f"USER QUESTION:\n{user_text}\n"
            ),
        }
    )

    resp = client.chat.completions.create(
        model=PENNY_MODEL,
        messages=messages,
        temperature=float(os.getenv("PENNY_TEMPERATURE", "0.4")),
    )

    return resp.choices[0].message.content.strip()
