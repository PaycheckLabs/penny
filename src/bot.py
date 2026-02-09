# NOTE: This file is intentionally "single-file" friendly for Railway deployments.
# Only minimal additions were made to inject synced Checks docs into the prompt.

import os
import re
import time
import json
import logging
import signal
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# -------------------------
# Logging
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("penny")

# -------------------------
# Env
# -------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("api_key", os.getenv("OPENAI", ""))).strip()

MODEL = os.getenv("MODEL", "gpt-5-nano").strip()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "280"))

# Optional: allow only one admin/user, or allow all
ALLOWED_TELEGRAM_USER_ID = os.getenv("ALLOWED_TELEGRAM_USER_ID", "").strip()

# Optional: Slash command trigger that "wakes" Penny in group chats
BOT_TRIGGER = os.getenv("BOT_TRIGGER", "/penny").strip()

# Optional
BOT_NAME = os.getenv("BOT_NAME", "Penny").strip()

# -------------------------
# Checks Whitepaper Docs (synced by GitHub Actions)
# -------------------------
# Expected landing path (per workflow): /app/src/data/checks_whitepaper on Railway
CHECKS_DOCS_DIR = os.getenv("CHECKS_DOCS_DIR", "").strip()


def _candidate_checks_docs_dirs() -> List[Path]:
    """
    Return plausible locations for the synced Checks docs, in order of preference.
    This is intentionally defensive because different deploy targets may have different working dirs.
    """
    here = Path(__file__).resolve()
    src_dir = here.parent  # .../src
    repo_root = src_dir.parent

    candidates: List[Path] = []

    # 1) Explicit env var, if provided
    if CHECKS_DOCS_DIR:
        candidates.append(Path(CHECKS_DOCS_DIR))

    # 2) Expected path in this repo layout
    candidates.append(src_dir / "data" / "checks_whitepaper")
    candidates.append(repo_root / "src" / "data" / "checks_whitepaper")

    # 3) Common alternates if someone changed the sync script
    candidates.append(repo_root / "docs" / "checks_whitepaper")
    candidates.append(repo_root / "data" / "checks_whitepaper")
    candidates.append(repo_root / "checks_whitepaper")

    # Normalize / de-dupe while preserving order
    seen = set()
    out: List[Path] = []
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            rp = p
        if str(rp) not in seen:
            seen.add(str(rp))
            out.append(rp)
    return out


def _read_markdown_files(root: Path, max_files: int = 200) -> List[Tuple[str, str]]:
    """
    Read markdown-ish files under root.
    Returns list of (relative_path, text).
    """
    items: List[Tuple[str, str]] = []
    if not root.exists() or not root.is_dir():
        return items

    count = 0
    for path in sorted(root.rglob("*")):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".markdown", ".mdx", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(root))
        items.append((rel, text))
        count += 1
    return items


def _score_doc_text(query_terms: List[str], text: str) -> int:
    """
    Tiny keyword overlap scorer. Fast and dependency-free.
    """
    if not query_terms or not text:
        return 0
    t = text.lower()
    score = 0
    for term in query_terms:
        if not term:
            continue
        # Favor exact term hits; count a few occurrences but cap to avoid long-doc dominance.
        hits = t.count(term)
        if hits:
            score += min(hits, 6)
    return score


def get_checks_docs_context(user_text: str, char_budget: int = 6000) -> Tuple[str, str]:
    """
    Returns (chosen_root, context_text). If not found, returns ("", "").
    Context is assembled from the most relevant markdown files.
    """
    user_text_l = (user_text or "").lower()

    # If the user isn't asking about Checks, bail early.
    checks_triggers = [
        "checks",
        "check token",
        "$check",
        "nft check",
        "checks platform",
        "whitepaper",
        "roadmap",
        "post-mvp",
        "auto-invest",
        "autoinvest",
        "auto investment",
    ]
    if not any(t in user_text_l for t in checks_triggers):
        return ("", "")

    # Basic term list for scoring
    # Keep short terms out to reduce noise
    raw_terms = re.findall(r"[a-z0-9\-]{3,}", user_text_l)
    # Add a couple normalized variants
    raw_terms += ["post-mvp", "postmvp", "auto-invest", "autoinvest", "auto", "investment", "roadmap"]
    query_terms = sorted(set(raw_terms))

    for root in _candidate_checks_docs_dirs():
        files = _read_markdown_files(root)
        if not files:
            continue

        ranked: List[Tuple[int, str, str]] = []
        for rel, txt in files:
            s = _score_doc_text(query_terms, txt)
            if s > 0:
                ranked.append((s, rel, txt))

        # If nothing scored, still provide a tiny index hint (so the model knows docs exist)
        ranked.sort(key=lambda x: x[0], reverse=True)

        parts: List[str] = []
        used = 0

        if ranked:
            for s, rel, txt in ranked[:8]:
                snippet = txt.strip()
                # Light trimming
                if len(snippet) > 1800:
                    snippet = snippet[:1800] + "\n…"
                block = f"FILE: {rel}\n{snippet}"
                if used + len(block) + 2 > char_budget:
                    break
                parts.append(block)
                used += len(block) + 2
        else:
            # No keyword matches; include a short manifest of available files
            manifest = "\n".join([f"- {rel}" for rel, _ in files[:40]])
            parts.append("No strong keyword match. Available files (partial):\n" + manifest)

        context = "\n\n".join(parts).strip()
        return (str(root), context)

    return ("", "")


# -------------------------
# OpenAI
# -------------------------
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY not set. Penny will not be able to answer.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------
# Conversation state
# -------------------------
# Keep a short rolling buffer per chat.
CHAT_HISTORY: Dict[int, Deque[Tuple[str, str]]] = {}
MAX_TURNS = 12

# -------------------------
# Base system prompt
# -------------------------
SYSTEM_PROMPT = """You are Penny, the official AI advisor for Paycheck Labs.

You are:
- Calm, precise, and helpful.
- Protective: you do not hallucinate. If you don't know, say so.
- Practical: you provide actionable steps, not vague statements.

Behavior rules:
- If asked about Paycheck Labs, Checks Platform, or Check Token, answer directly and accurately.
- If not asked, do not volunteer company internal details.
- If you have authoritative docs context provided in the conversation, treat it as the source of truth.
- Keep answers concise unless the user asks for more detail.
"""

# Optional: extra system context (kept empty by default)
SYSTEM_CONTEXT = os.getenv("SYSTEM_CONTEXT", "").strip()


def get_chat_history(chat_id: int) -> Deque[Tuple[str, str]]:
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = deque(maxlen=MAX_TURNS * 2)  # each turn is user+assistant
    return CHAT_HISTORY[chat_id]


def user_is_allowed(update: Update) -> bool:
    if not ALLOWED_TELEGRAM_USER_ID:
        return True
    try:
        return str(update.effective_user.id) == str(ALLOWED_TELEGRAM_USER_ID)
    except Exception:
        return False


def should_respond_in_chat(text: str) -> bool:
    """
    In group chats, only respond when the message starts with /penny (or whatever BOT_TRIGGER is).
    In DMs, respond to everything.
    """
    if not text:
        return False
    t = text.strip()
    return t.lower().startswith(BOT_TRIGGER.lower())


def strip_trigger(text: str) -> str:
    t = (text or "").strip()
    if t.lower().startswith(BOT_TRIGGER.lower()):
        return t[len(BOT_TRIGGER):].strip()
    return t


def build_messages(system_prompt: str, conversation: List[Tuple[str, str]], user_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

    system_context = SYSTEM_CONTEXT
    if system_context:
        messages.append({
            "role": "system",
            "content": system_context,
        })

    # If relevant, inject synced Checks whitepaper/docs context (read-only excerpts)
    checks_root, checks_ctx = get_checks_docs_context(user_text)
    if checks_ctx:
        messages.append({
            "role": "system",
            "content": (
                "Checks Docs Context (synced markdown excerpts; treat as authoritative within this bot)\n"
                f"Docs root: {checks_root}\n\n"
                f"{checks_ctx}"
            ),
        })

    # Add prior conversation (trimmed)
    for role, content in conversation[-(MAX_TURNS * 2):]:
        messages.append({
            "role": role,
            "content": content,
        })

    # Current user message
    messages.append({
        "role": "user",
        "content": user_text,
    })

    return messages


def openai_reply(conversation: List[Tuple[str, str]], user_text: str) -> str:
    if not client:
        return "OpenAI is not configured (missing OPENAI_API_KEY)."

    messages = build_messages(SYSTEM_PROMPT, conversation, user_text)

    try:
        resp = client.responses.create(
            model=MODEL,
            input=messages,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        log.exception("OpenAI call failed")
        return f"I hit an error calling the model. Check Railway logs and your OPENAI settings.\n\nError: {type(e).__name__}"


# -------------------------
# Telegram handlers
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not user_is_allowed(update):
        return
    await update.message.reply_text(
        f"Hi — I’m {BOT_NAME}. Ask me anything, or use `{BOT_TRIGGER} <question>` in group chats.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not user_is_allowed(update):
        return
    await update.message.reply_text(
        "Commands:\n"
        "/start — intro\n"
        "/help — this help\n\n"
        "In group chats, use:\n"
        f"{BOT_TRIGGER} <your question>",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not user_is_allowed(update):
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    # If group chat, require trigger; if DM, respond normally.
    is_group = update.effective_chat and update.effective_chat.type in ("group", "supergroup")

    if is_group:
        if not should_respond_in_chat(text):
            return
        user_text = strip_trigger(text)
        if not user_text:
            return
    else:
        user_text = text

    chat_id = update.effective_chat.id
    history = get_chat_history(chat_id)

    # Add user message to history
    history.append(("user", user_text))

    await update.message.chat.send_action(action=ChatAction.TYPING)

    reply = openai_reply(list(history), user_text)

    # Add assistant reply to history
    history.append(("assistant", reply))

    await update.message.reply_text(reply)


def _build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


def main() -> None:
    app = _build_app()
    log.info("Starting Penny Telegram bot...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
