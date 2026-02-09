#!/usr/bin/env python3
"""
Penny Telegram Bot (stable, minimal, and safe)

Key goals:
- Respond reliably in DMs, and in groups when invoked via /penny or @mention (or reply-to-bot).
- Do NOT talk about Paycheck Labs / Checks unless the user asks about it.
- When the user *does* ask about Paycheck Labs / Checks, attach synced docs context (via src/answer.py).
- Avoid blocking the asyncio event loop: OpenAI calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Local modules (in /src). Keep these imports simple so `python src/bot.py` works on Railway.
try:
    from answer import openai_reply as answer_openai_reply
except Exception:
    # Fallback for edge runtimes that run as a package (rare)
    from src.answer import openai_reply as answer_openai_reply  # type: ignore

try:
    from welcome_gate import is_allowed_user, register_welcome_gate_handlers
except Exception:
    from src.welcome_gate import is_allowed_user, register_welcome_gate_handlers  # type: ignore


logger = logging.getLogger("penny")

# ---------- Config ----------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("api_key") or ""
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "550"))

# Penny should be general by default. Product/company context is attached only when user asks.
BASE_SYSTEM_PROMPT = os.getenv(
    "PENNY_SYSTEM_PROMPT",
    "\n".join(
        [
            "You are Penny, a friendly, capable AI assistant.",
            "",
            "Default behavior:",
            "- Be general-purpose. Do NOT mention Paycheck Labs, Checks, or any products unless the user explicitly asks.",
            "- Keep replies clear and complete (finish sentences).",
            "- If the user greets you or asks small-talk, respond briefly and naturally.",
            "",
            "When the user asks about Paycheck Labs / Checks / the Checks whitepaper:",
            "- Use any provided 'Synced docs context' if present.",
            "- If the answer is not in the synced context, say so plainly rather than guessing.",
            "",
            "Safety and tone:",
            "- Be helpful, calm, and precise.",
            "- If something looks like sensitive info (keys, tokens, passwords), warn the user to rotate it and do not repeat it.",
        ]
    ),
)

# ---------- Small-talk detection (avoid calling OpenAI for simple greetings) ----------

GREETING_RE = re.compile(
    r"^\s*(?:@?\w+\s*)?"
    r"(?:hi|hello|hey|yo|gm|good\s+(?:morning|afternoon|evening)|sup|what'?s\s+up)"
    r"(?:\s+penny\b)?"
    r"[\s!?.]*$",
    re.IGNORECASE,
)
THANKS_RE = re.compile(r"^\s*(thanks|thank\s+you|thx|ty)\b[\s!?.]*$", re.IGNORECASE)
BYE_RE = re.compile(r"^\s*(bye|goodbye|later|cya|see\s+ya)\b[\s!?.]*$", re.IGNORECASE)
HOW_ARE_YOU_RE = re.compile(r"^\s*(hi|hello|hey)\b.*\bhow\s+are\s+you\b.*$", re.IGNORECASE)


def _is_small_talk(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) > 80:
        return False
    return bool(GREETING_RE.match(t) or THANKS_RE.match(t) or BYE_RE.match(t) or HOW_ARE_YOU_RE.match(t))


def _small_talk_reply(text: str) -> str:
    t = (text or "").lower()
    if THANKS_RE.match(t):
        return "You’re welcome. Want to keep going?"
    if BYE_RE.match(t):
        return "Catch you later."
    if HOW_ARE_YOU_RE.match(t):
        return "Doing great. How can I help you today?"
    return "Hey. How can I help?"


# ---------- Conversation memory (in-memory) ----------

@dataclass
class ConversationStore:
    max_turns: int = 12  # user+assistant pairs
    store: Dict[Tuple[int, int], List[Dict[str, str]]] = field(default_factory=dict)

    def get(self, chat_id: int, user_id: int) -> List[Dict[str, str]]:
        return list(self.store.get((chat_id, user_id), []))

    def append(self, chat_id: int, user_id: int, role: str, content: str) -> None:
        key = (chat_id, user_id)
        history = self.store.get(key, [])
        history.append({"role": role, "content": content})

        # keep last N turns (2 messages per turn)
        limit = self.max_turns * 2
        if len(history) > limit:
            history = history[-limit:]
        self.store[key] = history


MEMORY = ConversationStore(max_turns=int(os.getenv("PENNY_MAX_TURNS", "12")))


# ---------- Helpers ----------

def _strip_invocation(text: str, bot_username: Optional[str]) -> str:
    """Remove '/penny' prefix and @mentions to reduce prompt noise."""
    t = (text or "").strip()

    # Remove /penny at start
    if t.lower().startswith("/penny"):
        t = t[len("/penny") :].strip()

    # Remove @BotUsername mention
    if bot_username:
        mention = "@" + bot_username.lower()
        t = re.sub(rf"\s*{re.escape(mention)}\s*", " ", t, flags=re.IGNORECASE).strip()

    # Remove leading 'penny,' or 'penny:' in groups
    t = re.sub(r"^\s*penny\s*[:,\-]\s*", "", t, flags=re.IGNORECASE).strip()
    return t


async def _send_long(update: Update, text: str) -> None:
    """Telegram hard limit is ~4096 chars; keep safe."""
    if not update.message:
        return
    msg = text or ""
    if len(msg) <= 3500:
        await update.message.reply_text(msg)
        return

    chunk = []
    size = 0
    for line in msg.splitlines(keepends=True):
        if size + len(line) > 3500 and chunk:
            await update.message.reply_text("".join(chunk))
            chunk, size = [], 0
        chunk.append(line)
        size += len(line)
    if chunk:
        await update.message.reply_text("".join(chunk))


def _should_handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    msg = update.message
    if not msg or not msg.text:
        return False

    # Trigger 1: message is a reply to the bot
    if msg.reply_to_message and msg.reply_to_message.from_user and context.bot:
        try:
            if msg.reply_to_message.from_user.id == context.bot.id:
                return True
        except Exception:
            pass

    text_lower = msg.text.lower()

    # Trigger 2: explicit @mention
    try:
        username = (context.bot.username or "").lower()
    except Exception:
        username = ""
    if username and ("@" + username) in text_lower:
        return True

    # Trigger 3: /penny command is handled by CommandHandler
    return False


# ---------- Handlers ----------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Hey, I’m Penny. Ask me anything.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "\n".join(
            [
                "How to use Penny:",
                "- DM me with a question.",
                "- In groups: use /penny <your question> or @mention me.",
                "",
                "Tip: I won’t talk about Paycheck Labs / Checks unless you ask.",
            ]
        )
    )


async def cmd_penny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Group-friendly command: /penny <question>"""
    if not update.message:
        return

    if not is_allowed_user(update, context):
        # Gate messaging is handled by welcome_gate; keep this quiet.
        return

    user_text = " ".join(context.args or []).strip()
    if not user_text:
        await update.message.reply_text("Yep. What can I help with?")
        return

    await _handle_user_text(update, context, user_text)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.text:
        return

    if not is_allowed_user(update, context):
        return

    chat_type = msg.chat.type

    # Private chats: always handle.
    if chat_type == ChatType.PRIVATE:
        user_text = msg.text.strip()
        await _handle_user_text(update, context, user_text)
        return

    # Groups: handle only if invoked
    if chat_type in (ChatType.GROUP, ChatType.SUPERGROUP):
        if not _should_handle_group_message(update, context):
            return

        bot_username = (context.bot.username or "") if context.bot else ""
        user_text = _strip_invocation(msg.text, bot_username).strip()
        if not user_text:
            await msg.reply_text("Yep. What can I help with?")
            return
        await _handle_user_text(update, context, user_text)
        return


async def _handle_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE, raw_text: str) -> None:
    msg = update.message
    user = update.effective_user
    chat = update.effective_chat
    if not msg or not user or not chat:
        return

    bot_username = (context.bot.username or "") if context.bot else ""
    user_text = _strip_invocation(raw_text, bot_username).strip()

    # If it's simple small-talk, don't call OpenAI (and don't inject company context).
    if _is_small_talk(user_text):
        await _send_long(update, _small_talk_reply(user_text))
        return

    # Conversation history
    history = MEMORY.get(chat.id, user.id)
    MEMORY.append(chat.id, user.id, "user", user_text)

    # OpenAI call (off the event loop)
    try:
        client = context.bot_data.get("openai_client")
        if client is None:
            raise RuntimeError("OpenAI client not initialized")

        model = context.bot_data.get("openai_model", OPENAI_MODEL)
        max_tokens = context.bot_data.get("openai_max_output_tokens", OPENAI_MAX_OUTPUT_TOKENS)
        system_prompt = context.bot_data.get("system_prompt", BASE_SYSTEM_PROMPT)

        reply_text = await asyncio.to_thread(
            answer_openai_reply,
            client,
            system_prompt,
            history,
            user_text,
            model,
            max_tokens,
        )

        reply_text = (reply_text or "").strip()
        if not reply_text:
            reply_text = "I didn’t get a response back. Try again."

    except Exception as e:
        logger.exception("OpenAI reply failed: %s", e)
        reply_text = "Oops. I hit a technical issue. Please try again in a moment."

    MEMORY.append(chat.id, user.id, "assistant", reply_text)
    await _send_long(update, reply_text)


# ---------- Startup ----------

def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=35.0)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # store shared config
    app.bot_data["openai_client"] = openai_client
    app.bot_data["openai_model"] = OPENAI_MODEL
    app.bot_data["openai_max_output_tokens"] = OPENAI_MAX_OUTPUT_TOKENS
    app.bot_data["system_prompt"] = BASE_SYSTEM_PROMPT

    # core commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("penny", cmd_penny))

    # welcome gate (optional)
    register_welcome_gate_handlers(app)

    # text handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Penny starting…")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
