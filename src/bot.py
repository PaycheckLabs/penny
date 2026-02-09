# src/bot.py
import os
import re
import time
import json
import logging
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

# Existing KB helper
from knowledge_base import build_kb_context

# NEW: Docs excerpt helper (synced whitepaper/docs)
from answer import build_checks_context

logger = logging.getLogger("penny")
logging.basicConfig(level=logging.INFO)

# ----------------------------
# ENV
# ----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BOT_USERNAME = os.getenv("BOT_USERNAME", "").lstrip("@")
BOT_TRIGGER = os.getenv("BOT_TRIGGER", "penny").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or os.getenv("api_key", "") or os.getenv("OPENAI", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "500"))

# Admin controls
ADMIN_IDS = set()
_raw_admin = os.getenv("ADMIN_IDS", "").strip()
if _raw_admin:
    for part in _raw_admin.split(","):
        part = part.strip()
        if part.isdigit():
            ADMIN_IDS.add(int(part))

# Testing group target (optional)
TESTING_GROUP_ID = os.getenv("TESTING_GROUP_ID", "").strip()
if TESTING_GROUP_ID and TESTING_GROUP_ID.lstrip("-").isdigit():
    TESTING_GROUP_ID = int(TESTING_GROUP_ID)
else:
    TESTING_GROUP_ID = None

# NEW: Where the synced docs live (comma-separated, relative to repo root)
# From your screenshots, docs/ contains the synced markdown.
CHECKS_DOCS_DIRS = os.getenv("CHECKS_DOCS_DIRS", "docs,src/data/checks_whitepaper,src/data/checks_whitepaper/docs")
CHECKS_DOCS_MAX_CHARS = int(os.getenv("CHECKS_DOCS_MAX_CHARS", "6500"))

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# System prompt
# ----------------------------
SYSTEM_PROMPT = (
    "You are Penny, the official AI advisor for Paycheck Labs.\n"
    "You provide product guidance, community operations support, and educational explanations.\n"
    "Tone: calm, precise, protective. Prioritize clarity and correct process over hype.\n"
    "If you are unsure, say so and suggest the next best step.\n"
)

# ----------------------------
# Memory (simple in-process)
# ----------------------------
MAX_MEMORY_TURNS = 6
_memory: Dict[str, List[Dict[str, str]]] = {}

def _mem_key(chat_id: int, user_id: int) -> str:
    return f"{chat_id}:{user_id}"

def get_memory(chat_id: int, user_id: int) -> List[Dict[str, str]]:
    return _memory.get(_mem_key(chat_id, user_id), [])[-(MAX_MEMORY_TURNS * 2):]

def add_memory(chat_id: int, user_id: int, role: str, content: str) -> None:
    key = _mem_key(chat_id, user_id)
    _memory.setdefault(key, []).append({"role": role, "content": content})
    _memory[key] = _memory[key][-(MAX_MEMORY_TURNS * 2):]

# ----------------------------
# Utilities
# ----------------------------
def extract_output_text(resp) -> str:
    """
    Robust extraction for Responses API.
    """
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
    except Exception:
        pass

    try:
        out = []
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        out.append(getattr(c, "text", ""))
        text = "\n".join([t for t in out if t]).strip()
        return text
    except Exception:
        return ""

def should_respond(message_text: str, is_reply_to_bot: bool) -> bool:
    """
    In groups: respond if trigger word, @mention, or reply-to-bot.
    In DMs: always respond.
    """
    if not message_text:
        return False

    txt = message_text.strip()
    low = txt.lower()

    if is_reply_to_bot:
        return True

    if BOT_USERNAME and f"@{BOT_USERNAME.lower()}" in low:
        return True

    if low.startswith(BOT_TRIGGER + " "):
        return True
    if low == BOT_TRIGGER:
        return True

    return False

def strip_trigger_prefix(message_text: str) -> str:
    txt = message_text.strip()
    low = txt.lower()
    if low.startswith(BOT_TRIGGER + " "):
        return txt[len(BOT_TRIGGER) + 1 :].strip()
    if low == BOT_TRIGGER:
        return ""
    return txt

# ----------------------------
# Message builder (CORE)
# ----------------------------
def build_messages(chat_id: int, user_id: int, user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "developer", "content": SYSTEM_PROMPT}]

    # Existing: inject KB context (only when relevant)
    kb_context = build_kb_context(user_text, max_sections=3)
    if kb_context:
        msgs.append({"role": "developer", "content": kb_context})

    # NEW: inject Checks docs excerpts (only when relevant)
    # This is what makes "Auto-Investment" and other roadmap features discoverable.
    dirs = [d.strip() for d in CHECKS_DOCS_DIRS.split(",") if d.strip()]
    docs_context = build_checks_context(
        user_text=user_text,
        dirs=dirs,
        max_chars=CHECKS_DOCS_MAX_CHARS,
    )
    if docs_context:
        msgs.append({"role": "developer", "content": docs_context})

    # Memory
    msgs.extend(get_memory(chat_id, user_id))

    # User
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ----------------------------
# OpenAI call
# ----------------------------
def openai_reply(chat_id: int, user_id: int, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    messages = build_messages(chat_id, user_id, user_text)
    last_err = None
    for attempt in range(2):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=messages,
                max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                text={"verbosity": "low"},
                reasoning={"effort": "minimal"},
            )
            text = extract_output_text(resp)
            if text:
                return text, None
            last_err = "Empty model output"
        except Exception as e:
            last_err = repr(e)
            logger.exception("OpenAI call failed (attempt %s): %s", attempt + 1, last_err)
            time.sleep(0.6)
    return None, last_err

# ----------------------------
# Telegram handlers
# ----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Quick tips:\n"
        f"- In groups: start with {BOT_TRIGGER} or @{BOT_USERNAME}, or reply to me.\n"
        "- In DMs: just type normally.\n"
        "Examples:\n"
        f"{BOT_TRIGGER} Summarize this in 3 bullets\n"
        f"@{BOT_USERNAME} What should I do next?"
    )

# Stash last DM media (photo or GIF) for later posting
_dm_media_stash: Dict[int, Dict[str, str]] = {}

def _stash_dm_media(user_id: int, media_type: str, file_id: str) -> None:
    _dm_media_stash[user_id] = {"type": media_type, "file_id": file_id}

def _pop_dm_media(user_id: int) -> Optional[Dict[str, str]]:
    return _dm_media_stash.pop(user_id, None)

async def stash_dm_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or msg.chat.type != ChatType.PRIVATE or not msg.from_user:
        return

    # Photo
    if msg.photo:
        file_id = msg.photo[-1].file_id
        _stash_dm_media(msg.from_user.id, "photo", file_id)
        await msg.reply_text("Got it. Now send /posttestcaption <text> and I will post the image + caption to the group.")
        return

    # GIF / animation
    if msg.animation:
        file_id = msg.animation.file_id
        _stash_dm_media(msg.from_user.id, "animation", file_id)
        await msg.reply_text("Got it. Now send /posttestcaption <text> and I will post the GIF + caption to the group.")
        return

# /posttestcaption (DM only) -> posts last DM media (if any) + caption into the Testing Group
async def post_test_caption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    if msg.chat.type != ChatType.PRIVATE:
        await msg.reply_text("Please use /posttestcaption in DM with me.")
        return

    if ADMIN_IDS and msg.from_user and msg.from_user.id not in ADMIN_IDS:
        await msg.reply_text("Not authorized.")
        return

    caption = " ".join(context.args).strip() if context.args else ""
    if not caption:
        await msg.reply_text("Usage: /posttestcaption <caption text>")
        return

    if not TESTING_GROUP_ID:
        await msg.reply_text("TESTING_GROUP_ID is not set.")
        return

    if not msg.from_user:
        await msg.reply_text("Missing user.")
        return

    stashed = _pop_dm_media(msg.from_user.id)
    if not stashed:
        await msg.reply_text("I don't have any recent photo/GIF from you. Send me an image (or GIF) first, then run /posttestcaption.")
        return

    media_type = stashed.get("type")
    file_id = stashed.get("file_id")

    try:
        if media_type == "photo":
            await context.bot.send_photo(chat_id=TESTING_GROUP_ID, photo=file_id, caption=caption)
        elif media_type == "animation":
            await context.bot.send_animation(chat_id=TESTING_GROUP_ID, animation=file_id, caption=caption)
        else:
            await msg.reply_text("Unsupported media type.")
            return

        await msg.reply_text("Posted to the testing group.")
    except Exception as e:
        logger.exception("Failed to post to testing group: %s", repr(e))
        await msg.reply_text("Failed to post. Check logs.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return

    chat = msg.chat
    chat_id = chat.id
    user = msg.from_user
    if not user:
        return
    user_id = user.id

    text_in = msg.text.strip()

    # Determine whether this is a reply to the bot
    is_reply_to_bot = False
    if msg.reply_to_message and msg.reply_to_message.from_user and BOT_USERNAME:
        is_reply_to_bot = (msg.reply_to_message.from_user.username or "").lower() == BOT_USERNAME.lower()

    # DMs: always respond
    if chat.type == ChatType.PRIVATE:
        user_text = text_in
    else:
        # Groups: only respond if triggered
        if not should_respond(text_in, is_reply_to_bot):
            return
        user_text = strip_trigger_prefix(text_in)

    # Persist memory
    add_memory(chat_id, user_id, "user", user_text)

    # Call OpenAI
    answer, err = openai_reply(chat_id, user_id, user_text)
    if err:
        await msg.reply_text("I hit an error calling the model. Check Railway logs and your OPENAI settings.")
        return

    add_memory(chat_id, user_id, "assistant", answer)
    await msg.reply_text(answer)

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("posttestcaption", post_test_caption))

    # DM media stash (photos/animations)
    app.add_handler(MessageHandler(filters.PHOTO | filters.ANIMATION, stash_dm_media))

    # Text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting Penny bot...")
    app.run_polling()

if __name__ == "__main__":
    main()
