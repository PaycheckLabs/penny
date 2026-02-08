import os
import time
import logging
from collections import deque
from typing import Deque, Dict, Tuple, List, Optional

from telegram import Update, Message
from telegram.constants import ChatType
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("penny")

# ----------------------------
# Env
# ----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "250"))

BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyPennyBot").strip()  # no @
BOT_MENTION = os.getenv("BOT_MENTION", f"@{BOT_USERNAME}").strip()
COMMAND_TRIGGER = os.getenv("COMMAND_TRIGGER", "/penny").strip()

PENNY_TEST_CHAT_ID = os.getenv("PENNY_TEST_CHAT_ID", "").strip()  # optional
ADMIN_USER_IDS_RAW = os.getenv("ADMIN_USER_IDS", "").strip()  # optional: "123,456"
ADMIN_USER_IDS = set()
if ADMIN_USER_IDS_RAW:
    for part in ADMIN_USER_IDS_RAW.split(","):
        part = part.strip()
        if part.isdigit():
            ADMIN_USER_IDS.add(int(part))

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------
# Conversation memory (short + in-memory)
# Keyed by (chat_id, user_id)
# Stores a rolling window of turns: {"role": "...", "content": "..."}
# ----------------------------
MemoryKey = Tuple[int, int]
memory: Dict[MemoryKey, Deque[dict]] = {}
MEMORY_MAX_TURNS = 10            # 10 turns total (user+assistant)
MEMORY_TTL_SECONDS = 15 * 60     # 15 minutes
memory_timestamps: Dict[MemoryKey, float] = {}

# ----------------------------
# System prompt (tight + no payroll)
# ----------------------------
SYSTEM_PREAMBLE = (
    "You are Penny, a friendly AI assistant.\n"
    "Write like a helpful human. Keep replies concise, usually 1 to 6 lines.\n"
    "Do not use em dashes. Use normal punctuation.\n"
    "Do not mention Paycheck Labs or payroll unless the user explicitly asks who created you, "
    "who you work for, or asks about Paycheck Labs.\n"
    "If the user greets you, reply briefly and ask how you can help.\n"
    "Be supportive and curious. When useful, ask one short follow-up question.\n"
    "If the user requests unsafe or disallowed content, refuse briefly and offer a safe alternative.\n"
)

# ----------------------------
# Helpers
# ----------------------------
def _now() -> float:
    return time.time()

def cleanup_memory() -> None:
    """Remove expired conversation buffers."""
    t = _now()
    expired = [k for k, ts in memory_timestamps.items() if (t - ts) > MEMORY_TTL_SECONDS]
    for k in expired:
        memory_timestamps.pop(k, None)
        memory.pop(k, None)

def get_buffer(chat_id: int, user_id: int) -> Deque[dict]:
    key = (chat_id, user_id)
    if key not in memory:
        memory[key] = deque(maxlen=MEMORY_MAX_TURNS)
    memory_timestamps[key] = _now()
    return memory[key]

def add_turn(chat_id: int, user_id: int, role: str, content: str) -> None:
    buf = get_buffer(chat_id, user_id)
    buf.append({"role": role, "content": content})
    memory_timestamps[(chat_id, user_id)] = _now()

def is_private(update: Update) -> bool:
    return bool(update.message and update.message.chat and update.message.chat.type == ChatType.PRIVATE)

def message_starts_with_trigger(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return t.lower().startswith(COMMAND_TRIGGER.lower())

def message_starts_with_mention(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return t.lower().startswith(BOT_MENTION.lower())

def is_reply_to_penny(message: Message) -> bool:
    if not message.reply_to_message:
        return False
    r = message.reply_to_message
    if not r.from_user:
        return False
    # username can be None if privacy settings; we handle both cases
    if r.from_user.username and r.from_user.username.lower() == BOT_USERNAME.lower():
        return True
    return False

def should_respond(update: Update) -> bool:
    """Group chat gating rules."""
    if not update.message or not update.message.text:
        return False

    text = update.message.text.strip()
    if is_private(update):
        # In DM: respond to normal text, and triggers also work
        return True

    # In group/supergroup: only respond if trigger/mention/reply
    if message_starts_with_trigger(text):
        return True
    if message_starts_with_mention(text):
        return True
    if is_reply_to_penny(update.message):
        return True

    return False

def strip_routing_prefix(text: str) -> str:
    """Remove /penny or @mention prefix before sending to LLM."""
    t = text.strip()
    if message_starts_with_trigger(t):
        t = t[len(COMMAND_TRIGGER):].strip()
    if message_starts_with_mention(t):
        t = t[len(BOT_MENTION):].strip()
    return t

def safe_text(m: Optional[Message]) -> str:
    if not m:
        return ""
    if m.text:
        return m.text.strip()
    if m.caption:
        return m.caption.strip()
    return ""

def extract_openai_text(resp) -> str:
    """
    Robust extraction:
    - Prefer resp.output_text if present
    - Otherwise walk resp.output for message content
    """
    txt = ""
    if hasattr(resp, "output_text") and resp.output_text:
        txt = str(resp.output_text).strip()
        if txt:
            return txt

    # Walk structured output
    if hasattr(resp, "output") and resp.output:
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                content = getattr(item, "content", None) or []
                for part in content:
                    # different SDK builds expose different fields
                    ptype = getattr(part, "type", None)
                    if ptype in ("output_text", "text"):
                        val = getattr(part, "text", None) or getattr(part, "value", None)
                        if val:
                            txt = str(val).strip()
                            if txt:
                                return txt

    return ""

async def call_llm(chat_id: int, user_id: int, user_text: str, replied_text: str = "") -> str:
    if not client:
        return "LLM is not configured yet."

    cleanup_memory()

    # Build message list with short memory
    buf = list(get_buffer(chat_id, user_id))

    # If user replied to Penny, include the replied text explicitly
    # This helps when Telegram replies don’t include full context
    if replied_text:
        user_text = f"(Replying to: {replied_text})\n\n{user_text}".strip()

    messages = [{"role": "system", "content": SYSTEM_PREAMBLE}]
    messages.extend(buf)
    messages.append({"role": "user", "content": user_text})

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=messages,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        out = extract_openai_text(resp)

        # Retry once if empty output (common with GPT-5 family)
        if not out:
            resp2 = client.responses.create(
                model=OPENAI_MODEL,
                input=messages + [{"role": "user", "content": "Answer the user now in 1 to 6 short lines."}],
                max_output_tokens=max(OPENAI_MAX_OUTPUT_TOKENS, 350),
            )
            out = extract_openai_text(resp2)

        return out.strip()

    except Exception as e:
        log.exception("OpenAI call failed")
        return "I hit an error calling the model. Check Railway logs and your OPENAI settings."

# ----------------------------
# Commands
# ----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi. I’m Penny. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "You can talk to me normally in DMs.\n"
        "In group chats, use:\n"
        f"- {COMMAND_TRIGGER} your question\n"
        f"- {BOT_MENTION} your question\n"
        "- or reply to one of my messages"
    )

async def posttestcaption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    DM-only helper:
    User sends an image with a caption in Penny DM and uses /posttestcaption
    Penny reposts to the testing group.
    """
    if not update.message:
        return

    if not is_private(update):
        await update.message.reply_text("Use this in DM with me.")
        return

    if ADMIN_USER_IDS and update.message.from_user and update.message.from_user.id not in ADMIN_USER_IDS:
        await update.message.reply_text("Not allowed.")
        return

    if not PENNY_TEST_CHAT_ID:
        await update.message.reply_text("PENNY_TEST_CHAT_ID is not set in Railway.")
        return

    # Find last photo in the message (Telegram photos come as sizes)
    if not update.message.photo:
        await update.message.reply_text("Send an image with a caption, then run /posttestcaption.")
        return

    caption = (update.message.caption or "").strip()
    if not caption:
        await update.message.reply_text("Add a caption to the image first.")
        return

    try:
        file_id = update.message.photo[-1].file_id
        await context.bot.send_photo(chat_id=int(PENNY_TEST_CHAT_ID), photo=file_id, caption=caption)
        await update.message.reply_text("Posted to the testing group.")
    except Exception:
        log.exception("Failed to post to testing group")
        await update.message.reply_text("Could not post to the testing group. Check logs.")

# ----------------------------
# Main text handler
# ----------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    if not should_respond(update):
        return

    raw = update.message.text.strip()
    user_text = strip_routing_prefix(raw)

    # If the message is ONLY a trigger like "/penny" with no content
    if raw.lower().startswith(COMMAND_TRIGGER.lower()) and not user_text:
        await update.message.reply_text("Yep. What do you want to do?")
        return

    # Basic greeting shortcut (fast + cheap)
    lowered = user_text.lower().strip()
    greetings = {"hi", "hello", "hey", "yo", "sup", "gm", "good morning", "good evening"}
    if lowered in greetings:
        await update.message.reply_text("Hey. How can I help?")
        return

    chat_id = update.message.chat_id
    user_id = update.message.from_user.id if update.message.from_user else 0

    replied_text = ""
    if update.message.reply_to_message:
        replied_text = safe_text(update.message.reply_to_message)

    # Save the user turn into memory
    add_turn(chat_id, user_id, "user", user_text if user_text else raw)

    # Call LLM
    out = await call_llm(chat_id, user_id, user_text if user_text else raw, replied_text=replied_text)

    # If still empty, do not spam; give a helpful nudge
    if not out:
        await update.message.reply_text("I didn’t get a response back. Try again.")
        return

    # Save assistant turn into memory
    add_turn(chat_id, user_id, "assistant", out)

    await update.message.reply_text(out)

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("posttestcaption", posttestcaption))

    # Text messages (non-command)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Penny is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
