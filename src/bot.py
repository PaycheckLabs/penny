import os
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("penny")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "300"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

PENNY_NAME = os.getenv("PENNY_NAME", "Penny").strip()
BOT_MENTION = os.getenv("BOT_MENTION", "@HeyPennyBot").strip()  # include "@"
PENNY_COMMAND = "/penny"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Persona / System
# -----------------------------
SYSTEM_PREAMBLE = f"""
You are {PENNY_NAME}, a friendly, concise AI assistant.

Style rules:
- Keep replies short by default (1–5 lines). Expand only if the user asks.
- Sound human, warm, and conversational.
- Do not use em dashes (—). Use normal punctuation.
- Ask a simple follow-up question when it helps.
- If the user replies "yes" or "sure", infer the most recent context and continue.

Brand rules:
- Do NOT bring up Paycheck Labs, payroll, paychecks, HR, taxes, W-2s, or similar.
- Only mention Paycheck Labs if the user explicitly asks who made you, asks about your creator, or asks about the company.
- If user asks about Paycheck Labs, answer briefly and offer to share more.

Safety:
- Refuse harmful or disallowed requests briefly and redirect.
""".strip()

# -----------------------------
# Lightweight in-memory chat history
# -----------------------------
# Key: chat_id (int). Value: deque of messages like {"role": "...", "content": "..."}
HISTORY: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=12))

def _clean_trigger_prefix(text: str) -> str:
    """Remove /penny or @mention prefix if present."""
    t = (text or "").strip()
    if not t:
        return t

    low = t.lower()

    # /penny ...
    if low.startswith(PENNY_COMMAND):
        t = t[len(PENNY_COMMAND):].strip()

    # @HeyPennyBot ...
    # allow '@heypennybot' or 'heypennybot'
    mention_low = BOT_MENTION.lower().lstrip("@")
    if low.startswith(BOT_MENTION.lower()):
        t = t[len(BOT_MENTION):].strip()
    elif low.startswith("@" + mention_low):
        t = t[len("@" + mention_low):].strip()
    elif low.startswith(mention_low):
        t = t[len(mention_low):].strip()

    return t.strip()

def _is_private(update: Update) -> bool:
    return update.effective_chat and update.effective_chat.type == ChatType.PRIVATE

def _starts_with_trigger(text: str) -> bool:
    t = (text or "").lstrip()
    if not t:
        return False
    low = t.lower()
    return low.startswith(PENNY_COMMAND) or low.startswith(BOT_MENTION.lower()) or low.startswith(BOT_MENTION.lower().lstrip("@"))

def _is_reply_to_penny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """True if user is replying to one of Penny's messages."""
    msg = update.message
    if not msg or not msg.reply_to_message:
        return False
    replied = msg.reply_to_message

    # If bot is running, the replied message from the bot should have from_user.is_bot True.
    # We also match bot username if available.
    if replied.from_user and replied.from_user.is_bot:
        return True

    # Fallback: compare username if Telegram provides it
    bot_user = getattr(context, "bot", None)
    if bot_user and replied.from_user and replied.from_user.username and bot_user.username:
        return replied.from_user.username.lower() == bot_user.username.lower()

    return False

def _should_respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Group rules: only respond if /penny, @HeyPennyBot, or reply-to-Penny. Private: respond to all text."""
    msg = update.message
    if not msg or not msg.text:
        return False

    if _is_private(update):
        return True

    # group / supergroup
    if _starts_with_trigger(msg.text):
        return True

    if _is_reply_to_penny(update, context):
        return True

    return False

def _extract_openai_text(resp) -> str:
    """
    Robust text extraction across different response shapes.
    Prefers resp.output_text if present, else walks output items.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Walk resp.output (Responses API)
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    ctype = getattr(c, "type", "")
                    if ctype in ("output_text", "text"):
                        val = getattr(c, "text", None) or getattr(c, "value", None)
                        if isinstance(val, str) and val.strip():
                            chunks.append(val.strip())
        combined = "\n".join(chunks).strip()
        if combined:
            return combined

    return ""

async def _ask_llm(chat_id: int, user_text: str, replied_to_text: Optional[str] = None) -> str:
    """
    Build messages:
    - system preamble
    - recent history
    - if replying to Penny: include that message as assistant turn
    - current user message
    """
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PREAMBLE}]

    # Add short history
    if chat_id in HISTORY and HISTORY[chat_id]:
        messages.extend(list(HISTORY[chat_id]))

    # If user replied to Penny, inject that as last assistant message (if not already last)
    if replied_to_text:
        replied_to_text = replied_to_text.strip()
        if replied_to_text:
            # Avoid duplicating if last history item is already that assistant message
            if not (messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == replied_to_text):
                messages.append({"role": "assistant", "content": replied_to_text})

    messages.append({"role": "user", "content": user_text})

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )

    out = _extract_openai_text(resp)
    return out

def _history_add(chat_id: int, role: str, content: str) -> None:
    if not content or not content.strip():
        return
    HISTORY[chat_id].append({"role": role, "content": content.strip()})

# -----------------------------
# Handlers
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hi. I’m {PENNY_NAME}. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Use:\n"
        f"- {PENNY_COMMAND} <message>\n"
        f"- {BOT_MENTION} <message>\n"
        "- Reply to one of my messages\n"
        "\nIn DMs, you can also just type normally."
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not _should_respond(update, context):
            return

        chat_id = update.effective_chat.id
        raw_text = update.message.text or ""
        cleaned = _clean_trigger_prefix(raw_text)

        # If it was only the trigger with no content, ask a short question
        if not cleaned:
            await update.message.reply_text("Hey. What can I help with?")
            return

        replied_to_text = None
        if update.message.reply_to_message and _is_reply_to_penny(update, context):
            replied_to_text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""

        # Add user message to history
        _history_add(chat_id, "user", cleaned)

        # Ask model
        answer = await _ask_llm(chat_id=chat_id, user_text=cleaned, replied_to_text=replied_to_text)

        if not answer.strip():
            await update.message.reply_text("I didn’t get a response back. Try again.")
            return

        # Add assistant message to history
        _history_add(chat_id, "assistant", answer)

        await update.message.reply_text(answer)

    except Exception as e:
        log.exception("handle_text error: %s", e)
        await update.message.reply_text(
            "I hit an error calling the model. Check Railway logs and your OPENAI settings."
        )

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Note: We do NOT register /penny as a CommandHandler on purpose,
    # because we want "/penny <text>" to stay in handle_text and also work in groups.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.COMMAND, handle_text))  # catches "/penny ..." too

    log.info("Penny is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
