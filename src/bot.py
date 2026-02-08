import os
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple, Optional

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("penny-bot")

# ---------------------------
# Env config
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "350").strip())

# Telegram username without "@"
BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyPennyBot").strip().lstrip("@")
BOT_MENTION = f"@{BOT_USERNAME}"

# ---------------------------
# OpenAI client
# ---------------------------
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------
# Behavior + style
# ---------------------------
SYSTEM_PREAMBLE = """
You are Penny, a friendly general-purpose AI assistant.

Style rules:
- Write like a helpful human.
- Keep replies compact by default.
- Do not use em dashes.
- If the user says hi/hello, respond briefly and ask how you can help.
- Ask a short follow-up question when it helps the conversation.
- If the user asks for something unsafe or disallowed, refuse briefly and offer a safer alternative.

Content rules:
- Do not assume you do payroll, HR, W-2s, tax forms, or payroll processing.
- Do not bring up Paycheck Labs unless the user explicitly asks who created you, asks about your origin, or asks about Paycheck Labs.
- If asked who created you: say you were created by Paycheck Labs.
"""

# ---------------------------
# Lightweight memory (per chat)
# Stores last N user/assistant turns so follow-ups work.
# ---------------------------
MAX_TURNS = 10  # user+assistant pairs
chat_memory: Dict[int, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))


def _is_private_chat(update: Update) -> bool:
    return bool(update.effective_chat and update.effective_chat.type == "private")


def _text(update: Update) -> str:
    return (update.message.text or "").strip() if update.message else ""


def _reply_text(update: Update) -> Optional[str]:
    if not update.message or not update.message.reply_to_message:
        return None
    return (update.message.reply_to_message.text or "").strip()


def _reply_from_bot(update: Update) -> bool:
    """
    True if user is replying to a message authored by this bot.
    """
    if not update.message or not update.message.reply_to_message:
        return False
    bot = update.get_bot()
    reply_sender = update.message.reply_to_message.from_user
    return bool(reply_sender and bot and reply_sender.id == bot.id)


def _should_respond(update: Update) -> Tuple[bool, str]:
    """
    Returns (should_respond, cleaned_user_prompt)
    In groups: only respond to /penny, @HeyPennyBot, or replies to Penny.
    In DMs: respond to any text message.
    """
    text = _text(update)
    if not text:
        return False, ""

    if _is_private_chat(update):
        # In DMs, respond normally. Also allow /penny as a prefix.
        if text.lower().startswith("/penny"):
            cleaned = text[len("/penny") :].strip()
            return True, cleaned if cleaned else "Hi"
        return True, text

    # Group chat rules
    # 1) /penny prefix
    if text.lower().startswith("/penny"):
        cleaned = text[len("/penny") :].strip()
        return True, cleaned if cleaned else "Hi"

    # 2) @HeyPennyBot prefix
    if text.startswith(BOT_MENTION):
        cleaned = text[len(BOT_MENTION) :].strip()
        return True, cleaned if cleaned else "Hi"

    # 3) Reply to Penny
    if _reply_from_bot(update):
        return True, text

    return False, ""


def _add_to_memory(chat_id: int, role: str, content: str) -> None:
    chat_memory[chat_id].append((role, content))


def _build_openai_input(chat_id: int, user_prompt: str) -> list:
    """
    Build an input array for the Responses API using a small rolling context.
    """
    history = list(chat_memory[chat_id])
    input_items = [{"role": "system", "content": SYSTEM_PREAMBLE.strip()}]

    for role, content in history:
        input_items.append({"role": role, "content": content})

    input_items.append({"role": "user", "content": user_prompt})
    return input_items


def _friendly_error_message() -> str:
    return "I ran into a temporary issue calling the model. Try again in a moment."


def _clean_greeting(user_prompt: str) -> bool:
    low = user_prompt.strip().lower()
    return low in {"hi", "hello", "hey", "yo", "sup", "good morning", "good evening", "good afternoon"}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hey. How can I help you?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Use /penny <message> in group chats, or reply to me, or start with @HeyPennyBot.\n"
        "In DMs, just message me normally."
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        should_respond, prompt = _should_respond(update)
        if not should_respond:
            return

        chat_id = update.effective_chat.id

        # Quick compact greeting behavior
        if _clean_greeting(prompt):
            reply = "Hey. How can I help you?"
            _add_to_memory(chat_id, "user", prompt)
            _add_to_memory(chat_id, "assistant", reply)
            await update.message.reply_text(reply)
            return

        if not client:
            await update.message.reply_text("OpenAI is not configured yet. Missing OPENAI_API_KEY.")
            return

        # Add user message to memory before calling model
        _add_to_memory(chat_id, "user", prompt)

        # Build context-aware input
        openai_input = _build_openai_input(chat_id, prompt)

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=openai_input,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )

        output_text = (response.output_text or "").strip()
        if not output_text:
            output_text = "I did not get a response back. Try again."

        # Save assistant reply to memory
        _add_to_memory(chat_id, "assistant", output_text)

        await update.message.reply_text(output_text)

    except Exception as e:
        # Log the real error so Railway logs show the cause
        logger.exception("OpenAI call failed: %s", str(e))
        await update.message.reply_text(_friendly_error_message())


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Handle text messages (no commands)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
