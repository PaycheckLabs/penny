import os
import logging
import time
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

# Test group forwarding
PENNY_TEST_CHAT_ID = os.getenv("PENNY_TEST_CHAT_ID", "").strip()

# Admins (numeric Telegram user IDs) comma-separated
ADMIN_USER_IDS_RAW = os.getenv("ADMIN_USER_IDS", "").strip()
ADMIN_USER_IDS = set()
if ADMIN_USER_IDS_RAW:
    for part in ADMIN_USER_IDS_RAW.split(","):
        part = part.strip()
        if part.isdigit():
            ADMIN_USER_IDS.add(int(part))

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
""".strip()

# ---------------------------
# Lightweight memory (per chat)
# ---------------------------
MAX_TURNS = 10  # user+assistant pairs
chat_memory: Dict[int, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

# ---------------------------
# Pending state for /posttestcaption two-step flow (DM only)
# user_id -> (caption, timestamp)
# ---------------------------
pending_posttest: Dict[int, Tuple[str, float]] = {}
PENDING_TTL_SECONDS = 10 * 60  # 10 minutes


def _is_private_chat(update: Update) -> bool:
    return bool(update.effective_chat and update.effective_chat.type == "private")


def _text(update: Update) -> str:
    return (update.message.text or "").strip() if update.message else ""


def _reply_from_bot(update: Update) -> bool:
    if not update.message or not update.message.reply_to_message:
        return False
    bot = update.get_bot()
    reply_sender = update.message.reply_to_message.from_user
    return bool(reply_sender and bot and reply_sender.id == bot.id)


def _should_respond(update: Update) -> Tuple[bool, str]:
    """
    In groups: only respond to /penny, @HeyPennyBot, or replies to Penny.
    In DMs: respond to any text message.
    """
    text = _text(update)
    if not text:
        return False, ""

    if _is_private_chat(update):
        if text.lower().startswith("/penny"):
            cleaned = text[len("/penny") :].strip()
            return True, cleaned if cleaned else "Hi"
        return True, text

    if text.lower().startswith("/penny"):
        cleaned = text[len("/penny") :].strip()
        return True, cleaned if cleaned else "Hi"

    if text.startswith(BOT_MENTION):
        cleaned = text[len(BOT_MENTION) :].strip()
        return True, cleaned if cleaned else "Hi"

    if _reply_from_bot(update):
        return True, text

    return False, ""


def _add_to_memory(chat_id: int, role: str, content: str) -> None:
    chat_memory[chat_id].append((role, content))


def _build_openai_input(chat_id: int, user_prompt: str) -> list:
    history = list(chat_memory[chat_id])
    input_items = [{"role": "system", "content": SYSTEM_PREAMBLE}]
    for role, content in history:
        input_items.append({"role": role, "content": content})
    input_items.append({"role": "user", "content": user_prompt})
    return input_items


def _friendly_error_message() -> str:
    return "I ran into a temporary issue calling the model. Try again in a moment."


def _is_greeting(prompt: str) -> bool:
    low = prompt.strip().lower()
    return low in {"hi", "hello", "hey", "yo", "sup", "good morning", "good evening", "good afternoon"}


def _is_admin(update: Update) -> bool:
    user = update.effective_user
    return bool(user and user.id in ADMIN_USER_IDS)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hey. How can I help you?")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Group chats:\n"
        "- /penny <message>\n"
        f"- {BOT_MENTION} <message>\n"
        "- Reply to one of my messages\n\n"
        "DMs:\n"
        "- Just message me normally\n"
        "- /posttestcaption <caption> (admins only, forwards your next photo to the test group)"
    )


# ---------------------------
# /posttestcaption (DM only)
# ---------------------------
async def posttestcaption_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_private_chat(update):
        await update.message.reply_text("Use this in a DM with me.")
        return

    if not _is_admin(update):
        await update.message.reply_text("Sorry, this command is restricted.")
        return

    if not PENNY_TEST_CHAT_ID:
        await update.message.reply_text("Missing PENNY_TEST_CHAT_ID in Railway variables.")
        return

    # If user typed: /posttestcaption some caption
    caption = ""
    if context.args:
        caption = " ".join(context.args).strip()

    if caption:
        pending_posttest[update.effective_user.id] = (caption, time.time())
        await update.message.reply_text("Got it. Now send the image you want posted to the test group.")
    else:
        await update.message.reply_text(
            "Send a caption with the command, then send the image.\n\n"
            "Example:\n"
            "/posttestcaption New caption here"
        )


async def handle_private_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    DM-only photo handler to support:
    1) Photo caption begins with /posttestcaption <caption>
    2) Two-step flow: /posttestcaption <caption> then send photo
    """
    if not _is_private_chat(update):
        return

    if not _is_admin(update):
        return

    if not PENNY_TEST_CHAT_ID:
        await update.message.reply_text("Missing PENNY_TEST_CHAT_ID in Railway variables.")
        return

    msg = update.message
    if not msg or not msg.photo:
        return

    user_id = update.effective_user.id
    caption_text = (msg.caption or "").strip()

    final_caption = ""

    # Case 1: photo caption contains /posttestcaption ...
    if caption_text.lower().startswith("/posttestcaption"):
        final_caption = caption_text[len("/posttestcaption") :].strip()

    # Case 2: pending caption from prior command
    if not final_caption and user_id in pending_posttest:
        saved_caption, ts = pending_posttest[user_id]
        if time.time() - ts <= PENDING_TTL_SECONDS:
            final_caption = saved_caption
        # clear it either way
        pending_posttest.pop(user_id, None)

    # If still no caption, refuse (keeps flow clean)
    if not final_caption:
        await msg.reply_text(
            "I got the image. Now send:\n"
            "/posttestcaption <caption>\n"
            "Then resend the image, or add /posttestcaption in the image caption."
        )
        return

    try:
        # send the highest-res photo
        photo_file_id = msg.photo[-1].file_id
        await context.bot.send_photo(
            chat_id=int(PENNY_TEST_CHAT_ID),
            photo=photo_file_id,
            caption=final_caption,
        )
        await msg.reply_text("Posted to the Penny Testing Group.")
    except Exception as e:
        logger.exception("Failed to forward posttestcaption: %s", str(e))
        await msg.reply_text("I could not post that right now. Check Railway logs.")


# ---------------------------
# Main text handler (LLM)
# ---------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        should_respond, prompt = _should_respond(update)
        if not should_respond:
            return

        chat_id = update.effective_chat.id

        if _is_greeting(prompt):
            reply = "Hey. How can I help you?"
            _add_to_memory(chat_id, "user", prompt)
            _add_to_memory(chat_id, "assistant", reply)
            await update.message.reply_text(reply)
            return

        if not client:
            await update.message.reply_text("OpenAI is not configured yet. Missing OPENAI_API_KEY.")
            return

        _add_to_memory(chat_id, "user", prompt)

        openai_input = _build_openai_input(chat_id, prompt)

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=openai_input,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )

        output_text = (response.output_text or "").strip()
        if not output_text:
            output_text = "I did not get a response back. Try again."

        _add_to_memory(chat_id, "assistant", output_text)

        await update.message.reply_text(output_text)

    except Exception as e:
        logger.exception("OpenAI call failed: %s", str(e))
        await update.message.reply_text(_friendly_error_message())


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("posttestcaption", posttestcaption_command))

    # Photos (DM only, for forwarding feature)
    app.add_handler(MessageHandler(filters.PHOTO & filters.ChatType.PRIVATE, handle_private_photo))

    # Text (no commands)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
