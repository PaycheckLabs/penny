import os
import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -----------------------------
# Env / Config
# -----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PENNY_TEST_CHAT_ID = int(os.getenv("PENNY_TEST_CHAT_ID", "0"))

# Comma-separated numeric Telegram user IDs, e.g. "123456789,987654321"
ADMIN_USER_IDS = set(
    int(x.strip()) for x in os.getenv("ADMIN_USER_IDS", "").split(",") if x.strip()
)

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var.")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are Penny, an AI virtual assistant created by Paycheck Labs. "
    "Be conversational, clear, and helpful. Keep answers clean and concise. "
    "Ask a short follow-up question only when needed."
)

# -----------------------------
# Helpers
# -----------------------------
def is_admin(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    return uid in ADMIN_USER_IDS


# -----------------------------
# Commands
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hi, I'm Penny. What can I help you with today?")

# Utility: run in a group to get the chat id (safe to delete later)
async def chatid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    await update.message.reply_text(f"chat.id = {chat.id}\nchat.type = {chat.type}")

# Admin-only: post an image + caption into the Penny Testing Chat
# Run this in DM with Penny (or a private admin group) so it doesn't appear in the public chat.
#
# Usage:
#   /posttest https://example.com/image.png | Your caption text
async def posttest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    # Only admins can trigger this
    if not is_admin(update):
        return

    if PENNY_TEST_CHAT_ID == 0:
        await update.message.reply_text("Missing PENNY_TEST_CHAT_ID env var.")
        return

    raw = update.message.text or ""
    payload = raw.replace("/posttest", "", 1).strip()

    if not payload:
        await update.message.reply_text("Usage: /posttest <image_url> | <caption>")
        return

    parts = [p.strip() for p in payload.split("|", 1)]
    image_url = parts[0]
    caption = parts[1] if len(parts) > 1 else ""

    if not image_url.startswith("http"):
        await update.message.reply_text("Image must be a public http(s) URL.")
        return

    try:
        await context.bot.send_photo(
            chat_id=PENNY_TEST_CHAT_ID,
            photo=image_url,
            caption=caption,
        )
        # Extra polish: silent success (no confirmation message)
        return
    except Exception:
        logger.exception("Failed to post image to testing chat")
        await update.message.reply_text("Failed to post. Check logs.")


# -----------------------------
# Normal message handler (OpenAI)
# -----------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    # Let command handlers deal with commands
    if user_text.startswith("/"):
        return

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
        )
        reply = resp.choices[0].message.content.strip()
        await update.message.reply_text(reply)
    except Exception:
        logger.exception("Error calling OpenAI")
        await update.message.reply_text(
            "I hit an error calling the model. Check Railway logs and your OPENAI settings."
        )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chatid))
    app.add_handler(CommandHandler("posttest", posttest))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Penny bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
