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
# Config / Setup
# -----------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if you want

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var.")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are Penny, an AI virtual assistant created by Paycheck Labs. "
    "Be conversational, clear, and helpful. Keep answers concise, "
    "ask a brief follow-up question only when needed."
)

# -----------------------------
# Commands
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi, I'm Penny. What can I help you with today?"
    )

# ✅ Temporary utility command: use this in your testing group to get the chat ID
async def chatid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    await update.message.reply_text(
        f"chat.id = {chat.id}\nchat.type = {chat.type}"
    )

# -----------------------------
# Message Handler (ChatGPT)
# -----------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    # Ignore commands here (they're handled by CommandHandlers)
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

    except Exception as e:
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
    app.add_handler(CommandHandler("chatid", chatid))  # ✅ add this

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Penny bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
