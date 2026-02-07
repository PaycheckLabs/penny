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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("penny")


# --- OpenAI client (reads OPENAI_API_KEY from env) ---
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


SYSTEM_PROMPT = (
    "You are Penny, the official AI advisor for Paycheck Labs.\n"
    "Be calm, precise, and protective. Prioritize clarity and correct process over hype.\n"
    "If you are unsure, ask a short follow-up question.\n"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello. I’m Penny, the AI advisor for Paycheck Labs.\n\n"
        "Try:\n"
        "- /help\n"
        '- "What can you do?"\n'
        '- "Summarize Paycheck Labs in 3 bullets"'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/start - Introduction\n"
        "/help - This message\n\n"
        "You can also just type your question normally."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    lower = text.lower()

    greetings = {"hi", "hello", "hey", "yo", "sup", "gm", "good morning", "good evening"}
    if lower in greetings or any(lower.startswith(g + " ") for g in greetings):
        await update.message.reply_text(
            "Hey, I’m Penny.\n\n"
            "Try one of these:\n"
            "- /help\n"
            '- "What can you do?"\n'
            '- "Summarize Paycheck Labs in 3 bullets"'
        )
        return

    if "help" in lower or lower == "what can you do":
        await update.message.reply_text(
            "Right now I can:\n"
            "- Answer questions\n"
            "- Help you draft posts, docs, and announcements\n"
            "- Guide you through setup and troubleshooting\n\n"
            "Tell me what you want and I’ll respond."
        )
        return

    # --- LLM call ---
    try:
        client = get_openai_client()

        response = client.responses.create(
            model="gpt-5-nano",  # forces nano (cost controlled here)
            instructions=SYSTEM_PROMPT,
            input=text,
        )

        output = (response.output_text or "").strip()
        if not output:
            output = "I didn’t get a usable response back. Try again with a bit more detail."

        await update.message.reply_text(output)

    except Exception as e:
        # Log full error for Railway logs
        logger.exception("OpenAI call failed: %s", str(e))

        # Safe user message
        await update.message.reply_text(
            "I hit an error calling the language model. Check Railway logs and your OPENAI_API_KEY."
        )


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
