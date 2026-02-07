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
# Config
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "350"))

# Penny voice + safety posture (keep this short—cheap + consistent)
PENNY_INSTRUCTIONS = (
    "You are Penny, the official AI advisor for Paycheck Labs. "
    "Be calm, precise, and protective. Prioritize clarity and correct process over hype. "
    "If the user asks for financial advice, be educational and risk-aware, not promotional. "
    "Keep replies concise unless the user asks for more detail."
)

# Greetings that trigger the “menu-style” reply
GREETINGS = {"hi", "hello", "hey", "yo", "sup", "gm", "good morning", "good evening"}


# -----------------------------
# Setup logging (useful in Railway logs)
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("penny")


# -----------------------------
# OpenAI client (lazy init)
# -----------------------------
_openai_client = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# -----------------------------
# Telegram handlers
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello. I'm Penny — the AI advisor for Paycheck Labs.\n\n"
        "Try:\n"
        "• Ask a question\n"
        "• Type /help"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/start — Introduction\n"
        "/help — This message\n\n"
        "You can also just message me normally."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    lower = text.lower()

    # Friendly greeting -> quick menu
    if lower in GREETINGS or any(lower.startswith(g + " ") for g in GREETINGS):
        await update.message.reply_text(
            "Hey — I’m Penny.\n\n"
            "Try one:\n"
            "• “What can you do?”\n"
            "• “Summarize Paycheck Labs in 3 bullets”\n"
            "• “Help me plan the next Penny feature”"
        )
        return

    if "help" in lower or lower.strip() in {"what can you do", "what can you do?"}:
        await update.message.reply_text(
            "Right now I can:\n"
            "• Answer questions\n"
            "• Help with setup and testing\n"
            "• Draft clear, structured content\n\n"
            "Tell me what you want to do."
        )
        return

    # If no OpenAI key yet, fall back gracefully
    if not OPENAI_API_KEY:
        await update.message.reply_text(
            "I’m not connected to my language model yet.\n"
            "Add OPENAI_API_KEY and redeploy, then try again."
        )
        return

    # Call OpenAI (Responses API)
    try:
        client = get_openai_client()

        # Log what model you are using (so you can confirm in Railway logs)
        logger.info(f"OpenAI call: model={OPENAI_MODEL}, max_output_tokens={OPENAI_MAX_OUTPUT_TOKENS}")

        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=PENNY_INSTRUCTIONS,
            input=text,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )

        reply = (response.output_text or "").strip()
        if not reply:
            reply = "I didn’t get a usable response back. Try again with a bit more detail."

        await update.message.reply_text(reply)

    except Exception as e:
        logger.exception("OpenAI call failed")
        await update.message.reply_text(
            "I hit an error calling the language model. Check Railway logs for details."
        )


def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Reply to any non-command text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
