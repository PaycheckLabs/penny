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

# -------------------------
# Basic setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("penny")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "300"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = (
    "You are Penny, the official AI advisor for Paycheck Labs.\n"
    "Tone: calm, precise, helpful.\n"
    "Priority: clarity and correct process over hype.\n"
    "If you are unsure, say so and ask one short follow-up question.\n"
)

# -------------------------
# Commands
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello. I’m Penny, the AI advisor for Paycheck Labs.\n\n"
        "Try /help or ask me anything."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Available commands:\n"
        "/start\n"
        "/help\n\n"
        "You can also just type a question."
    )

# -------------------------
# OpenAI call
# -------------------------
def llm_reply(user_text: str) -> str:
    if not client:
        return "OpenAI is not configured yet. Set OPENAI_API_KEY in Railway variables."

    # Responses API call (simple single-turn)
    # Uses the model you set in OPENAI_MODEL, e.g. gpt-5-nano :contentReference[oaicite:1]{index=1}
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        text={"verbosity": "low"},
    )

    # Pull text safely
    try:
        return resp.output_text.strip()
    except Exception:
        # Fallback if SDK shape changes
        return "I ran into an issue generating a response. Try again."

# -------------------------
# Text handler
# -------------------------
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

    if "what can you do" in lower or lower == "help":
        await update.message.reply_text(
            "Right now I can:\n"
            "- Answer questions\n"
            "- Explain concepts clearly\n"
            "- Help draft posts and docs\n"
            "- Guide you through setup and testing"
        )
        return

    # Default: send to OpenAI
    try:
        answer = llm_reply(text)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.exception("LLM error: %s", e)
        await update.message.reply_text(
            "I hit an error calling the model. Check Railway logs and your OPENAI_API_KEY."
        )

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Reply to any non-command text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
