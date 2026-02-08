import os
import re
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
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("penny")

# -------------------------
# Env + Clients
# -------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

# OpenAI client is optional at runtime; bot can still run without it
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

log.info(f"Booting Penny. OpenAI enabled: {bool(client)} | Model: {OPENAI_MODEL}")

# -------------------------
# Personality + Rules
# -------------------------
SYSTEM_PREAMBLE = """
You are Penny, a friendly, supportive AI assistant.

Style:
- Be concise. Prefer 1 to 4 short sentences.
- Sound natural and human.
- No em dashes.
- Ask a brief follow-up question when it helps.
- Use bullets only when listing 3+ items.

Content rules:
- Do not mention Paycheck Labs, payroll, or paychecks in casual conversation.
- Only mention Paycheck Labs if the user asks who made you, who you work for, or asks about the company directly.
- If the user asks about Paycheck Labs, answer briefly and do not invent details. If you are unsure, say so.

Safety:
- Refuse harmful requests. Offer safer alternatives.
"""

PAYCHECK_TRIGGERS = re.compile(
    r"\b(paycheck\s?labs|checks platform|check token|\$check|paychain|paymart|who made you|who created you|who built you|your creator|your company)\b",
    re.IGNORECASE,
)

GREETINGS = {
    "hi", "hello", "hey", "yo", "sup", "gm", "good morning", "good evening", "good afternoon"
}

# -------------------------
# Commands
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey. I’m Penny.\n\nHow can I help?"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Here are a few things you can ask me:\n"
        "- Help me write a message\n"
        "- Summarize this text\n"
        "- Explain a topic simply\n"
        "- Brainstorm ideas\n\n"
        "What are you working on?"
    )

# -------------------------
# OpenAI helper
# -------------------------
def build_company_reply(user_text: str) -> str:
    # Keep this minimal for now; later you can replace with a proper knowledge file/database.
    return (
        "I’m Penny. I was created for a crypto and software team.\n"
        "If you tell me what you want to know about the project, I’ll keep it short and clear.\n\n"
        "What part are you curious about?"
    )

def call_openai(user_text: str) -> str:
    if not client:
        return "I’m running without my language model right now. Try again in a bit."

    try:
        # IMPORTANT: no temperature parameter (your model rejected it)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PREAMBLE.strip()},
                {"role": "user", "content": user_text.strip()},
            ],
            max_output_tokens=220,
        )

        # responses API output parsing
        text = (resp.output_text or "").strip()
        if not text:
            return "I’m here. What would you like to do?"

        # Hard cap in case the model runs long
        if len(text) > 900:
            text = text[:900].rstrip() + "…"

        return text

    except Exception as e:
        log.exception("OpenAI error")
        return "I hit an error calling the model. Check Railway logs and your OPENAI settings."

# -------------------------
# Message handler
# -------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    lower = text.lower()

    # Short greeting behavior
    if lower in GREETINGS or any(lower.startswith(g + " ") for g in GREETINGS):
        await update.message.reply_text("Hey, how’s it going?")
        return

    # If user is asking about the company/creator, allow a small response
    if PAYCHECK_TRIGGERS.search(text):
        await update.message.reply_text(build_company_reply(text))
        return

    # Normal assistant flow
    reply = call_openai(text)
    await update.message.reply_text(reply)

# -------------------------
# Main
# -------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Reply to any non-command text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Penny is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
