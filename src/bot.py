import os

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello. I’m Penny, the AI advisor for Paycheck Labs.\n\n"
        "I’m online and still coming together. More soon."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Introduction\n"
        "/help - This message\n\n"
        "You can also just type a message. Try:\n"
        "- “What can you do?”\n"
        "- “Summarize Paycheck Labs in 3 bullets”"
    )


# Handles normal text like "Hi"
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    lower = text.lower()

    # Friendly hello
    greetings = {"hi", "hello", "hey", "yo", "sup", "gm", "good morning", "good evening"}
    if lower in greetings or any(lower.startswith(g + " ") for g in greetings):
        await update.message.reply_text(
            "Hey, I’m Penny.\n\n"
            "Try one of these:\n"
            "- /help\n"
            "- “What can you do?”\n"
            "- “Summarize Paycheck Labs in 3 bullets”"
        )
        return

    # Simple help intent
    if lower in {"help", "/help", "what can you do", "what can you do?"}:
        await update.message.reply_text(
            "Right now I can:\n"
            "- Respond to messages\n"
            "- Guide you through setup and testing\n\n"
            "Next: I can be upgraded to answer questions about Paycheck Labs, crypto, and more."
        )
        return

    # Default fallback
    await update.message.reply_text(
        "Got it. I’m still in early build mode.\n\n"
        "Tell me what you want (example: “price check”, “news”, “write a post”), and I’ll respond."
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

    print("Penny is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
