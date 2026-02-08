import os
import logging
from typing import Optional

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
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("penny-bot")

# -----------------------------
# Config
# -----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# You can change these later without code changes by setting Railway env vars:
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "minimal").strip()
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "220"))

# Your bot username (Telegram @handle). Keep this exact.
BOT_MENTION = os.getenv("BOT_MENTION", "@HeyPennyBot").strip()

# Tight, human, non-corporate vibe.
# Also: do not mention Paycheck Labs unless asked directly.
SYSTEM_PREAMBLE = """
You are Penny, a friendly, concise AI assistant.

Style rules:
- Keep replies short and conversational.
- No em dashes.
- Ask a simple follow up question sometimes to keep the conversation going.
- Do not mention Paycheck Labs, payroll, paychecks, or company details unless the user directly asks who made you, who you work for, or asks about Paycheck Labs.
- Be helpful, supportive, and calm.
- If the user message is nonsense, unsafe, or inappropriate, respond responsibly and briefly.

If the user says "hi" or "hello", reply with something short like:
"Hey. How can I help?" or "Hi. How’s it going?"
""".strip()

# Create OpenAI client only if key exists
openai_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# We'll store the bot's Telegram user id after startup
BOT_USER_ID: Optional[int] = None


# -----------------------------
# Helpers
# -----------------------------
def is_triggered_message(update: Update) -> bool:
    """
    In group chats, only respond if:
      1) message starts with /penny
      2) message starts with @HeyPennyBot
      3) message is a reply to Penny
    In private chats, always respond.
    """
    msg = update.effective_message
    if not msg:
        return False

    chat = msg.chat
    text = (msg.text or "").strip()

    # Private DMs: respond normally
    if chat and chat.type == ChatType.PRIVATE:
        return True

    # Group/supergroup/channel style: require trigger
    if not text and not msg.reply_to_message:
        return False

    # Trigger 1: /penny
    if text.lower().startswith("/penny"):
        return True

    # Trigger 2: @HeyPennyBot
    if text.startswith(BOT_MENTION):
        return True

    # Trigger 3: Replying to Penny
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if BOT_USER_ID and msg.reply_to_message.from_user.id == BOT_USER_ID:
            return True

    return False


def strip_trigger_prefix(user_text: str) -> str:
    """
    Remove /penny or @HeyPennyBot prefix so the model sees the real request.
    """
    t = (user_text or "").strip()
    lower = t.lower()

    if lower.startswith("/penny"):
        t = t[len("/penny"):].strip()

    if t.startswith(BOT_MENTION):
        t = t[len(BOT_MENTION):].strip()

    # If user only typed the trigger, give a minimal prompt
    return t if t else "Hey"


async def call_llm(user_text: str) -> str:
    """
    Calls OpenAI Chat Completions with safe, compatible params.
    Uses max_completion_tokens (not max_tokens).
    """
    if not openai_client:
        return "I’m not fully set up yet. The AI key is missing."

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "developer", "content": SYSTEM_PREAMBLE},
                {"role": "user", "content": user_text},
            ],
            reasoning_effort=OPENAI_REASONING_EFFORT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        content = (resp.choices[0].message.content or "").strip()
        return content if content else "Got it. What would you like to do next?"

    except Exception as e:
        logger.exception("OpenAI call failed: %s", str(e))
        return "I hit an error calling the language model. Please check Railway logs and your OpenAI settings."


# -----------------------------
# Telegram Handlers
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Hey. How can I help?")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Try:\n"
        "- Ask a question\n"
        "- /penny summarize this\n"
        f"- {BOT_MENTION} draft a short reply\n"
    )


async def penny_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /penny <message>
    msg = update.effective_message
    text = (msg.text or "").strip()
    user_text = strip_trigger_prefix(text)

    reply = await call_llm(user_text)
    await msg.reply_text(reply)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    text = (msg.text or "").strip()
    if not text and not msg.reply_to_message:
        return

    # Group chats: ignore unless triggered
    if not is_triggered_message(update):
        return

    # If this was triggered via reply only, use the new message text as prompt
    user_text = strip_trigger_prefix(text)

    reply = await call_llm(user_text)
    await msg.reply_text(reply)


async def on_startup(app):
    global BOT_USER_ID
    me = await app.bot.get_me()
    BOT_USER_ID = me.id
    logger.info("Bot started as @%s (id=%s)", me.username, me.id)


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(on_startup).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("penny", penny_command))

    # Any normal text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    application.run_polling()


if __name__ == "__main__":
    main()
