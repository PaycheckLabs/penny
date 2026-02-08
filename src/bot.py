import os
import logging
from typing import Tuple, Optional

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("penny")

# -----------------------------
# Config
# -----------------------------
BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyPennyBot").lstrip("@")  # e.g. HeyPennyBot
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # change anytime in Railway env vars
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "220"))

SYSTEM_PREAMBLE = (
    "You are Penny, a friendly, compact, human-sounding AI assistant.\n"
    "\n"
    "Style rules:\n"
    "- Keep replies concise by default (usually 1 to 6 sentences).\n"
    "- Do not use em dashes.\n"
    "- Be conversational, supportive, and inquisitive when helpful.\n"
    "- If the user says hi/hello, respond briefly and ask how you can help.\n"
    "- If the user asks unclear or broad questions, ask a short clarifying question.\n"
    "\n"
    "Brand rules:\n"
    "- Do not bring up Paycheck Labs, payroll, paychecks, or company products in casual conversation.\n"
    "- Only mention Paycheck Labs if the user specifically asks who created you or directly asks about it.\n"
    "\n"
    "Safety:\n"
    "- Follow common-sense safety. Refuse harmful requests.\n"
)

# OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# Helpers
# -----------------------------
def _strip_trigger_prefix(text: str, trigger: str) -> str:
    """
    Removes the trigger prefix from the start of text, returning the remaining prompt.
    Handles punctuation/whitespace after the trigger.
    """
    t = text.strip()
    if not t.lower().startswith(trigger.lower()):
        return t
    remainder = t[len(trigger) :].lstrip()
    # Trim common separators after mention/command
    if remainder.startswith((",", ":", "-", "—")):
        remainder = remainder[1:].lstrip()
    return remainder


def should_respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns (should_respond, reason, prompt_text)
    Reasons: "dm", "slash", "mention", "reply"
    """
    msg = update.message
    if not msg or not msg.text:
        return (False, None, None)

    text = msg.text.strip()
    if not text:
        return (False, None, None)

    # --------------- IMPORTANT CHANGE ---------------
    # In private DMs, respond to normal messages (natural chat).
    if msg.chat and msg.chat.type == "private":
        return (True, "dm", text)
    # ------------------------------------------------

    # Group/supergroup/channel behavior: only respond when invoked.

    # 1) /penny (including /penny@BotName)
    # Note: If message is "/penny ..." it will often be handled by CommandHandler,
    # but we also gate here to support plain "/penny hello" text patterns.
    if text.lower().startswith("/penny"):
        prompt = text
        # remove /penny or /penny@BotName prefix
        # split first token then use remainder
        first_token = text.split(maxsplit=1)[0]
        remainder = text[len(first_token) :].strip()
        prompt = remainder if remainder else ""
        return (True, "slash", prompt)

    # 2) @HeyPennyBot mention at the start
    mention_prefix = f"@{BOT_USERNAME}".lower()
    if text.lower().startswith(mention_prefix):
        prompt = _strip_trigger_prefix(text, f"@{BOT_USERNAME}")
        return (True, "mention", prompt)

    # 3) Reply to Penny's message
    if msg.reply_to_message and msg.reply_to_message.from_user:
        try:
            if msg.reply_to_message.from_user.id == context.bot.id:
                return (True, "reply", text)
        except Exception:
            pass

    return (False, None, None)


async def llm_reply(user_text: str) -> str:
    """
    Calls OpenAI and returns assistant text.
    We avoid temperature because some models only support the default.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        user_text = "Hi"

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PREAMBLE},
            {"role": "user", "content": user_text},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    content = (resp.choices[0].message.content or "").strip()
    return content if content else "Hey. How can I help you?"


# -----------------------------
# Handlers
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey. How can I help you?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Quick tips:\n"
        "- In groups: use /penny, start with @HeyPennyBot, or reply to me.\n"
        "- In DMs: just type normally.\n"
        "\n"
        "Examples:\n"
        "- /penny write a short announcement\n"
        "- @HeyPennyBot summarize this in 3 bullets"
    )

async def penny_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /penny <prompt>
    """
    msg = update.message
    if not msg:
        return

    prompt = ""
    if context.args:
        prompt = " ".join(context.args).strip()

    try:
        reply = await llm_reply(prompt)
        await msg.reply_text(reply)
    except Exception as e:
        logger.exception("OpenAI error in /penny: %s", e)
        await msg.reply_text("I hit an error calling the model. Check Railway logs and your OPENAI settings.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles non-command text. In DMs: always respond.
    In groups: only respond when invoked (/penny, @mention, or reply).
    """
    ok, reason, prompt = should_respond(update, context)
    if not ok:
        return

    # In groups, if user invoked with @mention or /penny but gave no prompt, ask a short follow-up.
    if reason in {"slash", "mention"} and (prompt is None or not prompt.strip()):
        await update.message.reply_text("Hey. What do you want to do?")
        return

    try:
        reply = await llm_reply(prompt or "")
        await update.message.reply_text(reply)
    except Exception as e:
        logger.exception("OpenAI error in handle_text: %s", e)
        await update.message.reply_text("I hit an error calling the model. Check Railway logs and your OPENAI settings.")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("penny", penny_command))

    # Non-command text (DMs + invoked group messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
