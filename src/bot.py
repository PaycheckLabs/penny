import os
import re
import logging
from typing import Optional, Tuple

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
# Env
# -----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "220"))

# Optional: if you want to hard-code your bot mention instead of deriving it at runtime
# BOT_MENTION = os.getenv("BOT_MENTION", "@HeyPennyBot").strip()
BOT_MENTION = "@HeyPennyBot"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# System style + guardrails
# -----------------------------
SYSTEM_PREAMBLE = """
You are Penny, a friendly, compact, conversational AI assistant.

Style rules:
- Keep replies short by default (usually 1 to 4 lines).
- Be warm and human.
- Ask a simple follow-up question sometimes to keep the conversation moving.
- No em dashes. Use simple punctuation.
- Do not be long-winded unless the user asks for detail.

Brand rules:
- Do not bring up Paycheck Labs, payroll, paychecks, or company products in casual conversation.
- Only mention Paycheck Labs if the user directly asks who created you, what you are, or asks about the company.
- If asked about Paycheck Labs, answer briefly and offer to share more later.

Safety:
- Politely refuse unsafe or harmful requests.
""".strip()

# -----------------------------
# Memory: per chat + per thread
# Using Responses API previous_response_id
# -----------------------------
# Key: (chat_id, thread_id) -> last_response_id
LAST_RESPONSE_ID = {}

def thread_key(update: Update) -> Tuple[int, int]:
    """
    Thread support:
    - In groups/topics: message_thread_id is the topic thread.
    - Else: use 0.
    """
    chat_id = update.effective_chat.id
    thread_id = 0
    if update.message and update.message.message_thread_id:
        thread_id = update.message.message_thread_id
    return (chat_id, thread_id)

def extract_output_text(resp) -> str:
    """
    Try the convenience field first, then fall back to parsing output items.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback: parse resp.output list
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if item.get("type") == "message":
                for c in item.get("content", []) or []:
                    if c.get("type") == "output_text" and c.get("text"):
                        parts.append(c["text"])
        joined = "\n".join([p.strip() for p in parts if p and p.strip()]).strip()
        return joined if joined else ""
    except Exception:
        return ""

def is_private_chat(update: Update) -> bool:
    chat = update.effective_chat
    return chat and chat.type == "private"

def is_reply_to_penny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    msg = update.message
    if not msg or not msg.reply_to_message:
        return False
    replied = msg.reply_to_message
    # reply_to_message.from_user may be None sometimes
    if not replied.from_user:
        return False
    # safest: match bot id
    return replied.from_user.is_bot and replied.from_user.id == context.bot.id

def strip_trigger_prefix(text: str) -> str:
    """
    Remove /penny or @HeyPennyBot prefixes if present.
    """
    t = text.strip()

    # Remove leading /penny command (with optional bot username)
    t = re.sub(r"^/penny(@\w+)?\s*", "", t, flags=re.IGNORECASE)

    # Remove leading @HeyPennyBot mention
    mention_escaped = re.escape(BOT_MENTION)
    t = re.sub(rf"^{mention_escaped}\s*", "", t, flags=re.IGNORECASE)

    return t.strip()

def should_respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Rules:
    - Private DM: respond to normal messages (and /penny, @mention, reply all work too).
    - Group chat: only respond if:
        1) message starts with /penny
        2) message starts with @HeyPennyBot
        3) message is a reply to Penny
    """
    msg = update.message
    if not msg or not msg.text:
        return False

    text = msg.text.strip()

    if is_private_chat(update):
        return True

    # Group/supergroup:
    if re.match(r"^/penny(@\w+)?(\s|$)", text, flags=re.IGNORECASE):
        return True

    if text.lower().startswith(BOT_MENTION.lower()):
        return True

    if is_reply_to_penny(update, context):
        return True

    return False

# -----------------------------
# Handlers
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Quick tips:\n"
        "- Ask anything in a DM\n"
        "- In group chats: use /penny or @HeyPennyBot, or reply to me"
    )

async def penny_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /penny in any chat. If user adds text after the command, treat it as a prompt.
    """
    msg = update.message
    if not msg:
        return

    text = msg.text or ""
    prompt = strip_trigger_prefix(text)

    if not prompt:
        await msg.reply_text("Hey. What do you want to do?")
        return

    await respond_with_llm(update, context, prompt)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return

    if not should_respond(update, context):
        return

    user_text = strip_trigger_prefix(msg.text)

    # If user only typed the trigger with no content
    if not user_text:
        await msg.reply_text("Hey. What can I help with?")
        return

    await respond_with_llm(update, context, user_text)

async def respond_with_llm(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    key = thread_key(update)
    prev_id = LAST_RESPONSE_ID.get(key)

    try:
        # Use Responses API with previous_response_id so the model keeps context
        # previous_response_id is optional. :contentReference[oaicite:1]{index=1}
        resp = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PREAMBLE,
            input=user_text,
            previous_response_id=prev_id,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )

        # Store new response id so next message continues the thread
        resp_id = getattr(resp, "id", None)
        if resp_id:
            LAST_RESPONSE_ID[key] = resp_id

        out = extract_output_text(resp).strip()
        if not out:
            out = "Got it. What would you like to do next?"

        await update.message.reply_text(out)

    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        await update.message.reply_text(
            "I hit an error calling the model. Check Railway logs and your OPENAI settings."
        )

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("penny", penny_command))

    # All other text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
