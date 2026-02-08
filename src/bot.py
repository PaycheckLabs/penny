import os
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from telegram import Update, Message
from telegram.constants import ChatType
from telegram.ext import (
    Application,
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
log = logging.getLogger("penny")

# -----------------------------
# Env / Config
# -----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "320"))

PENNY_NAME = os.getenv("PENNY_NAME", "Penny").strip()
BOT_MENTION = os.getenv("BOT_MENTION", "@HeyPennyBot").strip()      # include "@"
BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyPennyBot").strip()     # no "@"
PENNY_COMMAND = "/penny"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# System prompt (compact + human)
# -----------------------------
SYSTEM_PREAMBLE = f"""
You are {PENNY_NAME}, a friendly, concise AI assistant.

Tone and format:
- Keep replies short by default (1–6 lines). Expand only if asked.
- Sound human, warm, and conversational.
- No em dashes. Use normal punctuation.
- If the user is vague, ask one helpful follow-up question.
- Avoid long lists unless the user asks for lists.

Brand rules:
- Do not bring up payroll, paychecks, HR, taxes, W-2s, or similar.
- Do not mention Paycheck Labs unless the user asks who made you or asks about the company.
- If asked about Paycheck Labs, answer briefly and offer to share more.

Safety:
- Refuse harmful requests briefly and redirect to safer help.
""".strip()

# -----------------------------
# In-memory history (lightweight)
# -----------------------------
# Per-chat rolling window (resets if Railway restarts)
HISTORY: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=14))

def add_history(chat_id: int, role: str, content: str) -> None:
    content = (content or "").strip()
    if content:
        HISTORY[chat_id].append({"role": role, "content": content})

# -----------------------------
# Trigger rules
# -----------------------------
def is_private(update: Update) -> bool:
    return bool(update.effective_chat and update.effective_chat.type == ChatType.PRIVATE)

def starts_with_trigger(text: str) -> bool:
    t = (text or "").lstrip()
    if not t:
        return False
    low = t.lower()
    return low.startswith(PENNY_COMMAND) or low.startswith(BOT_MENTION.lower()) or low.startswith(BOT_USERNAME.lower())

def strip_trigger_prefix(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    low = t.lower()

    if low.startswith(PENNY_COMMAND):
        t = t[len(PENNY_COMMAND):].strip()
        low = t.lower()

    if low.startswith(BOT_MENTION.lower()):
        t = t[len(BOT_MENTION):].strip()
        low = t.lower()

    if low.startswith(BOT_USERNAME.lower()):
        t = t[len(BOT_USERNAME):].strip()

    if t.startswith(":"):
        t = t[1:].strip()

    return t.strip()

def replied_to_bot(msg: Message) -> bool:
    if not msg.reply_to_message:
        return False
    r = msg.reply_to_message
    if r.from_user and r.from_user.is_bot:
        return True
    # Sometimes is_bot can be inconsistent; try username match too
    if r.from_user and r.from_user.username:
        return r.from_user.username.lower() == BOT_USERNAME.lower()
    return False

def should_respond(update: Update) -> Tuple[bool, Optional[str]]:
    """
    Returns (should_respond, replied_text_if_any)
    Group rules: respond only if:
      1) starts with /penny
      2) starts with @HeyPennyBot (or HeyPennyBot)
      3) reply to Penny's message
    DM rules: respond to any text.
    """
    msg = update.message
    if not msg or not msg.text:
        return False, None

    if is_private(update):
        return True, None

    if starts_with_trigger(msg.text):
        return True, None

    if replied_to_bot(msg):
        replied_text = msg.reply_to_message.text or msg.reply_to_message.caption or ""
        return True, replied_text.strip() or None

    return False, None

# -----------------------------
# OpenAI helpers
# -----------------------------
def extract_text(resp) -> str:
    """
    Robustly extract text from Responses API result.
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = getattr(resp, "output", None)
    if isinstance(out, list):
        parts: List[str] = []
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    ctype = getattr(c, "type", "")
                    if ctype in ("output_text", "text"):
                        val = getattr(c, "text", None) or getattr(c, "value", None)
                        if isinstance(val, str) and val.strip():
                            parts.append(val.strip())
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    return ""

def call_llm(chat_id: int, user_text: str, replied_to_text: Optional[str]) -> str:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PREAMBLE}]
    if HISTORY[chat_id]:
        messages.extend(list(HISTORY[chat_id]))

    # If user replied to Penny, inject that as the previous assistant message
    if replied_to_text:
        messages.append({"role": "assistant", "content": replied_to_text})

    messages.append({"role": "user", "content": user_text})

    log.info("OpenAI call | chat_id=%s | model=%s | user_len=%s", chat_id, OPENAI_MODEL, len(user_text))

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )
    return extract_text(resp)

# -----------------------------
# Telegram handlers
# -----------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hi. I’m {PENNY_NAME}. How can I help?")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "In groups, I respond when:\n"
        f"- You start with {PENNY_COMMAND}\n"
        f"- You start with {BOT_MENTION}\n"
        "- You reply to one of my messages\n\n"
        "In DMs, you can just type normally."
    )

async def penny_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /penny <message> command handler.
    Works in DMs and groups.
    """
    msg = update.message
    args_text = " ".join(context.args).strip() if context.args else ""
    if not args_text:
        await msg.reply_text("Hey. What can I help with?")
        return

    # Treat this exactly like a triggered message
    await handle_message(update, context, forced_text=args_text, replied_text=None)

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    forced_text: Optional[str] = None,
    replied_text: Optional[str] = None,
):
    msg = update.message
    chat_id = update.effective_chat.id

    try:
        incoming = forced_text if forced_text is not None else (msg.text or "")
        cleaned = strip_trigger_prefix(incoming)

        if not cleaned:
            await msg.reply_text("Hey. What can I help with?")
            return

        add_history(chat_id, "user", cleaned)

        answer = call_llm(chat_id, cleaned, replied_text)

        if not answer:
            await msg.reply_text("I didn’t get a response back. Try again.")
            return

        add_history(chat_id, "assistant", answer)
        await msg.reply_text(answer)

    except Exception as e:
        log.exception("LLM error: %s", e)
        await msg.reply_text("I hit an error calling the model. Check Railway logs and your OPENAI settings.")

async def message_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, replied_text = should_respond(update)
    if not ok:
        return

    await handle_message(update, context, forced_text=None, replied_text=replied_text)

def build_app() -> Application:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("penny", penny_cmd))

    # All normal text flows through the router
    app.add_handler(MessageHandler(filters.TEXT, message_router))

    return app

def main():
    app = build_app()
    log.info("Penny running. model=%s mention=%s username=%s", OPENAI_MODEL, BOT_MENTION, BOT_USERNAME)
    app.run_polling()

if __name__ == "__main__":
    main()
