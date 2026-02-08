import os
import re
import time
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional

from telegram import Update, Message
from telegram.constants import ChatType
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# NEW: welcome + hard gate module
from welcome_gate import register_welcome_gate_handlers

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("penny-bot")

# -------------------------
# Environment
# -------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "260").strip())

BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyPennyBot").strip().lstrip("@")
BOT_TRIGGER = os.getenv("BOT_TRIGGER", "/penny").strip()

# Optional
ENV = os.getenv("ENV", "production").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

# OpenAI client (only needs key if not using default env var behavior)
client = OpenAI(api_key=OPENAI_API_KEY if OPENAI_API_KEY else None)

# -------------------------
# Conversation Memory (lightweight)
# Keyed by (chat_id, user_id)
# -------------------------
# Store last N user+assistant turns to maintain thread continuity
MEMORY_TURNS = 10
memory: Dict[Tuple[int, int], Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=MEMORY_TURNS * 2))

def add_to_memory(chat_id: int, user_id: int, role: str, content: str) -> None:
    memory[(chat_id, user_id)].append({"role": role, "content": content})

def get_memory(chat_id: int, user_id: int) -> List[Dict[str, str]]:
    return list(memory[(chat_id, user_id)])

# -------------------------
# System style + guardrails
# -------------------------
SYSTEM_PROMPT = (
    "You are Penny, a helpful, friendly general-purpose AI assistant in Telegram.\n"
    "Style rules:\n"
    "- Be concise and conversational. Prefer 1 to 6 short sentences.\n"
    "- No em dashes. Use simple punctuation.\n"
    "- Ask a brief follow-up question when it helps.\n"
    "- Do not mention payroll, paychecks, or HR services.\n"
    "- Do not mention Paycheck Labs unless the user asks who made you or asks about the company.\n"
    "- If the user asks about who created you: say you were created by Paycheck Labs.\n"
    "- If the user asks unsafe or disallowed content, refuse briefly and offer a safer alternative.\n"
    "Math rule:\n"
    "- If the user asks for steps, show steps. If they ask for only the answer, give only the answer.\n"
)

# Friendly short greetings
GREETING_RE = re.compile(r"^(hi|hello|hey|yo|sup|what'?s up|good morning|good afternoon|good evening)\b", re.I)

def is_reply_to_penny(msg: Message) -> bool:
    if not msg.reply_to_message:
        return False
    # Telegram sets reply_to_message.from_user for bot messages too
    ru = msg.reply_to_message.from_user
    if not ru:
        return False
    # Most reliable: is_bot + username match
    if ru.is_bot and (ru.username or "").lower() == BOT_USERNAME.lower():
        return True
    return False

def starts_with_mention(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return t.lower().startswith(f"@{BOT_USERNAME}".lower())

def starts_with_trigger(text: str) -> bool:
    if not text:
        return False
    return text.strip().lower().startswith(BOT_TRIGGER.lower())

def strip_trigger_prefix(text: str) -> str:
    """Remove /penny or @HeyPennyBot prefix from the front, if present."""
    if not text:
        return ""
    t = text.strip()
    if starts_with_trigger(t):
        return t[len(BOT_TRIGGER):].strip()
    if starts_with_mention(t):
        return t[len(f"@{BOT_USERNAME}") :].strip()
    return t

def should_respond(update: Update) -> bool:
    """
    Chatroom rule:
      - respond only if /penny, @HeyPennyBot, or reply-to-Penny
    DM rule:
      - respond to any text (and /penny/@mention also work)
    """
    msg = update.message
    if not msg or not msg.text:
        return False

    # Never respond to ourselves
    if msg.from_user and msg.from_user.is_bot:
        return False

    chat_type = msg.chat.type
    text = msg.text.strip()

    # Private chats: respond to any message
    if chat_type == ChatType.PRIVATE:
        return True

    # Group chats: only respond under the 3 scenarios
    if starts_with_trigger(text):
        return True
    if starts_with_mention(text):
        return True
    if is_reply_to_penny(msg):
        return True

    return False

def extract_output_text(resp) -> str:
    """
    Responses API output is a structured list. Safely extract text.
    """
    try:
        out = ""
        for item in getattr(resp, "output", []) or []:
            content_list = getattr(item, "content", None)
            if not content_list:
                continue
            for c in content_list:
                t = getattr(c, "text", None)
                if t:
                    out += t
        return out.strip()
    except Exception:
        return ""

def build_messages(chat_id: int, user_id: int, user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "developer", "content": SYSTEM_PROMPT}]
    # Add short memory
    msgs.extend(get_memory(chat_id, user_id))
    msgs.append({"role": "user", "content": user_text})
    return msgs

def openai_reply(chat_id: int, user_id: int, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (assistant_text, error_message)
    """
    messages = build_messages(chat_id, user_id, user_text)

    # Minimal retries for transient failures
    last_err = None
    for attempt in range(2):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=messages,
                max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                # Keep output compact
                text={"verbosity": "low"},
                # Lower latency / less overthinking
                reasoning={"effort": "minimal"},
            )
            text = extract_output_text(resp)
            if text:
                return text, None
            last_err = "Empty model output"
        except Exception as e:
            last_err = repr(e)
            logger.exception("OpenAI call failed (attempt %s): %s", attempt + 1, last_err)
            time.sleep(0.6)

    return None, last_err

# -------------------------
# Handlers
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Quick tips:\n"
        f"- In groups: start with {BOT_TRIGGER} or @${BOT_USERNAME}, or reply to me.\n"
        "- In DMs: just type normally.\n"
        f"Examples:\n"
        f"{BOT_TRIGGER} Summarize this in 3 bullets\n"
        f"@{BOT_USERNAME} What should I do next?"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return

    if not should_respond(update):
        return

    raw_text = msg.text.strip()
    user_text = strip_trigger_prefix(raw_text)

    # If they triggered Penny but didn't ask anything
    if (starts_with_trigger(raw_text) or starts_with_mention(raw_text)) and not user_text:
        await msg.reply_text("Hey. What do you want to do?")
        return

    # Basic greeting shortcut (very short)
    if GREETING_RE.match(user_text if user_text else raw_text):
        await msg.reply_text("Hey. How can I help?")
        return

    chat_id = msg.chat.id
    user_id = msg.from_user.id if msg.from_user else 0

    # Store user message in memory (use the cleaned user_text if present)
    add_to_memory(chat_id, user_id, "user", user_text if user_text else raw_text)

    assistant_text, err = openai_reply(chat_id, user_id, user_text if user_text else raw_text)

    if assistant_text:
        # Store assistant response in memory
        add_to_memory(chat_id, user_id, "assistant", assistant_text)
        await msg.reply_text(assistant_text)
        return

    # If model failed, show a helpful, non-spammy message (and log details)
    logger.error("Model returned no response. Error: %s", err)
    await msg.reply_text("I ran into an issue answering that. Try again in a moment.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # NEW: register welcome GIF + hard gate verification handlers
    register_welcome_gate_handlers(app)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Text messages only
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running (%s). Model=%s", ENV, OPENAI_MODEL)
    app.run_polling()

if __name__ == "__main__":
    main()
