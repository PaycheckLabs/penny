import os
import re
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

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

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("penny")

# ----------------------------
# Environment
# ----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # you can change later
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "280"))

# Bot mention / command triggers (customize if you want)
BOT_MENTION = os.getenv("BOT_MENTION", "@HeyPennyBot").strip()  # must include "@"
BOT_COMMAND = os.getenv("BOT_COMMAND", "/penny").strip()        # must include "/"

# Optional: test group forwarding feature
PENNY_TEST_CHAT_ID = os.getenv("PENNY_TEST_CHAT_ID", "").strip()  # e.g. "-1001234567890"

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------
# Simple in-memory chat history (short-term memory)
# NOTE: this resets if Railway restarts the container.
# ----------------------------
MAX_TURNS = 10  # total messages stored per chat (user+assistant). Keep small for cost.
CHAT_MEMORY: Dict[Tuple[int, Optional[int]], Deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_TURNS))

# Key: (chat_id, topic_thread_id) so topics in groups stay separate
def memory_key(update: Update) -> Tuple[int, Optional[int]]:
    chat_id = update.effective_chat.id
    thread_id = getattr(update.effective_message, "message_thread_id", None)
    return (chat_id, thread_id)

# ----------------------------
# System prompt (personality + rules)
# ----------------------------
SYSTEM_PREAMBLE = """
You are Penny, a friendly, practical AI assistant.
Write like a helpful human. Keep responses short by default (2 to 6 sentences).
Do not use em dashes.
Be conversational and supportive. When it helps, ask one short follow-up question.

Do not mention Paycheck Labs or payroll/paychecks in casual conversation.
Only mention Paycheck Labs if the user asks who created you, what you are, or asks about the company directly.
If the user asks about Paycheck Labs, answer briefly and factually.

If the user asks for harmful, illegal, or unsafe instructions, refuse politely and suggest a safer alternative.

If the user says "yes" or "show me" or "do it" after you offered options, continue the same thread using the existing context.
""".strip()

# ----------------------------
# Helpers
# ----------------------------
def is_private_chat(update: Update) -> bool:
    chat = update.effective_chat
    return chat and chat.type == ChatType.PRIVATE

def bot_user_id(context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    try:
        return context.bot.id
    except Exception:
        return None

def should_respond_in_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tuple[bool, str]:
    """
    Group rules:
    1) User starts message with /penny
    2) User starts message with @HeyPennyBot (or BOT_MENTION)
    3) User replies to a message that Penny posted
    """
    msg = update.effective_message
    if not msg:
        return (False, "")

    text = (msg.text or msg.caption or "").strip()
    if not text and not msg.reply_to_message:
        return (False, "")

    # Case 3: reply to Penny
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.id == bot_user_id(context):
            # In replies, we should use the user's text as-is (no prefix needed)
            return (True, text)

    # Case 1: /penny prefix
    if text.lower().startswith(BOT_COMMAND.lower()):
        cleaned = text[len(BOT_COMMAND):].strip()
        return (True, cleaned)

    # Case 2: @HeyPennyBot prefix
    if BOT_MENTION and text.lower().startswith(BOT_MENTION.lower()):
        cleaned = text[len(BOT_MENTION):].strip()
        return (True, cleaned)

    return (False, "")

def strip_leading_triggers(text: str) -> str:
    t = text.strip()
    if t.lower().startswith(BOT_COMMAND.lower()):
        t = t[len(BOT_COMMAND):].strip()
    if BOT_MENTION and t.lower().startswith(BOT_MENTION.lower()):
        t = t[len(BOT_MENTION):].strip()
    return t

def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", s)).strip()

def build_input_with_memory(key: Tuple[int, Optional[int]], user_text: str) -> List[dict]:
    """
    Build an input list for the Responses API that includes system + recent memory.
    """
    history = list(CHAT_MEMORY[key])
    msgs = [{"role": "system", "content": SYSTEM_PREAMBLE}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_text})
    return msgs

def extract_output_text(resp) -> str:
    """
    Prefer response.output_text, but fall back to scanning common structures.
    """
    try:
        t = getattr(resp, "output_text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
    except Exception:
        pass

    # Fallback: try dict-like access
    try:
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        # Look for output_text key
        if isinstance(data, dict):
            t = data.get("output_text")
            if isinstance(t, str) and t.strip():
                return t.strip()
            # Search nested output -> content -> text
            out = data.get("output", [])
            if isinstance(out, list):
                chunks = []
                for item in out:
                    content = item.get("content", []) if isinstance(item, dict) else []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            chunks.append(c.get("text", ""))
                        elif isinstance(c, dict) and "text" in c:
                            chunks.append(c.get("text", ""))
                joined = "\n".join([x for x in chunks if isinstance(x, str) and x.strip()]).strip()
                if joined:
                    return joined
    except Exception:
        pass

    return ""

async def call_llm(key: Tuple[int, Optional[int]], user_text: str) -> str:
    if not client:
        return "I am not configured yet. The OpenAI key is missing on the server."

    messages = build_input_with_memory(key, user_text)

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=messages,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        text = extract_output_text(resp)
        return normalize_whitespace(text)
    except Exception as e:
        logger.exception("OpenAI call failed")
        return "I hit an error calling the language model. Check Railway logs and your OPENAI settings."

def remember_turn(key: Tuple[int, Optional[int]], user_text: str, assistant_text: str) -> None:
    if user_text.strip():
        CHAT_MEMORY[key].append({"role": "user", "content": user_text.strip()})
    if assistant_text.strip():
        CHAT_MEMORY[key].append({"role": "assistant", "content": assistant_text.strip()})

# ----------------------------
# Commands
# ----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Quick commands:\n"
        f"- {BOT_COMMAND} <message>\n"
        f"- {BOT_MENTION} <message>\n"
        "- Reply to me in a chat to continue the thread\n"
        "\nIn DMs, you can just type normally."
    )

async def penny_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /penny <text> - always allowed
    msg = update.effective_message
    text = (msg.text or "").strip()
    user_text = text[len(BOT_COMMAND):].strip() if text.lower().startswith(BOT_COMMAND.lower()) else ""
    if not user_text:
        await msg.reply_text("Yep. What do you want to talk about?")
        return

    key = memory_key(update)
    answer = await call_llm(key, user_text)
    if not answer:
        answer = "I did not get a response back. Try again."

    remember_turn(key, user_text, answer)
    await msg.reply_text(answer)

async def posttestcaption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /posttestcaption <caption>
    Sends the caption to the test group. (Optional)
    """
    msg = update.effective_message
    if not msg:
        return

    if not PENNY_TEST_CHAT_ID:
        await msg.reply_text("PENNY_TEST_CHAT_ID is not set on the server.")
        return

    text = (msg.text or "").strip()
    caption = text.replace("/posttestcaption", "", 1).strip()
    if not caption:
        await msg.reply_text("Usage: /posttestcaption <caption>")
        return

    try:
        await context.bot.send_message(chat_id=PENNY_TEST_CHAT_ID, text=caption)
        await msg.reply_text("Posted to the test group.")
    except Exception:
        logger.exception("posttestcaption failed")
        await msg.reply_text("I could not post to the test group. Check Railway logs.")

# ----------------------------
# Main message handler
# ----------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    raw_text = (msg.text or msg.caption or "").strip()
    if not raw_text and not msg.reply_to_message:
        return

    # Decide if we should respond
    if is_private_chat(update):
        # In DMs, respond to any text.
        # Still allow the user to include /penny or @HeyPennyBot, we strip it.
        user_text = strip_leading_triggers(raw_text)
        if not user_text:
            await msg.reply_text("Hey. How can I help?")
            return
    else:
        # In groups/supergroups: respond only if triggered
        ok, extracted = should_respond_in_group(update, context)
        if not ok:
            return
        user_text = extracted if extracted else ""
        user_text = strip_leading_triggers(user_text)
        if not user_text:
            await msg.reply_text("Yep. What do you need?")
            return

    # Small friendly fast-path for greetings (keeps it human and short)
    lower = user_text.lower()
    if lower in {"hi", "hello", "hey", "yo", "sup", "gm", "good morning", "good evening"}:
        await msg.reply_text("Hey. How can I help?")
        return

    # Call LLM with short-term memory
    key = memory_key(update)
    answer = await call_llm(key, user_text)

    if not answer:
        answer = "I did not get a response back. Try again."

    remember_turn(key, user_text, answer)
    await msg.reply_text(answer)

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("penny", penny_command))
    app.add_handler(CommandHandler("posttestcaption", posttestcaption))

    # Text handler (non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
