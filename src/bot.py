import os
import re
import time
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional

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
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "350"))
ENV = os.getenv("ENV", "production").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Behavior settings
# -----------------------------
COMMAND_TRIGGER = "/penny"
MENTION_TRIGGER_DEFAULT = "@HeyPennyBot"  # You can change later if you rename the bot
MAX_TURNS = 10  # conversation memory size (per chat)

# Chat memory:
# chat_id -> deque of (role, content)
CHAT_MEMORY: Dict[int, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

# -----------------------------
# System prompt (style + scope)
# -----------------------------
SYSTEM_PREAMBLE = """You are Penny, a friendly general-purpose AI assistant.

Style rules:
- Sound human, warm, and concise.
- Avoid em dashes. Use normal punctuation.
- Default to short replies. Ask one helpful follow-up question when it makes sense.
- If the user’s request is unclear, ask a brief clarifying question.
- If the user asks for steps, and you offered steps moments ago, continue that thread.

Brand rules:
- Do NOT assume payroll, taxes, or HR.
- Do NOT bring up Paycheck Labs or company products in casual conversation.
- Only mention Paycheck Labs if the user explicitly asks who made you, who you work with, or asks about the company/products.
- If asked about Paycheck Labs, keep it short and factual.

Safety:
- Refuse harmful or illegal requests.
- Be helpful and responsible.
"""

# -----------------------------
# Utilities
# -----------------------------
def _clean_text(text: str) -> str:
    return (text or "").strip()

def _is_private_chat(update: Update) -> bool:
    return bool(update.effective_chat and update.effective_chat.type == "private")

def _bot_username(context: ContextTypes.DEFAULT_TYPE) -> str:
    # Telegram username includes no @ in the API value; we add it
    try:
        uname = context.bot.username or ""
        uname = uname.strip()
        return f"@{uname}" if uname else ""
    except Exception:
        return ""

def _should_respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Rules:
    - In private DMs: respond to normal messages (and triggers still work).
    - In groups/supergroups: respond only if:
        1) message starts with /penny
        2) message starts with @HeyPennyBot OR @<actual_bot_username>
        3) message is a reply to Penny (reply_to_message.from_user.id == bot id)
    """
    msg = update.effective_message
    if not msg:
        return False

    text = _clean_text(msg.text or msg.caption or "")
    if not text and not msg.reply_to_message:
        return False

    if _is_private_chat(update):
        return True

    # Group chat rules
    bot_un = _bot_username(context)
    mention_triggers = [MENTION_TRIGGER_DEFAULT]
    if bot_un:
        mention_triggers.append(bot_un)

    starts_with_command = text.lower().startswith(COMMAND_TRIGGER)
    starts_with_mention = any(text.startswith(m) for m in mention_triggers)

    replied_to_bot = False
    if msg.reply_to_message and msg.reply_to_message.from_user:
        replied_to_bot = (msg.reply_to_message.from_user.id == context.bot.id)

    return starts_with_command or starts_with_mention or replied_to_bot

def _strip_triggers(text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Remove /penny or @mention prefix so the model sees the actual request.
    """
    t = _clean_text(text)
    if not t:
        return t

    # Remove /penny prefix
    if t.lower().startswith(COMMAND_TRIGGER):
        t = t[len(COMMAND_TRIGGER):].strip()
        return t if t else "Hi"

    # Remove @mention prefix
    bot_un = _bot_username(context)
    mention_triggers = [MENTION_TRIGGER_DEFAULT]
    if bot_un:
        mention_triggers.append(bot_un)

    for m in mention_triggers:
        if t.startswith(m):
            t = t[len(m):].strip()
            return t if t else "Hi"

    return t

def _push_memory(chat_id: int, role: str, content: str) -> None:
    content = _clean_text(content)
    if not content:
        return
    CHAT_MEMORY[chat_id].append((role, content))

def _build_input(chat_id: int, user_text: str, reply_context: Optional[str]) -> List[dict]:
    """
    Build Responses API input as a list of messages.
    Include:
    - system preamble
    - short reply-context if user replied to Penny
    - recent memory turns
    - current user message
    """
    items: List[dict] = []

    items.append({
        "role": "system",
        "content": [{"type": "input_text", "text": SYSTEM_PREAMBLE}],
    })

    if reply_context:
        items.append({
            "role": "system",
            "content": [{"type": "input_text", "text": f"Context: The user is replying to this message: {reply_context}"}],
        })

    # memory
    for role, content in CHAT_MEMORY[chat_id]:
        items.append({
            "role": role,
            "content": [{"type": "input_text", "text": content}],
        })

    # current user
    items.append({
        "role": "user",
        "content": [{"type": "input_text", "text": user_text}],
    })

    return items

def _extract_output_text(resp) -> str:
    """
    Prefer resp.output_text.
    If empty, attempt to extract from resp.output structure.
    """
    try:
        txt = (resp.output_text or "").strip()
        if txt:
            return txt
    except Exception:
        pass

    # Fallback parse
    try:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []) or []:
                    # content parts can vary; prefer any text fields we can find
                    if isinstance(c, dict):
                        if "text" in c and isinstance(c["text"], str):
                            parts.append(c["text"])
                        elif c.get("type") in ("output_text", "input_text") and isinstance(c.get("text"), str):
                            parts.append(c["text"])
        return "\n".join([p.strip() for p in parts if p.strip()]).strip()
    except Exception:
        return ""

async def _call_llm(chat_id: int, user_text: str, reply_context: Optional[str]) -> str:
    """
    Calls OpenAI Responses API with memory.
    Avoids parameters that certain models reject.
    """
    input_items = _build_input(chat_id, user_text, reply_context)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=input_items,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )

    answer = _extract_output_text(resp).strip()
    return answer

# -----------------------------
# Handlers
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi. I’m Penny. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Quick ways to talk to me:\n"
        "- In groups: start with /penny or @HeyPennyBot, or reply to one of my messages\n"
        "- In DMs: just message me normally\n\n"
        "Try:\n"
        "- /penny Summarize this in 3 bullets\n"
        "- /penny Help me write a short caption\n"
        "- /penny What is 864 x 327?"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    chat_id = update.effective_chat.id
    raw_text = _clean_text(msg.text or msg.caption or "")

    # Only respond when rules allow
    if not _should_respond(update, context):
        return

    # Reply context (if user replied to Penny)
    reply_context = None
    if msg.reply_to_message and msg.reply_to_message.text:
        # Only use reply context if they replied to Penny directly
        if msg.reply_to_message.from_user and msg.reply_to_message.from_user.id == context.bot.id:
            reply_context = _clean_text(msg.reply_to_message.text)

    # Normalize user input
    user_text = _strip_triggers(raw_text, context)
    if not user_text:
        user_text = "Hi"

    # Save user message to memory
    _push_memory(chat_id, "user", user_text)

    try:
        answer = await _call_llm(chat_id, user_text, reply_context)

        if not answer:
            answer = "Got it. What would you like to do next?"

        # Save assistant answer to memory
        _push_memory(chat_id, "assistant", answer)

        await msg.reply_text(answer)

    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        await msg.reply_text(
            "I hit an error calling the model. Check Railway logs and your OPENAI settings."
        )

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Catch all text messages (commands excluded)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # Also allow captions (photos w/ text, etc.)
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_text))

    logger.info("Penny is running (%s). Model=%s", ENV, OPENAI_MODEL)
    app.run_polling()

if __name__ == "__main__":
    main()
