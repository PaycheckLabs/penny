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
from welcome_gate import register_welcome_gate_handlers

# NEW: Paycheck Labs knowledge base (lightweight retrieval)
from knowledge_base import build_kb_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("penny-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "260").strip())

BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyPennyBot").strip().lstrip("@")
BOT_TRIGGER = (os.getenv("BOT_TRIGGER") or os.getenv("COMMAND_TRIGGER") or "/penny").strip()

def _parse_int_env(name: str, default: int = 0) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _parse_admin_ids() -> set[int]:
    raw = (os.getenv("ADMIN_USER_IDS", "") or "").strip()
    if not raw:
        return set()
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    out: set[int] = set()
    for p in parts:
        try:
            out.add(int(p))
        except ValueError:
            continue
    return out

TEST_GROUP_CHAT_ID = _parse_int_env("PENNY_TEST_CHAT_ID", 0)
if not TEST_GROUP_CHAT_ID:
    TEST_GROUP_CHAT_ID = _parse_int_env("TEST_GROUP_CHAT_ID", 0)

ADMIN_IDS = _parse_admin_ids()
single_admin = _parse_int_env("ADMIN_TELEGRAM_USER_ID", 0)
if single_admin:
    ADMIN_IDS.add(single_admin)

ENV = os.getenv("ENV", "production").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

client = OpenAI(api_key=OPENAI_API_KEY if OPENAI_API_KEY else None)

# -------------------------
# Conversation Memory
# -------------------------
MEMORY_TURNS = 10
memory: Dict[Tuple[int, int], Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=MEMORY_TURNS * 2))

def add_to_memory(chat_id: int, user_id: int, role: str, content: str) -> None:
    memory[(chat_id, user_id)].append({"role": role, "content": content})

def get_memory(chat_id: int, user_id: int) -> List[Dict[str, str]]:
    return list(memory[(chat_id, user_id)])

# -------------------------
# NEW: DM media stash for /posttestcaption
# -------------------------
DM_MEDIA_TTL_SECONDS = int(os.getenv("DM_MEDIA_TTL_SECONDS", "900"))  # 15 minutes
_last_dm_media: Dict[int, Dict[str, str]] = {}  # user_id -> {"type": "photo|animation", "file_id": "...", "ts": "..."}

def _stash_dm_media(user_id: int, media_type: str, file_id: str) -> None:
    _last_dm_media[user_id] = {"type": media_type, "file_id": file_id, "ts": str(int(time.time()))}

def _pop_recent_dm_media(user_id: int) -> Optional[Dict[str, str]]:
    data = _last_dm_media.get(user_id)
    if not data:
        return None
    try:
        ts = int(data.get("ts", "0"))
    except ValueError:
        ts = 0
    if time.time() - ts > DM_MEDIA_TTL_SECONDS:
        _last_dm_media.pop(user_id, None)
        return None
    # keep it for reuse, do NOT pop by default
    return data

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

GREETING_RE = re.compile(r"^(hi|hello|hey|yo|sup|what'?s up|good morning|good afternoon|good evening)\b", re.I)

def is_reply_to_penny(msg: Message) -> bool:
    if not msg.reply_to_message:
        return False
    ru = msg.reply_to_message.from_user
    if not ru:
        return False
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
    if not text:
        return ""
    t = text.strip()
    if starts_with_trigger(t):
        return t[len(BOT_TRIGGER):].strip()
    if starts_with_mention(t):
        return t[len(f"@{BOT_USERNAME}") :].strip()
    return t

def should_respond(update: Update) -> bool:
    msg = update.message
    if not msg or not msg.text:
        return False
    if msg.from_user and msg.from_user.is_bot:
        return False
    chat_type = msg.chat.type
    text = msg.text.strip()
    if chat_type == ChatType.PRIVATE:
        return True
    if starts_with_trigger(text) or starts_with_mention(text) or is_reply_to_penny(msg):
        return True
    return False

def extract_output_text(resp) -> str:
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

    # NEW: inject KB context only when relevant
    kb_context = build_kb_context(user_text, max_sections=3)
    if kb_context:
        msgs.append({"role": "developer", "content": kb_context})

    msgs.extend(get_memory(chat_id, user_id))
    msgs.append({"role": "user", "content": user_text})
    return msgs

def openai_reply(chat_id: int, user_id: int, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    messages = build_messages(chat_id, user_id, user_text)
    last_err = None
    for attempt in range(2):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=messages,
                max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                text={"verbosity": "low"},
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi. How can I help?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Quick tips:\n"
        f"- In groups: start with {BOT_TRIGGER} or @{BOT_USERNAME}, or reply to me.\n"
        "- In DMs: just type normally.\n"
        "Examples:\n"
        f"{BOT_TRIGGER} Summarize this in 3 bullets\n"
        f"@{BOT_USERNAME} What should I do next?"
    )

# Stash last DM media (photo or GIF) for later posting
async def stash_dm_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or msg.chat.type != ChatType.PRIVATE or not msg.from_user:
        return

    # Photo
    if msg.photo:
        file_id = msg.photo[-1].file_id
        _stash_dm_media(msg.from_user.id, "photo", file_id)
        await msg.reply_text("Got it. Now send /posttestcaption <text> and I will post the image + caption to the group.")
        return

    # GIF / animation
    if msg.animation:
        file_id = msg.animation.file_id
        _stash_dm_media(msg.from_user.id, "animation", file_id)
        await msg.reply_text("Got it. Now send /posttestcaption <text> and I will post the GIF + caption to the group.")
        return

# /posttestcaption (DM only) -> posts last DM media (if any) + caption into the Testing Group
async def post_test_caption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    if msg.chat.type != ChatType.PRIVATE:
        await msg.reply_text("Please use /posttestcaption in DM with me.")
        return

    if ADMIN_IDS and msg.from_user and msg.from_user.id not in ADMIN_IDS:
        await msg.reply_text("Not authorized.")
        return

    if not TEST_GROUP_CHAT_ID:
        await msg.reply_text("PENNY_TEST_CHAT_ID is not set in Railway variables.")
        return

    caption = " ".join(context.args).strip() if context.args else ""
    if not caption:
        await msg.reply_text("Usage:\n/posttestcaption <your caption text>")
        return

    user_id = msg.from_user.id if msg.from_user else 0
    media = _pop_recent_dm_media(user_id)

    try:
        if media and media.get("type") == "photo":
            await context.bot.send_photo(chat_id=TEST_GROUP_CHAT_ID, photo=media["file_id"], caption=caption)
            await msg.reply_text("Posted image + caption to the Penny Testing Group ✅")
        elif media and media.get("type") == "animation":
            await context.bot.send_animation(chat_id=TEST_GROUP_CHAT_ID, animation=media["file_id"], caption=caption)
            await msg.reply_text("Posted GIF + caption to the Penny Testing Group ✅")
        else:
            await context.bot.send_message(chat_id=TEST_GROUP_CHAT_ID, text=caption)
            await msg.reply_text("Posted text to the Penny Testing Group ✅ (send a photo/GIF first if you want media)")
    except Exception as e:
        logger.exception("post_test_caption failed: %r", e)
        await msg.reply_text(f"Failed to post. Error: {e!r}")

# TEMP: Get Telegram GIF file_id (send GIF to Penny in a DM)
async def debug_get_gif_file_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    if msg.chat.type != ChatType.PRIVATE:
        return
    if msg.animation:
        file_id = msg.animation.file_id
        await msg.reply_text(f"GIF file_id:\n{file_id}")
        logger.info("WELCOME_GIF_FILE_ID=%s", file_id)
    else:
        await msg.reply_text("No GIF animation detected. Please send the GIF as an animation (GIF).")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return

    if not should_respond(update):
        return

    raw_text = msg.text.strip()
    user_text = strip_trigger_prefix(raw_text)

    if (starts_with_trigger(raw_text) or starts_with_mention(raw_text)) and not user_text:
        await msg.reply_text("Hey. What do you want to do?")
        return

    if GREETING_RE.match(user_text if user_text else raw_text):
        await msg.reply_text("Hey. How can I help?")
        return

    chat_id = msg.chat.id
    user_id = msg.from_user.id if msg.from_user else 0

    add_to_memory(chat_id, user_id, "user", user_text if user_text else raw_text)
    assistant_text, err = openai_reply(chat_id, user_id, user_text if user_text else raw_text)

    if assistant_text:
        add_to_memory(chat_id, user_id, "assistant", assistant_text)
        await msg.reply_text(assistant_text)
        return

    logger.error("Model returned no response. Error: %s", err)
    await msg.reply_text("I ran into an issue answering that. Try again in a moment.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    register_welcome_gate_handlers(app)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("posttestcaption", post_test_caption))

    # stash media in DM for later posting
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & (filters.PHOTO | filters.ANIMATION), stash_dm_media))

    # TEMP: Listen for GIFs in DM and reply with file_id (kept for convenience)
    app.add_handler(MessageHandler(filters.ANIMATION, debug_get_gif_file_id))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Penny is running (%s). Model=%s", ENV, OPENAI_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
