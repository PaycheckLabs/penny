import os
import time
import json
import logging
import traceback
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple
from pathlib import Path

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from openai import OpenAI

from welcome_gate import is_allowed_user
from knowledge_base import should_attach_kb_context, build_kb_context

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("penny-bot")

# -----------------------------------------------------------------------------
# Environment / Config
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "450").strip())

# Optional: where the synced Checks docs/whitepaper markdown lives on disk
CHECKS_DOCS_DIR = os.getenv("CHECKS_DOCS_DIR", "").strip()
# Upper bound on how much doc text to inject per reply (keeps prompts small)
CHECKS_DOCS_MAX_CHARS = int(os.getenv("CHECKS_DOCS_MAX_CHARS", "8000").strip())

BOT_USERNAME = os.getenv("BOT_USERNAME", "@HeyPennyBot").strip()

# Limits
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12").strip())

# -----------------------------------------------------------------------------
# Validate required env
# -----------------------------------------------------------------------------
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# In-memory conversation store (per chat)
# -----------------------------------------------------------------------------
chat_history: Dict[int, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS))

# -----------------------------------------------------------------------------
# System prompt
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Penny, the official AI advisor for Paycheck Labs.
Your job is to provide product guidance, community operations help, and educational responses.

Tone:
- Calm, precise, helpful.
- Clarity over hype.
- If you are unsure, ask a short follow-up question.
- If a user requests financial advice, do not give it. Offer general education and risk reminders.

Rules:
- Do not invent features or claim details are in docs unless you have them.
- If you can’t find something in the synced docs context, say so plainly.
"""

# -----------------------------------------------------------------------------
# Docs retrieval (best-effort local scan)
# -----------------------------------------------------------------------------
def _repo_root() -> Path:
    # bot.py is in src/; repo root is one level up
    return Path(__file__).resolve().parent.parent


def _candidate_docs_dirs() -> list[Path]:
    # 1) explicit env var
    dirs: list[Path] = []
    if CHECKS_DOCS_DIR:
        dirs.append(Path(CHECKS_DOCS_DIR))

    root = _repo_root()
    # 2) common sync destinations (support both)
    dirs.extend(
        [
            root / "src" / "data" / "checks_whitepaper",
            root / "docs",
            root / "src" / "docs",
        ]
    )

    # de-dupe, keep existing only
    out: list[Path] = []
    seen = set()
    for d in dirs:
        try:
            d = d.resolve()
        except Exception:
            pass
        if str(d) in seen:
            continue
        seen.add(str(d))
        if d.exists() and d.is_dir():
            out.append(d)
    return out


def _read_markdown_files(base: Path) -> list[tuple[str, str]]:
    """Return list of (path, text) for .md files under base."""
    files: list[Path] = []
    files.extend(sorted(base.glob("*.md")))
    files.extend(sorted(base.rglob("*.md")))

    seen = set()
    out: list[tuple[str, str]] = []
    for fp in files:
        if not fp.is_file():
            continue
        try:
            key = str(fp.resolve())
        except Exception:
            key = str(fp)
        if key in seen:
            continue
        seen.add(key)

        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not txt.strip():
            continue

        # try to show a stable relative path for citations
        try:
            rel = str(fp.resolve().relative_to(_repo_root()))
        except Exception:
            rel = str(fp)
        out.append((rel, txt))
    return out


def _tokenize(s: str) -> set[str]:
    import re as _re

    toks = _re.findall(r"[a-zA-Z0-9_\$]{2,}", s.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "you",
        "are",
        "was",
        "were",
        "what",
        "when",
        "where",
        "how",
        "why",
        "can",
        "could",
        "should",
        "would",
        "about",
        "onto",
    }
    return {t for t in toks if t not in stop}


def checks_docs_context(query: str) -> str:
    """Best-effort retrieval from synced markdown docs. Returns a short context string."""
    qset = _tokenize(query)
    if not qset:
        return ""

    chunks: list[tuple[float, str]] = []  # (score, text)

    for d in _candidate_docs_dirs():
        for rel, txt in _read_markdown_files(d):
            parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
            for p in parts:
                pset = _tokenize(p)
                if not pset:
                    continue
                overlap = len(qset & pset)
                if overlap == 0:
                    continue
                boost = 1.5 if any(t in rel.lower() for t in qset) else 0.0
                score = overlap + boost
                snippet = f"[{rel}]\n{p}"
                chunks.append((score, snippet))

    if not chunks:
        return ""

    chunks.sort(key=lambda x: x[0], reverse=True)

    out: list[str] = []
    total = 0
    for score, snippet in chunks[:40]:
        if total >= CHECKS_DOCS_MAX_CHARS:
            break
        add = snippet + "\n---\n"
        if total + len(add) > CHECKS_DOCS_MAX_CHARS and out:
            break
        out.append(add)
        total += len(add)

    return "".join(out).strip()


# -----------------------------------------------------------------------------
# Message building + OpenAI call
# -----------------------------------------------------------------------------
def build_messages(chat_id: int, user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Best-effort: inject relevant snippets from synced Checks docs/whitepaper
    try:
        ctx = checks_docs_context(user_text)
    except Exception:
        ctx = ""

    if ctx:
        msgs.insert(
            0,
            {
                "role": "system",
                "content": (
                    "Use ONLY the following synced Checks docs excerpts as an authoritative source when answering questions about Checks.\n"
                    "If the excerpts do not contain the answer, say you could not find it in the synced docs and ask a follow-up question.\n\n"
                    f"{ctx}"
                ),
            },
        )

    # Attach KB context only when useful (your existing logic)
    try:
        if should_attach_kb_context(user_text):
            kb_context = build_kb_context(user_text)
            if kb_context:
                msgs.append(
                    {
                        "role": "system",
                        "content": f"Paycheck Labs internal context (use only if relevant):\n{kb_context}",
                    }
                )
    except Exception:
        pass

    # Add history
    for role, content in chat_history[chat_id]:
        msgs.append({"role": role, "content": content})

    # Add current user message
    msgs.append({"role": "user", "content": user_text})
    return msgs


def openai_reply(chat_id: int, user_text: str) -> str:
    messages = build_messages(chat_id, user_text)

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=messages,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        out = (resp.output_text or "").strip()
        return out if out else "I didn’t get a response back. Try again."
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        logger.error(traceback.format_exc())
        return "I hit an error calling the model. Check Railway logs and your OPENAI settings."


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Hi — I’m Penny. Ask me anything about Paycheck Labs or Checks.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "Commands:\n"
        "/start — start\n"
        "/help — help\n\n"
        "You can also just message me normally."
    )


def _should_respond_in_group(update: Update) -> bool:
    """Respond in groups only when directly mentioned or replied-to, to avoid spam."""
    if not update.message:
        return False

    text = update.message.text or ""
    # If user replied to bot
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        if update.message.reply_to_message.from_user.username:
            if update.message.reply_to_message.from_user.username.lower() in (BOT_USERNAME or "").lower():
                return True
        # Some bots may not have username populated; allow reply anyway
        return True

    # If bot username mentioned
    if BOT_USERNAME and BOT_USERNAME.lower() in text.lower():
        return True

    return False


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    # Gatekeeping (your existing allowlist logic)
    if not is_allowed_user(update):
        return

    chat_id = update.effective_chat.id if update.effective_chat else 0
    chat_type = update.effective_chat.type if update.effective_chat else None

    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    # Group handling: only respond when mentioned/replied to
    if chat_type in (ChatType.GROUP, ChatType.SUPERGROUP):
        if not _should_respond_in_group(update):
            return

    # Store user message in history
    chat_history[chat_id].append(("user", user_text))

    # Get reply
    reply = openai_reply(chat_id, user_text)

    # Store assistant reply in history
    chat_history[chat_id].append(("assistant", reply))

    await update.message.reply_text(reply)


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Penny bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
