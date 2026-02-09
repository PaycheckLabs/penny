import os
import time
import logging
import secrets
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ChatPermissions,
)
from telegram.constants import ChatMemberStatus, ParseMode, ChatType
from telegram.ext import (
    ContextTypes,
    ChatMemberHandler,
    CallbackQueryHandler,
    ChatJoinRequestHandler,
    MessageHandler,
    filters,
)

log = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------

PAYCHECK_WEBSITE = "https://www.paycheck.io/"
PAYCHECK_X = "https://x.com/PaycheckIO"
PAYCHECK_MEDIUM = "https://medium.com/@paycheck"

VERIFY_TIMEOUT_SECONDS = int(os.getenv("VERIFY_TIMEOUT_SECONDS", "180"))
KICK_IF_NOT_VERIFIED = os.getenv("KICK_IF_NOT_VERIFIED", "true").lower() == "true"

CAPTCHA_MAX_ATTEMPTS = int(os.getenv("CAPTCHA_MAX_ATTEMPTS", "3"))
KICK_ON_TOO_MANY_WRONG = os.getenv("KICK_ON_TOO_MANY_WRONG", "true").lower() == "true"

WELCOME_GIF_FILE_ID = os.getenv("WELCOME_GIF_FILE_ID", "").strip()
WELCOME_GIF_LOCAL_PATH = os.getenv("WELCOME_GIF_LOCAL_PATH", "").strip()

CB_VERIFY_PREFIX = "verify_gate:"
CAPTCHA_ICONS: List[str] = ["✅", "⭐", "🔷", "🍀"]

# -----------------------------
# Optional allowlist / gating helpers (used by bot.py)
# -----------------------------

WELCOME_GATE_ENABLED = os.getenv("WELCOME_GATE_ENABLED", "false").lower() == "true"

def _parse_id_set(env_name: str) -> set[int]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out

ADMIN_USER_IDS = _parse_id_set("ADMIN_USER_IDS") or _parse_id_set("PENNY_ADMIN_USER_IDS")
ALLOWED_USER_IDS = _parse_id_set("ALLOWED_USER_IDS")  # if set, only these users can use Penny

def is_allowed_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Lightweight gate for Penny's AI responses.

    - If ALLOWED_USER_IDS is set, only those IDs (plus admins) can use Penny.
    - If WELCOME_GATE_ENABLED is true, users currently in the verification flow (PENDING) are blocked.
    """
    user = getattr(update, "effective_user", None)
    chat = getattr(update, "effective_chat", None)
    if not user or not chat:
        return False

    uid = int(user.id)

    if uid in ADMIN_USER_IDS:
        return True

    if ALLOWED_USER_IDS and uid not in ALLOWED_USER_IDS:
        return False

    if WELCOME_GATE_ENABLED and chat.type in ("group", "supergroup"):
        # If user is currently pending verification, block AI usage until verified.
        if _key(chat.id, uid) in PENDING:
            return False

    return True

# Anti-duplicate welcome window (seconds)
WELCOME_DEDUP_SECONDS = int(os.getenv("WELCOME_DEDUP_SECONDS", "20"))


# -----------------------------
# In-memory state (MVP)
# -----------------------------

@dataclass
class PendingVerification:
    chat_id: int
    user_id: int
    created_at: float
    welcome_message_id: Optional[int] = None
    token: str = ""
    correct_index: int = 0
    attempts: int = 0

# pending verifications
PENDING: Dict[str, PendingVerification] = {}

# dedup: last time we started flow per user per chat
LAST_WELCOME_TS: Dict[str, float] = {}  # key: f"{chat_id}:{user_id}" -> timestamp


def _key(chat_id: int, user_id: int) -> str:
    return f"{chat_id}:{user_id}"


def _recently_welcomed(chat_id: int, user_id: int) -> bool:
    k = _key(chat_id, user_id)
    now = time.time()
    ts = LAST_WELCOME_TS.get(k, 0.0)
    if now - ts < WELCOME_DEDUP_SECONDS:
        return True
    LAST_WELCOME_TS[k] = now
    return False


def _restricted_perms() -> ChatPermissions:
    return ChatPermissions(
        can_send_messages=False,
        can_send_audios=False,
        can_send_documents=False,
        can_send_photos=False,
        can_send_videos=False,
        can_send_video_notes=False,
        can_send_voice_notes=False,
        can_send_polls=False,
        can_send_other_messages=False,
        can_add_web_page_previews=False,
        can_change_info=False,
        can_invite_users=False,
        can_pin_messages=False,
        can_manage_topics=False,
    )


def _allowed_perms() -> ChatPermissions:
    return ChatPermissions(
        can_send_messages=True,
        can_send_audios=True,
        can_send_documents=True,
        can_send_photos=True,
        can_send_videos=True,
        can_send_video_notes=True,
        can_send_voice_notes=True,
        can_send_polls=True,
        can_send_other_messages=True,
        can_add_web_page_previews=True,
        can_invite_users=True,
    )


async def _apply_hard_gate(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=_restricted_perms(),
        )
    except Exception:
        log.exception("Failed to restrict user %s in chat %s", user_id, chat_id)


async def _lift_gate(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=_allowed_perms(),
        )
    except Exception:
        log.exception("Failed to unrestrict user %s in chat %s", user_id, chat_id)


async def _kick_user(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        await context.bot.unban_chat_member(chat_id=chat_id, user_id=user_id)
    except Exception:
        log.exception("Failed to kick user %s in chat %s", user_id, chat_id)


def _welcome_caption(first_name: str, required_icon: str) -> str:
    return (
        f"Welcome, {first_name} 👋\n\n"
        "I’m Penny. You’re in my testing group for Penny v1.1 by Paycheck Labs.\n\n"
        f"Before you can chat, tap the {required_icon} button below.\n\n"
        "Talk to me with:\n\n"
        "▸ @HeyPennyBot\n"
        "▸ /penny + your message\n"
        "▸ Reply to one of my messages\n\n"
        "Paycheck links are in the buttons below."
    )


def _build_keyboard(chat_id: int, user_id: int, token: str) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("Paycheck Website", url=PAYCHECK_WEBSITE),
            InlineKeyboardButton("Paycheck X", url=PAYCHECK_X),
        ],
        [
            InlineKeyboardButton("Paycheck Medium", url=PAYCHECK_MEDIUM),
        ],
    ]

    captcha_buttons: List[InlineKeyboardButton] = []
    for idx, icon in enumerate(CAPTCHA_ICONS):
        cd = f"{CB_VERIFY_PREFIX}{chat_id}:{user_id}:{token}:{idx}"
        captcha_buttons.append(InlineKeyboardButton(icon, callback_data=cd))

    rows.append(captcha_buttons)
    return InlineKeyboardMarkup(rows)


def _build_verified_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Paycheck Website", url=PAYCHECK_WEBSITE),
            InlineKeyboardButton("Paycheck X", url=PAYCHECK_X),
        ],
        [
            InlineKeyboardButton("Paycheck Medium", url=PAYCHECK_MEDIUM),
        ],
    ])


async def _timeout_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    user_id = data.get("user_id")
    if chat_id is None or user_id is None:
        return

    k = _key(chat_id, user_id)
    pv = PENDING.get(k)
    if not pv:
        return

    if KICK_IF_NOT_VERIFIED:
        await _kick_user(chat_id, user_id, context)

    PENDING.pop(k, None)

    if pv.welcome_message_id:
        try:
            await context.bot.edit_message_caption(
                chat_id=chat_id,
                message_id=pv.welcome_message_id,
                caption="Verification window expired. Please re-join and verify.",
                reply_markup=_build_verified_keyboard(),
            )
        except Exception:
            pass


def _parse_callback(data: str) -> Optional[Tuple[int, int, str, int]]:
    try:
        payload = data[len(CB_VERIFY_PREFIX):]
        chat_id_str, user_id_str, token, choice_str = payload.split(":")
        return int(chat_id_str), int(user_id_str), token, int(choice_str)
    except Exception:
        return None


async def _start_verification_flow(chat_id: int, user, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Shared entry point. Called from multiple update types.
    """
    if user.is_bot:
        return

    user_id = user.id
    if _recently_welcomed(chat_id, user_id):
        return

    first_name = user.first_name or "there"

    # 1) Restrict immediately
    await _apply_hard_gate(chat_id, user_id, context)

    # 2) Captcha state
    token = secrets.token_hex(4)
    correct_index = secrets.randbelow(len(CAPTCHA_ICONS))
    required_icon = CAPTCHA_ICONS[correct_index]

    caption = _welcome_caption(first_name, required_icon)
    markup = _build_keyboard(chat_id, user_id, token)

    # 3) Send welcome message
    sent = None
    try:
        if WELCOME_GIF_FILE_ID:
            sent = await context.bot.send_animation(
                chat_id=chat_id,
                animation=WELCOME_GIF_FILE_ID,
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
            )
        elif WELCOME_GIF_LOCAL_PATH:
            with open(WELCOME_GIF_LOCAL_PATH, "rb") as f:
                sent = await context.bot.send_animation(
                    chat_id=chat_id,
                    animation=f,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
        else:
            sent = await context.bot.send_message(
                chat_id=chat_id,
                text=caption,
                reply_markup=markup,
            )
    except Exception:
        log.exception("Failed sending welcome content in chat %s", chat_id)

    pv = PendingVerification(
        chat_id=chat_id,
        user_id=user_id,
        created_at=time.time(),
        welcome_message_id=(sent.message_id if sent else None),
        token=token,
        correct_index=correct_index,
        attempts=0,
    )
    PENDING[_key(chat_id, user_id)] = pv

    # 4) Schedule timeout
    job_name = f"verify_timeout:{chat_id}:{user_id}"
    for j in context.job_queue.get_jobs_by_name(job_name):
        j.schedule_removal()

    context.job_queue.run_once(
        _timeout_job,
        when=VERIFY_TIMEOUT_SECONDS,
        name=job_name,
        data={"chat_id": chat_id, "user_id": user_id},
    )


# -----------------------------
# Join request handler (only used if invite link requires approval)
# -----------------------------
async def on_join_request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    req = update.chat_join_request
    if not req:
        return

    chat_id = req.chat.id
    user = req.from_user

    try:
        await context.bot.approve_chat_join_request(chat_id=chat_id, user_id=user.id)
    except Exception:
        log.exception("Failed to approve join request for user %s in chat %s", user.id, chat_id)
        return

    # After approval, Telegram may or may not emit chat_member reliably.
    # Start flow immediately here too.
    await _start_verification_flow(chat_id, user, context)


# -----------------------------
# Fallback: NEW_CHAT_MEMBERS service message (most reliable)
# -----------------------------
async def on_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.new_chat_members:
        return

    # Only groups/supergroups
    if msg.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return

    chat_id = msg.chat.id
    for user in msg.new_chat_members:
        await _start_verification_flow(chat_id, user, context)


# -----------------------------
# ChatMember updates (nice when it fires)
# -----------------------------
async def on_chat_member_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cm = update.chat_member
    if not cm:
        return

    user = cm.new_chat_member.user
    if user.is_bot:
        return

    old = cm.old_chat_member
    new = cm.new_chat_member

    # A join is usually LEFT/KICKED -> MEMBER/RESTRICTED.
    if old.status in (ChatMemberStatus.LEFT, ChatMemberStatus.KICKED) and new.status in (
        ChatMemberStatus.MEMBER,
        ChatMemberStatus.RESTRICTED,
    ):
        await _start_verification_flow(cm.chat.id, user, context)


# -----------------------------
# Captcha button handler
# -----------------------------
async def on_verify_captcha(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    if not query.data.startswith(CB_VERIFY_PREFIX):
        return

    parsed = _parse_callback(query.data)
    if not parsed:
        await query.answer("Invalid verification.", show_alert=True)
        return

    chat_id, target_user_id, token, choice_index = parsed
    clicker = query.from_user
    if not clicker:
        return

    if clicker.id != target_user_id:
        await query.answer("This verification is not for you.", show_alert=True)
        return

    k = _key(chat_id, target_user_id)
    pv = PENDING.get(k)
    if not pv:
        await query.answer("Verification expired. Please re-join.", show_alert=True)
        return

    if token != pv.token:
        await query.answer("Verification expired. Please re-join.", show_alert=True)
        return

    # Correct
    if choice_index == pv.correct_index:
        await query.answer("Verified ✅")

        await _lift_gate(chat_id, target_user_id, context)

        job_name = f"verify_timeout:{chat_id}:{target_user_id}"
        for j in context.job_queue.get_jobs_by_name(job_name):
            j.schedule_removal()

        PENDING.pop(k, None)

        try:
            await query.edit_message_caption(
                caption="Verified ✅ Welcome in!",
                reply_markup=_build_verified_keyboard(),
            )
        except Exception:
            pass
        return

    # Wrong
    pv.attempts += 1
    remaining = max(0, CAPTCHA_MAX_ATTEMPTS - pv.attempts)

    if remaining <= 0:
        await query.answer("Too many wrong attempts.", show_alert=True)
        if KICK_ON_TOO_MANY_WRONG:
            await _kick_user(chat_id, target_user_id, context)
        PENDING.pop(k, None)
        try:
            await query.edit_message_caption(
                caption="Verification failed. Please re-join and try again.",
                reply_markup=_build_verified_keyboard(),
            )
        except Exception:
            pass
        return

    # Rotate challenge
    pv.token = secrets.token_hex(4)
    pv.correct_index = secrets.randbelow(len(CAPTCHA_ICONS))
    required_icon = CAPTCHA_ICONS[pv.correct_index]

    await query.answer(f"Wrong button. Try again. Attempts left: {remaining}", show_alert=True)

    try:
        await query.edit_message_caption(
            caption=_welcome_caption(clicker.first_name or "there", required_icon),
            reply_markup=_build_keyboard(chat_id, target_user_id, pv.token),
        )
    except Exception:
        pass


def register_welcome_gate_handlers(application) -> None:
    # If join requests are enabled, approve and start flow
    application.add_handler(ChatJoinRequestHandler(on_join_request))

    # Most reliable join signal in groups
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, on_new_chat_members))

    # Nice-to-have when chat_member updates arrive
    application.add_handler(ChatMemberHandler(on_chat_member_update, ChatMemberHandler.CHAT_MEMBER))

    # Captcha button clicks
    application.add_handler(CallbackQueryHandler(on_verify_captcha, pattern=f"^{CB_VERIFY_PREFIX}"))
