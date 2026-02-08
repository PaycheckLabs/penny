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
from telegram.constants import ChatMemberStatus, ParseMode
from telegram.ext import (
    ContextTypes,
    ChatMemberHandler,
    CallbackQueryHandler,
)

log = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------

PAYCHECK_WEBSITE = "https://www.paycheck.io/"
PAYCHECK_X = "https://x.com/PaycheckIO"
PAYCHECK_MEDIUM = "https://medium.com/@paycheck"

VERIFY_TIMEOUT_SECONDS = int(os.getenv("VERIFY_TIMEOUT_SECONDS", "180"))  # 3 minutes
KICK_IF_NOT_VERIFIED = os.getenv("KICK_IF_NOT_VERIFIED", "true").lower() == "true"

# Wrong-attempt behavior
CAPTCHA_MAX_ATTEMPTS = int(os.getenv("CAPTCHA_MAX_ATTEMPTS", "3"))
KICK_ON_TOO_MANY_WRONG = os.getenv("KICK_ON_TOO_MANY_WRONG", "true").lower() == "true"

# Your welcome GIF
WELCOME_GIF_FILE_ID = os.getenv("WELCOME_GIF_FILE_ID", "").strip()
WELCOME_GIF_LOCAL_PATH = os.getenv("WELCOME_GIF_LOCAL_PATH", "").strip()

# Callback prefix
CB_VERIFY_PREFIX = "verify_gate:"  # keep short, callback_data has length limits

# Captcha icons
CAPTCHA_ICONS: List[str] = ["✅", "⭐", "🔷", "🍀"]


# -----------------------------
# In-memory state (MVP)
# Swap to Redis/DB later for restart safety
# -----------------------------

@dataclass
class PendingVerification:
    chat_id: int
    user_id: int
    created_at: float
    welcome_message_id: Optional[int] = None
    token: str = ""                 # short per-user token
    correct_index: int = 0          # which icon is correct
    attempts: int = 0               # wrong attempts so far

PENDING: Dict[str, PendingVerification] = {}  # key: f"{chat_id}:{user_id}"


def _key(chat_id: int, user_id: int) -> str:
    return f"{chat_id}:{user_id}"


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
        await context.bot.unban_chat_member(chat_id=chat_id, user_id=user_id)  # kick behavior
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
    # 1) Info links
    rows: List[List[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("Paycheck Website", url=PAYCHECK_WEBSITE),
            InlineKeyboardButton("Paycheck X", url=PAYCHECK_X),
        ],
        [
            InlineKeyboardButton("Paycheck Medium", url=PAYCHECK_MEDIUM),
        ],
    ]

    # 2) Captcha buttons (one row)
    captcha_buttons: List[InlineKeyboardButton] = []
    for idx, icon in enumerate(CAPTCHA_ICONS):
        # callback_data format: verify_gate:<chat_id>:<user_id>:<token>:<choice_index>
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
    # verify_gate:<chat_id>:<user_id>:<token>:<choice_index>
    try:
        payload = data[len(CB_VERIFY_PREFIX):]
        chat_id_str, user_id_str, token, choice_str = payload.split(":")
        return int(chat_id_str), int(user_id_str), token, int(choice_str)
    except Exception:
        return None


async def on_chat_member_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    result = update.chat_member
    if not result:
        return

    chat = result.chat
    new = result.new_chat_member
    old = result.old_chat_member

    user = new.user
    if user.is_bot:
        return

    joined = (old.status in [ChatMemberStatus.LEFT, ChatMemberStatus.KICKED]) and (
        new.status in [ChatMemberStatus.MEMBER, ChatMemberStatus.RESTRICTED]
    )
    if not joined:
        return

    chat_id = chat.id
    user_id = user.id
    first_name = user.first_name or "there"

    # 1) Restrict immediately
    await _apply_hard_gate(chat_id, user_id, context)

    # 2) Create captcha state
    token = secrets.token_hex(4)  # short token, fits callback_data limits
    correct_index = secrets.randbelow(len(CAPTCHA_ICONS))
    required_icon = CAPTCHA_ICONS[correct_index]

    # 3) Send welcome GIF + links + captcha buttons
    caption = _welcome_caption(first_name, required_icon)
    markup = _build_keyboard(chat_id, user_id, token)

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

    # Only the intended user can verify
    if clicker.id != target_user_id:
        await query.answer("This verification is not for you.", show_alert=True)
        return

    k = _key(chat_id, target_user_id)
    pv = PENDING.get(k)
    if not pv:
        await query.answer("Verification expired. Please re-join.", show_alert=True)
        return

    # Token must match (prevents reusing old buttons)
    if token != pv.token:
        await query.answer("Verification expired. Please re-join.", show_alert=True)
        return

    # Correct choice
    if choice_index == pv.correct_index:
        await query.answer("Verified ✅")

        await _lift_gate(chat_id, target_user_id, context)

        # Cancel timeout job
        job_name = f"verify_timeout:{chat_id}:{target_user_id}"
        for j in context.job_queue.get_jobs_by_name(job_name):
            j.schedule_removal()

        PENDING.pop(k, None)

        # Update message caption + remove captcha buttons
        try:
            await query.edit_message_caption(
                caption="Verified ✅ Welcome in!",
                reply_markup=_build_verified_keyboard(),
            )
        except Exception:
            pass
        return

    # Wrong choice
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

    # Rotate challenge after wrong attempt (makes brute-force harder)
    pv.token = secrets.token_hex(4)
    pv.correct_index = secrets.randbelow(len(CAPTCHA_ICONS))
    required_icon = CAPTCHA_ICONS[pv.correct_index]

    await query.answer(f"Wrong button. Try again. Attempts left: {remaining}", show_alert=True)

    # Update caption + refreshed keyboard with new token and requirement
    try:
        await query.edit_message_caption(
            caption=_welcome_caption(clicker.first_name or "there", required_icon),
            reply_markup=_build_keyboard(chat_id, target_user_id, pv.token),
        )
    except Exception:
        pass


def register_welcome_gate_handlers(application) -> None:
    application.add_handler(ChatMemberHandler(on_chat_member_update, ChatMemberHandler.CHAT_MEMBER))
    application.add_handler(CallbackQueryHandler(on_verify_captcha, pattern=f"^{CB_VERIFY_PREFIX}"))
