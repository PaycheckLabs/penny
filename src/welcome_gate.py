import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional

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
# Config (edit as needed)
# -----------------------------

PAYCHECK_WEBSITE = "https://www.paycheck.io/"
PAYCHECK_X = "https://x.com/PaycheckIO"
PAYCHECK_MEDIUM = "https://medium.com/@paycheck"

# Hard-gate settings
VERIFY_TIMEOUT_SECONDS = int(os.getenv("VERIFY_TIMEOUT_SECONDS", "180"))  # 3 minutes default
KICK_IF_NOT_VERIFIED = os.getenv("KICK_IF_NOT_VERIFIED", "true").lower() == "true"

# Your welcome GIF:
# Prefer using a Telegram file_id for reliability (fast + no hosting needed).
WELCOME_GIF_FILE_ID = os.getenv("WELCOME_GIF_FILE_ID", "").strip()
# If you prefer a local path (less ideal on cloud deploys unless bundled):
WELCOME_GIF_LOCAL_PATH = os.getenv("WELCOME_GIF_LOCAL_PATH", "").strip()

# Callback data prefix
CB_VERIFY_PREFIX = "verify_gate:"


# -----------------------------
# In-memory state
# (fine for MVP; swap to Redis/DB later)
# -----------------------------

@dataclass
class PendingVerification:
    chat_id: int
    user_id: int
    created_at: float
    welcome_message_id: Optional[int] = None

PENDING: Dict[str, PendingVerification] = {}  # key: f"{chat_id}:{user_id}"


def _key(chat_id: int, user_id: int) -> str:
    return f"{chat_id}:{user_id}"


def _welcome_caption(first_name: str) -> str:
    # Keep this short because it’s a GIF caption.
    return (
        f"Welcome, {first_name} 👋\n"
        "I’m Penny, the AI Advisor for Paycheck Labs.\n\n"
        "To chat here, please verify:\n"
        "Tap Verify ✅ below."
    )


def _keyboard(chat_id: int, user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Paycheck Website", url=PAYCHECK_WEBSITE),
            InlineKeyboardButton("Paycheck X", url=PAYCHECK_X),
        ],
        [
            InlineKeyboardButton("Paycheck Medium", url=PAYCHECK_MEDIUM),
        ],
        [
            InlineKeyboardButton("Verify ✅", callback_data=f"{CB_VERIFY_PREFIX}{chat_id}:{user_id}"),
        ],
    ])


def _restricted_perms() -> ChatPermissions:
    # Hard gate: no sending messages/media until verified.
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
    # You can loosen/tighten these based on your group rules.
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
    # Restrict user immediately on join
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=_restricted_perms(),
        )
    except Exception:
        log.exception("Failed to restrict user %s in chat %s", user_id, chat_id)


async def _lift_gate(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Restore permissions when verified
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=_allowed_perms(),
        )
    except Exception:
        log.exception("Failed to unrestrict user %s in chat %s", user_id, chat_id)


async def _timeout_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Runs after VERIFY_TIMEOUT_SECONDS to handle unverified users.
    """
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    user_id = data.get("user_id")
    k = _key(chat_id, user_id)

    pv = PENDING.get(k)
    if not pv:
        return  # already verified/cleared

    # Still pending -> enforce action
    if KICK_IF_NOT_VERIFIED:
        try:
            await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
            await context.bot.unban_chat_member(chat_id=chat_id, user_id=user_id)  # makes it a "kick"
        except Exception:
            log.exception("Failed to kick unverified user %s in chat %s", user_id, chat_id)

    # Clean up state
    PENDING.pop(k, None)

    # Optional: update the welcome message buttons/caption to show expired
    if pv.welcome_message_id:
        try:
            await context.bot.edit_message_caption(
                chat_id=chat_id,
                message_id=pv.welcome_message_id,
                caption="Verification window expired. Please re-join and verify.",
            )
        except Exception:
            pass


async def on_chat_member_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Fires on membership changes. We care about "user joined".
    """
    result = update.chat_member
    if not result:
        return

    chat = result.chat
    new = result.new_chat_member
    old = result.old_chat_member

    # Only handle real users, not the bot itself
    user = new.user
    if user.is_bot:
        return

    # Detect join event:
    # old: left/kicked -> new: member/restricted
    joined = (old.status in [ChatMemberStatus.LEFT, ChatMemberStatus.KICKED]) and (
        new.status in [ChatMemberStatus.MEMBER, ChatMemberStatus.RESTRICTED]
    )
    if not joined:
        return

    chat_id = chat.id
    user_id = user.id
    first_name = user.first_name or "there"

    # 1) Restrict immediately (hard gate)
    await _apply_hard_gate(chat_id, user_id, context)

    # 2) Send welcome GIF + buttons
    caption = _welcome_caption(first_name)
    markup = _keyboard(chat_id, user_id)

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
            # Local file path. Works only if file exists on the deployed filesystem.
            with open(WELCOME_GIF_LOCAL_PATH, "rb") as f:
                sent = await context.bot.send_animation(
                    chat_id=chat_id,
                    animation=f,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
        else:
            # Fallback: no GIF set
            sent = await context.bot.send_message(
                chat_id=chat_id,
                text=caption,
                reply_markup=markup,
            )
    except Exception:
        log.exception("Failed sending welcome content in chat %s", chat_id)

    # 3) Store pending verification + schedule timeout enforcement
    pv = PendingVerification(
        chat_id=chat_id,
        user_id=user_id,
        created_at=time.time(),
        welcome_message_id=(sent.message_id if sent else None),
    )
    PENDING[_key(chat_id, user_id)] = pv

    # Replace existing job (in case they re-join quickly)
    job_name = f"verify_timeout:{chat_id}:{user_id}"
    for j in context.job_queue.get_jobs_by_name(job_name):
        j.schedule_removal()

    context.job_queue.run_once(
        _timeout_job,
        when=VERIFY_TIMEOUT_SECONDS,
        name=job_name,
        data={"chat_id": chat_id, "user_id": user_id},
    )


async def on_verify_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles clicking Verify ✅.
    """
    query = update.callback_query
    if not query or not query.data:
        return

    if not query.data.startswith(CB_VERIFY_PREFIX):
        return

    await query.answer()  # remove "loading..."

    # Parse payload: verify_gate:<chat_id>:<user_id>
    payload = query.data[len(CB_VERIFY_PREFIX):]
    try:
        chat_id_str, user_id_str = payload.split(":")
        chat_id = int(chat_id_str)
        target_user_id = int(user_id_str)
    except Exception:
        return

    clicker = query.from_user
    if not clicker:
        return

    # Only the target user can verify themselves
    if clicker.id != target_user_id:
        await query.answer("This verify button isn’t for you.", show_alert=True)
        return

    k = _key(chat_id, target_user_id)
    pv = PENDING.get(k)
    if not pv:
        # Already verified/expired
        try:
            await query.edit_message_caption(caption="You’re already verified ✅")
        except Exception:
            pass
        return

    # Lift restrictions
    await _lift_gate(chat_id, target_user_id, context)

    # Clear state + cancel timeout job
    PENDING.pop(k, None)
    job_name = f"verify_timeout:{chat_id}:{target_user_id}"
    for j in context.job_queue.get_jobs_by_name(job_name):
        j.schedule_removal()

    # Update the welcome message to reflect success (optional)
    try:
        # Keep same links but remove Verify button
        verified_markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Paycheck Website", url=PAYCHECK_WEBSITE),
                InlineKeyboardButton("Paycheck X", url=PAYCHECK_X),
            ],
            [InlineKeyboardButton("Paycheck Medium", url=PAYCHECK_MEDIUM)],
        ])
        await query.edit_message_caption(
            caption="Verified ✅ Welcome in!",
            reply_markup=verified_markup,
        )
    except Exception:
        pass


def register_welcome_gate_handlers(application) -> None:
    """
    Call this from your main bot setup.
    """
    application.add_handler(ChatMemberHandler(on_chat_member_update, ChatMemberHandler.CHAT_MEMBER))
    application.add_handler(CallbackQueryHandler(on_verify_button, pattern=f"^{CB_VERIFY_PREFIX}"))
