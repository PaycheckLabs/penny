"""
knowledge_base.py

Static, lightweight reference notes for Penny.

IMPORTANT:
- These notes should NOT be injected for general conversation.
- We only inject when the user clearly asks about Paycheck Labs / Checks / related products.
"""

from __future__ import annotations

from typing import Dict, List


# -----------------------------
# High-confidence activation gate
# -----------------------------

def _is_checks_query(text: str) -> bool:
    t = (text or "").lower()
    triggers = [
        "paycheck labs",
        "checks platform",
        "check token",
        "$check",
        "checks.xyz",
        "checks whitepaper",
        "whitepaper",
        "checks roadmap",
        "roadmap for checks",
        "paychain",
        "paymart",
        "nft check",
        "nft checks",
        "mint a check",
        "redeem a check",
        "penny bot",
        "heypennybot",
    ]
    return any(k in t for k in triggers)


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


# -----------------------------
# KB Sections (keep short & factual)
# -----------------------------

KB_SECTIONS: Dict[str, str] = {
    "paycheck_overview": (
        "Paycheck Labs is building a blockchain and AI-focused ecosystem centered on programmable payments.\n"
        "Products mentioned across docs and discussions include the Checks Platform (programmable NFT Checks),\n"
        "Check Token ($CHECK), and planned future products like Paymart and Paychain."
    ),
    "checks_platform_summary": (
        "Checks Platform (NFT Checks protocol): users can mint programmable NFT Checks that represent value and rules.\n"
        "Common concepts include vesting schedules, escrow-like flows, auto-compounding/DeFi integrations, and redemption.\n"
        "If details are not present in synced docs, say so rather than guessing."
    ),
    "check_token_summary": (
        "Check Token ($CHECK): Paycheck Labs ecosystem token. Supply is described as 100B in prior internal notes.\n"
        "Utility is tied to the Checks Platform (e.g., supporting mint/redeem workflows and ecosystem incentives).\n"
        "If a user asks for exact tokenomics, prefer synced docs/whitepaper context and provide qualifiers."
    ),
    "roadmap_hint": (
        "Roadmap questions: answer using synced docs excerpts first.\n"
        "If the specific item isn’t present in synced docs, say it’s not confirmed in the synced set and ask\n"
        "if they want to share the exact link/section they’re referencing."
    ),
}

# Keywords are intentionally specific to avoid accidental triggers.
SECTION_KEYWORDS: Dict[str, List[str]] = {
    "paycheck_overview": ["paycheck labs"],
    "checks_platform_summary": ["checks platform", "nft check", "nft checks", "mint a check", "redeem a check"],
    "check_token_summary": ["check token", "$check"],
    "roadmap_hint": ["roadmap", "post-mvp", "post mvp", "coming later", "future features"],
}


def build_kb_context(user_text: str) -> str:
    """
    Return a small reference context blob, or "" if we should not inject anything.
    """
    if not _is_checks_query(user_text):
        return ""

    selected: List[str] = []
    for key, kws in SECTION_KEYWORDS.items():
        if _contains_any(user_text, kws):
            selected.append(KB_SECTIONS[key])

    # If user clearly asked about Paycheck/Checks but nothing matched, give a tiny general section.
    if not selected:
        selected.append(KB_SECTIONS["checks_platform_summary"])

    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for s in selected:
        if s not in seen:
            seen.add(s)
            out.append(s)

    return "\n\n".join(out).strip()
