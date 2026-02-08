# src/knowledge_base.py
"""
Paycheck Labs / Penny Knowledge Base (v1)

Goal:
- Provide a curated, safe, and editable source of truth that Penny can use for:
  - Company info (Paycheck Labs)
  - Penny assistant info (how to use, testing group norms)
  - Official links
  - High-level product references (with clear "planned vs in development" labels)

This is intentionally lightweight for MVP:
- No database
- No embeddings
- Simple keyword routing + snippet assembly

Later upgrades:
- Chunking + embeddings + retrieval
- Admin-only KB updates from Telegram
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# -----------------------------
# Canonical links (company-focused)
# -----------------------------
PAYCHECK_LINKS: Dict[str, str] = {
    "Paycheck Website": "https://www.paycheck.io/",
    "Paycheck X": "https://x.com/PaycheckIO",
    "Paycheck Medium": "https://medium.com/@paycheck",
}

# Product / documentation links (high-level)
PRODUCT_LINKS: Dict[str, str] = {
    # Provided by you as the key canonical domain for Checks documentation
    "Checks Website": "https://checks.xyz/",
    # Whitepaper URL can vary. We reference it as "on checks.xyz" instead of guessing a path.
    "Checks Whitepaper": "See the Whitepaper on checks.xyz",
}


# -----------------------------
# Knowledge base sections
# Keep these short, factual, and easy to edit.
# -----------------------------
@dataclass(frozen=True)
class KBSection:
    id: str
    title: str
    keywords: Tuple[str, ...]
    content: str


KB_SECTIONS: List[KBSection] = [
    KBSection(
        id="paycheck_overview",
        title="Paycheck Labs overview",
        keywords=("paycheck labs", "paycheck", "company", "team", "about", "who built you", "who made you"),
        content=(
            "Paycheck Labs is the team building Penny and other blockchain and AI projects.\n"
            "The fastest way to get official details is the website and the News page."
        ),
    ),
    KBSection(
        id="penny_overview",
        title="Penny overview",
        keywords=("penny", "assistant", "bot", "v1.1", "version", "what can you do", "features"),
        content=(
            "I’m Penny, an AI virtual assistant by Paycheck Labs.\n"
            "Right now I’m in active testing. The goal is to be helpful, fast, and reliable.\n"
            "If something breaks or looks wrong, share what you typed and what happened so we can fix it."
        ),
    ),
    KBSection(
        id="penny_how_to_talk",
        title="How to talk to Penny",
        keywords=("how do i", "talk to you", "use penny", "@heypennybot", "/penny", "reply"),
        content=(
            "In groups, you can talk to me in three ways:\n"
            "• @HeyPennyBot\n"
            "• /penny + your message\n"
            "• Reply to one of my messages\n"
            "In DMs, just type normally."
        ),
    ),
    KBSection(
        id="testing_group_rules",
        title="Testing group expectations",
        keywords=("testing group", "test", "bug", "issue", "feedback", "report", "working", "not working"),
        content=(
            "This is a testing group, so rough edges are expected.\n"
            "If I mess up, please share:\n"
            "• what you typed\n"
            "• what you expected\n"
            "• what happened instead\n"
            "That feedback helps us improve quickly."
        ),
    ),
    KBSection(
        id="welcome_gate",
        title="Welcome gate verification",
        keywords=("verify", "verification", "captcha", "welcome gate", "why can’t i chat", "restricted", "unlock"),
        content=(
            "This group uses a welcome gate to reduce spam.\n"
            "If you can’t chat yet, tap the correct verification button under the welcome message.\n"
            "Once verified, you’ll be able to send messages normally."
        ),
    ),
    KBSection(
        id="official_links",
        title="Official Paycheck links",
        keywords=("links", "website", "twitter", "x", "medium", "official", "where can i find"),
        content=(
            "Official Paycheck links:\n"
            f"• Website: {PAYCHECK_LINKS['Paycheck Website']}\n"
            f"• X: {PAYCHECK_LINKS['Paycheck X']}\n"
            f"• Medium: {PAYCHECK_LINKS['Paycheck Medium']}"
        ),
    ),

    # -----------------------------
    # Products and initiatives (high-level)
    # -----------------------------
    KBSection(
        id="paychain_planned",
        title="Paychain blockchain (planned)",
        keywords=("paychain", "layer 1", "l1", "payments network", "block explorer", "payscan", "virtual machine", "pvm"),
        content=(
            "Paychain is a planned future blockchain initiative focused on payments and tokenization.\n"
            "It is described as a custom Layer 1 network, with supporting components like a block explorer."
        ),
    ),
    KBSection(
        id="checks_platform_dev",
        title="Checks Platform (in development)",
        keywords=("checks", "checks platform", "nft checks", "check token", "check token $check", "protocol", "checks.xyz", "whitepaper"),
        content=(
            "The Checks Platform is in development and centers around NFT Checks, which are programmable financial agreements.\n"
            "For the most accurate and complete product info, use the canonical docs and Whitepaper on:\n"
            f"• {PRODUCT_LINKS['Checks Website']}"
        ),
    ),
    KBSection(
        id="tech_foundation",
        title="Technology foundation",
        keywords=("technology", "tech", "stack", "ai", "artificial intelligence", "llm", "machine learning", "oracles", "zk", "zero-knowledge", "encryption", "mpc", "did", "soulbound", "nft"),
        content=(
            "Paycheck Labs references a technology foundation spanning blockchain and AI.\n"
            "Examples mentioned include smart contract systems, AI and large language models, machine learning, and privacy/security primitives.\n"
            "Details evolve, so use the website for the latest descriptions."
        ),
    ),

    # -----------------------------
    # FAQ / News / Associate Program
    # -----------------------------
    KBSection(
        id="faq_help",
        title="FAQ on the website",
        keywords=("faq", "frequently asked", "questions", "what is paycheck labs", "security", "secure", "benefit my business"),
        content=(
            "The Paycheck website includes an FAQ section.\n"
            "If you want official phrasing for questions about what Paycheck Labs is, how blockchain can help, or security posture, "
            "the FAQ is the best source."
        ),
    ),
    KBSection(
        id="news_updates",
        title="News updates",
        keywords=("news", "updates", "latest", "announcements", "release", "listing", "launch"),
        content=(
            "Official announcements and updates can be found on the Paycheck website’s News section.\n"
            "For quick links, use the buttons below, and check the News page for the latest posts."
        ),
    ),
    KBSection(
        id="associate_program",
        title="Paycheck Labs Associate Program (ASC)",
        keywords=("associate", "associate program", "asc", "workforce", "apply", "sign up", "rewards", "benefits"),
        content=(
            "Paycheck Labs has an Associate Program (often shortened to ASC).\n"
            "People can read about the program on the website and apply to become an associate."
        ),
    ),
    KBSection(
        id="roadmap_hint",
        title="What’s coming next",
        keywords=("roadmap", "next", "coming", "soon", "plans", "knowledge base", "price", "crypto", "api"),
        content=(
            "Planned upgrades include expanding the Paycheck Labs knowledge base, richer commands, and API features "
            "like crypto price checks.\n"
            "We’re building step by step and validating everything in the testing group."
        ),
    ),
]


# -----------------------------
# Retrieval helpers (simple MVP)
# -----------------------------
def normalize(text: str) -> str:
    return (text or "").lower().strip()


def find_relevant_sections(user_text: str, max_sections: int = 3) -> List[KBSection]:
    """
    Lightweight keyword match. Picks the most relevant sections for the prompt.
    """
    t = normalize(user_text)
    if not t:
        return []

    scored: List[Tuple[int, KBSection]] = []
    for sec in KB_SECTIONS:
        score = 0
        for kw in sec.keywords:
            if kw in t:
                score += 2
        for w in sec.title.lower().split():
            if w and w in t:
                score += 1
        if score > 0:
            scored.append((score, sec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sections]]


def build_kb_context(user_text: str, max_sections: int = 3) -> str:
    """
    Returns a short block of context to inject into the model prompt when relevant.
    """
    secs = find_relevant_sections(user_text, max_sections=max_sections)
    if not secs:
        return ""

    parts: List[str] = ["Paycheck Labs Knowledge Base (curated):"]
    for sec in secs:
        parts.append(f"\n[{sec.title}]\n{sec.content}")

    return "\n".join(parts)
