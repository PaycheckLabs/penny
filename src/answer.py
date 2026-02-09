"""
answer.py

Single responsibility: turn (history + user_text) into a model reply, optionally
attaching context from synced Checks / Paycheck Labs docs.

Design goals:
- Fast: keyword-based retrieval from local markdown files (no embeddings by default).
- Safe: only attach company/product context when user explicitly asks.
- Stable: no side effects at import time; pure functions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional: local static KB (lightweight). If missing, we still work.
try:
    from knowledge_base import build_kb_context as _build_kb_context  # type: ignore
except Exception:
    _build_kb_context = None  # type: ignore


_WORD_RE = re.compile(r"[a-z0-9_\$]{2,}", re.IGNORECASE)

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you", "are", "was", "were", "what",
    "when", "where", "which", "who", "whom", "why", "how", "can", "could", "should", "would",
    "about", "into", "over", "under", "then", "than", "also", "just", "like", "have", "has", "had",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokens(text: str) -> List[str]:
    return [t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOP]


def _is_checks_query(text: str) -> bool:
    """
    High-confidence trigger so Penny doesn't talk about Paycheck Labs / Checks unless prompted.

    IMPORTANT: Do NOT trigger on generic words like "about", "company", "token", "wallet", etc.
    """
    t = (text or "").lower()
    triggers = [
        "paycheck labs",
        "checks platform",
        "check token",
        "$check",
        "checks.xyz",
        "paychain",
        "paymart",
        "penny bot",
        "heypennybot",
        "checks whitepaper",
        "whitepaper",
        "checks roadmap",
        "roadmap for checks",
        "nft check",
        "nft checks",
        "mint a check",
        "redeem a check",
    ]
    return any(k in t for k in triggers)


def _repo_root() -> Path:
    # src/answer.py -> repo root
    return Path(__file__).resolve().parent.parent


def _candidate_dirs() -> List[Path]:
    """
    Where the synced docs may land. We include conservative defaults and allow overrides.

    If you want to force a single location, set:
      CHECKS_DOCS_DIR=/app/docs   (Railway)
    """
    root = _repo_root()

    env_primary = os.getenv("CHECKS_DOCS_DIR", "").strip()
    env_fallback = os.getenv("CHECKS_DOCS_FALLBACK_DIRS", "").strip()

    dirs: List[Path] = []
    if env_primary:
        dirs.append(Path(env_primary))

    # Common locations (including where your workflow currently puts files)
    dirs.extend(
        [
            root / "docs",  # <-- your current synced files (CORE_CONTRACT.md, ROADMAP.md, etc.)
            root / "docs" / "checks_whitepaper",
            root / "docs" / "checks_whitepaper_md",
            root / "src" / "docs",
            root / "src" / "data" / "checks_whitepaper",
            root / "src" / "data" / "checks_whitepaper_md",
        ]
    )

    if env_fallback:
        for part in env_fallback.split(","):
            p = part.strip()
            if p:
                dirs.append(Path(p))

    # de-dupe while preserving order
    seen = set()
    out: List[Path] = []
    for d in dirs:
        try:
            key = str(d.resolve())
        except Exception:
            key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _iter_markdown_files(base: Path) -> Iterable[Path]:
    if not base.exists() or not base.is_dir():
        return []
    # Shallow + small recursive search (docs trees are usually shallow)
    files = list(base.glob("*.md")) + list(base.glob("*.markdown"))
    files += list(base.glob("**/*.md"))[:200]
    return files


def _read_files(dirs: Sequence[Path], max_files: int = 250, max_bytes: int = 900_000) -> List[Tuple[str, str]]:
    """
    Returns list of (source_name, text). We cap file count and size to stay fast.
    """
    docs: List[Tuple[str, str]] = []
    count = 0
    for d in dirs:
        for fp in _iter_markdown_files(d):
            if count >= max_files:
                return docs
            try:
                if not fp.is_file():
                    continue
                if fp.stat().st_size > max_bytes:
                    continue
                text = fp.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    docs.append((str(fp.relative_to(_repo_root())), text))
                    count += 1
            except Exception:
                continue
    return docs


def _split_blocks(text: str) -> List[str]:
    # split on blank lines; keeps headings with paragraphs
    raw = re.split(r"\n\s*\n+", text.strip())
    blocks = [b.strip() for b in raw if b and b.strip()]
    return blocks


def _score_block(q_tokens: set, block: str) -> int:
    b_tokens = set(_tokens(block))
    if not b_tokens:
        return 0
    overlap = len(q_tokens & b_tokens)
    if overlap == 0:
        return 0

    # small boosts for exact phrase matches
    bl = block.lower()
    if "auto" in q_tokens and "investment" in q_tokens:
        if "auto-invest" in bl or "auto invest" in bl or "autoinvest" in bl:
            overlap += 6
    if "$check" in q_tokens and "$check" in bl:
        overlap += 4
    if "roadmap" in q_tokens and "roadmap" in bl:
        overlap += 3

    return overlap


def extract_checks_context(query: str, max_chars: int = 6000) -> str:
    """
    Pull the most relevant markdown blocks from synced docs.

    Returns a single text blob that is safe to inject into a system message.
    """
    query = _normalize(query)
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return ""

    docs = _read_files(_candidate_dirs())
    if not docs:
        return ""

    scored: List[Tuple[int, str, str]] = []
    for source, text in docs:
        for block in _split_blocks(text):
            score = _score_block(q_tokens, block)
            if score > 0:
                scored.append((score, source, block))

    if not scored:
        # Fallback: substring search for "auto invest" or exact query tokens.
        q = query.lower()
        for source, text in docs:
            tl = text.lower()
            if ("auto" in q and "invest" in q and ("auto-invest" in tl or "auto invest" in tl or "autoinvest" in tl)):
                # grab nearby lines: keep it simple, take matching paragraphs
                chunks = []
                for block in _split_blocks(text):
                    bl = block.lower()
                    if "auto" in bl and "invest" in bl:
                        chunks.append(f"[{source}]\n{block}")
                out = "\n\n".join(chunks).strip()
                return out[:max_chars].strip()
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)

    out_parts: List[str] = []
    used_chars = 0
    used_sources = set()

    for score, source, block in scored[:40]:
        # Avoid repeating the same file too much
        if source in used_sources and score < 4:
            continue
        used_sources.add(source)

        piece = f"[{source}]\n{block}".strip()
        if used_chars + len(piece) + 2 > max_chars:
            break
        out_parts.append(piece)
        used_chars += len(piece) + 2

    return "\n\n".join(out_parts).strip()


def _build_messages(
    system_prompt: str,
    history: Sequence[Dict[str, str]],
    user_text: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]

    # Attach context only when user explicitly asks.
    ctx_parts: List[str] = []

    if _is_checks_query(user_text):
        docs_ctx = extract_checks_context(user_text)
        if docs_ctx:
            ctx_parts.append("Synced docs context (authoritative for Paycheck Labs / Checks questions):\n" + docs_ctx)

        if _build_kb_context is not None:
            kb_ctx = _build_kb_context(user_text) or ""
            kb_ctx = kb_ctx.strip()
            if kb_ctx:
                ctx_parts.append("Static reference notes (use only if relevant):\n" + kb_ctx)

    if ctx_parts:
        messages.append(
            {
                "role": "system",
                "content": "\n\n".join(ctx_parts).strip(),
            }
        )

    # history (already role/content dicts)
    for m in history:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text.strip()})
    return messages


def openai_reply(
    client,
    system_prompt: str,
    history: Sequence[Dict[str, str]],
    user_text: str,
    model: str = "gpt-4.1-mini",
    max_output_tokens: int = 550,
) -> str:
    """
    Synchronous OpenAI call (wrap this in asyncio.to_thread in async apps).
    """
    user_text = _normalize(user_text)
    if not user_text:
        return ""

    messages = _build_messages(system_prompt, history, user_text)

    resp = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_output_tokens,
        temperature=0.4,
    )
    return (resp.output_text or "").strip()
