"""
answer.py

Lightweight retrieval from locally-synced Checks docs.

This file does NOT call external services.
It reads markdown/text files under: docs/checks_whitepaper/
and returns a short "developer" context block to inject into the model.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple


DOCS_ROOT = Path(__file__).resolve().parent.parent  # repo root (src/..)
CHECKS_DOCS_DIR = DOCS_ROOT / "docs" / "checks_whitepaper"

_ALLOWED_EXTS = {".md", ".txt"}


def _keywords(query: str) -> List[str]:
    q = query.lower()
    stop = {
        "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
        "with", "from", "into", "your", "about", "tell", "more", "some", "does", "doing",
        "have", "will", "would", "should", "could", "just", "like", "than", "then",
        "please", "penny", "checks", "check", "token", "platform", "whitepaper",
    }
    words = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", q)
    kws = [w for w in words if len(w) >= 4 and w not in stop]

    seen = set()
    out: List[str] = []
    for w in kws:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:12]


def _is_checks_related(query: str) -> bool:
    q = query.lower()
    triggers = [
        "checks", "check token", "$check", "nft check", "whitepaper", "gitbook",
        "post-mvp", "mvp", "roadmap", "auto-invest", "auto investment", "autoinvest",
        "vesting", "escrow", "redeem", "mint",
    ]
    return any(t in q for t in triggers)


@lru_cache(maxsize=1)
def _load_docs() -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    if not CHECKS_DOCS_DIR.exists():
        return docs

    for p in CHECKS_DOCS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in _ALLOWED_EXTS:
            try:
                docs.append((str(p.relative_to(DOCS_ROOT)), p.read_text(encoding="utf-8", errors="replace")))
            except Exception:
                continue
    return docs


def _chunk_text(text: str, max_chunk_chars: int = 1800) -> List[str]:
    parts = re.split(r"\n(?=(?:#{1,6}\s)|(?:\n{2,}))", text)
    cleaned = [p.strip() for p in parts if p and p.strip()]

    chunks: List[str] = []
    buf = ""
    for part in cleaned:
        if not buf:
            buf = part
            continue
        if len(buf) + 2 + len(part) <= max_chunk_chars:
            buf = buf + "\n\n" + part
        else:
            chunks.append(buf)
            buf = part
    if buf:
        chunks.append(buf)
    return chunks


def _score_chunk(chunk: str, kws: List[str]) -> int:
    c = chunk.lower()
    score = 0
    for w in kws:
        score += min(c.count(w), 10)

    if "post-mvp" in c:
        score += 6
    if "roadmap" in c:
        score += 3
    if "auto-invest" in c or "auto investment" in c or "autoinvest" in c:
        score += 10

    return score


def get_checks_context(user_text: str, max_chars: int = 6000) -> str:
    if not _is_checks_related(user_text):
        return ""

    docs = _load_docs()
    if not docs:
        return ""

    kws = _keywords(user_text) or ["post-mvp", "roadmap", "feature", "mvp"]

    scored: List[Tuple[int, str, str]] = []  # (score, path, chunk)
    for rel_path, text in docs:
        for chunk in _chunk_text(text):
            s = _score_chunk(chunk, kws)
            if s > 0:
                scored.append((s, rel_path, chunk))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)

    picked: List[Tuple[str, str]] = []
    used_files = set()
    for score, rel_path, chunk in scored:
        if rel_path not in used_files or len(used_files) < 2:
            picked.append((rel_path, chunk))
            used_files.add(rel_path)
        if len(picked) >= 4:
            break

    out_lines: List[str] = []
    out_lines.append(
        "CHECKS_DOCS_CONTEXT (local, synced from checks-gitbook). "
        "Use this as the primary source for Checks/whitepaper questions. "
        "If the answer is not in these excerpts, say so and ask a follow-up."
    )
    out_lines.append("")

    for rel_path, chunk in picked:
        snippet = chunk.strip()
        out_lines.append(f"[Source: {rel_path}]")
        out_lines.append(snippet)
        out_lines.append("")

    ctx = "\n".join(out_lines).strip()

    if len(ctx) > max_chars:
        ctx = ctx[:max_chars].rstrip() + "\n\n[...truncated...]"

    return ctx
