"""
checks_rag.py

Fast, keyword-based retrieval over the locally-synced Checks docs/whitepaper markdown.

Why this exists:
- Embedding-based RAG can be slow/costly and may introduce startup delays on Railway.
- This module provides a simple, stable retriever that works immediately once files are present.

Usage:
    from checks_rag import retrieve_checks_context
    ctx = retrieve_checks_context("Tell me about Auto-Investment post-MVP")

This module is intentionally side-effect free at import time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

_WORD_RE = re.compile(r"[a-z0-9_\$]{2,}", re.IGNORECASE)

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you", "are", "was", "were", "what",
    "when", "where", "which", "who", "whom", "why", "how", "can", "could", "should", "would",
    "about", "into", "over", "under", "then", "than", "also", "just", "like", "have", "has", "had",
}

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _tokens(text: str) -> List[str]:
    return [t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOP]

def _candidate_dirs() -> List[Path]:
    root = _repo_root()
    env_primary = os.getenv("CHECKS_DOCS_DIR", "").strip()
    env_fallback = os.getenv("CHECKS_DOCS_FALLBACK_DIRS", "").strip()

    dirs: List[Path] = []
    if env_primary:
        dirs.append(Path(env_primary))

    dirs.extend(
        [
            root / "docs",
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

    # de-dupe
    seen = set()
    out: List[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def _iter_md_files(base: Path) -> Iterable[Path]:
    if not base.exists() or not base.is_dir():
        return []
    files = list(base.glob("*.md")) + list(base.glob("*.markdown"))
    files += list(base.glob("**/*.md"))[:200]
    return files

def _read_files(dirs: Sequence[Path], max_files: int = 250, max_bytes: int = 900_000) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    count = 0
    root = _repo_root()
    for d in dirs:
        for fp in _iter_md_files(d):
            if count >= max_files:
                return docs
            try:
                if not fp.is_file():
                    continue
                if fp.stat().st_size > max_bytes:
                    continue
                text = fp.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    try:
                        src = str(fp.relative_to(root))
                    except Exception:
                        src = str(fp)
                    docs.append((src, text))
                    count += 1
            except Exception:
                continue
    return docs

def _split_blocks(text: str) -> List[str]:
    raw = re.split(r"\n\s*\n+", (text or "").strip())
    return [b.strip() for b in raw if b and b.strip()]

def _score(q_tokens: set, block: str) -> int:
    bt = set(_tokens(block))
    if not bt:
        return 0
    overlap = len(q_tokens & bt)
    if overlap == 0:
        return 0

    bl = block.lower()
    if "auto" in q_tokens and "investment" in q_tokens:
        if "auto-invest" in bl or "auto invest" in bl or "autoinvest" in bl:
            overlap += 6
    if "roadmap" in q_tokens and "roadmap" in bl:
        overlap += 3
    return overlap

def retrieve_checks_context(query: str, max_chars: int = 6000) -> str:
    """
    Return relevant excerpt blocks labeled with [source_file].

    Returns "" if nothing is found (e.g., docs not present or no match).
    """
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return ""

    docs = _read_files(_candidate_dirs())
    if not docs:
        return ""

    scored: List[Tuple[int, str, str]] = []
    for src, text in docs:
        for block in _split_blocks(text):
            s = _score(q_tokens, block)
            if s > 0:
                scored.append((s, src, block))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    used = 0
    for s, src, block in scored[:40]:
        piece = f"[{src}]\n{block}".strip()
        if used + len(piece) + 2 > max_chars:
            break
        out.append(piece)
        used += len(piece) + 2

    return "\n\n".join(out).strip()
