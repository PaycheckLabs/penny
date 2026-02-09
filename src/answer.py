# src/answer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

_STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","been","being","it","this","that","these","those","as","not","no","do",
    "does","did","can","could","should","would","will","just","about","into","over","under","more","most",
    "your","you","we","our","they","their","them","i","me","my"
}

_EXCLUDE_DIRS = {
    ".git", ".github", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", "dist", "build"
}

def _repo_root() -> Path:
    # /app/src/answer.py -> /app/src -> /app
    return Path(__file__).resolve().parents[1]

def _normalize_words(text: str) -> List[str]:
    # keep hyphenated terms useful by splitting into words
    text = text.lower().replace("-", " ")
    words = re.findall(r"[a-z0-9]{2,}", text)
    return [w for w in words if w not in _STOPWORDS]

def _iter_md_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    out: List[Path] = []
    for p in dir_path.rglob("*.md"):
        if not p.is_file():
            continue
        if any(part in _EXCLUDE_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)

def _iter_md_files_repo_wide(repo: Path) -> List[Path]:
    out: List[Path] = []
    for p in repo.rglob("*.md"):
        if not p.is_file():
            continue
        if any(part in _EXCLUDE_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)

def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def _score_chunk(q_words: List[str], chunk: str) -> int:
    if not q_words:
        return 0

    c_low = chunk.lower()
    c_words = set(_normalize_words(chunk))

    score = 0
    # Base word hits
    for w in q_words:
        if w in c_words:
            score += 2

    # Strong boost for exact phrase-ish matches of key terms
    # specifically help for "auto investment" / "auto-investment"
    if "auto" in q_words and "investment" in q_words:
        if "auto investment" in c_low or "auto-investment" in c_low:
            score += 10
        # still boost if both words appear anywhere
        if "auto" in c_words and "investment" in c_words:
            score += 5

    # Small boost if the query contains a rare-looking token and it appears
    # (example: "autoinvestment")
    for w in q_words:
        if len(w) >= 10 and w in c_low:
            score += 3

    return score

def build_checks_context(
    user_text: str,
    dirs: Optional[Iterable[str]] = None,
    max_chars: int = 8000,
    max_files: int = 80,
    max_chunks: int = 8,
) -> str:
    """
    Returns a developer-message string containing the most relevant excerpts
    from synced Checks docs/whitepaper markdown.

    If no matches are found in the provided dirs, falls back to scanning all .md
    in the repo (excluding .github/.git/etc.) so we don't miss where the sync landed.
    """
    repo = _repo_root()

    # Default to the common places you’ve used so far + what your screenshots show.
    if dirs is None:
        dirs = [
            "docs",
            "src/data/checks_whitepaper",
            "src/data/checks_whitepaper/docs",
        ]

    q_words = _normalize_words(user_text)
    if not q_words:
        return ""

    # Build directory list
    search_dirs: List[Path] = []
    for d in dirs:
        d = (d or "").strip()
        if not d:
            continue
        search_dirs.append((repo / d).resolve())

    # Collect markdown files from those dirs
    md_files: List[Path] = []
    for d in search_dirs:
        md_files.extend(_iter_md_files(d))

    # If nothing found in those dirs, fallback to repo-wide scan
    if not md_files:
        md_files = _iter_md_files_repo_wide(repo)

    # De-dupe while preserving order
    seen = set()
    files: List[Path] = []
    for f in md_files:
        if f not in seen:
            seen.add(f)
            files.append(f)

    if not files:
        return ""

    scored: List[Tuple[int, str, str]] = []  # (score, relpath, chunk)
    for f in files[:max_files]:
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not raw:
            continue

        rel = str(f.relative_to(repo)) if f.is_relative_to(repo) else str(f)
        for ch in _chunk_text(raw):
            s = _score_chunk(q_words, ch)
            if s > 0:
                scored.append((s, rel, ch.strip()))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[:max_chunks]

    out: List[str] = []
    out.append(
        "CHECKS WHITEPAPER / DOCS EXCERPTS (use these as source of truth when relevant):\n"
        "- If the answer is not supported by these excerpts, say you can't find it in the synced docs.\n"
        "- Prefer quoting or paraphrasing these excerpts.\n"
    )

    for i, (s, rel, ch) in enumerate(picked, start=1):
        out.append(f"\n[Excerpt {i} | {rel} | score={s}]\n{ch}\n")

    combined = "\n".join(out).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "\n\n[truncated]\n"

    return combined
