# src/answer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

# Small stopword list (keep it tiny; we just need basic scoring)
_STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","been","being","it","this","that","these","those","as","not","no","do",
    "does","did","can","could","should","would","will","just","about","into","over","under","more","most",
    "your","you","we","our","they","their","them","i","me","my"
}

def _repo_root() -> Path:
    # src/answer.py -> src -> repo root
    return Path(__file__).resolve().parents[1]

def _normalize_words(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]{2,}", text.lower())
    return [w for w in words if w not in _STOPWORDS]

def _iter_md_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted([p for p in dir_path.rglob("*.md") if p.is_file()])

def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def _score_chunk(q_words: List[str], chunk: str) -> int:
    if not q_words:
        return 0
    c_words = set(_normalize_words(chunk))
    score = 0
    for w in q_words:
        if w in c_words:
            score += 2
    # bonus for exact phrase-ish matches
    q_join = " ".join(q_words[:6])
    if q_join and q_join in chunk.lower():
        score += 3
    return score

def build_checks_context(
    user_text: str,
    dirs: Optional[Iterable[str]] = None,
    max_chars: int = 6500,
    max_files: int = 30,
    max_chunks: int = 6,
) -> str:
    """
    Returns a developer-message string containing the most relevant excerpts
    from the synced Checks whitepaper/docs.

    This is intentionally lightweight (no embeddings) but works well for
    strong keyword sections like "Auto-Investment".
    """
    repo = _repo_root()

    # Where your workflow/script likely placed the synced markdown.
    # From your screenshots, /docs has CORE_CONTRACT.md, PROMPT_PACK.md, ROADMAP.md.
    # Also keep a fallback for future: src/data/checks_whitepaper
    if dirs is None:
        dirs = ["docs", "src/data/checks_whitepaper", "src/data/checks_whitepaper/docs"]

    search_dirs: List[Path] = []
    for d in dirs:
        p = (repo / d).resolve()
        search_dirs.append(p)

    md_files: List[Path] = []
    for d in search_dirs:
        md_files.extend(_iter_md_files(d))

    # De-dupe while preserving order
    seen = set()
    unique_files: List[Path] = []
    for f in md_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    if not unique_files:
        return ""

    q_words = _normalize_words(user_text)
    if not q_words:
        return ""

    scored: List[Tuple[int, str, str]] = []  # (score, file_rel, chunk)
    for f in unique_files[:max_files]:
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Light trim per file to keep things fast
        raw = raw.strip()
        if not raw:
            continue

        chunks = _chunk_text(raw)
        rel = str(f.relative_to(repo))

        for ch in chunks:
            s = _score_chunk(q_words, ch)
            if s > 0:
                scored.append((s, rel, ch))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[:max_chunks]

    # Build the final context block
    out_parts: List[str] = []
    out_parts.append(
        "CHECKS WHITEPAPER / DOCS EXCERPTS (use these as the source of truth when relevant):\n"
        "- If the answer is not supported by these excerpts, say you can’t find it in the synced docs.\n"
        "- Prefer quoting or paraphrasing the excerpts.\n"
    )

    for idx, (s, rel, ch) in enumerate(picked, start=1):
        ch = ch.strip()
        out_parts.append(f"\n[Excerpt {idx} | {rel} | score={s}]\n{ch}\n")

    combined = "\n".join(out_parts).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "\n\n[truncated]\n"

    return combined
