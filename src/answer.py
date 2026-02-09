import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

# This module is intentionally lightweight.
# It exists so bot.py can optionally delegate answering logic here without crashing.

CHECKS_DOCS_DIR = os.getenv("CHECKS_DOCS_DIR", "").strip()


def _candidate_checks_docs_dirs() -> List[Path]:
    here = Path(__file__).resolve()
    src_dir = here.parent
    repo_root = src_dir.parent

    candidates: List[Path] = []
    if CHECKS_DOCS_DIR:
        candidates.append(Path(CHECKS_DOCS_DIR))

    candidates.append(src_dir / "data" / "checks_whitepaper")
    candidates.append(repo_root / "src" / "data" / "checks_whitepaper")
    candidates.append(repo_root / "docs" / "checks_whitepaper")
    candidates.append(repo_root / "data" / "checks_whitepaper")
    candidates.append(repo_root / "checks_whitepaper")

    seen = set()
    out: List[Path] = []
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            rp = p
        if str(rp) not in seen:
            seen.add(str(rp))
            out.append(rp)
    return out


def _read_markdown_files(root: Path, max_files: int = 200) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    if not root.exists() or not root.is_dir():
        return items

    count = 0
    for path in sorted(root.rglob("*")):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".markdown", ".mdx", ".txt"}:
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        items.append((str(path.relative_to(root)), txt))
        count += 1
    return items


def _score_doc_text(query_terms: List[str], text: str) -> int:
    if not query_terms or not text:
        return 0
    t = text.lower()
    score = 0
    for term in query_terms:
        hits = t.count(term)
        if hits:
            score += min(hits, 6)
    return score


def get_checks_docs_context(user_text: str, char_budget: int = 6000) -> Tuple[str, str]:
    user_text_l = (user_text or "").lower()
    triggers = [
        "checks", "check token", "$check", "nft check", "checks platform",
        "whitepaper", "roadmap", "post-mvp", "auto-invest", "autoinvest", "auto investment"
    ]
    if not any(t in user_text_l for t in triggers):
        return ("", "")

    raw_terms = re.findall(r"[a-z0-9\-]{3,}", user_text_l)
    raw_terms += ["post-mvp", "postmvp", "auto-invest", "autoinvest", "investment", "roadmap"]
    query_terms = sorted(set(raw_terms))

    for root in _candidate_checks_docs_dirs():
        files = _read_markdown_files(root)
        if not files:
            continue

        ranked: List[Tuple[int, str, str]] = []
        for rel, txt in files:
            s = _score_doc_text(query_terms, txt)
            if s > 0:
                ranked.append((s, rel, txt))

        ranked.sort(key=lambda x: x[0], reverse=True)

        parts: List[str] = []
        used = 0
        if ranked:
            for s, rel, txt in ranked[:8]:
                snippet = txt.strip()
                if len(snippet) > 1800:
                    snippet = snippet[:1800] + "\n…"
                block = f"FILE: {rel}\n{snippet}"
                if used + len(block) + 2 > char_budget:
                    break
                parts.append(block)
                used += len(block) + 2
        else:
            manifest = "\n".join([f"- {rel}" for rel, _ in files[:40]])
            parts.append("No strong keyword match. Available files (partial):\n" + manifest)

        return (str(root), "\n\n".join(parts).strip())

    return ("", "")


def build_messages(system_prompt: str, conversation: List[Tuple[str, str]], user_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    root, ctx = get_checks_docs_context(user_text)
    if ctx:
        messages.append({
            "role": "system",
            "content": (
                "Checks Docs Context (synced markdown excerpts; treat as authoritative within this bot)\n"
                f"Docs root: {root}\n\n{ctx}"
            ),
        })

    for role, content in conversation[-12:]:
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})
    return messages


def openai_reply(
    client,
    system_prompt: str,
    conversation: List[Tuple[str, str]],
    user_text: str,
    model: str = "gpt-5-nano",
    max_output_tokens: int = 280,
) -> str:
    """
    Minimal OpenAI response helper for bot.py to import.
    Expects `client` to be an OpenAI() client instance.
    """
    messages = build_messages(system_prompt, conversation, user_text)
    resp = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_output_tokens,
    )
    return (resp.output_text or "").strip()
