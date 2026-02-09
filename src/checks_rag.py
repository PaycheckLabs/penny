import os
import re
import json
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI

# -------------------------
# Config
# -------------------------
EMBED_MODEL = os.getenv("CHECKS_EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("CHECKS_TOP_K", "6"))
MAX_CHUNK_CHARS = int(os.getenv("CHECKS_MAX_CHUNK_CHARS", "1600"))

# Where your workflow synced the whitepaper files:
# (adjust if your actual folder name differs)
WHITEPAPER_DIR = Path(os.getenv("CHECKS_WHITEPAPER_DIR", "src/data/checks_whitepaper"))

# Cache file stored in-repo so Railway can reuse it between deploys (if persistent disk),
# but also fine if it rebuilds occasionally.
CACHE_DIR = Path(os.getenv("CHECKS_CACHE_DIR", "src/data/.checks_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VEC_CACHE_PATH = CACHE_DIR / "checks_vectors.jsonl"
META_PATH = CACHE_DIR / "checks_meta.json"

client = OpenAI()


# -------------------------
# Helpers
# -------------------------
def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _cosine(a: List[float], b: List[float]) -> float:
    # safe cosine similarity
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _strip_md_noise(text: str) -> str:
    # Keep it readable for embeddings
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # remove code fences
    text = re.sub(r"<[^>]+>", " ", text)               # remove HTML
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Chunk by paragraphs, then pack into <= max_chars.
    Keeps semantic coherence without extra deps.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # if paragraph itself is huge, hard-split
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks

def _list_whitepaper_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    files = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".md", ".txt"):
            files.append(p)
    return sorted(files)

def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batched embeddings call.
    """
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    # OpenAI returns in the same order
    return [d.embedding for d in resp.data]


# -------------------------
# Index build / load
# -------------------------
def _current_corpus_fingerprint(files: List[Path]) -> str:
    """
    Fingerprint all file contents; changes when whitepaper changes.
    """
    h = hashlib.sha256()
    for f in files:
        h.update(str(f).encode("utf-8"))
        h.update(b"\0")
        h.update(_read_text_file(f).encode("utf-8", errors="ignore"))
        h.update(b"\0\0")
    return h.hexdigest()

def build_or_load_index(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Returns:
      {
        "items": [
          {"id": "...", "source": "path/to/file.md", "chunk": "...", "embedding": [..]},
          ...
        ]
      }
    """
    files = _list_whitepaper_files(WHITEPAPER_DIR)
    if not files:
        return {"items": []}

    fingerprint = _current_corpus_fingerprint(files)

    # Check metadata to see if we can reuse cached embeddings
    if not force_rebuild and META_PATH.exists() and VEC_CACHE_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint and meta.get("embed_model") == EMBED_MODEL:
                items = []
                with VEC_CACHE_PATH.open("r", encoding="utf-8") as f:
                    for line in f:
                        items.append(json.loads(line))
                return {"items": items}
        except Exception:
            pass

    # Rebuild
    items: List[Dict[str, Any]] = []
    batch_texts: List[str] = []
    batch_refs: List[Tuple[str, str]] = []  # (source, chunk)

    for f in files:
        raw = _read_text_file(f)
        cleaned = _strip_md_noise(raw)
        if not cleaned:
            continue

        for chunk in _chunk_text(cleaned):
            chunk_id = _sha256_text(f"{f.as_posix()}::{chunk[:200]}")
            items.append({
                "id": chunk_id,
                "source": str(f.as_posix()),
                "chunk": chunk,
                "embedding": None,  # fill later
            })
            batch_texts.append(chunk)
            batch_refs.append((str(f.as_posix()), chunk))

    # Embed in batches to avoid huge single requests
    embedded: List[List[float]] = []
    BATCH = 64
    for i in range(0, len(batch_texts), BATCH):
        embedded.extend(_embed_texts(batch_texts[i:i + BATCH]))

    for i, vec in enumerate(embedded):
        items[i]["embedding"] = vec

    # Save cache
    with VEC_CACHE_PATH.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    META_PATH.write_text(json.dumps({
        "fingerprint": fingerprint,
        "embed_model": EMBED_MODEL,
        "file_count": len(files),
        "item_count": len(items),
    }, indent=2), encoding="utf-8")

    return {"items": items}


# -------------------------
# Retrieval
# -------------------------
_INDEX: Optional[Dict[str, Any]] = None

def get_index() -> Dict[str, Any]:
    global _INDEX
    if _INDEX is None:
        _INDEX = build_or_load_index(force_rebuild=False)
    return _INDEX

def retrieve_checks_context(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    idx = get_index()
    items = idx.get("items", [])
    if not items:
        return []

    q_vec = _embed_texts([query])[0]

    scored = []
    for it in items:
        score = _cosine(q_vec, it["embedding"])
        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [it for _, it in scored[:top_k]]

    # Return minimal fields the prompt needs
    return [{"source": p["source"], "chunk": p["chunk"]} for p in picked]


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Turn chunks into a citation-friendly block the model can quote/summarize from.
    """
    if not chunks:
        return "No whitepaper context found."

    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[WP{i}] SOURCE: {c['source']}\n{c['chunk']}\n")
    return "\n".join(lines)
