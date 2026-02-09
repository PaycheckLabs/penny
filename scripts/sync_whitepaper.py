#!/usr/bin/env python3
"""
Sync markdown/text docs from a source repo folder into this repo.

Used by GitHub Actions to keep Penny's local knowledge base updated from checks-gitbook.

Behavior:
- Copies only .md and .txt files
- Skips .gitbook/ and common non-doc paths
- Mirrors folder structure
- Deletes destination files that no longer exist in source (true sync)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Set


ALLOWED_EXTS: Set[str] = {".md", ".txt"}

# Folders to ignore anywhere in the path
IGNORE_DIR_NAMES: Set[str] = {
    ".git",
    ".github",
    ".gitbook",          # contains assets + gitbook internals
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}

# Also ignore specific path fragments (normalized with '/')
IGNORE_PATH_FRAGMENTS: Set[str] = {
    ".gitbook/assets",   # large binaries not needed for RAG
}


def _norm_rel(p: Path) -> str:
    return str(p.as_posix()).lstrip("./")


def _should_ignore(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    for frag in IGNORE_PATH_FRAGMENTS:
        if rel.startswith(frag) or f"/{frag}/" in f"/{rel}/":
            return True
    parts = [x for x in rel.split("/") if x]
    return any(part in IGNORE_DIR_NAMES for part in parts)


def _iter_source_files(src_dir: Path) -> Iterable[Path]:
    for p in src_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = _norm_rel(p.relative_to(src_dir))
        if _should_ignore(rel):
            continue
        if p.suffix.lower() in ALLOWED_EXTS:
            yield p


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sync_docs(src_dir: Path, dest_dir: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source directory not found: {src_dir}")

    _safe_mkdir(dest_dir)

    # Build mapping of expected destination files
    expected: Set[str] = set()

    for src_file in _iter_source_files(src_dir):
        rel = src_file.relative_to(src_dir)
        rel_str = _norm_rel(rel)
        expected.add(rel_str)

        dest_file = dest_dir / rel
        _safe_mkdir(dest_file.parent)
        shutil.copy2(src_file, dest_file)

    # Delete dest files that are no longer in source (true sync)
    for dest_file in dest_dir.rglob("*"):
        if dest_file.is_dir():
            continue
        rel = _norm_rel(dest_file.relative_to(dest_dir))
        if _should_ignore(rel):
            continue
        if dest_file.suffix.lower() not in ALLOWED_EXTS:
            continue
        if rel not in expected:
            dest_file.unlink()

    # Cleanup empty folders
    for d in sorted(dest_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                next(d.iterdir())
            except StopIteration:
                d.rmdir()


def main() -> None:
    src = Path(os.environ.get("WP_SRC_DIR", "")).resolve()
    dest = Path(os.environ.get("WP_DEST_DIR", "")).resolve()

    if not src or str(src) in ("/", ".") or not str(src):
        raise SystemExit("WP_SRC_DIR env var not set correctly.")
    if not dest or str(dest) in ("/", ".") or not str(dest):
        raise SystemExit("WP_DEST_DIR env var not set correctly.")

    print(f"Syncing whitepaper docs:\n- from: {src}\n- to:   {dest}")
    sync_docs(src, dest)
    print("Sync complete.")


if __name__ == "__main__":
    main()
