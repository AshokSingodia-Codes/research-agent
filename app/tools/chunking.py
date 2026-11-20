"""Document and page chunking for vector store evidence indexing."""

from __future__ import annotations

import re
from typing import NamedTuple

MIN_CHUNK_CHARS = 80


class DocumentChunk(NamedTuple):
    chunk_id: str
    doc_id: str
    source_url: str
    domain: str
    text: str
    page: int | None = None


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Splits an over-long paragraph into overlapping windows, preferring sentence boundaries."""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_document_text(
    doc_id: str,
    source_url: str,
    domain: str,
    text: str,
    *,
    size: int = 800,
    overlap: int = 120,
    max_chunks: int = 25,
) -> list[DocumentChunk]:
    """Paragraph-aware chunking for extracted web or PDF text."""
    if not text:
        return []

    pieces: list[str] = []
    buf = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) > size:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_windows(para, size, overlap))
        elif len(buf) + len(para) + 1 <= size:
            buf = f"{buf}\n{para}".strip()
        else:
            pieces.append(buf)
            buf = para
    if buf:
        pieces.append(buf)

    valid_pieces = [p for p in pieces if len(p) >= MIN_CHUNK_CHARS][:max_chunks]
    return [
        DocumentChunk(
            chunk_id=f"{doc_id}#{i}",
            doc_id=doc_id,
            source_url=source_url,
            domain=domain,
            text=p,
        )
        for i, p in enumerate(valid_pieces, 1)
    ]

