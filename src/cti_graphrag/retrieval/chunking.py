"""Split CTI report documents into retrievable chunks, preserving traceable metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cti_graphrag.ingestion.cti_reports import Document

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    # Skip markdown heading lines and blank lines; keep prose sentences only.
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    joined = " ".join(lines)
    return [s.strip() for s in _SENTENCE_RE.split(joined) if s.strip()]


def chunk_document(doc: Document, max_sentences: int = 3, overlap: int = 1) -> list[Chunk]:
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    step = max(1, max_sentences - overlap)
    idx = 0
    chunk_num = 0
    while idx < len(sentences):
        window = sentences[idx : idx + max_sentences]
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::chunk{chunk_num}",
                doc_id=doc.doc_id,
                text=text,
                metadata={
                    "source": doc.source,
                    "title": doc.title,
                    "publication_date": doc.publication_date,
                    "related_actors": doc.related_actors,
                    "related_malware": doc.related_malware,
                    "related_cves": doc.related_cves,
                    "sectors": doc.sectors,
                    "path": doc.path,
                },
            )
        )
        chunk_num += 1
        idx += step
    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, **kwargs))
    return chunks
