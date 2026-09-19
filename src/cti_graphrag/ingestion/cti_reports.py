"""Ingest unstructured CTI report documents (Markdown with a metadata header).

Each report keeps metadata (source, doc id, publication date, related
entities) alongside its body text so retrieved chunks can always be traced
back to their original source -- the traceability requirement for the
document-based RAG pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_REPORTS_DIR = Path(__file__).resolve().parents[3] / "data" / "sample" / "cti_reports"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    publication_date: str
    text: str
    related_actors: list[str] = field(default_factory=list)
    related_malware: list[str] = field(default_factory=list)
    related_cves: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    path: str | None = None

    def all_related_entities(self) -> list[str]:
        return [*self.related_actors, *self.related_malware, *self.sectors]


def _parse_scalar(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip('"')


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    header_block, body = match.groups()
    metadata: dict = {}
    for line in header_block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = _parse_scalar(value)
    return metadata, body.strip()


def parse_report_file(path: Path) -> Document:
    raw_text = path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(raw_text)
    return Document(
        doc_id=metadata.get("doc_id", path.stem),
        title=metadata.get("title", path.stem),
        source=metadata.get("source", "unknown"),
        publication_date=metadata.get("publication_date", ""),
        text=body,
        related_actors=metadata.get("related_actors", []),
        related_malware=metadata.get("related_malware", []),
        related_cves=metadata.get("related_cves", []),
        sectors=metadata.get("sectors", []),
        path=str(path),
    )


def load_cti_reports(directory: Path | None = None) -> list[Document]:
    directory = directory or SAMPLE_REPORTS_DIR
    return [parse_report_file(p) for p in sorted(directory.glob("*.md"))]
