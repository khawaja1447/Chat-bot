"""PDF -> pages -> chunks, with document and page identity preserved throughout."""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragbot.config import MAX_PAGES_PER_DOC, RagConfig


class IngestError(ValueError):
    """Raised for a PDF we cannot usefully index, with a message fit for the UI."""


@dataclass
class Chunk:
    text: str
    doc: str
    page: int
    chunk_id: str

    @property
    def metadata(self) -> dict:
        return {"doc": self.doc, "page": self.page, "chunk_id": self.chunk_id}

    @property
    def citation(self) -> str:
        return f"{self.doc} p.{self.page}"


@dataclass
class Document:
    name: str
    pages: list = field(default_factory=list)   # [(page_number, text)]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_document(file_obj, name: str) -> Document:
    """
    Read a PDF into per-page text.

    Page identity has to survive ingestion — if pages are concatenated first, no
    amount of downstream metadata can put a citation back together.
    """
    raw = file_obj.read()
    if not raw:
        raise IngestError(f"{name} is empty.")
    try:
        doc = pymupdf.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise IngestError(f"{name} could not be opened as a PDF ({exc}).") from exc

    try:
        if len(doc) > MAX_PAGES_PER_DOC:
            raise IngestError(
                f"{name} has {len(doc)} pages; the limit is {MAX_PAGES_PER_DOC}. "
                "Embeddings run locally on CPU, so larger documents take a very "
                "long time. Split it and upload a section."
            )
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()

    if not any(text.strip() for _, text in pages):
        raise IngestError(
            f"No text could be extracted from {name}. Scanned or image-only "
            "documents need OCR before they can be indexed."
        )
    return Document(name=name, pages=pages)


def _splitter(config: RagConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def chunk_document(document: Document, config: RagConfig) -> list:
    """
    Split each page independently.

    Chunking the whole document as one string lets a window span a page break and
    glue the end of one section to the start of an unrelated next one. Per-page
    splitting costs a little redundancy at boundaries and buys exact citations.
    """
    splitter = _splitter(config)
    chunks = []
    for page_no, page_text in document.pages:
        if not page_text.strip():
            continue
        for i, text in enumerate(splitter.split_text(page_text)):
            chunks.append(
                Chunk(
                    text=text,
                    doc=document.name,
                    page=page_no,
                    chunk_id=f"{document.name}::p{page_no}::c{i}",
                )
            )
    return chunks
