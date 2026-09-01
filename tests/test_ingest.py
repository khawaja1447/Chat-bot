"""PDF ingestion: page identity, rejection cases, per-page chunking."""

from __future__ import annotations

import pytest

from ragbot.config import MAX_PAGES_PER_DOC, RagConfig
from ragbot.ingest import Document, IngestError, chunk_document, extract_document

CONFIG = RagConfig(chunk_size=200, chunk_overlap=50)


def test_extract_preserves_page_numbers(pdf_factory):
    doc = extract_document(pdf_factory(["alpha text", "beta text"]), "two.pdf")
    assert [n for n, _ in doc.pages] == [1, 2]
    assert "alpha" in doc.pages[0][1]
    assert "beta" in doc.pages[1][1]
    assert doc.page_count == 2


def test_image_only_pdf_is_rejected(pdf_factory):
    with pytest.raises(IngestError, match="No text could be extracted"):
        extract_document(pdf_factory([None, None]), "scan.pdf")


def test_oversized_pdf_is_rejected(pdf_factory):
    with pytest.raises(IngestError, match="limit is"):
        extract_document(pdf_factory([None] * (MAX_PAGES_PER_DOC + 1)), "huge.pdf")


def test_empty_upload_is_rejected():
    import io

    with pytest.raises(IngestError, match="empty"):
        extract_document(io.BytesIO(b""), "nothing.pdf")


def test_non_pdf_is_rejected():
    import io

    with pytest.raises(IngestError, match="could not be opened"):
        extract_document(io.BytesIO(b"this is plainly not a pdf"), "fake.pdf")


def test_chunks_never_straddle_a_page_boundary(pdf_factory):
    doc = extract_document(pdf_factory(["alpha " * 80, "beta " * 80]), "two.pdf")
    chunks = chunk_document(doc, CONFIG)

    assert len(chunks) > 2, "expected several chunks per page at this chunk size"
    for chunk in chunks:
        # A chunk that spanned the page break would contain both marker words.
        assert not ("alpha" in chunk.text and "beta" in chunk.text)
    assert {c.page for c in chunks} == {1, 2}


def test_chunk_ids_are_unique_and_traceable(pdf_factory):
    doc = extract_document(pdf_factory(["alpha " * 80, "beta " * 80]), "two.pdf")
    chunks = chunk_document(doc, CONFIG)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for chunk in chunks:
        assert chunk.chunk_id.startswith(f"two.pdf::p{chunk.page}::")
        assert chunk.citation == f"two.pdf p.{chunk.page}"


def test_blank_pages_are_skipped_but_do_not_shift_numbering(pdf_factory):
    doc = extract_document(pdf_factory(["first", None, "third"]), "gappy.pdf")
    chunks = chunk_document(doc, CONFIG)
    # Page 2 contributes nothing, but page 3 must still be called page 3.
    assert {c.page for c in chunks} == {1, 3}


def test_chunk_overlap_preserves_boundary_context():
    # Built in code, not via a PDF: insert_textbox rewraps text on the way in,
    # which would break an assertion about an exact character run.
    config = RagConfig(chunk_size=400, chunk_overlap=100)
    doc = Document(name="solid.pdf", pages=[(1, "A" * 1200)])
    chunks = chunk_document(doc, config)
    assert len(chunks) > 1
    assert chunks[0].text[-100:] == chunks[1].text[:100]
