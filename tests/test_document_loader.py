import json

import pymupdf

from src import document_loader


def make_pdf(path, *pages):
    with pymupdf.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        pdf.save(path)


def test_pdf_export_cleans_text_and_keeps_source_and_page_numbers(tmp_path, monkeypatch):
    name = "Attendance.PDF"
    make_pdf(tmp_path / name, "  First    policy.  ", "Second policy.")
    output = tmp_path / "processed" / "documents.json"
    monkeypatch.setattr(document_loader, "DOCUMENTS_PATH", output)
    load = document_loader.read_documents
    monkeypatch.setattr(document_loader, "read_documents", lambda: load(tmp_path))

    document_loader.main()

    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"source": name, "page": 1, "text": "First policy."},
        {"source": name, "page": 2, "text": "Second policy."},
    ]


def test_empty_folder(tmp_path):
    assert document_loader.read_documents(tmp_path) == ([], [])
    assert document_loader.document_records([]) == []


def test_blank_page_keeps_its_number_and_textless_pdf_is_skipped(tmp_path):
    mixed = tmp_path / "mixed.pdf"
    blank = tmp_path / "blank.pdf"
    make_pdf(mixed, "First page.", "", "Third page.")
    make_pdf(blank, "")

    documents, skipped = document_loader.read_documents(tmp_path)

    assert skipped == [(blank, "No extractable text.")]
    assert document_loader.document_records(documents) == [
        {"source": mixed.name, "page": 1, "text": "First page."},
        {"source": mixed.name, "page": 2, "text": ""},
        {"source": mixed.name, "page": 3, "text": "Third page."},
    ]
