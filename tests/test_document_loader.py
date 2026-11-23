# python -m pytest tests/test_document_loader.py -v
import csv
import json

import fitz
import pytest

from src.document_loader import (
    load_sources,
    clean_text,
    load_documents,
    save_documents,
)


@pytest.fixture
def sample_data(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    pdf_path = raw_dir / "sample.pdf"

    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "AIU Attendance Policy")

        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "Students must attend lectures."
        )

        pdf.save(pdf_path)

    sources_file = tmp_path / "sources.csv"

    metadata = {
        "document_id": "AIU-TEST",
        "document_title": "Attendance Policy",
        "original_filename": "sample.pdf",
        "official_source_url": "https://aiu.edu.eg",
        "document_category": "Academic Policies",
        "language": "English",
        "document_date_as_printed": "2026",
        "document_date_type": "Creation",
        "page_count": "2",
        "review_status": "accepted",
        "review_notes": "",
        "version_notes": ""
    }

    with open(
        sources_file, "w",
        encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=metadata.keys()
        )
        writer.writeheader()
        writer.writerow(metadata)

    return raw_dir, sources_file


# Test 1: Source inventory loading
def test_load_sources(sample_data):
    _, sources_file = sample_data

    sources = load_sources(sources_file)

    assert len(sources) == 1
    assert "sample.pdf" in sources
    assert sources["sample.pdf"]["document_id"] == "AIU-TEST"


# Test 2: Text cleaning
def test_clean_text():
    text = "  AIU    Policy\n\n\n\nAttendance  "

    assert clean_text(text) == "AIU Policy\n\nAttendance"


# Test 3: PDF text extraction
def test_load_documents(sample_data):
    raw_dir, sources_file = sample_data

    documents = load_documents(raw_dir, sources_file)

    assert len(documents) == 2
    assert "AIU Attendance Policy" in documents[0]["text"]
    assert "Students must attend lectures." in documents[1]["text"]


# Test 4: Metadata preservation
def test_metadata_preservation(sample_data):
    raw_dir, sources_file = sample_data

    documents = load_documents(raw_dir, sources_file)

    assert documents[0]["document_id"] == "AIU-TEST"
    assert documents[0]["document_title"] == "Attendance Policy"
    assert documents[0]["source"] == "sample.pdf"
    assert documents[0]["source_url"] == "https://aiu.edu.eg"


# Test 5: Page numbering and IDs
def test_page_numbering(sample_data):
    raw_dir, sources_file = sample_data

    documents = load_documents(raw_dir, sources_file)

    assert documents[0]["page"] == 1
    assert documents[1]["page"] == 2

    assert documents[0]["page_id"] == "AIU-TEST-P001"
    assert documents[1]["page_id"] == "AIU-TEST-P002"


# Test 6: JSON output
def test_save_documents(sample_data, tmp_path):
    raw_dir, sources_file = sample_data

    documents = load_documents(raw_dir, sources_file)

    output_file = tmp_path / "processed" / "documents.json"

    save_documents(documents, output_file)

    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as file:
        saved_documents = json.load(file)

    assert saved_documents == documents


# Test 7: Empty PDF folder
def test_empty_folder(sample_data, tmp_path):
    _, sources_file = sample_data

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_documents(empty_dir, sources_file)


# Test 8: Unregistered PDF
def test_unregistered_pdf(sample_data):
    raw_dir, sources_file = sample_data

    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(raw_dir / "unregistered.pdf")

    with pytest.raises(ValueError):
        load_documents(raw_dir, sources_file)


# Test 9: Verified visual correction
def test_verified_correction(sample_data, tmp_path):
    raw_dir, sources_file = sample_data

    corrections_file = tmp_path / "corrections.json"

    corrections = [
        {
            "document_id": "AIU-TEST",
            "page": 1,
            "extraction_method": "visual_form_summary",
            "review_status": "verified",
            "text": "Reviewed image content."
        }
    ]

    corrections_file.write_text(
        json.dumps(corrections),
        encoding="utf-8"
    )

    documents = load_documents(
        raw_dir,
        sources_file,
        corrections_file
    )

    assert len(documents) == 2
    assert "AIU Attendance Policy" in documents[0]["text"]
    assert "Reviewed image content." in documents[0]["text"]

    assert (
        documents[0]["extraction_method"]
        == "native_text_plus_visual_summary"
    )

    assert documents[0]["image_review_status"] == "verified"


# Test 10: Unverified correction
def test_unverified_correction(sample_data, tmp_path):
    raw_dir, sources_file = sample_data

    corrections_file = tmp_path / "corrections.json"

    corrections = [
        {
            "document_id": "AIU-TEST",
            "page": 1,
            "extraction_method": "visual_form_summary",
            "review_status": "pending_human_review",
            "text": "Unverified information."
        }
    ]

    corrections_file.write_text(
        json.dumps(corrections),
        encoding="utf-8"
    )

    documents = load_documents(
        raw_dir,
        sources_file,
        corrections_file
    )

    assert "Unverified information." not in documents[0]["text"]

    assert documents[0]["extraction_method"] == "native_text"


# Test 11: Native extraction method
def test_native_extraction_method(sample_data):
    raw_dir, sources_file = sample_data

    documents = load_documents(raw_dir, sources_file)

    assert all(
        doc["extraction_method"] == "native_text"
        for doc in documents
    )