import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from docx import Document
import pymupdf

from src import rebuild_chunks
from src.chunking import chunk_documents
from src.clean_text import clean_text
from src.document_loader import document_records, extract_docx, extract_pdf, read_documents


class DocumentProcessingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.raw = Path(temporary.name)

    def make_pdf(self, filename, *pages):
        path = self.raw / filename
        with pymupdf.open() as document:
            for text in pages:
                page = document.new_page()
                if text:
                    page.insert_text((72, 72), text)
            document.save(path)
        return path

    def test_pdf_extraction_and_starting_page_in_chunks(self):
        path = self.make_pdf("Fixture.PDF", "I.\nATTENDANCE RULE\n\nRemain enrolled.", "", "Third page.")
        parts = extract_pdf(path)
        self.assertEqual([part.page for part in parts], [1, 2, 3])
        self.assertEqual(extract_pdf(path), parts)
        self.assertIn("Remain enrolled.", parts[0].text)
        self.assertEqual(parts[1].text, "")
        chunk, = chunk_documents(document_records([(path, parts)]))
        self.assertEqual(chunk["source"], path.name)
        self.assertEqual(chunk["page"], 1)
        self.assertIn("Third page.", chunk["text"])

    def test_docx_body_order_tables_and_heading_metadata(self):
        path = self.raw / "Fixture.docx"
        document = Document()
        document.add_heading("Local fixture", level=1)
        document.add_paragraph("First paragraph.")
        document.add_table(rows=1, cols=1).cell(0, 0).text = "Table cell."
        document.add_paragraph("Last paragraph.")
        document.save(path)

        parts = extract_docx(path)
        chunk, = chunk_documents(document_records([(path, parts)]))
        self.assertEqual(chunk["text"], "Local fixture\n\nFirst paragraph.\n\nTable cell.\n\nLast paragraph.")
        self.assertEqual(parts[0].headings, ("Local fixture",))
        self.assertIsNone(chunk["page"])

    def test_pdf_rejects_recognized_non_pdf_formats(self):
        fixtures = {
            "page.pdf": "<html><body><p>HTML text.</p></body></html>",
            "drawing.pdf": (
                '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">'
                '<text x="10" y="30">SVG text.</text></svg>'
            ),
        }
        valid = self.make_pdf("valid.pdf", "Valid policy text.")
        for name, markup in fixtures.items():
            path = self.raw / name
            path.write_text(markup, encoding="utf-8")
            with pymupdf.open(path) as document:
                self.assertFalse(document.is_pdf)
                self.assertIn("text.", document[0].get_text().lower())

        documents, skipped = read_documents(self.raw)
        self.assertEqual([path for path, _ in documents], [valid])
        self.assertEqual({path.name: reason for path, reason in skipped},
                         dict.fromkeys(fixtures, "File content is not a PDF."))

    def test_pdf_rejects_a_file_that_requires_structural_repair(self):
        path = self.make_pdf("truncated.pdf", "Recoverable policy text.")
        contents = path.read_bytes()
        path.write_bytes(contents[:contents.rindex(b"\nxref")])

        with pymupdf.open(path) as document:
            self.assertTrue(document.is_pdf)
            self.assertTrue(document.is_repaired)
            self.assertIn("Recoverable policy text.", document[0].get_text())

        documents, skipped = read_documents(self.raw)
        self.assertEqual(documents, [])
        self.assertEqual(skipped, [(path, "PDF required structural repair. Obtain an intact copy.")])

    def test_docx_extracts_each_merged_cell_once_and_keeps_distinct_cells(self):
        path = self.raw / "merged.docx"
        document = Document()
        table = document.add_table(rows=2, cols=3)
        table.cell(0, 0).merge(table.cell(1, 0)).text = "Vertical merge."
        table.cell(0, 1).merge(table.cell(0, 2)).text = "Horizontal merge."
        table.cell(1, 1).text = "Same text."
        table.cell(1, 2).text = "Same text."
        document.add_paragraph("After the table.")
        document.save(path)

        part, = extract_docx(path)
        self.assertEqual(part.text.split("\n\n"), [
            "Vertical merge.", "Horizontal merge.", "Same text.", "Same text.", "After the table.",
        ])

    def test_discovery_skips_empty_and_corrupt_files_and_keeps_documents_separate(self):
        pdf = self.make_pdf("same.pdf", "PDF passage.")
        docx = self.raw / "same.docx"
        document = Document()
        document.add_paragraph("DOCX passage.")
        document.save(docx)
        self.make_pdf("empty.pdf", "")
        Document().save(self.raw / "empty.docx")
        (self.raw / "broken.pdf").write_bytes(b"invalid PDF")
        (self.raw / "broken.docx").write_bytes(b"invalid DOCX")
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), "Password-protected fixture.")
            document.save(self.raw / "encrypted.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256,
                          owner_pw="fixture-owner", user_pw="fixture-reader")
        with pymupdf.open() as document:
            page = document.new_page()
            pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16), False)
            pixels.clear_with(0)
            page.insert_image(pymupdf.Rect(72, 72, 144, 144), pixmap=pixels)
            document.save(self.raw / "image-only.pdf")
        (self.raw / "generated_cleaned.txt").write_text("Ignore this.", encoding="utf-8")
        documents, skipped = read_documents(self.raw)
        self.assertEqual({path for path, _ in documents}, {pdf, docx})
        errors = {path.name: reason for path, reason in skipped}
        self.assertEqual(set(errors), {"empty.pdf", "empty.docx", "broken.pdf", "broken.docx",
                                      "encrypted.pdf", "image-only.pdf"})
        self.assertEqual(errors["encrypted.pdf"], "PDF requires a password.")
        self.assertEqual(errors["empty.pdf"], "No extractable text.")
        self.assertEqual(errors["image-only.pdf"], "No extractable text.")
        chunks = chunk_documents(document_records(documents))
        self.assertEqual({chunk["text"] for chunk in chunks}, {"PDF passage.", "DOCX passage."})
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 2)

    def test_clean_preserves_structure_and_unicode(self):
        text = "  Heading\r\n\r\n\r\n  •\t75%  \r\n 1-  March 2022\r\n  Cre\u0300me — student@aiu.edu.eg "
        self.assertEqual(clean_text(clean_text(text)), clean_text(text))
        self.assertEqual(clean_text(" \t\n "), "")
        self.assertEqual(clean_text(text), "Heading\n\n• 75%\n1- March 2022\nCrème — student@aiu.edu.eg")

    def test_duplicate_text_is_reported_after_normalization(self):
        for name, text in [("a.docx", "Café rules."), ("b.docx", "Cafe\u0301  rules."),
                           ("c.docx", "Other rules.")]:
            document = Document()
            document.add_paragraph(text)
            document.save(self.raw / name)

        documents, skipped = read_documents(self.raw)
        self.assertEqual([path.name for path, _ in documents], ["a.docx", "c.docx"])
        self.assertEqual(skipped, [(self.raw / "b.docx", "Duplicate extracted text of a.docx.")])

    def test_rebuild_writes_only_the_required_json_outputs(self):
        path = self.make_pdf("rules.pdf", "One    policy.")
        outputs = self.raw / "processed"
        documents_path = outputs / "documents.json"
        chunks_path = outputs / "chunks.json"
        with (
            patch.object(rebuild_chunks, "read_documents", side_effect=lambda: read_documents(self.raw)),
            patch.object(rebuild_chunks, "DOCUMENTS_PATH", documents_path),
            patch.object(rebuild_chunks, "CHUNKS_PATH", chunks_path),
            patch("builtins.print"),
        ):
            self.assertEqual(rebuild_chunks.main(), 0)

        self.assertEqual({output.name for output in outputs.iterdir()}, {"documents.json", "chunks.json"})
        expected = {"source": path.name, "page": 1, "text": "One policy."}
        self.assertEqual(json.loads(documents_path.read_bytes()), [expected])
        self.assertEqual(json.loads(chunks_path.read_bytes()),
                         [{"chunk_id": "rules.pdf#p1_c1", **expected}])


if __name__ == "__main__":
    unittest.main()
