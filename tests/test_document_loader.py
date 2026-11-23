from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from docx import Document
import pymupdf

from src.document_loader import extract_docx, read_documents


class DocumentLoaderTests(unittest.TestCase):
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

    def test_duplicate_text_is_reported_after_normalization(self):
        for name, text in [("a.docx", "Café rules."), ("b.docx", "Cafe\u0301  rules."),
                           ("c.docx", "Other rules.")]:
            document = Document()
            document.add_paragraph(text)
            document.save(self.raw / name)

        documents, skipped = read_documents(self.raw)
        self.assertEqual([path.name for path, _ in documents], ["a.docx", "c.docx"])
        self.assertEqual(skipped, [(self.raw / "b.docx", "Duplicate extracted text of a.docx.")])

    def test_empty_source_folder_returns_no_documents(self):
        self.assertEqual(read_documents(self.raw), ([], []))
