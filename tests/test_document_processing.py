import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from docx import Document
import pymupdf

from scripts import export_documents
from src import rebuild_chunks
from src.chunking import chunk_document
from src.document_loader import TextPart, extract_docx, extract_pdf, read_documents


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

    def test_pdf_extraction_and_page_ranges(self):
        path = self.make_pdf("Fixture.PDF", "I.\nATTENDANCE RULE\n\nRemain enrolled.", "", "Third page.")
        parts = extract_pdf(path)
        self.assertEqual([part.page for part in parts], [1, 2, 3])
        self.assertEqual(extract_pdf(path), parts)
        self.assertIn("Remain enrolled.", parts[0].text)
        self.assertEqual(parts[1].text, "")
        chunk, = chunk_document(path, parts)
        self.assertEqual(chunk["source"], path.name)
        self.assertEqual(chunk["source_path"], path.as_posix())
        self.assertEqual((chunk["page_start"], chunk["page_end"]), (1, 3))
        self.assertEqual(chunk["section_heading"], "I. ATTENDANCE RULE")

    def test_docx_body_order_tables_and_heading_metadata(self):
        path = self.raw / "Fixture.docx"
        document = Document()
        document.add_heading("Local fixture", level=1)
        document.add_paragraph("First paragraph.")
        document.add_table(rows=1, cols=1).cell(0, 0).text = "Table cell."
        document.add_paragraph("Last paragraph.")
        document.save(path)

        parts = extract_docx(path)
        chunk, = chunk_document(path, parts)
        self.assertEqual(chunk["text"], "Local fixture\n\nFirst paragraph.\n\nTable cell.\n\nLast paragraph.")
        self.assertEqual(chunk["section_heading"], "Local fixture")
        self.assertIsNone(chunk["page_start"])
        self.assertIsNone(chunk["page_end"])

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
        chunks = [chunk for path, parts in documents for chunk in chunk_document(path, parts)]
        self.assertEqual({chunk["text"] for chunk in chunks}, {"PDF passage.", "DOCX passage."})
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 2)

    def test_rebuild_uses_cli_chunk_size_and_overlap(self):
        path = self.raw / "rules.pdf"
        parts = [TextPart("one two three four five six", page=1)]
        model = Mock(max_seq_length=256)
        model.tokenizer.return_value = {"input_ids": [1]}
        with (
            patch("sys.argv", ["rebuild", "--chunk-size", "3", "--overlap", "1"]),
            patch.object(rebuild_chunks, "read_documents", return_value=([(path, parts)], [])),
            patch.object(rebuild_chunks, "load_model", return_value=model),
            patch.object(rebuild_chunks, "atomic_write") as save,
            patch("builtins.print"),
        ):
            self.assertEqual(rebuild_chunks.main(), 0)

        self.assertEqual([chunk["text"] for chunk in json.loads(save.call_args.args[1])],
                         ["one two three", "three four five", "five six"])


    def test_document_export_preserves_loader_text_and_canonical_chunks(self):
        pdf = self.make_pdf("z-policy.pdf", "First page.", "", "Third page.")
        docx = self.raw / "a-note.docx"
        document = Document()
        document.add_paragraph("Caf\u00e9 policy.")
        document.save(docx)
        output = self.raw / "data" / "processed" / "documents.json"
        canonical = output.with_name("document_chunks.json")
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"canonical chunk output")

        with (
            patch.object(export_documents, "PROJECT_ROOT", self.raw),
            patch.object(export_documents, "DOCUMENTS_PATH", output),
            patch.object(export_documents, "read_documents", side_effect=lambda: read_documents(self.raw)),
            patch("builtins.print"),
        ):
            export_documents.main()
            first_export = output.read_bytes()
            export_documents.main()

        self.assertEqual(output.read_bytes(), first_export)
        self.assertIn("Caf\u00e9".encode("utf-8"), first_export)
        self.assertIn(b'\n    "source_path": "a-note.docx"', first_export)
        self.assertEqual(json.loads(first_export), [
            {"source": docx.name, "source_path": docx.name, "page": None, "text": "Caf\u00e9 policy."},
            *[
                {"source": pdf.name, "source_path": pdf.name, "page": part.page, "text": part.text}
                for part in extract_pdf(pdf)
            ],
        ])
        self.assertEqual(canonical.read_bytes(), b"canonical chunk output")
        self.assertFalse(output.with_name("chunks.json").exists())

    def test_document_export_sorts_by_relative_path_and_numeric_page(self):
        output = self.raw / "data" / "processed" / "documents.json"
        paths = [self.raw / "data" / "raw" / collection / "policy.pdf" for collection in ("z", "a")]
        documents = [
            (paths[0], [TextPart("Last document.", page=1)]),
            (paths[1], [TextPart("Eleventh page.", page=11), TextPart("Second page.", page=2)]),
        ]
        with (
            patch.object(export_documents, "PROJECT_ROOT", self.raw),
            patch.object(export_documents, "DOCUMENTS_PATH", output),
            patch.object(export_documents, "read_documents", return_value=(documents, [])),
            patch("builtins.print"),
        ):
            export_documents.main()

        records = json.loads(output.read_bytes())
        self.assertEqual([(record["source_path"], record["page"]) for record in records], [
            ("data/raw/a/policy.pdf", 2),
            ("data/raw/a/policy.pdf", 11),
            ("data/raw/z/policy.pdf", 1),
        ])


if __name__ == "__main__":
    unittest.main()
