import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from docx import Document
import pymupdf

from src import rebuild_chunks
from src.chunking import chunk_document, chunk_parts, chunk_text
from src.clean_text import clean_text
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
        chunks = [chunk for path, parts in documents for chunk in chunk_document(path, parts)]
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

    def test_chunk_prefers_complete_paragraphs_and_retains_exact_overlap(self):
        chunks = chunk_text("one two three four\n\nfive six seven eight\n\nnine ten",
                            chunk_size=5, overlap=1)
        self.assertEqual(chunks[0], "one two three four")
        self.assertEqual(chunks[-1].split()[-1], "ten")
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left.split()[-1:], right.split()[:1])
            self.assertLessEqual(len(left.split()), 5)

    def test_chunk_coverage_and_page_provenance_across_boundaries(self):
        parts = [
            TextPart("I. FIRST RULE\n\none repeated words", page=1),
            TextPart("", page=2),
            TextPart("two repeated words\n\nthree four five", page=3),
        ]
        path = self.raw / "policy.pdf"
        fits = lambda passage: len(passage.split()) <= 6
        options = {"chunk_size": 7, "overlap": 2, "fits": fits}
        chunks = chunk_document(path, parts, **options)
        self.assertEqual(chunk_document(path, parts, **options), chunks)
        self.assertEqual([(chunk["page_start"], chunk["page_end"]) for chunk in chunks],
                         [(1, 1), (1, 3), (3, 3)])
        expected_words = " ".join(part.text for part in parts).split()
        recovered_words = []
        for ordinal, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk["chunk_id"], f"{path.as_posix()}#{ordinal}")
            self.assertEqual(chunk["section_heading"], "I. FIRST RULE")
            self.assertTrue(fits(chunk["text"]))
            words = chunk["text"].split()
            if recovered_words:
                self.assertEqual(recovered_words[-2:], words[:2])
                words = words[2:]
            recovered_words.extend(words)
        self.assertEqual(recovered_words, expected_words)

    def test_empty_short_and_invalid_chunk_inputs(self):
        self.assertEqual(chunk_text(" \n\t "), [])
        self.assertEqual(chunk_text("Short passage."), ["Short passage."])
        self.assertEqual(chunk_text("one two three four", chunk_size=2, overlap=0),
                         ["one two", "three four"])
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=2, overlap=2)

    def test_tight_token_budget_keeps_chunks_moving(self):
        parts = [TextPart("one two three four five six")]
        chunks = chunk_parts(parts, chunk_size=5, overlap=4, fits=lambda text: len(text.split()) <= 2)
        self.assertEqual([chunk["text"] for chunk in chunks],
                         ["one two", "two three", "three four", "four five", "five six"])

    def test_token_budget_drops_overlap_when_it_blocks_new_content(self):
        parts = [TextPart("a bb cc d")]
        chunks = chunk_parts(parts, chunk_size=4, overlap=1,
                             fits=lambda text: len(text.replace(" ", "")) <= 3)
        self.assertEqual([chunk["text"] for chunk in chunks], ["a bb", "cc d"])
        with self.assertRaisesRegex(ValueError, "single word exceeds"):
            chunk_parts([TextPart("largeword")], fits=lambda text: len(text) <= 3)


if __name__ == "__main__":
    unittest.main()
