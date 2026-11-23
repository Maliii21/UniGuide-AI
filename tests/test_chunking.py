from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.chunking import chunk_document, chunk_parts, chunk_text
from src.document_loader import TextPart


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.raw = Path(temporary.name)

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
