import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.generate_embeddings import EMBEDDING_DIMENSION, generate_embeddings, save_knowledge_base
from src.semantic_search import load_knowledge_base, semantic_search


def unit_vector(index):
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    vector[index] = 1
    return vector


class LocalEncoder:
    max_seq_length = 256

    def tokenizer(self, text, **kwargs):
        return {"input_ids": text.split()}

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return unit_vector(1)
        return np.stack([unit_vector(index) for index, _ in enumerate(texts)])


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.metadata = root / "chunks.json"
        self.vectors = root / "vectors.npy"
        self.chunks = [
            {"chunk_id": "a", "source": "fixture.pdf", "text": "First passage", "page_start": 1},
            {"chunk_id": "b", "source": "fixture.pdf", "text": "Second passage", "page_start": 2},
        ]
        self.model = LocalEncoder()

    def test_embedding_shape_dtype_and_round_trip(self):
        vectors = generate_embeddings(self.chunks, self.model)
        self.assertEqual(vectors.shape, (2, 384))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
        save_knowledge_base(self.chunks, vectors, self.metadata, self.vectors)
        chunks, loaded = load_knowledge_base(self.metadata, self.vectors)
        self.assertEqual(chunks, self.chunks)
        np.testing.assert_array_equal(loaded, vectors)

    def test_rejects_invalid_embedding_values_and_dimensions(self):
        invalid_vectors = (
            (np.zeros((2, 4)), "384"),
            (np.stack([unit_vector(0)]), "2 rows"),
            (np.full((2, 384), np.nan), "finite"),
            (np.zeros((2, 384)), "normalized"),
        )
        for vectors, message in invalid_vectors:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                save_knowledge_base(self.chunks, vectors, self.metadata, self.vectors)

    def test_rejects_empty_chunk_list_and_overlong_chunk(self):
        with self.assertRaisesRegex(ValueError, "No document chunks"):
            generate_embeddings([], self.model)
        self.chunks[0]["text"] = "word " * 257
        with self.assertRaisesRegex(ValueError, "input limit"):
            generate_embeddings(self.chunks, self.model)

    def test_detects_reordered_metadata_with_same_count(self):
        vectors = generate_embeddings(self.chunks, self.model)
        save_knowledge_base(self.chunks, vectors, self.metadata, self.vectors)
        self.metadata.write_text(json.dumps(self.chunks[::-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not match"):
            load_knowledge_base(self.metadata, self.vectors)

    def test_detects_reordered_vectors_with_same_shape(self):
        vectors = generate_embeddings(self.chunks, self.model)
        save_knowledge_base(self.chunks, vectors, self.metadata, self.vectors)
        np.save(self.vectors, vectors[::-1])
        with self.assertRaisesRegex(ValueError, "do not match"):
            load_knowledge_base(self.metadata, self.vectors)

    def test_missing_knowledge_base_files(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing knowledge-base file"):
            load_knowledge_base(self.metadata, self.vectors)
        vectors = generate_embeddings(self.chunks, self.model)
        save_knowledge_base(self.chunks, vectors, self.metadata, self.vectors)
        self.vectors.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "vectors.npy"):
            load_knowledge_base(self.metadata, self.vectors)

    def test_cosine_similarity_and_top_k(self):
        vectors = generate_embeddings(self.chunks, self.model)
        ranked = semantic_search("  attendance  ", self.model, self.chunks, vectors, top_k=20)
        self.assertEqual([result["chunk_id"] for result in ranked], ["b", "a"])
        self.assertEqual([result["score"] for result in ranked], [1.0, 0.0])
        self.assertEqual(ranked[0]["page_start"], 2)
        self.assertEqual(semantic_search("attendance", self.model, self.chunks, vectors, top_k=1),
                         ranked[:1])

    def test_empty_question_and_nonpositive_top_k_skip_encoding(self):
        self.assertEqual(semantic_search(" \t ", None, [], None), [])
        self.assertEqual(semantic_search("attendance", None, [], None, top_k=0), [])


if __name__ == "__main__":
    unittest.main()
