import unittest
from unittest.mock import Mock

import numpy as np

from src.embeddings import embed_query, embed_texts, normalize_rows
from src.generate_embeddings import EMBEDDING_DIMENSION
from src.retrieval import cosine_similarity, format_context, retrieve, top_k_indices
from src.semantic_search import semantic_search


def passage(source, number, text, *, section="Scope"):
    path = f"data/raw/{source}"
    return {
        "source": source,
        "source_path": path,
        "chunk_id": f"{path}#{number}",
        "page_start": number,
        "page_end": number + 1,
        "section_heading": section,
        "section_headings": [section],
        "text": text,
    }


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        query = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        query[0] = 1
        self.model.encode.return_value = query

    def search(self, question, chunks, scores, top_k, **options):
        vectors = np.zeros((len(chunks), EMBEDDING_DIMENSION), dtype=np.float32)
        for row, score in enumerate(scores):
            vectors[row, 0] = score
            vectors[row, row + 1] = np.sqrt(1 - score ** 2)
        return semantic_search(question, self.model, chunks, vectors, top_k, **options)

    def test_matching_document_can_outrank_a_higher_scoring_cross_reference(self):
        cross_reference = passage("Campus-Handbook.pdf", 1, "See the library loan policy for details.")
        policy = passage("Library-Loan-Policy.pdf", 1, "Loans may be renewed at the library desk.")
        question = "How do I renew library loans?"
        chunks = [cross_reference, policy]
        self.assertEqual(self.search(question, chunks, [0.81, 0.78], 1, diversify=False)[0]["source"],
                         cross_reference["source"])
        result, = self.search(question, chunks, [0.81, 0.78], 1)
        self.assertEqual(result["source"], policy["source"])
        self.assertAlmostEqual(result["score"], 0.78)

    def test_broader_pool_returns_distinct_sections_of_the_same_policy(self):
        title = "Equipment-Checkout.pdf"
        repeated = "The first desk opens on Monday at noon for all borrowers."
        chunks = [passage(title, index, repeated) for index in range(1, 4)]
        rules = passage(title, 4, "Bring a campus card and complete the checkout form.", section="Procedure")
        chunks.append(rules)
        results = self.search("How does equipment checkout work?", chunks, [0.90, 0.89, 0.88, 0.83], 2)
        self.assertEqual([item["chunk_id"] for item in results],
                         [chunks[0]["chunk_id"], rules["chunk_id"]])
        self.assertEqual({key: results[1][key] for key in rules}, rules)

    def test_adds_verified_continuation_with_its_own_metadata_and_full_text(self):
        overlap = "staff will confirm in the system whether the equipment"
        anchor = passage("Checkout.pdf", 1, "At the front desk, " + overlap)
        neighbor = passage("Checkout.pdf", 2, overlap + " can be borrowed. Print a receipt from the self-service terminal before you leave the checkout desk.", section="Pickup")
        result, = self.search("What is the checkout process?", [anchor, neighbor], [0.85, 0.75], 1)
        context, = result["context"]
        self.assertEqual({key: context[key] for key in neighbor}, neighbor)
        self.assertAlmostEqual(context["score"], 0.75)
        self.assertNotIn("context", anchor)
        self.model.encode.assert_called_once()

    def test_skips_unverified_weak_or_duplicate_neighbors(self):
        shared = "one two three four five six seven eight nine ten eleven twelve"
        new_text = " followed by many more words explaining an important part of the policy in detail."
        anchor = passage("Checkout.pdf", 1, shared)
        different_source = passage("Another.pdf", 2, shared + new_text)
        different_source["chunk_id"] = "data/raw/Checkout.pdf#2"
        neighbor = passage("Checkout.pdf", 2, shared + new_text)
        no_overlap = passage("Checkout.pdf", 2, "Different first words" + new_text)
        duplicate = passage("Checkout.pdf", 2, shared)
        cases = [(different_source, 0.79), (neighbor, 0.60), (no_overlap, 0.79), (duplicate, 0.79)]
        for candidate, score in cases:
            with self.subTest(candidate=candidate["text"], source=candidate["source"], score=score):
                result, = self.search("checkout", [anchor, candidate], [0.85, score], 1)
                self.assertNotIn("context", result)
        results = self.search("checkout", [anchor, neighbor], [0.85, 0.79], 2)
        self.assertTrue(all("context" not in result for result in results))

    def test_recovers_suppressed_adjacent_top_hit_with_new_evidence(self):
        overlap = ("Visitors need to show their campus badge at reception before picking up "
                   "any equipment from the desk.")
        anchor = passage("Checkout.pdf", 1, overlap)
        neighbor = passage("Checkout.pdf", 2, overlap + " For exceptions to this rule the supervisor must sign a separate receipt at the desk.")
        other = passage("Facilities.pdf", 1, "Equipment can also be picked up at the annex.")
        results = self.search("Where can I get equipment?", [anchor, neighbor, other], [0.90, 0.87, 0.83], 2)
        self.assertEqual([item["chunk_id"] for item in results], [anchor["chunk_id"], other["chunk_id"]])
        self.assertEqual(results[0]["context"][0]["chunk_id"], neighbor["chunk_id"])


# ---- Tests for src/embeddings.py and src/retrieval.py ----


def vector(*values):
    result = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    result[:len(values)] = values
    return result


def make_chunks(count):
    return [{"chunk_id": f"data/raw/Doc{i}.pdf#1", "source": f"Doc{i}.pdf", "source_path": f"data/raw/Doc{i}.pdf",
             "text": f"Passage number {i} about a different topic.", "page_start": i + 1, "page_end": i + 1}
            for i in range(count)]


def make_index(scores):
    matrix = np.zeros((len(scores), EMBEDDING_DIMENSION), dtype=np.float32)
    for row, score in enumerate(scores):
        matrix[row, 0] = score
        matrix[row, row + 1] = np.sqrt(1 - score ** 2)
    return matrix


class CosineTests(unittest.TestCase):
    def test_matches_manual_formula_on_unnormalized_vectors(self):
        query = np.array([3.0, 4.0])
        matrix = np.array([[3.0, 4.0], [4.0, -3.0], [-6.0, -8.0], [0.0, 0.0]])
        np.testing.assert_allclose(cosine_similarity(query, matrix), [1.0, 0.0, -1.0, 0.0], atol=1e-12)

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            cosine_similarity(np.ones(3), np.ones((2, 4)))

    def test_top_k_is_ordered_stable_and_clamped(self):
        scores = np.array([0.2, 0.9, 0.9, 0.1])
        self.assertEqual(top_k_indices(scores, 3).tolist(), [1, 2, 0])
        self.assertEqual(top_k_indices(scores, 10).tolist(), [1, 2, 0, 3])
        self.assertEqual(top_k_indices(scores, 0).tolist(), [])


class EmbeddingHelperTests(unittest.TestCase):
    def test_normalize_rows_and_zero_vector(self):
        np.testing.assert_allclose(np.linalg.norm(normalize_rows(np.array([[3, 4], [0, 2]])), axis=1), 1)
        with self.assertRaisesRegex(ValueError, "zero vector"):
            normalize_rows(np.zeros((1, 4)))

    def test_embed_query_normalizes_and_validates(self):
        model = Mock()
        model.encode.return_value = vector(5)  # not unit length
        result = embed_query("  attendance  ", model)
        np.testing.assert_allclose(np.linalg.norm(result), 1)
        self.assertEqual(model.encode.call_args.args[0], "attendance")
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                embed_query(bad, model)
        model.encode.return_value = np.ones(7, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "384"):
            embed_query("x", model)

    def test_embed_texts_shape_and_input_checks(self):
        model = Mock()
        model.encode.return_value = np.stack([vector(1), vector(0, 1)])
        self.assertEqual(embed_texts(["a", "b"], model).shape, (2, EMBEDDING_DIMENSION))
        for bad in ("single string", [], ["ok", " "]):
            with self.assertRaises(ValueError):
                embed_texts(bad, model)


class RetrieveTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.encode.return_value = vector(1)
        self.chunks = make_chunks(4)
        self.index = make_index([0.5, 0.9, 0.7, 0.1])

    def test_plain_ranking_scores_and_rank_field(self):
        results = retrieve("attendance", self.model, self.chunks, self.index, top_k=3, diversify=False)
        self.assertEqual([r["chunk_id"] for r in results],
                         [self.chunks[1]["chunk_id"], self.chunks[2]["chunk_id"], self.chunks[0]["chunk_id"]])
        np.testing.assert_allclose([r["score"] for r in results], [0.9, 0.7, 0.5], atol=1e-6)
        self.assertEqual([r["rank"] for r in results], [1, 2, 3])
        self.assertNotIn("rank", self.chunks[1])  # input chunks are not mutated

    def test_agrees_with_semantic_search_in_both_modes(self):
        for diversify in (False, True):
            expected = semantic_search("attendance", self.model, self.chunks, self.index, 3, diversify=diversify)
            actual = retrieve("attendance", self.model, self.chunks, self.index, 3, diversify=diversify)
            self.assertEqual([r["chunk_id"] for r in actual], [r["chunk_id"] for r in expected])
            np.testing.assert_allclose([r["score"] for r in actual], [r["score"] for r in expected], atol=1e-6)

    def test_min_score_filters_weak_matches_and_renumbers(self):
        results = retrieve("attendance", self.model, self.chunks, self.index, 4, min_score=0.6, diversify=False)
        self.assertEqual([r["rank"] for r in results], [1, 2])
        self.assertEqual(retrieve("attendance", self.model, self.chunks, self.index, 4, min_score=0.99), [])

    def test_empty_question_and_bad_top_k_skip_encoding(self):
        self.assertEqual(retrieve("  ", None, self.chunks, self.index), [])
        self.assertEqual(retrieve("attendance", None, self.chunks, self.index, top_k=0), [])

    def test_rejects_mismatched_index(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            retrieve("attendance", self.model, self.chunks[:2], self.index)

    def test_format_context_includes_source_pages_and_continuation(self):
        results = [{**self.chunks[0], "rank": 1, "score": 0.9, "page_end": 3,
                    "context": [{**self.chunks[1], "page_start": 3, "page_end": 3}]}]
        text = format_context(results)
        self.assertIn("[Source 1: Doc0.pdf, pages 1-3]\nPassage number 0", text)
        self.assertIn("[Source 1: Doc1.pdf, page 3 (continuation)]\nPassage number 1", text)
        self.assertEqual(format_context([]), "")


if __name__ == "__main__":
    unittest.main()
