import unittest
from unittest.mock import Mock

import numpy as np

from src.generate_embeddings import EMBEDDING_DIMENSION
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


if __name__ == "__main__":
    unittest.main()
