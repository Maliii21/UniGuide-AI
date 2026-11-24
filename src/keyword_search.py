
import math
import re

from collections import Counter



STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "from", "with", "by", "and", "or",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "you", "your", "we", "our",
    "he", "she", "they", "their", "them",
    "what", "which", "who", "how", "when", "where",
    "do", "does", "did", "can", "could",
    "would", "should", "will", "have", "has", "had"
}


TERM_NORMALIZATION = {

    "students": "student",

    "transferred": "transfer",
    "transfers": "transfer",
    "transferring": "transfer",

    "courses": "course",

    "withdrawal": "withdraw",
    "withdrawals": "withdraw",
    "withdrawn": "withdraw",
    "withdrawing": "withdraw",

    "registration": "register",
    "registered": "register",
    "registering": "register",

    "admissions": "admission",

    "requirements": "requirement",
    "required": "requirement",
    "requires": "requirement",

    "credits": "credit",
    "hours": "hour",

    "semesters": "semester",

    "examinations": "exam",
    "examination": "exam",
    "exams": "exam",

    "fees": "fee",
    "documents": "document"
}


def tokenize(text):
    """Tokenize and normalize academic terminology."""

    tokens = re.findall(r"\b\w+\b", text.lower())

    normalized_tokens = []

    for token in tokens:

        if token in STOPWORDS:
            continue

        token = TERM_NORMALIZATION.get(token, token)

        normalized_tokens.append(token)

    return normalized_tokens



class BM25Search:
    """Keyword-based retrieval using BM25."""

    def __init__(self, metadata, k1=1.5, b=0.75):

        if not metadata:
            raise ValueError("Metadata cannot be empty.")

        self.metadata = metadata
        self.k1 = k1
        self.b = b

        # Tokenize all document chunks
        self.corpus = [
            tokenize(chunk["text"])
            for chunk in metadata
        ]

        self.num_documents = len(self.corpus)

        self.doc_lengths = [
            len(tokens) for tokens in self.corpus
        ]

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.num_documents
        )

        if self.avg_doc_length == 0:
            raise ValueError("Corpus contains no tokens.")

        # Calculate term frequencies
        self.term_frequencies = [
            Counter(tokens) for tokens in self.corpus
        ]

        # Calculate document frequencies
        self.document_frequencies = Counter()

        for tokens in self.corpus:
            self.document_frequencies.update(set(tokens))

        # Calculate inverse document frequencies
        self.idf = {}

        for term, frequency in self.document_frequencies.items():

            self.idf[term] = math.log(
                1 + (
                    self.num_documents - frequency + 0.5
                ) / (
                    frequency + 0.5
                )
            )

    def search(self, query, top_k=5):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_tokens = set(tokenize(query))

        scores = []

        for index, frequencies in enumerate(
            self.term_frequencies
        ):

            score = 0.0

            for term in query_tokens:

                if term not in frequencies:
                    continue

                tf = frequencies[term]

                numerator = tf * (self.k1 + 1)

                denominator = tf + self.k1 * (
                    1 - self.b
                    + self.b * (
                        self.doc_lengths[index]
                        / self.avg_doc_length
                    )
                )

                score += (
                    self.idf[term]
                    * numerator
                    / denominator
                )

            scores.append((index, score))

        # Sort results by BM25 score
        scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for index, score in scores:

            if score <= 0:
                continue

            result = self.metadata[index].copy()
            result["bm25_score"] = float(score)

            results.append(result)

            if len(results) >= top_k:
                break

        return results


def main():

    from .vector_store import load_embedding_data

    print("Loading metadata...")

    _, metadata, _ = load_embedding_data()

    print("Building BM25 search engine...")

    engine = BM25Search(metadata)

    print("Indexed chunks:", engine.num_documents)
    print("Vocabulary size:", len(engine.idf))

    questions = [
        "What is the minimum GPA required for transfer students?",
        "How can I withdraw from a course?",
        "How can I register for courses?",
        "What is the weather in Alexandria today?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("Question:", question)

        results = engine.search(question, top_k=5)

        for rank, result in enumerate(results, start=1):

            print(
                f"Rank {rank} | "
                f"{result['chunk_id']} | "
                f"BM25: {result['bm25_score']:.6f}"
            )

            print("Document:", result["document_title"])

    print("\nBM25 search test completed successfully.")


if __name__ == "__main__":
    main()
