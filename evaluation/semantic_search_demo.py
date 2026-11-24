
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.embeddings import load_model, embed_query
from src.vector_store import (
    load_embedding_data,
    load_index,
    search
)


def main():

    print("Loading embedding data...")

    _, metadata, config = load_embedding_data()

    print("Loading FAISS index...")

    index = load_index()

    print("Loading embedding model...")

    model = load_model(
        model_name=config["model_name"],
        device="cpu"
    )

    print("\nSemantic search is ready.")
    print("Enter an English question.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a valid question.")
            continue

        query_embedding = embed_query(question, model)

        results = search(
            index,
            query_embedding,
            metadata,
            top_k=5
        )

        print("\nTop 5 retrieved chunks:")

        for rank, result in enumerate(results, start=1):

            print("\n" + "=" * 70)

            print("Rank:", rank)

            print("Chunk ID:", result["chunk_id"])

            print(
                "Similarity:",
                f"{result['similarity_score']:.6f}"
            )

            print("Document:", result["document_title"])

            print("Source:", result["source"])

            print("Page:", result["page"])

            print("Source URL:", result["source_url"])

            print("\nRetrieved Text:")

            print(result["text"])

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
