
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    BASE_DIR / "evaluation" / "llm_experiments"
)

CHUNKS_FILE = (
    BASE_DIR / "data" / "processed" / "chunks.json"
)

QUESTIONS_FILE = (
    EXPERIMENT_DIR / "development_questions.json"
)

OUTPUT_FILE = (
    EXPERIMENT_DIR / "evaluation_contexts.json"
)


# Source excerpt boundaries.
# None means retain the original boundary.

EXCERPT_RULES = {
    "AIU-003-P002-C002": (
        "Academic Semesters & Course Registration",
        "Program Study Duration"
    ),
    "AIU-008-P003-C001": (
        "The transcript issued with the AIU seal",
        "VI."
    ),
    "AIU-024-P002-C001": (
        None,
        None
    ),
    "AIU-015-P002-C001": (
        "- Students",
        "- Alumni and Others"
    ),
    "AIU-016-P003-C002": (
        "- Security",
        "- Vandalism and Harassment"
    ),
    "AIU-021-P001-C001": (
        "SCOPE OF POLICY",
        None
    ),
    "AIU-021-P002-C001": (
        "Required Documents:",
        "VI."
    ),
    "AIU-024-P002-C002": (
        None,
        "VI."
    ),
    "AIU-021-P001-C002": (
        None,
        None
    ),
    "AIU-003-P003-C001": (
        "Study Dismissal & Academic Probation",
        "AIU reserves the right to issue a disciplinary"
    ),
    "AIU-015-P001-C001": (
        None,
        None
    ),
    "AIU-015-P003-C001": (
        None,
        None
    )
}


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def extract_excerpt(text, start_marker, end_marker):

    start = (
        text.index(start_marker)
        if start_marker is not None
        else 0
    )

    end = (
        text.index(end_marker, start)
        if end_marker is not None
        else len(text)
    )

    excerpt = text[start:end].strip()

    if not excerpt:
        raise ValueError("Empty evidence excerpt.")

    if excerpt not in text:
        raise ValueError("Excerpt differs from source.")

    return excerpt


def main():

    questions = load_json(QUESTIONS_FILE)
    chunks = load_json(CHUNKS_FILE)

    chunk_map = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    context_ids = list(dict.fromkeys(
        chunk_id
        for question in questions
        for chunk_id in question["context_chunk_ids"]
    ))

    if set(context_ids) != set(EXCERPT_RULES):
        raise ValueError("Unexpected context chunk IDs.")

    output = {}

    original_chars = 0
    minimized_chars = 0

    for chunk_id in context_ids:

        chunk = chunk_map[chunk_id]
        original_text = chunk["text"]

        start_marker, end_marker = EXCERPT_RULES[chunk_id]

        excerpt = extract_excerpt(
            original_text,
            start_marker,
            end_marker
        )

        output[chunk_id] = {
            "chunk_id": chunk_id,
            "document_title": chunk["document_title"],
            "page": chunk["page"],
            "text": excerpt
        }

        original_chars += len(original_text)
        minimized_chars += len(excerpt)

        print(
            chunk_id,
            "| Original:", len(original_text),
            "| Minimized:", len(excerpt),
            "| PASSED"
        )

    reduction = (
        1 - minimized_chars / original_chars
    ) * 100

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nUnique context chunks:", len(output))
    print("Original characters:", original_chars)
    print("Minimized characters:", minimized_chars)
    print(f"Character reduction: {reduction:.2f}%")

    print("Output file:", OUTPUT_FILE)
    print("Context preparation: PASSED")
    print("API requests: 0")


if __name__ == "__main__":
    main()
