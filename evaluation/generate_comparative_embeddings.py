
import argparse
import hashlib
import json
import time

from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "data" / "embeddings" / "comparative"
)

BATCH_SIZE = 16
WINDOW_OVERLAP = 64
EXPECTED_DIMENSION = 768


MODELS = [
    {
        "model_name": "intfloat/multilingual-e5-base",
        "alias": "multilingual_e5",
        "document_prefix": "passage: ",
        "query_prefix": "query: "
    },
    {
        "model_name": "intfloat/e5-base-v2",
        "alias": "english_e5",
        "document_prefix": "passage: ",
        "query_prefix": "query: "
    },
    {
        "model_name": "BAAI/bge-base-en-v1.5",
        "alias": "bge_base",
        "document_prefix": "",
        "query_prefix": (
            "Represent this sentence for "
            "searching relevant passages: "
        )
    }
]


# ==========================================
# 1. Dataset Loading
# ==========================================

def load_dataset(file_path):

    source_bytes = file_path.read_bytes()

    dataset_hash = hashlib.sha256(
        source_bytes
    ).hexdigest()

    chunks = json.loads(
        source_bytes.decode("utf-8-sig")
    )

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Invalid or empty dataset.")

    chunk_ids = set()

    for chunk in chunks:

        if not isinstance(chunk, dict):
            raise ValueError("Invalid chunk record.")

        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Invalid chunk ID.")

        if chunk_id in chunk_ids:
            raise ValueError("Duplicate chunk ID.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty chunk text.")

        chunk_ids.add(chunk_id)

    return chunks, dataset_hash


# ==========================================
# 2. Token Safety
# ==========================================

def count_tokens(text, tokenizer):

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False
        )
    )


def build_windows(text, model, prefix):

    tokenizer = model.tokenizer
    max_length = model.max_seq_length

    formatted_text = prefix + text

    if count_tokens(
        formatted_text,
        tokenizer
    ) <= max_length:

        return [formatted_text]

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False
    )

    prefix_cost = count_tokens(
        prefix,
        tokenizer
    )

    budget = max_length - prefix_cost

    if budget <= 0:
        raise ValueError("Invalid token budget.")

    overlap = min(
        WINDOW_OVERLAP,
        budget - 1
    )

    windows = []
    coverage = set()

    start = 0

    while start < len(token_ids):

        end = min(
            start + budget,
            len(token_ids)
        )

        while end > start:

            window_text = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

            formatted_text = prefix + window_text

            actual_count = count_tokens(
                formatted_text,
                tokenizer
            )

            if actual_count <= max_length:
                break

            end -= 1

        if end <= start:
            raise ValueError(
                "Cannot build a valid window."
            )

        windows.append(formatted_text)

        coverage.update(range(start, end))

        if end == len(token_ids):
            break

        start = max(
            start + 1,
            end - overlap
        )

    if len(coverage) != len(token_ids):
        raise ValueError("Incomplete token coverage.")

    return windows


# ==========================================
# 3. Embedding Validation
# ==========================================

def validate_embeddings(
    embeddings,
    expected_count,
    expected_dim
):

    expected_shape = (
        expected_count,
        expected_dim
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Invalid shape: {embeddings.shape}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError("Invalid embedding dtype.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            "Embeddings are not normalized."
        )


# ==========================================
# 4. Generate Model Embeddings
# ==========================================

def generate_embeddings(
    chunks,
    config,
    output_dir,
    dataset_hash,
    device,
    cache_dir,
    local_files_only
):

    print("\n" + "=" * 70)
    print("Model:", config["model_name"])

    start = time.perf_counter()

    model = SentenceTransformer(
        config["model_name"],
        device=device,
        cache_folder=cache_dir,
        local_files_only=local_files_only
    )

    loading_seconds = (
        time.perf_counter() - start
    )

    dimension = model.get_embedding_dimension()

    if dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Unexpected dimension: {dimension}"
        )

    all_windows = []
    window_ranges = []
    windowed_chunk_ids = []

    start = time.perf_counter()

    for chunk in chunks:

        windows = build_windows(
            chunk["text"],
            model,
            config["document_prefix"]
        )

        first = len(all_windows)

        all_windows.extend(windows)

        last = len(all_windows)

        window_ranges.append(
            (first, last)
        )

        if len(windows) > 1:

            windowed_chunk_ids.append(
                chunk["chunk_id"]
            )

    maximum_input_tokens = max(
        count_tokens(text, model.tokenizer)
        for text in all_windows
    )

    if maximum_input_tokens > model.max_seq_length:
        raise ValueError("Token limit exceeded.")

    preparation_seconds = (
        time.perf_counter() - start
    )

    print("Chunks:", len(chunks))
    print("Encoding inputs:", len(all_windows))
    print("Windowed chunks:", len(windowed_chunk_ids))
    print("Maximum input tokens:", maximum_input_tokens)

    start = time.perf_counter()

    window_embeddings = model.encode(
        all_windows,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    window_embeddings = np.asarray(
        window_embeddings,
        dtype=np.float32
    )

    encoding_seconds = (
        time.perf_counter() - start
    )

    validate_embeddings(
        window_embeddings,
        len(all_windows),
        dimension
    )

    # Aggregate windows into one vector per chunk.

    start = time.perf_counter()

    final_embeddings = []

    for first, last in window_ranges:

        vectors = window_embeddings[first:last]

        if len(vectors) == 1:

            vector = vectors[0]

        else:

            vector = vectors.mean(axis=0)

            norm = np.linalg.norm(vector)

            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(
                    "Invalid aggregated vector."
                )

            vector = vector / norm

        final_embeddings.append(vector)

    final_embeddings = np.asarray(
        final_embeddings,
        dtype=np.float32
    )

    aggregation_seconds = (
        time.perf_counter() - start
    )

    validate_embeddings(
        final_embeddings,
        len(chunks),
        dimension
    )

    output_file = (
        output_dir / f"{config['alias']}.npy"
    )

    np.save(
        output_file,
        final_embeddings,
        allow_pickle=False
    )

    reloaded = np.load(
        output_file,
        allow_pickle=False
    )

    if not np.array_equal(
        final_embeddings,
        reloaded
    ):
        raise ValueError(
            "Embedding persistence validation failed."
        )

    print("Embedding shape:", final_embeddings.shape)
    print("Encoding time:", round(encoding_seconds, 3))
    print("Saved:", output_file)
    print("Validation: PASSED")

    return {
        "model": config["model_name"],
        "alias": config["alias"],
        "device": device,
        "document_prefix": config["document_prefix"],
        "query_prefix": config["query_prefix"],
        "dataset_sha256": dataset_hash,
        "embedding_file": output_file.name,
        "embedding_shape": list(final_embeddings.shape),
        "dtype": str(final_embeddings.dtype),
        "batch_size": BATCH_SIZE,
        "window_overlap": WINDOW_OVERLAP,
        "aggregation": "mean_then_l2",
        "normalized": True,
        "windowed_chunks": len(windowed_chunk_ids),
        "windowed_chunk_ids": windowed_chunk_ids,
        "total_encoding_inputs": len(all_windows),
        "maximum_input_tokens": maximum_input_tokens,
        "model_loading_seconds": loading_seconds,
        "preparation_seconds": preparation_seconds,
        "encoding_seconds": encoding_seconds,
        "aggregation_seconds": aggregation_seconds,
        "chunks_per_second": (
            len(chunks) / encoding_seconds
        ),
        "file_size_kib": (
            output_file.stat().st_size / 1024
        )
    }


# ==========================================
# 5. Main Experiment
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate comparable embeddings "
            "for UniGuide-AI."
        )
    )

    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=CHUNKS_FILE
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--cache-dir",
        default=None
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true"
    )

    args = parser.parse_args()

    chunks, dataset_hash = load_dataset(
        args.chunks_file
    )

    output_dir = args.output_dir.resolve()

    production_dir = (
        BASE_DIR / "data" / "embeddings"
    ).resolve()

    if output_dir == production_dir:
        raise ValueError(
            "Do not write experimental files "
            "directly into the production directory."
        )

    # Refuse to overwrite previous experiments.

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    print("Dataset chunks:", len(chunks))
    print("Dataset SHA256:", dataset_hash)
    print("Output directory:", output_dir)

    summaries = []

    for config in MODELS:

        summary = generate_embeddings(
            chunks=chunks,
            config=config,
            output_dir=output_dir,
            dataset_hash=dataset_hash,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only
        )

        summaries.append(summary)

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    with open(
        output_dir / "chunk_ids.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunk_ids,
            file,
            ensure_ascii=False,
            indent=2
        )

    with open(
        output_dir / "comparison_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summaries,
            file,
            ensure_ascii=False,
            indent=2
        )

    final_hash = hashlib.sha256(
        args.chunks_file.read_bytes()
    ).hexdigest()

    if final_hash != dataset_hash:
        raise ValueError(
            "Dataset changed during the experiment."
        )

    print("\n" + "=" * 70)
    print("COMPARATIVE EMBEDDING SUMMARY")

    for summary in summaries:

        print(
            summary["alias"],
            "| Shape:", summary["embedding_shape"],
            "| Windows:", summary["windowed_chunks"],
            "| Time:",
            round(summary["encoding_seconds"], 3)
        )

    print("Dataset integrity: PASSED")
    print("All embedding validations: PASSED")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
