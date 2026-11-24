# python -m pytest tests/test_embeddings.py -v -rs
#
# Most tests use a small fake model, so they run offline and instantly.
# Tests that need the real multilingual-e5-base model are skipped
# automatically when it cannot be loaded (-rs shows the reason).
import hashlib
import json

import numpy as np
import pytest

from src.embeddings import (
    CONFIG_NAME,
    EMBEDDING_DIMENSION,
    MAX_CHUNK_TOKENS,
    METADATA_NAME,
    MODEL_NAME,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    VECTORS_NAME,
    build_index,
    embed_chunks,
    embed_passages,
    embed_query,
    load_chunks,
    load_index,
    load_model,
    passage_text,
    save_index,
    validate_chunks,
    validate_embeddings,
)


def unit_vector(text):
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
    vector = np.random.default_rng(seed).normal(size=EMBEDDING_DIMENSION)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class FakeModel:
    """Deterministic stand-in for the E5 model (no download needed)."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **options):
        self.calls.append((texts, options))

        if isinstance(texts, str):
            return unit_vector(texts)

        return np.stack([unit_vector(text) for text in texts])


@pytest.fixture
def chunks():
    return [
        {
            "document_id": "AIU-TEST",
            "page_id": "AIU-TEST-P001",
            "chunk_id": "AIU-TEST-P001-C001",
            "chunk_index": 1,
            "document_title": "Attendance Policy",
            "source": "attendance.pdf",
            "source_url": "https://aiu.edu.eg/attendance.pdf",
            "language": "English",
            "page": 1,
            "token_count": 9,
            "text": "Students must attend at least 75% of lectures.",
        },
        {
            "document_id": "AIU-TEST",
            "page_id": "AIU-TEST-P002",
            "chunk_id": "AIU-TEST-P002-C001",
            "chunk_index": 1,
            "document_title": "Attendance Policy",
            "source": "attendance.pdf",
            "source_url": "https://aiu.edu.eg/attendance.pdf",
            "language": "Arabic",
            "page": 2,
            "token_count": 8,
            "text": "يجب على الطلاب حضور ٧٥٪ على الأقل من المحاضرات.",
        },
    ]


@pytest.fixture(scope="module")
def real_model():
    try:
        return load_model()
    except Exception as error:
        pytest.skip(f"multilingual-e5-base could not be loaded ({type(error).__name__}): {error}")


# Test 1: Model loading (real model)
def test_load_model(real_model):
    dimension = getattr(real_model, "get_embedding_dimension", None) or real_model.get_sentence_embedding_dimension
    assert dimension() == EMBEDDING_DIMENSION


def test_model_name_matches_chunking_tokenizer():
    pytest.importorskip("transformers")

    from src.chunking import CHUNK_SIZE, MODEL_NAME as CHUNKING_MODEL_NAME

    assert MODEL_NAME == CHUNKING_MODEL_NAME
    assert CHUNK_SIZE <= MAX_CHUNK_TOKENS


# Test 2: Reading chunks
def test_load_chunks(tmp_path, chunks):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    assert load_chunks(path) == chunks

    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "missing.json")

    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_chunks(path)


# Test 3: Vector dimensions
def test_embedding_shape(chunks):
    vectors = embed_chunks(chunks, FakeModel())

    assert vectors.shape == (2, EMBEDDING_DIMENSION)
    assert vectors.dtype == np.float32
    assert embed_query("attendance", FakeModel()).shape == (EMBEDDING_DIMENSION,)


# Test 4: One vector per chunk
def test_embedding_count(chunks):
    many = [
        {**chunks[0], "chunk_id": f"AIU-TEST-P001-C{number:03d}"}
        for number in range(1, 8)
    ]

    assert embed_chunks(many, FakeModel(), batch_size=3).shape[0] == 7

    class DropsOneRow(FakeModel):
        def encode(self, texts, **options):
            return super().encode(texts, **options)[:-1]

    with pytest.raises(ValueError, match="Expected 2 embeddings"):
        embed_chunks(chunks, DropsOneRow())


# Test 5: Empty input
def test_empty_input(chunks):
    for bad in ([], "single string", [""], ["ok", "   "]):
        with pytest.raises(ValueError):
            embed_passages(bad, FakeModel())

    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            embed_query(bad, FakeModel())

    with pytest.raises(ValueError):
        validate_chunks([])

    with pytest.raises(ValueError, match="empty text"):
        validate_chunks([{**chunks[0], "text": "   "}])


# Test 6: Normalization
def test_normalized_embeddings(chunks):
    model = FakeModel()
    vectors = embed_chunks(chunks, model)

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert model.calls[0][1]["normalize_embeddings"] is True

    class NotNormalized(FakeModel):
        def encode(self, texts, **options):
            return super().encode(texts, **options) * 5

    with pytest.raises(ValueError, match="normalized"):
        embed_chunks(chunks, NotNormalized())


# Test 7: E5 prefixes
def test_e5_prefixes_are_applied(chunks):
    model = FakeModel()

    embed_chunks(chunks, model)
    embed_query("  What is the minimum attendance?  ", model)

    passages = model.calls[0][0]
    assert passages == [
        PASSAGE_PREFIX + chunk["document_title"] + "\n" + chunk["text"]
        for chunk in chunks
    ]
    assert model.calls[1][0] == QUERY_PREFIX + "What is the minimum attendance?"


# Test 7b: The document title is embedded with the chunk
def test_passage_text_adds_the_document_title(chunks):
    assert passage_text(chunks[0]) == "Attendance Policy\n" + chunks[0]["text"]
    assert passage_text({**chunks[0], "document_title": "  "}) == chunks[0]["text"]
    assert passage_text({"text": "No title here."}) == "No title here."


# Test 8: Arabic support (real model)
def test_arabic_embedding(real_model):
    vector = embed_query("ما هو الحد الأدنى لنسبة الحضور؟", real_model)

    assert vector.shape == (EMBEDDING_DIMENSION,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)
    assert np.all(np.isfinite(vector))


# Test 9: English support (real model)
def test_english_embedding(real_model):
    vectors = embed_passages(
        ["Students must attend at least 75% of lectures."], real_model
    )

    assert vectors.shape == (1, EMBEDDING_DIMENSION)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-3)
    assert np.all(np.isfinite(vectors))


# Test 10: Metadata is preserved
def test_metadata_preservation(tmp_path, chunks):
    save_index(chunks, embed_chunks(chunks, FakeModel()), tmp_path)

    loaded_chunks, _, _ = load_index(tmp_path)

    assert loaded_chunks == chunks
    assert loaded_chunks[1]["text"] == chunks[1]["text"]  # Arabic survives


# Test 11: Saving the index
def test_save_index(tmp_path, chunks):
    vectors = embed_chunks(chunks, FakeModel())

    save_index(chunks, vectors, tmp_path)

    assert np.load(tmp_path / VECTORS_NAME).shape == (2, EMBEDDING_DIMENSION)
    assert (tmp_path / METADATA_NAME).exists()

    config = json.loads((tmp_path / CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["model_name"] == MODEL_NAME
    assert config["dimension"] == EMBEDDING_DIMENSION
    assert config["chunk_count"] == 2

    with pytest.raises(ValueError):
        save_index(chunks, vectors[:1], tmp_path)


# Test 12: Reloading the index
def test_load_index(tmp_path, chunks):
    vectors = embed_chunks(chunks, FakeModel())
    save_index(chunks, vectors, tmp_path)

    loaded_chunks, loaded_vectors, config = load_index(tmp_path)

    assert np.array_equal(loaded_vectors, vectors)
    assert [chunk["chunk_id"] for chunk in loaded_chunks] == [
        chunk["chunk_id"] for chunk in chunks
    ]
    assert config["chunk_count"] == 2

    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "missing")

    config_path = tmp_path / CONFIG_NAME
    config["chunk_count"] = 99
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        load_index(tmp_path)


# Test 13: Unique IDs and required metadata
def test_unique_ids(chunks):
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        validate_chunks([chunks[0], {**chunks[1], "chunk_id": chunks[0]["chunk_id"]}])

    broken = {key: value for key, value in chunks[0].items() if key != "source_url"}

    with pytest.raises(ValueError, match="source_url"):
        validate_chunks([broken])


def test_chunk_must_fit_the_model_input_limit(chunks):
    assert validate_chunks([{**chunks[0], "token_count": MAX_CHUNK_TOKENS}])

    with pytest.raises(ValueError, match="limit"):
        validate_chunks([{**chunks[0], "token_count": MAX_CHUNK_TOKENS + 1}])

    with pytest.raises(ValueError, match="token_count"):
        validate_chunks([{**chunks[0], "token_count": 0}])


# Test 14: No NaN or infinity
def test_finite_values():
    good = np.eye(2, EMBEDDING_DIMENSION, dtype=np.float32)

    assert validate_embeddings(good, expected_count=2).shape == (2, EMBEDDING_DIMENSION)

    for bad_value in (np.nan, np.inf):
        bad = good.copy()
        bad[0, 0] = bad_value

        with pytest.raises(ValueError, match="NaN or infinite"):
            validate_embeddings(bad)

    with pytest.raises(ValueError, match="shape"):
        validate_embeddings(np.ones((2, 5), dtype=np.float32))


# Test 15: Full build from chunks.json
def test_build_index_end_to_end(tmp_path, chunks):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    built_chunks, vectors = build_index(
        chunks_path, tmp_path / "index", model=FakeModel()
    )

    reloaded_chunks, reloaded_vectors, _ = load_index(tmp_path / "index")

    assert built_chunks == reloaded_chunks == chunks
    assert np.array_equal(vectors, reloaded_vectors)
