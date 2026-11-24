
import json

import numpy as np
import pytest

from src.embeddings import (
    load_chunks,
    embed_documents,
    embed_query,
    validate_embeddings,
    save_embeddings,
)


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "C001",
            "text": "University course registration policy."
        },
        {
            "chunk_id": "C002",
            "text": "Students must attend their classes."
        }
    ]


@pytest.fixture
def sample_embeddings():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=np.float32
    )


# ==========================================
# Fake Embedding Model
# ==========================================

class FakeTokenizer:

    def encode(self, text, **kwargs):
        return list(range(len(text.split()) + 2))


class FakeModel:

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 512
        self.last_inputs = None
        self.last_kwargs = None

    def get_sentence_embedding_dimension(self):
        return 3

    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):

        self.last_inputs = texts
        self.last_kwargs = kwargs

        return np.tile(
            np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32
            ),
            (len(texts), 1)
        )


@pytest.fixture
def fake_model():
    return FakeModel()


# ==========================================
# 1. Dataset Loading Tests
# ==========================================

def test_load_valid_chunks(tmp_path, sample_chunks):

    path = tmp_path / "chunks.json"

    path.write_text(
        json.dumps(sample_chunks),
        encoding="utf-8"
    )

    result = load_chunks(path)

    assert result == sample_chunks


def test_load_empty_dataset(tmp_path):

    path = tmp_path / "chunks.json"

    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        load_chunks(path)


def test_duplicate_chunk_ids(tmp_path):

    chunks = [
        {"chunk_id": "C001", "text": "First"},
        {"chunk_id": "C001", "text": "Second"}
    ]

    path = tmp_path / "chunks.json"

    path.write_text(
        json.dumps(chunks),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_chunks(path)


def test_empty_chunk_text(tmp_path):

    chunks = [
        {"chunk_id": "C001", "text": "   "}
    ]

    path = tmp_path / "chunks.json"

    path.write_text(
        json.dumps(chunks),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_chunks(path)


# ==========================================
# 2. Embedding Validation Tests
# ==========================================

def test_valid_embeddings(sample_embeddings):

    validate_embeddings(
        sample_embeddings,
        expected_count=2,
        expected_dim=3
    )


def test_invalid_embedding_count(sample_embeddings):

    with pytest.raises(ValueError):

        validate_embeddings(
            sample_embeddings,
            expected_count=3,
            expected_dim=3
        )


def test_invalid_embedding_dimension(sample_embeddings):

    with pytest.raises(ValueError):

        validate_embeddings(
            sample_embeddings,
            expected_count=2,
            expected_dim=4
        )


def test_non_normalized_embeddings():

    embeddings = np.array(
        [[2.0, 0.0, 0.0]],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        validate_embeddings(
            embeddings,
            expected_count=1,
            expected_dim=3
        )


def test_nan_embeddings():

    embeddings = np.array(
        [[np.nan, 0.0, 1.0]],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        validate_embeddings(
            embeddings,
            expected_count=1,
            expected_dim=3
        )


def test_infinite_embeddings():

    embeddings = np.array(
        [[np.inf, 0.0, 1.0]],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        validate_embeddings(
            embeddings,
            expected_count=1,
            expected_dim=3
        )


def test_zero_vector():

    embeddings = np.zeros(
        (1, 3),
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        validate_embeddings(
            embeddings,
            expected_count=1,
            expected_dim=3
        )


# ==========================================
# 3. Document Encoding Tests
# ==========================================

def test_document_embedding_shape(
    sample_chunks,
    fake_model
):

    embeddings = embed_documents(
        sample_chunks,
        fake_model
    )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32

    assert np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0
    )


def test_document_prefix(
    sample_chunks,
    fake_model
):

    embed_documents(
        sample_chunks,
        fake_model
    )

    assert fake_model.last_inputs == [
        "passage: " + chunk["text"]
        for chunk in sample_chunks
    ]

    assert (
        fake_model.last_kwargs["normalize_embeddings"]
        is True
    )


def test_document_token_limit(fake_model):

    chunks = [
        {
            "chunk_id": "C001",
            "text": "word " * 600
        }
    ]

    with pytest.raises(ValueError):

        embed_documents(
            chunks,
            fake_model
        )

    assert fake_model.last_inputs is None


# ==========================================
# 4. Query Encoding Tests
# ==========================================

def test_query_embedding_shape(fake_model):

    embedding = embed_query(
        "How can I register?",
        fake_model
    )

    assert embedding.shape == (3,)

    assert np.isclose(
        np.linalg.norm(embedding),
        1.0
    )


def test_query_prefix(fake_model):

    embed_query(
        "How can I register?",
        fake_model
    )

    assert fake_model.last_inputs == [
        "query: How can I register?"
    ]

    assert (
        fake_model.last_kwargs["normalize_embeddings"]
        is True
    )


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query(query, fake_model):

    with pytest.raises(ValueError):

        embed_query(
            query,
            fake_model
        )


def test_query_token_limit(fake_model):

    query = "word " * 600

    with pytest.raises(ValueError):

        embed_query(
            query,
            fake_model
        )

    assert fake_model.last_inputs is None


# ==========================================
# 5. Persistence Tests
# ==========================================

def test_save_embeddings(
    tmp_path,
    sample_chunks,
    sample_embeddings
):

    output_dir = tmp_path / "embeddings"

    save_embeddings(
        sample_embeddings,
        sample_chunks,
        output_dir=output_dir,
        model_name="test-model"
    )

    assert (output_dir / "embeddings.npy").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "config.json").exists()

    loaded_embeddings = np.load(
        output_dir / "embeddings.npy",
        allow_pickle=False
    )

    metadata = json.loads(
        (output_dir / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    config = json.loads(
        (output_dir / "config.json").read_text(
            encoding="utf-8"
        )
    )

    assert np.array_equal(
        loaded_embeddings,
        sample_embeddings
    )

    assert metadata == sample_chunks

    assert config["model_name"] == "test-model"
    assert config["embedding_dimension"] == 3
    assert config["number_of_embeddings"] == 2
    assert config["normalized"] is True
    assert config["document_prefix"] == "passage: "
    assert config["query_prefix"] == "query: "
