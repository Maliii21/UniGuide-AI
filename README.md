## Embeddings and Retrieval

This stage turns the 202 chunks from `data/processed/chunks.json` into vectors and finds the chunks most similar to a student's question (Arabic or English).

### Model selection

| Setting | Value |
|---|---|
| Model | `intfloat/multilingual-e5-base` |
| Embedding dimension | 768 |
| Maximum input | 512 tokens |
| Languages | Arabic and English |

The model supports both languages, and it uses the same tokenizer as the chunking stage, so the 400-token chunks leave room for the document title, the `passage: ` prefix and the special tokens. Other candidates from the project guide (`multilingual-e5-small`, `multilingual-e5-large`, BGE-M3) were **not** benchmarked.

### Vector storage

The index is stored as a NumPy matrix plus a JSON metadata file, not in ChromaDB or FAISS.

- The assignment requires the similarity calculation to be implemented in NumPy.
- 202 chunks x 768 values (155,136 numbers, about 0.6 MB) fit easily in memory, so an exact search over every chunk is fast.
- The same files can be loaded into a vector database later if the corpus grows.

### How it works

1. Each chunk is embedded as `passage: <document title>` + new line + `<chunk text>`. Many chunks never name their policy, and adding the title improved ranking (see Evaluation).
2. Each question is embedded as `query: <question>`.
3. All vectors are L2-normalized (unit length) and validated: correct shape, no NaN or infinite values.
4. Cosine similarity is computed with NumPy: `cos(q, d) = (q . d) / (||q|| * ||d||)`.
5. The Top-K chunks (default `K = 5`) are returned with their score, rank and source metadata.

### Files

| File | Purpose |
|---|---|
| `src/embeddings.py` | Validate chunks, create embeddings in batches, save and load the index |
| `src/retrieval.py` | Cosine similarity, Top-K search, `Retriever` class, prompt context formatting |
| `data/embeddings/embeddings.npy` | Float32 matrix, shape `(202, 768)` |
| `data/embeddings/chunks_metadata.json` | The chunks in the same row order as the matrix |
| `data/embeddings/embedding_config.json` | Model name, dimension, prefixes, chunk count |

`data/embeddings/` is generated locally and ignored by Git.

### Running

Run from the project root, after the data processing steps:

```bash
python src/embeddings.py
python -X utf8 src/retrieval.py "What is the minimum attendance percentage?" --top-k 5
```

The first run downloads the model (about 1 GB).

### Using the retriever (for the LLM stage)

```python
from src.retrieval import Retriever, format_context

retriever = Retriever()                      # loads the model and the index once
results = retriever.search("What is the minimum attendance percentage?", top_k=5)
context = format_context(results)            # source blocks for the LLM prompt
```

Each result contains the chunk metadata (`chunk_id`, `document_id`, `document_title`, `page`, `source_url`, `text`), plus `score` and `rank`.

### Tests

```bash
python -m pytest tests/test_embeddings.py tests/test_retrieval.py -v -rs
```

| Suite | Tests |
|---|---:|
| `tests/test_embeddings.py` | 17 |
| `tests/test_retrieval.py` | 13 |

Twenty-five of these tests use a small fake model and run offline. Five load the real model (model loading, Arabic and English embeddings, and two search tests including an Arabic question that finds English text). If the model cannot be loaded, those five are skipped and `-rs` shows why. Together with the data processing tests, the full suite is 58 tests, all passing.

### Evaluation

Twelve English and Arabic questions were checked against the document we expected to answer each one. This measures whether the **right document** appears, not whether the exact answer passage does.

| Embedding input | Correct document first | In top 3 | In top 5 |
|---|---:|---:|---:|
| Chunk text only | 6/12 | 11/12 | 11/12 |
| Document title + chunk text (used) | 7/12 | 11/12 | 12/12 |

This is why the default is `top_k = 5`. The sample is small and the questions and expected documents were chosen by us, so treat it as a sanity check, not a benchmark.

### Limitations

- **Arabic questions and English chunks.** For the Arabic question "What is the minimum attendance percentage?", the top results were Arabic form text. The chunk that states the 75% rule (`AIU-018-P001-C001`) was not in the top 5, although the English version of the question finds it in the top 2.
- **Scores are close together.** Relevant and irrelevant chunks scored between 0.73 and 0.79 for the same question, so a score threshold (`min_score`) is not a reliable way to detect "no answer".
- **Possible improvement.** Search with both the Arabic question and its English translation and keep each chunk's higher score.
