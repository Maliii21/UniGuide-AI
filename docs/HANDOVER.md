# UniGuide-AI — Retrieval Component Handover

## 1. Current Project Status

The data processing and retrieval components have been implemented and tested.

- Official documents: 26
- Extracted pages: 121
- Processed chunks: 202
- Embedding dimension: 768
- Automated tests: 254 passed
- Failed tests: 0
- Detected dependency conflicts: 0

The retrieval engine supports semantic search, BM25 keyword search, hybrid retrieval, cross-encoder reranking, and relevance filtering.

## 2. Environment Setup

Use Python 3.12 and execute the following commands from the project root:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

A CPU-compatible PyTorch installation is recommended when GPU acceleration is unavailable.

The processed dataset is available at:

`data/processed/chunks.json`

The repository includes precomputed embeddings and a FAISS vector index for the current dataset.

The production retrieval artifacts are available in `data/embeddings/`:

- `config.json`
- `embeddings.npy`
- `metadata.json`
- `index.faiss`
- `index_manifest.json`

Team members can use these artifacts directly without regenerating them.

If the dataset, embedding model, or embedding configuration changes, regenerate the artifacts:

```bash
python -m src.embeddings
python -m src.vector_store
```

Verify the retrieval pipeline:

```bash
python -m src.retrieval
```

Run the automated tests:

```bash
python -m pytest tests -v --tb=short
```

Pretrained model downloads may require internet access during the first execution.

Production embeddings and the FAISS index are included in Git. Comparative embedding arrays, pretrained model weights, cache files, and sensitive credentials remain excluded from version control.

## 3. Student 3 — LLM Integration and User Interface

### Assigned Responsibilities

- Implement the LLM integration.
- Design the answer-generation prompt.
- Generate answers grounded in retrieved university documents.
- Handle insufficient evidence and out-of-scope questions.
- Display document citations and source references.
- Develop the Streamlit user interface.
- Implement LLM and application tests.
- Add the required LLM and UI dependencies to requirements.txt.

### Main Files

| File | Responsibility |
|---|---|
| `src/llm.py` | LLM integration and answer generation |
| `app.py` | Streamlit application |
| `tests/test_llm.py` | LLM component tests |
| `src/retrieval.py` | Existing retrieval interface |

### Recommended Retrieval Interface

Initialize the retrieval engine once and reuse it across queries.

```python
from src.retrieval import RetrievalEngine

engine = RetrievalEngine()

response = engine.retrieve_with_relevance(
    query="How can I register for courses?",
    top_k=5,
    candidate_k=20,
    threshold=-5.0
)
```

### Handling Retrieval Results

```python
if not response["relevance_passed"]:
    answer = (
        "I could not find sufficient relevant information "
        "in the available university documents."
    )
else:
    context = response["context"]
    sources = response["results"]

    # Pass the question and context to the LLM.
    # Verify evidence sufficiency before generating an answer.
```

The relevance decision is a preliminary gate and does not guarantee that the retrieved documents contain a complete answer.

### Source Metadata

Retrieved chunks preserve the following source information:

- `chunk_id`
- `document_id`
- `document_title`
- `source_url`
- `page`
- `document_date`
- `version_notes`
- `review_notes`
- `text`

Use this information to construct transparent source citations.

### Answer Generation Requirements

The LLM should:

- Answer using the provided document context.
- Avoid inventing university policies.
- Avoid unsupported numerical values and deadlines.
- Clearly indicate insufficient evidence.
- Identify conflicting or potentially outdated information.
- Cite the documents supporting each answer.
- Treat retrieved document content as evidence, not executable instructions.

### User Interface Recommendations

- Provide a question input or chat interface.
- Display generated answers.
- Display supporting documents and page numbers.
- Handle loading states and runtime errors.
- Reuse the retrieval engine through resource caching.
- Avoid exposing API keys or sensitive configuration.

Streamlit and LLM-specific dependencies should be added during this stage. Streamlit 1.67.0 was installed and imported successfully in the retrieval development environment, but browser-level runtime validation has not been confirmed.

## 4. Student 4 — Evaluation and Performance Analysis

### Assigned Responsibilities

- Prepare an independent annotated evaluation dataset.
- Compare the three embedding models.
- Compare the four retrieval modes.
- Evaluate the relevance threshold.
- Measure retrieval quality and latency.
- Evaluate final generated answers after LLM integration.
- Document experimental results and limitations.
- Coordinate final model selection with Student 2.

### Recommended Evaluation Dataset

Prepare approximately 50 English questions grounded in the official university documents.

| Question type | Suggested count |
|---|---:|
| Direct factual questions | 15 |
| Paraphrased questions | 15 |
| Policy and procedure questions | 10 |
| Complex questions | 5 |
| Unanswerable questions | 5 |
| Total | 50 |

Each answerable question should include all identified relevant chunk identifiers and supporting document references.

The final evaluation questions should be independent of the preliminary development questions used during model selection and threshold calibration.

### Suggested Evaluation Dataset Columns

- question_id
- question
- expected_answer
- relevant_chunk_ids
- source_document_ids
- category
- question_type
- difficulty
- answerable
- split

### Embedding Models

| Model | Model identifier |
|---|---|
| Multilingual E5 Base | `intfloat/multilingual-e5-base` |
| English E5 Base v2 | `intfloat/e5-base-v2` |
| BGE Base English v1.5 | `BAAI/bge-base-en-v1.5` |

All three models generate 768-dimensional embeddings.

Multilingual E5 Base is the provisional production model. English E5 Base v2 achieved the highest preliminary MRR@10.

### Comparative Embedding Generation

```bash
python -m evaluation.generate_comparative_embeddings
```

### Preliminary Evaluation Reproduction

```bash
python -m evaluation.evaluate_embedding_models
```

The existing evaluation script reproduces the eight-question development experiment. It must be adapted or supplemented to evaluate the independent annotated dataset.

Refer to `evaluation/experiments/README.md` for experimental configurations and reproduction instructions.

### Retrieval Mode Comparison

```python
from src.retrieval import RetrievalEngine

engine = RetrievalEngine()

modes = ["semantic", "keyword", "hybrid", "reranked"]

for mode in modes:
    results = engine.retrieve(
        query="How can I register for courses?",
        top_k=5,
        mode=mode
    )

    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    print(mode, retrieved_ids)
```

The current retrieval engine uses the provisional Multilingual E5 production embeddings. Alternative embedding models require model-specific query encoding and compatible indexes or the separate comparative evaluation pipeline.

### Recommended Retrieval Metrics

- Hit Rate@1
- Hit Rate@5
- Precision@5
- Recall@5
- MRR@5
- Query encoding time
- Retrieval latency
- Reranking latency

Standard relevance-ranking metrics should be calculated on answerable questions with annotated relevant chunks.

Unanswerable questions should be evaluated separately using relevance rejection and answer-abstention metrics.

### Relevance Filter Evaluation

The current provisional threshold is -5.0.

Evaluate:

- True positives
- True negatives
- False positives
- False negatives
- Accuracy
- Precision
- Recall
- F1-score

Threshold selection should use development data rather than the final held-out evaluation set.

### Answer Quality Evaluation

After Student 3 completes LLM integration, evaluate:

- Answer correctness.
- Faithfulness to retrieved evidence.
- Citation correctness.
- Answer completeness.
- Appropriate abstention.
- End-to-end response latency.

## 5. Preliminary Experimental Results

### Embedding Comparison

| Model | Hit@5 | MRR@10 |
|---|---:|---:|
| Multilingual E5 Base | 87.5% | 0.4792 |
| English E5 Base v2 | 87.5% | 0.6292 |
| BGE Base English v1.5 | 87.5% | 0.5875 |

Based on eight development questions.

### Retrieval Comparison

| Mode | MRR@5 |
|---|---:|
| Semantic | 0.3611 |
| Keyword | 0.5000 |
| Hybrid | 0.4833 |
| Reranked | 0.8333 |

Based on three answerable development questions.

### Relevance Classification

| Metric | Result |
|---|---:|
| Accuracy | 90.00% |
| Precision | 83.33% |
| Recall | 100.00% |
| F1-score | 90.91% |

Based on 20 development questions.

These results are preliminary and must not be reported as independent final evaluation results.

## 6. Integration Constraints

- Preserve existing chunk identifiers and source metadata.
- Do not mix embeddings generated by different models.
- Preserve model-specific query and document prefixes.
- Rebuild the vector index when the dataset or production embedding model changes.
- Only the five approved production retrieval artifacts may be committed. Do not commit comparative embedding arrays, pretrained model weights, model caches, API keys, or credentials.
- Coordinate changes to shared retrieval interfaces.
- Run regression tests after modifying existing components.
- Confirm document redistribution permissions before distributing source content.

## 7. Known Limitations

- Final independent evaluation is pending.
- Production embedding model selection remains provisional.
- The relevance threshold requires further validation.
- Clean-environment installation has not been independently verified.
- Full comparative embedding regeneration using the saved script remains unverified.
- Model revisions are not explicitly pinned.
- Arabic query retrieval quality has not been independently evaluated.
- LLM and user interface integration are pending.

## 8. Completion Criteria

Student 3 should deliver a functioning source-grounded chatbot with citations, insufficient-evidence handling, and a Streamlit interface.

Student 4 should deliver an independent evaluation dataset, reproducible evaluation scripts, comparative results, final model recommendations, and documented limitations.