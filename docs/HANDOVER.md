# UniGuide-AI — System Integration and Evaluation Handover

## 1. Current Project Status

The core UniGuide-AI system has been implemented, integrated, tested, and deployed.

| Component | Status |
|---|---|
| Data collection and preprocessing | Completed |
| Embedding generation and vector storage | Completed |
| Semantic and keyword retrieval | Completed |
| Hybrid retrieval and cross-encoder reranking | Completed |
| Relevance filtering | Completed |
| Preliminary LLM comparison | Completed |
| Gemini integration | Completed |
| Automatic Arabic query translation | Completed |
| Citation validation | Completed |
| RAG pipeline integration | Completed |
| Conversation management | Completed |
| Streamlit chatbot | Completed |
| Shared retrieval resources | Completed |
| Cloud deployment | Completed |
| Functional acceptance testing | Completed |
| Independent comprehensive evaluation | Pending final verification |

### System Statistics

| Metric | Value |
|---|---:|
| Official university documents | 26 |
| Extracted pages | 121 |
| Processed chunks | 202 |
| Embedding dimension | 768 |
| Automated tests passed | 555/555 |
| Automated test failures | 0 |
| Online functional acceptance checks | 10/10 |

**Repository:** https://github.com/Maliii21/UniGuide-AI

**Live application:** https://uniguide-ai.streamlit.app/

**Current chatbot branch:** `feature/chatbot-ui`

The remaining major academic task is independent evaluation and performance analysis.

## 2. Environment Setup

Use Python 3.12 and execute the following commands from the project root:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The application uses CPU-compatible PyTorch. A dedicated GPU is not required.

### API Configuration

Create a `.env` file using `.env.example` as a template.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Configure:

```dotenv
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_TIMEOUT_MS=120000
```

Never commit actual API credentials.

### Production Retrieval Artifacts

The repository includes the following precomputed artifacts in `data/embeddings/`:

- `config.json`
- `embeddings.npy`
- `metadata.json`
- `index.faiss`
- `index_manifest.json`

Initial embedding regeneration is not required.

Verify the retrieval pipeline:

```bash
python -m src.retrieval
```

If the dataset, embedding model, or relevant embedding configuration changes, regenerate the artifacts:

```bash
python -m src.embeddings
python -m src.vector_store
python -m src.retrieval
```

Pretrained model downloads may require internet access during the first execution.

### Run the Chatbot

```bash
python -m streamlit run app.py
```

Open http://localhost:8501.

### Run Regression Tests

```bash
python -m pytest tests/ -q
```

The latest recorded complete test suite contains 555 passing tests.

## 3. Student 3 — Completed Deliverables

Student 3 completed the LLM integration, end-to-end RAG orchestration, chatbot interface, and cloud deployment.

### Implementation Files

| File | Responsibility |
|---|---|
| `src/llm.py` | Gemini configuration, prompts, generation, retries, and API metadata |
| `src/query_translator.py` | Arabic detection and query translation |
| `src/citation_validator.py` | Citation identifier extraction and validation |
| `src/rag_pipeline.py` | End-to-end RAG orchestration |
| `src/conversation.py` | Session conversation management |
| `src/shared_retrieval.py` | Shared and synchronized retrieval resources |
| `app.py` | Bilingual Streamlit chatbot |
| `.streamlit/config.toml` | Streamlit configuration |
| `.env.example` | Environment configuration template |
| `.gitattributes` | Cross-platform retrieval artifact integrity |
| `evaluation/llm_experiments/` | LLM evaluation results and documentation |

### Production Configuration

| Parameter | Value |
|---|---|
| Embedding model | intfloat/multilingual-e5-base |
| Embedding dimension | 768 |
| Vector index | FAISS IndexFlatIP |
| Keyword retrieval | BM25 |
| Rank fusion | Reciprocal Rank Fusion |
| RRF constant | 60 |
| Reranking model | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Candidate count | 20 |
| Final retrieved chunks | 5 |
| Relevance threshold | -5.0 |
| Default LLM | gemini-3.1-flash-lite |
| Alternative evaluated LLM | gemini-3.5-flash-lite |
| Request timeout | 120,000 milliseconds |
| Execution device | CPU |

The embedding model, LLM selection, and relevance threshold remain provisional pending independent evaluation.

### End-to-End Workflow

1. Receive the original user question.
2. Detect Arabic content.
3. Translate Arabic or mixed-language questions into English for retrieval.
4. Perform semantic and BM25 retrieval.
5. Combine rankings through RRF.
6. Rerank retrieved candidates.
7. Apply relevance filtering.
8. Generate an answer using the retrieved context and original question.
9. Handle insufficient evidence or generation errors.
10. Validate generated citation identifiers.
11. Present the answer and validated source references.

## 4. Evaluation Interfaces

### Retrieval-Only Evaluation

Initialize the retrieval engine once and reuse it across questions.

```python
from src.retrieval import RetrievalEngine

engine = RetrievalEngine(device="cpu")

results = engine.retrieve(
    query="How can I register for courses?",
    top_k=5,
    mode="reranked"
)

for result in results:
    print(result["chunk_id"])
    print(result["document_title"])
```

Supported retrieval modes:

- `semantic`
- `keyword`
- `hybrid`
- `reranked`

### Relevance-Aware Retrieval

```python
response = engine.retrieve_with_relevance(
    query="How can I register for courses?",
    top_k=5,
    candidate_k=20,
    rrf_k=60,
    threshold=-5.0
)

print(response["relevance_passed"])
print(response["max_score"])
print(response["retained_count"])
print(response["results"])
```

The relevance gate does not guarantee complete answerability.

### End-to-End RAG Evaluation

Use the production RAG pipeline to evaluate the integrated system rather than manually reconstructing its individual stages.

```python
from src.rag_pipeline import RAGPipeline

questions = [
    "Is summer registration mandatory?",
    "How can I appeal my final grade?",
    "هل التسجيل في الفصل الدراسي الصيفي إجباري؟"
]

with RAGPipeline(device="cpu") as pipeline:
    for question in questions:
        result = pipeline.ask(question)

        print("Question:", result.question)
        print("Status:", result.status)
        print("Answer:", result.answer)
        print("Retrieval query:", result.retrieval_query)
        print("Retrieved chunks:", result.retrieved_chunks)
        print("Relevance passed:", result.relevance_passed)
        print("Citation validation:", result.citation_validation)
        print("Total latency:", result.total_latency_seconds)
```

The pipeline should be initialized once per evaluation run to avoid unnecessary model loading.

### Response Statuses

| Status | Interpretation |
|---|---|
| `success` | Answer generated with valid citation identifiers |
| `insufficient_context` | Available evidence was rejected or considered insufficient |
| `citation_error` | Generated citation identifiers failed validation |
| `llm_error` | Translation or answer-generation failure |

Technical failures must be recorded separately from correct abstention.

A successful status does not automatically establish factual correctness or semantic citation faithfulness.

### Source Metadata

Retrieved chunks preserve:

- `chunk_id`
- `document_id`
- `document_title`
- `source_url`
- `page`
- `document_date`
- `version_notes`
- `review_notes`
- `text`

Preserve these identifiers when constructing evaluation ground truth.

## 5. Student 4 — Evaluation and Performance Analysis

### Assigned Responsibilities

- Prepare an independent annotated evaluation dataset.
- Compare the three embedding models.
- Compare the four retrieval modes.
- Independently evaluate the relevance threshold.
- Measure retrieval quality and latency.
- Evaluate final generated answers.
- Compare responses with and without RAG.
- Evaluate Arabic and English system behavior.
- Document results, errors, and limitations.
- Coordinate final model recommendations with Students 2 and 3.

### Recommended Evaluation Dataset

Prepare approximately 50 English questions grounded in the official university documents, with additional Arabic questions for bilingual evaluation where feasible.

| Question Type | Suggested Count |
|---|---:|
| Direct factual questions | 15 |
| Paraphrased questions | 15 |
| Policy and procedure questions | 10 |
| Complex questions | 5 |
| Unanswerable questions | 5 |
| Total | 50 |

Each answerable question should include its expected answer and all identified supporting chunk identifiers.

Unanswerable questions should include a documented reason explaining why the available knowledge base cannot answer them.

### Suggested Dataset Columns

```text
question_id
question
expected_answer
relevant_chunk_ids
source_document_ids
category
question_type
difficulty
answerable
split
```

Use `evaluation/test_questions.csv` for the annotated evaluation dataset and `evaluation/results.csv` for evaluation results, or introduce clearly documented supplementary result files.

### Evaluation Independence

- Do not reuse preliminary development questions as independent final test questions.
- Avoid placing paraphrases of the same underlying fact across development and test splits.
- Use development data for threshold calibration and model selection.
- Preserve a held-out test set for final reporting.
- Freeze the selected configuration before final testing.
- Validate expected answers and supporting chunks against the original documents.

## 6. Embedding Model Evaluation

Compare the following models:

| Model | Identifier |
|---|---|
| Multilingual E5 Base | intfloat/multilingual-e5-base |
| English E5 Base v2 | intfloat/e5-base-v2 |
| BGE Base English v1.5 | BAAI/bge-base-en-v1.5 |

All three models generate 768-dimensional embeddings.

### Generate Comparative Embeddings

```bash
python -m evaluation.generate_comparative_embeddings
```

### Reproduce the Preliminary Comparison

```bash
python -m evaluation.evaluate_embedding_models
```

The existing evaluation script reproduces the eight-question development experiment. It must be adapted or supplemented to evaluate the independent annotated dataset.

Refer to `evaluation/experiments/README.md`.

Use model-specific query encoding, document prefixes, and compatible embedding indexes.

Do not overwrite production retrieval artifacts during comparative experiments.

## 7. Retrieval Evaluation

Compare semantic, keyword, hybrid, and reranked retrieval using identical questions and ground-truth annotations.

```python
from src.retrieval import RetrievalEngine

engine = RetrievalEngine(device="cpu")

for mode in ["semantic", "keyword", "hybrid", "reranked"]:
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

### Recommended Metrics

- Hit Rate@1
- Hit Rate@5
- Precision@5
- Recall@5
- MRR@5
- MRR@10
- Query encoding time
- Retrieval latency
- Reranking latency

Calculate standard ranking metrics on answerable questions with annotated relevant chunks.

Evaluate unanswerable questions separately through relevance rejection and answer-abstention metrics.

### Relevance Threshold Evaluation

The current threshold is -5.0.

Measure:

- True positives and true negatives.
- False positives and false negatives.
- Accuracy.
- Precision.
- Recall.
- F1-score.

Use development data to select a threshold and report final results on held-out questions.

## 8. End-to-End Answer Quality Evaluation

Evaluate the production pipeline using the independent annotated question set.

### Evaluation Criteria

| Criterion | Description |
|---|---|
| Correctness | Agreement with verified university policy |
| Completeness | Coverage of required facts, conditions, and exceptions |
| Faithfulness | Support for generated claims in retrieved evidence |
| Citation validity | Correct citation identifiers |
| Citation faithfulness | Whether cited passages support associated claims |
| Appropriate abstention | Correct rejection of unanswerable questions |
| False abstention | Rejection of answerable questions |
| Latency | End-to-end processing time |
| Token consumption | Translation and generation token usage |
| Reliability | API failures, retries, and technical errors |

Use consistent scoring criteria and retain raw model outputs for reproducibility.

Independent human verification is recommended, particularly for policy conditions, deadlines, numerical values, and Arabic responses.

### RAG versus Non-RAG Comparison

Compare two experimental conditions:

| Condition | Configuration |
|---|---|
| Without RAG | Same Gemini model answering without retrieved university documents |
| With RAG | Production UniGuide-AI pipeline with retrieved context and citation validation |

Use identical evaluation questions and the same LLM model.

The non-RAG baseline requires a separate evaluation implementation. Do not simply pass an empty context to the production grounded-answer prompt, because that prompt is explicitly designed to abstain when evidence is missing.

Keep generation settings consistent wherever possible and document unavoidable differences between the two conditions.

Do not require document citations from the non-RAG baseline when no documents are supplied. Evaluate shared answer-quality criteria fairly and report RAG-specific citation metrics separately.

### Suggested Result Fields

```text
question_id
experiment_condition
model
retrieval_mode
retrieved_chunk_ids
generated_answer
response_status
relevance_passed
max_relevance_score
correctness_score
completeness_score
faithfulness_score
citation_validity
citation_faithfulness
abstention_correct
retrieval_latency_seconds
translation_latency_seconds
generation_latency_seconds
total_latency_seconds
total_tokens
api_attempts
error_type
review_notes
```

Record unavailable or inapplicable metrics explicitly rather than replacing them with misleading zero values.

## 9. Preliminary Experimental Results

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

### LLM Comparison

| Metric | Gemini 3.5 Flash-Lite | Gemini 3.1 Flash-Lite |
|---|---:|---:|
| Completed questions | 12/12 | 12/12 |
| Weighted quality score | 83.54% | 95.83% |
| Mean successful latency | 3.30 s | 14.17 s |
| Valid citation identifiers | 22/22 | 30/30 |
| Correct abstention | 2/2 | 2/2 |
| False abstention | 1/10 | 0/10 |
| Successful-response tokens | 8,760 | 9,098 |

Based on 12 development questions with fixed contexts and AI-assisted, non-blind scoring.

Gemini 3.1 Flash-Lite was provisionally selected because of its stronger evaluated answer quality despite higher latency.

Refer to:

- `evaluation/llm_experiments/final_model_selection.md`
- `evaluation/llm_experiments/comparison_results_v3.json`
- `evaluation/llm_experiments/comparison_summary_v3.json`
- `evaluation/llm_experiments/quality_scores_v3.csv`
- `evaluation/llm_experiments/quality_scoring_rubric.md`

### Arabic Translation Case Study

| Metric | Before Translation | After Translation |
|---|---:|---:|
| Known supporting chunks retrieved | 0/2 | 2/2 |
| Tracked supporting-chunk Recall@5 | 0% | 100% |
| Answer status | Insufficient context | Success |

This diagnostic experiment used one Arabic question about mandatory summer registration. It does not establish general Arabic retrieval accuracy.

All results in this section are preliminary and must not be presented as independent final evaluation results.

## 10. Deployment and Operational Notes

The application is deployed through Streamlit Community Cloud.

- Entry point: `app.py`
- Current branch: `feature/chatbot-ui`
- Python version: 3.12
- Credentials: Streamlit Secrets
- Retrieval execution: CPU

The application uses cached shared retrieval resources while maintaining separate session conversation histories.

JSON retrieval artifacts must preserve LF line endings to ensure SHA-256 hash consistency across Windows and Linux.

Do not delete the deployment branch until the application has been successfully migrated to the intended production branch.

## 11. Integration Constraints

- Preserve existing chunk identifiers and source metadata.
- Do not mix embeddings generated by different models.
- Preserve model-specific document and query prefixes.
- Rebuild the vector index when the dataset or production embedding model changes.
- Do not overwrite production artifacts during comparative experiments.
- Do not commit comparative embedding arrays, pretrained model weights, caches, API keys, or credentials.
- Preserve cross-platform artifact integrity.
- Coordinate changes to shared retrieval and generation interfaces.
- Run regression tests after modifying existing components.
- Record evaluation configurations and raw results.
- Confirm document redistribution permissions before distributing source content.

## 12. Known Limitations

- Independent comprehensive evaluation remains pending.
- Production embedding and LLM selections remain provisional.
- The relevance threshold requires independent validation.
- Preliminary experiments use small development samples.
- Citation identifier validation does not establish semantic faithfulness.
- Arabic retrieval quality requires broader evaluation.
- Query translation may alter the meaning of some questions.
- Conversation history is not used as multi-turn retrieval context.
- Large-scale concurrent-user performance has not been benchmarked.
- Pretrained model revisions are not explicitly pinned.
- Full comparative embedding regeneration using the saved script remains independently unverified.
- Some university policies may contain outdated, conflicting, or incomplete information.

## 13. Student 4 Completion Criteria

The final evaluation deliverables should include:

- An independently annotated evaluation dataset.
- Reproducible evaluation scripts.
- Comparative embedding model results.
- Comparative retrieval-mode results.
- Relevance threshold evaluation.
- End-to-end answer-quality evaluation.
- RAG versus non-RAG comparison.
- Arabic and English evaluation results.
- Latency, token consumption, and reliability analysis.
- Final model and configuration recommendations.
- Documented failure cases and limitations.
- Updated project documentation containing verified final results.

The final evaluation must distinguish development experiments, automated regression tests, functional acceptance checks, and independent system performance measurements.

## 14. Handover Status

The integrated chatbot is operational and has completed local and cloud functional testing.

Students 1, 2, and 3 have completed their core implementation responsibilities. Student 4 can proceed with independent evaluation using the existing retrieval engine, production RAG pipeline, annotated university documents, and preliminary experimental artifacts.

Any recommended changes to production models or configurations should be reviewed collaboratively and followed by regression testing and deployment verification.