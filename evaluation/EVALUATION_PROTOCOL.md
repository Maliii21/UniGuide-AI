# UniGuide-AI — Evaluation Protocol

## 1. Objective

This evaluation aims to systematically assess the retrieval performance, answer quality, reliability, multilingual capabilities, and computational efficiency of UniGuide-AI.

The evaluation extends the preliminary experiments conducted during the embedding, retrieval, and LLM development stages using a larger, systematically annotated and reviewed dataset. Complete independence from all historical experiments cannot be guaranteed because some preliminary evaluation questions were unavailable.

## 2. Research Questions

- RQ1: Which embedding model provides the most effective semantic retrieval?
- RQ2: Which retrieval strategy achieves the highest retrieval quality?
- RQ3: Which relevance threshold provides the best balance between accepting relevant queries and rejecting irrelevant queries?
- RQ4: How accurately does the complete RAG pipeline answer university-related questions?
- RQ5: Does RAG improve answer quality compared with a standalone LLM?
- RQ6: How effectively does the system handle English and Arabic questions?

## 3. Knowledge Base

The evaluation uses the existing UniGuide-AI knowledge base:

| Property | Value |
|---|---:|
| University policy documents | 26 |
| Document pages | 121 |
| Text chunks | 202 |
| Supported query languages | English and Arabic |

The corpus must remain unchanged during comparative experiments.

## 4. Evaluation Dataset

The finalized dataset contains 70 annotated questions, including 56 human-reviewed questions and 14 AI-reviewed questions approved with dataset owner authorization. All 70 questions passed structural validation with zero errors and zero warnings.

### 4.1 Language and Question Distribution

| Question type | English | Arabic | Total |
|---|---:|---:|---:|
| Direct factual | 15 | 6 | 21 |
| Paraphrased | 15 | 6 | 21 |
| Procedural | 10 | 4 | 14 |
| Complex reasoning | 5 | 2 | 7 |
| Unanswerable | 5 | 2 | 7 |
| Total | 50 | 20 | 70 |

### 4.2 Dataset Splitting

| Split | English | Arabic | Total |
|---|---:|---:|---:|
| Development | 20 | 8 | 28 |
| Held-out test | 30 | 12 | 42 |
| Total | 50 | 20 | 70 |

The development set is used for model selection, retrieval configuration, and relevance threshold calibration.

The held-out test set is reserved for final evaluation after configuration selection.

Questions testing the same underlying facts must belong to the same split.

## 5. Dataset Schema

The dataset is stored in `evaluation/test_questions.csv` using UTF-8 encoding.

| Field | Description |
|---|---|
| question_id | Unique question identifier |
| question | Original question text |
| language | en or ar |
| expected_answer | Reference answer |
| relevant_chunk_ids | JSON array of supporting chunk IDs |
| source_document_ids | JSON array of supporting document IDs |
| category | Policy topic |
| question_type | Question classification |
| difficulty | easy, medium, or hard |
| answerable | true or false |
| split | development or test |
| fact_group_id | Identifier for related underlying facts |
| annotation_notes | Evidence, assumptions, and limitations |
| review_status | pending or verified |

## 6. Ground Truth Annotation

Reference answers must be derived from the university documents rather than generated system responses.

Annotation requirements:

- Identify all directly supporting chunks.
- Preserve numerical values, conditions, exceptions, and procedural requirements.
- Verify evidence against the original documents.
- Identify alternative supporting evidence where applicable.
- Record document version limitations.
- Avoid unresolved policy conflicts in ordinary answerable questions.
- Verify unanswerable questions against the complete corpus.
- Use empty evidence arrays for questions with no supporting answer evidence.
- Record the reason for unanswerability.
- Require documented review before freezing the dataset.
- Distinguish human verification from AI-assisted review.
- Record reviewer identity, review decisions, corrections, and limitations.
- Verify dataset structure and SHA-256 file integrity before freezing.
- Use the development set exclusively for model selection and parameter tuning.
- Reserve the held-out test set for final evaluation after configuration selection.

For complex questions, annotation notes must identify the individual facts required for a complete answer.

## 7. Data Leakage Prevention

Previously used development questions and their underlying facts must be excluded from the held-out test set.

Known preliminary experiments include:

- Embedding model comparison.
- Retrieval strategy comparison.
- Relevance threshold validation.
- LLM model comparison.
- End-to-end functional testing.

Paraphrases and translations of previously evaluated facts must not be treated as independent test questions.

The complete question list from the earlier relevance threshold experiment was not recovered. Therefore, complete independence from all historical experiments cannot be guaranteed.

No configuration tuning is permitted using held-out test results.

## 8. Embedding Model Evaluation

The following models will be compared:

| Model | Identifier |
|---|---|
| Multilingual E5 | intfloat/multilingual-e5-base |
| English E