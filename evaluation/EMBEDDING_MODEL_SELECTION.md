# Embedding Model Evaluation and Selection

## 1. Objective

This experiment evaluates three embedding models for UniGuide-AI and investigates the impact of Arabic-to-English query translation on retrieval effectiveness.

## 2. Experimental Setup

- Corpus: 26 university documents, 121 pages, and 202 chunks.
- Evaluation dataset: Frozen version 1.0.0.
- Development set: 28 questions, including 25 answerable and 3 unanswerable questions.
- Answerable questions: 18 English and 7 Arabic.
- Embedding dimensions: 768.
- Search method: Exact cosine similarity using L2-normalized vectors.
- Execution environment: CPU.
- Evaluation cutoffs: K = 1, 3, 5, and 10.
- Metrics: Hit Rate, Precision, Recall, MRR, nDCG, and retrieval latency.

The 42 held-out test questions were excluded from model selection. Unanswerable questions were excluded from ranking metric averages.

Previously generated comparative embeddings were reused after validating corpus integrity, chunk alignment, embedding dimensions, data types, finite values, and normalization.

## 3. Embedding Model Comparison

| Model | Hit@5 | Recall@5 | MRR@10 | nDCG@10 | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Multilingual E5 Base | 72% | 70% | 0.6237 | 0.6683 | 63.11 ms |
| English E5 Base v2 | 72% | 70% | 0.5667 | 0.5908 | 94.75 ms |
| BGE Base EN v1.5 | 64% | 62% | 0.5533 | 0.5837 | 81.27 ms |

A total of 84 development evaluations were completed successfully.

Multilingual E5 achieved the highest overall MRR and nDCG, while matching English E5 in overall Recall@5.

## 4. Arabic Query Translation Ablation Study

Seven answerable Arabic development questions were translated into English using the existing production query translation component. Identical translations were evaluated across all three embedding models, resulting in 21 additional retrieval evaluations.

| Model | Arabic Recall@5 Before | Arabic Recall@5 After | Arabic MRR@10 After | Combined Recall@5 |
|---|---:|---:|---:|---:|
| Multilingual E5 | 14.29% | 96.43% | 0.8571 | 93% |
| English E5 | 0% | 96.43% | 0.7857 | 97% |
| BGE Base | 0% | 96.43% | 0.8095 | 89% |

Combined performance was calculated using the 18 original English questions and seven translated Arabic questions.

For Multilingual E5, query translation improved Arabic Recall@5 by 82.14 percentage points.

An AI-assisted semantic review found that all seven translations preserved the main meaning of the original questions.

## 5. Selected Production Model

**Selected model:** `intfloat/multilingual-e5-base`

Although English E5 achieved the highest combined Recall@5 after translation, Multilingual E5 was retained as an engineering trade-off based on:

- Stronger overall evidence ranking.
- Higher MRR on translated Arabic questions.
- Native multilingual support.
- Lower observed retrieval latency in the original-language experiment.
- Compatibility with the existing production embedding and FAISS infrastructure.

The four-percentage-point combined recall advantage of English E5 is acknowledged. The selected model is not claimed to outperform every alternative on every metric.

The existing Arabic-to-English query translation component will also be retained.

## 6. Limitations

- Model selection was based on a relatively small development dataset.
- Only seven answerable Arabic questions were included in the translation ablation.
- Ground-truth annotations may not include every valid supporting chunk.
- Historical model revisions were not pinned.
- Reused embedding artifacts were structurally validated but not independently regenerated.
- Translation quality was reviewed with AI assistance rather than independent human assessment.
- Retrieval latency excludes translation, reranking, and answer generation.
- Results measure semantic retrieval effectiveness rather than complete RAG answer quality.

## 7. Conclusion

A total of 105 retrieval evaluations were completed across the embedding comparison and translation ablation experiments.

The findings demonstrate that query translation substantially improves retrieval of English university policy documents for the evaluated Arabic questions.

Multilingual E5 with Arabic-to-English query translation will be used in subsequent retrieval strategy evaluation. The held-out test set remains reserved for final evaluation.