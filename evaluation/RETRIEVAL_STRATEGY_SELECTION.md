# Retrieval Strategy Evaluation and Selection

## 1. Objective

Evaluate the effectiveness and computational efficiency of four retrieval strategies and determine a suitable configuration for the UniGuide-AI RAG system.

## 2. Experimental Setup

- Embedding model: `intfloat/multilingual-e5-base`
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Corpus: 202 chunks from 26 university documents.
- Dataset: Frozen version 1.0.0.
- Development questions: 28 (25 answerable and 3 unanswerable).
- Retrieval strategies: Semantic, BM25, Hybrid, and Reranked Hybrid.
- Candidate pool: 20.
- RRF constant: 60.
- Evaluation cutoffs: K = 1, 3, 5, and 10.
- Execution environment: CPU.
- Arabic questions were translated into English using the existing production translation component.

All 112 retrieval evaluations were completed successfully. The relevance threshold was disabled to isolate retrieval effectiveness. The 42 held-out test questions were not evaluated.

## 3. Overall Results

| Strategy | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | Mean Latency | P95 Latency |
|---|---:|---:|---:|---:|---:|---:|
| Semantic | 96% | 93% | 0.8333 | 0.8394 | 69.06 ms | 86.11 ms |
| BM25 | 96% | 95% | 0.8467 | 0.8626 | 0.88 ms | 1.14 ms |
| Hybrid | 100% | 99% | 0.9133 | 0.9177 | 68.28 ms | 76.29 ms |
| Reranked Hybrid | 100% | 99% | 0.9267 | 0.9325 | 1545.31 ms | 1717.88 ms |

Ranking metrics were calculated over the 25 answerable questions. Latency measurements included all 28 development questions.

## 4. Language-Specific Results

| Strategy | English Recall@5 | English nDCG@5 | Translated Arabic Recall@5 | Translated Arabic nDCG@5 |
|---|---:|---:|---:|---:|
| Semantic | 91.67% | 0.8288 | 96.43% | 0.8667 |
| BM25 | 94.44% | 0.8639 | 96.43% | 0.8591 |
| Hybrid | 100% | 0.8950 | 96.43% | 0.9760 |
| Reranked Hybrid | 100% | 0.9434 | 96.43% | 0.9046 |

## 5. Findings

### Hybrid Retrieval

Hybrid retrieval improved Recall@5 by six percentage points over semantic search and four percentage points over BM25.

The results demonstrate the complementary strengths of semantic similarity and keyword matching. Hybrid retrieval successfully recovered relevant evidence for all 25 answerable development questions.

### Cross-Encoder Reranking

Reranking maintained the same Hit@5 and Recall@5 as hybrid retrieval while improving overall MRR@5 and nDCG@5.

However, reranking reduced ranking effectiveness on the translated Arabic subgroup and increased mean retrieval latency by approximately 22.6 times compared with hybrid retrieval.

### Efficiency

Hybrid retrieval provided the strongest observed balance between evidence coverage, ranking quality, and computational efficiency.

Reranked hybrid retrieval achieved the highest overall ranking metrics but introduced additional computational overhead.

## 6. Selected Configuration

The existing reranked hybrid retrieval configuration is retained for subsequent relevance threshold evaluation.

- Embedding model: Multilingual E5 Base.
- Keyword retrieval: BM25.
- Fusion method: Reciprocal Rank Fusion.
- Candidate pool: 20.
- RRF constant: 60.
- Reranker: MS MARCO MiniLM-L6-v2.
- Production output: Top 5 chunks.

This selection is an engineering trade-off based on overall ranking quality and compatibility with the existing cross-encoder-based relevance filtering mechanism.

Hybrid retrieval remains a viable lower-latency alternative. The benefit of relevance filtering will be evaluated separately before finalizing the complete pipeline.

## 7. Limitations

- The evaluation uses a small development dataset.
- Ranking metrics depend on the completeness of ground-truth annotations.
- The Arabic subgroup contains only seven answerable questions.
- Latency measurements exclude model loading, query translation, and answer generation.
- Measurements were collected on a single CPU environment.
- Relevance scores do not directly establish question answerability.
- The experiments measure retrieval quality rather than final answer quality.

## 8. Conclusion

Hybrid retrieval substantially improved evidence coverage compared with standalone semantic and keyword retrieval.

Cross-encoder reranking provided a smaller additional improvement in overall ranking quality at a considerably higher computational cost.

The saved retrieval rankings and reranker scores will be reused for relevance threshold optimization without repeating model inference.