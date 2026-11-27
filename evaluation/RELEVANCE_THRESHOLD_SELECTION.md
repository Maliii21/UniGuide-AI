# Relevance Threshold Evaluation and Selection

## 1. Objective

Evaluate the query-level relevance gate and select a threshold that minimizes premature rejection of answerable questions while reducing the acceptance of irrelevant context.

## 2. Experimental Setup

- Dataset version: 1.0.0.
- Development questions: 28.
- Answerable questions: 25.
- Unanswerable questions: 3.
- Embedding model: Multilingual E5 Base.
- Retrieval strategy: Reranked Hybrid.
- Candidate pool: 20.
- RRF constant: 60.
- Top-K: 5.
- Threshold candidates evaluated: 78.
- Baseline threshold: -5.0.

Previously saved cross-encoder scores were reused with the existing production relevance filter. No additional model inference or API calls were required.

The held-out test questions were not evaluated.

## 3. Evaluation Results

| Metric | Baseline (-5.0) | Selected (-8.0) |
|---|---:|---:|
| True Positives | 24 | 25 |
| True Negatives | 2 | 1 |
| False Positives | 1 | 2 |
| False Negatives | 1 | 0 |
| Accuracy | 92.86% | 92.86% |
| Precision | 96.00% | 92.59% |
| Recall | 96.00% | 100.00% |
| F1-score | 96.00% | 96.15% |
| Specificity | 66.67% | 33.33% |
| Balanced Accuracy | 81.33% | 66.67% |

## 4. Error Analysis

At the baseline threshold, AR-006 was incorrectly rejected despite being answerable.

Lowering the threshold to -8.0 recovered this question but additionally accepted AR-008, an unanswerable question concerning university library opening hours.

EN-019 remained a false positive under both thresholds because the retrieved context was related to student ID replacement but did not establish the requested processing duration.

These results demonstrate that retrieval relevance does not necessarily imply answerability.

## 5. Selected Threshold

The threshold of -8.0 is selected provisionally for subsequent end-to-end evaluation.

The selection prioritizes answerable-question recall, followed by specificity. It also achieves the highest observed F1-score under the evaluated selection procedure.

Although the selected threshold reduces specificity and balanced accuracy, it eliminates premature rejection of answerable questions on the development dataset.

The downstream LLM is instructed to abstain when the supplied evidence is insufficient. Its effectiveness must be evaluated independently before adopting the selected threshold in production.

## 6. Limitations

- Only three unanswerable development questions were available.
- Answerability labels were used as a proxy for retrieval relevance.
- Relevance scores are not calibrated answerability probabilities.
- The selected threshold may overfit the development dataset.
- Perfect development recall does not guarantee equivalent held-out performance.
- Accepting an unanswerable question does not necessarily result in an incorrect generated answer.

## 7. Conclusion

The relevance gate is retained as an initial context screening mechanism rather than a definitive answerability classifier.

The selected threshold of -8.0 will be evaluated within the complete RAG pipeline before the final configuration is frozen. The existing production threshold remains unchanged until this validation is completed.