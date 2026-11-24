# Embedding Model Comparison

## 1. Objective

This experiment compares three embedding models for semantic retrieval in UniGuide-AI, a university-focused Retrieval-Augmented Generation (RAG) system.

## 2. Dataset

- Source documents: 26 university policy documents.
- Total chunks: 202.
- Content: English and Arabic.
- Dataset file: `data/processed/chunks.json`.
- Development questions: 8.
- Total retrieval evaluations: 24.

## 3. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Embedding dimension | 768 |
| Maximum sequence length | 512 tokens |
| Batch size | 16 |
| Device | CPU |
| Normalization | L2 |
| Similarity metric | Cosine similarity |
| Search implementation | Exact NumPy search |
| Window overlap | 64 tokens |
| Window aggregation | Mean pooling followed by L2 normalization |

### Models

| Model | Document prefix | Query prefix |
|---|---|---|
| intfloat/multilingual-e5-base | passage: | query: |
| intfloat/e5-base-v2 | passage: | query: |
| BAAI/bge-base-en-v1.5 | None | Represent this sentence for searching relevant passages: |

A trailing space is included after each non-empty prefix.

## 4. Embedding Generation Results

| Metric | Multilingual E5 | English E5 | BGE Base |
|---|---:|---:|---:|
| Generated vectors | 202 | 202 | 202 |
| Encoding inputs | 202 | 204 | 204 |
| Windowed chunks | 0 | 2 | 2 |
| Maximum input tokens | 404 | 512 | 512 |
| Document encoding time (s) | 116.27 | 114.37 | 160.99 |

All generated embeddings passed shape, normalization, finite-value, and persistence validation.

## 5. Preliminary Retrieval Results

| Metric | Multilingual E5 | English E5 | BGE Base |
|---|---:|---:|---:|
| Hit Rate@1 | 25% | 50% | 50% |
| Hit Rate@3 | 62.5% | 75% | 50% |
| Hit Rate@5 | 87.5% | 87.5% | 87.5% |
| Hit Rate@10 | 87.5% | 87.5% | 87.5% |
| MRR@10 | 0.479167 | 0.629167 | 0.587500 |
| Mean target rank | 6.375 | 3.625 | 5.375 |
| Mean retrieval latency (ms) | 90.23 | 85.02 | 118.80 |

## 6. Reproducibility

The saved evaluation script was executed using the original embedding files.

- Completed evaluations: 24/24.
- Target rank agreement: 100%.
- Dataset integrity: Passed.
- Chunk alignment: Passed.
- Model configuration validation: Passed.

All reported ranking metrics matched the original experiment. Execution times varied between runs.

Full regeneration using the saved generation script has not yet been executed.

## 7. Model Selection

English E5 achieved the highest preliminary MRR@10 and Hit Rate@3.

However, Multilingual E5 remains the provisional production model because of its multilingual compatibility, existing pipeline integration, and ability to process all current chunks without additional token windows.

English E5 remains a strong alternative for the final evaluation.

The final model selection will be determined through a larger independent evaluation.

## 8. Known Limitations

- The development dataset contains only eight questions.
- Each question has one tracked target chunk rather than exhaustive relevance annotations.
- Hit Rate and MRR do not establish complete retrieval relevance.
- All three models failed to retrieve the tracked withdrawal request form within the top 10 results.
- Model-specific tokenization and windowing affect the comparison.
- Latency measurements depend on hardware and runtime conditions.
- The experiment evaluates semantic retrieval, not the complete RAG answer-generation pipeline.
- Model revisions are not pinned, so exact future numerical reproducibility is not guaranteed.

## 9. Reproduction Instructions

Run the following commands from the repository root.

Generate comparative embeddings:

```bash
python -m evaluation.generate_comparative_embeddings
```

Evaluate the generated embeddings:

```bash
python -m evaluation.evaluate_embedding_models
```

Use `--help` to inspect optional arguments, including custom cache directories, experiment directories, and offline model loading.

The default generation output directory is `data/embeddings/comparative/`. Generation refuses to overwrite an existing experiment directory.

## 10. Saved Results

| File | Description |
|---|---|
| chunk_ids.json | Ordered chunk identifiers |
| comparison_summary.json | Original embedding generation statistics |
| preliminary_retrieval_comparison.json | Original retrieval evaluation |
| reproduced_retrieval_comparison.json | Reproduced retrieval evaluation |

Generated embedding arrays and model weights are excluded from Git version control and can be regenerated locally.

## 11. Final Evaluation Handover

Student 4 will conduct an independent evaluation using a larger question dataset, comprehensive relevance annotations, retrieval metrics, retrieval strategy comparisons, and answer-quality assessment.

The eight development questions used here should not be treated as an independent final test set.