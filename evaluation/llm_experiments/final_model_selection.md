# UniGuide-AI — V3 Model Selection

## Decision
Use `gemini-3.1-flash-lite` as the provisional default for production integration. Keep `gemini-3.5-flash-lite` configurable as an alternative; do not implement silent automatic fallback without testing its quality and behavior.

## Evidence
The fixed-context development experiment contains 12 matched questions per model, 24 successful answers and one preserved 504 failure. The V2 baseline was imported unchanged. Uploaded question, context, prompt and baseline hashes match the recorded fingerprints. The original chunks and evaluator scripts were not supplied for independent hash verification here.

| Metric | Gemini 3.5 Flash Lite | Gemini 3.1 Flash Lite |
|---|---:|---:|
| Mean weighted quality | 83.54% | 95.83% |
| Correctness criterion = 2 | 11/12 | 12/12 |
| Correct abstentions | 2/2 | 2/2 |
| False abstentions | 1/10 | 0/10 |
| Material unsupported or contradicted claims | 1/12 | 0/12 |
| Valid individual citation IDs | 22/22 | 30/30 |
| Mean successful latency | 3.2958 s | 14.1687 s |
| Median successful latency | 3.0485 s | 9.6092 s |
| Mean successful tokens | 730 | 758.17 |
| Completion | 12/12 | 12/12 |
| Top-level attempt success | 12/12 | 12/13 |
| Automatic retries | 0 | 0 |
| Manual recovery runs | 0 | 1 |

## Review conventions
The uploaded rubric weights correctness, faithfulness, completeness, citation quality and instruction following at 30, 25, 20, 15 and 10 percent. The review is AI-assisted, non-blind and single-reviewer, pending team confirmation. Correctness evaluates the central answer separately from completeness. Missing reference details may also reduce instruction following because the prompt requires sufficient completeness. Grouped IDs receive partial citation and instruction credit even when all individual IDs exist. Appropriate abstention without citation is acceptable. DEV-09 baseline falsely denies that the context addresses the conditional final chance: this is counted as both false abstention and a contradicted claim about the context, not an invented policy. Its zero score reflects failure across the five criteria. DEV-06 candidate receives full credit for the requested reference procedure, although it does not separately state the beyond-control condition. These judgment calls should be reviewed by the team. Failed requests are excluded from quality scores. See quality_scores_v3.csv for individual rationales and comparison_summary_v3.json for metric denominators and source hashes.

## Rationale and trade-offs
The candidate provides a materially better answer to DEV-09 and a fuller procedure for DEV-06. Correctness and faithfulness take priority in a university policy assistant. The candidate is about 4.30 times slower on average for successful generations and experienced one 504 timeout (119.7712 seconds), recovered in a separate run. Successful latency excludes that failure. The candidate's mean token count is approximately 3.86% higher. No causal claim about model speed or service reliability is justified from these small noncontemporaneous samples.

## Limitations
Only 12 development questions, question-informed minimized contexts, fixed-context rather than end-to-end RAG, reused baseline, different execution times, potentially different provider defaults, availability-driven candidate screening, and rubric finalized after observing baseline answers. Earlier V1/V2 experiments remain historical evidence, not pooled into V3. The review is not independent human annotation. Citation ID validity does not establish citation entailment. The production model decision is provisional pending Student 4's independent evaluation and actual retrieval integration tests.

## Integration plan
Use a configurable model name, grounded prompt, retrieval relevance gate, evidence sufficiency checks, citation validation, bilingual responses, bounded error handling and explicit service-error messages. Validate actual retrieval-to-answer behavior before release. Begin Streamlit UI development after a minimal tested LLM interface is available. No additional model-comparison API calls are required at this stage.
