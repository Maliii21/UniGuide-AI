:::writing{variant="document" id="83621"}
# UniGuide-AI — LLM Evaluation Rubric

## 1. Objective

The objective is to compare candidate language models for grounded university policy question answering and select a suitable model for integration into UniGuide-AI.

## 2. Candidate Models

- Gemini 3.5 Flash Lite
- Gemini 3.1 Flash Lite

## 3. Experimental Setup

Both models are evaluated using the same 12 development questions, minimized policy contexts, prompt template, and request-handling configuration.

The Gemini 3.5 Flash Lite responses are imported from the verified V2 experiment. Gemini 3.1 Flash Lite responses will be generated during V3.

## 4. Quality Evaluation Criteria

| Criterion | Weight |
|---|---:|
| Correctness | 30% |
| Faithfulness | 25% |
| Completeness | 20% |
| Citation Quality | 15% |
| Instruction Following | 10% |
| Total | 100% |

Each criterion is scored using a three-point scale:

- 0: Incorrect, missing, or substantially inadequate.
- 1: Partially satisfactory.
- 2: Fully satisfactory.

### Correctness

Measures whether the answer correctly addresses the question according to the reference answer and supporting policy context.

### Faithfulness

Measures whether factual claims are supported by the supplied context without fabricated information.

### Completeness

Measures whether the answer includes all necessary conditions, procedures, exceptions, deadlines, and numerical requirements relevant to the question.

### Citation Quality

Measures whether citations use valid chunk identifiers, follow the required format, and support the associated claims.

For unanswerable questions, an appropriate abstention without citations receives full citation credit when no supporting citation is necessary. Fabricated or misleading citations are penalized.

### Instruction Following

Measures adherence to the requested language, grounding rules, response format, and other prompt instructions.

## 5. Weighted Quality Score

The weighted quality score is calculated as:

Quality Score = (30C + 25F + 20M + 15S + 10I) / 2

Where C, F, M, S, and I represent correctness, faithfulness, completeness, citation quality, and instruction following, respectively.

The maximum score is 100.

## 6. Additional Metrics

The evaluation also reports:

- Fully correct answer rate.
- Mean weighted quality score.
- Abstention accuracy.
- False abstention rate.
- Hallucination rate.
- Citation validity rate.
- Mean and median successful generation latency.
- Mean token usage.
- Completion rate.
- API attempt success rate.
- Retry count.

Operational metrics are reported separately from answer quality.

## 7. Evaluation Procedure

Responses are manually assessed against the reference answers and supplied policy contexts using the same scoring criteria for both models.

Quality scores and reviewer explanations are stored separately from the original generation results.

Failed API requests are recorded as operational failures rather than scored as generated answers. Quality comparisons report the number of evaluated responses and use matched questions when coverage differs.

## 8. Model Selection Strategy

Answer correctness and faithfulness are the primary considerations. Completeness, citation quality, instruction following, operational reliability, latency, and token consumption are also considered.

A model with material unsupported policy claims should not be selected solely because it achieves a higher aggregate score or faster response time.

## 9. Experimental Limitations

- The development dataset contains only 12 questions.
- The contexts were manually minimized using knowledge of the development questions.
- The experiment evaluates fixed-context generation rather than the complete retrieval-augmented pipeline.
- The models are evaluated at different times, so server conditions may differ.
- Baseline responses are reused rather than regenerated.
- Provider-default generation settings may differ between models.
- Candidate selection was influenced by API availability.
- Baseline answers were observed before this rubric was finalized.
- Independent end-to-end evaluation is required before drawing broader conclusions about system performance.
:::