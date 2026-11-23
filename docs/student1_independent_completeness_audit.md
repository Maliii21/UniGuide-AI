# Student 1 independent completeness and correctness audit

**Verdict: PASS WITH LIMITATIONS.** The audited PDF corpus is ready for Student 2 handoff and for the Student 1 portion of the assignment. The current submission work should use the published branch or its clean clone. The audit reproduced the complete corpus and found no blocker for the current 26-PDF dataset. The original audit reproduced loader defects on malformed and DOCX fixtures outside that corpus. The approved local follow-up below records their resolution and the remaining Word numbering limitation. Policy currency, the registration deadline conflict, and redistribution permission still require separate decisions.

## Scope and actual project structure

I read the AIE417 assignment PDF and the four-student plan in `UniGuide_AI_Plan.docx`, then audited a fresh GitHub clone of Student 1 at `6c7b40bc10f6ab58395d4a48834a1894ca96f7ae` and a detached Student 2 working tree at `4888bfe30b34a81195cf5ecfa8ebc532cb97e5fa`. The production code and published tests came from those commits. The older `D:\UniGuide_AI` workspace supplied the validated baseline artifacts for byte comparisons.

| Published Student 1 path | Role |
| --- | --- |
| `data/sources.csv`, `scripts/acquire_sources.py` | Source records, SHA-256 checks, and local acquisition |
| `data/raw/`, `data/processed/` | Local inputs and generated JSON after setup |
| `src/document_loader.py`, `src/clean_text.py` | PDF/DOCX extraction and text normalization |
| `src/chunking.py`, `src/rebuild_chunks.py` | Chunking, metadata, and the rebuild command |
| `tests/test_document_processing.py` | Processing fixtures and pipeline tests |
| `README.md`, `requirements.txt`, `docs/student1_completion_report.md`, `docs/student1_final_verification_report.md` | Setup, scope, and previous evidence |

The published tree has one implementation per Student 1 module. I parsed its Python files and checked its README and report links. The placeholder `app.py`, embedding, retrieval, LLM and evaluation files on Student 1's branch are other team stages. The processing pipeline imports none of them. The raw PDFs and generated artifacts are kept local.

## Assignment requirement matrix

The assignment requires at least 20 relevant documents and a complete team RAG implementation. The plan assigns acquisition, extraction, cleaning and chunks to Student 1. The LLM, UI, and at least 10 questions evaluated with and without RAG are team deliverables outside this audit.

| Requirement | Implementation | Evidence and status | Remaining action |
| --- | --- | --- | --- |
| Acquire at least 20 distinct, relevant official sources | `data/sources.csv`, `scripts/acquire_sources.py` | **PASS**. A fresh acquisition fetched 26 AIU-hosted PDFs with exact manifest checksums. The corpus covers academic rules and university services relevant to students. | — |
| Record identity, title, URL, date, language and type | `data/sources.csv` | **PASS**. All rows have distinct identities and matching paths. All 23 populated printed dates occurred in extracted text. Unknown date fields are blank. | Keep printed, effective and publication dates distinct. |
| Verify applicable editions and permission for distribution | Manifest source and version notes | **NOT VERIFIED**. Officially hosted bytes and current applicability are separate. The reviewed PDFs provide no verified redistribution permission in the manifest. | Confirm versions where needed and permission or a course-approved submission method. |
| Load and extract the supported source documents | `src/document_loader.py` | **PARTIAL**. All 26 current PDFs and the ordinary DOCX fixture passed. Mislabeled formats, repaired PDFs and merged DOCX cells expose the edge cases below. | Approve and test narrowly scoped fixes. |
| Normalize text while preserving source wording and structure, and remove repeated headers | `src/clean_text.py`, PyMuPDF extraction | **PARTIAL**. NFC, whitespace, paragraphs, and current corpus word preservation passed. There is no general repeated-header removal step. The positional sample found repeated table column labels worth retaining. | Confirm whether generic header removal is required. Preserve meaningful table labels. |
| Flag image-only scans | Manifest and `read_documents()` | **PASS** for the submitted corpus with manual review. The manifest identifies five PDFs with image-only annex material. An entirely image-only test file is skipped with the same no-text message as a blank file. | Keep visible form limitations in the handoff. Add scan-specific diagnostics if needed. |
| Chunk near the plan's starting settings while protecting policy context | `src/chunking.py`, `src/rebuild_chunks.py` | **PASS**. Defaults of 500 words and 75-word overlap are inside the plan's 400–600 and 50–100 starting ranges. Actual passages fit the 256-token encoder budget and cover the extracted source text. | Keep overlap and the tokenizer settings documented. |
| Preserve stable IDs, file identity, PDF page and available section metadata | `chunk_document()` and manifest | **PASS**. 290 unique IDs, file joins, exact source spans, and one-based page endpoints independently checked. | For citations, join `source_path` to `local_file_path`; use the PDF page range. |
| Deliver reproducible chunks usable by Student 2 | Student 1 rebuild and published Student 2 loader and encoder | **PASS**. The independent rebuilds matched the validated JSON and vector/index artifacts. All six retrieval comparisons reproduced. | Pass the published retrieval entry point and the source flags to the team. |
| Verify documented Windows setup | `README.md`, `requirements.txt` | **PASS** on Windows Python 3.12.7. Clean virtual environment, dependency check, official acquisition, tests, and rebuild passed with a cached model. | First-time model acquisition requires access to the model host or a supplied cache. |

## Acquisition, integrity, and extraction

- The clean clone downloaded all 26 PDFs directly from the official HTTPS URLs. Each matched the manifest and the pre-existing local original. A tampered-local-file fixture was rejected before any download and preserved for inspection. All raw hashes and normalized full texts were unique. The closest of 325 normalized five-word-shingle pairs was 0.5402, between the domestic and international admissions policies, which cover different audiences.
- The 26 originals opened as PDFs with no encryption or repair. They contain **121 pages**, all with native text. Repeated extraction returned identical page-text records. Every published date or effective date left unknown in the manifest remains unknown here. All four semester-edition flags and three editorial/template flags still have supporting text in the files.
- A page-image scan corroborated nine annex pages across Transcript p. 3, Withdrawal pp. 2–3, Attendance pp. 2–3, Registration pp. 4–6 and Incomplete Course p. 2. Course Registration pp. 4–5 yielded just five and three text words. Native extraction alone cannot represent their form instructions.
- Verified PDF text directly at Attendance p. 1, Course Registration p. 2, Transcript p. 3 and Undergraduate Admission p. 1. The checked clauses, credit-hour values, heading order, printed date and one transcript table label/value appeared in the expected page text order. This is representative reading-order evidence. PDF text-stream order and flattened tables remain a general limitation.

## Cleaning, chunking, and citation checks

Cleaning was idempotent on all 121 pages. Comparing whitespace-delimited tokens before and after cleaning, with NFC applied on both sides, verified preservation of all 32,936 source words and symbols in order. The fixtures also exercise decomposed characters, bullets, blank lines, headings, numbers, percentages, dates and punctuation.

The **290 chunks** contain **84–229 words** and **127–256 tokens**, including special tokens. All **264 within-document transitions** overlap by exactly 75 words for this corpus. Repeated-text and multi-page fixtures checked continuity across a blank page and preservation of section labels. The normal builder prefers available paragraph/list/section boundaries. The token ceiling can still split long policy text, and Top-K retrieval sometimes omits a relevant continuation or detail.

For each chunk I matched its exact text substring against an independently reconstructed cleaned page stream, then calculated first and last page from the source offsets. This covered all source words with the accounted overlap. Every chunk joined to one manifest row and contained stable `chunk_id`, source fields and heading metadata. The separate manifest `document_id` is available through that join. The singular heading can be null when the plural field records multiple sections. Headings recognized in the source text are useful context, and the PDF page range is the verified citation anchor.

## Automated tests and coverage changes

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Published Student 1 | 10 | 0 | 0 |
| Local Student 1 test candidate | 11 | 0 | 0 |
| Published combined Student 2 | 23 | 0 | 0 |

The candidate changes are isolated to `tests/test_document_processing.py` in the audit clone. The existing loader test now includes encrypted and genuinely image-only PDFs alongside blank, corrupt, unsupported-extension and valid documents. The extraction and cleaning tests add determinism, empty-input and idempotence assertions. One new test reconstructs repeated text from exact overlap and checks stable IDs, page ranges and token limits across a blank page. The follow-up adds an assertion for the explicit error when a single oversized word exceeds the token budget. All tests use local synthetic fixtures and require no source PDF download. The corpus findings and defect reproductions below were checked separately.

## End-to-end reproducibility and Student 2 compatibility

The Student 1 rebuild produced **26 processed documents, zero skipped documents and 290 chunks**. The independent Student 2 rebuild in the detached worktree produced the same JSON plus **290 normalized 384-dimensional float32 embeddings**. Compared with the validated original files, all three byte sequences matched.

| Artifact | SHA-256 |
| --- | --- |
| `data/processed/document_chunks.json` | `f4b25ae5db983c2e22452186fe47bd797c0a94cf5757120968338f33fdcff098` |
| `data/embeddings/document_embeddings.npy` | `2a13f8b80e3f1ba8cab2601cf0f54ede530bee89e4eb854f192d5e3d72a1e15e` |
| `data/embeddings/document_embeddings.manifest.json` | `a0afd420b2f8a90af39090520c959ff1794c3244b73195592f1ee9ffe8daf2ae` |

Student 2's published reader accepted the exact eight-field chunk schema and verified index integrity. It reproduced all six documented cosine-only and diversified rankings, rounded scores and continuation hits. The attendance query returned the correct minimum from `4-Attendance-Policy.pdf`, page 1, as the first result with cosine score 0.5798. These results verify repeatability on this environment and the six tested questions.

## Clean clone setup

I cloned the published Student 1 branch from GitHub and created an empty virtual environment with `py -3.12 -m venv .venv`. Installing the published `requirements.txt` succeeded, `pip check` found no broken requirements, and the new interpreter reported Python 3.12.7. The three exact top-level pins resolve for this environment. The README steps for acquisition, testing, and rebuilding worked from the clone root. The acquisition command was tested against live AIU URLs and rerun to verify existing files. The rebuild used the local cached CPU MiniLM model and reproduced the baseline hash. Linux and a first-time model download were outside this verification.

## Severity and exact follow-up

| Finding | Severity and impact | Action before broader use |
| --- | --- | --- |
| Files disguised as PDFs can be indexed | **IMPORTANT**. PyMuPDF recognized synthetic HTML and SVG saved with a `.pdf` suffix, and the loader accepted them as PDFs. Their assigned PDF-style page citations can be misleading. All 26 current files were confirmed as genuine PDFs. | Resolved locally. Require `document.is_pdf` and regress both mislabeled formats. |
| Repairable corruption is silent | **IMPORTANT**. A synthetic truncated cross-reference table opened with `is_repaired=True` and the loader accepted recovered text without a warning. The verified corpus contains no repaired files. | Resolved locally. Reject `document.is_repaired` with a request for an intact copy and regress the repaired fixture. |
| DOCX layout can duplicate text or lose numbering | **IMPORTANT** if DOCX sources are added. A two-column merged cell yielded its text twice, and automatic list numbering was absent in a synthetic fixture. The submitted corpus is PDF-only. | Merged-cell duplication is resolved locally. Automatic list labels remain a documented limitation. |
| Repeated page headers are not generally removed | **LIMITATION** for the current corpus. The audit tested repeated strings in the upper 14% and lower 10% of pages at matching vertical positions. It found repeated admissions table labels such as “Field” and “Minimum HS Grade”, which carry meaning. | Review confirmed boilerplate individually before introducing a removal rule. |
| Nine pages contain image-only annex material | **LIMITATION** for English policy-body retrieval. The forms and examples in five documents remain outside the extracted corpus. | Use the PDF directly for exact form details or verify any transcription that enters scope. |
| Course Registration add deadline differs between body and forms | **IMPORTANT** and a blocker for giving a definitive main-semester add deadline. The p. 2 body gives week one. Forms on pp. 4–5 say add/drop through week two. | Obtain AIU clarification and carry the conflict into evaluation until it is resolved. |
| Four admissions editions and three editorial/template policies | **LIMITATION** for the assignment text corpus and **IMPORTANT** for live advice. The four admissions covers call for semesterly updates. Disability, Medical Service and Social Media retain editor/placeholder content. | Confirm applicability or approval before using these for current guidance. |
| Redistribution permission and local workspace drift | **IMPORTANT** for the submission package. The published instructions reproduce the local data. The older workspace still has pre-rename module filenames. | Submit from the published tree and confirm the course-approved method for delivering the dataset before distributing original PDFs or derived text. |
| Future reproducibility across dependency or model changes | **OPTIONAL** for this verified run. The top-level packages are pinned; transitive versions and the model revision are not locked by the repository. | Record a lock and model revision if the team requires byte-identical rebuilds across dates or machines. |

The original audit made no production-source edits, commits, merges, or pushes. The separately approved loader edits and permanent test candidate remain local for publication review. The current PDF-only technical handoff can proceed with the source notes above.

## Approved local loader follow-up

The corrections are confined to the Student 1 loader. PyMuPDF content detection rejects HTML and SVG disguised as PDFs. The loader rejects structurally repaired PDFs and tells the user to obtain an intact copy. Merged DOCX cells are visited once by their underlying cell identity, with distinct cells retaining their text and order.

All three new regression methods failed on the published loader and passed with the local fixes. The loader suite also covers ordinary tables, encrypted files, empty and image-only files, malformed input, repeated source text, stable IDs and token limits. The local follow-up passed **14 Student 1 tests and 27 combined tests**, with zero failures or skips. The Student 2 test tree overlaid the Student 1 loader and tests on the published retrieval commit.

Automatic DOCX list numbers and bullets remain outside the extracted paragraph text. The list item wording and any literal markers remain available. Check the original Word layout whenever the automatically rendered labels carry meaning.

Both isolated rebuilds processed 26 of 26 PDFs with zero skips. The Student 1 chunk JSON and Student 2 chunk JSON, embedding file, and index manifest matched the SHA-256 values listed above byte for byte. The six retrieval questions matched the recorded ranked passages, rounded scores and continuation passages in both modes. The validated originals were read only.
