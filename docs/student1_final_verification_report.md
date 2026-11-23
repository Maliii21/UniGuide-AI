# Student 1 final verification and quality assurance

**Verdict: PASS WITH LIMITATIONS.** The renamed processing pipeline is ready for handoff review. The Student 1 rebuild processed 26 official AIU PDFs and 121 pages into 290 chunks, with zero skipped documents. The chunks matched the validated baseline byte for byte. Student 2 regenerated the same embeddings and reproduced the previous retrieval results. The 12 source-review flags carry forward.

## Scope and organization

Verification used separate Windows QA clones based on the published Student 1 commit `d2362531ecbfb60e5de96b65fa0650d611d928f0` and Student 2 commit `a58b0244150606d18ec1a8773b5b4267a3a7d3bb`. The original workspace supplied the input PDFs and baseline hashes. Generated files went into the QA clones. The rename and reference edits are unstaged and uncommitted.

| File in the proposed structure | Responsibility |
| --- | --- |
| `data/sources.csv`, `scripts/acquire_sources.py` | Official URLs, document IDs, checksums, source flags, and verified acquisition |
| `src/document_loader.py` | PDF and DOCX loading, extraction, and invalid or duplicate input detection |
| `src/clean_text.py` | Unicode NFC and whitespace normalization |
| `src/chunking.py` | Section-aware, overlapping chunks with source and page metadata |
| `src/rebuild_chunks.py`, `tests/test_document_processing.py` | Student 1 rebuild command and processing tests |

The loader moved from `src/extract_pdf.py` to `src/document_loader.py`. The chunker moved from `src/chunk_documents.py` to `src/chunking.py`. Both new filenames were already tracked as empty placeholders. The proposed trees contain one implementation of each module. The new loader matches the former file exactly. The new chunker differs from the former file only in its loader imports. Public functions, including `extract_pdf()`, remain available from their new modules.

I updated package imports and direct-script fallbacks in `src/chunking.py` and `src/rebuild_chunks.py`, the test imports, the README structure notes, and the completion report. Student 2's `src/rebuild_knowledge_base.py` has only the two chunker import paths changed. Its embedding and retrieval modules use the shared rebuild code and required no edits. I checked the file trees and references on all five branch snapshots. The application placeholder had no imports to update. AST checks and a filename/reference scan on both QA trees found no remaining operational imports or references to the retired module names.

## Responsibility checklist

| Responsibility | Result | Evidence |
| --- | --- | --- |
| Acquisition and provenance | PASS WITH LIMITATIONS | All 26 PDFs match the checked-in manifest and independently downloaded copies from the prior verification. Approval status, applicable editions and redistribution permission have separate review flags or notes. |
| Loading and extraction | PASS WITH LIMITATIONS | All PDFs and native text pages loaded successfully. Representative policy clauses, lists and a table row matched their PDF locations and order. Image-only annex material and complex layouts have coverage limits. |
| Cleaning and text preservation | PASS | NFC and whitespace normalization preserved all 32,936 extracted whitespace-delimited words in order after NFC. Cleaning was idempotent on all 121 pages. |
| Chunking and metadata | PASS | Verified source coverage, 264 overlaps, stable IDs, exact page ranges, section fields and MiniLM token counts. |
| Repeatability and Student 2 handoff | PASS | Both rebuilds produced the same chunk JSON. The combined rebuild matched the original embeddings and index manifest byte for byte, and both retrieval modes matched the six-question evaluation. |
| Automated verification | PASS | Student 1 suite 10/10, complete suite 23/23, and focused PDF and metadata checks passed. |

## Dataset and extraction findings

- The 26 source records use distinct document IDs, official AIU HTTPS URLs, local paths and raw SHA-256 hashes. Both the raw hashes and full normalized text were unique across the 26 files. Every file opened without encryption or PyMuPDF repair and supplied useful English policy text. All recorded page counts matched the actual PDFs.
- The acquisition command, `python -m scripts.acquire_sources`, validated all 26 local copies against the manifest. Every checksum also matched the earlier independently downloaded AIU snapshot. This establishes consistency with that snapshot. No new HTTP downloads or verification of current editions were needed for the rename audit.
- All 23 recorded printed dates appeared in the PDF text. Unverified publication, revision and effective-date fields remain blank. The manifest lists usage considerations and the 12 review flags. [Source manifest](../data/sources.csv)
- Representative extraction checks covered Attendance pages 1–2, Course Registration page 2, Transcript page 3 and Undergraduate Admission page 1. The checks confirmed values such as the 75% attendance requirement, main-semester credit limits, a printed date and ordered section text. The Transcript example preserved the selected label and value in order, with matching source row coordinates. The extractor returns plain text, so a table's grid and general visual reading order are outside the verified format.
- Temporary fixtures verified the distinct error for an encrypted PDF, the no-text response for empty and wholly image-only PDFs, and rejection of a damaged PDF. The valid fixture still processed alongside them. DOCX extraction, body order and headings were covered by the existing suite.

## Cleaning, chunking and metadata findings

The targeted cleaning fixture preserved decomposed accented text as NFC, bullets, headings, paragraph breaks, numbers, percentages, dates and punctuation while removing excess whitespace. Repeated cleaning returned identical text. A repeated long paragraph reconstructed exactly after removing known overlap, and a multi-page fixture retained both section labels.

The rebuild used the default 500-word cap and 75-word overlap. Actual MiniLM inputs were 127–256 tokens, including special tokens. All 264 adjacent same-document pairs overlapped by exactly 75 words. Independent source-span checks accounted for every extracted word and established exact page boundaries for each generated chunk.

All 290 records had unique IDs, original filenames, source paths, page ranges, text, and section-heading metadata. Each source path joins to precisely one manifest document ID. The chunk schema keeps that path as the source identifier, with the manifest's separate `document_id` available through the join. The singular `section_heading` field is null when a passage spans multiple headings, which are carried in `section_headings`.

## End-to-end and Student 2 compatibility

| Check | Result |
| --- | --- |
| `python -m src.rebuild_chunks` | 26 processed, 121 pages independently counted, 290 chunks, 0 skipped |
| Student 1 chunk JSON | Byte-for-byte baseline match, SHA-256 `f4b25ae5db983c2e22452186fe47bd797c0a94cf5757120968338f33fdcff098` |
| Student 2 rebuild | Accepted all 290 records and reproduced the chunk JSON, the `(290, 384)` `float32` embeddings, and the index manifest byte for byte |
| Student 2 retrieval | All six questions reproduced the recorded cosine-only and diversified rankings, scores, and continuations |
| Real query and citation | The 75% attendance rule was returned first from `4-Attendance-Policy.pdf`, PDF page 1, cosine score `0.5798` |

The rebuilds used the same Python 3.12 environment and cached CPU MiniLM model. The original PDFs and validated output files were preserved.

## Automated tests and limitations

Both suites ran via `python -m unittest discover -s tests -v`. Student 1 had **10 passed, 0 failed, 0 skipped**. The combined tree had **23 passed, 0 failed, 0 skipped**. Test changes were limited to imports. Encrypted, image-only and specific real-PDF behavior were also checked with temporary fixtures and corpus checks. The repository suite still has no dedicated encrypted or wholly image-only fixtures. The real-PDF checks cover representative layouts and all chunk source spans, with visual reading order checked on representative pages.

| Issue | Severity and handoff effect |
| --- | --- |
| Image-only forms or examples | Medium. Five PDFs contain such material across nine pages: Transcript p. 3, Withdrawal pp. 2–3, Attendance pp. 2–3, Course Registration pp. 4–6, and Incomplete Course p. 2. This limits form-field answers. The English policy-text handoff can proceed. |
| Course Registration deadline conflict | High for answers about the main-semester add deadline. The p. 2 body gives week one, while the forms on pp. 4–5 say add/drop through week two. AIU clarification is required before presenting a definitive deadline. This blocks that specific answer. |
| Four admissions edition flags | Medium. The Transfer, International Undergraduate, Undergraduate and Change Program policies call for semesterly updates. Confirm the applicable edition before presenting current admissions requirements. This limits live guidance. |
| Three editorial or template flags | Medium. Disability, Medical Service and Social Media contain editorial or placeholder text. Approval and contact details require confirmation before use as live guidance. |
| PDF layouts, headings and OCR | Low for core text handoff. The text index can flatten tables and omits image-only content. Visual heading coverage varies. Verify the PDF directly for layout-dependent answers. |
| Reuse and redistribution permission | High for redistribution. The local text handoff is usable. Repository or hosted distribution remains subject to permission for the underlying material. |

## Corrections and GitHub follow-up

The prepared corrections are the two filename moves and their import and documentation updates. This audit found no processing defect that blocks the text handoff. The outstanding source issues are recorded above and in the [completion report](student1_completion_report.md).

After approval, create a normal follow-up commit on `feature/data-processing` with the Student 1 changes. Merge that commit into `feature/retrieval` and include the two Student 2 import updates in the follow-up. The published Student 1 commit is an ancestor of Student 2. The README was the only shared file with divergent published edits, and a three-way simulation merged it cleanly. Fetch both tips and repeat ancestry and test checks at the time of integration. This plan preserves both published histories. The two feature branches are the only targets for the follow-up.
