# Student 1 completion report

**Verdict: PASS WITH LIMITATION.** The verified text corpus meets the assignment's minimum of 20 documents and is ready for the team handoff. Confirmation is still needed for certain policy editions, the conflicting registration deadline, and reuse or redistribution permission. The image-only forms remain outside the text index.

## Scope and recovery

The course assignment PDF (AIE417, Assignment 2, p. 1) requires at least 20 relevant documents and, for the team submission, document chunking, embeddings, NumPy cosine similarity, Top-K retrieval, a pretrained LLM, a working interface, and at least 10 questions comparing answers with and without RAG. This audit applies the Student 1 responsibilities in the project brief. The six-question retrieval evaluation supports the handoff. The team-level chatbot evaluation belongs to a later stage.

The workspace had no Git history for this path, and no separate copy was found in the project or the checked Documents and Downloads folders. This report restores the original path using the previous handoff record and checks the results against current PDFs, source records, and index artifacts.

## Assignment requirements and evidence

| Student 1 requirement | Result | Verified evidence |
| --- | --- | --- |
| Collect at least 20 relevant, distinct official AIU sources | PASS | 26 official policy PDFs in `data/raw/`, all usable for the English policy-text corpus. |
| Acquire publicly accessible documents and record official provenance | PASS | All 26 recorded URLs appeared in AIU's policy index and returned the exact bytes of the local files. |
| Record source identity, title, URL, language, date and usage details | PASS WITH LIMITATION | All 26 records include document identity and usage notes. Twenty-three carry a printed creation, development, or unspecified cover date. Actual publication/revision and effective date/year fields are blank where unverified. |
| Establish permission for reuse or redistribution | NOT VERIFIED | AIU publicly links the PDFs. The audit located no reuse licence for these copies in the checked catalogue or PDF text. Public access establishes provenance. |
| Deduplicate the corpus and detect duplicate text | PASS | 26 distinct byte hashes and distinct normalized full texts. The loader now reports and skips later documents with identical normalized text. |
| Inspect document versions | PASS WITH LIMITATION | The source copies match current catalogue downloads. Four admissions covers call for semesterly updates, and three documents have editorial or template content. Applicable editions or approval dates remain unverified. |
| Detect scanned, empty, corrupt, inaccessible and problematic content | PASS WITH LIMITATION | All 121 pages have native text, and current files opened without repair or passwords. Five PDFs contain image-only material. Automated fixtures cover empty and invalid files, and the source-review table records visual limitations. |
| Extract supported source formats as applicable | PASS WITH LIMITATION | All current sources are PDFs. Production supports PDF and DOCX, and tests exercise both. HTML was used for discovery. There are no raw HTML or TXT source documents in this corpus. |
| Preserve document boundaries, page references, and available headings | PASS | All 290 chunks match spans from one PDF, with independently verified page ranges, unique IDs, chapter/section labels present in the source text, and matching provenance records. |
| Normalize Unicode, whitespace and line breaks while preserving policy text | PASS | NFC, whitespace cleanup and idempotence verified across the corpus. All 32,936 source tokens defined by whitespace splitting were retained in order after NFC. |
| Manually inspect images and use OCR where essential | PASS WITH LIMITATION | Forms and examples were reviewed. Search covers the main English procedures. Exact form content and conflicting form instructions have review limits documented below. |
| Support configurable chunk size and overlap within MiniLM limits | PASS | Defaults are 500 words and 75 words of overlap, now configurable through the rebuild CLI. All saved chunks fit the model's 256-token limit. |
| Prefer section and paragraph boundaries | PASS | The chunker prefers headings, paragraphs and list boundaries when the word and token budgets permit. Long blocks can be split and overlap preserves continuity. |
| Deliver reproducible processed chunks and usable Student 2 input | PASS | Two default in-memory runs matched the saved 290-chunk JSON byte for byte. The saved index passed its hash and ordering checks. Both retrieval modes returned the expected Attendance result. |
| Maintain representative extraction, cleaning, chunking and traceability tests | PASS | 10 document-processing tests passed. The full project suite passed 23 tests. |
| Deliver raw files, source manifest, acquisition report and processed data | PASS | The artifacts are present, cross-referenced and documented below. |

Source catalogue and data inventory. [AIU Policies and Procedures](https://aiu.edu.eg/aiu-policies-and-procedures/) · [Source manifest](../data/sources.csv)

## Acquisition and inventory

**26 accepted for text-layer processing, 0 rejected, 12 flagged.** The flags include five documents with image-only material across nine pages, four admissions editions, and three editorial or template issues. The registration discrepancy is part of an existing image-content flag.

Every local SHA-256 matched both the manifest and the currently linked official PDF. All IDs, filenames, URLs, source paths, page counts and chunk counts join consistently with the index. The manifest now records usage considerations without asserting a licence. The `2024/07` URL directory is a hosting path. Use the printed date and its stated type when available.

All 26 file hashes and normalized texts were unique. The highest similarity among 325 pairs was 0.5402 by Jaccard similarity over normalized five-word sequences. The domestic and international undergraduate policies cover different populations and eligibility provisions and are separate documents.

| ID | Original filename | PDF pages | Chunks | Review status |
| --- | --- | ---: | ---: | --- |
| AIU-001 | 1-Admission-Transfer-Undergrad-Policy.pdf | 3 | 6 | Edition review |
| AIU-002 | 1-AIU-Outcomes-Assessment-in-Academic-Programs-and-Non-Academic-Units-Policy.pdf | 6 | 14 | No specific flag |
| AIU-003 | 10-Enrollment-Policy.pdf | 4 | 13 | No specific flag |
| AIU-004 | 10-Volunteering-Policy-at-AIU.pdf | 12 | 48 | No specific flag |
| AIU-005 | 11-AIU-Students-Mental-Health-and-Well-being-Policy.pdf | 6 | 15 | No specific flag |
| AIU-006 | 11-Student-identification-card-Policy.pdf | 2 | 5 | No specific flag |
| AIU-007 | 12-Student-Record-Privacy-retention-Policy.pdf | 3 | 8 | No specific flag |
| AIU-008 | 13-Transcript-Policy.pdf | 4 | 8 | Image-only content |
| AIU-009 | 14-Tuition-and-Fees-Policy.pdf | 3 | 8 | No specific flag |
| AIU-010 | 15-Withdraw-from-AIU-Policy.pdf | 4 | 6 | Image-only content |
| AIU-011 | 16-Academic-calendar-Policy.pdf | 3 | 5 | No specific flag |
| AIU-012 | 17-Dress-Code-Policy.pdf | 3 | 5 | No specific flag |
| AIU-013 | 18-International-Admission-Undergrad-Policy.pdf | 10 | 17 | Edition review |
| AIU-014 | 2-Admission-Undergrad-Policy.pdf | 13 | 19 | Edition review |
| AIU-015 | 2.3-Email-Communication-Policy.pdf | 4 | 12 | No specific flag |
| AIU-016 | 2.5-Internet-Use-Policy.pdf | 5 | 15 | No specific flag |
| AIU-017 | 3-Class-Schedule-Policy.pdf | 2 | 6 | No specific flag |
| AIU-018 | 4-Attendance-Policy.pdf | 3 | 5 | Image-only content |
| AIU-019 | 4-Disability-policy.pdf | 8 | 35 | Editorial review |
| AIU-020 | 5-Course-Registration-Policy.pdf | 7 | 8 | Image-only content |
| AIU-021 | 6-Incomplete-Policy.pdf | 3 | 4 | Image-only content |
| AIU-022 | 6-Medical-service-policy.pdf | 3 | 7 | Editorial review |
| AIU-023 | 7-Admission-Change-program-Policy.pdf | 2 | 4 | Edition review |
| AIU-024 | 8-Appeals-Complaints-and-Grevances-Policy.pdf | 3 | 5 | No specific flag |
| AIU-025 | 9-Course-Audit-Policy.pdf | 3 | 6 | No specific flag |
| AIU-026 | 9-Social-media-policy.pdf | 2 | 6 | Editorial review |
| **Total** | **26 PDFs** | **121** | **290** | **12 flagged** |

## Source review and impact on answers

Each item below retains usable English policy text and therefore leaves the 20-document minimum satisfied. A matching official download establishes which copy was used. Version or editorial flags limit what the chatbot can present as current or approved.

| Source and PDF pages | Finding | Effect on chatbot answers |
| --- | --- | --- |
| Transcript, p. 3 | The sample transcript and key are images. The issuance procedure and description of the key are searchable. | Basic request and issuance questions are supported. Exact sample fields and layout require the PDF. |
| Withdrawal, pp. 2–3 | Arabic withdrawal-related forms are images. The required forms and Registrar submission procedure are searchable. | The chatbot can cover the general submission process. Individual form fields and instructions need visual confirmation. |
| Attendance, pp. 2–3 | The absence request and attendance-sheet examples are embedded images. The attendance rules, supporting-document requirement and review responsibilities are searchable. | Core attendance and excuse questions are supported. Form-field questions require the original. |
| Course Registration, pp. 2, 4–6 | Image-only registration, add/drop and withdrawal forms. The body on p. 2 gives the first week for main-semester additions and the end of the second week for eligible drops. The forms on pp. 4–5 state that add/drop runs through the end of the second week. | General registration and credit-hour rules are searchable. An authoritative answer about the current add deadline requires AIU clarification of the conflict. |
| Incomplete Course, p. 2 | The incomplete-course form is an image. The application steps and document requirement are searchable. | Process questions are supported. Form fields are outside retrieved text. |
| Transfer Admission, p. 1 | The cover calls for semesterly updates. Applicable edition unverified. | Current transfer eligibility, conditions and deadlines require edition confirmation. |
| International Undergraduate Admission, p. 1 | The cover calls for semesterly updates. Applicable edition unverified. | Current international eligibility and document requirements require edition confirmation. |
| Undergraduate Admission, p. 1 | The cover calls for semesterly updates. Applicable edition unverified. | Current intake requirements and cutoffs require edition confirmation. |
| Change Program, p. 1 | The cover calls for semesterly updates. Applicable edition unverified. | Current program-transfer rules require edition confirmation. |
| Disability, pp. 2 and 7 | The official copy contains an inline editorial comment on internal email and placeholder contact guidance. Approval status unverified. | The chatbot should avoid representing editorial text as an approved procedure or inventing a contact. |
| Medical Service, p. 2 | The contact section contains template guidance. Approval status unverified. | The clinic overview is searchable. Specific contact details require a confirmed source. |
| Social Media, pp. 1–2 | The cover and contact section include ownership or contact placeholders. Approval status unverified. | The chatbot should avoid inferring a policy owner or contact from template text. |

OCR is unnecessary for the core English policy-text handoff because the substantive procedures above are searchable. Exact form fields would need verified transcription or OCR if they enter scope. OCR alone would preserve the conflicting registration statements, so AIU clarification is still needed.

## Processing results and corrections

- Added canonical Unicode NFC normalization to `src/clean_text.py`. The regression test checks a decomposed accent along with line breaks, bullets and ordinary policy punctuation.
- Added duplicate extracted-text detection in `src/document_loader.py`. It compares case-preserving text after NFC and whitespace normalization, keeps the first file in sorted order and reports subsequent matches by filename. Documents that differ only in images still need review of those images.
- Exposed `--chunk-size` and `--overlap` through `src/rebuild_chunks.py` and passed them through `src/chunking.py`. A regression test verifies that both options change the generated passages. Default settings are unchanged.
- All 290 chunks traced to contiguous source words, with no gaps across documents' extracted text. All 264 adjacent within-document pairs overlapped by 75 words. MiniLM inputs ranged from 127 to 256 tokens including special tokens. Sparse captions on form pages still have valid one-based page references.
- Two full read/extract/chunk passes matched the stored JSON byte for byte. The cached CPU MiniLM model regenerated the `(290, 384)` embedding array exactly in memory. The saved index and all original PDFs retained their SHA-256 checksums.
- **23 tests passed, 0 failed, 0 skipped** via `python -m unittest discover -s tests -v`. The full suite covers PDF/DOCX extraction, empty and invalid documents, Unicode, duplicates, configurable chunking, page and section metadata, index integrity and retrieval.

## Handoff to Student 2 and the next students

The `data/processed/document_chunks.json` remains the source of passage text and metadata. Keep its row order aligned with the locally generated vectors and index manifest. Join `source_path` to `local_file_path` in [sources.csv](../data/sources.csv) to find the official URL, and use `page_start` and `page_end` for citations. The real CPU model returned `4-Attendance-Policy.pdf`, page 1, as rank one for the minimum-attendance question in both ranking modes. Its score was `0.5798`, and the passage contained the 75% requirement.

The complete local pipeline produced the verification results above. This branch ships the processing stage. See the README to acquire and rebuild the text locally.

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -m src.rebuild_chunks
```

Student 2's retrieval branch supplies the combined embedding rebuild and semantic search. Both builds use the same tokenizer and chunk defaults.

The text corpus and retrieval are ready for Students 3 and 4 to continue development. The final team submission still requires the pretrained LLM, a working chatbot interface, and the assignment's 10-question comparison with and without RAG. The source-review limitations above carry forward.
