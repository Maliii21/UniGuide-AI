# UniGuide-AI
AIU Student Assistant powered by RAG

UniGuide AI uses official Alamein International University policy documents as its source corpus. This branch contains the source inventory and the document-processing pipeline. The [Student 1 report](docs/student1_completion_report.md) describes the audit and the 12 source-review flags.

## Set up and rebuild on Windows

Run these commands from the repository root. They use Python 3.12.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m scripts.acquire_sources
& .\.venv\Scripts\python.exe -m src.rebuild_chunks
```

On Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` for the remaining commands. Acquisition reads official HTTPS URLs from [data/sources.csv](data/sources.csv) and verifies each PDF's SHA-256 before saving it. It stops on a checksum mismatch and preserves existing copies. The first chunk build needs the CPU MiniLM model; later runs can use its local cache with `HF_HUB_OFFLINE=1`.

The loader is in `src/document_loader.py`, text cleanup is in `src/clean_text.py`, and chunking is in `src/chunking.py`.

Original PDFs and verbatim derived text stay local while redistribution permission is unverified. The canonical output is `data/processed/document_chunks.json`. It is ignored by Git and can be reproduced from the source manifest. Chunk IDs retain the original filename, PDF page range, and recognizable section headings. The default is 500 words with 75 words of overlap before the MiniLM token cap. Use `--chunk-size` and `--overlap` when intentionally changing the index.

Student 2's retrieval branch builds on this branch and contains the embedding and search commands.

## Inspect extracted text locally

From the repository root, run the optional export after acquiring the source documents.

```powershell
& .\.venv\Scripts\python.exe -m scripts.export_documents
(Get-Content -Raw -Encoding UTF8 data/processed/documents.json | ConvertFrom-Json) | Select-Object -First 1
```

`data/processed/documents.json` contains the original filename, project-relative source path, page number, and extracted text before cleaning or chunking. Records are sorted by source path and numeric page, with `null` for DOCX pages. The export is local and ignored by Git. The normal rebuild still writes `document_chunks.json` for Student 2.

## Run Student 1 tests

From the repository root, install the test requirements and run the complete suite.

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python.exe -m pytest tests/ -v
```

Run a single file by replacing `tests/` with its path.

| File | Coverage |
| --- | --- |
| `tests/test_document_loader.py` | Format validation, repaired PDFs, merged DOCX cells, duplicate text, and an empty source folder |
| `tests/test_clean_text.py` | Unicode normalization, whitespace, lists, and idempotence |
| `tests/test_chunking.py` | Overlap, token limits, source coverage, IDs, and page provenance |
| `tests/test_document_processing.py` | Extraction-to-chunk metadata, mixed input, the rebuild CLI, and the optional text export |

The tests also support `python -m unittest discover -s tests -v`.
