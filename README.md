# UniGuide-AI
AIU Student Assistant powered by RAG

UniGuide AI uses official Alamein International University policy documents as its source corpus. This branch contains the source inventory and the document-processing pipeline. The [Student 1 report](docs/student1_completion_report.md) describes the audit and the 12 source-review flags.

## Set up, preprocess, and test

Run these commands from the repository root. They use Python 3.12.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m scripts.acquire_sources
& .\.venv\Scripts\python.exe -m src.rebuild_chunks
& .\.venv\Scripts\python.exe -m pytest tests/ -v
```

On Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` for the remaining commands. Place the 26 original PDFs listed in [data/sources.csv](data/sources.csv) in `data/raw/`. `python -m scripts.acquire_sources` downloads missing PDFs from the official AIU HTTPS URLs and checks all copies against the manifest's SHA-256 values. The manifest starts with `filename`, `title`, `source_url`, `year`, and `category`. Unverified years are blank.

The rebuild uses PyMuPDF and produces two UTF-8 JSON files.

- `data/processed/documents.json` contains `source`, one-based `page`, and cleaned `text`.
- `data/processed/chunks.json` contains `chunk_id`, `source`, `page`, and `text`. If a chunk spans pages, `page` identifies where its text starts.

The chunker uses `tiktoken` with the `cl100k_base` encoding. It prefers paragraph boundaries, caps chunks at 500 tokens, and overlaps adjacent chunks of the same document by 75 tokens. The first run may download the tokenizer vocabulary.

Run `python -m pytest tests/test_document_loader.py -v` or `python -m pytest tests/test_chunking.py -v` for the dedicated suites. Git ignores new raw files and generated outputs. Use the local rebuild to provide `chunks.json` for Student 2. The repository already contains the 26 source PDFs.

For raw page inspection, `python -m scripts.export_documents` writes `data/processed/documents_raw.json` with `source_path` and uncleaned text. The rebuild writes the assignment outputs independently.
