# UniGuide-AI
AIU Student Assistant powered by RAG

UniGuide AI uses official Alamein International University policy documents as its source corpus. This branch contains the source inventory and the document-processing, embedding, and retrieval pipelines. The [Student 1 report](docs/student1_completion_report.md) describes the audit and the 12 source-review flags.

## Set up and rebuild on Windows

Run these commands from the repository root. They use Python 3.12.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m scripts.acquire_sources
& .\.venv\Scripts\python.exe -m src.rebuild_knowledge_base
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` for the remaining commands. Acquisition reads official HTTPS URLs from [data/sources.csv](data/sources.csv) and verifies each PDF's SHA-256 before saving it. It stops on a checksum mismatch and preserves existing copies. The first chunk build needs the CPU MiniLM model; later runs can use its local cache with `HF_HUB_OFFLINE=1`.

The loader is in `src/document_loader.py`, text cleanup is in `src/clean_text.py`, and chunking is in `src/chunking.py`.

Original PDFs and verbatim derived text stay local while redistribution permission is unverified. The generated index files are ignored by Git and can be reproduced from the source manifest. Chunk IDs retain the original filename, PDF page range, and recognizable section headings. The default is 500 words with 75 words of overlap before the MiniLM token cap. Use `--chunk-size` and `--overlap` when intentionally changing the index.

The full rebuild creates the processed JSON, vectors, and integrity manifest together. It uses the same processing functions as `src.rebuild_chunks`. Use the full rebuild whenever inputs or chunking settings change.

## Retrieval

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m src.semantic_search --top-k 3 --question "What is the minimum attendance percentage required at AIU?"
```

Student 3 can import `load_model` from `src.generate_embeddings` and `load_knowledge_base` and `semantic_search` from `src.semantic_search`. Load the model and index once. Each row carries passage text, a cosine score, source path, PDF page range, available section headings and optional continuation passages. Match `source_path` with `local_file_path` in the source manifest for official URLs. Use `diversify=False` or `--cosine-only` for the original rank order.

The [six-question retrieval evaluation](evaluation/retrieval_comparison.txt) records rankings, scores and observed evidence gaps. The single-corpus benchmark uses 26 PDFs, 290 passages and 290 normalized 384-dimensional vectors.

## Team integration

Review the data-processing branch first, then the retrieval branch, which includes the processing commit in its ancestry. Preserve the existing UI and evaluation branch files when integrating. Once the LLM and Streamlit entry point are implemented, the owner can choose the reviewed integration branch and `app.py` in Streamlit Community Cloud and set Python 3.12. Add the UI's dependencies to `requirements.txt` when that code is integrated. A Git checkout starts without the index. The deployment needs a reproducible build or provisioned index using data authorized for that environment. Keep API credentials in deployment secrets.
