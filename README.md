## Data Collection and Processing

The UniGuide-AI data processing pipeline prepares official Alamein International University (AIU) policy documents for a Retrieval-Augmented Generation (RAG) system supporting Arabic and English queries.

### Dataset Overview

| Metric | Value |
|---|---:|
| Official PDF documents | 26 |
| Extracted pages | 121 |
| Extracted words | 33,599 |
| Pages enhanced with visual summaries | 9 |
| Generated chunks | 202 |
| Maximum chunk size | 400 tokens |
| Minimum chunk size | 34 tokens |
| Average chunk size | 264.77 tokens |
| Target chunk overlap | 60 tokens |
| Passed tests | 27/27 |

### Data Processing Pipeline

1. **Data Collection:** Collect official university policy documents from the AIU website.
2. **Data Validation:** Check PDF readability, duplicate files, and text availability.
3. **Source Inventory:** Record document metadata, source URLs, page counts, verification statuses, and review notes.
4. **Text Extraction:** Extract page-level text using PyMuPDF.
5. **Text Cleaning:** Normalize whitespace while preserving the original content and paragraph structure.
6. **Image Content Processing:** Supplement native text extraction with reviewed visual summaries where necessary.
7. **Document Chunking:** Split extracted text into overlapping, token-based chunks.
8. **Metadata Preservation:** Retain document identifiers, page numbers, source URLs, and review information.
9. **Quality Assurance:** Validate the processing pipeline through automated tests.

### OCR Experiments

Basic and enhanced OCR approaches were investigated using PyMuPDF and Tesseract OCR.

The experiments included:

- Arabic and English text recognition.
- High-resolution image rendering.
- Grayscale conversion.
- Contrast enhancement and sharpening.
- OCR configuration adjustments.

Recognition errors were observed in complex forms, tables, and Arabic image-based content. Therefore, raw OCR output was not approved for ingestion.

A hybrid approach was adopted, combining native PDF text extraction with visual form summaries marked as verified in the correction inventory.

Nine pages across five documents were enhanced using this approach.

### Chunking Strategy

The system implements page-aware, boundary-aware token chunking with overlapping content.

| Parameter | Configuration |
|---|---|
| Tokenizer | intfloat/multilingual-e5-base |
| Model maximum sequence length | 512 tokens |
| Maximum chunk size | 400 tokens |
| Target overlap | 60 tokens |
| Overlap percentage | 15% |
| Page boundary preservation | Enabled |
| Metadata preservation | Enabled |

The tokenizer is used to calculate token lengths accurately for both Arabic and English content. Natural sentence boundaries are preferred where possible, while character offsets preserve the original extracted text.

The selected chunk size leaves sufficient space for the embedding model's input prefix and special tokens.

### Generated Data Files

| File | Description |
|---|---|
| `data/sources.csv` | Source inventory and document metadata |
| `data/ocr_corrections.json` | Page-level visual summaries and review statuses |
| `data/processed/documents.json` | Extracted page-level text and metadata |
| `data/processed/chunks.json` | Processed chunks ready for embedding |

### Reproducing the Data Processing Pipeline

Run the following commands from the project root:

```bash
python Validate_data.py
python src/document_loader.py
python src/chunking.py
python src/update_source_inventory.py
```

### Testing and Validation

Run the data processing tests using:

```bash
python -m pytest tests/test_document_loader.py tests/test_chunking.py -v
```

Test results:

| Test Suite | Passed |
|---|---:|
| Document Loader | 11/11 |
| Chunking | 16/16 |
| Total | 27/27 |

The tests cover text cleaning, metadata preservation, unique identifiers, page boundaries, overlapping content, Arabic text preservation, text coverage, JSON serialization, and embedding input length.

### Limitations

- OCR accuracy is affected by complex layouts, tables, and scanned Arabic forms.
- Some university policies contain outdated editions or conflicting information requiring institutional confirmation.
- Page-aware chunking may produce shorter chunks for pages with limited content.
- Retrieval effectiveness must be evaluated after embedding and indexing.
- Public accessibility of source documents does not automatically establish redistribution permission.