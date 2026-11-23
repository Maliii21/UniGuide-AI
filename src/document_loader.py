from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import pymupdf
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError

if __package__:
    from .clean_text import clean_text
else:
    from clean_text import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
DOCUMENTS_PATH = PROCESSED_FOLDER / "documents.json"
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


@dataclass(frozen=True)
class TextPart:
    text: str
    page: int | None = None
    headings: tuple[str, ...] = ()


def discover_documents(folder=RAW_FOLDER):
    return sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.name,
    )


def extract_pdf(path):
    with pymupdf.open(path) as document:
        if not document.is_pdf:
            raise ValueError("File content is not a PDF.")
        if document.needs_pass:
            raise ValueError("PDF requires a password.")
        if document.is_repaired:
            raise ValueError("PDF required structural repair. Obtain an intact copy.")
        return [TextPart(page.get_text("text"), page.number + 1) for page in document]


def docx_paragraphs(container):
    """Yield body and table paragraphs in document order."""
    for item in container.iter_inner_content():
        if isinstance(item, Paragraph):
            yield item
        else:
            seen_cells = set()
            for row in item.rows:
                for cell in row.cells:
                    # Merged grid positions share the same underlying cell, even across rows.
                    cell_xml = cell._tc
                    if cell_xml in seen_cells:
                        continue
                    seen_cells.add(cell_xml)
                    yield from docx_paragraphs(cell)


def extract_docx(path):
    paragraphs = list(docx_paragraphs(Document(path)))
    headings = tuple(
        paragraph.text for paragraph in paragraphs
        if paragraph.style.name.startswith("Heading ")
    )
    return [TextPart("\n\n".join(paragraph.text for paragraph in paragraphs), headings=headings)]


def read_documents(folder=RAW_FOLDER):
    """Return (documents, skipped), keeping per-file errors out of the rest of the build."""
    documents = []
    skipped = []
    indexed_texts = {}
    for path in discover_documents(folder):
        try:
            extractor = extract_pdf if path.suffix.lower() == ".pdf" else extract_docx
            parts = extractor(path)
            text = " ".join(clean_text(" ".join(part.text for part in parts)).split())
            if not text:
                raise ValueError("No extractable text.")
            fingerprint = sha256(text.encode("utf-8")).digest()
            if fingerprint in indexed_texts:
                raise ValueError(f"Duplicate extracted text of {indexed_texts[fingerprint].name}.")
        except (OSError, ValueError, RuntimeError, BadZipFile, KeyError,
                PackageNotFoundError, XMLSyntaxError) as error:
            skipped.append((path, str(error)))
        else:
            indexed_texts[fingerprint] = path
            documents.append((path, parts))
    return documents, skipped


def document_records(documents):
    return [
        {"source": path.name, "page": part.page, "text": clean_text(part.text)}
        for path, parts in documents
        for part in parts
    ]


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                            dir=path.parent, delete=False) as staged:
        staged_path = Path(staged.name)
        try:
            json.dump(records, staged, ensure_ascii=False, indent=2)
            staged.write("\n")
            staged.close()
            os.replace(staged_path, path)
        finally:
            staged.close()
            staged_path.unlink(missing_ok=True)


def main():
    documents, skipped = read_documents()
    for path, reason in skipped:
        print(f"Skipped {path.name}: {reason}")
    records = document_records(documents)
    write_json(DOCUMENTS_PATH, records)
    print(f"Exported {len(records)} page record(s) from {len(documents)} document(s).")


if __name__ == "__main__":
    main()
