from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
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
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


@dataclass(frozen=True)
class TextPart:
    text: str
    page: int | None = None
    headings: tuple[str, ...] = ()


def discover_documents(folder=RAW_FOLDER):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_pdf(path):
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("PDF requires a password.")
        return [TextPart(page.get_text("text"), page.number + 1) for page in document]


def docx_paragraphs(container):
    """Yield body and table paragraphs in document order."""
    for item in container.iter_inner_content():
        if isinstance(item, Paragraph):
            yield item
        else:
            for row in item.rows:
                for cell in row.cells:
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


def main():
    documents, skipped = read_documents()
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    for path, parts in documents:
        output = PROCESSED_FOLDER / f"{path.name}.txt"
        output.write_text("\n\n".join(part.text for part in parts), encoding="utf-8")
        print(f"{path.name} -> {output.name}")
    for path, reason in skipped:
        print(f"Skipped {path.name}: {reason}")
    print(f"Extracted {len(documents)} document(s), skipped {len(skipped)}.")


if __name__ == "__main__":
    main()
