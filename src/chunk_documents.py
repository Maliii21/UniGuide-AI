from pathlib import Path
import re

if __package__:
    from .clean_text import clean_text
    from .extract_pdf import PROJECT_ROOT, TextPart
else:
    from clean_text import clean_text
    from extract_pdf import PROJECT_ROOT, TextPart

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

# PyMuPDF can put a Roman numeral on its own line above a heading.
PDF_HEADING = re.compile(r"(?m)^[IVXLCDM]+\.[ \t\n]+[A-Z][A-Z &/-]+(?: \([^\n]+\))?$")
BLOCK_START = re.compile(r"(?m)\n\n(?=\S)|^(?=(?:[-•–]|\d+[.)-])(?: |$))")


def chunk_parts(parts, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, fits=None):
    """Group paragraphs and list items, keeping page and heading offsets."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller.")

    texts = []
    page_starts = {}
    heading_spans = {}
    offset = 0
    for part in parts:
        text = clean_text(part.text)
        if not text:
            continue
        texts.append(text)
        page_starts[offset] = part.page
        matches = list(PDF_HEADING.finditer(text)) if part.page is not None else []
        for heading in map(clean_text, part.headings):
            if heading:
                matches.extend(re.finditer(rf"(?m)^{re.escape(heading)}$", text))
        for match in matches:
            heading_spans[offset + match.start()] = (
                offset + match.end(), " ".join(match.group().split())
            )
        offset += len(text) + 2

    text = "\n\n".join(texts)
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    starts = {word.start(): index for index, word in enumerate(words)}
    block_offsets = set(page_starts) | set(heading_spans)
    block_offsets.update(match.end() for match in BLOCK_START.finditer(text))
    # Keep a heading with the first block of its section when selecting a boundary.
    for start, (end, _) in heading_spans.items():
        following = re.search(r"\S", text[end:])
        next_block = end + following.start() if following else end
        block_offsets -= {point for point in block_offsets if start < point <= next_block
                          and point not in heading_spans}
    boundaries = sorted(starts[point] for point in block_offsets if point in starts)
    page_indexes = [(starts[point], page) for point, page in page_starts.items()]
    heading_indexes = sorted((starts[point], title) for point, (_, title) in heading_spans.items())

    def passage(start, end):
        return text[words[start].start():words[end - 1].end()]

    chunks = []
    start = previous_end = 0
    while previous_end < len(words):
        end = min(start + chunk_size, len(words))
        if fits and not fits(passage(start, end)):
            # Find the largest prefix that fits the model without truncation.
            low, high = start, end
            while low + 1 < high:
                middle = (low + high) // 2
                if fits(passage(start, middle)):
                    low = middle
                else:
                    high = middle
            end = low
        if end <= previous_end:
            if start == previous_end:
                raise ValueError("A single word exceeds the embedding model's input limit.")
            # The overlap filled the model window. Retry from the next new word.
            start = previous_end
            continue
        if end < len(words):
            # Prefer nearby paragraph and list boundaries without wasting the window.
            earliest = max(previous_end + 1, end - (end - start) // 3)
            end = max((point for point in boundaries if earliest <= point <= end), default=end)

        pages = [
            page for index, (point, page) in enumerate(page_indexes)
            if page is not None and point < end
            and (index + 1 == len(page_indexes) or page_indexes[index + 1][0] > start)
        ]
        active_heading = next((title for point, title in reversed(heading_indexes) if point <= start), None)
        headings = ([active_heading] if active_heading else []) + [
            title for point, title in heading_indexes if start < point < end
        ]
        chunks.append({
            "text": passage(start, end),
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "section_heading": headings[0] if len(headings) == 1 else None,
            "section_headings": headings,
        })
        previous_end = end
        start = max(start + 1, end - overlap)
    return chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    return [chunk["text"] for chunk in chunk_parts([TextPart(text)], chunk_size, overlap)]


def chunk_document(path, parts, *, fits=None, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    path = Path(path).absolute()
    source_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    return [
        {
            "chunk_id": f"{source_path.as_posix()}#{index}",
            "source": path.name,
            "source_path": source_path.as_posix(),
            **chunk,
        }
        for index, chunk in enumerate(chunk_parts(parts, chunk_size=chunk_size, overlap=overlap, fits=fits), start=1)
    ]


def main():
    if __package__:
        from .rebuild_chunks import main as rebuild
    else:
        from rebuild_chunks import main as rebuild

    return rebuild()


if __name__ == "__main__":
    raise SystemExit(main())
