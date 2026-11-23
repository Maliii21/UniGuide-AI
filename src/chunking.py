from bisect import bisect_right
from itertools import groupby
import re

import tiktoken

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
ENCODING_NAME = "cl100k_base"


def chunk_document(source, pages, encoding):
    """Split one document into overlapping token windows with starting-page citations."""
    pages = [page for page in pages if page["text"]]
    if not pages:
        return []

    page_offsets = []
    offset = 0
    for page in pages:
        page_offsets.append(offset)
        offset += len(page["text"].encode("utf-8")) + 2

    text = "\n\n".join(page["text"] for page in pages)
    data = text.encode("utf-8")
    tokens = encoding.encode_ordinary(text)
    offsets = [0]
    for token_bytes in encoding.decode_tokens_bytes(tokens):
        offsets.append(offsets[-1] + len(token_bytes))

    # A byte-pair token can end inside a Unicode character.
    utf8_boundaries = {
        index for index, position in enumerate(offsets)
        if position == len(data) or (data[position] & 0xC0) != 0x80
    }
    paragraph_offsets = {match.end() for match in re.finditer(rb"\n\n", data)}
    chunks = []
    start = 0
    while start < len(tokens):
        limit = min(start + CHUNK_SIZE, len(tokens))
        if limit == len(tokens):
            candidates = [limit]
        else:
            # Prefer nearby paragraph breaks, falling back to a safe token boundary.
            candidates = sorted(
                range(start + CHUNK_OVERLAP + 1, limit + 1),
                key=lambda end: (end >= limit - CHUNK_SIZE // 3 and offsets[end] in paragraph_offsets, end),
                reverse=True,
            )
        for end in candidates:
            if end not in utf8_boundaries or (end < len(tokens) and end - CHUNK_OVERLAP not in utf8_boundaries):
                continue
            passage = data[offsets[start]:offsets[end]].decode("utf-8")
            if encoding.encode_ordinary(passage) == tokens[start:end]:
                break
        else:
            raise ValueError(f"Cannot split {source} at a UTF-8 boundary with the requested overlap.")

        first_text_offset = offsets[end] - len(passage.lstrip().encode("utf-8"))
        page = pages[bisect_right(page_offsets, first_text_offset) - 1]["page"]
        chunks.append({
            "chunk_id": f"{source}#p{page}_c{len(chunks) + 1}",
            "source": source,
            "page": page,
            "text": passage,
        })
        if end == len(tokens):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def chunk_documents(records):
    """Chunk page records in document order, without overlapping across documents."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return [
        chunk
        for source, pages in groupby(records, key=lambda record: record["source"])
        for chunk in chunk_document(source, pages, encoding)
    ]


def main():
    if __package__:
        from .rebuild_chunks import main as rebuild
    else:
        from rebuild_chunks import main as rebuild
    return rebuild()


if __name__ == "__main__":
    raise SystemExit(main())
