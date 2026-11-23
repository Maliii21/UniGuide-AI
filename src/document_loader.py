import csv
import json
import re
from pathlib import Path

import fitz

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
SOURCES_FILE = BASE_DIR / 'data' / 'sources.csv'
CORRECTIONS_FILE = BASE_DIR / 'data' / 'ocr_corrections.json'
OUTPUT_FILE = BASE_DIR / 'data' / 'processed' / 'documents.json'


def load_sources(sources_file=SOURCES_FILE):
    """Load document metadata indexed by filename."""
    sources = {}
    with open(sources_file, encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            filename = row['original_filename'].strip()
            if filename in sources:
                raise ValueError(f'Duplicate filename: {filename}')
            sources[filename] = row
    return sources


def clean_text(text):
    """Normalize whitespace while preserving paragraphs."""
    text = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def load_corrections(corrections_file=None):
    """Load page-level visual summaries and their review status."""
    corrections = {}
    if corrections_file is None:
        return corrections
    with open(corrections_file, encoding='utf-8-sig') as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError('Corrections must be a JSON list.')
    for record in records:
        page = record['page']
        if type(page) is not int or page < 1:
            raise ValueError(f'Invalid correction page: {page}')
        key = (record['document_id'], page)
        if key in corrections:
            raise ValueError(f'Duplicate correction: {key}')
        if record['review_status'] not in {'verified', 'pending_human_review'}:
            raise ValueError(f'Invalid correction review status: {key}')
        if not isinstance(record['text'], str) or not clean_text(record['text']):
            raise ValueError(f'Empty or invalid correction: {key}')
        corrections[key] = record
    return corrections


def load_documents(folder_path=RAW_DIR, sources_file=SOURCES_FILE, corrections_file=None):
    """Extract native PDF text and append only verified visual summaries."""
    documents = []
    sources = load_sources(sources_file)
    corrections = load_corrections(corrections_file)
    pdf_files = sorted(Path(folder_path).glob('*.pdf'))
    if not pdf_files:
        raise FileNotFoundError('No PDF files found.')
    actual = {pdf.name for pdf in pdf_files}
    expected = set(sources)
    if actual != expected:
        raise ValueError(f'Missing PDFs: {expected - actual}\nUnregistered PDFs: {actual - expected}')
    matched = set()
    for pdf_path in pdf_files:
        metadata = sources[pdf_path.name]
        with fitz.open(pdf_path) as pdf:
            if len(pdf) != int(metadata['page_count']):
                raise ValueError(f'Page count mismatch: {pdf_path.name}')
            for page_number, page in enumerate(pdf, start=1):
                native_text = clean_text(page.get_text('text'))
                text = native_text
                extraction_method = 'native_text'
                image_review_status = 'not_required'
                key = (metadata['document_id'], page_number)
                correction = corrections.get(key)
                if correction is not None:
                    matched.add(key)
                    image_review_status = correction['review_status']
                    if image_review_status == 'verified':
                        text = clean_text(native_text + '\n\n[Reviewed image content]\n' + correction['text'])
                        extraction_method = 'native_text_plus_visual_summary' if native_text else 'visual_summary'
                    else:
                        print(f'WARNING: Unverified correction skipped: {key}')
                if not text:
                    print(f'WARNING: Empty page {page_number} in {pdf_path.name}')
                    continue
                documents.append({
                    'document_id': metadata['document_id'],
                    'page_id': f"{metadata['document_id']}-P{page_number:03d}",
                    'document_title': metadata['document_title'],
                    'source': pdf_path.name,
                    'source_url': metadata['official_source_url'],
                    'category': metadata['document_category'],
                    'language': metadata['language'],
                    'document_date': metadata['document_date_as_printed'],
                    'document_date_type': metadata['document_date_type'],
                    'page': page_number,
                    'review_status': metadata['review_status'],
                    'review_notes': metadata['review_notes'],
                    'version_notes': metadata['version_notes'],
                    'extraction_method': extraction_method,
                    'image_review_status': image_review_status,
                    'text': text,
                })
            print(f'Loaded: {pdf_path.name} ({len(pdf)} pages)')
    unmatched = set(corrections) - matched
    if unmatched:
        raise ValueError(f'Corrections without matching pages: {unmatched}')
    return documents


def save_documents(documents, output_path=OUTPUT_FILE):
    """Save extracted pages as UTF-8 JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(documents, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    documents = load_documents(corrections_file=CORRECTIONS_FILE)
    save_documents(documents)
    total_documents = len({doc['document_id'] for doc in documents})
    total_words = sum(len(doc['text'].split()) for doc in documents)
    enhanced_pages = sum(doc['extraction_method'] in {'native_text_plus_visual_summary', 'visual_summary'} for doc in documents)
    pending_pages = sum(doc['image_review_status'] == 'pending_human_review' for doc in documents)
    print('\n' + '=' * 50)
    print('DOCUMENT LOADING COMPLETED')
    print('=' * 50)
    print(f'Total documents: {total_documents}')
    print(f'Total extracted pages: {len(documents)}')
    print(f'Total extracted words: {total_words}')
    print(f'Enhanced pages: {enhanced_pages}')
    print(f'Pending review pages: {pending_pages}')
    print(f'Output saved to: {OUTPUT_FILE}')
