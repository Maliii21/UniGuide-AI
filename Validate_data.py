# python -m pip install --no-cache-dir "PyMuPDF==1.24.14"
# python -c "import fitz; print('Installation successful!')"
# python validate_data.py

from pathlib import Path
import hashlib
import fitz

data_dir = Path("data/raw")
pdf_files = sorted(data_dir.glob("*.pdf"))

print(f"Total PDF files: {len(pdf_files)}")

hashes = {}

for pdf_path in pdf_files:
    file_hash = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()

    if file_hash in hashes:
        print(f"DUPLICATE: {pdf_path.name}")
        print(f"  Same as: {hashes[file_hash]}")
    else:
        hashes[file_hash] = pdf_path.name

    try:
        with fitz.open(pdf_path) as doc:
            empty_pages = [
                i + 1
                for i, page in enumerate(doc)
                if not page.get_text().strip()
            ]

            print(
                f"{pdf_path.name}: "
                f"{len(doc)} pages, "
                f"{len(empty_pages)} empty-text pages"
            )

    except Exception as e:
        print(f"ERROR reading {pdf_path.name}: {e}")


