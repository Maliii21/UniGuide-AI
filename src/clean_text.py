
from pathlib import Path
import re
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"


def clean_text(text):
    text = unicodedata.normalize("NFC", text)

    # Normalize Windows and Unix line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unnecessary spaces while preserving line breaks
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove spaces at the beginning and end of lines
    lines = [line.strip() for line in text.split("\n")]

    # Remove excessive consecutive blank lines
    cleaned_lines = []
    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()


def main():
    for path in sorted(PROCESSED_FOLDER.glob("*.txt")):
        if path.stem.lower().endswith("_cleaned"):
            continue
        original = path.read_text(encoding="utf-8")
        cleaned = clean_text(original)
        output = PROCESSED_FOLDER / f"{path.stem}_cleaned.txt"
        output.write_text(cleaned, encoding="utf-8")
        print(f"{path.name}: {len(original)} -> {len(cleaned)} characters ({output.name})")


if __name__ == "__main__":
    main()
