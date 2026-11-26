
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evaluation" / "dataset_manifest.json"


def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


def main():
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    errors = []

    print("=" * 60)
    print("UNIGUIDE-AI DATASET INTEGRITY VERIFICATION")
    print("=" * 60)

    for relative_path, expected_hash in manifest["file_hashes"].items():
        path = ROOT / relative_path

        if not path.is_file():
            errors.append(f"Missing file: {relative_path}")
            print(f"[FAIL] {relative_path}")
            continue

        actual_hash = calculate_sha256(path)

        if actual_hash.lower() != expected_hash.lower():
            errors.append(f"Hash mismatch: {relative_path}")
            print(f"[FAIL] {relative_path}")
        else:
            print(f"[PASS] {relative_path}")

    print("\n" + "=" * 60)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")

        print("DATASET INTEGRITY VERIFICATION FAILED")
        return 1

    print("DATASET INTEGRITY VERIFIED")
    print("All frozen files match the recorded SHA-256 hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
