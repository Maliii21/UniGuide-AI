"""Retrieve the manifest's official, checksum-pinned PDF copies for local use."""

import csv
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_HOSTS = {"aiu.edu.eg", "www.aiu.edu.eg"}


def acquire_sources():
    with (ROOT / "data" / "sources.csv").open(encoding="utf-8", newline="") as manifest:
        rows = list(csv.DictReader(manifest))

    for row in rows:
        name, url, digest = (row[key] for key in ("original_filename", "official_source_url", "sha256"))
        target = ROOT / "data" / "raw" / name
        if target.parent != ROOT / "data" / "raw" or target.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF filename in the manifest: {name}")
        source = urlsplit(url)
        if source.scheme != "https" or source.hostname not in OFFICIAL_HOSTS:
            raise ValueError(f"Expected an official HTTPS source for {name}.")
        if target.exists():
            if sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Existing file differs from the manifest: {target}")
            continue
        with urlopen(url, timeout=30) as response:
            final = urlsplit(response.url)
            if final.scheme != "https" or final.hostname not in OFFICIAL_HOSTS:
                raise ValueError(f"The download left the official AIU site: {name}")
            contents = response.read()
        if not contents.startswith(b"%PDF-") or sha256(contents).hexdigest() != digest:
            raise ValueError(f"The official PDF differs from the manifest: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as destination:
            destination.write(contents)

    return len(rows)


if __name__ == "__main__":
    try:
        print(f"Verified {acquire_sources()} local AIU PDFs against the manifest.")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
