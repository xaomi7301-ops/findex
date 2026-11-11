"""Download official Python documentation as a genuine plain-text corpus."""

import argparse
import shutil
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://docs.python.org/3.13/archives/python-3.13-docs-text.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Python 3.13 docs text corpus")
    parser.add_argument("--root", type=Path, default=Path("data/python_docs"))
    args = parser.parse_args()
    archive = Path("data/_downloads/python-3.13-docs-text.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        print(f"Downloading {URL}")
        try:
            with urlopen(URL, timeout=60) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
        except Exception:
            archive.unlink(missing_ok=True)
            raise

    extracted = 0
    with ZipFile(archive) as package:
        for member in package.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or path.suffix.lower() != ".txt":
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                print(f"Skipping unsafe filename {member.filename!r}")
                continue
            segments = list(path.parts)
            if len(segments) > 1 and segments[0].startswith("python-"):
                segments = segments[1:]
            output = args.root.joinpath(*segments)
            output.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1

    print(f"Extracted {extracted} genuine documentation files into {args.root}")


if __name__ == "__main__":
    main()
