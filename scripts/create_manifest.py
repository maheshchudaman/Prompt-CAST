import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(files):
            handle.write(json.dumps({"id": f"{index:08d}", "path": str(path), "relative_path": str(path.relative_to(root)), "sha256": file_hash(path)}) + "\n")
    print(f"wrote {len(files)} records to {output}")


if __name__ == "__main__":
    main()
