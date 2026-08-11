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


def load_scene_categories(path):
    """Parses ADE20K's sceneCategories.txt: one 'image_stem category_name' pair per line."""
    categories = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed sceneCategories.txt line: {line!r}")
        stem, category = parts
        categories[stem] = category.replace("_", " ")
    return categories


def main():
    parser = argparse.ArgumentParser(description="Build a text-conditioned manifest for ADE20K, using scene category as the prompt.")
    parser.add_argument("--root", required=True, help="Directory containing ADE20K images")
    parser.add_argument("--scene-categories", required=True, help="Path to ADE20K's sceneCategories.txt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    categories = load_scene_categories(args.scene_categories)
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, []
    with output.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(files):
            category = categories.get(path.stem)
            if category is None:
                skipped.append(path.stem)
                continue
            record = {
                "id": f"{index:08d}",
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "sha256": file_hash(path),
                "prompt": category,
            }
            handle.write(json.dumps(record) + "\n")
            written += 1

    print(f"wrote {written} records to {output}")
    if skipped:
        print(f"skipped {len(skipped)} image(s) with no matching scene-category entry, e.g. {skipped[:5]}")


if __name__ == "__main__":
    main()
