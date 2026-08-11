import json

import pytest
import torch
from PIL import Image

from castnet.data import ManifestDataset
from scripts.create_ade20k_text_manifest import load_scene_categories


def build_fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    names_to_categories = {
        "ADE_train_00000001": "airport_terminal",
        "ADE_train_00000002": "bedroom",
        "ADE_train_00000003": "street",
    }
    for name in names_to_categories:
        Image.new("RGB", (48, 48), color=(100, 120, 140)).save(images / f"{name}.jpg")

    scene_file = tmp_path / "sceneCategories.txt"
    scene_file.write_text("\n".join(f"{name} {category}" for name, category in names_to_categories.items()) + "\n")
    return images, scene_file


def test_load_scene_categories_parses_and_normalizes_underscores(tmp_path):
    _, scene_file = build_fixture(tmp_path)
    categories = load_scene_categories(scene_file)
    assert categories == {
        "ADE_train_00000001": "airport terminal",
        "ADE_train_00000002": "bedroom",
        "ADE_train_00000003": "street",
    }


def test_manifest_dataset_returns_prompt_when_required(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (48, 48), color=(10, 20, 30)).save(image_path)
    manifest_path.write_text(json.dumps({"id": "0", "path": str(image_path), "prompt": "bedroom"}) + "\n")

    ds = ManifestDataset(str(manifest_path), image_size=32, require_prompt=True)
    sample = ds[0]
    assert sample["prompt"] == "bedroom"
    assert sample["target"].shape == (3, 32, 32)


def test_manifest_dataset_rejects_missing_prompts(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (48, 48), color=(10, 20, 30)).save(image_path)
    manifest_path.write_text(json.dumps({"id": "0", "path": str(image_path)}) + "\n")  # no "prompt" field

    with pytest.raises(ValueError, match="no prompt"):
        ManifestDataset(str(manifest_path), image_size=32, require_prompt=True)


def test_default_manifest_dataset_unaffected_by_prompt_support(tmp_path):
    # require_prompt defaults to False - existing manifests without a "prompt"
    # field must keep working exactly as before this feature was added.
    manifest_path = tmp_path / "manifest.jsonl"
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (48, 48), color=(10, 20, 30)).save(image_path)
    manifest_path.write_text(json.dumps({"id": "0", "path": str(image_path)}) + "\n")

    ds = ManifestDataset(str(manifest_path), image_size=32)
    sample = ds[0]
    assert "prompt" not in sample


def test_full_pipeline_manifest_to_text_conditioned_forward_pass(tmp_path):
    open_clip = pytest.importorskip("open_clip", reason="open_clip_torch is an optional dependency")
    from torch.utils.data import DataLoader

    from castnet.model import CASTNet, CASTNetConfig
    from castnet.text_encoder import FrozenTextEncoder

    images, scene_file = build_fixture(tmp_path)
    from scripts.create_ade20k_text_manifest import main as build_manifest_main
    import sys

    manifest_path = tmp_path / "manifest.jsonl"
    argv = sys.argv
    sys.argv = ["create_ade20k_text_manifest.py", "--root", str(images), "--scene-categories", str(scene_file), "--output", str(manifest_path)]
    try:
        build_manifest_main()
    finally:
        sys.argv = argv

    ds = ManifestDataset(str(manifest_path), image_size=32, require_prompt=True)
    loader = DataLoader(ds, batch_size=len(ds))
    batch = next(iter(loader))

    encoder = FrozenTextEncoder()
    text_embedding = encoder(batch["prompt"])

    model = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4, retain_ratio=0.25, text_dim=encoder.output_dim))
    output = model(batch["corrupted"], batch["valid_mask"], text_embedding=text_embedding)

    assert output["completed"].shape == batch["target"].shape
    assert torch.allclose(output["completed"] * batch["valid_mask"], batch["target"] * batch["valid_mask"], atol=1e-6)
