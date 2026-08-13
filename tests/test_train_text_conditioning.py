import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image


def build_fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    names_to_categories = {f"img_{i:03d}": "bedroom" if i % 2 == 0 else "street" for i in range(6)}
    for name in names_to_categories:
        Image.new("RGB", (48, 48), color=(100, 120, 140)).save(images / f"{name}.jpg")
    scene_file = tmp_path / "sceneCategories.txt"
    scene_file.write_text("\n".join(f"{name} {cat}" for name, cat in names_to_categories.items()) + "\n")
    return images, scene_file


def test_train_py_runs_end_to_end_with_text_conditioning(tmp_path):
    pytest.importorskip("open_clip", reason="open_clip_torch is an optional dependency")

    repo_root = Path(__file__).resolve().parents[1]
    images, scene_file = build_fixture(tmp_path)

    manifest_path = tmp_path / "manifest.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "create_ade20k_text_manifest.py"),
            "--root", str(images),
            "--scene-categories", str(scene_file),
            "--output", str(manifest_path),
        ],
        check=True,
        cwd=repo_root,
    )

    config = {
        "seed": 123,
        "experiment": f"pytest_text_train_{tmp_path.name}",
        "data": {"train_manifest": str(manifest_path), "image_size": 32, "mask_seed": 2026},
        "model": {"base_channels": 16, "attention_heads": 4, "retain_ratio": 0.25, "csam_temperature": 0.07, "residual_scale": 0.25, "text_dim": 512},
        "training": {"epochs": 1, "batch_size": 3, "learning_rate": 0.0002, "beta1": 0.5, "beta2": 0.999, "lambda_frequency": 0.1, "lambda_edge": 0.1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(repo_root / "train.py"), "--config", str(config_path), "--max-steps", "2"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"train.py failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    run_dir = repo_root / "runs" / config["experiment"]
    metrics_path = run_dir / "metrics.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "latest.pt"

    assert metrics_path.exists()
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # --max-steps 2
    first_record = json.loads(lines[0])
    assert "loss_g" in first_record and "loss_d" in first_record

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["device"] in ("cuda", "mps", "cpu")

    assert checkpoint_path.exists()

    # cleanup: this test writes into the real runs/ dir like any other experiment would
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)
