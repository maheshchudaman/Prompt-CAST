# Prompt-CAST: Text-Conditioned Image Inpainting with a Frozen Vision-Language Prior

Prompt-CAST extends [CAST-Net](https://github.com/maheshchudaman/CAST-Net) with optional text-guided inpainting: a short prompt can steer what the Cross-Scale Affinity Mixer (CSAM) retrieves/synthesizes in the masked region, via a FiLM-conditioned query modulation driven by a frozen CLIP text encoder. Sparse Global Context (SGC) and Frequency-Aware Residual Refinement (FARR) are unchanged from CAST-Net.

- **Sparse Global Context (SGC)**: learned top-k context selection with N x K attention. Unchanged from CAST-Net.
- **Cross-Scale Affinity Mixer (CSAM)**: texture retrieval restricted to observed source locations, now with an optional `text_dim` parameter enabling FiLM-based prompt conditioning of the hole queries before the affinity computation.
- **Frequency-Aware Residual Refinement (FARR)**: mask-restricted residual refinement with frequency and edge losses. Unchanged from CAST-Net.

## Research-integrity status

**The text-conditioning path is architecture only.** It is implemented and unit-tested, and the full pipeline (ADE20K manifest with scene-category prompts -> dataset -> frozen CLIP encoder -> text-conditioned forward pass) has been verified end-to-end against a synthetic fixture. **No training has been run, no checkpoint exists, and no quality metric (PSNR/SSIM/FID/CLIP-score) has been measured.** Do not cite this repository as evidence that text-conditioned inpainting works well, or at all, until real training and evaluation exist and are released.

The base CAST-Net path (`text_dim=None`, the default) carries CAST-Net's own research-integrity status: implementation-ready, not yet validating the manuscript's numerical results until those experiments are executed and released.

## Mask convention

`valid_mask = 1` means observed/known pixel and `valid_mask = 0` means missing pixel, throughout training, evaluation, inference, tests, and equations - identical to CAST-Net.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# only if you want text-conditioned inpainting (frozen CLIP text encoder):
pip install open_clip_torch
```

## Verify the implementation

```bash
pytest -q
python scripts/smoke_experiment.py --output runs/smoke
python scripts/smoke_experiment_text.py --output runs/smoke_text
```

Both smoke runs use synthetic tensors and a synthetic (non-CLIP) prompt embedding. They verify optimization, checkpoint reload, deterministic inference, and exact preservation of observed pixels - including through the FiLM text-conditioning path. Neither is a quality benchmark.

## Dataset manifests

Datasets are not redistributed. After accepting the relevant terms:

```bash
# unconditional manifest (any dataset)
python scripts/create_manifest.py --root /data/celeba_hq/train --output manifests/celeba_hq_train.jsonl

# text-conditioned manifest for ADE20K, using its own sceneCategories.txt as the prompt source
python scripts/create_ade20k_text_manifest.py \
  --root /data/ade20k/images \
  --scene-categories /data/ade20k/sceneCategories.txt \
  --output manifests/ade20k_train_text.jsonl
```

Keep identical test manifests and mask seeds for every baseline.

## Training

```bash
python train.py --config configs/celeba_hq.yaml
```

Text-conditioned training requires a manifest built with `create_ade20k_text_manifest.py` (or any manifest whose records carry a `"prompt"` field) and a config with `model.text_dim` set to the CLIP text encoder's output dimension (512 for `ViT-B-32-quickgelu`). A dedicated `configs/ade20k_text.yaml` and training-script wiring for this path have not been added yet - only the model, data, and text-encoder building blocks have been built and verified so far.

For reported results, create separate configuration files for at least three declared seeds. Each run writes a configuration snapshot, environment record, JSONL loss log, checkpoint, and hashes under `runs/<experiment>`.

## Evaluation

```bash
python evaluate.py \
  --config configs/celeba_hq.yaml \
  --checkpoint runs/celeba_hq_seed42/latest.pt \
  --output results/celeba_hq_seed42.jsonl
```

Evaluation writes per-image records and a summary containing the code/configuration/checkpoint provenance. LPIPS and FID should be added through pinned, documented metric packages before final benchmarking; the current evaluator implements hole PSNR and full-image SSIM. A CLIP-score metric (does the completed region match its prompt) has not been added yet.

## Inference

```bash
python inference.py --config configs/celeba_hq.yaml --checkpoint checkpoint.pt \
  --image input.jpg --valid-mask valid_mask.png --output completed.png
```

## Relationship to CAST-Net

This repository started from the CAST-Net implementation ([github.com/maheshchudaman/CAST-Net](https://github.com/maheshchudaman/CAST-Net), commit `02bdbda6b9c74c94e9dea30439567f3e0dd6c300`) and adds the text-conditioning path on top without modifying SGC or FARR. If you only need the base (unconditional) architecture, use CAST-Net directly.

## Release requirement

The paper should link to an immutable Git tag or archived DOI, not only the moving default branch. Release exact configurations, split/mask manifests where licensing permits, per-image metrics, aggregate scripts, checkpoints with SHA-256 hashes, and the source identifiers for every paper figure.
