# CAST-Net: Context-Aware Sparse-Transformer GAN for Image Inpainting

This repository is a clean, independent reference implementation of the CAST-Net architecture described in the accompanying manuscript. It contains the three named components:

- Sparse Global Context (SGC): learned top-k context selection with N x K attention.
- Cross-Scale Affinity Mixer (CSAM): texture retrieval restricted to observed source locations.
- Frequency-Aware Residual Refinement (FARR): mask-restricted residual refinement with frequency and edge losses.

## Research-integrity status

The code is implementation-ready but **does not validate any numerical result in the manuscript until the experiments are executed and the resulting artifacts are released**. Do not cite planning forecasts or author-supplied panels as reproduced findings merely because this repository exists.

## Mask convention

`valid_mask = 1` means observed/known pixel and `valid_mask = 0` means missing pixel. This convention is asserted throughout training, evaluation, inference, tests, and the manuscript equations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Verify the implementation

```bash
pytest -q
python scripts/smoke_experiment.py --output runs/smoke
```

The smoke run uses synthetic tensors. It verifies optimization, checkpoint reload, deterministic inference, and exact preservation of observed pixels; it is not a quality benchmark.

## Dataset manifests

Datasets are not redistributed. After accepting the relevant terms, create immutable split manifests:

```bash
python scripts/create_manifest.py --root /data/celeba_hq/train --output manifests/celeba_hq_train.jsonl
```

Repeat for validation and test splits. Keep identical test manifests and mask seeds for every baseline.

## Training

```bash
python train.py --config configs/celeba_hq.yaml
```

For reported results, create separate configuration files for at least three declared seeds. Each run writes a configuration snapshot, environment record, JSONL loss log, checkpoint, and hashes under `runs/<experiment>`.

## Evaluation

```bash
python evaluate.py \
  --config configs/celeba_hq.yaml \
  --checkpoint runs/celeba_hq_seed42/latest.pt \
  --output results/celeba_hq_seed42.jsonl
```

Evaluation writes per-image records and a summary containing the code/configuration/checkpoint provenance. LPIPS and FID should be added through pinned, documented metric packages before final benchmarking; the current evaluator implements hole PSNR and full-image SSIM.

## Inference

```bash
python inference.py --config configs/celeba_hq.yaml --checkpoint checkpoint.pt \
  --image input.jpg --valid-mask valid_mask.png --output completed.png
```

## Release requirement

The paper should link to an immutable Git tag or archived DOI, not only the moving default branch. Release exact configurations, split/mask manifests where licensing permits, per-image metrics, aggregate scripts, checkpoints with SHA-256 hashes, and the source identifiers for every paper figure.
