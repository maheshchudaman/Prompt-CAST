# Reproducibility protocol

## Required record for every result

- repository release tag and Git commit
- configuration file and SHA-256 hash
- dataset version, licence, split manifest, and manifest hash
- fixed mask-bank algorithm, seed, identifiers, and area-bin statistics
- Python, PyTorch, CUDA, cuDNN, GPU, and resolved package versions
- random seed and deterministic-algorithm status
- complete training/validation logs
- selected-checkpoint rule and checkpoint SHA-256 hash
- exact evaluation command and raw per-image metrics
- baseline repository, commit, weights, preprocessing, and failure count

## Experimental sequence

1. Run `pytest -q` in a clean environment.
2. Run the deterministic synthetic smoke experiment.
3. Freeze dataset split manifests and the test mask bank.
4. Train each claimed dataset with at least three seeds.
5. Evaluate every method on identical images, masks, resolution, and compositing.
6. Compute mean, standard deviation, and paired bootstrap 95% confidence intervals.
7. Generate manuscript tables and figures from archived outputs, never by manual transcription.
8. Have two authors compare the generated tables against aggregate files.
9. Create an immutable release and archive it with a DOI.

## Figure provenance table

Before submission create `results/figure_source_ids.csv` with:

```text
figure,panel,dataset,image_id,mask_id,method,run_id,checkpoint_sha256,output_path
```

Do not label a panel as experimental if any of these fields cannot be recovered.

## Current status

- implementation and unit tests: available
- synthetic deterministic smoke record: generated during verification
- trained CAST-Net checkpoints: pending
- dataset manifests and fixed mask bank: pending author-controlled datasets
- repeated-seed benchmark results: pending
- paper table/figure provenance: pending

This status must be updated honestly with each release.
