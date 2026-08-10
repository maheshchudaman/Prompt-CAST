import argparse
import json
from pathlib import Path

import torch

from castnet.losses import generator_objective
from castnet.model import CASTNet, CASTNetConfig
from castnet.reproducibility import seed_everything, sha256, write_json


def main():
    """Synthetic smoke test for the text-conditioned CSAM path.

    Mirrors scripts/smoke_experiment.py exactly, but builds the model with
    text_dim set and drives training with a synthetic prompt embedding
    (a fixed random vector) rather than a real CLIP encoder output, since
    CLIP itself is frozen and not part of what this smoke test needs to
    verify. This checks that the training loop (forward, loss, backward,
    optimizer step, checkpoint save/reload) works correctly through the
    new FiLM head - it says nothing about inpainting quality, since there
    is no real image data or trained CLIP text encoder involved.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/smoke_text")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(123)
    text_dim = 512
    model = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4, retain_ratio=0.25, text_dim=text_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target = torch.rand(2, 3, 64, 64) * 2 - 1
    valid = torch.ones(2, 1, 64, 64)
    valid[:, :, 16:48, 16:48] = 0
    corrupted = target * valid
    prompt_embedding = torch.randn(2, text_dim)  # stand-in for a frozen CLIP text embedding

    before = model(corrupted, valid, text_embedding=prompt_embedding)["completed"].detach()
    outputs = model(corrupted, valid, text_embedding=prompt_embedding)
    losses = generator_objective(outputs, target, valid)
    optimizer.zero_grad()
    losses["total"].backward()
    film_grad_before_step = model.csam.film.weight.grad.detach().clone()
    optimizer.step()

    checkpoint = output / "smoke_text.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    clone = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4, retain_ratio=0.25, text_dim=text_dim))
    clone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    clone.eval()
    with torch.no_grad():
        reloaded = clone(corrupted, valid, text_embedding=prompt_embedding)["completed"]
        expected = model.eval()(corrupted, valid, text_embedding=prompt_embedding)["completed"]
    max_error = float((reloaded - expected).abs().max())

    record = {
        "status": "pass" if max_error < 1e-7 else "fail",
        "loss": float(losses["total"].detach()),
        "reload_max_abs_error": max_error,
        "checkpoint_sha256": sha256(checkpoint),
        "known_pixels_preserved": bool(torch.allclose(reloaded * valid, target * valid, atol=1e-6)),
        "film_head_received_nonzero_gradient": bool(film_grad_before_step.abs().max() > 0),
        "output_changed_after_one_step": bool((reloaded - before).abs().max() > 1e-8),
        "note": "synthetic prompt embedding (random vector), not a real CLIP output; verifies the training loop and FiLM path mechanically, not inpainting quality",
    }
    write_json(output / "smoke_text_record.json", record)
    print(json.dumps(record, indent=2))
    if record["status"] != "pass" or not record["known_pixels_preserved"] or not record["film_head_received_nonzero_gradient"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
