import argparse
import json
from pathlib import Path

import torch

from castnet.losses import generator_objective
from castnet.model import CASTNet, CASTNetConfig
from castnet.reproducibility import seed_everything, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(123)
    model = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4, retain_ratio=0.25))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target = torch.rand(2, 3, 64, 64) * 2 - 1
    valid = torch.ones(2, 1, 64, 64)
    valid[:, :, 16:48, 16:48] = 0
    corrupted = target * valid
    before = model(corrupted, valid)["completed"].detach()
    losses = generator_objective(model(corrupted, valid), target, valid)
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    checkpoint = output / "smoke.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    clone = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4, retain_ratio=0.25))
    clone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    clone.eval()
    with torch.no_grad():
        reloaded = clone(corrupted, valid)["completed"]
        expected = model.eval()(corrupted, valid)["completed"]
    max_error = float((reloaded - expected).abs().max())
    record = {"status": "pass" if max_error < 1e-7 else "fail", "loss": float(losses["total"].detach()), "reload_max_abs_error": max_error, "checkpoint_sha256": sha256(checkpoint), "known_pixels_preserved": bool(torch.allclose(reloaded * valid, target * valid, atol=1e-6))}
    write_json(output / "smoke_record.json", record)
    print(json.dumps(record, indent=2))
    if record["status"] != "pass" or not record["known_pixels_preserved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
