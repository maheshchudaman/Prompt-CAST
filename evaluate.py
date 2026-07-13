import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from castnet.config import load_yaml, model_config
from castnet.data import ManifestDataset
from castnet.metrics import global_ssim, masked_psnr
from castnet.model import CASTNet
from castnet.reproducibility import environment_record, git_revision, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CASTNet(model_config(config)).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    dataset = ManifestDataset(config["data"]["test_manifest"], config["data"]["image_size"], config["data"]["mask_seed"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    records = []
    with torch.no_grad():
        for batch in loader:
            target, corrupted, valid = batch["target"].to(device), batch["corrupted"].to(device), batch["valid_mask"].to(device)
            completed = model(corrupted, valid)["completed"]
            records.append({"image_id": batch["image_id"][0], "psnr_hole": float(masked_psnr(completed, target, valid)), "ssim_full": float(global_ssim(completed, target))})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    summary = {"count": len(records), "mean_psnr_hole": sum(r["psnr_hole"] for r in records) / max(len(records), 1), "mean_ssim_full": sum(r["ssim_full"] for r in records) / max(len(records), 1), "checkpoint_sha256": sha256(args.checkpoint), "config_sha256": sha256(args.config), "git_commit": git_revision(), "environment": environment_record()}
    write_json(output.with_suffix(".summary.json"), summary)


if __name__ == "__main__":
    main()
