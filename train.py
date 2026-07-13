import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from castnet.config import load_yaml, model_config
from castnet.data import ManifestDataset
from castnet.discriminator import PatchDiscriminator, discriminator_hinge, generator_hinge
from castnet.losses import generator_objective
from castnet.model import CASTNet
from castnet.reproducibility import environment_record, git_revision, seed_everything, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / config["experiment"]
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config.yaml"
    snapshot.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    write_json(run_dir / "run_metadata.json", {"git_commit": git_revision(), "config_sha256": sha256(snapshot), "environment": environment_record(), "seed": config["seed"], "status": "started"})

    dataset = ManifestDataset(config["data"]["train_manifest"], config["data"]["image_size"], config["data"]["mask_seed"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    model = CASTNet(model_config(config)).to(device)
    discriminator = PatchDiscriminator(config["model"]["base_channels"]).to(device)
    optimizer_g = torch.optim.Adam(model.parameters(), config["training"]["learning_rate"], betas=(config["training"]["beta1"], config["training"]["beta2"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), config["training"]["learning_rate"], betas=(config["training"]["beta1"], config["training"]["beta2"]))
    step = 0
    log_path = run_dir / "metrics.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(config["training"]["epochs"]):
            for batch in loader:
                target = batch["target"].to(device)
                corrupted = batch["corrupted"].to(device)
                valid = batch["valid_mask"].to(device)
                output = model(corrupted, valid)
                optimizer_d.zero_grad(set_to_none=True)
                loss_d = discriminator_hinge(discriminator(target, valid), discriminator(output["completed"].detach(), valid))
                loss_d.backward()
                optimizer_d.step()
                optimizer_g.zero_grad(set_to_none=True)
                losses = generator_objective(output, target, valid, config["training"]["lambda_frequency"], config["training"]["lambda_edge"])
                adversarial = generator_hinge(discriminator(output["completed"], valid))
                loss_g = losses["total"] + 0.1 * adversarial
                loss_g.backward()
                optimizer_g.step()
                step += 1
                record = {"epoch": epoch, "step": step, "loss_g": float(loss_g.detach()), "loss_d": float(loss_d.detach()), **{key: float(value.detach()) for key, value in losses.items()}}
                log.write(json.dumps(record) + "\n")
                log.flush()
                if args.max_steps and step >= args.max_steps:
                    break
            checkpoint = run_dir / "latest.pt"
            torch.save({"model": model.state_dict(), "discriminator": discriminator.state_dict(), "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(), "epoch": epoch, "step": step, "config": config}, checkpoint)
            if args.max_steps and step >= args.max_steps:
                break
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update({"status": "complete", "steps": step, "checkpoint_sha256": sha256(checkpoint)})
    write_json(run_dir / "run_metadata.json", metadata)


if __name__ == "__main__":
    main()
