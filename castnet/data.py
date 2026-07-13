import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def free_form_validity_mask(size, seed, min_strokes=4, max_strokes=12):
    rng = random.Random(seed)
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(min_strokes, max_strokes)):
        points = []
        x, y = rng.randrange(size), rng.randrange(size)
        for _ in range(rng.randint(2, 6)):
            x = max(0, min(size - 1, x + rng.randint(-size // 3, size // 3)))
            y = max(0, min(size - 1, y + rng.randint(-size // 3, size // 3)))
            points.append((x, y))
        draw.line(points, fill=0, width=rng.randint(max(2, size // 40), max(3, size // 10)))
    return TF.pil_to_tensor(image).float() / 255.0


class ManifestDataset(Dataset):
    def __init__(self, manifest, image_size=256, mask_seed=2026):
        self.records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image_size = image_size
        self.mask_seed = mask_seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        target = TF.pil_to_tensor(image).float() / 127.5 - 1.0
        valid = free_form_validity_mask(self.image_size, self.mask_seed + index)
        corrupted = target * valid
        return {"target": target, "corrupted": corrupted, "valid_mask": valid, "image_id": record.get("id", str(index))}
