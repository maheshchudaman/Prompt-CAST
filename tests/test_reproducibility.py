import torch

from castnet.model import CASTNet, CASTNetConfig
from castnet.reproducibility import seed_everything


def test_seeded_initialization_is_repeatable():
    seed_everything(7)
    first = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4))
    seed_everything(7)
    second = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4))
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
