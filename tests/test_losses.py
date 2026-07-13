import torch

from castnet.losses import edge_charbonnier, frequency_l1, hole_l1


def test_losses_are_finite():
    prediction = torch.zeros(1, 3, 16, 16)
    target = torch.ones_like(prediction)
    valid = torch.ones(1, 1, 16, 16)
    valid[:, :, 4:12, 4:12] = 0
    assert torch.isfinite(hole_l1(prediction, target, valid))
    assert torch.isfinite(edge_charbonnier(prediction, target, valid))
    assert torch.isfinite(frequency_l1(prediction, target))
