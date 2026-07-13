import torch

from castnet.model import CASTNet, CASTNetConfig


def small_model():
    return CASTNet(CASTNetConfig(base_channels=16, attention_heads=4, retain_ratio=0.25))


def test_output_shapes_and_known_pixel_preservation():
    model = small_model()
    target = torch.rand(2, 3, 64, 64) * 2 - 1
    valid = torch.ones(2, 1, 64, 64)
    valid[:, :, 20:44, 16:48] = 0
    output = model(target * valid, valid, return_diagnostics=True)
    assert output["completed"].shape == target.shape
    assert torch.allclose(output["completed"] * valid, target * valid, atol=1e-6)
    assert output["diagnostics"]["sgc"]["indices"].shape[-1] == 64


def test_all_valid_mask_reproduces_input():
    model = small_model().eval()
    target = torch.rand(1, 3, 64, 64) * 2 - 1
    valid = torch.ones(1, 1, 64, 64)
    with torch.no_grad():
        completed = model(target, valid)["completed"]
    assert torch.equal(completed, target)


def test_gradients_reach_token_scorer():
    model = small_model()
    target = torch.rand(1, 3, 64, 64) * 2 - 1
    valid = torch.ones(1, 1, 64, 64)
    valid[:, :, 12:52, 12:52] = 0
    model(target * valid, valid)["completed"].mean().backward()
    assert model.sgc.scorer.weight.grad is not None
