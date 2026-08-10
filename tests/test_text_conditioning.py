import torch

from castnet.model import CASTNet, CASTNetConfig


def small_model(text_dim=None):
    return CASTNet(CASTNetConfig(base_channels=16, attention_heads=4, retain_ratio=0.25, text_dim=text_dim))


def test_unconditional_model_rejects_text_embedding():
    model = small_model(text_dim=None)
    target = torch.rand(1, 3, 64, 64) * 2 - 1
    valid = torch.ones(1, 1, 64, 64)
    valid[:, :, 12:52, 12:52] = 0
    prompt = torch.randn(1, 512)
    try:
        model(target * valid, valid, text_embedding=prompt)
        assert False, "expected ValueError when text_dim=None but text_embedding is passed"
    except ValueError:
        pass


def test_zero_init_film_matches_unconditional_output():
    # Compared within a single model instance, not across two separately
    # constructed models: constructing an extra nn.Linear (even one that is
    # immediately zeroed) consumes RNG draws and shifts the random init of
    # every later layer, so two "same seed" models are not actually
    # weight-identical once their parameter counts differ. Isolating the
    # comparison to one instance removes that confound and tests only what
    # this claim is actually about: does zero-init FiLM act as identity.
    model = small_model(text_dim=512)
    target = torch.rand(1, 3, 64, 64) * 2 - 1
    valid = torch.ones(1, 1, 64, 64)
    valid[:, :, 12:52, 12:52] = 0
    prompt = torch.randn(1, 512)

    model.eval()
    with torch.no_grad():
        out_with_prompt = model(target * valid, valid, text_embedding=prompt)["completed"]
        out_no_prompt = model(target * valid, valid, text_embedding=None)["completed"]

    assert torch.allclose(out_with_prompt, out_no_prompt, atol=1e-6)


def test_gradients_reach_film_head():
    model = small_model(text_dim=512)
    target = torch.rand(1, 3, 64, 64) * 2 - 1
    valid = torch.ones(1, 1, 64, 64)
    valid[:, :, 12:52, 12:52] = 0
    prompt = torch.randn(1, 512, requires_grad=False)
    model(target * valid, valid, text_embedding=prompt)["completed"].mean().backward()
    assert model.csam.film.weight.grad is not None


def test_known_pixel_preservation_with_text_conditioning():
    model = small_model(text_dim=512)
    target = torch.rand(1, 3, 64, 64) * 2 - 1
    valid = torch.ones(1, 1, 64, 64)
    valid[:, :, 20:44, 16:48] = 0
    prompt = torch.randn(1, 512)
    output = model(target * valid, valid, text_embedding=prompt)
    assert torch.allclose(output["completed"] * valid, target * valid, atol=1e-6)
