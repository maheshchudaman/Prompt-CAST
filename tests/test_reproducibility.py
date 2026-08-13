import torch

from castnet.model import CASTNet, CASTNetConfig
from castnet.reproducibility import environment_record, seed_everything


def test_seeded_initialization_is_repeatable():
    seed_everything(7)
    first = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4))
    seed_everything(7)
    second = CASTNet(CASTNetConfig(base_channels=16, attention_heads=4))
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))


def test_environment_record_gpu_field_is_never_falsely_cpu():
    # On any machine with CUDA or MPS available, the recorded "gpu" field
    # must reflect that accelerator, not silently fall back to "cpu" - a
    # bug that previously made MPS (Apple Silicon) runs misreport their
    # hardware in run_metadata.json.
    record = environment_record()
    if torch.cuda.is_available():
        assert record["gpu"] == torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        assert record["gpu"] != "cpu"
        assert "MPS" in record["gpu"]
    else:
        assert record["gpu"] == "cpu"
