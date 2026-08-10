import torch
from torch import nn


class FrozenTextEncoder(nn.Module):
    """Wraps a pretrained CLIP text tower, frozen, for prompt embeddings.

    Requires open_clip_torch, which is an optional dependency: only import
    this module if text-conditioned inpainting is actually used.
    """

    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cpu"):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "FrozenTextEncoder requires open_clip_torch. Install it with "
                "`pip install open_clip_torch` to use text-conditioned inpainting."
            ) from exc
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(device)
        self.output_dim = model.text_projection.shape[1]
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, prompts):
        """prompts: list[str] of length B. Returns (B, output_dim) float tensor."""
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(prompts).to(device)
        return self.model.encode_text(tokens).float()
