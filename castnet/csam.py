import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossScaleAffinityMixer(nn.Module):
    """Mask-aware texture retrieval from observed source locations.

    Optionally text-conditioned: when text_dim is set, a frozen prompt
    embedding modulates the hole queries via FiLM (Eq. 5a-5b in the
    text-guided extension note) before the affinity computation in Eq. 5.
    With text_dim=None, or with a zero-initialized FiLM head and no prompt
    passed at forward time, behaviour is bit-for-bit identical to the base
    (unconditional) module.
    """

    def __init__(self, fine_channels, channels, temperature=0.07, text_dim=None):
        super().__init__()
        self.temperature = temperature
        self.fine_project = nn.Conv2d(fine_channels, channels, 1)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.output = nn.Conv2d(channels, channels, 1)

        self.text_dim = text_dim
        if text_dim is not None:
            self.film = nn.Linear(text_dim, channels * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def forward(self, bottleneck, fine_feature, valid_mask, text_embedding=None, return_diagnostics=False):
        if text_embedding is not None and self.film is None:
            raise ValueError("text_embedding was provided but this module was built with text_dim=None")
        fine = self.fine_project(fine_feature)
        fine = F.interpolate(fine, bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        mixed = bottleneck + fine
        b, c, h, w = mixed.shape
        valid = F.interpolate(valid_mask.float(), (h, w), mode="nearest").flatten(1).bool()
        has_valid = valid.any(dim=1, keepdim=True)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~has_valid.squeeze(1)

        q = F.normalize(self.query(bottleneck).flatten(2).transpose(1, 2), dim=-1)
        if text_embedding is not None:
            gamma, beta = self.film(text_embedding).chunk(2, dim=-1)  # each (B, C)
            q = F.normalize((1.0 + gamma.unsqueeze(1)) * q + beta.unsqueeze(1), dim=-1)
        k = F.normalize(self.key(mixed).flatten(2).transpose(1, 2), dim=-1)
        v = self.value(mixed).flatten(2).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(1, 2)) / max(self.temperature, 1e-6)
        logits = logits.masked_fill(~safe_valid[:, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1)
        transported = torch.matmul(weights, v).transpose(1, 2).reshape(b, c, h, w)
        transported = transported * has_valid.view(b, 1, 1, 1)
        hole = 1.0 - F.interpolate(valid_mask.float(), (h, w), mode="nearest")
        gate = self.gate(torch.cat([bottleneck, transported], dim=1))
        output = bottleneck + hole * gate * self.output(transported)
        if return_diagnostics:
            return output, {"affinity": weights, "valid_sources": valid}
        return output
