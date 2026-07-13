import torch
from torch.nn import functional as F


def masked_psnr(prediction, target, valid_mask, eps=1e-8):
    hole = 1.0 - valid_mask
    mse = ((prediction - target).square() * hole).sum() / (hole.sum() * prediction.shape[1] + eps)
    return 10.0 * torch.log10(4.0 / (mse + eps))


def global_ssim(prediction, target):
    prediction = (prediction + 1.0) / 2.0
    target = (target + 1.0) / 2.0
    mu_x = prediction.mean(dim=(-2, -1), keepdim=True)
    mu_y = target.mean(dim=(-2, -1), keepdim=True)
    var_x = ((prediction - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    var_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    covariance = ((prediction - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)) / ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    return score.mean()
