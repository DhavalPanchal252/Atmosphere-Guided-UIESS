"""
loss.py — UIESS loss functions
================================
Original: Chen & Pei (2022) — UIESS
v3 additions (marked [NEW v3]):
  - TVLoss: total variation regulariser (was inline in train.py)
  - VGGPerceptualLoss: VGG-16 feature + style loss (was inline in train.py)
Both were used in the original training but defined inline in the notebook.
Moving them here keeps train_v3.py focused on training logic only.
"""

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ──────────────────────────────────────────────────────────────────────────────
# SSIM  (original)
# ──────────────────────────────────────────────────────────────────────────────

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, window=None,
         size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1  if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window    = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2

    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    v1, v2 = 2.0 * sigma12 + C2, sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ret = ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)
    return (ret, torch.mean(v1 / v2)) if full else ret


class SSIM(nn.Module):
    """Differentiable SSIM loss (original)."""

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()
        self.window_size  = window_size
        self.size_average = size_average
        self.val_range    = val_range
        self.channel = 1
        self.window  = create_window(window_size)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window  = window
            self.channel = channel
        return ssim(img1, img2, window=window,
                    window_size=self.window_size,
                    size_average=self.size_average,
                    val_range=1)


# ──────────────────────────────────────────────────────────────────────────────
# [NEW v3] Total Variation Loss
# ──────────────────────────────────────────────────────────────────────────────

class TVLoss(nn.Module):
    """
    [NEW v3] Total Variation regulariser — penalises high-frequency noise in
    generated images by minimising intensity differences between neighbouring
    pixels. Promotes smooth, coherent output especially in homogeneous regions.

    Lossweight=1 matches the λ_tv=0.3 weighting applied in train_v3.py.
    """

    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x, w_x  = x.size(2), x.size(3)
        count_h    = self._tensor_size(x[:, :, 1:, :])
        count_w    = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)


# ──────────────────────────────────────────────────────────────────────────────
# [NEW v3] VGG Perceptual Loss
# ──────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    [NEW v3] Perceptual loss using VGG-16 feature maps.

    Combines:
      - Feature loss (L1 on intermediate activations) for semantic consistency
      - Style loss   (L1 on Gram matrices) for texture matching

    Layers used by default (feature_layers=[0,1,2], style_layers=[2,3])
    match the configuration from the paper experiments.

    The λ_perceptual=0.0005/2 weighting in train_v3.py keeps this loss from
    dominating pixel-level objectives.
    """

    def __init__(self, weighting=1, resize=False):
        super().__init__()
        vgg    = torchvision.models.vgg16(pretrained=True).features
        blocks = [
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval(),
        ]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks    = nn.ModuleList(blocks)
        self.resize    = resize
        self.weighting = weighting
        self.mean = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std  = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inp, target,
                feature_layers=(0, 1, 2), style_layers=(2, 3)):
        if inp.shape[1] != 3:
            inp    = inp.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        inp    = (inp    - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            inp    = F.interpolate(inp,    mode="bilinear", size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode="bilinear", size=(224, 224), align_corners=False)

        loss = 0.0
        x, y = inp, target
        for i, block in enumerate(self.blocks):
            x, y = block(x), block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                ax = x.reshape(x.shape[0], x.shape[1], -1)
                ay = y.reshape(y.shape[0], y.shape[1], -1)
                loss += F.l1_loss(ax @ ax.permute(0, 2, 1),
                                  ay @ ay.permute(0, 2, 1))
        return self.weighting * loss