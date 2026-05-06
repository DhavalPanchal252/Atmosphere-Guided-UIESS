"""
models.py — UIESS model architecture
======================================
Original: Chen & Pei (2022) — UIESS
v3 additions (marked [NEW v3]):
  - StyleTransformUnit: the latent transform unit T used in the original paper
    is included here explicitly so train_v3.py and inference.py can import it
    cleanly without redefining it.

Note: StyleTransformUnit was defined inline in the original train.py.
Moving it here makes it a first-class part of the architecture module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Weight initialisation  (original)
# ──────────────────────────────────────────────────────────────────────────────

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


# ──────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler  (original)
# ──────────────────────────────────────────────────────────────────────────────

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, \
            "Decay must start before training ends!"
        self.n_epochs = n_epochs
        self.offset   = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch)


# ──────────────────────────────────────────────────────────────────────────────
# Custom normalisation layers  (original)
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps      = eps
        self.momentum = momentum
        self.weight   = None
        self.bias     = None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, \
            "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        out = F.batch_norm(
            x.contiguous().view(1, b * c, h, w),
            self.running_mean.repeat(b),
            self.running_var.repeat(b),
            self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps    = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta  = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean  = x.view(x.size(0), -1).mean(1).view(*shape)
        std   = x.view(x.size(0), -1).std(1).view(*shape)
        x     = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks  (original)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super().__init__()
        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


# ──────────────────────────────────────────────────────────────────────────────
# Encoders  (original)
# ──────────────────────────────────────────────────────────────────────────────

class ContentEncoder(nn.Module):
    """Shared content encoder — instance norm strips style, preserving structure."""

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StyleEncoder(nn.Module):
    """
    Domain-specific style encoder — no instance norm, so style statistics
    are preserved. Outputs a fixed-dim style vector via global average pooling.

    Used for three domains:
      real_sty_Enc  — real-world underwater (domain A)
      syn_sty_Enc   — synthetic underwater  (domain B)
      atm_sty_Enc   — atmospheric clean     (domain C) [NEW v3]
    All three share this same architecture class.
    """

    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.ReLU(inplace=True),
        ]
        for _ in range(2):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            dim *= 2
        for _ in range(n_downsample - 2):
            layers += [
                nn.Conv2d(dim, dim, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, style_dim, 1, 1, 0),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# MLP  (original)
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super().__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


# ──────────────────────────────────────────────────────────────────────────────
# Generator  (original)
# ──────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Decoder with AdaIN style injection."""

    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super().__init__()
        dim    = dim * 2 ** n_upsample
        layers = []
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        num_adain  = self.get_num_adain_params()
        self.mlp   = MLP(style_dim, num_adain)

    def get_num_adain_params(self):
        return sum(2 * m.num_features for m in self.modules()
                   if m.__class__.__name__ == "AdaptiveInstanceNorm2d")

    def assign_adain_params(self, adain_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias   = adain_params[:, :m.num_features].contiguous().view(-1)
                m.weight = adain_params[:, m.num_features:2*m.num_features].contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def forward(self, content_code, style_code):
        self.assign_adain_params(self.mlp(style_code))
        return self.model(content_code)


# ──────────────────────────────────────────────────────────────────────────────
# [NEW v3] Style Transform Unit
# ──────────────────────────────────────────────────────────────────────────────

class StyleTransformUnit(nn.Module):
    """
    [NEW v3] Latent transform T: maps degraded underwater style → clean style.

    Implemented as an MLP with a skip connection (residual), which encourages
    the network to learn the *delta* from degraded to clean rather than the
    absolute mapping. This is the module whose output is guided by Latm in v3.

    Input/output: style vector Z_S ∈ R^{style_dim}  (after global avg pool)
    """

    def __init__(self, dim=64, style_dim=8):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(style_dim, dim),
            nn.PReLU(),
            nn.Linear(dim, style_dim),
        )

    def forward(self, style_code):
        return style_code + self.estimator(style_code).view(-1, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scale Discriminator  (original)
# ──────────────────────────────────────────────────────────────────────────────

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module("disc_%d" % i, nn.Sequential(
                *discriminator_block(in_channels, 64, normalize=False),
                *discriminator_block(64, 128),
                *discriminator_block(128, 256),
                *discriminator_block(256, 512),
                nn.Conv2d(512, 1, 3, padding=1),
            ))
        self.downsample = nn.AvgPool2d(
            in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        return sum(torch.mean((out - gt) ** 2) for out in self.forward(x))

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs