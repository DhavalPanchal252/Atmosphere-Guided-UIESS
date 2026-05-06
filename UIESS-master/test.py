"""
Atmosphere-Guided UIESS — Inference Script
==========================================
Paper: "Atmosphere-Guided Domain Adaptation for Underwater Image Enhancement
        via Content and Style Separation"
Author: Dhaval Panchal, Prof. Srimanta Mandal — DAIICT, Gandhinagar

Supports four inference modes selectable via --mode:
  real        Enhance real-world underwater images         (testA folder)
  syn         Enhance synthetic underwater images          (testB folder)
  sweep       Controllable enhancement via alpha sweep     (latent manipulation)
  i2i         Cross-domain image-to-image translation demo

Shared modules used:
  models.py   — ContentEncoder, StyleEncoder, Generator, StyleTransformUnit
  datasets.py — is_image_file, load_img

Saved model layout expected under --model_dir:
  c_Enc_<N>.pth
  G_<N>.pth
  real_sty_Enc_<N>.pth
  syn_sty_Enc_<N>.pth
  T_<N>.pth

where <N> is --checkpoint (default 35).

Quick start with the proposed Atmosphere-Guided model:
  python inference.py \\
      --model_dir output/saved_models/Atmospher_Guided_UIESS \\
      --test_dir  /path/to/your/images \\
      --mode real --checkpoint 35

Quick start with the original pretrained UIESS baseline:
  python inference.py \\
      --model_dir UIESS-master/saved_models/UIESS \\
      --test_dir  /path/to/your/images \\
      --mode real --checkpoint 35
"""

import argparse
import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

# ── Shared modules ─────────────────────────────────────────────────────────────
from models import (
    ContentEncoder,
    Generator,
    StyleEncoder,
    StyleTransformUnit,
)
from datasets import is_image_file, load_img


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Atmosphere-Guided UIESS inference",
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--model_dir",  type=str, required=True,
                   help="Folder containing pretrained .pth checkpoints\n"
                        "  Proposed: output/saved_models/Atmospher_Guided_UIESS\n"
                        "  Baseline: UIESS-master/saved_models/UIESS")
    p.add_argument("--test_dir",   type=str, default=None,
                   help="Input image folder (real-world or synthetic)")
    p.add_argument("--syn_dir",    type=str, default=None,
                   help="Synthetic image folder (used for --mode i2i)")
    p.add_argument("--out_dir",    type=str, default="output/inference",
                   help="Root output directory")
    p.add_argument("--mode",       type=str, default="real",
                   choices=["real", "syn", "sweep", "i2i"],
                   help="Which inference to run (default: real)")
    p.add_argument("--checkpoint", type=int, default=35,
                   help="Epoch number to load checkpoints from")

    # Model hyper-params — must match the checkpoint
    p.add_argument("--dim",          type=int, default=40)
    p.add_argument("--style_dim",    type=int, default=8)
    p.add_argument("--n_residual",   type=int, default=3)
    p.add_argument("--n_downsample", type=int, default=2)
    p.add_argument("--seed",         type=int, default=123)
    p.add_argument("--gpu",          type=str, default="0")

    # Sweep options
    p.add_argument("--alphas", type=str,
                   default="0,0.25,0.5,0.75,1.0,1.25,1.5",
                   help="Comma-separated alpha values for enhancement sweep\n"
                        "  0 = original, 1 = full enhancement, >1 = over-enhanced")

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _pad_to_multiple(img, k=4):
    """Resize PIL image so both dims are divisible by k."""
    w, h = img.size
    return img.resize((w // k * k or w, h // k * k or h))


class ImageFolder(Dataset):
    """Single-folder dataset used by all inference modes."""

    def __init__(self, folder, transform):
        self.tfm   = transform
        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder) if is_image_file(f)])
        if not self.files:
            raise ValueError(f"No images found in {folder}")

    def __getitem__(self, i):
        path = self.files[i]
        img  = load_img(path)
        orig_size = img.size          # (W, H) before padding
        img  = _pad_to_multiple(img, 4)
        return {"img": self.tfm(img), "name": path, "orig_size": orig_size}

    def __len__(self):
        return len(self.files)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_models(opt, device):
    """Instantiate and load all generator-side models from shared models.py."""

    def _ckpt(name):
        path = os.path.join(opt.model_dir, f"{name}_{opt.checkpoint}.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Make sure --checkpoint matches the epoch number in the filename.")
        return torch.load(path, map_location=device)

    c_Enc        = ContentEncoder(dim=opt.dim, n_downsample=opt.n_downsample,
                                  n_residual=opt.n_residual).to(device)
    G            = Generator(dim=opt.dim, n_upsample=opt.n_downsample,
                             n_residual=opt.n_residual,
                             style_dim=opt.style_dim).to(device)
    real_sty_Enc = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample,
                                style_dim=opt.style_dim).to(device)
    syn_sty_Enc  = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample,
                                style_dim=opt.style_dim).to(device)
    T            = StyleTransformUnit(dim=opt.dim, style_dim=opt.style_dim).to(device)

    c_Enc.load_state_dict(_ckpt("c_Enc"))
    G.load_state_dict(_ckpt("G"))
    real_sty_Enc.load_state_dict(_ckpt("real_sty_Enc"))
    syn_sty_Enc.load_state_dict(_ckpt("syn_sty_Enc"))
    T.load_state_dict(_ckpt("T"))

    for m in [c_Enc, G, real_sty_Enc, syn_sty_Enc, T]:
        m.eval()

    total_params = sum(
        p.numel() for m in [c_Enc, G, real_sty_Enc, T]
        for p in m.parameters())
    print(f"Models loaded from epoch {opt.checkpoint}  "
          f"({total_params / 1e6:.2f}M inference parameters)")

    return dict(c_Enc=c_Enc, G=G,
                real_sty_Enc=real_sty_Enc, syn_sty_Enc=syn_sty_Enc, T=T)


# ──────────────────────────────────────────────────────────────────────────────
# Shared utility
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor, orig_size=None):
    """Convert a (1,C,H,W) or (C,H,W) tensor to a PIL image.

    If orig_size=(W,H) is given the output is bicubically resized back to the
    original input resolution so the saved file matches the source image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.mul(255).add_(0.5).clamp_(0, 255)
           .permute(1, 2, 0).cpu().numpy().astype("uint8"))
    img = Image.fromarray(arr)
    if orig_size is not None:
        img = img.resize(orig_size, Image.BICUBIC)
    return img


def _make_loader(folder):
    """Return a batch-1 DataLoader for an ImageFolder."""
    return DataLoader(
        ImageFolder(folder, transforms.ToTensor()),
        batch_size=1, shuffle=False, num_workers=0)


# ──────────────────────────────────────────────────────────────────────────────
# Inference modes
# ──────────────────────────────────────────────────────────────────────────────

def run_real(models, test_dir, out_dir, device):
    """
    Enhance real-world underwater images using the real-domain style path.

    Pipeline:  XA → c_Enc → cA
                         real_sty_Enc → sA → T → en_sA → G(cA, en_sA) → enhanced
    Output is saved at the original input resolution.
    """
    out_path = os.path.join(out_dir, "real_enhanced")
    os.makedirs(out_path, exist_ok=True)

    c_Enc, G, real_sty_Enc, T = (
        models[k] for k in ["c_Enc", "G", "real_sty_Enc", "T"])

    times = []
    for batch in _make_loader(test_dir):
        img       = batch["img"].to(device)
        name      = os.path.basename(batch["name"][0])
        orig_size = batch["orig_size"]          # (W, H) tuple of tensors
        orig_size = (orig_size[0].item(), orig_size[1].item())

        t0 = time.time()
        with torch.no_grad():
            enh = G(c_Enc(img), T(real_sty_Enc(img)))
        times.append(time.time() - t0)

        _tensor_to_pil(enh, orig_size).save(os.path.join(out_path, name))
        print(f"  {name}")

    if len(times) > 1:
        avg_ms = 1000 * sum(times[1:]) / len(times[1:])   # skip warm-up
        print(f"\n{len(times)} images — avg {avg_ms:.1f} ms/image  "
              f"({1000/avg_ms:.0f} FPS)")
    print(f"Saved to {out_path}")


def run_syn(models, syn_dir, out_dir, device):
    """
    Enhance synthetic underwater images using the synthetic-domain style path.

    Pipeline:  XB → c_Enc → cB
                         syn_sty_Enc → sB → T → en_sB → G(cB, en_sB) → enhanced
    """
    out_path = os.path.join(out_dir, "syn_enhanced")
    os.makedirs(out_path, exist_ok=True)

    c_Enc, G, syn_sty_Enc, T = (
        models[k] for k in ["c_Enc", "G", "syn_sty_Enc", "T"])

    for batch in _make_loader(syn_dir):
        img       = batch["img"].to(device)
        name      = os.path.basename(batch["name"][0])
        orig_size = batch["orig_size"]
        orig_size = (orig_size[0].item(), orig_size[1].item())

        with torch.no_grad():
            enh = G(c_Enc(img), T(syn_sty_Enc(img)))

        _tensor_to_pil(enh, orig_size).save(os.path.join(out_path, name))
        print(f"  {name}")

    print(f"Saved to {out_path}")


def run_sweep(models, test_dir, out_dir, alphas, device):
    """
    Controllable enhancement sweep via latent interpolation.

    For each alpha in --alphas:
        Z_controlled = Z_S + alpha * (Z_S→C − Z_S)

    alpha = 0     → original style  (no enhancement)
    alpha = 1     → full enhancement
    alpha > 1     → over-enhanced / saturated
    alpha < 0     → further degraded

    Outputs:
      alpha_sweep/<stem>_alpha_strip.png   — all alphas in one horizontal strip
      alpha_sweep/<stem>/alpha_X.XX.png    — per-alpha at original resolution
    """
    out_path = os.path.join(out_dir, "alpha_sweep")
    os.makedirs(out_path, exist_ok=True)

    c_Enc, G, real_sty_Enc, T = (
        models[k] for k in ["c_Enc", "G", "real_sty_Enc", "T"])

    for batch in _make_loader(test_dir):
        img       = batch["img"].to(device)
        stem      = os.path.splitext(os.path.basename(batch["name"][0]))[0]
        orig_size = batch["orig_size"]
        orig_size = (orig_size[0].item(), orig_size[1].item())

        with torch.no_grad():
            c    = c_Enc(img)
            s    = real_sty_Enc(img)
            en_s = T(s)

            # Build a horizontal strip: [input | alpha_0 | alpha_1 | ...]
            strip = img
            for alpha in alphas:
                z_ctrl = s + alpha * (en_s - s)
                strip  = torch.cat([strip, G(c, z_ctrl)], dim=-1)

        save_image(strip,
                   os.path.join(out_path, f"{stem}_alpha_strip.png"),
                   normalize=True, value_range=(0, 1))

        # Individual per-alpha saves at original resolution
        alpha_dir = os.path.join(out_path, stem)
        os.makedirs(alpha_dir, exist_ok=True)
        with torch.no_grad():
            for alpha in alphas:
                z_ctrl = s + alpha * (en_s - s)
                enh    = G(c, z_ctrl)
                _tensor_to_pil(enh, orig_size).save(
                    os.path.join(alpha_dir, f"alpha_{alpha:.2f}.png"))

        print(f"  {stem}  [alphas: {alphas}]")

    print(f"Saved to {out_path}")


def run_i2i(models, test_dir, syn_dir, out_dir, device):
    """
    Cross-domain image-to-image translation demo.

    For each real image XR paired (cyclically) with a synthetic image XS,
    produces a five-column strip:
        input | self-recon | real→syn-style | syn→real-style | enhanced

    Saved to:  i2i_translation/<stem>_i2i.png
    """
    out_path = os.path.join(out_dir, "i2i_translation")
    os.makedirs(out_path, exist_ok=True)

    c_Enc, G, real_sty_Enc, syn_sty_Enc, T = (
        models[k] for k in ["c_Enc", "G", "real_sty_Enc", "syn_sty_Enc", "T"])

    real_dl  = _make_loader(test_dir)
    syn_dl   = _make_loader(syn_dir)
    syn_iter = iter(syn_dl)

    for r_batch in real_dl:
        try:
            s_batch = next(syn_iter)
        except StopIteration:
            syn_iter = iter(syn_dl)
            s_batch  = next(syn_iter)

        XR   = r_batch["img"].to(device)
        XS   = s_batch["img"].to(device)
        stem = os.path.splitext(os.path.basename(r_batch["name"][0]))[0]

        with torch.no_grad():
            cR, sR = c_Enc(XR), real_sty_Enc(XR)
            cS, sS = c_Enc(XS), syn_sty_Enc(XS)
            XRR    = G(cR, sR)       # real self-reconstruction
            XRS    = G(cR, sS)       # real content + syn style
            XSR    = G(cS, sR)       # syn content + real style
            en_R   = G(cR, T(sR))    # real enhanced

        grid = torch.cat([XR, XRR, XRS, XSR, en_R], dim=-1)
        save_image(grid,
                   os.path.join(out_path, f"{stem}_i2i.png"),
                   normalize=True, value_range=(0, 1))
        print(f"  {stem}  [input | self-recon | R→S | S→R | enhanced]")

    print(f"Saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    opt = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(opt.seed)

    print(f"Device: {device}")
    print(f"Mode:   {opt.mode}")
    print(f"Models: {opt.model_dir}  (epoch {opt.checkpoint})\n")

    models = load_models(opt, device)
    alphas = [float(a) for a in opt.alphas.split(",")]

    if opt.mode == "real":
        if opt.test_dir is None:
            raise ValueError("--test_dir required for --mode real")
        run_real(models, opt.test_dir, opt.out_dir, device)

    elif opt.mode == "syn":
        folder = opt.syn_dir or opt.test_dir
        if folder is None:
            raise ValueError("--syn_dir (or --test_dir) required for --mode syn")
        run_syn(models, folder, opt.out_dir, device)

    elif opt.mode == "sweep":
        if opt.test_dir is None:
            raise ValueError("--test_dir required for --mode sweep")
        print(f"Alpha values: {alphas}\n")
        run_sweep(models, opt.test_dir, opt.out_dir, alphas, device)

    elif opt.mode == "i2i":
        if opt.test_dir is None or opt.syn_dir is None:
            raise ValueError("--test_dir and --syn_dir both required for --mode i2i")
        run_i2i(models, opt.test_dir, opt.syn_dir, opt.out_dir, device)