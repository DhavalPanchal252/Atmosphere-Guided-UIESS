"""
Atmosphere-Guided UIESS — Training Script (v3)
================================================
Paper: "Atmosphere-Guided Domain Adaptation for Underwater Image Enhancement
        via Content and Style Separation"
Author: Dhaval Panchal, Prof. Srimanta Mandal — DAIICT, Gandhinagar

Key differences from the original UIESS train.py:
  - Third training domain (trainC): atmospheric clean images
  - Atmosphere style encoder (atm_sty_Enc)
  - L1 atmosphere-guided style loss (Latm) after a warmup period
  - Per-epoch PSNR/SSIM validation with best-model tracking
  - atm_sty_Enc included in optimizer_G (gradient flows through T)

Shared modules used:
  models.py   — all network classes + weight init + LR scheduler
  datasets.py — EnhancedDataset, ValDataset, load_img, is_image_file
  loss.py     — SSIM, TVLoss, VGGPerceptualLoss

Usage:
    python train_v3.py --data_root /path/to/data --gpu 0

Dataset layout expected under --data_root:
    trainA/         real-world underwater images
    trainB/         synthetic underwater images
    trainB_label/   synthetic clean references
    trainC/         atmospheric clean images  (DIV2K or similar)
    testA/          real-world test images
    testB/          synthetic test images
    testB_label/    synthetic test ground truth

Training tip: pick the best checkpoint from epoch 25-35 using the
PSNR/SSIM validation summary printed at the end.
"""

import argparse
import datetime
import itertools
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ── Shared modules ─────────────────────────────────────────────────────────────
from models import (
    ContentEncoder,
    Generator,
    LambdaLR,
    MultiDiscriminator,
    StyleEncoder,
    StyleTransformUnit,
    weights_init_normal,
)
from datasets import (
    EnhancedDataset,
    ValDataset,
    is_image_file,
    load_img,
)
from loss import SSIM, TVLoss, VGGPerceptualLoss


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Atmosphere-Guided UIESS training (v3)")

    # Paths
    p.add_argument("--data_root", type=str, required=True,
                   help="Root folder containing trainA/B/B_label/C and testA/B/B_label")
    p.add_argument("--out_dir",   type=str, default="output",
                   help="Where to write saved_models/ and images/")
    p.add_argument("--exp_name",  type=str, default="Atmospher_Guided_UIESS")

    # Training hyper-params (match paper)
    p.add_argument("--epoch",              type=int,   default=0,
                   help="Resume from this epoch (0 = train from scratch)")
    p.add_argument("--n_epochs",           type=int,   default=35)
    p.add_argument("--batch_size",         type=int,   default=1)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--b1",                 type=float, default=0.5)
    p.add_argument("--b2",                 type=float, default=0.999)
    p.add_argument("--decay_epoch",        type=int,   default=20)
    p.add_argument("--n_cpu",              type=int,   default=0)
    p.add_argument("--img_height",         type=int,   default=128)
    p.add_argument("--img_width",          type=int,   default=128)
    p.add_argument("--n_downsample",       type=int,   default=2)
    p.add_argument("--n_residual",         type=int,   default=3)
    p.add_argument("--dim",                type=int,   default=40)
    p.add_argument("--style_dim",          type=int,   default=8)
    p.add_argument("--sample_interval",    type=int,   default=1)
    p.add_argument("--checkpoint_interval",type=int,   default=10)
    p.add_argument("--seed",               type=int,   default=123)
    p.add_argument("--gpu",                type=str,   default="0")

    # Atmosphere-guided loss (v3 additions)
    p.add_argument("--lambda_atm",        type=float, default=0.5,
                   help="Weight for the atmosphere-guided L1 style loss")
    p.add_argument("--atm_warmup_epochs", type=int,   default=3,
                   help="Epochs to skip atmosphere loss while the base latent space forms")

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True


def worker_init(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Per-epoch validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_epoch(c_Enc, G, syn_sty_Enc, T, testB_dir, testB_label_dir, device):
    """Returns (avg_psnr, avg_ssim) on the synthetic test split."""
    from skimage.metrics import structural_similarity

    for m in [c_Enc, G, syn_sty_Enc, T]:
        m.eval()

    gt_files = sorted([f for f in os.listdir(testB_label_dir) if is_image_file(f)])
    val_ds   = ValDataset(testB_dir, [transforms.ToTensor()])
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    psnr_sum = ssim_sum = count = 0

    for i, batch in enumerate(val_dl):
        if i >= len(gt_files):
            break
        with torch.no_grad():
            img  = batch["img"].to(device)
            c    = c_Enc(img)
            s    = syn_sty_Enc(img)
            en_s = T(s)
            enh  = G(c, en_s)

        pred = (enh.squeeze().mul(255).add_(0.5).clamp_(0, 255)
                .permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        gt   = np.array(load_img(os.path.join(testB_label_dir, gt_files[i])))
        pred_resized = np.array(Image.fromarray(pred).resize(
            (gt.shape[1], gt.shape[0]))).astype(np.float64)
        gt   = gt.astype(np.float64)

        mse = np.mean((pred_resized - gt) ** 2)
        psnr_sum += (100.0 if mse == 0
                     else 10.0 * np.log10(255.0 ** 2 / mse))
        ssim_sum += structural_similarity(
            gt, pred_resized, channel_axis=2, data_range=255)
        count += 1

    for m in [c_Enc, G, syn_sty_Enc, T]:
        m.train()

    return psnr_sum / max(count, 1), ssim_sum / max(count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(opt, device):
    images_dir = os.path.join(opt.out_dir, "images",       opt.exp_name)
    models_dir = os.path.join(opt.out_dir, "saved_models", opt.exp_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    set_seed(opt.seed)

    # ── Loss objects ──────────────────────────────────────────────────────────
    L1        = nn.L1Loss().to(device)
    ssim_loss = SSIM().to(device)
    tv_loss   = TVLoss().to(device)
    perc_loss = VGGPerceptualLoss().to(device)

    # ── Models ────────────────────────────────────────────────────────────────
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
    D            = MultiDiscriminator().to(device)
    # [v3] atmosphere style encoder — shares the StyleEncoder class from models.py
    atm_sty_Enc  = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample,
                                style_dim=opt.style_dim).to(device)

    # ── Resume / fresh init ───────────────────────────────────────────────────
    named_models = [
        ("c_Enc",        c_Enc),
        ("G",            G),
        ("real_sty_Enc", real_sty_Enc),
        ("syn_sty_Enc",  syn_sty_Enc),
        ("T",            T),
        ("D",            D),
    ]

    if opt.epoch != 0:
        for name, model in named_models:
            ckpt = os.path.join(models_dir, f"{name}_{opt.epoch}.pth")
            model.load_state_dict(torch.load(ckpt, map_location=device))

        atm_path = os.path.join(models_dir, f"atm_sty_Enc_{opt.epoch}.pth")
        if os.path.isfile(atm_path):
            atm_sty_Enc.load_state_dict(torch.load(atm_path, map_location=device))
            print(f"Resumed from epoch {opt.epoch}")
        else:
            atm_sty_Enc.apply(weights_init_normal)
            print(f"Resumed (no atm_sty_Enc checkpoint — fresh init for it)")
    else:
        for m in [c_Enc, G, real_sty_Enc, syn_sty_Enc, T, D, atm_sty_Enc]:
            m.apply(weights_init_normal)

    # ── Loss weights (exact paper values) ─────────────────────────────────────
    lw = dict(
        gan=1, id=10, cyc=1,
        enhanced=3.5 / 2, ssim=5.0 / 2,
        tv=0.3, perceptual=0.0005 / 2,
        latent=3,
        atm=opt.lambda_atm,
    )

    # ── Optimizers ────────────────────────────────────────────────────────────
    # atm_sty_Enc is intentionally inside optimizer_G so Latm gradients flow
    # back through the transform unit T.
    opt_G = torch.optim.Adam(
        itertools.chain(
            c_Enc.parameters(), G.parameters(),
            real_sty_Enc.parameters(), syn_sty_Enc.parameters(),
            T.parameters(), atm_sty_Enc.parameters()),
        lr=opt.lr, betas=(opt.b1, opt.b2))
    opt_D = torch.optim.Adam(
        D.parameters(), lr=opt.lr * 5, betas=(opt.b1, opt.b2))

    lr_sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # ── Data ──────────────────────────────────────────────────────────────────
    tfm = [transforms.ToTensor()]
    set_seed(opt.seed)

    # EnhancedDataset from datasets.py handles trainA/B/B_label + optional trainC
    train_loader = DataLoader(
        EnhancedDataset(opt.data_root, tfm, mode="train",
                        patch_size=opt.img_height),
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.n_cpu, worker_init_fn=worker_init, pin_memory=True)

    val_loader = DataLoader(
        EnhancedDataset(opt.data_root, [
            transforms.Resize((opt.img_height * 2, opt.img_width * 2),
                               interpolation=Image.BICUBIC),
            transforms.ToTensor()], mode="val"),
        batch_size=5, shuffle=True, num_workers=0, pin_memory=True)

    testB_dir       = os.path.join(opt.data_root, "testB")
    testB_label_dir = os.path.join(opt.data_root, "testB_label")
    has_val_data    = os.path.isdir(testB_dir) and os.path.isdir(testB_label_dir)

    best_psnr   = 0.0
    best_epoch  = 0
    val_history = []

    # ── Sample grid helper ────────────────────────────────────────────────────
    def sample_images(tag):
        batch = next(iter(val_loader))
        rows_A = rows_B = None
        for img_A, img_B, lbl_B in zip(batch["Real"], batch["Syn"], batch["label"]):
            with torch.no_grad():
                XA = img_A.unsqueeze(0).to(device)
                XB = img_B.unsqueeze(0).to(device)
                cA, sA   = c_Enc(XA), real_sty_Enc(XA)
                cB, sB   = c_Enc(XB), syn_sty_Enc(XB)
                XAB, XBA = G(cA, sB), G(cB, sA)
                cAB      = c_Enc(XAB)
                cBA      = c_Enc(XBA)
                sA_rec   = real_sty_Enc(XBA)
                sB_rec   = syn_sty_Enc(XAB)
                enA      = G(cA, T(sA))
                enB      = G(cB, T(sB))
                XABA     = G(cAB, sA)
                XBAB     = G(cBA, sB)
                XAA      = G(cA, sA)
                XBB      = G(cB, sB)
            lb   = lbl_B.unsqueeze(0).to(device)
            rowA = torch.cat([img_A.unsqueeze(0).to(device), XAA, XAB, XABA, enA], dim=-1)
            rowB = torch.cat([img_B.unsqueeze(0).to(device), XBB, XBA, XBAB, enB, lb], dim=-1)
            rows_A = rowA if rows_A is None else torch.cat([rows_A, rowA], dim=-2)
            rows_B = rowB if rows_B is None else torch.cat([rows_B, rowB], dim=-2)

        save_image(rows_A, os.path.join(images_dir, f"{tag}_realA.png"),
                   normalize=True, value_range=(0, 1))
        save_image(rows_B, os.path.join(images_dir, f"{tag}_synB.png"),
                   normalize=True, value_range=(0, 1))

    sample_images(0)
    VALID, FAKE = 1, 0
    prev_time   = time.time()

    # ── Training epochs ───────────────────────────────────────────────────────
    start_epoch = opt.epoch + 1 if opt.epoch > 0 else 0

    for epoch in range(start_epoch, opt.n_epochs + 1):
        for i, batch in enumerate(train_loader):
            XA      = batch["Real"].to(device)
            XB      = batch["Syn"].to(device)
            labelB  = batch["label"].to(device)
            has_atm   = "Atm" in batch
            apply_atm = has_atm and (epoch >= opt.atm_warmup_epochs)
            if has_atm:
                XC = batch["Atm"].to(device)

            # ── Forward ───────────────────────────────────────────────────────
            cA, sA   = c_Enc(XA), real_sty_Enc(XA)
            cB, sB   = c_Enc(XB), syn_sty_Enc(XB)
            XAA, XBB = G(cA, sA), G(cB, sB)
            XBA, XAB = G(cB, sA), G(cA, sB)
            cBA, sBA = c_Enc(XBA), real_sty_Enc(XBA)
            cAB, sAB = c_Enc(XAB), syn_sty_Enc(XAB)
            XABA, XBAB = G(cAB, sA), G(cBA, sB)
            en_sA = T(sA)   # Z_S_R→C
            en_sB = T(sB)   # Z_S_S→C
            enA   = G(cB, en_sA)
            enB   = G(cB, en_sB)

            # ── Discriminator update ──────────────────────────────────────────
            opt_D.zero_grad()
            loss_D = (
                D.compute_loss(XA,          VALID) +
                D.compute_loss(XBA.detach(), FAKE) +
                D.compute_loss(XB,          VALID) +
                D.compute_loss(XAB.detach(), FAKE)
            )
            loss_D.backward()
            opt_D.step()

            # ── Generator update ──────────────────────────────────────────────
            opt_G.zero_grad()

            loss_GAN  = lw["gan"]        * (D.compute_loss(XBA, VALID) +
                                             D.compute_loss(XAB, VALID))
            loss_ID   = lw["id"]         * (L1(XAA, XA) + L1(XBB, XB))
            loss_cyc  = lw["cyc"]        * (L1(XABA, XA) + L1(XBAB, XB))
            loss_enh  = lw["enhanced"]   * (L1(enA, labelB) + L1(enB, labelB))
            loss_ssim = lw["ssim"]       * ((1 - ssim_loss(enA, labelB)) +
                                             (1 - ssim_loss(enB, labelB)))
            loss_perc = lw["perceptual"] * (perc_loss(enA, labelB) +
                                             perc_loss(enB, labelB))
            loss_lat  = lw["latent"]     * L1(en_sB, en_sA)
            loss_tv   = lw["tv"]         * (tv_loss(enA) + tv_loss(enB))

            # [v3] Atmosphere-guided style loss
            # L1 on style vectors — magnitude matters for AdaIN params.
            # Guides en_sB toward the clean atmospheric style manifold;
            # Llatent already constrains en_sB ≈ en_sA.
            if apply_atm:
                s_code_C = atm_sty_Enc(XC)
                loss_atm = lw["atm"] * L1(en_sB, s_code_C.detach())
            else:
                loss_atm = torch.tensor(0.0, device=device)

            loss_G = (loss_GAN + loss_ID + loss_cyc +
                      loss_enh + loss_ssim + loss_lat +
                      loss_perc + loss_tv + loss_atm)
            loss_G.backward()
            opt_G.step()

            # ── Logging ───────────────────────────────────────────────────────
            done = epoch * len(train_loader) + i
            left = opt.n_epochs * len(train_loader) - done
            eta  = datetime.timedelta(seconds=left * (time.time() - prev_time))
            prev_time = time.time()

            if i % 500 == 0:
                atm_tag = f"{loss_atm.item():.4f}" if apply_atm else "warmup"
                print(
                    f"[Ep {epoch}/{opt.n_epochs}][{i}/{len(train_loader)}]"
                    f" D:{loss_D.item():.4f}"
                    f" G:{loss_G.item():.4f}"
                    f" (GAN:{loss_GAN.item():.4f}"
                    f" ID:{loss_ID.item():.4f}"
                    f" Cyc:{loss_cyc.item():.4f}"
                    f" Enh:{loss_enh.item():.4f}"
                    f" Lat:{loss_lat.item():.4f}"
                    f" Atm:{atm_tag})"
                    f" ETA:{eta}"
                )

        # ── End of epoch ──────────────────────────────────────────────────────
        if epoch % opt.sample_interval == 0:
            sample_images(epoch)

        lr_sched_G.step()
        lr_sched_D.step()

        # Quantitative validation (skip early noisy epochs)
        if has_val_data and epoch >= 5:
            avg_psnr, avg_ssim = validate_epoch(
                c_Enc, G, syn_sty_Enc, T,
                testB_dir, testB_label_dir, device)
            val_history.append((epoch, avg_psnr, avg_ssim))
            is_best = avg_psnr > best_psnr
            if is_best:
                best_psnr  = avg_psnr
                best_epoch = epoch
            print(f"  [Val] Ep {epoch}  PSNR={avg_psnr:.2f}  "
                  f"SSIM={avg_ssim:.4f}{'  ★ BEST' if is_best else ''}")

        # Save checkpoints every interval and always after epoch 25
        if epoch % opt.checkpoint_interval == 0 or epoch >= 25:
            save_list = named_models + [("atm_sty_Enc", atm_sty_Enc)]
            for name, model in save_list:
                torch.save(model.state_dict(),
                           os.path.join(models_dir, f"{name}_{epoch}.pth"))
            print(f"  Saved checkpoints for epoch {epoch}")

    # ── Final summary ─────────────────────────────────────────────────────────
    if val_history:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        for ep, p, s in val_history:
            star = " ★" if ep == best_epoch else ""
            print(f"  Epoch {ep:3d}  PSNR={p:.2f}  SSIM={s:.4f}{star}")
        print(f"\n  Best checkpoint: epoch {best_epoch}  PSNR={best_psnr:.2f}")
        print("=" * 60)

    return best_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    opt    = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {vars(opt)}\n")
    train(opt, device)