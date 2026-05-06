"""
datasets.py — UIESS dataset definitions
========================================
Original: Chen & Pei (2022) — UIESS
v3 additions (marked [NEW v3]):
  - trainC atmospheric clean image loading in EnhancedDataset
  - ValDataset: lightweight dataset for per-epoch quantitative validation
"""

import os
import random

import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Utilities  (original)
# ──────────────────────────────────────────────────────────────────────────────

def is_image_file(filename):
    return any(filename.endswith(ext)
               for ext in [".bmp", ".png", ".jpg", ".jpeg", ".JPG", ".PNG"])


def load_img(filepath):
    return Image.open(filepath).convert("RGB")


def get_patch(imgs, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = imgs[0].size
    tp = patch_size * scale
    ip = tp // scale
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy
    return tuple(img.crop((ty, tx, ty + tp, tx + tp)) for img in imgs)


def augmentation(imgs, flip_h=True, rot=True):
    info_aug = {"flip_h": False, "flip_v": False, "trans": False}
    if random.random() < 0.5 and flip_h:
        imgs = [ImageOps.flip(img) for img in imgs]
        info_aug["flip_h"] = True
    if rot:
        if random.random() < 0.5:
            imgs = [ImageOps.mirror(img) for img in imgs]
            info_aug["flip_v"] = True
        if random.random() < 0.5:
            imgs = [img.rotate(180) for img in imgs]
        info_aug["trans"] = True
    return tuple(imgs), info_aug


# ──────────────────────────────────────────────────────────────────────────────
# Main training / validation dataset  (original + [NEW v3] trainC block)
# ──────────────────────────────────────────────────────────────────────────────

class EnhancedDataset(Dataset):
    """
    Loads pairs from trainA (real), trainB (synthetic), trainB_label (clean ref).

    [NEW v3] If a trainC/ folder exists under root, atmospheric clean images are
    loaded alongside each sample and returned as batch["Atm"]. This enables the
    atmosphere-guided style loss in train_v3.py. The folder is optional — if
    absent, the dataset behaves identically to the original UIESS version.
    """

    def __init__(self, root, transforms_=None, mode="train", patch_size=128):
        super().__init__()
        self.transform  = transforms.Compose(transforms_)
        self.mode       = mode
        self.patch_size = patch_size

        if mode == "train":
            self.filesA = sorted([
                os.path.join(root, "trainA", x)
                for x in os.listdir(os.path.join(root, "trainA"))
                if is_image_file(x)])
            self.filesB = sorted([
                os.path.join(root, "trainB", x)
                for x in os.listdir(os.path.join(root, "trainB"))
                if is_image_file(x)])
            self.labelB = sorted([
                os.path.join(root, "trainB_label", x)
                for x in os.listdir(os.path.join(root, "trainB_label"))
                if is_image_file(x)])

            # [NEW v3] atmospheric clean images — optional third domain
            trainC_path = os.path.join(root, "trainC")
            self.filesC = (
                sorted([os.path.join(trainC_path, x)
                        for x in os.listdir(trainC_path)
                        if is_image_file(x)])
                if os.path.isdir(trainC_path) else []
            )
        else:
            self.filesA = sorted([
                os.path.join(root, "testA", x)
                for x in os.listdir(os.path.join(root, "testA"))
                if is_image_file(x)])
            self.filesB = sorted([
                os.path.join(root, "testB", x)
                for x in os.listdir(os.path.join(root, "testB"))
                if is_image_file(x)])
            self.labelB = sorted([
                os.path.join(root, "testB_label", x)
                for x in os.listdir(os.path.join(root, "testB_label"))
                if is_image_file(x)])
            self.filesC = []  # no atmosphere supervision at test time

    def __getitem__(self, index):
        img_A   = load_img(self.filesA[index % len(self.filesA)])
        img_B   = load_img(self.filesB[index])
        label_B = load_img(self.labelB[index])

        if self.mode == "train":
            img_A          = get_patch([img_A], self.patch_size)[0]
            img_B, label_B = get_patch([img_B, label_B], self.patch_size)
            (img_A, img_B, label_B), _ = augmentation([img_A, img_B, label_B])

        if self.mode == "val":
            w, h   = img_A.size
            new_w  = w // 4 * 4 if w % 4 else w
            new_h  = h // 4 * 4 if h % 4 else h
            if new_w != w or new_h != h:
                img_A = img_A.resize((new_w, new_h))

        sample = {
            "Real":  self.transform(img_A),
            "Syn":   self.transform(img_B),
            "label": self.transform(label_B),
        }

        # [NEW v3] attach atmosphere image if trainC is available
        if self.filesC:
            img_C = load_img(self.filesC[index % len(self.filesC)])
            if self.mode == "train":
                img_C       = get_patch([img_C], self.patch_size)[0]
                (img_C,), _ = augmentation([img_C])
            sample["Atm"] = self.transform(img_C)

        return sample

    def __len__(self):
        return len(self.filesA) if self.mode == "val" else len(self.filesB)


class EnhancedValDataset(Dataset):
    """Original single-folder validation dataset (used by test.py and train.py)."""

    def __init__(self, transforms_=None, dataset_path="train", patch_size=128):
        super().__init__()
        self.transform = transforms.Compose(transforms_)
        self.files = [
            os.path.join(dataset_path, x)
            for x in os.listdir(dataset_path)
            if is_image_file(x)]

    def __getitem__(self, index):
        img = load_img(self.files[index % len(self.files)])
        w, h  = img.size
        new_w = w // 4 * 4 if w % 4 else w
        new_h = h // 4 * 4 if h % 4 else h
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))
        return {"img": self.transform(img), "name": self.files[index % len(self.files)]}

    def __len__(self):
        return len(self.files)


# ──────────────────────────────────────────────────────────────────────────────
# [NEW v3] Quantitative validation dataset
# ──────────────────────────────────────────────────────────────────────────────

class ValDataset(Dataset):
    """
    [NEW v3] Lightweight dataset for per-epoch PSNR/SSIM tracking in train_v3.py.
    Loads images from a single folder, pads to multiples of 4, and returns
    the original filename so ground-truth can be matched by name.
    """

    def __init__(self, dir_path, transforms_):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path) if is_image_file(f)])

    def __getitem__(self, index):
        img = load_img(self.files[index])
        w, h = img.size
        img  = img.resize((w // 4 * 4 or w, h // 4 * 4 or h))
        return {"img": self.transform(img), "name": self.files[index]}

    def __len__(self):
        return len(self.files)