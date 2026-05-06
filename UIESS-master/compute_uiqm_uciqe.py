"""
Underwater Image Quality Metrics: UIQM and UCIQE
=================================================
Notebook-ready version for Lightning AI.

Input  (raw degraded):  /teamspace/studios/this_studio/squid
Output (enhanced):      /teamspace/studios/this_studio/output/squid_atm_v3/test_REAL_image/35
"""

import os, math, csv
import numpy as np
from scipy import ndimage
from PIL import Image
from skimage import color
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
input_dir  = "/teamspace/studios/this_studio/squid"
output_dir = "/teamspace/studios/this_studio/output/squid_atm_v3/test_REAL_image/35"
save_csv   = True   # save results to CSV alongside images
# ────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ============================================================================
#  UIQM  –  Underwater Image Quality Measure  (Panetta et al., 2016)
# ============================================================================

def _mu_alpha(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = int(math.ceil(alpha_L * K))
    T_a_R = int(math.floor(alpha_R * K))
    weight = 1.0 / (K - T_a_L - T_a_R)
    return weight * sum(x[T_a_L + 1 : K - T_a_R])

def _sigma_alpha(x, mu):
    return sum((p - mu) ** 2 for p in x) / len(x)

def _uicm(img):
    R = img[:, :, 0].flatten()
    G = img[:, :, 1].flatten()
    B = img[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2.0) - B
    mu_RG  = _mu_alpha(RG)
    mu_YB  = _mu_alpha(YB)
    s_RG   = _sigma_alpha(RG, mu_RG)
    s_YB   = _sigma_alpha(YB, mu_YB)
    l_val  = math.sqrt(mu_RG ** 2 + mu_YB ** 2)
    r_val  = math.sqrt(s_RG + s_YB)
    return -0.0268 * l_val + 0.1586 * r_val

def _sobel(x):
    dx  = ndimage.sobel(x, axis=0)
    dy  = ndimage.sobel(x, axis=1)
    mag = np.hypot(dx, dy)
    mx  = np.max(mag)
    if mx != 0:
        mag *= 255.0 / mx
    return mag

def _eme(x, window_size):
    k1 = int(x.shape[1] // window_size)
    k2 = int(x.shape[0] // window_size)
    if k1 == 0 or k2 == 0:
        return 0.0
    w   = 2.0 / (k1 * k2)
    x   = x[: k2 * window_size, : k1 * window_size]
    val = 0.0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:(k+1) * window_size,
                      l * window_size:(l+1) * window_size]
            mx = np.max(block)
            mn = np.min(block)
            if mn > 0 and mx > 0:
                val += math.log(mx / mn)
    return w * val

def _uism(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Rs, Gs, Bs = _sobel(R), _sobel(G), _sobel(B)
    r_eme = _eme(np.multiply(Rs, R), 10)
    g_eme = _eme(np.multiply(Gs, G), 10)
    b_eme = _eme(np.multiply(Bs, B), 10)
    return 0.299 * r_eme + 0.587 * g_eme + 0.114 * b_eme

def _uiconm(img, window_size):
    k1 = int(img.shape[1] // window_size)
    k2 = int(img.shape[0] // window_size)
    if k1 == 0 or k2 == 0:
        return 0.0
    w   = -1.0 / (k1 * k2)
    img = img[: k2 * window_size, : k1 * window_size]
    val = 0.0
    for l in range(k1):
        for k in range(k2):
            block = img[k * window_size:(k+1) * window_size,
                        l * window_size:(l+1) * window_size, :]
            mx  = np.max(block)
            mn  = np.min(block)
            top = mx - mn
            bot = mx + mn
            if bot == 0.0 or top == 0.0 or math.isnan(top) or math.isnan(bot):
                continue
            val += ((top / bot)) * math.log(top / bot)
    return w * val

def compute_uiqm(img_rgb):
    x = np.array(img_rgb, dtype=np.float64)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm   = c1 * uicm + c2 * uism + c3 * uiconm
    return uiqm, uicm, uism, uiconm

# ============================================================================
#  UCIQE  –  Underwater Color Image Quality Evaluation  (Yang & Sowmya, 2015)
# ============================================================================

def compute_uciqe(img_rgb):
    """
    Compute UCIQE with normalized components to match paper scale [0, 1].

    - sigma_c : std of chroma (from Lab, normalized to [0,1])
    - con_l   : luminance contrast (L normalized to [0,1])
    - mu_s    : mean saturation (from HSV, already [0,1])
    """
    img = np.array(img_rgb, dtype=np.uint8)

    # --- Lab color space (skimage: L=[0,100], a,b≈[-128,127]) ---
    lab = color.rgb2lab(img)
    L   = lab[:, :, 0] / 100.0          # normalize to [0, 1]
    a   = (lab[:, :, 1] + 128.0) / 255.0  # normalize to [0, 1]
    b   = (lab[:, :, 2] + 128.0) / 255.0  # normalize to [0, 1]

    # Chroma on normalized a, b (centered at 0.5)
    chroma  = np.sqrt((a - 0.5) ** 2 + (b - 0.5) ** 2)
    sigma_c = np.std(chroma)

    # Luminance contrast (top 1% – bottom 1% of normalized L)
    L_sorted = np.sort(L.flatten())
    n    = len(L_sorted)
    top1 = int(math.ceil(0.99 * n))
    bot1 = max(int(math.floor(0.01 * n)), 1)
    con_l = np.mean(L_sorted[top1:]) - np.mean(L_sorted[:bot1])

    # Mean saturation from HSV (already in [0, 1])
    hsv  = color.rgb2hsv(img)
    mu_s = np.mean(hsv[:, :, 1])

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s
    return uciqe, sigma_c, con_l, mu_s

# ============================================================================
#  Evaluate a directory
# ============================================================================

def evaluate_directory(dir_path, label=""):
    paths = sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    print(f"\n{'='*80}")
    print(f"  [{label}]  {dir_path}  –  {len(paths)} image(s)")
    print(f"{'='*80}\n")

    rows = []
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)
        print(f"  [{i}/{len(paths)}] {name} ... ", end="", flush=True)
        try:
            img = np.array(Image.open(p).convert("RGB"))
            uiqm, uicm, uism, uiconm = compute_uiqm(img)
            uciqe, sigma_c, con_l, mu_s = compute_uciqe(img)
            rows.append({
                "filename": name,
                "UIQM": round(uiqm, 4),  "UICM": round(uicm, 4),
                "UISM": round(uism, 4),   "UIConM": round(uiconm, 4),
                "UCIQE": round(uciqe, 4), "sigma_c": round(sigma_c, 4),
                "con_l": round(con_l, 4), "mu_s": round(mu_s, 4),
            })
            print(f"UIQM={uiqm:.4f}  |  UCIQE={uciqe:.4f}")
        except Exception as e:
            print(f"ERROR → {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n{'─'*80}")
        print(f"  Mean UIQM  = {df['UIQM'].mean():.4f}  ±  {df['UIQM'].std():.4f}")
        print(f"  Mean UCIQE = {df['UCIQE'].mean():.4f}  ±  {df['UCIQE'].std():.4f}")
        print(f"{'─'*80}")

    if save_csv and not df.empty:
        csv_path = os.path.join(dir_path, f"metrics_{label}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Saved → {csv_path}\n")

    return df

# ============================================================================
#  RUN
# ============================================================================

print("📊  Computing UIQM & UCIQE metrics ...\n")

df_input  = evaluate_directory(input_dir,  label="input_raw")
df_output = evaluate_directory(output_dir, label="enhanced")

# ── Side-by-side comparison (matched by filename) ──────────────────────────
if not df_input.empty and not df_output.empty:
    merged = pd.merge(
        df_input[["filename", "UIQM", "UCIQE"]],
        df_output[["filename", "UIQM", "UCIQE"]],
        on="filename", suffixes=("_input", "_enhanced"), how="inner"
    )
    if not merged.empty:
        merged["UIQM_gain"]  = merged["UIQM_enhanced"]  - merged["UIQM_input"]
        merged["UCIQE_gain"] = merged["UCIQE_enhanced"] - merged["UCIQE_input"]
        
        comp_csv = os.path.join(output_dir, "comparison.csv")
        merged.to_csv(comp_csv, index=False)
        print(f"  ✅ Comparison saved → {comp_csv}")

print("\n🏁  Done!\n")

print("="*60)
print("  Final Results Table")
print("="*60)
print(f"| {'Methods':<15} | {'UIQM':<15} | {'UCIQE':<15} |")
print(f"|{'-'*17}|{'-'*17}|{'-'*17}|")
if not df_input.empty:
    print(f"| {'Input':<15} | {df_input['UIQM'].mean():<15.4f} | {df_input['UCIQE'].mean():<15.4f} |")
if not df_output.empty:
    print(f"| {'Ours':<15} | {df_output['UIQM'].mean():<15.4f} | {df_output['UCIQE'].mean():<15.4f} |")
print("="*60)
