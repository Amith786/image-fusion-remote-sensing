"""
fusion.py — Importable fusion engine for the TACFUSION application.

Public API
----------
    run_fusion(eo_path, sar_path, output_base=None, save_graphs=False) -> dict

    Returns
    -------
    {
        "fused_array"  : np.ndarray  BGR uint8 (256×256)
        "scene_mode"   : str         'normal' | 'night' | 'fog' | 'fire'
        "psnr_wavelet" : float
        "ssim_wavelet" : float
        "psnr_dl"      : float
        "ssim_dl"      : float
        "psnr_final"   : float
        "ssim_final"   : float
        "log_path"     : str | None
        "graph_paths"  : dict        {'psnr': path, 'ssim': path} or {}
    }

CLI
---
    python fusion.py          (interactive image picker, unchanged UX)
"""

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import csv
import os
from datetime import datetime
from metrics import calculate_metrics
from plots import plot_graphs

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(2, 16, 3, padding=1)
        self.relu    = nn.ReLU()
        self.conv2   = nn.Conv2d(16, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

def fuse_channel(c1, c2):
    """Haar DWT fusion for a single image channel (returns uint8 256×256)."""
    c1 = np.float32(cv2.resize(c1, (256, 256)))
    c2 = np.float32(cv2.resize(c2, (256, 256)))

    LL1, (LH1, HL1, HH1) = pywt.dwt2(c1, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(c2, 'haar')
    LL = (LL1 + LL2) / 2
    LH = np.maximum(LH1, LH2)
    HL = np.maximum(HL1, HL2)
    HH = np.maximum(HH1, HH2)
    fused = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    return np.clip(fused, 0, 255).astype('uint8')


def _run_dl(gray1, gray2):
    """FusionNet inference on a grayscale pair → normalised uint8 array."""
    t1 = torch.tensor(gray1 / 255.0).unsqueeze(0).unsqueeze(0).float()
    t2 = torch.tensor(gray2 / 255.0).unsqueeze(0).unsqueeze(0).float()

    model  = FusionNet()
    output = model(t1, t2)

    arr = output.squeeze().detach().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).astype('uint8')


def _adaptive_select(gray1, gray2, fused_color, fused_dl):
    """
    Choose the final BGR output based on EO image statistics.

    Returns
    -------
    final      : np.ndarray  BGR uint8
    scene_mode : str         'normal' | 'night' | 'fog' | 'fire'
    """
    mean_i = float(np.mean(gray1))
    std_i  = float(np.std(gray1))

    if mean_i < 60 and std_i < 40:
        scene = "night"
        final = cv2.cvtColor(cv2.equalizeHist(gray1), cv2.COLOR_GRAY2BGR)

    elif std_i < 25:
        scene = "fog"
        final = fused_color.copy()

    elif mean_i > 170 and std_i > 50:
        scene = "fire"
        final = cv2.cvtColor(np.maximum(gray1, gray2), cv2.COLOR_GRAY2BGR)

    else:
        scene = "normal"
        dl_bgr = cv2.cvtColor(fused_dl, cv2.COLOR_GRAY2BGR)
        final  = (0.6 * fused_color + 0.4 * dl_bgr).astype('uint8')

    return final, scene

def run_fusion(eo_path, sar_path, output_base=None, save_graphs=False):
    """
    Fuse an EO image with a SAR image.

    Parameters
    ----------
    eo_path     : str   Path to the EO (optical) image.
    sar_path    : str   Path to the SAR image.
    output_base : str   Root for logs/graphs. Defaults to 'output' in cwd.
    save_graphs : bool  Write PSNR/SSIM chart PNGs to disk. Default False
                        (set True for CLI use; keep False in Flask to avoid
                        matplotlib GUI calls on a headless server).

    Returns
    -------
    dict  — see module docstring for full key list.
    """

    img1 = cv2.imread(eo_path)
    img2 = cv2.imread(sar_path)
    if img1 is None:
        raise FileNotFoundError(f"EO image not found: {eo_path}")
    if img2 is None:
        raise FileNotFoundError(f"SAR image not found: {sar_path}")

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)
    fused_color = cv2.merge([
        fuse_channel(b1, b2),
        fuse_channel(g1, g2),
        fuse_channel(r1, r2),
    ])


    gray1    = cv2.cvtColor(img1.astype('uint8'), cv2.COLOR_BGR2GRAY)
    gray2    = cv2.cvtColor(img2.astype('uint8'), cv2.COLOR_BGR2GRAY)
    fused_dl = _run_dl(gray1, gray2)

    final, scene_mode = _adaptive_select(gray1, gray2, fused_color, fused_dl)

    (psnr_wavelet, ssim_wavelet,
     psnr_dl,      ssim_dl,
     psnr_final,   ssim_final) = calculate_metrics(
        gray1, fused_color, fused_dl, final
    )

    graph_paths = {}
    log_path    = None

    if output_base is None:
        output_base = "output"

    now       = datetime.now()
    ts        = now.strftime("%Y%m%d_%H%M%S")
    date_dir  = os.path.join(output_base, now.strftime("%Y-%m"), now.strftime("%d"))
    fused_dir = os.path.join(date_dir, "fused")
    psnr_dir  = os.path.join(fused_dir, "Graph", "psnr")
    ssim_dir  = os.path.join(fused_dir, "Graph", "ssim")
    log_dir   = os.path.join(date_dir, "logs")

    for d in (psnr_dir, ssim_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    if save_graphs:
        graph_paths = plot_graphs(
            psnr_wavelet, psnr_dl, psnr_final,
            ssim_wavelet, ssim_dl, ssim_final,
            psnr_dir, ssim_dir, ts,
        )

    log_file     = os.path.join(log_dir, "fusion_log.csv")
    write_header = not os.path.exists(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Time", "EO", "SAR", "SceneMode",
                "PSNR_Wavelet", "SSIM_Wavelet",
                "PSNR_DL",      "SSIM_DL",
                "PSNR_Final",   "SSIM_Final",
            ])
        writer.writerow([
            ts,
            os.path.basename(eo_path),
            os.path.basename(sar_path),
            scene_mode,
            round(psnr_wavelet, 4), round(ssim_wavelet, 4),
            round(psnr_dl,      4), round(ssim_dl,      4),
            round(psnr_final,   4), round(ssim_final,   4),
        ])
    log_path = log_file

    return {
        "fused_array":   final,
        "scene_mode":    scene_mode,
        "psnr_wavelet":  round(psnr_wavelet, 4),
        "ssim_wavelet":  round(ssim_wavelet, 4),
        "psnr_dl":       round(psnr_dl,      4),
        "ssim_dl":       round(ssim_dl,      4),
        "psnr_final":    round(psnr_final,   4),
        "ssim_final":    round(ssim_final,   4),
        "log_path":      log_path,
        "graph_paths":   graph_paths,
    }


if __name__ == "__main__":
    eo_folder  = "input/eo"
    sar_folder = "input/sar"

    eo_files  = os.listdir(eo_folder)
    sar_files = os.listdir(sar_folder)

    print("\nAvailable EO Images:")
    for i, f in enumerate(eo_files):
        print(f"  {i}: {f}")

    print("\nAvailable SAR Images:")
    for i, f in enumerate(sar_files):
        print(f"  {i}: {f}")

    eo_idx  = int(input("\nSelect EO image index: "))
    sar_idx = int(input("Select SAR image index: "))

    eo_path  = os.path.join(eo_folder,  eo_files[eo_idx])
    sar_path = os.path.join(sar_folder, sar_files[sar_idx])

    print(f"\nSelected EO : {eo_files[eo_idx]}")
    print(f"Selected SAR: {sar_files[sar_idx]}")

    result = run_fusion(eo_path, sar_path, save_graphs=True)

    print(f"\nScene mode : {result['scene_mode']}")
    print("\n--- METRICS COMPARISON ---")
    print(f"Wavelet  → PSNR: {result['psnr_wavelet']:<8}  SSIM: {result['ssim_wavelet']}")
    print(f"DL       → PSNR: {result['psnr_dl']:<8}  SSIM: {result['ssim_dl']}")
    print(f"Adaptive → PSNR: {result['psnr_final']:<8}  SSIM: {result['ssim_final']}")
    print(f"\nLog : {result['log_path']}")

    cv2.imshow("Final Adaptive Output", result["fused_array"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
